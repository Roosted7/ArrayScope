"""Crop / axis-range and montage index-window scrub input must coalesce.

A rapid scrub issues many intermediate windows faster than the render loop can
process them.  The ``RenderCoordinator`` collapses that burst to the latest
value: intermediate windows supersede one another in a single pending request,
so a synchronous backlog of ``N`` scrub events becomes a handful of render
passes -- never ``N`` full plan/apply passes over stale windows.  Latest-wins
must still land the final window exactly, keep every processed step rendering,
and settle identically to stepping slowly, with the ADR-0051 montage watchdog
silent throughout.

This pins the input-cadence contract that the displayed-axis crop scrub relies
on (the scalar slice-index fast path already had it; the crop / index-window
range path is extended onto the same principle in ``state_sync`` /
``render``).  Without coalescing every intermediate window would flush its own
render, so the ``passes <= ...`` assertions are the red-first guard.
"""

from __future__ import annotations

import numpy as np

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    restore_default_backend,
    use_pyqtgraph_backend,
)
from tests.ui.helpers import (
    process_events as _process_events,
)


def _coordinator(win):
    coordinator = getattr(win, "render_coordinator", None)
    assert coordinator is not None
    return coordinator


def _committed_current(win) -> bool:
    frame = getattr(win, "_committed_display_frame", None)
    checker = getattr(win.renderer, "_is_committed_display_frame_current", None)
    return bool(frame is not None and callable(checker) and checker(frame))


def _committed_axis_range_text(win, axis: int):
    frame = getattr(win, "_committed_display_frame", None)
    if frame is None:
        return None
    return frame.geometry.view_state.axis_range_text[int(axis)]


def _committed_montage_sources(win):
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return None
    return tuple(int(tile.source_index) for tile in session.plan.tiles)


def _settled_on_crop(win, axis: int, window_text: str) -> bool:
    # The COMMITTED frame -- not just view_state -- must show the final window,
    # so a stale predecessor frame can never satisfy the wait (a lost final
    # step would hang here instead of passing silently).
    return bool(
        _committed_axis_range_text(win, axis) == window_text
        and frame_session_settled(win)
        and _committed_current(win)
    )


def _settled_on_montage(win, indices) -> bool:
    return bool(
        _committed_montage_sources(win) == tuple(indices)
        and frame_session_settled(win)
        and _committed_current(win)
    )


def _stall_assertions(win) -> int:
    return int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0)


def test_crop_window_scrub_backlog_coalesces_to_latest(qtbot):
    """A synchronous backlog of crop-window scrubs collapses to the latest."""

    settings = use_pyqtgraph_backend()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.random.default_rng(0).standard_normal((60, 48, 6)).astype("float32"))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        # Seed the crop so the burst below is a continuation window scrub.
        win._on_slice_text_changed(0, "10:40")
        qtbot.waitUntil(
            lambda: _settled_on_crop(win, 0, "10:40"), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )

        coordinator = _coordinator(win)
        flushed_before = int(coordinator.flushed)
        coalesced_before = int(coordinator.coalesced)

        # Fire the whole burst in ONE event-loop turn (no processing between
        # steps): a real queued backlog of stale windows.
        burst = 16
        windows = [f"{10 + step}:{40 + step}" for step in range(burst)]
        for text in windows:
            win._on_slice_text_changed(0, text)

        # Latest-wins in one drain: the pending request holds only the final
        # window; every earlier window supersedes its predecessor.
        assert int(coordinator.coalesced) - coalesced_before >= burst - 2

        # A bounded handful of passes -- not one per input -- flush to settle.
        qtbot.waitUntil(
            lambda: _settled_on_crop(win, 0, windows[-1]),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        flushed_delta = int(coordinator.flushed) - flushed_before
        assert flushed_delta <= 4, (
            f"expected the {burst}-window backlog to coalesce, saw {flushed_delta} passes"
        )

        # Final window lands exactly in the committed frame, and the scrubbed
        # chip mirrors it (the immediate single-axis sync of the continuation
        # fast path).
        assert win.view_state.axis_range_text[0] == windows[-1]
        assert win.dimension_strip.chip(0).slice_edit.text() == windows[-1]

        # Settled state is identical to stepping slowly, watchdog silent.
        assert _settled_on_crop(win, 0, windows[-1])
        assert _stall_assertions(win) == 0
    finally:
        win.close()
        restore_default_backend(settings)


def test_montage_index_window_scrub_backlog_coalesces_to_latest(qtbot):
    """The montage index-window scrub path coalesces a backlog the same way."""

    settings = use_pyqtgraph_backend()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.random.default_rng(1).standard_normal((40, 40, 32)).astype("float32"))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        # Promote axis 2 to a montage window, then scrub that window.
        win._on_slice_text_changed(2, "4:16")
        qtbot.waitUntil(
            lambda: _settled_on_montage(win, range(4, 16)),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert win.view_state.montage_axis == 2

        coordinator = _coordinator(win)
        flushed_before = int(coordinator.flushed)
        coalesced_before = int(coordinator.coalesced)

        burst = 12
        windows = [f"{4 + step}:{16 + step}" for step in range(burst)]
        for text in windows:
            win._on_slice_text_changed(2, text)

        assert int(coordinator.coalesced) - coalesced_before >= burst - 2

        final_indices = tuple(range(4 + burst - 1, 16 + burst - 1))
        qtbot.waitUntil(
            lambda: _settled_on_montage(win, final_indices),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        flushed_delta = int(coordinator.flushed) - flushed_before
        assert flushed_delta <= 4, (
            f"expected the {burst}-window backlog to coalesce, saw {flushed_delta} passes"
        )

        assert win.view_state.montage_axis == 2
        assert tuple(win.view_state.montage_indices) == final_indices
        assert _settled_on_montage(win, final_indices)
        assert _stall_assertions(win) == 0
    finally:
        win.close()
        restore_default_backend(settings)
