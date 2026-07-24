"""Resident crop-window scrub short-circuits to a rebind (no producer work).

Scrubbing the displayed-axis crop window of an already-resident montage plane
re-samples the same canonical GPU pages at a shifted origin.  The demand/planning
layer otherwise never consults physical residency, so every shifted window is a
fresh typed target and each tile is re-evaluated (one display-cache miss and one
producer per tile per step) even though the pixels are already resident.  With
the opt-in ``resident_crop_rebind`` capability the planner rebinds the resident
pages before the ladder plans, scheduling ZERO producers for the resident tiles.

The capability is gated OFF by default: the rebind reuses the predecessor
window's auto-level evidence (the maturity contract) instead of re-anchoring, so
it is only pixel-exact against the CPU oracle while the level window is stable
(verified here with statistically uniform data).  A crop whose pages are NOT
resident, or any pixel-affecting identity change, falls through to the ordinary
evaluation.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)

_MONTAGE_INDICES = tuple(range(30, 230, 2))


def _preparation_completed(win) -> int:
    lanes = win.kernel.diagnostics().lanes
    return int((lanes.get("display_preparation") or {}).get("completed", 0) or 0)


def _busy_pump_until(predicate, budget_s, label) -> None:
    # A busy pump, not qtbot.waitUntil: a real event loop never idles, so the
    # low-priority planning continuations only run under sustained queue
    # pressure.  The idle qtbot loop would dispatch them instantly and hide the
    # field scheduling entirely.
    app = QtWidgets.QApplication.instance()
    deadline = perf_counter() + budget_s
    while not predicate():
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        assert perf_counter() < deadline, f"{label} failed to settle within {budget_s:.1f}s"


def _uniform_source() -> np.ndarray:
    # Statistically uniform across every crop window so the auto-level window is
    # stable and a level-reusing rebind stays pixel-exact.  A per-tile constant
    # gradient would shift the window and is deliberately out of scope for the
    # gated rebind.
    return np.random.default_rng(20260724).standard_normal((336, 336, 272), dtype=np.float32)


def _cropped_state(win, start: int):
    state = win.view_state
    state = state.with_axis_range(
        0, indices=tuple(range(start, start + 200)), text=f"{start}:{start + 200}"
    )
    state = state.with_axis_range(1, indices=tuple(range(66, 266)), text="66:266")
    return state.with_montage_axis(2, columns=10, indices=_MONTAGE_INDICES, text="30:2:230")


def _crop_settled(win, start: int) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    ranges = tuple(getattr(session.view_state, "axis_range_indices", None) or ())
    indices = tuple(ranges[0] or ()) if ranges else ()
    if not indices or int(indices[0]) != int(start):
        return False
    return frame_session_settled(win)


def _fill_full_plane_then_crop(win, start: int) -> None:
    full = win.view_state.with_montage_axis(
        2, columns=10, indices=_MONTAGE_INDICES, text="30:2:230"
    )
    win._set_view_state(full)
    win.update_image_view()
    _busy_pump_until(
        lambda: frame_session_settled(win),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "full-plane fill",
    )
    win._apply_slice_state(
        0,
        _cropped_state(win, start),
        reason="slice-range",
        interactive=True,
        immediate_axis_only=False,
    )
    _busy_pump_until(
        lambda: _crop_settled(win, start),
        INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
        "cold cropped fill",
    )


def test_resident_crop_scrub_schedules_no_producers(qtbot):
    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _fill_full_plane_then_crop(win, 94)

        # Every scrub step shifts a fully-resident window: zero producers.
        for step in range(4):
            start = 96 + step
            before = _preparation_completed(win)
            win._on_slice_text_changed(0, f"{start}:{start + 200}")
            _busy_pump_until(
                lambda start=start: _crop_settled(win, start),
                INTERACTION_SETTLE_HARD_LIMIT_MS / 1000.0,
                f"resident scrub {start}",
            )
            assert _preparation_completed(win) - before == 0, (
                "a resident crop scrub must schedule no display-preparation producers"
            )

        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
        assert_wgpu_frame_matches_cpu_reference(win)
    finally:
        win.close()
        restore_default_backend(settings)


def test_cold_crop_scrub_falls_back_to_evaluation(qtbot):
    """A crop window whose pages are NOT resident keeps the ordinary evaluation.

    Starting already cropped never builds the canonical full-plane pages, so
    each shifted window is a cold local identity.  The residency probe withholds
    the rebind and the planner schedules the missing producers, proving the
    short-circuit is strictly residency-gated (partial residency => only the
    missing work runs; here nothing is resident, so all of it does).
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    win = make_backend_window(qtbot, _uniform_source(), backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        win._set_view_state(_cropped_state(win, 94))
        win.update_image_view()
        _busy_pump_until(
            lambda: _crop_settled(win, 94),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 4 / 1000.0,
            "cold cropped fill",
        )
        before = _preparation_completed(win)
        win._on_slice_text_changed(0, "96:296")
        _busy_pump_until(
            lambda: _crop_settled(win, 96),
            INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0,
            "cold scrub",
        )
        assert _preparation_completed(win) - before > 0, (
            "a non-resident crop window must still schedule its producers"
        )
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)


def test_slice_change_does_not_reuse_residency(qtbot):
    """A pixel-affecting identity change (slice index) never rebinds residency.

    The rebind is only legal for a pure window shift under an unchanged content
    key.  Advancing the non-displayed slice index changes the content key, so
    the residency short-circuit must decline and the planner must evaluate.
    """

    settings = use_wgpu_backend(
        extra_settings={"montage_quality_policy": "resident", "resident_crop_rebind": True}
    )
    data = _uniform_source()
    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(780, 760)
    try:
        win.show()
        _fill_full_plane_then_crop(win, 94)
        # Re-slice the montage-adjacent axis via a fresh montage window (new
        # source indices): a genuine content change, not a window shift.
        before = _preparation_completed(win)
        shifted = tuple(index + 1 for index in _MONTAGE_INDICES)
        state = win.view_state.with_montage_axis(2, columns=10, indices=shifted, text="31:2:231")
        win._set_view_state(state)
        win.update_image_view()

        def montage_settled() -> bool:
            session = getattr(win.renderer, "_frame_session", None)
            if session is None or session.plan is None:
                return False
            plan_sources = tuple(int(tile.source_index) for tile in session.plan.tiles)
            return bool(plan_sources == shifted and frame_session_settled(win))

        _busy_pump_until(
            montage_settled, INTERACTION_SETTLE_HARD_LIMIT_MS * 2 / 1000.0, "content change"
        )
        assert _preparation_completed(win) - before > 0, (
            "a content-identity change must not be served by a resident rebind"
        )
    finally:
        win.close()
        restore_default_backend(settings)
