"""Montage index-window scroll must settle to target quality both directions.

Regression net for the trace-proven idle stall (2026-07-15,
``/tmp/arrayscope-stall-18-1.trace.jsonl`` seq 9703-9708 and
``/tmp/arrayscope-stall-65-2.trace.jsonl`` seq 37576-37585, decoded in
``docs/redesign/coverage-stall-2026-07-15.md``): after an index-window
retarget the shared-transform DESIRED pass is barred behind the reset
``first_pass_histogram_published`` flag, and when the rough level evidence
for the newly scrolled sources completed AFTER the last acknowledgement
commit, no hook remained to publish it — a closed wait cycle with an idle
kernel and ``required_target_unsettled_tiles()`` non-empty until the
watchdog asserted.  The deterministic half of the fix is unit-tested in
``tests/window/test_montage_backend.py`` (late first-pass evidence arms the
publication flush); this test drives the live shape offscreen: an
FFT-over-montage-axis pipeline (shared-transform owner) on the VisPy shader
backend, scrolled down and back up, must reach ``required_target_settled``
in both directions with zero watchdog stall assertions.
"""

from __future__ import annotations

import numpy as np

from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_vispy_backend,
)

_FILL_TIMEOUT_MS = 60_000
_SCROLL_TIMEOUT_MS = 30_000


def _window_settled(win, indices) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    plan_sources = tuple(int(tile.source_index) for tile in session.plan.tiles)
    if plan_sources != tuple(indices):
        return False
    return frame_session_settled(win)


def _scroll_to(win, qtbot, indices) -> None:
    state = win.view_state.with_montage_axis(
        2,
        columns=5,
        indices=tuple(indices),
        text=f"{indices[0]}:{indices[-1] + 1}",
    )
    win._set_view_state(state)
    win.update_image_view()
    qtbot.waitUntil(lambda: _window_settled(win, indices), timeout=_SCROLL_TIMEOUT_MS)


def test_fft_montage_scroll_down_then_up_settles_required_target(qtbot):
    settings = use_vispy_backend(extra_settings={"montage_quality_policy": "resident"})
    from arrayscope.operations.pipeline import CenteredFFT

    rng = np.random.default_rng(20260715)
    data = rng.standard_normal((96, 96, 36), dtype=np.float32)

    win = make_backend_window(qtbot, data)
    win.resize(520, 420)
    try:
        win.show()
        # FFT over the montage axis: non-commuting for display LOD, so the
        # shared-transform fanout owns every tile's producer — the exact
        # scheduling shape of the stalled traces.
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()

        initial = tuple(range(0, 20))
        state = win.view_state.with_montage_axis(
            2,
            columns=5,
            indices=initial,
            text="0:20",
        )
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(lambda: _window_settled(win, initial), timeout=_FILL_TIMEOUT_MS)

        # Scroll down (new sources appended), then back up (new sources
        # prepended) — both retarget directions of the live repro.
        _scroll_to(win, qtbot, tuple(range(6, 26)))
        _scroll_to(win, qtbot, tuple(range(2, 22)))

        session = win.renderer._frame_session
        assert session.required_target_unsettled_tiles() == ()
        # The watchdog must never have asserted: the kernel may not sit idle
        # while required tiles are unsettled (dossier exit gate).
        assert int(getattr(win.renderer, "_montage_stall_assertions", 0) or 0) == 0
    finally:
        win.close()
        restore_default_backend(settings)
