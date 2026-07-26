"""A 1-tile montage scroll must retain the aggregate level population.

The semantic ``montage_level_key`` embeds the selected index window, so a
scroll retarget starts a fresh per-key tile map.  The retained per-source
statistics live in the family-keyed memory cache, but before the rehydration
fix nothing bulk-seeded the new key from that cache: the population trickled
back one bounded evidence batch at a time while ``frame_effects`` read the
aggregate with ``allow_partial=True`` during the first pass.  The window/level
and histogram were therefore briefly computed over a handful of sources -- a
visible dip toward one tile's range -- before recovering.

This drives a settled montage, scrolls it by exactly one tile, and asserts the
provisional level source the display would read immediately after the retarget
still covers the retained population (not a rough handful).  Red before the
fix (source_count collapses to one commit batch); green after.
"""

from __future__ import annotations

import numpy as np

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_wgpu_backend,
)


def _window_settled(win, indices) -> bool:
    session = getattr(win.renderer, "_frame_session", None)
    if session is None or session.plan is None:
        return False
    plan_sources = tuple(int(t.source_index) for t in session.plan.tiles)
    if plan_sources != tuple(indices):
        return False
    return frame_session_settled(win)


def _run(qtbot):
    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})

    rng = np.random.default_rng(7)
    height = width = 96
    depth = 70
    base = rng.random((height, width), dtype=np.float32)
    data = np.empty((height, width, depth), dtype=np.float32)
    for k in range(depth):
        # Distinct per-slice ranges (~[0, k+1]) make an aggregate computed over
        # a rough handful of sources unmistakably narrower than the full window.
        data[:, :, k] = base * float(k + 1)

    win = make_backend_window(qtbot, data, backend="wgpu", require_gpu_atlas=True)
    win.resize(1100, 850)
    try:
        win.show()
        cols = 10
        window_size = 50
        initial = tuple(range(window_size))
        state = win.view_state.with_montage_axis(2, columns=cols, indices=initial, text="0:50")
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: _window_settled(win, initial), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )

        baseline_session = win.renderer._frame_session
        baseline = win.renderer._montage_level_source_for_session(
            baseline_session, allow_partial=True
        )
        assert baseline is not None
        assert baseline.source_count == window_size
        baseline_high = float(baseline.histogram_range[1])

        # Scroll by exactly one tile: slice 0 leaves, slice 50 enters.
        target = tuple(range(1, window_size + 1))
        state = win.view_state.with_montage_axis(2, columns=cols, indices=target, text="1:51")
        win._set_view_state(state)
        win.update_image_view()

        # The seam frame_effects reads during the shader first pass
        # (allow_partial=True). Immediately after the retarget it must already
        # reflect the retained population, not a single evidence batch.
        session = win.renderer._frame_session
        source = win.renderer._montage_level_source_for_session(session, allow_partial=True)
        assert source is not None, "no provisional level source after retarget"
        # 49 of 50 sources stayed on screen and are in the family cache; only the
        # one genuinely new tile may be missing. A dip to one commit batch (~4)
        # is the bug.
        assert source.source_count >= window_size - 1, (
            f"aggregate collapsed to {source.source_count} sources after a 1-tile "
            f"scroll (expected >= {window_size - 1}); retained levels not reused"
        )
        # The high bound must not dip below the retained window's high.
        assert float(source.histogram_range[1]) >= baseline_high - 1.0

        qtbot.waitUntil(
            lambda: _window_settled(win, target), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
    finally:
        win.close()
        restore_default_backend(settings)


def test_wgpu_one_tile_scroll_retains_level_population(qtbot):
    _run(qtbot)
