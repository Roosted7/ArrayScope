"""A pan away and back must not republish the whole montage (WGPU).

The defect this pins is render-eagerness §3.2: the WGPU executor already keeps
out-of-view pages resident, but the session dropped every out-of-view tile from
``TilePresentationDelta.active_tiles``.  On ``gpu_atlas`` a tile missing from
that tuple disappears from ``_wgpu_committed["tiles"]`` whether or not a removal
is emitted, and its absence breaks ``previous_commit["tiles"] <= payloads`` —
the precondition the executor tests before applying a delta instead of
republishing the montage.  So a camera move that changed no data paid a full
republication just to put the same pixels back.

The regime is chosen against the failure mode this repo keeps hitting — a green
result resting on a test that cannot fail:

* the montage is large enough that the visible set turns over completely on the
  pan (a two-tile regime has already produced one retracted conclusion here);
* tile content is a per-source gradient, so a tile drawn in the wrong slot is a
  pixel difference rather than an identical constant fill;
* payloads carry ``TileIdentity`` (they come from the real pipeline), because a
  payload without one skips WGPU binding reuse entirely;
* the montage must **complete**, not merely keep its retained tiles visible.
  The R2 violation quoted in the progressive-render contract — zoom in, zoom
  out, settle at 135 of 272 forever with no stall firing — passes straight
  through a retention-only assertion.
"""

from __future__ import annotations

import contextlib

import numpy as np

from arrayscope.tools.framebuffer_reference import assert_wgpu_frame_matches_cpu_reference
from tests.settle import settle_window
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    restore_default_backend,
    use_pyqtgraph_backend,
    use_wgpu_backend,
)

SOURCES = 64
COLUMNS = 8
ROWS = 8
TILE = 160


def _montage_data() -> np.ndarray:
    """Per-source gradients: a tile shown in the wrong slot is visible."""

    yy = np.arange(TILE, dtype=np.float32)[:, None, None]
    xx = np.arange(TILE, dtype=np.float32)[None, :, None]
    zz = np.arange(SOURCES, dtype=np.float32)[None, None, :]
    return np.ascontiguousarray(yy * 3.0 + xx * 1.0 + zz * 5000.0)


def _open_montage(qtbot, *, backend: str):
    win = make_backend_window(
        qtbot, _montage_data(), backend=backend, require_gpu_atlas=backend == "wgpu"
    )
    win.resize(900, 700)
    win.show()
    win._set_view_state(
        win.view_state.with_montage_axis(
            2, columns=COLUMNS, indices=tuple(range(SOURCES)), text=":"
        )
    )
    win.update_image_view()
    settle_window(win, lambda: frame_session_settled(win), label="montage open")
    return win


def _tile_extent(win) -> tuple[float, float]:
    height, width = (int(value) for value in win.renderer._frame_session.plan.display_shape[:2])
    return width / COLUMNS, height / ROWS


def _goto(win, box) -> None:
    x0, x1, y0, y1 = box
    win.img_view.getView().setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
    win.update_image_view()
    # A slow offscreen fill is not this test's subject; the assertions below
    # read counters and payload state, not wall time.
    with contextlib.suppress(Exception):
        settle_window(win, lambda: frame_session_settled(win), label="pan leg")


def _republications(win) -> int:
    return int(getattr(win.img_view, "_wgpu_binding_full_republications", 0) or 0)


def test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope(qtbot):
    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    win = _open_montage(qtbot, backend="wgpu")
    try:
        tile_w, tile_h = _tile_extent(win)
        home = (0.0, 3 * tile_w, 0.0, 2 * tile_h)
        away_x = (5 * tile_w, 8 * tile_w, 0.0, 2 * tile_h)
        away_y = (0.0, 3 * tile_w, (ROWS - 2) * tile_h, ROWS * tile_h)

        _goto(win, home)
        _goto(win, home)
        session = win.renderer._frame_session
        assert session.retain_out_of_view_payloads is True
        home_visible = set(session.visible_tile_numbers)
        assert home_visible, "the zoomed view must still show tiles"

        _goto(win, away_x)
        session = win.renderer._frame_session
        away_visible = set(session.visible_tile_numbers)
        # The visible set genuinely turns over: this is not a nudge.
        assert not (home_visible & away_visible)
        # Every tile the camera left keeps its presentation slot.
        assert home_visible <= set(session.retained_out_of_view)
        assert home_visible <= set(session._last_active_tiles)
        assert_wgpu_frame_matches_cpu_reference(win)

        # The first return leg is measured on its own: coming back to a region
        # whose pixels never changed must cost no republication at all.
        before = _republications(win)
        _goto(win, home)
        assert _republications(win) - before == 0
        assert_wgpu_frame_matches_cpu_reference(win)

        # Then the whole up/down cycle, against the property that actually
        # generalises: a pan must not cost one whole-montage republication per
        # leg.  Retention can still shrink between legs — background work
        # supersedes an out-of-view wrapper and the tile steps out of the scope
        # — and a shrink does republish.  Interleaved in-process A/B measured
        # 4-6 republications per four-leg cycle without retention and 0-2 with
        # it, so this bound separates the two without pinning the exact count.
        legs = (away_y, home, away_x, home)
        before = _republications(win)
        for leg in legs:
            _goto(win, leg)
            assert_wgpu_frame_matches_cpu_reference(win)
            session = win.renderer._frame_session
            # The presentation scope never collapses back to the viewport.
            assert set(session.visible_tile_numbers) < set(session._last_active_tiles)
        assert _republications(win) - before < len(legs)

        # Completion, not merely retention: the R2 pathology this program has
        # already met keeps the retained tiles visible forever and never
        # finishes the round.
        settle_window(
            win,
            lambda: not win.renderer._frame_session.required_target_unsettled_tiles(),
            label="required target completion",
        )
        session = win.renderer._frame_session
        assert session.visible_plan_complete()
        assert session.retained_out_of_view_bytes <= session.retained_out_of_view_budget_bytes
    finally:
        win.close()
        restore_default_backend(settings)


def test_cpu_item_backend_does_not_retain_out_of_view_payloads(qtbot):
    """Scope pin: this change is the ``gpu_atlas`` payload-retention slice.

    PyQtGraph owns real per-tile scene items with their own residency policy
    (render-eagerness §2), and its victim policy is a separate chip.
    """

    settings = use_pyqtgraph_backend(extra_settings={"montage_quality_policy": "resident"})
    win = _open_montage(qtbot, backend="pyqtgraph")
    try:
        tile_w, tile_h = _tile_extent(win)
        _goto(win, (0.0, 3 * tile_w, 0.0, 2 * tile_h))
        _goto(win, (5 * tile_w, 8 * tile_w, 0.0, 2 * tile_h))
        session = win.renderer._frame_session
        assert session.retain_out_of_view_payloads is False
        assert not session.retained_out_of_view
    finally:
        win.close()
        restore_default_backend(settings)
