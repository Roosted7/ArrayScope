"""A data change hides its out-of-view tiles first and removes them later.

The product rule these pin: when the data changes — an operation, a reload,
changed indexing, a different view mode — a quick pan must never reveal the
previous data, but updating what is actually on screen outranks that
bookkeeping and must not wait for it.

Two distinct defects, one per backend, and they are deliberately not tested
the same way:

* ``gpu_atlas`` (WGPU) draws world-space instances and a pan submits nothing
  but a camera uniform, so an out-of-view instance left standing after a data
  change is drawn from pages that still hold the superseded data with no
  per-tile predicate anywhere in the path to refuse it.  That is a physical
  defect and it gets a physical oracle: read the render target after a pure
  camera move and compare against a capture of the previous data.
* ``cpu_item`` (PyQtGraph) hides each tile individually when it leaves the
  viewport, so the same pan reveals nothing.  Its test is therefore a GUARD,
  not a reproduction, and says so: it fails only if a future change starts
  leaving out-of-view items drawn.
* ``cpu_item``'s compact preview atlas is the exception to that, and gets a
  reproduction of its own.  It is one Qt item covering many tiles, so it is
  the one place on this backend where a superseded tile could still reach the
  screen — and it exists only above 256 planned tiles, which is exactly the
  large-montage case.

The regime is chosen against the failure mode this repo keeps hitting — a
green result resting on a test that cannot fail:

* the ordering test uses 272 tiles against 24 visible, so "out-of-view first"
  would be an order of magnitude of extra work rather than a rounding error;
* the atlas test uses the field montage (272 sources, 128 px each) because
  the atlas is built only at ``PYQTGRAPH_PREVIEW_ATLAS_MIN_TILES`` (256) or
  more planned tiles, AND only when fitting the montage demands a reduced
  LOD.  Anything smaller on either count passes against the unfixed code;
* every stale-reveal oracle carries its own mutant: the same region captured
  once the new data has settled must NOT match the old capture, which is what
  proves the comparison can fail at all;
* all of them assert the montage COMPLETES.  The R2 violation quoted in the
  progressive-render contract — settled at 135 of 272 forever, no stall firing
  — passes straight through an assertion that only checks what stayed visible.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from arrayscope.tools.framebuffer_reference import _qimage_rgba_array
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    frame_session_settled,
    make_backend_window,
    process_events,
    restore_default_backend,
    use_pyqtgraph_backend,
    use_wgpu_backend,
)

TILE = 32
SLICES = 3
# The ordering test needs the montage the requirement names: 272 tiles, so that
# doing the out-of-view work first would be an order of magnitude of extra work
# rather than a rounding error.  The two stale-reveal tests need a superseded
# out-of-view set and a legible render target, not a big one -- and at 272 their
# completion wait sat right on the interaction settle cap (about one run in
# three timed out serially, which is how these tests joined the documented
# ``tests/gpu`` + ``tests/ui`` contention casualties).  So they get the small
# grid and the ordering test keeps the large one.
LARGE = (272, 17, 16)
SMALL = (64, 8, 8)


def _montage_data(sources: int, *, tile: int = TILE) -> np.ndarray:
    """Three slices with the SAME value distribution and different tiles.

    Slice 0 is the source order reversed.  Identical global range, so an
    auto-window change can never be mistaken for a content change; completely
    different per-tile brightness, so a tile drawn from the wrong slice is
    legible in the render target -- a per-TILE difference is not, at montage
    scale: mirroring each tile in x moves at most 95 against a global span of
    315 000, under one grey level, which is how the first version of this
    fixture managed to pass without the change.

    Every tile also carries its own y/x gradient, so a constant fill can never
    stand in for the right tile: the atlas test below reads pixels back.
    """

    yy = np.arange(tile, dtype=np.float32)[:, None, None]
    xx = np.arange(tile, dtype=np.float32)[None, :, None]
    zz = np.arange(sources, dtype=np.float32)[None, None, :]
    base = np.ascontiguousarray(yy * 3.0 + xx * 1.0 + zz * 5000.0)
    reversed_base = np.ascontiguousarray(base[:, :, ::-1])
    # Slice 0 -- the one the tests change TO -- is the reversed arrangement,
    # so it differs from both slices displayed before it.
    return np.ascontiguousarray(np.stack([reversed_base, base, base], axis=-1))


def _open(qtbot, *, backend, grid=LARGE):
    sources, columns, _rows = grid
    win = make_backend_window(
        qtbot, _montage_data(sources), backend=backend, require_gpu_atlas=backend == "wgpu"
    )
    win._test_montage_grid = grid
    win.resize(900, 700)
    win.show()
    win._set_view_state(
        win.view_state.with_montage_axis(
            2, columns=columns, indices=tuple(range(sources)), text=":"
        )
    )
    win.update_image_view()
    with contextlib.suppress(Exception):
        qtbot.waitUntil(
            lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
    # One data change before the measured one, to a slice the test never
    # displays again.  The FIRST slice step of a session pays the operation
    # pipeline's own warm-up (measured at ~5 s against ~0.3 s for every step
    # after it), which has nothing to do with the subject and does not fit
    # under the interaction settle cap.  Slice 0 stays untouched, so the
    # measured change is still a display-cache MISS for every visible tile —
    # which is the regime where the stale window exists at all.
    win._set_view_state(win.view_state.with_slice(3, 2))
    win.render(reason="test-pipeline-warmup")
    with contextlib.suppress(Exception):
        qtbot.waitUntil(
            lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
    return win


def _tile_extent(win) -> tuple[float, float]:
    _sources, columns, rows = win._test_montage_grid
    height, width = (int(v) for v in win.renderer._frame_session.plan.display_shape[:2])
    return width / columns, height / rows


def _goto(win, qtbot, box) -> None:
    x0, x1, y0, y1 = box
    win.img_view.getView().setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
    win.update_image_view()
    # A slow offscreen fill is not the subject; the assertions read state and
    # the render target, never wall time.
    with contextlib.suppress(Exception):
        qtbot.waitUntil(
            lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )


def _home(win) -> tuple[float, float, float, float]:
    """A deliberately small viewport: 9 visible tiles, 263 out of view at 272.

    Small on purpose twice over.  It makes "out-of-view first" an order of
    magnitude of extra work rather than a rounding error, and it keeps the cold
    fill of the visible set well inside the one interaction settle cap on a
    loaded machine -- these tests joined the documented ``tests/gpu`` +
    ``tests/ui`` contention casualties at 24 visible tiles.  Square rather than
    wide because the WGPU oracle reads a square render target, and a wide box
    letterboxes it until the lit area is too small to compare.
    """

    tile_w, tile_h = _tile_extent(win)
    return (0.0, 3 * tile_w, 0.0, 3 * tile_h)


def _camera_only(win, box) -> None:
    """Move the camera without asking the renderer for anything.

    This is the whole point of the WGPU case: no commit runs, so no per-tile
    staleness predicate gets a chance to refuse anything.
    """

    x0, x1, y0, y1 = box
    win.img_view.getView().setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)


def _capture(win) -> np.ndarray:
    from arrayscope.gpu.command_protocol import SetDisplayMapping, UpdateTileInstances

    view = win.img_view
    # A zero-upload submit so the readback target holds the currently bound
    # instances regardless of how the live surface presents.
    view._submit_wgpu(
        (
            SetDisplayMapping(view._wgpu_mapping_state),
            view._wgpu_camera_command(),
            UpdateTileInstances(view._wgpu_tile_instances()),
        )
    )
    return np.asarray(view._wgpu_executor.read_target())[..., :3].astype(np.int16)


def _lit_fraction(frame: np.ndarray) -> float:
    return float(np.mean(frame.max(axis=-1) > 4))


def _identical_fraction(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32)).max(axis=-1) <= 2))


def _change_the_data(win) -> None:
    """One data change: the displayed slice index moves.

    This is the ``retarget_index_window`` seam — the session object survives,
    so ``invalidate_tiled_presentation`` is never reached and nothing else
    tells the surface to stop drawing the predecessor.

    The target slice is deliberately one that has never been displayed.  A
    display-cache HIT for every visible tile lets the commit inside
    ``render()`` publish the successor's payloads immediately, which narrows
    the drawn set as a side effect and closes the stale window before a test
    can look at it.  Pre-warming the slices therefore makes the whole defect
    unobservable — which it was, once, in this file's first draft.
    """

    win._set_view_state(win.view_state.with_slice(3, 0))
    win.render(reason="test-data-change")


def test_wgpu_data_change_stops_drawing_out_of_view_tiles_before_a_pan_can_reveal_them(qtbot):
    settings = use_wgpu_backend(extra_settings={"montage_quality_policy": "resident"})
    win = _open(qtbot, backend="wgpu", grid=SMALL)
    try:
        _sources, columns, rows = SMALL
        tile_w, tile_h = _tile_extent(win)
        home = _home(win)
        # Wider than ``home``: the ViewBox fits the box to the widget, and a box
        # this shape is what leaves enough lit pixels in the square readback
        # target for the comparison to mean anything (measured 40% lit against
        # 5% for a narrower one, which is the assertion below).
        away = (
            (columns - 4) * tile_w,
            columns * tile_w,
            (rows - 3) * tile_h,
            rows * tile_h,
        )

        # Reference: what the away region looks like drawn from the ORIGINAL
        # data, settled.  Captured before anything is invalidated.
        _goto(win, qtbot, away)
        _goto(win, qtbot, away)
        away_before = _capture(win)
        assert _lit_fraction(away_before) > 0.1, "the away region never filled; nothing to compare"

        _goto(win, qtbot, home)
        _goto(win, qtbot, home)
        session = win.renderer._frame_session
        visible_before = set(session.visible_tile_numbers)
        assert visible_before, "the zoomed view must show tiles"
        # The window this test exists for only exists when the surface is
        # drawing more than the viewport.
        drawn_before = {int(t) for t in (win.img_view._wgpu_committed or {}).get("tiles", ()) or ()}
        assert drawn_before - visible_before, (
            "no out-of-view tile was drawn before the data change, so this "
            "fixture cannot reveal one"
        )

        _change_the_data(win)

        session = win.renderer._frame_session
        visible_now = set(session.visible_tile_numbers)
        drawn_now = {int(t) for t in (win.img_view._wgpu_committed or {}).get("tiles", ()) or ()}
        assert not (drawn_now - visible_now), (
            "the surface is still drawing superseded out-of-view tiles: "
            f"{sorted(drawn_now - visible_now)[:12]}"
        )

        # The pan itself: camera only, no renderer call, no commit.  This is
        # the moment a per-tile predicate would never run.
        _camera_only(win, away)
        revealed = _capture(win)
        assert _identical_fraction(revealed, away_before) < 0.99, (
            "a pure camera move onto the invalidated region reproduced the "
            "previous data pixel for pixel"
        )
        assert _lit_fraction(revealed) < 0.02, (
            "invalidated out-of-view tiles are still drawn after the data change"
        )

        # The oracle's own mutant.  Without this the comparison above could be
        # passing because the two captures can never match.
        _goto(win, qtbot, away)
        _goto(win, qtbot, away)
        settled = _capture(win)
        assert _lit_fraction(settled) > 0.1, "the away region never refilled with the new data"
        assert _identical_fraction(settled, away_before) < 0.9, (
            "the new slice draws the same pixels as the old one, so the "
            "stale-reveal comparison could not have failed"
        )

        # Completion, not merely absence: a montage that hides everything and
        # never finishes would satisfy every assertion above.
        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert not win.renderer._frame_session.required_target_unsettled_tiles()
    finally:
        win.close()
        restore_default_backend(settings)


def test_cpu_item_data_change_leaves_no_out_of_view_tile_drawn(qtbot):
    """Guard, not a reproduction.

    ``cpu_item`` hides each tile as it leaves the viewport, so a pan onto a
    superseded tile shows the background even without the seam.  What this
    pins is that the property keeps holding: if a future change starts leaving
    out-of-view items drawn — the compact preview atlas is the standing
    candidate, since it is one item covering many tiles — this fails.
    """

    settings = use_pyqtgraph_backend(extra_settings={"montage_quality_policy": "resident"})
    win = _open(qtbot, backend="pyqtgraph", grid=SMALL)
    try:
        _goto(win, qtbot, _home(win))
        _goto(win, qtbot, _home(win))

        _change_the_data(win)

        session = win.renderer._frame_session
        visible = set(session.visible_tile_numbers)
        layer = win.img_view._montage_tile_layer
        drawn = {int(tile) for tile in layer._physically_presented_tiles()}
        assert not (drawn - visible), (
            f"superseded out-of-view tiles are still drawn: {sorted(drawn - visible)[:12]}"
        )
        # Hidden, not destroyed: the whole point of hiding rather than
        # removing is that residency survives for the way back.
        assert layer._states, "hiding must not discard the retained tile states"

        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert not win.renderer._frame_session.required_target_unsettled_tiles()
    finally:
        win.close()
        restore_default_backend(settings)


# --- the compact preview atlas -------------------------------------------
#
# ``cpu_item``'s one remaining way for a superseded tile to reach the screen.
# The atlas is a single Qt item covering many tiles, and it is built only when
# a commit plans at least ``PYQTGRAPH_PREVIEW_ATLAS_MIN_TILES`` (256) of them,
# so anything smaller CANNOT reproduce this and passes against the unfixed
# code.  It also needs sources large enough that fitting the montage to the
# widget demands a reduced LOD -- at 32 px a 17x16 montage fits at level 0 and
# every upsert is exact, so no atlas is ever built.
ATLAS_TILE = 128
ATLAS = (272, 17, 16)


def _open_atlas(qtbot):
    """A field montage stopped at the moment the preview atlas is installed.

    Deliberately NOT settled.  The atlas is a bridge: every commit narrows it
    to that commit's active set and exact payloads retire its members one by
    one, so a settled montage has few or none left.  The window this test
    needs is the one the user actually sees on a cold field montage -- the
    aggregate preview standing while the exact fill is still arriving.
    """

    sources, columns, _rows = ATLAS
    win = make_backend_window(qtbot, _montage_data(sources, tile=ATLAS_TILE), backend="pyqtgraph")
    win._test_montage_grid = ATLAS
    win.resize(950, 950)
    win.show()
    with contextlib.suppress(Exception):
        qtbot.waitUntil(
            lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
    win._set_view_state(
        win.view_state.with_montage_axis(
            2, columns=columns, indices=tuple(range(sources)), text=":"
        )
    )
    win.update_image_view()
    layer = win.img_view._montage_tile_layer
    for _ in range(400):
        process_events(qtbot, count=2)
        if layer.preview_atlas_active_tiles:
            break
    return win


def _atlas_boxes(win, members):
    """Pick two disjoint 3x3 tile blocks: one holding atlas members, one not.

    Computed from the ACTUAL membership rather than pinned, because which
    tiles the atlas still carries depends on how far the exact fill got.
    """

    sources, columns, rows = ATLAS
    tile_w, tile_h = _tile_extent(win)

    def block(col0, row0):
        return {
            row * columns + col
            for row in range(row0, row0 + 3)
            for col in range(col0, col0 + 3)
            if row * columns + col < sources
        }

    def box(col0, row0):
        return (col0 * tile_w, (col0 + 3) * tile_w, row0 * tile_h, (row0 + 3) * tile_h)

    blocks = [(col0, row0) for row0 in range(0, rows - 2, 3) for col0 in range(0, columns - 2, 3)]
    away_col, away_row = max(blocks, key=lambda b: len(members & block(*b)))
    away_tiles = block(away_col, away_row)
    # Home is the block furthest from away, so the away members are firmly
    # outside the viewport the data change is served for.
    home_col, home_row = max(
        (b for b in blocks if not (block(*b) & away_tiles)),
        key=lambda b: abs(b[0] - away_col) + abs(b[1] - away_row),
    )
    return box(home_col, home_row), box(away_col, away_row), members & away_tiles


def _grab(win) -> np.ndarray:
    viewport = win.img_view.graphicsView.viewport()
    viewport.repaint()
    return _qimage_rgba_array(viewport.grab().toImage())[..., :3].astype(np.int16)


def _drawn_fraction(win, frame: np.ndarray) -> float:
    """Fraction of the viewport raster that is not the scene background.

    ``_lit_fraction`` is the WGPU render target's oracle and does not carry
    over: PyQtGraph's scene background is a light grey, so "brighter than
    black" is true of every pixel including the empty ones.  Asking the view
    for its own background colour is the same thing the Qt-raster CPU
    reference oracle does.
    """

    colour = win.img_view.graphicsView.backgroundBrush().color()
    background = np.asarray(
        (int(colour.red()), int(colour.green()), int(colour.blue())), dtype=np.int16
    )
    return float(np.mean(np.abs(frame - background).max(axis=-1) > 8))


def test_cpu_item_preview_atlas_stops_drawing_what_a_data_change_superseded(qtbot):
    """A superseded tile cannot reach the screen through the atlas either.

    The atlas is narrowed to each commit's active set, and on ``cpu_item``
    that is the viewport (out-of-view payload retention is ``gpu_atlas``-only,
    ``FrameController._retains_out_of_view_payloads``).  So the one state that
    leaves an atlas member outside the viewport is a camera move with NO
    commit between it and the data change -- which is a user panning or
    zooming a large montage while its first fill is still arriving, and then
    stepping the slice.  That is the sequence here.

    The oracle is pixels, not bookkeeping: after the data change the camera
    moves back onto the superseded block WITHOUT asking the renderer for
    anything, and the viewport raster must be background.  Its own mutant is
    the capture taken before the change -- if that were not lit, the
    assertion could never fail.
    """

    settings = use_pyqtgraph_backend(extra_settings={"montage_quality_policy": "resident"})
    win = _open_atlas(qtbot)
    try:
        layer = win.img_view._montage_tile_layer
        members = set(layer.preview_atlas_active_tiles)
        assert members, (
            "the compact preview atlas never formed, so this test cannot "
            "reproduce the defect it exists for"
        )
        home, away, away_members = _atlas_boxes(win, members)
        assert away_members, (
            "no atlas member landed in the away block; the fixture would be "
            "asserting about tiles the atlas never covered"
        )

        # Camera only, both legs: a commit would narrow the atlas to the
        # viewport and close the window before the data change arrives.
        _camera_only(win, away)
        away_before = _grab(win)
        assert _drawn_fraction(win, away_before) > 0.1, (
            "the away block was never drawn; nothing to compare"
        )
        _camera_only(win, home)

        session = win.renderer._frame_session
        drawn_before = {int(tile) for tile in layer._physically_presented_tiles()}
        assert away_members <= drawn_before, (
            "the away atlas members are not drawn, so no pan could reveal them"
        )

        _change_the_data(win)

        session = win.renderer._frame_session
        visible = set(session.visible_tile_numbers)
        assert away_members - visible, (
            "every away atlas member stayed in view across the data change, "
            "so none of them was superseded and this test measures nothing"
        )
        # The pan itself, and the assertion this test rests on.  No renderer
        # call, so no per-tile predicate runs.  Pixels first deliberately: the
        # bookkeeping check below agrees with it, but a green result here has
        # to be a green result about the screen.
        _camera_only(win, away)
        revealed = _grab(win)
        assert _drawn_fraction(win, revealed) < 0.02, (
            "a pure camera move onto the superseded block still draws it -- "
            f"{_drawn_fraction(win, revealed):.3f} of the viewport is not "
            f"background, and {_identical_fraction(revealed, away_before):.3f} "
            "of it is pixel-identical to the previous data"
        )
        drawn = {int(tile) for tile in layer._physically_presented_tiles()}
        assert not (drawn & away_members), (
            f"the atlas is still drawing superseded tiles: {sorted(drawn & away_members)[:12]}"
        )

        # Hidden, not destroyed, and the montage still finishes.
        assert layer._states, "hiding must not discard the retained tile states"
        _goto(win, qtbot, away)
        _goto(win, qtbot, away)
        settled = _grab(win)
        assert _drawn_fraction(win, settled) > 0.1, (
            "the away block never refilled with the new data"
        )
        assert _identical_fraction(settled, away_before) < 0.9, (
            "the new slice draws the same pixels as the old one, so the "
            "stale-reveal comparison could not have failed"
        )
        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert not win.renderer._frame_session.required_target_unsettled_tiles()
    finally:
        win.close()
        restore_default_backend(settings)


@pytest.mark.parametrize("backend", ["pyqtgraph", "wgpu"])
def test_data_change_serves_the_visible_set_before_any_out_of_view_removal(qtbot, backend):
    """The ordering property, as a property rather than a comment.

    272 tiles, 9 of them visible.  Before this, every commit on the way to the
    visible set's first pixels carried the other 263 tiles' removals — each one
    re-emitted in the ``removals`` tuple of every commit until the backend
    acknowledged it, so nine commits meant nine times 248 at a wider viewport.
    The assertion is that ZERO out-of-view removal reaches a commit before
    ``visible_plan_complete()``, and that the deferred queue then drains to
    empty on its own.
    """

    settings = (use_wgpu_backend if backend == "wgpu" else use_pyqtgraph_backend)(
        extra_settings={"montage_quality_policy": "resident"}
    )
    win = _open(qtbot, backend=backend)
    try:
        from arrayscope.window.frame_session import FrameSession

        _goto(win, qtbot, _home(win))
        _goto(win, qtbot, _home(win))
        # Warm the target slice for the VISIBLE set only.  Unlike the two
        # stale-reveal tests, this one's subject is what the commits carry, not
        # what the surface draws, and the retarget queues an out-of-view
        # invalidation whether or not the visible tiles are cached — cached
        # tiles reach the retarget only for the visible display tiles, so every
        # out-of-view tile takes the miss branch either way (the guard below
        # fails outright if that stops being true).  What the warm buys is a
        # 272-tile fill that fits inside the one interaction settle cap on a
        # loaded machine.
        for index in (0, 2):
            win._set_view_state(win.view_state.with_slice(3, index))
            win.render(reason="test-target-warmup")
            with contextlib.suppress(Exception):
                qtbot.waitUntil(
                    lambda: frame_session_settled(win), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
                )
        session = win.renderer._frame_session
        visible = set(session.visible_tile_numbers)
        out_of_view = len(tuple(session.plan.tiles)) - len(visible)
        assert len(visible) < out_of_view / 4, (
            "visible and out-of-view must differ by an order of magnitude or "
            f"the ordering is unmeasurable: {len(visible)} vs {out_of_view}"
        )

        early_removals: list[int] = []
        original = FrameSession.build_tile_presentation

        def recording(self, *args, **kwargs):
            state, delta = original(self, *args, **kwargs)
            if not self.visible_plan_complete():
                early_removals.extend(
                    int(tile)
                    for tile in delta.removals
                    if int(tile) not in self.visible_tile_numbers
                )
            return state, delta

        FrameSession.build_tile_presentation = recording
        try:
            _change_the_data(win)
            session = win.renderer._frame_session
            assert session.deferred_out_of_view_invalidations, (
                "the data change queued no deferred out-of-view invalidation, "
                "so this test is measuring nothing"
            )
            qtbot.waitUntil(
                lambda: win.renderer._frame_session.visible_plan_complete(),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
            assert not early_removals, (
                "out-of-view removals rode the commits that owed the screen "
                f"its first pixels: {len(early_removals)} entries"
            )
        finally:
            FrameSession.build_tile_presentation = original

        # Deferred, not dropped: the queue converges without a timer of its
        # own and without an unbounded backlog.
        qtbot.waitUntil(
            lambda: not win.renderer._frame_session.deferred_out_of_view_invalidations,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        session = win.renderer._frame_session
        assert session.deferred_out_of_view_drained >= out_of_view - len(visible)

        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert not win.renderer._frame_session.required_target_unsettled_tiles()
    finally:
        win.close()
        restore_default_backend(settings)
