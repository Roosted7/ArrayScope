import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    clear_arrayscope_settings as _clear_arrayscope_settings,
)
from tests.ui.helpers import (
    process_events as _process_events,
)
from tests.ui.test_montage_interactions import _committed_tile_payload


def _committed_montage_tile_images(win):
    """Return {montage_index: committed tile image} for the live montage."""

    session = win.renderer._frame_session
    images = {}
    for tile in session.plan.tiles:
        payload = _committed_tile_payload(win, tile)
        if payload is not None and getattr(payload, "image", None) is not None:
            images[int(tile.montage_index)] = np.asarray(payload.image).copy()
    return images


def _wait_for_committed_montage_tiles(win, qtbot, count, *, expected_shape=None):
    def _ready():
        _process_events(qtbot, count=5)
        images = _committed_montage_tile_images(win)
        return len(images) >= count and (
            expected_shape is None
            or all(tuple(image.shape[:2]) == tuple(expected_shape) for image in images.values())
        )

    qtbot.waitUntil(_ready, timeout=min(8000, INTERACTION_SETTLE_HARD_LIMIT_MS))


def test_tiled_dimension_x_y_buttons_promote_range_to_image_crop(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1), text=":"))
        win.render(reason="test")
        _process_events(qtbot)

        chip = win.dimension_strip.chip(2)
        assert chip.x_button.isEnabled()
        assert chip.y_button.isEnabled()
        win.set_dimension_role("x", 2)

        assert win.view_state.image_axes == (0, 2)
        assert win.view_state.montage_axis is None
        assert win.view_state.axis_range_indices[2] == (0, 1)
        assert win.view_state.axis_range_text[2] == ":"
        assert win.view_state.slice_indices[1] == 1
    finally:
        win.close()


def test_middle_horizontal_drag_steps_highlighted_dimension(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5, 6), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win.resize(900, 700)
        win.show()
        _process_events(qtbot)
        strip = win.dimension_strip
        target = strip.scroll_target_axis()
        assert target == 2
        assert strip.chip(target).property("indexScrollTarget") is True

        view = win.img_view
        viewport = view.graphicsView.viewport()
        start = QtCore.QPointF(viewport.width() / 2.0, viewport.height() / 2.0)
        middle = QtCore.Qt.MouseButton.MiddleButton

        def send(event_type, position, button, buttons):
            event = QtGui.QMouseEvent(
                event_type,
                position,
                button,
                buttons,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )
            assert view.eventFilter(viewport, event) is True
            _process_events(qtbot)

        before = win.view_state.slice_indices[target]
        send(QtCore.QEvent.Type.MouseButtonPress, start, middle, middle)
        send(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(start.x() + 50.0, start.y()),
            QtCore.Qt.MouseButton.NoButton,
            middle,
        )
        assert strip.chip(target).property("indexScrollActive") is True
        send(
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(start.x() + 50.0, start.y()),
            middle,
            QtCore.Qt.MouseButton.NoButton,
        )

        assert win.view_state.slice_indices[target] == min(win.data.shape[target] - 1, before + 5)
        assert strip.chip(target).property("indexScrollActive") is False
    finally:
        win.close()


def test_demoting_cropped_image_axis_preserves_it_as_montage(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 5 * 4, dtype=float).reshape(2, 5, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_axis_range(1, indices=(0, 2, 4), text="0:5:2"))

        win.set_dimension_role("x", 2)

        assert win.view_state.image_axes == (0, 2)
        assert win.view_state.montage_axis == 1
        assert win.view_state.montage_indices == (0, 2, 4)
        assert win.view_state.montage_text == "0:5:2"
        assert win.view_state.axis_range_indices[1] is None
    finally:
        win.close()


def test_demoting_full_image_axis_centers_scalar_slice(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(6 * 5 * 4, dtype=float).reshape(6, 5, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_slice(0, 0))

        win.set_dimension_role("y", 2)

        assert win.view_state.image_axes == (2, 1)
        assert win.view_state.slice_indices[0] == 3
    finally:
        win.close()


def test_swapping_cropped_x_axis_to_y_keeps_existing_montage_axis(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(6 * 5 * 4, dtype=float).reshape(6, 5, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        state = (
            win.view_state.with_image_axes(1, 2)
            .with_montage_axis(0, indices=(0, 2, 4), text="0:6:2")
            .with_axis_range(2, indices=(1, 2, 3), text="1:4")
        )
        win._set_view_state(state)

        win.set_dimension_role("y", 2)

        assert win.view_state.image_axes == (2, 1)
        assert win.view_state.montage_axis == 0
        assert win.view_state.montage_indices == (0, 2, 4)
        assert win.view_state.axis_range_indices[2] == (1, 2, 3)
    finally:
        win.close()


def test_empty_tiled_slice_text_clears_to_midpoint_scalar(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1, 2), text=":"))
        win._on_slice_text_changed(2, "")
        _process_events(qtbot)

        assert win.view_state.montage_axis is None
        assert win.view_state.axis_range_indices[2] is None
        assert win.view_state.slice_indices[2] == 2
    finally:
        win.close()


def test_invalid_slice_text_restores_state_and_shows_status(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.state_sync as state_sync
    from arrayscope.window import ArrayScopeWindow

    messages = []
    monkeypatch.setattr(
        state_sync,
        "show_status_message",
        lambda _window, message, **_kwargs: messages.append(str(message)),
    )

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        before = win.view_state

        win._on_slice_text_changed(2, "abc")
        _process_events(qtbot)

        assert win.view_state == before
        assert messages
        assert "Could not understand" in messages[-1]
        assert win.dimension_strip.chip(2).slice_edit.text() == "2"
    finally:
        win.close()


def test_raw_index_list_text_creates_montage_selection(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)

        win._on_slice_text_changed(2, "0,2;4")
        _process_events(qtbot)

        assert win.view_state.montage_axis == 2
        assert win.view_state.montage_indices == (0, 2, 4)
        assert win.view_state.montage_text == "0 2 4"
    finally:
        win.close()


def test_raw_index_list_montage_renders_each_selected_source(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 60), dtype=float)
    for index in range(data.shape[2]):
        data[:, :, index] = float(index)
    selected = (7, 8, 9, 11, 14, 56)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)

        win._on_slice_text_changed(2, "7 8 9 11 14 56")
        for _ in range(120):
            _process_events(qtbot, count=2)
            session = getattr(win.renderer, "_frame_session", None)
            if session is not None and len(getattr(session, "rendered_tiles", {})) == len(selected):
                break

        session = win.renderer._frame_session
        rendered = [session.rendered_tiles[tile] for tile in sorted(session.rendered_tiles)]
        assert tuple(tile.tile.source_index for tile in rendered) == selected
        for rendered_tile in rendered:
            assert np.nanmin(rendered_tile.image) == float(rendered_tile.tile.source_index)
            assert np.nanmax(rendered_tile.image) == float(rendered_tile.tile.source_index)
    finally:
        win.close()


def test_live_profile_from_axis_sets_exactly_one_profile_axis(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.profile_axes = (0, 1)
        win._enable_live_profile_for_axis(2)
        _process_events(qtbot)

        assert win.profile_axes == (2,)
        assert win.view_state.line_axis == 2
        assert win.profile_dock.isVisible()
    finally:
        win.close()


def test_add_operation_leaves_dimension_strip_columns_consistent(qtbot):
    """Ring 1 (offscreen) — the stale-reflow failure reproduces offscreen.

    Regression (2026-07-18 dogfood report): adding an operation transiently
    narrowed the strip's parent (dock/canvas-preserve churn), wrapping the
    chips to an extra row; the strip stayed wrapped after the width came back
    until the next data-driven relayout (e.g. scrolling an index).
    """
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9 * 10 * 11, dtype=float).reshape(8, 9, 10, 11))
    qtbot.addWidget(win)
    try:
        win.resize(1000, 900)
        win.show()
        _process_events(qtbot, count=30)
        # The user closed the operation dock earlier in the session, so adding
        # an operation must not reopen it — and must not leave the strip laid
        # out for a width it no longer has.
        win._operation_dock_user_visible = False
        strip = win.dimension_strip
        assert strip._columns == strip._column_count()

        win.request_operation("centered_fft", 0)
        _process_events(qtbot, count=40)

        assert not win.operation_dock.isVisible()
        assert strip._columns == strip._column_count()
    finally:
        win.close()


def test_add_operation_does_not_resize_viewport_with_dock_open(qtbot):
    """Adding an operation while the operation dock is ALREADY open must not
    change the render viewport size -- no layout moves, so nothing should flash.

    Regression: a mid-relayout ``layoutChanged`` read a transiently-narrow
    width, wrapped the chips to an extra row, and grew the dimension-strip
    height for one turn; through the shared central layout that briefly shrank
    the render viewport before it settled back.
    """
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.random.default_rng(0).random((8, 40, 40)).astype(np.float32))
    qtbot.addWidget(win)
    try:
        win.resize(1000, 800)
        win.show()
        qtbot.waitExposed(win)
        _process_events(qtbot, count=30)
        win.layout_manager.set_operation_dock_visible_from_user(True)
        _process_events(qtbot, count=30)
        assert win.operation_dock.isVisible()

        viewport = win.img_view.graphicsView.viewport()
        strip_scroll = win.dims_scroll
        before_vp = viewport.height()
        before_strip = strip_scroll.height()

        # Spy on every height the strip is actually given during the settle --
        # a sub-turn transient (mid-relayout height grow) would otherwise be
        # reverted before a poll could observe it.
        applied_heights = []
        real_set_fixed = strip_scroll.setFixedHeight
        strip_scroll.setFixedHeight = lambda h: (applied_heights.append(int(h)), real_set_fixed(h))[
            1
        ]

        win.request_operation("centered_fft", 1)
        _process_events(qtbot, count=40)

        # No height applied during the op-add may exceed the pre-add height:
        # that grow is exactly what briefly shrank the viewport.
        assert all(h <= before_strip for h in applied_heights), applied_heights
        assert strip_scroll.height() == before_strip
        assert viewport.height() == before_vp
    finally:
        win.close()


def test_montage_x_y_swap_is_an_instant_display_transform(qtbot):
    """A montage X/Y swap is a pure DISPLAY transform, like an axis flip.

    On a backend that renders canonical tiles (``display_axis_transpose``) an
    X/Y axis-order swap must NOT re-materialize or re-orient the cached tile
    payloads: they stay in canonical (sorted-image-axes) orientation and are
    reused verbatim, while the backend applies the swap at draw time and the
    hover/ROI value-readout indexes the canonical array with swapped
    coordinates.  Regression guard for two failure modes: (1) re-rendering the
    tiles on a swap (the payload would transpose / eval count would climb), and
    (2) leaving the pre-swap orientation on screen (hover would read the old
    value).  Runs on the default pyqtgraph backend, which is canonical.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    # A SQUARE image plane (N, N, K): a transpose keeps tile shape and slice
    # indices identical, so nothing but the display transform changes.  Per-plane
    # data is asymmetric so a stale (non-transposed) readout is detectable:
    # value = x + 10*y + 100*k.
    n_side, n_tiles = 5, 4
    yy, xx, kk = np.mgrid[0:n_side, 0:n_side, 0:n_tiles]
    data = (xx + 10 * yy + 100 * kk).astype(np.float32)

    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(
            win.view_state.with_montage_axis(2, indices=tuple(range(n_tiles)), text=":")
        )
        win.render(reason="test-montage")
        _wait_for_committed_montage_tiles(win, qtbot, n_tiles, expected_shape=(n_side, n_side))

        before = _committed_montage_tile_images(win)
        assert len(before) == n_tiles
        assert win.view_state.image_axes == (0, 1)

        # Hover at a fixed, asymmetric tile-local pixel BEFORE the swap.  Tile 0,
        # local (col=1, row=0) -> value x=1, y=0 -> 1.
        tile0 = win.renderer._frame_session.plan.tiles[0]

        def _hover_value():
            _process_events(qtbot, count=5)
            context = win.display_geometry.context_for_view_point(tile0.x0 + 1, tile0.y0 + 0)
            if context is None:
                return None
            return win.renderer._hover_value_from_display(context.mapping)

        qtbot.waitUntil(
            lambda: _hover_value() is not None,
            timeout=min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
        assert _hover_value() == pytest.approx(1.0)

        evals_before = int(win.operation_evaluator.image_evaluations)

        # Swap X/Y: click Y on the axis currently acting as X.
        x_axis = win.view_state.image_axes[1]
        win.set_dimension_role("y", x_axis)
        assert win.view_state.image_axes == (1, 0)

        _wait_for_committed_montage_tiles(win, qtbot, n_tiles, expected_shape=(n_side, n_side))
        after = _committed_montage_tile_images(win)

        # 1) Tile payloads are CANONICAL and reused verbatim -- unchanged across
        #    the swap (never transposed into the payload), and no tile was
        #    re-evaluated (a display transform costs no compute).
        for index, old_image in before.items():
            assert index in after, f"tile {index} dropped after swap"
            np.testing.assert_array_equal(
                after[index],
                old_image,
                err_msg=f"tile {index} payload changed on the swap (should stay canonical)",
            )
        assert int(win.operation_evaluator.image_evaluations) == evals_before, (
            "an X/Y swap re-evaluated tiles instead of reusing canonical payloads"
        )

        # 2) The DISPLAY transposed: the same screen pixel now reads the
        #    transposed value.  local (col=1, row=0) under image_axes=(1,0)
        #    maps screen-col->axis0, screen-row->axis1 -> value 10.
        qtbot.waitUntil(
            lambda: _hover_value() == pytest.approx(10.0),
            timeout=min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
    finally:
        win.close()


def test_wgpu_montage_x_y_swap_reuses_gpu_residency(qtbot):
    """On wgpu an X/Y swap rebinds resident textures -- zero new uploads.

    The strongest form of the instant-transpose contract: the GPU upload count
    is flat across the swap (existing textures are re-sampled with a swapped UV
    walk), the canonical payloads are unchanged, and hover reads the transposed
    value.
    """

    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "wgpu")
    settings.sync()
    from arrayscope.window import ArrayScopeWindow

    n_side, n_tiles = 5, 4
    yy, xx, kk = np.mgrid[0:n_side, 0:n_side, 0:n_tiles]
    data = (xx + 10 * yy + 100 * kk).astype(np.float32)

    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        if type(win.img_view).__name__ != "WgpuSurface":
            pytest.skip("wgpu backend unavailable in this environment")
        win._set_view_state(
            win.view_state.with_montage_axis(2, indices=tuple(range(n_tiles)), text=":")
        )
        win.render(reason="test-montage")
        _wait_for_committed_montage_tiles(win, qtbot, n_tiles, expected_shape=(n_side, n_side))
        before = _committed_montage_tile_images(win)
        assert len(before) == n_tiles

        tile0 = win.renderer._frame_session.plan.tiles[0]

        def _hover_value():
            _process_events(qtbot, count=5)
            context = win.display_geometry.context_for_view_point(tile0.x0 + 1, tile0.y0 + 0)
            if context is None:
                return None
            return win.renderer._hover_value_from_display(context.mapping)

        qtbot.waitUntil(
            lambda: _hover_value() is not None,
            timeout=min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
        uploads_before = int(win.img_view._wgpu_executor.uploads_total)

        x_axis = win.view_state.image_axes[1]
        win.set_dimension_role("y", x_axis)
        assert win.view_state.image_axes == (1, 0)
        _wait_for_committed_montage_tiles(win, qtbot, n_tiles, expected_shape=(n_side, n_side))

        # No new GPU uploads: the swap re-sampled resident textures.
        assert int(win.img_view._wgpu_executor.uploads_total) == uploads_before, (
            "an X/Y swap re-uploaded tiles instead of rebinding GPU residency"
        )
        after = _committed_montage_tile_images(win)
        for index, old_image in before.items():
            np.testing.assert_array_equal(after[index], old_image)
        qtbot.waitUntil(
            lambda: _hover_value() == pytest.approx(10.0),
            timeout=min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS),
        )
    finally:
        win.close()


def test_wgpu_transposed_nonsquare_montage_floor_admission_survives(qtbot):
    """A transposed NON-SQUARE montage with reduced-LOD floors must not crash.

    Regression for a page-LOD orientation mismatch: the pyramid pages are
    canonical (transpose-invariant), but the reduced-floor ``requested_lod``
    described its source extent with the DISPLAY-oriented ``plan.tile_shape``.
    On a transposed non-square montage the two disagreed and
    ``PageBackedPresentation`` raised "requested page LOD source shape disagrees
    with native source coverage" inside floor admission (async, logged not
    raised), stalling the tiles.
    """

    import logging

    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", "wgpu")
    settings.setValue("montage_quality_policy", "resident")
    settings.sync()
    from arrayscope.window import ArrayScopeWindow

    # Many NON-SQUARE planes so each tile renders heavily reduced (floors),
    # montaged on axis 0 -> image_axes (1, 2).
    k, n, m = 64, 256, 192
    data = np.random.default_rng(0).standard_normal((k, n, m)).astype(np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)

    floor_errors: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.exc_info and "page LOD source shape disagrees" in str(record.exc_info[1]):
                floor_errors.append(record)

    handler = _Capture()
    logging.getLogger().addHandler(handler)
    try:
        _process_events(qtbot)
        if type(win.img_view).__name__ != "WgpuSurface":
            pytest.skip("wgpu backend unavailable in this environment")
        win._set_view_state(
            win.view_state.with_image_axes(1, 2).with_montage_axis(
                0, indices=tuple(range(k)), text=":"
            )
        )
        win.render(reason="test-montage")
        _process_events(qtbot, count=60)
        # Transpose back and forth: builds floors in one orientation and reuses
        # the canonical pages in the other.
        for _ in range(4):
            win._set_view_state(win.view_state.transposed_image_axes())
            win.update_image_view()
            _process_events(qtbot, count=40)
        assert not floor_errors, (
            "transposed non-square floor admission raised a page-LOD shape mismatch"
        )
    finally:
        logging.getLogger().removeHandler(handler)
        win.close()


def _assert_committed_frame_not_stale(win, qtbot, *, label):
    """Assert the watchdog's ``committed_frame_stale`` predicate stays clear.

    Mirrors the exact terminal condition the ADR-0051 stall watchdog checks
    (``frame_runtime.py``): with the required target fully settled and its
    first pixels presented, the committed display frame must satisfy
    ``_is_committed_display_frame_current``.  A lagging render-generation stamp
    fails only that predicate while every pixel is current -- the field defect.

    Uses a bare ``processEvents`` pump rather than a timed wait: the session is
    already settled by the caller, and a timed wait would let an incidental
    single-plane viewport-update timer re-commit and mask the persistent stall
    the field captured (montage sessions have no such timer, so they never
    self-heal).
    """

    from pyqtgraph.Qt import QtWidgets

    app = QtWidgets.QApplication.instance()
    renderer = win.renderer
    session = renderer._frame_session
    assert session is not None, f"{label}: no live frame session"
    for _ in range(10):
        app.processEvents()
    assert session.required_target_unsettled_tiles() == (), (
        f"{label}: target not settled -- reproduction precondition unmet"
    )
    assert bool(session.required_first_pixels_presented()), (
        f"{label}: first pixels not presented -- reproduction precondition unmet"
    )
    frame = getattr(win, "_committed_display_frame", None)
    assert frame is not None, f"{label}: no committed display frame"
    committed_frame_stale = not renderer._is_committed_display_frame_current(frame)
    assert not committed_frame_stale, (
        f"{label}: committed_frame_stale fired -- committed frame stamped at "
        f"generation {int(frame.key.render_generation)} but live generation is "
        f"{renderer._render_generation.current} while every pixel-affecting "
        f"identity component matches (the watchdog would assert an idle stall)"
    )
    assert int(getattr(renderer, "_montage_stall_assertions", 0) or 0) == 0, (
        f"{label}: a stall assertion fired"
    )


def _issue_settle_tail_renders(win, qtbot, *, count=3):
    """Replay the redundant renders an interaction issues as it settles.

    An interactive scroll/swap fires many ``request_render`` calls; the last
    ones land on an already-settled presentation and produce no fresh commit.
    Each still advances the global render generation, so the committed frame's
    generation stamp is left behind -- the exact tail seen in the field traces
    (``arrayscope-stall-78-1`` slice tail, ``arrayscope-stall-445-7`` gens
    1053-1061 at a fixed view state).
    """

    for _ in range(count):
        win.renderer.request_render(reason="settle-tail", interactive=True)
        _process_events(qtbot, count=10)


def test_x_y_swap_settle_leaves_committed_frame_current(qtbot):
    """Scenario A: an X/Y swap that settles must not strand a stale committed frame.

    Field repro ``arrayscope-stall-78-1`` (signature session 78,
    ``committed_frame_stale=1``, kernel idle): the user changed X/Y and the
    view settled, then the interaction's redundant settle-tail renders landed
    on the session-reuse path -- which refreshes ``session.render_generation``
    but commits nothing when the presentation is already complete.  The
    committed display frame kept the pre-tail generation stamp while every
    pixel-affecting identity component (document, geometry, request key)
    matched, so the currency predicate rejected the frame and the watchdog
    declared an idle stall with nothing armed to recommit.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    n_side, n_tiles = 5, 4
    yy, xx, kk = np.mgrid[0:n_side, 0:n_side, 0:n_tiles]
    data = (xx + 10 * yy + 100 * kk).astype(np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(
            win.view_state.with_montage_axis(2, indices=tuple(range(n_tiles)), text=":")
        )
        win.render(reason="test-montage")
        _wait_for_committed_montage_tiles(win, qtbot, n_tiles, expected_shape=(n_side, n_side))
        assert win.view_state.image_axes == (0, 1)

        # Swap X/Y and let the transposed display settle.
        win.set_dimension_role("y", win.view_state.image_axes[1])
        assert win.view_state.image_axes == (1, 0)
        _wait_for_committed_montage_tiles(win, qtbot, n_tiles, expected_shape=(n_side, n_side))

        _issue_settle_tail_renders(win, qtbot)
        _assert_committed_frame_not_stale(win, qtbot, label="x/y swap settle")
    finally:
        win.close()


def test_indexed_dim_scroll_settle_leaves_committed_frame_current(qtbot):
    """Scenario B: a cropped indexed-dim montage scroll must settle clean.

    Field repro ``arrayscope-stall-445-6/7`` (signature session 445,
    ``committed_frame_stale=1``, 20 presented tiles, kernel idle, revision 60):
    a cropped montage (range crops on both image axes, swapped + flipped axes)
    scrolled a slice-range.  The maintainer confirmed the pixels render
    correctly -- the watchdog fired purely because the committed frame's
    render-generation stamp lagged the settle-tail renders while every
    pixel-affecting identity component matched.
    """

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    depth = 12
    data = np.arange(40 * 44 * depth, dtype=np.float32).reshape(40, 44, depth)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        montage_indices = tuple(range(0, depth, 2))
        state = (
            win.view_state.with_image_axes(1, 0)
            .with_montage_axis(2, indices=montage_indices, text="0::2")
            .with_axis_range(0, indices=tuple(range(4, 34, 2)), text="4:2:34")
            .with_axis_range(1, indices=tuple(range(2, 40, 2)), text="2:2:40")
        )
        win._set_view_state(state)
        win.render(reason="test-cropped-montage")
        _wait_for_committed_montage_tiles(win, qtbot, len(montage_indices))

        # Scroll the crop window on the first image axis (a slice-range scroll).
        scrolled = win.view_state.with_axis_range(0, indices=tuple(range(6, 36, 2)), text="6:2:36")
        win._set_view_state(scrolled)
        win.render(reason="slice-range")
        _wait_for_committed_montage_tiles(win, qtbot, len(montage_indices))

        _issue_settle_tail_renders(win, qtbot)
        _assert_committed_frame_not_stale(win, qtbot, label="indexed-dim scroll settle")
    finally:
        win.close()
