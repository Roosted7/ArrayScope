import os
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from pytestqt.exceptions import TimeoutError as QtBotTimeoutError

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import (
    assert_panel_invariants as _assert_panel_invariants,
    assert_size_close as _assert_size_close,
    clear_arrayscope_settings as _clear_arrayscope_settings,
    panel_body as _panel_body,
    process_events as _process_events,
    view_action as _view_action,
    wait_for_panel_preserve as _wait_for_panel_preserve,
)


def _tile_result(tile, value):
    image = np.full((tile.height, tile.width), value, dtype=np.float32)
    return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))


def _select_image_backend(name: str) -> None:
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", str(name))
    settings.sync()


def _tile_for_callback(win, call):
    requested_index = int(call["key"][-1])
    return next(
        tile
        for tile in win.renderer._frame_session.plan.tiles
        if int(tile.montage_index) == requested_index
    )


def _committed_tile_payload(win, tile):
    frame = getattr(win, "_committed_display_frame", None)
    value_source = getattr(frame, "value_source", None)
    payloads = getattr(value_source, "payloads", {})
    return payloads.get(int(tile.montage_index))


def _committed_tile_has_value(win, tile, value):
    payload = _committed_tile_payload(win, tile)
    if payload is None:
        return False
    expected = np.full((int(tile.height), int(tile.width)), value, dtype=np.float32)
    return np.array_equal(np.asarray(payload.image), expected)


def _display_levels(win):
    return tuple(float(value) for value in win.img_view.getLevels())


def _visible_backend_acknowledgements(win, backend):
    if backend == "pyqtgraph":
        return {
            int(tile_number): state.acknowledged_identity
            for tile_number, state in win.img_view._montage_tile_layer.states.items()
            if state.visible and state.item.isVisible()
        }
    layer = win.img_view._vispy_gpu_montage_layer
    pool = layer._pool
    drawn = {}
    for page_index, payloads in enumerate(layer._page_payloads_by_index):
        if (
            page_index >= len(layer._visuals_by_page)
            or not layer._visuals_by_page[page_index].visible
        ):
            continue
        for tile_number in payloads:
            resident_key = pool.tile_resident_keys.get(int(tile_number))
            if resident_key is not None:
                drawn[int(tile_number)] = pool.acknowledged_identities.get(
                    resident_key
                )
    return drawn


def _assert_view_contains_applied_montage_plan(win):
    plan = win.renderer._frame_session.plan
    height, width = tuple(int(value) for value in plan.display_shape[:2])
    view_range = win.img_view.getView().viewRange()
    assert view_range[0][0] <= 0.0
    assert view_range[0][1] >= float(width)
    assert view_range[1][0] <= 0.0
    assert view_range[1][1] >= float(height)


def _wait_for_committed_session_geometry(win, qtbot, *, expected_indices=None):
    try:
        qtbot.waitUntil(
            lambda: (
                win.renderer._frame_session is not None
                and win.renderer._frame_session.visible_plan_complete()
                and win.display_geometry.montage
                == win.renderer._frame_session.plan.geometry
                and (
                    expected_indices is None
                    or win.renderer._frame_session.plan.geometry.indices
                    == tuple(expected_indices)
                )
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
    except QtBotTimeoutError:
        session = win.renderer._frame_session
        level_stats = win.renderer._montage_level_stats_for_session(session)
        pytest.fail(
            "committed session geometry did not converge: "
            f"active={session.frame_plan.active_region_ids!r}, "
            f"unsettled={session.required_target_unsettled_tiles()!r}, "
            f"backend_ack={session.lifecycle.backend_presented_identities!r}, "
            f"geometry_matches={win.display_geometry.montage == session.plan.geometry!r}, "
            f"committed_indices={win.display_geometry.montage.indices!r}, "
            f"plan_indices={session.plan.geometry.indices!r}, "
            f"commit_outcome={win.renderer._last_montage_commit_outcome!r}, "
            f"display_committed={session.display_committed!r}, "
            f"level_rank={level_stats.rank!r}, "
            f"level_sources={level_stats.source_indices!r}, "
            f"semantic_evidence={session.semantic_level_evidence_diagnostics()!r}, "
            f"kernel={win.kernel.diagnostics()!r}, "
            f"stage_cache={win.operation_evaluator.stage_cache_diagnostics()!r}"
        )
    return win.renderer._frame_session


def _assert_committed_tile_value(win, tile, value):
    payload = _committed_tile_payload(win, tile)
    assert payload is not None
    expected = np.full((int(tile.height), int(tile.width)), value, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(payload.image), expected)


def _reset_warm_tile_state(win):
    """Return the window to a semantically cold tile state.

    The tiled pipeline renders plain slices through the montage tile lane and
    keeps their payloads resident (evaluator cache, retained payload store,
    committed frame).  Tests that pin the cold montage contract (loading
    placeholders, progressive level warm-up) must drop that warm state first,
    as a document/context change would.
    """

    win.operation_evaluator.clear_cache()
    win.renderer._retained_tiled_payload_store().clear_for_document_or_context_change("test-cold-start")
    # The slice render's session would re-acknowledge its payloads on the next
    # presentation commit; drop it so no warm tile state can be resurrected.
    win.renderer._frame_session = None
    frame = getattr(win, "_committed_display_frame", None)
    payloads = getattr(getattr(frame, "value_source", None), "payloads", None)
    if isinstance(payloads, dict):
        payloads.clear()


def _use_slice_zero(win, qtbot):
    win._set_view_state(win.view_state.with_slice(2, 0))
    win.render(reason="test-initial-slice")
    _process_events(qtbot, count=20)
    qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
    _reset_warm_tile_state(win)


def _settle_initial_render(win, qtbot):
    """Wait for the startup render (which flows through the montage tile lane
    since the tiled pipeline handles single slices too) and drop its warm
    tiles, so tests control exactly which tiles are warm and which calls the
    patched montage controller sees."""

    qtbot.waitUntil(lambda: getattr(win, "_committed_display_frame", None) is not None, timeout=3000)
    qtbot.waitUntil(lambda: not win.montage_tile_evaluation_controller.is_busy(), timeout=3000)
    _reset_warm_tile_state(win)


def test_montage_status_does_not_remain_computing(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.core.cache_status import CacheStatus
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        assert win.operation_evaluator.last_status.status != CacheStatus.COMPUTING
    finally:
        win.close()


def test_montage_visible_subset_hover_uses_source_index_not_local_tile_zero(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=float)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        win.img_view.getView().setRange(xRange=(0, 2), yRange=(7, 8), padding=0)
        win.update_image_view()
        _wait_for_committed_session_geometry(win, qtbot)
        tile_10 = win.renderer._frame_session.plan.tiles[10]
        qtbot.waitUntil(
            lambda: win.display_geometry.context_for_view_point(
                tile_10.x0 + 1,
                tile_10.y0 + 1,
            )
            is not None,
            timeout=3000,
        )
        context = win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1)

        assert context is not None
        assert context.context_text.endswith("d2=10")
    finally:
        win.close()


def test_panned_montage_hover_reads_committed_display_coordinates(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        win.img_view.getView().setRange(xRange=(0, 2), yRange=(7, 8), padding=0)
        win.update_image_view()

        _wait_for_committed_session_geometry(win, qtbot)
        tile_10 = win.renderer._frame_session.plan.tiles[10]
        qtbot.waitUntil(
            lambda: win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1)
            is not None,
            timeout=3000,
        )
        context = win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1)

        assert context is not None
        assert context.mapping.local_x == 1
        assert context.mapping.local_y == 1
        assert win.renderer._hover_value_from_display(context.mapping) == pytest.approx(10.0)
    finally:
        win.close()


def test_montage_update_after_shifted_origin_preserves_world_view_range(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)

        win.img_view.getView().setRange(xRange=(0, 2), yRange=(6, 8), padding=0)
        win.update_image_view()
        _wait_for_committed_session_geometry(win, qtbot)
        before = win.img_view.getView().viewRange()

        win.update_image_view()
        _process_events(qtbot, count=20)

        after = win.img_view.getView().viewRange()
        tile_10 = win.renderer._frame_session.plan.tiles[10]
        assert after[0] == pytest.approx(before[0])
        assert after[1] == pytest.approx(before[1])
        assert win.display_geometry.context_for_view_point(tile_10.x0 + 1, tile_10.y0 + 1) is not None
    finally:
        win.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_montage_tile_count_increase_preserves_manual_zoom_when_not_near_auto(qtbot, backend):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.model.tile_identity import acknowledged_identity_satisfies_target
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    source_plane = np.linspace(0.0, 1.0, 384 * 640, dtype=np.float32).reshape(384, 640)
    data = source_plane[:, :, None] + np.arange(20, dtype=np.float32)[None, None, :]
    win = ArrayScopeWindow(data)
    win.resize(1200, 820)
    win.show()
    qtbot.addWidget(win)
    try:
        qtbot.waitUntil(
            lambda: (
                win.img_view.graphicsView.viewport().width() >= 600
                and win.img_view.graphicsView.viewport().height() >= 400
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(5)), text=":"))
        win.render(reason="test-montage")
        _wait_for_committed_session_geometry(win, qtbot, expected_indices=range(5))
        win.img_view.getView().setRange(xRange=(0, 320), yRange=(0, 384), padding=0)
        qtbot.waitUntil(lambda: win.img_view.viewport_controller.mode.value == "user", timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        before = win.img_view.getView().viewRange()

        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage-more-tiles")
        session = _wait_for_committed_session_geometry(win, qtbot, expected_indices=range(20))

        view_range = win.img_view.getView().viewRange()
        assert view_range[0] == pytest.approx(before[0], abs=0.03)
        assert view_range[1] == pytest.approx(before[1], abs=0.03)
        # Full semantic-evidence convergence is an independent, lower-priority
        # side-work contract covered by test_semantic_level_evidence. It must
        # not be a prerequisite for validating camera continuity and physical
        # target acknowledgement here.
        active = set(int(tile) for tile in session.frame_plan.active_region_ids)
        assert active == {0}
        backend_visible = _visible_backend_acknowledgements(win, backend)
        assert active <= set(backend_visible)
        for tile_number, identity in backend_visible.items():
            lifecycle = session.lifecycle.peek(int(tile_number))
            assert lifecycle is not None and lifecycle.target is not None
            assert acknowledged_identity_satisfies_target(identity, lifecycle.target.identity)
    finally:
        win.close()
        settings.setValue("image_rendering_backend", "pyqtgraph")
        settings.sync()


def test_switching_to_larger_montage_auto_fits_when_tiles_would_be_hidden(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.img_view.getView().setRange(xRange=(0, 3), yRange=(0, 2), padding=0)

        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-switch-to-montage")
        _process_events(qtbot, count=80)

        _assert_view_contains_applied_montage_plan(win)
        assert win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel") is not None
    finally:
        win.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_pre_event_loop_complex_montage_eventually_fits_committed_plan(qtbot, backend):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    win = ArrayScopeWindow(np.arange(12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8))
    qtbot.addWidget(win)
    try:
        win.show()
        win.operation_coordinator.load_operations(
            (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))
        )
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()
        win._set_view_state(
            win.view_state.with_montage_axis(
                2,
                columns=3,
                indices=tuple(range(1, 7)),
                text="1:7",
            )
        )
        win.update_image_view()
        win.resize(1200, 820)

        qtbot.waitUntil(
            lambda: (
                win.renderer._frame_session is not None
                and win.renderer._frame_session.visible_plan_complete()
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        def committed_plan_is_fitted():
            session = win.renderer._frame_session
            height, width = tuple(int(value) for value in session.plan.display_shape[:2])
            view_range = win.img_view.getView().viewRange()
            return bool(
                view_range[0][0] <= 0.0
                and view_range[0][1] >= float(width)
                and view_range[1][0] <= 0.0
                and view_range[1][1] >= float(height)
            )

        qtbot.waitUntil(committed_plan_is_fitted, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        session = win.renderer._frame_session
        viewport = win.img_view.graphicsView.viewport()
        assert tuple(session.viewport_shape) == (viewport.height(), viewport.width())
    finally:
        win.close()
        settings.setValue("image_rendering_backend", "pyqtgraph")
        settings.sync()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_montage_viewport_resize_is_not_dropped_before_first_image_commit(
    qtbot,
    backend,
):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    win = ArrayScopeWindow(np.zeros((12, 10, 8), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win._set_view_state(
            win.view_state.with_montage_axis(
                2,
                columns=3,
                indices=tuple(range(1, 7)),
                text="1:7",
            )
        )
        win.img_view.image = None
        notifications = []
        win._on_image_viewport_resized = lambda **kwargs: notifications.append(kwargs)

        win.img_view.graphicsView.resizeEvent(
            QtGui.QResizeEvent(QtCore.QSize(741, 619), QtCore.QSize(447, 493))
        )

        assert len(notifications) == 1
        assert notifications[0]["previous_viewport_size"] is not None
        assert notifications[0]["base_view_range"] is not None
    finally:
        win.close()
        settings.setValue("image_rendering_backend", "pyqtgraph")
        settings.sync()


def test_montage_tile_count_increase_auto_adjusts_when_near_auto(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 12), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(10)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        action = win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel")
        if action is not None:
            action.linkActivated.emit("action")
            _process_events(qtbot, count=5)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(12)), text=":"))
        win.render(reason="test-montage-few-more-tiles")
        _process_events(qtbot, count=80)

        _assert_view_contains_applied_montage_plan(win)
    finally:
        win.close()


def test_montage_manual_resize_is_single_camera_transaction(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((20, 20, 30), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win.resize(900, 700)
        win.show()
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=None, indices=tuple(range(30)), text=":"))
        win.render(reason="test-montage-resize")
        _process_events(qtbot, count=80)
        view = win.img_view.getView()
        view.setRange(xRange=(-100.0, 300.0), yRange=(-100.0, 300.0), padding=0)
        _process_events(qtbot, count=5)
        win.img_view.viewport_controller.mode = ViewportMode.USER
        before_size = win.img_view.graphicsView.viewport().size()
        before = view.viewRange()
        range_changes = []
        view.sigRangeChanged.connect(lambda *_args: range_changes.append(view.viewRange()))

        win.resize(500, 700)
        _process_events(qtbot, count=20)

        after_size = win.img_view.graphicsView.viewport().size()
        after = view.viewRange()
        before_x_units = (before[0][1] - before[0][0]) / before_size.width()
        before_y_units = (before[1][1] - before[1][0]) / before_size.height()
        after_x_units = (after[0][1] - after[0][0]) / after_size.width()
        after_y_units = (after[1][1] - after[1][0]) / after_size.height()
        assert after_x_units == pytest.approx(before_x_units)
        assert after_y_units == pytest.approx(before_y_units)
        assert range_changes == []
    finally:
        win.close()


def test_montage_auto_fit_skips_when_fit_mode_is_enabled(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(5)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        action = win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel")
        if action is not None:
            action.linkActivated.emit("action")
            _process_events(qtbot, count=5)
        win.display_toolbar.fit_action.trigger()
        _process_events(qtbot, count=10)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT

        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage-fit-mode")
        _process_events(qtbot, count=80)

        assert win.img_view.viewport_controller.mode == ViewportMode.FIT
        assert win.statusBar().findChild(QtWidgets.QLabel, "ArrayScopeStatusActionLabel") is None
    finally:
        win.close()


def test_montage_interactive_render_commits_tiled_frame(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 8, dtype=float).reshape(2, 3, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.render(reason="test-montage")
        qtbot.waitUntil(lambda: getattr(getattr(win, "_committed_display_frame", None), "scene", None) is not None, timeout=3000)

        frame = win._committed_display_frame
        assert frame.is_tiled
        assert frame.scene.resident_region_ids
    finally:
        win.close()


def test_montage_commits_cached_tiles_immediately_with_loading_placeholders(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    _select_image_backend("pyqtgraph")
    from arrayscope.display.montage import MontageTileState, make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _settle_initial_render(win, qtbot)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        win._set_view_state(state)
        plan = make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3)
        win.operation_evaluator.store_montage_tile_result(plan.tiles[0], montage_axis=2, colormap_lut=None, result=_tile_result(plan.tiles[0], 10))
        win.update_image_view()

        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=3000,
        )

        states = win.renderer._frame_session.ensure_tile_states()
        assert states[0] == MontageTileState.LOADED
        assert all(state == MontageTileState.LOADED for state in states)
        _assert_committed_tile_value(win, plan.tiles[0], 10)
    finally:
        win.close()


def test_montage_loading_overlay_clears_after_final_delayed_commit(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()
        win.renderer._show_frame_session_loading_overlay(win.renderer._frame_session)
        assert win.img_view._evaluation_overlay.isVisible()

        qtbot.waitUntil(
            lambda: bool(
                win.renderer._frame_session.visible_plan_complete()
                and (
                    win.img_view._evaluation_overlay is None
                    or not win.img_view._evaluation_overlay.isVisible()
                )
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
        assert win.renderer._frame_session.visible_plan_complete()
    finally:
        win.close()


def test_montage_ready_display_payloads_commit_immediately(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.stage_fanin import StageFanInState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    session = SimpleNamespace(
        session_id=999,
        key=("test-session",),
        render_generation=win.renderer._capture_render_generation(),
        stage_fan_in=StageFanInState(),
        final_commit_pending=False,
        flush_pending=False,
        pending_removals=set(),
        dirty_payloads={1: None},
    )
    calls = []
    try:
        _process_events(qtbot)
        _settle_initial_render(win, qtbot)
        monkeypatch.setattr(
            win.renderer,
            "_is_current_frame_session",
            lambda session_id, key: session_id == 999 and key == ("test-session",),
        )
        monkeypatch.setattr(
            win.renderer,
            "commit_frame_session_presentation",
            lambda _session: calls.append(
                (
                    bool(session.final_commit_pending),
                    bool(session.flush_pending),
                )
            ),
        )
        win.renderer._frame_session = session

        win.renderer.apply_ready_montage_display(session)

        assert session.final_commit_pending
        assert session.flush_pending
        assert calls
        assert calls[0] == (True, True)
    finally:
        win.close()


def test_montage_active_tiles_are_all_accounted_for(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 12, dtype=np.float32).reshape(2, 2, 12))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(12)), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: win.renderer._frame_session is not None, timeout=3000)

        active_ids = set(win.renderer._frame_session.visible_tile_numbers)
        qtbot.waitUntil(
            lambda: all(
                win.renderer._frame_session.ensure_tile_states()[int(tile.montage_index)] != MontageTileState.UNLOADED
                for tile in win.renderer._frame_session.plan.tiles
                if int(tile.montage_index) in active_ids
            ),
            timeout=3000,
        )
        states = win.renderer._frame_session.ensure_tile_states()
        for tile in win.renderer._frame_session.plan.tiles:
            if int(tile.montage_index) in active_ids:
                assert states[tile.montage_index] in {
                    MontageTileState.LOADED,
                    MontageTileState.LOADING,
                    MontageTileState.SKIPPED,
                }
    finally:
        win.close()


def test_montage_pan_schedules_viewport_update(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 12, dtype=np.float32).reshape(2, 2, 12))
    qtbot.addWidget(win)
    calls = []
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(12)), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: getattr(win.renderer._frame_session, "display_committed", False), timeout=3000)
        monkeypatch.setattr(win.renderer, "retarget_montage_viewport", lambda: calls.append("retargeted"))

        win.img_view.getView().setRange(xRange=(6, 9), yRange=(0, 2), padding=0)

        qtbot.waitUntil(lambda: bool(calls), timeout=1000)
    finally:
        win.close()


def test_cached_montage_tile_rebinds_to_current_layout(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    _select_image_backend("pyqtgraph")
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        old_state = win.view_state.with_montage_axis(2, columns=1, indices=(0, 1, 2), text=":")
        old_plan = make_montage_plan(old_state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=1)
        win.operation_evaluator.store_montage_tile_result(old_plan.tiles[1], montage_axis=2, colormap_lut=None, result=_tile_result(old_plan.tiles[1], 11))
        new_state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        win._set_view_state(new_state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=3000,
        )

        new_tile = win.renderer._frame_session.plan.tiles[1]
        _assert_committed_tile_value(win, new_tile, 11)
    finally:
        win.close()


def test_montage_skipped_tiles_show_detailed_warning(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((2, 2, 2), dtype=np.float32))
    qtbot.addWidget(win)
    messages = []
    monkeypatch.setattr(render_module.Qt.QtWidgets.QMessageBox, "warning", lambda _parent, _title, message: messages.append(message))
    try:
        _process_events(qtbot)
        win.renderer._warn_montage_tiles_skipped(
            skipped_count=2,
            tile_bytes=2048,
            budget_bytes=1024,
            tile_shape=(16, 16),
        )

        assert messages
        assert "skipped 2 tile" in messages[0]
        assert "over the visible render budget" in messages[0]
        assert "Tile shape is (16, 16)" in messages[0]
    finally:
        win.close()


def test_montage_auto_window_button_applies_current_semantic_bounds_immediately(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=80)
        expected = (0.0, 11.0)
        original_session = win.renderer._frame_session

        win.img_view.setLevels(2.0, 8.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (2.0, 8.0)

        win.auto_window_levels()

        assert win.renderer._frame_session is original_session
        assert win.renderer._explicit_user_level_source is None
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == expected
        _process_events(qtbot, count=20)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == expected
    finally:
        win.close()


def test_montage_zoom_in_does_not_shrink_level_source_coverage(qtbot, monkeypatch):
    from arrayscope.window.frame_effects import FramePipelineEffects

    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    tile_count = 6
    win = ArrayScopeWindow(np.arange(2 * 2 * tile_count, dtype=np.float32).reshape(2, 2, tile_count))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=6, indices=tuple(range(tile_count)), text=":"))
        win.update_image_view()

        session = win.renderer._frame_session
        effects = FramePipelineEffects(win.renderer, session)
        for index, tile in enumerate(session.plan.tiles):
            effects.admit_tile_result(tile, _tile_result(tile, 100 * (index + 1)))
        win.renderer.apply_montage_presentation(session)
        _process_events(qtbot)

        before_levels = tuple(round(float(value), 6) for value in win.img_view.getLevels())
        before_histogram = tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds())
        assert session.applied_level_source.source_count == tile_count
        assert session.applied_level_source.expected_count == tile_count

        win.img_view.getView().setRange(xRange=(0.0, 2.0), yRange=(0.0, 2.0), padding=0)
        win.renderer.apply_montage_viewport_retarget()
        _process_events(qtbot)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == before_levels
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == before_histogram
        assert win.renderer._frame_session.applied_level_source.source_count == tile_count
        assert win.renderer._frame_session.applied_level_source.expected_count == tile_count
    finally:
        win.close()


def test_enabling_montage_with_cached_tile_preserves_relative_window_fractions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    _select_image_backend("pyqtgraph")
    from arrayscope.core.window_levels import relative_levels
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _use_slice_zero(win, qtbot)
        win.img_view.setHistogramDataBounds((0.0, 19.0))
        win.img_view.setLevels(5.0, 15.0)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(1, 2), text="1:3")
        plan = make_montage_plan(state, axis=2, indices=(1, 2), tile_shape=(4, 5), columns=3)
        tile = plan.tiles[0]
        win.operation_evaluator.store_montage_tile_result(tile, montage_axis=2, colormap_lut=None, result=_tile_result(tile, 100.0))
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=3000,
        )

        bounds = tuple(float(value) for value in win.img_view.getHistogramDataBounds())
        expected = relative_levels((5.0, 15.0), (0.0, 19.0), bounds)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == tuple(
            round(float(value), 6) for value in expected
        )
        assert tuple(round(float(value), 6) for value in bounds) == (99.0, 219.0)
    finally:
        win.close()


def test_shifting_montage_range_preserves_relative_window_fractions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    _select_image_backend("pyqtgraph")
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)

    def result_for(tile, image):
        image = np.asarray(image, dtype=np.float32)
        return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))

    try:
        _process_events(qtbot)
        _use_slice_zero(win, qtbot)
        win.img_view.setHistogramDataBounds((0.0, 19.0))
        win.img_view.setLevels(5.0, 15.0)
        first_state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text="0:2")
        first_plan = make_montage_plan(first_state, axis=2, indices=(0, 1), tile_shape=(4, 5), columns=2)
        for tile in first_plan.tiles:
            win.operation_evaluator.store_montage_tile_result(tile, montage_axis=2, colormap_lut=None, result=result_for(tile, data[:, :, tile.source_index]))
        win._set_view_state(first_state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (0.0, 119.0),
            timeout=3000,
        )
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (31.315789, 93.947368)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (0.0, 119.0)

        shifted_state = win.view_state.with_montage_axis(2, columns=2, indices=(1, 2), text="1:3")
        win._set_view_state(shifted_state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (131.315789, 193.947368),
            timeout=3000,
        )

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (131.315789, 193.947368)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (100.0, 219.0)
    finally:
        win.close()


def test_cached_montage_commit_uses_all_loaded_tiles_for_initial_histogram(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    _select_image_backend("pyqtgraph")
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    tile_count = 6
    win = ArrayScopeWindow(np.zeros((2, 2, tile_count), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=tile_count, indices=tuple(range(tile_count)), text=":")
        plan = make_montage_plan(state, axis=2, indices=tuple(range(tile_count)), tile_shape=(2, 2), columns=tile_count)
        values = (0.0, 0.0, 0.0, 0.0, 1000.0, 1200.0)
        for tile, value in zip(plan.tiles, values):
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, value),
            )
        win.img_view.getView().setRange(xRange=(0, 20), yRange=(0, 2), padding=0)
        win._set_view_state(state)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: win.renderer._frame_session.visible_plan_complete(),
            timeout=3000,
        )

        bounds = tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds())
        assert bounds[1] > 1000.0
        assert win.renderer._frame_session.applied_level_source.source_count == tile_count
        assert win.renderer._frame_session.applied_level_source.expected_count == tile_count
    finally:
        win.close()


def test_fft_montage_stage_cache_hit_keeps_per_tile_slab_plans(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.operations.regions import region_text
    from arrayscope.operations.stage_cache import StageValue
    from arrayscope.operations.stage_materialization import StageMaterializationResult
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **_kwargs: 1)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"))

        materializer = win.operation_evaluator.stage_materializer

        def cached_stage(document_key, candidate):
            key = materializer.key_for_candidate(document_key, candidate)
            data = np.zeros(tuple(candidate.shape), dtype=np.dtype(candidate.dtype))
            value = StageValue(
                data=data,
                region=candidate.region,
                stage_index=int(candidate.stage_index),
                nbytes=int(data.nbytes),
                priority=str(candidate.priority),
            )
            return StageMaterializationResult("hit", key, value=value)

        monkeypatch.setattr(materializer, "request_stage", cached_stage)

        win.update_image_view()

        session = win.renderer._frame_session
        planned_regions = [
            region_text(session.stage_fan_in.tile_stage_plans[index].region_plan.final_region)
            for index in range(4)
        ]

        assert len(set(planned_regions)) == 4
    finally:
        win.close()


def test_operation_backed_complex_montage_tile_layer_rewindows_rgb_from_histogram_levels(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()
    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=0),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()

        qtbot.waitUntil(lambda: getattr(win.renderer._frame_session, "display_committed", False), timeout=3000)
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=3000)
        assert any(state.rgb_base is not None for state in win.img_view._montage_tile_layer.states.values())

        low, high = win.img_view.getHistogramDataBounds()
        desired = ((float(low) + float(high)) / 2.0, float(high))
        # A user command must supersede automatic level work still attached to
        # an otherwise committed progressive montage session.
        win.renderer._frame_session.force_auto = True
        win.img_view.histogram.setLevels(*desired)
        win.img_view._on_histogram_level_change_finished()
        assert win.renderer._frame_session.force_auto is False
        assert win.renderer._frame_session.level_generation.target_levels == desired
        qtbot.waitUntil(
            lambda: all(tuple(state.levels) == desired for state in win.img_view._montage_tile_layer.states.values()),
            timeout=1000,
        )
        assert win.renderer._frame_session.has_stale_level_presentations() is False
    finally:
        win.close()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
@pytest.mark.parametrize(
    "transition",
    ("operation", "channel-real", "complex-mode", "axes"),
)
def test_semantic_montage_transition_never_leaves_old_tiles_visible(
    qtbot,
    backend,
    transition,
):
    _clear_arrayscope_settings()
    if backend == "vispy" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip(
            "physical VisPy transition acknowledgement requires a real OpenGL display"
        )
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.core.view_state import ChannelMode
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.display.model.tile_identity import acknowledged_identity_satisfies_target
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT, FFTShift
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()
    data = np.arange(12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        if image_view_backend_capabilities(win.img_view).name != backend:
            pytest.skip(f"{backend} backend unavailable in this Qt environment")
        _process_events(qtbot)
        if transition != "operation":
            win.operation_coordinator.load_operations(
                (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))
            )
            win._set_document(win.operation_coordinator.document)
            win._coerce_channel_for_current_dtype()
            if transition == "complex-mode":
                win._set_view_state(win.view_state.with_channel(ChannelMode.ABS))
        state = win.view_state.with_montage_axis(
            2,
            columns=3,
            indices=tuple(range(1, 7)),
            text="1:7",
        )
        win._set_view_state(state)
        win.update_image_view()
        try:
            qtbot.waitUntil(
                lambda: bool(
                    getattr(win.renderer._frame_session, "display_committed", False)
                    and win.renderer._frame_session.visible_plan_complete()
                    and set(_visible_backend_acknowledgements(win, backend))
                    == set(range(6))
                ),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
        except Exception:
            stalled = win.renderer._frame_session
            pytest.fail(
                repr(
                    {
                        "display_committed": stalled.display_committed,
                        "visible_complete": stalled.visible_plan_complete(),
                        "required": sorted(stalled.required_tile_numbers()),
                        "unsettled": sorted(stalled.required_target_unsettled_tiles()),
                        "pending": sorted(stalled.pending_tiles),
                        "active_requests": sorted(stalled.active_tile_requests),
                        "dirty": sorted(stalled.dirty_payloads),
                        "acks": sorted(_visible_backend_acknowledgements(win, backend)),
                        "draw_pending": bool(win.img_view.presentationDrawPending()),
                        "stale_levels": stalled.has_stale_level_presentations(),
                        "level_snapshot": stalled.level_presentation_snapshot(),
                        "level_target": stalled.level_generation.target_levels,
                        "applied_levels": getattr(
                            stalled.applied_level_source, "levels", None
                        ),
                        "lifecycle": stalled.lifecycle.counters(),
                    }
                )
            )
        assert set(_visible_backend_acknowledgements(win, backend)) == set(range(6))
        previous = win.renderer._frame_session

        if transition == "operation":
            win.operation_coordinator.load_operations(
                (CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2))
            )
            win._set_document(win.operation_coordinator.document)
            win._coerce_channel_for_current_dtype()
        elif transition == "channel-real":
            win._set_view_state(win.view_state.with_channel(ChannelMode.REAL))
        elif transition == "complex-mode":
            win._set_view_state(win.view_state.with_channel(ChannelMode.COMPLEX))
        else:
            win._set_view_state(win.view_state.transposed_image_axes())
        win.update_image_view()

        qtbot.waitUntil(
            lambda: win.renderer._frame_session is not previous,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        current = win.renderer._frame_session
        assert current.semantic_key != previous.semantic_key
        for tile_number, acknowledged_identity in _visible_backend_acknowledgements(
            win,
            backend,
        ).items():
            lifecycle = current.lifecycle.peek(int(tile_number))
            assert lifecycle is not None and lifecycle.target is not None
            assert acknowledged_identity_satisfies_target(
                acknowledged_identity,
                lifecycle.target.identity,
            )

        qtbot.waitUntil(current.visible_plan_complete, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        final_acknowledgements = _visible_backend_acknowledgements(win, backend)
        assert set(final_acknowledgements) == set(current.visible_tile_numbers), {
            "physical": set(final_acknowledgements),
            "visible_targets": set(current.visible_tile_numbers),
            "lifecycle_presented": set(current.lifecycle.presented_tiles),
            "backend_identities": set(
                current.lifecycle.backend_presented_identities
            ),
            "rows": current.diagnostic_tile_identity_rows(
                limit=len(current.visible_tile_numbers),
                include_all_visible=True,
            ),
        }
        rows = {
            int(row["tile"]): row
            for row in current.diagnostic_tile_identity_rows(
                limit=len(current.visible_tile_numbers),
                include_all_visible=True,
            )
        }
        for tile_number, acknowledged_identity in final_acknowledgements.items():
            lifecycle = current.lifecycle.peek(int(tile_number))
            assert lifecycle is not None and lifecycle.target is not None
            assert acknowledged_identity_satisfies_target(
                acknowledged_identity,
                lifecycle.target.identity,
            )
            assert rows[tile_number]["drawable"] is True
            assert rows[tile_number]["target_source"] == acknowledged_identity.source_index
            assert (
                rows[tile_number]["acknowledged_source"]
                == acknowledged_identity.source_index
            )
    finally:
        win.close()
        settings.setValue(
            "image_rendering_backend",
            ImageRenderingBackendChoice.PYQTGRAPH.value,
        )
        settings.sync()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_viewport_montage_retarget_never_leaves_old_tiles_visible(qtbot, backend):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.display.model.tile_identity import acknowledged_identity_satisfies_target
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.sync()

    data = np.arange(12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        if image_view_backend_capabilities(win.img_view).name != backend:
            pytest.skip(f"{backend} backend unavailable in this Qt environment")
        _process_events(qtbot)
        initial = win.view_state.with_montage_axis(2, columns=3, indices=tuple(range(6)), text="0:6")
        win._set_view_state(initial)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: bool(getattr(win.renderer._frame_session, "display_committed", False)),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert _visible_backend_acknowledgements(win, backend)
        previous = win.renderer._frame_session
        previous_semantic_key = previous.semantic_key

        retargeted = win.view_state.with_montage_axis(
            2,
            columns=3,
            indices=tuple(range(2, 8)),
            text="2:8",
        )
        win._set_view_state(retargeted)
        win.update_image_view()

        qtbot.waitUntil(
            lambda: tuple(
                int(tile.source_index)
                for tile in win.renderer._frame_session.plan.tiles
            )
            == tuple(range(2, 8)),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        current = win.renderer._frame_session
        qtbot.waitUntil(
            lambda: current.required_target_settled(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert current is previous
        assert current.semantic_key == previous_semantic_key
        for tile_number, acknowledged_identity in _visible_backend_acknowledgements(
            win,
            backend,
        ).items():
            lifecycle = current.lifecycle.peek(int(tile_number))
            assert lifecycle is not None and lifecycle.target is not None
            assert acknowledged_identity_satisfies_target(
                acknowledged_identity,
                lifecycle.target.identity,
            )
    finally:
        win.close()
        settings.setValue(
            "image_rendering_backend",
            ImageRenderingBackendChoice.PYQTGRAPH.value,
        )
        settings.sync()


@pytest.mark.parametrize("backend", ("pyqtgraph", "vispy"))
def test_one_index_source_window_retarget_remaps_59_without_black_frame(
    qtbot,
    monkeypatch,
    backend,
):
    """R8C.1: a 100:160 -> 101:161 shift is one placement transaction."""

    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
    from arrayscope.display.backend_contract import image_view_backend_capabilities
    from arrayscope.display.model.tile_identity import acknowledged_identity_satisfies_target
    from arrayscope.display.backends import surface_for_view
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", backend)
    settings.setValue("montage_quality_policy", "resident")
    settings.sync()

    if backend == "vispy" and os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        pytest.skip("physical VisPy source-window continuity requires a real OpenGL display")

    data = np.empty((6, 8, 161), dtype=np.float32)
    for source_index in range(data.shape[2]):
        data[..., source_index] = float(source_index)

    win = ArrayScopeWindow(data)
    win.resize(1200, 700)
    win.show()
    qtbot.addWidget(win)
    try:
        if image_view_backend_capabilities(win.img_view).name != backend:
            pytest.skip(f"{backend} backend unavailable in this Qt environment")
        initial_indices = tuple(range(100, 160))
        target_indices = tuple(range(101, 161))
        initial = win.view_state.with_montage_axis(
            2,
            columns=10,
            indices=initial_indices,
            text="100:160",
        )
        win._set_view_state(initial)
        win.update_image_view()
        qtbot.waitUntil(
            lambda: bool(
                win.renderer._frame_session is not None
                and win.renderer._frame_session.visible_plan_complete()
                and len(_visible_backend_acknowledgements(win, backend)) == 60
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        win.img_view.setLevels(-1.0, 200.0)
        qtbot.waitUntil(
            lambda: not win.renderer._frame_session.has_stale_level_presentations(),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        surface = surface_for_view(win.img_view)
        original_invalidate = surface.invalidate_tiled_presentation
        original_present = surface.present_tiled
        observations = []
        reports = []
        tracking = {"active": False}

        def record_physical(phase):
            if not tracking["active"]:
                return
            acknowledgements = _visible_backend_acknowledgements(win, backend)
            compatible = 0
            incompatible = 0
            for tile_number, identity in acknowledgements.items():
                expected_source = target_indices[int(tile_number)]
                if int(getattr(identity, "source_index", -1)) == expected_source:
                    compatible += 1
                else:
                    incompatible += 1
            observation = (str(phase), len(acknowledgements), compatible, incompatible)
            if not observations or observations[-1] != observation:
                observations.append(observation)

        def invalidate(reason):
            result = original_invalidate(reason)
            record_physical(f"invalidate:{reason}")
            return result

        def present(presentation):
            report = original_present(presentation)
            if tracking["active"]:
                reports.append(report)
                record_physical("present")
            return report

        monkeypatch.setattr(surface, "invalidate_tiled_presentation", invalidate)
        monkeypatch.setattr(surface, "present_tiled", present)

        evaluations_before = int(win.operation_evaluator.image_evaluations)
        tracking["active"] = True
        shifted = win.view_state.with_montage_axis(
            2,
            columns=10,
            indices=target_indices,
            text="101:161",
        )
        win._set_view_state(shifted)
        win.update_image_view()

        try:
            qtbot.waitUntil(
                lambda: bool(
                    tuple(
                        int(tile.source_index)
                        for tile in win.renderer._frame_session.plan.tiles
                    )
                    == target_indices
                    and win.renderer._frame_session.visible_plan_complete()
                    and len(_visible_backend_acknowledgements(win, backend)) == 60
                ),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
        except Exception:
            stalled = win.renderer._frame_session
            pytest.fail(
                repr(
                    {
                        "plan": tuple(int(tile.source_index) for tile in stalled.plan.tiles),
                        "complete": stalled.visible_plan_complete(),
                        "acknowledged": len(_visible_backend_acknowledgements(win, backend)),
                        "dirty": tuple(stalled.dirty_payloads),
                        "upserts": tuple(stalled.pending_payload_upserts),
                        "removals": tuple(stalled.pending_removals),
                        "source_window": stalled.source_window_changed_pending,
                        "atomic_reject": getattr(stalled, "_atomic_fast_reject_reason", None),
                        "unsettled": stalled.required_target_unsettled_tiles(),
                        "first_pass": (
                            stalled.first_pass_quality,
                            stalled.first_pass_histogram_published,
                            stalled.first_pass_pixels_presented(),
                        ),
                        "ladder_states": getattr(stalled.pipeline, "last_plan_states", None),
                        "ladder_steps": getattr(stalled.pipeline, "last_plan_steps", None),
                        "rung_pending": tuple(stalled.pending_rung_materializations),
                        "active": tuple(stalled.active_tile_requests),
                        "identity_rows": stalled.diagnostic_tile_identity_rows(
                            limit=4,
                            include_all_visible=False,
                        ),
                        "observations": observations,
                        "report_count": len(reports),
                    }
                )
            )
        current = win.renderer._frame_session

        assert current.tile_compute_cache_hits >= 59
        assert int(win.operation_evaluator.image_evaluations) - evaluations_before <= 1
        assert observations
        assert min(compatible for _phase, _visible, compatible, _bad in observations) >= 59, observations
        assert max(60 - compatible for _phase, _visible, compatible, _bad in observations) <= 1, observations
        assert all(incompatible == 0 for _phase, _visible, _compatible, incompatible in observations), observations

        acknowledgements = _visible_backend_acknowledgements(win, backend)
        assert len(acknowledgements) == 60
        for tile_number, acknowledged_identity in acknowledgements.items():
            lifecycle = current.lifecycle.peek(int(tile_number))
            assert lifecycle is not None and lifecycle.target is not None
            assert acknowledged_identity_satisfies_target(
                acknowledged_identity,
                lifecycle.target.identity,
            )

        assert current.plan.geometry.indices == target_indices
        tile = current.plan.tiles[0]
        context = win.display_geometry.context_for_view_point(tile.x0 + 1, tile.y0 + 1)
        assert context is not None
        assert context.context_text.endswith("d2=101")
        assert win.renderer._hover_value_from_display(context.mapping) == pytest.approx(101.0)

        frame = win._committed_display_frame
        assert frame.value_source.payloads[0].source_index == 101
        assert frame.value_source.value_at(context.mapping) == pytest.approx(101.0)
        roi = RoiSelection(
            "source-window-roi",
            "Source window ROI",
            RoiGeometry(
                RoiKind.RECTANGLE,
                rect=(float(tile.x0), float(tile.y0), 2.0, 2.0),
            ),
        )
        roi_values = win._committed_tiled_roi_values((roi,), collect_histograms=False)
        assert roi_values is not None
        assert roi_values[0][roi.id][1].mean == pytest.approx(101.0)
    finally:
        win.close()
        settings.setValue(
            "image_rendering_backend",
            ImageRenderingBackendChoice.PYQTGRAPH.value,
        )
        settings.sync()


def test_large_complex_montage_auto_uses_tile_layer(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.ones((840, 840, 3), dtype=np.complex64)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()

        qtbot.waitUntil(lambda: getattr(win.renderer._frame_session, "display_committed", False), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)

        assert win.img_view.montageDisplayMode().endswith("tile_layer")
    finally:
        win.close()


def test_large_complex_montage_tile_layer_histogram_drag_does_not_update_base_image_item(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.PYQTGRAPH.value)
    settings.sync()
    data = np.ones((840, 840, 3), dtype=np.complex64)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_image_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        qtbot.waitUntil(lambda: bool(win.img_view._montage_tile_layer.states), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        qtbot.waitUntil(
            lambda: (
                not getattr(win.renderer._frame_session, "dirty_payloads", {})
                and not getattr(win.renderer._frame_session, "pending_payload_upserts", {})
                and not getattr(win.renderer._frame_session, "pending_removals", set())
                and not getattr(win.renderer._frame_session, "active_tile_requests", set())
            ),
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )

        def fail_base_image_item_update(*args, **kwargs):
            raise AssertionError("base ImageItem update during tile-layer histogram drag")

        monkeypatch.setattr(win.img_view.imageItem, "setImage", fail_base_image_item_update)
        low, high = win.img_view.getHistogramDataBounds()
        win.img_view.histogram.setLevels((float(low) + float(high)) / 2.0, float(high))
        # Wait for the coalesced preview's actual tile-layer work. A fixed
        # qWait races the 33 ms timer under xdist and makes the last timing
        # channel depend on host load.
        qtbot.waitUntil(
            lambda: win.img_view.lastImageUploadTiming().tile_layer_visible_items > 0,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        win.img_view._on_histogram_level_change_finished()

        timing = win.img_view.lastImageUploadTiming()
        # The coalesced preview and its bounded tile continuation have distinct
        # timing channels, so either may be the last record. The semantic mode
        # is the surface state, not whichever nested timing scope finished last.
        assert win.img_view.montageDisplayMode() == "tile_layer"
        assert timing.tile_layer_visible_items > 0
        assert timing.tile_layer_items_updated <= timing.tile_layer_visible_items
        assert timing.tile_layer_texture_uploads == 0
    finally:
        win.close()


def test_visible_render_budget_uses_app_setting():
    from types import SimpleNamespace

    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window.render import RenderOrchestrator

    renderer = RenderOrchestrator.__new__(RenderOrchestrator)
    renderer.win = SimpleNamespace(app_settings=AppSettingsState(render_memory_budget_mb=256))

    assert renderer._visible_render_budget_bytes() == 256 * 1024 * 1024
