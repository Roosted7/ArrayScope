import numpy as np
import pytest
from dataclasses import replace

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
from tests.ui.helpers import clear_arrayscope_settings, process_events


def _tile_result(tile, value):
    image = np.full((tile.height, tile.width), value, dtype=np.float32)
    return EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes))


def test_rapid_slice_burst_is_coalesced_and_latest(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win, "render", lambda **kwargs: calls.append((kwargs, win.view_state.slice_indices[2])))
    try:
        win._on_slice_index_changed(2, 1)
        win._on_slice_index_changed(2, 2)
        win._on_slice_index_changed(2, 3)

        assert calls == []
        qtbot.waitUntil(lambda: bool(calls), timeout=500)

        assert len(calls) < 3
        assert calls[-1][1] == 3
    finally:
        win.close()


def test_slice_control_updates_before_render_completion(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    render_calls = []

    def blocked_render(**kwargs):
        render_calls.append((kwargs, win.view_state.slice_indices[2]))

    monkeypatch.setattr(win, "render", blocked_render)
    try:
        win._on_slice_index_changed(2, 4)

        assert render_calls == []
        assert win.view_state.slice_indices[2] == 4
        assert win.widgets["spins"]["slice_indices"][2].value() == 4
        assert win.dimension_strip.chip(2).slice_edit.text() == "4"

        qtbot.waitUntil(lambda: bool(render_calls), timeout=500)
        assert render_calls[-1][1] == 4
    finally:
        win.close()


def test_rapid_scroll_latest_control_state_not_blocked_by_slow_commit(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 8, dtype=np.float32).reshape(4, 5, 8))
    qtbot.addWidget(win)
    render_calls = []

    def recording_render(**kwargs):
        render_calls.append((kwargs.get("reason"), win.view_state.slice_indices[2]))

    monkeypatch.setattr(win, "render", recording_render)
    try:
        for index in (1, 2, 3, 6):
            win._on_slice_index_changed(2, index)
            assert win.view_state.slice_indices[2] == index
            assert win.widgets["spins"]["slice_indices"][2].value() == index
            assert win.dimension_strip.chip(2).slice_edit.text() == str(index)

        assert render_calls == []
        qtbot.waitUntil(lambda: bool(render_calls), timeout=500)
        assert render_calls[-1] == ("slice", 6)
        assert len(render_calls) < 4
    finally:
        win.close()


def test_hot_cached_montage_schedules_no_tile_evaluation(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
            )
        calls = []
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))

        win._set_view_state(state)
        win.update_montage_view()

        qtbot.waitUntil(lambda: win._current_montage_canvas is not None, timeout=250)
        assert calls == []
        assert win._montage_cached_tiles_last_session == 2
        assert win._montage_missing_tiles_last_session == 0
    finally:
        win.close()


def test_hot_cached_tile_layer_clean_flush_updates_zero_items(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.app.settings_state import MontageDisplayBackendChoice
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":")
        plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
        for tile in plan.tiles:
            win.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=2,
                colormap_lut=None,
                result=_tile_result(tile, int(tile.source_index) + 1),
            )
        calls = []
        monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
        win.app_settings = replace(win.app_settings, montage_display_backend=MontageDisplayBackendChoice.TILE_LAYER)

        win._set_view_state(state)
        win.update_montage_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=500)
        first_sources = {tile: state.source_array_id for tile, state in win.img_view._montage_tile_layer.states.items()}

        win.update_montage_view()
        timing = win.img_view.lastImageUploadTiming()
        second_sources = {tile: state.source_array_id for tile, state in win.img_view._montage_tile_layer.states.items()}

        assert calls == []
        assert second_sources == first_sources
        assert all(str(source[0]) == "montage_tile" for source in second_sources.values())
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 2
        assert timing.tile_layer_upload_ms == 0.0
        assert timing.visible_bytes == 0

        win._commit_montage_session_canvas(win._montage_session, force=True)
        timing = win.img_view.lastImageUploadTiming()

        assert calls == []
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 2
        assert timing.tile_layer_upload_ms == 0.0
        assert timing.visible_bytes == 0
    finally:
        win.close()


def test_vispy_montage_pyqtgraph_range_change_schedules_viewport_tile_update(qtbot, monkeypatch):
    pytest.importorskip("vispy")

    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice, MontageDisplayBackendChoice
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
    settings.sync()

    win = ArrayScopeWindow(np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8))
    qtbot.addWidget(win)
    scheduled = []
    try:
        win.app_settings = replace(win.app_settings, montage_display_backend=MontageDisplayBackendChoice.TILE_LAYER)
        process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.update_montage_view()
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "vispy_tile_layer", timeout=3000)
        monkeypatch.setattr(win, "_schedule_montage_viewport_update", lambda: scheduled.append(win.img_view.getView().viewRange()))

        assert win.img_view._vispy_canvas_native.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        win.img_view.getView().setRange(xRange=(0.0, 4.0), yRange=(0.0, 2.0), padding=0)
        process_events(qtbot)

        assert win.img_view.rendering_backend_name == "vispy"
        assert scheduled
    finally:
        win.close()
        clear_arrayscope_settings()


def test_vispy_montage_view_range_change_expands_visible_tile_set(qtbot, monkeypatch):
    pytest.importorskip("vispy")

    clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import ImageRenderingBackendChoice, MontageDisplayBackendChoice
    from arrayscope.window import ArrayScopeWindow

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.VISPY.value)
    settings.sync()

    win = ArrayScopeWindow(np.arange(4 * 100 * 8, dtype=np.float32).reshape(4, 100, 8))
    qtbot.addWidget(win)
    try:
        win.resize(360, 240)
        win.show()
        win.app_settings = replace(win.app_settings, montage_display_backend=MontageDisplayBackendChoice.TILE_LAYER)
        process_events(qtbot)
        monkeypatch.setattr(
            win.montage_tile_evaluation_controller,
            "start_latest",
            lambda _fn, **kwargs: len(getattr(win._montage_session, "active_tile_requests", ())) + 1,
        )
        win._set_view_state(win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(8)), text=":"))
        win.update_montage_view()
        initial_visible = len(win._montage_session.visible_tiles)

        win.img_view.getView().setRange(xRange=(0.0, 807.0), yRange=(0.0, 4.0), padding=0)
        win._run_montage_viewport_update()
        expanded_visible = len(win._montage_session.visible_tiles)

        assert initial_visible < 8
        assert expanded_visible == 8
    finally:
        win.close()
        clear_arrayscope_settings()


def test_cold_montage_tile_patches_without_side_panel_refresh(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    operation_refreshes = []
    inspection_refreshes = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    monkeypatch.setattr(win, "_update_operation_dock", lambda: operation_refreshes.append("operation"))
    monkeypatch.setattr(win, "_refresh_inspection_dock", lambda: inspection_refreshes.append("inspection"))
    try:
        process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view(defer_side_panels=True)
        operation_refreshes.clear()
        inspection_refreshes.clear()

        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 9))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 9, dtype=np.float32)), timeout=250)

        assert operation_refreshes == []
        assert inspection_refreshes == []
    finally:
        win.close()
