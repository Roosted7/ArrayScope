import time

import numpy as np
import pytest

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.evaluator import EvaluationResult
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


def test_montage_roi_gap_source_is_nan(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=40)

        gap_x = win._current_montage_geometry.tile_width
        source = win._roi_source_image()

        assert np.isnan(source[:, gap_x]).all()
    finally:
        win.close()


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
        win.update_montage_view()
        qtbot.waitUntil(lambda: win._current_montage_canvas is not None and win._current_montage_canvas.origin_y != 0, timeout=3000)

        canvas = win._current_montage_canvas
        assert canvas.origin_y != 0
        tile_10 = canvas.full_plan.tiles[10]
        qtbot.waitUntil(
            lambda: win.display_geometry.context_for_display_point(
                tile_10.x0 - win._current_montage_canvas.origin_x + 1,
                tile_10.y0 - win._current_montage_canvas.origin_y + 1,
            )
            is not None,
            timeout=3000,
        )
        canvas = win._current_montage_canvas
        context = win.display_geometry.context_for_display_point(tile_10.x0 - canvas.origin_x + 1, tile_10.y0 - canvas.origin_y + 1)

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
        win.update_montage_view()
        qtbot.waitUntil(lambda: win._current_montage_canvas is not None and win._current_montage_canvas.origin_y != 0, timeout=3000)

        canvas = win._current_montage_canvas
        tile_10 = canvas.full_plan.tiles[10]
        qtbot.waitUntil(
            lambda: win.display_geometry.context_for_display_point(tile_10.x0 - win._current_montage_canvas.origin_x + 1, tile_10.y0 - win._current_montage_canvas.origin_y + 1)
            is not None,
            timeout=3000,
        )
        canvas = win._current_montage_canvas
        context = win.display_geometry.context_for_display_point(tile_10.x0 - canvas.origin_x + 1, tile_10.y0 - canvas.origin_y + 1)

        assert context is not None
        assert context.mapping.local_x == 1
        assert context.mapping.local_y == 1
        assert context.mapping.display_y != context.mapping.local_y
        assert win._hover_value_from_display(context.mapping) == pytest.approx(10.0)
    finally:
        win.close()


def test_montage_does_not_call_make_montage_for_interactive_render(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.window import ArrayScopeWindow

    monkeypatch.setattr(render_module, "make_montage", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")), raising=False)
    win = ArrayScopeWindow(np.arange(2 * 3 * 8, dtype=float).reshape(2, 3, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)

        assert win._current_montage_canvas is not None
    finally:
        win.close()


def test_montage_commits_cached_tiles_immediately_with_loading_placeholders(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState, make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        win._set_view_state(state)
        plan = make_montage_plan(state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=3)
        win.operation_evaluator.store_montage_tile_result(plan.tiles[0], montage_axis=2, colormap_lut=None, result=_tile_result(plan.tiles[0], 10))
        monkeypatch.setattr(win, "_schedule_next_montage_tile", lambda _session: None)

        win.update_montage_view()

        canvas = win._current_montage_canvas
        assert canvas is not None
        assert canvas.tile_states[0] == MontageTileState.LOADED
        assert canvas.tile_states[1] == MontageTileState.LOADING
        np.testing.assert_array_equal(canvas.data[0:2, 0:2], np.full((2, 2), 10, dtype=np.float32))
        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
    finally:
        win.close()


def test_montage_loading_overlays_wait_for_slow_callback(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
        assert win.img_view._evaluation_overlay is None or not win.img_view._evaluation_overlay.isVisible()

        calls[0]["on_slow"]()

        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
        assert win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()


def test_montage_loading_overlay_is_session_delayed(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **_kwargs: 1)
    try:
        _process_events(qtbot)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        assert win.img_view._evaluation_overlay is None or not win.img_view._evaluation_overlay.isVisible()

        qtbot.waitUntil(lambda: win.img_view._evaluation_overlay is not None and win.img_view._evaluation_overlay.isVisible(), timeout=500)

        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
    finally:
        win.close()


def test_montage_canvas_tiles_are_all_accounted_for(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 12, dtype=np.float32).reshape(2, 2, 12))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(12)), text=":"))
        monkeypatch.setattr(win, "_schedule_next_montage_tile", lambda _session: None)
        win.update_montage_view()

        canvas = win._current_montage_canvas
        assert canvas is not None
        for tile in canvas.full_plan.tiles:
            in_canvas = (
                tile.x0 < canvas.canvas_rect[2]
                and tile.x0 + tile.width > canvas.canvas_rect[0]
                and tile.y0 < canvas.canvas_rect[3]
                and tile.y0 + tile.height > canvas.canvas_rect[1]
            )
            if in_canvas:
                assert canvas.tile_states[tile.montage_index] in {
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
        win.update_montage_view()
        monkeypatch.setattr(win, "update_montage_view", lambda *args, **kwargs: calls.append((args, kwargs)))

        win.img_view.getView().setRange(xRange=(6, 9), yRange=(0, 2), padding=0)

        qtbot.waitUntil(lambda: bool(calls), timeout=1000)
    finally:
        win.close()


def test_cached_montage_tile_rebinds_to_current_layout(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        old_state = ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=1, indices=(0, 1, 2), text=":")
        old_plan = make_montage_plan(old_state, axis=2, indices=(0, 1, 2), tile_shape=(2, 2), columns=1)
        win.operation_evaluator.store_montage_tile_result(old_plan.tiles[1], montage_axis=2, colormap_lut=None, result=_tile_result(old_plan.tiles[1], 11))
        monkeypatch.setattr(win, "_schedule_next_montage_tile", lambda _session: None)

        new_state = win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":")
        win._set_view_state(new_state)
        win.update_montage_view()

        canvas = win._current_montage_canvas
        new_tile = canvas.full_plan.tiles[1]
        x0 = new_tile.x0 - canvas.origin_x
        y0 = new_tile.y0 - canvas.origin_y
        np.testing.assert_array_equal(canvas.data[y0 : y0 + 2, x0 : x0 + 2], np.full((2, 2), 11, dtype=np.float32))
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
        win._warn_montage_tiles_skipped(
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


def test_montage_schedules_missing_tiles_on_montage_lane(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []

    def capture_start_latest(_fn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", capture_start_latest)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()

        assert len(calls) == 2
        assert calls[0]["key"][0] == "montage_tile"
        assert calls[0]["replace_group"].startswith("montage-tile:")
        assert win.montage_tile_evaluation_controller.pool.maxThreadCount() == 2
        assert win.visible_evaluation_controller.pool.maxThreadCount() == 1
        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 1))
        assert all(call["key"][0] == "montage_tile" for call in calls)
    finally:
        win.close()


def test_montage_finished_tile_updates_canvas_before_all_tiles_finish(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        tile = win._montage_session.plan.tiles[0]

        calls[0]["on_done"](_tile_result(tile, 7))
        qtbot.waitUntil(lambda: win._current_montage_canvas.tile_states[0] == MontageTileState.LOADED, timeout=1000)

        canvas = win._current_montage_canvas
        assert canvas.tile_states[0] == MontageTileState.LOADED
        assert canvas.tile_states[1] == MontageTileState.LOADING
        np.testing.assert_array_equal(canvas.data[0:2, 0:2], np.full((2, 2), 7, dtype=np.float32))
    finally:
        win.close()


def test_montage_finished_tile_patches_without_rebuilding_canvas(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.montage_renderer as montage_renderer_module
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        data_id = id(win._current_montage_canvas.data)
        monkeypatch.setattr(
            montage_renderer_module,
            "make_montage_viewport_canvas",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("canvas rebuilt")),
        )

        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 7))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 7, dtype=np.float32)), timeout=1000)

        assert id(win._current_montage_canvas.data) == data_id
        np.testing.assert_array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 7, dtype=np.float32))
    finally:
        win.close()


def test_montage_progressive_tile_commit_preserves_current_levels(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        win.img_view.setLevels(2.0, 8.0)

        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 100))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 100, dtype=np.float32)), timeout=1000)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (2.0, 8.0)
    finally:
        win.close()


def test_montage_loading_canvas_preserves_levels_until_first_real_tile(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win.img_view.setLevels(2.0, 8.0)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (2.0, 8.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (0.0, 9.0)
        assert tuple(win.img_view.image.shape[:2]) != tuple(win._current_montage_canvas.data.shape[:2])

        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 100))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 100, dtype=np.float32)), timeout=1000)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (99.444444, 100.777778)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (99.0, 101.0)

        tile1 = win._montage_session.visible_tiles[1]
        calls[1]["on_done"](_tile_result(tile1, 200))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 3:5], np.full((2, 2), 200, dtype=np.float32)), timeout=1000)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (121.888889, 190.555556)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (99.0, 202.0)
    finally:
        win.close()


def test_enabling_montage_with_cached_tile_preserves_relative_window_fractions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.img_view.setLevels(5.0, 15.0)
        state = win.view_state.with_montage_axis(2, columns=3, indices=(1, 2), text="1:3")
        plan = make_montage_plan(state, axis=2, indices=(1, 2), tile_shape=(4, 5), columns=3)
        tile = plan.tiles[0]
        win.operation_evaluator.store_montage_tile_result(tile, montage_axis=2, colormap_lut=None, result=_tile_result(tile, 100.0))
        monkeypatch.setattr(win, "_schedule_next_montage_tile", lambda _session: None)

        win._set_view_state(state)
        win.update_montage_view()
        _process_events(qtbot, count=10)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (99.526316, 100.578947)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (99.0, 101.0)
    finally:
        win.close()


def test_shifting_montage_range_preserves_relative_window_fractions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
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
        win.img_view.setLevels(5.0, 15.0)
        first_state = win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text="0:2")
        first_plan = make_montage_plan(first_state, axis=2, indices=(0, 1), tile_shape=(4, 5), columns=2)
        for tile in first_plan.tiles:
            win.operation_evaluator.store_montage_tile_result(tile, montage_axis=2, colormap_lut=None, result=result_for(tile, data[:, :, tile.source_index]))
        monkeypatch.setattr(win, "_schedule_next_montage_tile", lambda _session: None)

        win._set_view_state(first_state)
        win.update_montage_view()
        _process_events(qtbot, count=10)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (31.315789, 93.947368)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (0.0, 119.0)

        shifted_state = win.view_state.with_montage_axis(2, columns=2, indices=(1, 2), text="1:3")
        win._set_view_state(shifted_state)
        win.update_montage_view()
        _process_events(qtbot, count=10)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (105.0, 115.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (100.0, 119.0)
    finally:
        win.close()


def test_montage_degenerate_previous_levels_do_not_become_one_tile_window(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((200, 200, 8), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index * 1000.0
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        initial_levels = tuple(round(float(value), 6) for value in win.img_view.getLevels())
        assert initial_levels == (-0.5, 0.5)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.update_montage_view()
        assert calls

        tile0 = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile0, 1000.0))
        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=1000)

        levels = tuple(round(float(value), 6) for value in win.img_view.getLevels())
        histogram_bounds = tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds())

        assert levels == (990.0, 1010.0)
        assert histogram_bounds == (990.0, 1010.0)
        assert levels != (1000.0, 1000.0)
        assert histogram_bounds != (1000.0, 1000.0)
    finally:
        win.close()


def test_montage_panning_without_new_tiles_does_not_change_levels(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((200, 200, 8), dtype=np.float32)
    for index in range(data.shape[2]):
        data[:, :, index] = index * 1000.0
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.update_montage_view()
        assert len(calls) >= 2

        for callback_index, value in ((0, 1000.0), (1, 2000.0)):
            tile = win._montage_session.plan.tiles[callback_index]
            calls[callback_index]["on_done"](_tile_result(tile, value))
            _process_events(qtbot, count=10)

        before_levels = tuple(round(float(value), 6) for value in win.img_view.getLevels())
        before_bounds = tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds())
        before_stats = win._montage_level_stats_for_session(win._montage_session)

        win.img_view.getView().setRange(xRange=(402, 605), yRange=(0, 199), padding=0)
        _process_events(qtbot, count=30)
        win.update_montage_view()
        _process_events(qtbot, count=10)

        after_stats = win._montage_level_stats_for_session(win._montage_session)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == before_levels
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == before_bounds
        assert after_stats.rank == before_stats.rank
        assert after_stats.source_indices == before_stats.source_indices
    finally:
        win.close()


def test_montage_visible_tiles_do_not_define_relative_levels(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()

        tile0 = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile0, 100))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 0:2], np.full((2, 2), 100, dtype=np.float32)), timeout=1000)
        first_levels = tuple(float(value) for value in win.img_view.getLevels())

        tile1 = win._montage_session.plan.tiles[1]
        calls[1]["on_done"](_tile_result(tile1, 1000))
        qtbot.waitUntil(lambda: np.array_equal(win._current_montage_canvas.data[0:2, 3:5], np.full((2, 2), 1000, dtype=np.float32)), timeout=1000)

        assert tuple(float(value) for value in win.img_view.getLevels()) != first_levels
        assert tuple(float(value) for value in win.img_view.getLevels()) == (99.0, 1010.0)
    finally:
        win.close()


def test_fft_montage_schedules_shared_stage_before_tile_workers(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    stage_calls = []
    tile_calls = []
    monkeypatch.setattr(win.stage_evaluation_controller, "start_latest", lambda _fn, **kwargs: stage_calls.append(kwargs) or len(stage_calls))
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: tile_calls.append(kwargs) or len(tile_calls))
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"))

        win.update_montage_view()

        assert len(stage_calls) == 1
        assert tile_calls == []
        assert win._montage_session.stage_waiting_tiles
    finally:
        win.close()


def test_fft_montage_attaches_to_inflight_stage_instead_of_tile_recompute(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    qtbot.addWidget(win)
    stage_calls = []
    tile_calls = []
    monkeypatch.setattr(win.stage_evaluation_controller, "start_latest", lambda _fn, **kwargs: stage_calls.append(kwargs) or len(stage_calls))
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: tile_calls.append(kwargs) or len(tile_calls))
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"))

        win.update_montage_view()
        win.update_montage_view()

        assert len(stage_calls) == 1
        assert tile_calls == []
        assert win._montage_session.attached_stage_requests
        assert win._montage_session.stage_waiting_tiles
    finally:
        win.close()


def test_operation_backed_complex_montage_rewindows_rgb_from_histogram_levels(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=0),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()

        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=3000)
        before = np.array(win.img_view.imageDisp, copy=True)
        assert win.img_view._rgbBaseImage is not None

        low, high = win.img_view.getHistogramDataBounds()
        win.img_view.setLevels((float(low) + float(high)) / 2.0, float(high))
        _process_events(qtbot, count=10)

        assert not np.array_equal(win.img_view.imageDisp, before)
    finally:
        win.close()


def test_operation_backed_complex_montage_tile_layer_rewindows_rgb_from_histogram_levels(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=0),))
        win._set_document(win.operation_coordinator.document)
        win._coerce_channel_for_current_dtype()
        win._montage_tile_layer_policy = lambda _geometry, _data: True
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()

        qtbot.waitUntil(lambda: getattr(win._montage_session, "display_committed", False), timeout=3000)
        qtbot.waitUntil(lambda: win.img_view.montageDisplayMode() == "tile_layer", timeout=3000)
        assert any(state.rgb_base is not None for state in win.img_view._montage_tile_layer.states.values())
        first_item = next(iter(win.img_view._montage_tile_layer.states.values())).item
        before = np.array(first_item.image, copy=True)

        low, high = win.img_view.getHistogramDataBounds()
        win.img_view.setLevels((float(low) + float(high)) / 2.0, float(high))
        _process_events(qtbot, count=10)

        assert not np.array_equal(first_item.image, before)
    finally:
        win.close()


def test_stale_montage_tile_result_does_not_mutate_current_ui_state(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"))
        win.update_montage_view()
        old_callback = calls[0]["on_done"]
        old_tile = win._montage_session.plan.tiles[0]

        win._set_view_state(win.view_state.with_montage_axis(2, columns=2, indices=(2, 3), text="2:"))
        win.update_montage_view()
        win.img_view.setEvaluationOverlay(True, "new overlay")
        current_session_id = win._montage_session.session_id
        current_canvas = win._current_montage_canvas
        old_callback(_tile_result(old_tile, 99))

        assert win._montage_session.session_id == current_session_id
        assert win._current_montage_canvas is current_canvas
        assert win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()


def test_visible_render_budget_uses_app_setting():
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window.render import RenderMixin

    mixin = RenderMixin()
    mixin.app_settings = AppSettingsState(render_memory_budget_mb=256)

    assert mixin._visible_render_budget_bytes() == 256 * 1024 * 1024


def test_montage_memory_warning_uses_viewport_canvas_estimate(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.montage_renderer as montage_renderer
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((512, 512, 100), dtype=np.float32)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    messages = []
    monkeypatch.setattr(montage_renderer, "show_status_message", lambda _parent, text, **_kwargs: messages.append(str(text)))
    monkeypatch.setattr(win.montage_tile_evaluation_controller, "start_latest", lambda _fn, **_kwargs: None)
    try:
        win.app_settings = AppSettingsState(render_memory_budget_mb=128)
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=10, indices=tuple(range(100)), text=":"))
        win.update_montage_view()

        assert not any("Montage would allocate" in message for message in messages)
        assert not any("Montage viewport canvas would allocate" in message for message in messages)
    finally:
        win.close()
