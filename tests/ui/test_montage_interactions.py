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


def test_montage_byte_budget_limits_visible_tiles():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.render import RenderMixin

    state = ViewState.from_shape((10, 10, 10)).with_montage_axis(2, indices=tuple(range(10)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(10)), tile_shape=(10, 10), columns=5, gap=1)

    selected, used, skipped = RenderMixin()._select_visible_montage_tiles_by_budget(
        plan.tiles,
        tile_shape=(10, 10),
        output_dtype=np.float32,
        rgb=False,
        budget_bytes=10 * 10 * 4 * 2,
    )

    assert tuple(tile.source_index for tile in selected) == (0,)
    assert used == 10 * 10 * 8
    assert skipped == 9


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
    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        assert getattr(win.img_view, "_montage_tile_overlay_items", []) == []
        assert win.img_view._evaluation_overlay is None or not win.img_view._evaluation_overlay.isVisible()

        calls[0]["on_slow"]()

        assert getattr(win.img_view, "_montage_tile_overlay_items", [])
        assert win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()


def test_montage_loading_overlay_is_session_delayed(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", lambda _fn, **_kwargs: 1)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()
        assert win.img_view._evaluation_overlay is None or not win.img_view._evaluation_overlay.isVisible()

        qtbot.waitUntil(lambda: win.img_view._evaluation_overlay is not None and win.img_view._evaluation_overlay.isVisible(), timeout=500)

        assert getattr(win.img_view, "_montage_tile_overlay_items", [])
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


def test_montage_schedules_missing_tiles_one_at_a_time(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []

    def capture_start_latest(_fn, **kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", capture_start_latest)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"))
        win.update_montage_view()

        assert len(calls) == 1
        assert calls[0]["key"][0] == "montage_tile"
        tile = win._montage_session.plan.tiles[0]
        calls[0]["on_done"](_tile_result(tile, 1))
        assert len(calls) == 2
        assert calls[1]["key"][0] == "montage_tile"
    finally:
        win.close()


def test_montage_finished_tile_updates_canvas_before_all_tiles_finish(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
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


def test_stale_montage_tile_result_does_not_mutate_current_ui_state(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4))
    qtbot.addWidget(win)
    calls = []
    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", lambda _fn, **kwargs: calls.append(kwargs) or len(calls))
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
