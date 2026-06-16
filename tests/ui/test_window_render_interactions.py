import time

import numpy as np
import pytest

from tests.ui.helpers import (
    assert_panel_invariants as _assert_panel_invariants,
    assert_size_close as _assert_size_close,
    clear_arrayscope_settings as _clear_arrayscope_settings,
    panel_body as _panel_body,
    process_events as _process_events,
    view_action as _view_action,
    wait_for_panel_preserve as _wait_for_panel_preserve,
)


def test_channel_auto_manual_coercion_matrix(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.core.view_state import ChannelMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 2, dtype=float).reshape(4, 5, 2))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert win.view_state.channel == ChannelMode.REAL
        win.request_operation("combine_real_imag", 2)
        _process_events(qtbot, count=20)
        assert win.view_state.channel == ChannelMode.COMPLEX

        win.clear_operations()
        _process_events(qtbot, count=20)
        win._on_channel_clicked("real")
        assert win._channel_user_selected
        win.request_operation("combine_real_imag", 2)
        _process_events(qtbot, count=20)
        assert win.view_state.channel == ChannelMode.REAL

        win._on_channel_clicked("imag")
        assert win.view_state.channel == ChannelMode.IMAG
        win.clear_operations()
        _process_events(qtbot, count=20)
        assert win.view_state.channel == ChannelMode.REAL

        win.request_operation("combine_real_imag", 2)
        _process_events(qtbot, count=20)
        win._on_channel_clicked("abs")
        win.clear_operations()
        _process_events(qtbot, count=20)
        assert win.view_state.channel == ChannelMode.ABS
    finally:
        win.close()


def test_reload_button_uses_standard_tool_button_chrome(qtbot, tmp_path):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.ui.widgets import TOOL_BUTTON_STYLE
    from arrayscope.window import ArrayScopeWindow

    data_path = tmp_path / "array.npy"
    np.save(data_path, np.arange(20 * 30, dtype=float).reshape(20, 30))
    win = ArrayScopeWindow(np.load(data_path), filepath=data_path)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        assert isinstance(win._reload_btn, QtWidgets.QToolButton)
        assert win._reload_btn.styleSheet() == TOOL_BUTTON_STYLE
        assert win._reload_btn.isVisible()
    finally:
        win.close()


def test_hover_reads_display_without_scalar_evaluation(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        monkeypatch.setattr(win.pixel_evaluation_controller, "start", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scalar eval scheduled")))
        scene_pos = win.img_view.getView().mapViewToScene(QtCore.QPointF(2.1, 1.1))
        win.getPixel(scene_pos)
        _process_events(qtbot, count=5)
        text = win.widgets["labels"]["pixelValue"].text().lower()
        assert "updating" not in text
        assert "7" in text
    finally:
        win.close()


def test_hover_reads_op_backed_display_without_scalar_evaluation(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.request_operation("reverse", 0)
        _process_events(qtbot, count=30)
        monkeypatch.setattr(win.pixel_evaluation_controller, "start", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scalar eval scheduled")))
        scene_pos = win.img_view.getView().mapViewToScene(QtCore.QPointF(2.1, 1.1))
        win.getPixel(scene_pos)
        _process_events(qtbot, count=5)
        text = win.widgets["labels"]["pixelValue"].text().lower()
        assert "updating" not in text
        assert "12" in text
    finally:
        win.close()


def test_relative_window_levels_preserve_fractions_across_2d_slice_scroll(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 2), dtype=np.float32)
    data[:, :, 0] = np.arange(20, dtype=np.float32).reshape(4, 5)
    data[:, :, 1] = 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.img_view.setLevels(5.0, 15.0)
        _process_events(qtbot, count=5)

        win._on_slice_index_changed(2, 1)
        win.render_coordinator.flush_now()
        _process_events(qtbot, count=30)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (105.0, 115.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (100.0, 119.0)
    finally:
        win.close()


def test_relative_window_levels_survive_fast_scroll_with_render_in_flight(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    captured = []
    try:
        _process_events(qtbot, count=20)
        win.img_view.setLevels(5.0, 15.0)
        _process_events(qtbot, count=5)
        monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", lambda _fn, **kwargs: captured.append(kwargs) or len(captured))

        win._on_slice_index_changed(2, 1)
        win.render_coordinator.flush_now()
        win._on_slice_index_changed(2, 2)
        win.render_coordinator.flush_now()

        latest = data[:, :, 2]
        captured[-1]["on_done"](EvaluationResult(DisplayImage(latest), 0.0, latest.shape, int(latest.nbytes)))
        _process_events(qtbot, count=20)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (205.0, 215.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (200.0, 219.0)
    finally:
        win.close()


def test_relative_window_levels_match_for_cached_and_uncached_images(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.operations.evaluator import EvaluationResult
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 3), dtype=np.float32)
    for index in range(3):
        data[:, :, index] = index * 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    captured = []
    try:
        _process_events(qtbot, count=20)
        win.img_view.setLevels(5.0, 15.0)
        _process_events(qtbot, count=5)
        monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", lambda _fn, **kwargs: captured.append(kwargs) or len(captured))

        win._on_slice_index_changed(2, 1)
        win.render_coordinator.flush_now()
        slice1 = data[:, :, 1]
        captured[-1]["on_done"](EvaluationResult(DisplayImage(slice1), 0.0, slice1.shape, int(slice1.nbytes)))
        _process_events(qtbot, count=20)
        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (105.0, 115.0)

        slice2_state = win.view_state.with_slice(2, 2)
        slice2 = data[:, :, 2]
        win.operation_evaluator.store_image_result(slice2_state, None, EvaluationResult(DisplayImage(slice2), 0.0, slice2.shape, int(slice2.nbytes)))
        win._on_slice_index_changed(2, 2)
        win.render_coordinator.flush_now()
        _process_events(qtbot, count=20)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (205.0, 215.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (200.0, 219.0)
    finally:
        win.close()


def test_absolute_window_levels_preserve_numbers_across_2d_slice_scroll(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 2), dtype=np.float32)
    data[:, :, 0] = np.arange(20, dtype=np.float32).reshape(4, 5)
    data[:, :, 1] = 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win.img_view.setLevels(5.0, 15.0)
        _process_events(qtbot, count=5)

        win._on_slice_index_changed(2, 1)
        win.render_coordinator.flush_now()
        _process_events(qtbot, count=30)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (5.0, 15.0)
        assert tuple(round(float(value), 6) for value in win.img_view.getHistogramDataBounds()) == (100.0, 119.0)
    finally:
        win.close()


def test_auto_window_resets_absolute_levels_once(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((4, 5, 2), dtype=np.float32)
    data[:, :, 0] = np.arange(20, dtype=np.float32).reshape(4, 5)
    data[:, :, 1] = 100.0 + np.arange(20, dtype=np.float32).reshape(4, 5)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.widgets["buttons"]["display"]["window_absolute"].setChecked(True)
        win.widgets["buttons"]["display"]["window_relative"].setChecked(False)
        win.img_view.setLevels(5.0, 15.0)
        win._on_slice_index_changed(2, 1)
        win.render_coordinator.flush_now()
        _process_events(qtbot, count=20)

        win.auto_window_levels()
        _process_events(qtbot, count=30)

        assert tuple(round(float(value), 6) for value in win.img_view.getLevels()) == (100.0, 119.0)
    finally:
        win.close()


def test_strict_ui_mode_raises_callback_exceptions(monkeypatch):
    from arrayscope.app.errors import handle_ui_exception

    monkeypatch.setenv("ARRAYSCOPE_STRICT_UI", "1")

    with pytest.raises(RuntimeError, match="boom"):
        handle_ui_exception("test callback", RuntimeError("boom"))
