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


def test_strict_ui_mode_raises_callback_exceptions(monkeypatch):
    from arrayscope.app.errors import handle_ui_exception

    monkeypatch.setenv("ARRAYSCOPE_STRICT_UI", "1")

    with pytest.raises(RuntimeError, match="boom"):
        handle_ui_exception("test callback", RuntimeError("boom"))
