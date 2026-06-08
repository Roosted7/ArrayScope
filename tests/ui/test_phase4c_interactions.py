import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.clear()
    settings.sync()


def _view_action(win, text):
    for action in win.menuBar().actions():
        if action.text() == "View":
            for child in action.menu().actions():
                if child.text() == text:
                    return child
    raise AssertionError(f"View action not found: {text}")


def test_managed_docks_survive_repeated_menu_and_direct_close(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        for text, dock_name in (("Inspection", "inspection_dock"), ("Profile", "profile_dock"), ("Operations", "operation_dock")):
            action = _view_action(win, text)
            dock = getattr(win, dock_name)
            for _ in range(5):
                action.trigger()
                _process_events(qtbot, count=12)
                assert dock.isVisible()
                dock.close()
                _process_events(qtbot, count=35)
                assert not dock.isVisible()
            action.trigger()
            _process_events(qtbot, count=12)
            assert dock.isVisible()
    finally:
        win.close()


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


def test_toolbar_fit_and_one_to_one_are_viewport_commands(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.one_to_one_image()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.ONE_TO_ONE
        win.fit_image_to_view()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT
        assert win.display_toolbar.aspect_combo.findData("square_fov") == -1
    finally:
        win.close()


def test_cached_hover_value_does_not_show_updating(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5, dtype=float).reshape(4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        index = (1, 2)
        win.operation_evaluator.scalar(win.view_state, index)
        win._request_pixel_value(index, 2, 1, "", win.img_view.getView().mapViewToScene(win.img_view.getView().viewRect().center()))
        _process_events(qtbot, count=5)
        assert "updating" not in win.widgets["labels"]["pixelValue"].text().lower()
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


def test_roi_statistics_refresh_is_debounced(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(40 * 40, dtype=float).reshape(40, 40))
    qtbot.addWidget(win)
    calls = []
    original = win._compute_roi_inspection_snapshot
    monkeypatch.setattr(
        win,
        "_compute_roi_inspection_snapshot",
        lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1],
    )
    try:
        _process_events(qtbot, count=20)
        calls.clear()
        win.img_view.createRoi("rectangle", rect=(2, 2, 6, 6))
        win.img_view.createRoi("rectangle", rect=(4, 4, 6, 6))
        win.img_view.createRoi("rectangle", rect=(6, 6, 6, 6))
        _process_events(qtbot, count=20)

        assert len(calls) == 1
        assert win.inspection_dock.roi_model.rowCount() == 3
    finally:
        win.close()
