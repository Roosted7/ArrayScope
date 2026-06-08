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


def test_operations_dock_does_not_auto_reopen_after_user_close(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        _process_events(qtbot, count=25)
        action = _view_action(win, "Operations")
        assert win.operation_dock.isVisible()
        assert action.isChecked()

        action.trigger()
        _process_events(qtbot, count=20)
        win._update_operation_dock()
        win.layout_manager.sync_progressive_docks()
        _process_events(qtbot, count=20)

        assert not win.operation_dock.isVisible()
        assert not action.isChecked()
    finally:
        win.close()


def test_floating_inspection_show_does_not_redock(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        win.layout_manager.set_managed_dock_floating(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        assert win.inspection_dock.isFloating()

        action.trigger()
        _process_events(qtbot, count=15)
        assert not win.inspection_dock.isVisible()
        action.trigger()
        _process_events(qtbot, count=15)

        assert win.inspection_dock.isVisible()
        assert win.inspection_dock.isFloating()
    finally:
        win.close()


def test_reset_layout_redocks_managed_docks(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        win.layout_manager.set_managed_dock_floating(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        assert win.inspection_dock.isFloating()

        win.reset_layout()
        _process_events(qtbot, count=20)

        assert not win.inspection_dock.isFloating()
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
        def fail_render(*args, **kwargs):
            raise AssertionError("Fit/1:1 must not render")

        win.render = fail_render
        win.display_toolbar.one_to_one_action.trigger()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.ONE_TO_ONE
        win.display_toolbar.fit_action.trigger()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT
        assert not hasattr(win.display_toolbar, "aspect_combo")
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


def test_render_with_hidden_profile_and_live_profile_off_skips_line_plot(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.layout_manager.set_managed_dock_visible(win.profile_dock, False, reason="test", preserve_canvas=False)
        win.widgets["buttons"]["display"]["live_profile"].setChecked(False)
        _process_events(qtbot, count=10)
        monkeypatch.setattr(win, "update_line_plot", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hidden profile work ran")))

        win.render(reason="hidden-profile-test")
        _process_events(qtbot, count=10)
    finally:
        win.close()


def test_render_refreshes_inspection_once_on_image_commit(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        calls = []
        original = win._refresh_inspection_dock
        monkeypatch.setattr(win, "_refresh_inspection_dock", lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1])

        win.render(reason="inspection-refresh-test")
        _process_events(qtbot, count=10)

        assert len(calls) == 1
    finally:
        win.close()
