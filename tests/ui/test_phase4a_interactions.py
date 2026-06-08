import os

import numpy as np
import pytest

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


def test_render_preserves_viewport_for_same_display_shape(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        view = win.img_view.getView()
        view.setRange(xRange=(1, 3), yRange=(1, 3), padding=0)
        before = view.viewRange()

        win._on_slice_index_changed(2, 1)
        _process_events(qtbot, count=20)

        np.testing.assert_allclose(view.viewRange(), before, atol=1e-9)
    finally:
        win.close()


def test_dock_show_hide_preserves_image_view_size(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        win.resize(700, 420)
        _process_events(qtbot)
        before_size = win.img_view.size()

        win._show_inspection_dock()
        _process_events(qtbot, count=30)
        win._set_inspection_dock_visible_from_user(False)
        _process_events(qtbot, count=30)

        after_size = win.img_view.size()
        assert abs(after_size.width() - before_size.width()) <= 1
        assert abs(after_size.height() - before_size.height()) <= 1
    finally:
        win.close()


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


def test_strict_ui_mode_raises_callback_exceptions(monkeypatch):
    from arrayscope.app.errors import handle_ui_exception

    monkeypatch.setenv("ARRAYSCOPE_STRICT_UI", "1")

    with pytest.raises(RuntimeError, match="boom"):
        handle_ui_exception("test callback", RuntimeError("boom"))


def test_inspection_dock_defaults_left_and_stays_closed_after_direct_close(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)

        win._show_inspection_dock()
        _process_events(qtbot, count=12)
        assert win.inspection_dock.isVisible()
        assert win.dockWidgetArea(win.inspection_dock) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea

        win.inspection_dock.close()
        _process_events(qtbot, count=12)
        assert not win.inspection_dock.isVisible()
        assert not win._inspection_dock_user_visible

        win._refresh_inspection_dock()
        _process_events(qtbot, count=12)
        assert not win.inspection_dock.isVisible()
    finally:
        win.close()


def test_direct_closing_docked_inspection_does_not_restore_canvas_snapshot(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(32 * 32, dtype=float).reshape(32, 32))
    qtbot.addWidget(win)
    try:
        win.resize(900, 620)
        _process_events(qtbot)
        win._show_inspection_dock()
        _process_events(qtbot, count=20)
        before = win.img_view.size()
        before_window = win.size()

        win.inspection_dock.close()
        _process_events(qtbot, count=30)
        after = win.img_view.size()
        after_window = win.size()

        assert abs(after.width() - before.width()) <= 2
        assert after_window.width() < before_window.width()
        assert not win.inspection_dock.isVisible()
        assert win._inspection_dock_user_visible is False
    finally:
        win.close()


def test_profile_dock_defaults_bottom_when_opened_from_view_menu(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(32 * 32, dtype=float).reshape(32, 32))
    qtbot.addWidget(win)
    try:
        win.move(0, 0)
        win.resize(700, 420)
        _process_events(qtbot, count=20)
        win._set_profile_dock_visible_from_user(True)
        _process_events(qtbot, count=40)

        assert win.profile_dock.isVisible()
        assert win.dockWidgetArea(win.profile_dock) == QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
    finally:
        win.close()


def test_opening_dock_uses_current_dock_extent_for_window_growth(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(24 * 24, dtype=float).reshape(24, 24))
    qtbot.addWidget(win)
    try:
        win.setGeometry(420, 20, 320, 500)
        _process_events(qtbot, count=20)
        win.inspection_dock.resize(420, 300)
        width_delta, height_delta = win.layout_manager._dock_extent_for_area(win.inspection_dock)

        assert width_delta >= 420
        assert height_delta == 0

        win._set_inspection_dock_visible_from_user(True)
        _process_events(qtbot, count=40)
        assert win.inspection_dock.isVisible()
    finally:
        win.close()


def test_docks_have_size_grips_and_managed_detach(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        for dock in (win.inspection_dock, win.profile_dock, win.operation_dock):
            assert dock.features() & QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            assert dock.findChildren(QtWidgets.QSizeGrip)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        assert win.panel_manager.location("inspection") == PanelLocation.DETACHED
    finally:
        win.close()
