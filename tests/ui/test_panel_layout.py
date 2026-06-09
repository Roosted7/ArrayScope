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


def _view_submenu_action(win, submenu_text, action_text):
    for action in win.menuBar().actions():
        if action.text() != "View":
            continue
        for child in action.menu().actions():
            if child.text() != submenu_text:
                continue
            submenu = child.menu()
            for grandchild in submenu.actions():
                if grandchild.text() == action_text:
                    return grandchild
    raise AssertionError(f"View submenu action not found: {submenu_text}/{action_text}")


def test_panel_open_hide_preserves_central_widget_size_with_resize_transaction(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        win.resize(900, 620)
        _process_events(qtbot, count=20)
        target = QtCore.QSize(win.centralWidget().size())

        win._show_inspection_dock()
        _wait_for_panel_preserve(qtbot)
        _assert_size_close(win.centralWidget().size(), target)

        win._set_inspection_dock_visible_from_user(False)
        _wait_for_panel_preserve(qtbot)
        _assert_size_close(win.centralWidget().size(), target)
    finally:
        win.close()


def test_canvas_preserve_controller_records_diagnostics(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        win.resize(900, 620)
        _process_events(qtbot, count=20)

        win._show_inspection_dock()
        _wait_for_panel_preserve(qtbot)

        diagnostics = win.layout_manager.canvas_preserver.diagnostics()
        assert diagnostics.generation > 0
        assert diagnostics.mode == "best_effort"
        assert diagnostics.last_transition == "show-docked"
        assert diagnostics.target_canvas_size is not None
        assert diagnostics.final_canvas_size is not None
        assert diagnostics.events
    finally:
        win.close()


def test_panel_preserve_transaction_does_not_move_window_position(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        win.move(100, 80)
        win.resize(900, 620)
        _process_events(qtbot, count=20)
        before = QtCore.QPoint(win.pos())
        if win.pos() != before:
            return

        win._show_inspection_dock()
        _wait_for_panel_preserve(qtbot)
        assert win.pos() == before

        win._set_inspection_dock_visible_from_user(False)
        _wait_for_panel_preserve(qtbot)
        assert win.pos() == before
    finally:
        win.close()


def test_panel_resize_behavior_off_does_not_resize_main_window(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState, PanelResizeBehavior
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        win.resize(900, 620)
        _process_events(qtbot, count=20)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme,
            prefetch_nearby_slices=win.app_settings.prefetch_nearby_slices,
            panel_resize_behavior=PanelResizeBehavior.OFF,
        )
        before_central_size = win.centralWidget().size()
        before_generation = win.layout_manager.canvas_preserver.generation

        win._show_inspection_dock()
        _wait_for_panel_preserve(qtbot)

        assert win.layout_manager.canvas_preserver.generation == before_generation
        assert win.centralWidget().size().width() < before_central_size.width()
    finally:
        win.close()


def test_canvas_preserve_strong_wayland_applies_and_releases_constraints(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import AppSettingsState, PanelResizeBehavior
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme,
            prefetch_nearby_slices=win.app_settings.prefetch_nearby_slices,
            panel_resize_behavior=PanelResizeBehavior.STRONG_WAYLAND,
            fft_backend=win.app_settings.fft_backend,
            fft_workers=win.app_settings.fft_workers,
            memory_profile=win.app_settings.memory_profile,
            render_memory_budget_mb=win.app_settings.render_memory_budget_mb,
        )
        preserver = win.layout_manager.canvas_preserver
        preserver._qt_platform_name = lambda: "wayland"
        target = QtCore.QSize(win.size().width() + 12, win.size().height())
        preserver._generation += 1
        generation = preserver.generation

        preserver._apply_strong_preserve_constraints(target, generation)

        assert preserver.constraints_active
        assert win.minimumSize() == win.maximumSize()

        preserver._release_strong_preserve_constraints(generation)
        _process_events(qtbot, count=35)
        assert not preserver.constraints_active
        assert win.minimumSize() != target
        assert win.maximumSize() != target
        assert preserver.diagnostics().strong_used
    finally:
        win.close()


def test_panel_strong_preserve_release_ignores_stale_generation(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        target = QtCore.QSize(win.size().width() + 12, win.size().height())
        preserver = win.layout_manager.canvas_preserver
        preserver._generation += 1
        generation = preserver.generation

        preserver._apply_strong_preserve_constraints(target, generation)
        assert win.minimumSize() == target
        assert win.maximumSize() == target

        preserver._generation += 1
        preserver._release_strong_preserve_constraints(generation)
        assert win.minimumSize() == target
        assert win.maximumSize() == target

        preserver._release_strong_preserve_constraints(preserver.generation)
        assert win.minimumSize() != target
        assert win.maximumSize() != target
    finally:
        win.close()


def test_strong_wayland_skips_strong_path_off_wayland(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore
    from arrayscope.app.settings_state import AppSettingsState, PanelResizeBehavior
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        win.app_settings = AppSettingsState(
            theme=win.app_settings.theme,
            prefetch_nearby_slices=win.app_settings.prefetch_nearby_slices,
            panel_resize_behavior=PanelResizeBehavior.STRONG_WAYLAND,
            fft_backend=win.app_settings.fft_backend,
            fft_workers=win.app_settings.fft_workers,
            memory_profile=win.app_settings.memory_profile,
            render_memory_budget_mb=win.app_settings.render_memory_budget_mb,
        )
        preserver = win.layout_manager.canvas_preserver
        preserver._qt_platform_name = lambda: "offscreen"
        preserver._platform = "offscreen"
        preserver._generation += 1
        generation = preserver.generation
        current = win.centralWidget().size()
        target = QtCore.QSize(current.width() + 30, current.height())

        preserver._correct_canvas_size(
            target,
            attempts=1,
            generation=generation,
            dock_extents=(),
            size_constraints=preserver._capture_window_size_constraints(),
            strong_used=False,
            allow_strong=True,
        )
        _process_events(qtbot, count=5)

        diagnostics = preserver.diagnostics()
        assert not preserver.constraints_active
        assert not diagnostics.strong_used
        assert any("strong_skipped" in event for event in diagnostics.events)
    finally:
        win.close()


def test_view_menu_panel_resize_behavior_persists(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import PanelResizeBehavior
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        off_action = _view_submenu_action(win, "Panel Resize Behavior", "Off")
        strong_action = _view_submenu_action(win, "Panel Resize Behavior", "Strong Wayland")
        best_action = _view_submenu_action(win, "Panel Resize Behavior", "Best effort")
        assert best_action.isChecked()
        off_action.trigger()
        _process_events(qtbot, count=5)
        assert off_action.isChecked()
        assert win.app_settings.panel_resize_behavior == PanelResizeBehavior.OFF
        strong_action.trigger()
        _process_events(qtbot, count=5)
        assert strong_action.isChecked()
        assert win.app_settings.panel_resize_behavior == PanelResizeBehavior.STRONG_WAYLAND
    finally:
        win.close()

    second = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(second)
    try:
        _process_events(qtbot, count=20)
        action = _view_submenu_action(second, "Panel Resize Behavior", "Strong Wayland")
        assert action.isChecked()
        assert second.app_settings.panel_resize_behavior == PanelResizeBehavior.STRONG_WAYLAND
    finally:
        second.close()


def test_inspection_dock_defaults_left_and_stays_closed_after_managed_title_close(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore, QtWidgets
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)

        win._show_inspection_dock()
        _process_events(qtbot, count=12)
        assert win.inspection_dock.isVisible()
        assert win.dockWidgetArea(win.inspection_dock) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea

        close_button = win.inspection_dock.findChild(QtWidgets.QToolButton, "ManagedDockCloseButton")
        assert close_button is not None
        close_button.click()
        _process_events(qtbot, count=12)
        assert not win.inspection_dock.isVisible()
        assert not win._inspection_dock_user_visible

        win._refresh_inspection_dock()
        _process_events(qtbot, count=12)
        assert not win.inspection_dock.isVisible()
    finally:
        win.close()


def test_managed_title_closing_docked_inspection_does_not_restore_canvas_snapshot(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
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

        close_button = win.inspection_dock.findChild(QtWidgets.QToolButton, "ManagedDockCloseButton")
        assert close_button is not None
        close_button.click()
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


def test_view_menu_uses_managed_dock_actions(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=20)

        assert win.inspection_dock.isVisible()
        assert action.isChecked()
    finally:
        win.close()


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


def test_detached_inspection_show_does_not_redock_until_requested(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)
        assert win.panel_manager.location("inspection") == PanelLocation.DETACHED

        action.trigger()
        _process_events(qtbot, count=15)
        assert win.panel_manager.location("inspection") == PanelLocation.HIDDEN
        action.trigger()
        _process_events(qtbot, count=15)

        assert win.panel_manager.location("inspection") == PanelLocation.DOCKED
    finally:
        win.close()


def test_reset_layout_redocks_managed_docks(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        assert win.panel_manager.location("inspection") == PanelLocation.DETACHED

        win.reset_layout()
        _process_events(qtbot, count=20)

        assert win.panel_manager.location("inspection") != PanelLocation.DETACHED
    finally:
        win.close()


def test_closing_detached_dialog_unchecks_view_action(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)

        dialog = win.panel_manager.panel_for_dock(win.inspection_dock).dialog
        assert dialog is not None
        assert action.isChecked()

        dialog.close()
        _process_events(qtbot, count=10)

        assert win.panel_manager.location("inspection") == PanelLocation.HIDDEN
        assert not action.isChecked()

        action.trigger()
        _process_events(qtbot, count=15)
        assert win.panel_manager.location("inspection") == PanelLocation.DOCKED
        assert win.inspection_dock.findChild(type(win.inspection_dock.stats_table)) is win.inspection_dock.stats_table
    finally:
        win.close()


def test_managed_dock_title_buttons_detach_hide_and_redock(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)

        from pyqtgraph.Qt import QtWidgets

        detach_button = win.inspection_dock.findChild(QtWidgets.QToolButton, "ManagedDockDetachButton")
        assert detach_button is not None
        detach_button.click()
        _process_events(qtbot, count=15)
        assert win.panel_manager.location("inspection") == PanelLocation.DETACHED

        dialog = win.panel_manager.panel_for_dock(win.inspection_dock).dialog
        assert dialog is not None
        assert dialog.findChild(type(win.inspection_dock.stats_table)) is win.inspection_dock.stats_table
        redock_button = dialog.findChild(QtWidgets.QToolButton, "DetachedPanelRedockButton")
        assert redock_button is not None
        redock_button.click()
        _process_events(qtbot, count=15)
        assert win.panel_manager.location("inspection") == PanelLocation.DOCKED
        assert win.inspection_dock.findChild(type(win.inspection_dock.stats_table)) is win.inspection_dock.stats_table

        close_button = win.inspection_dock.findChild(QtWidgets.QToolButton, "ManagedDockCloseButton")
        assert close_button is not None
        close_button.click()
        _process_events(qtbot, count=15)
        assert win.panel_manager.location("inspection") == PanelLocation.HIDDEN
        assert not win._inspection_dock_user_visible
    finally:
        win.close()


def test_managed_dock_title_drag_detaches_panel(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore, QtTest
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.layout_manager.set_managed_dock_visible(win.inspection_dock, True, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=10)

        title_bar = win.inspection_dock.titleBarWidget()
        assert title_bar is not None
        QtTest.QTest.mousePress(title_bar, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(10, 10))
        QtTest.QTest.mouseMove(title_bar, QtCore.QPoint(80, 10))
        QtTest.QTest.mouseRelease(title_bar, QtCore.Qt.MouseButton.LeftButton, pos=QtCore.QPoint(80, 10))
        _process_events(qtbot, count=15)

        assert win.panel_manager.location("inspection") == PanelLocation.DETACHED
    finally:
        win.close()


def test_detached_hidden_reopen_redock_hide_reopen_preserves_body(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        panel = win.panel_manager.panel_for_dock(win.inspection_dock)
        body = _panel_body(panel)
        _assert_panel_invariants(win, "inspection", PanelLocation.DOCKED)

        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.DETACHED)

        panel.dialog.close()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.HIDDEN)

        action.trigger()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.DOCKED)

        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        redock_button = panel.dialog.findChild(QtWidgets.QToolButton, "DetachedPanelRedockButton")
        redock_button.click()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.DOCKED)

        action.trigger()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.HIDDEN)

        action.trigger()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.DOCKED)
    finally:
        win.close()


def test_hide_detached_panel_destroys_dialog_and_recovers_body(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        panel = win.panel_manager.panel_for_dock(win.inspection_dock)
        body = _panel_body(panel)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        assert panel.dialog is not None

        action.trigger()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.HIDDEN)

        action.trigger()
        _process_events(qtbot, count=15)
        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.DOCKED)
        assert win.inspection_dock.findChild(type(win.inspection_dock.stats_table)) is win.inspection_dock.stats_table
    finally:
        win.close()


def test_reset_layout_after_detached_hidden_panel_has_no_stale_dialog(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=15)
        panel = win.panel_manager.panel_for_dock(win.inspection_dock)
        body = _panel_body(panel)
        win.layout_manager.detach_managed_dock(win.inspection_dock, reason="test", preserve_canvas=False)
        _process_events(qtbot, count=15)
        action.trigger()
        _process_events(qtbot, count=15)
        _assert_panel_invariants(win, "inspection", PanelLocation.HIDDEN)

        win.reset_layout()
        _process_events(qtbot, count=20)

        assert _panel_body(panel) is body
        _assert_panel_invariants(win, "inspection", PanelLocation.HIDDEN)
        assert win.inspection_dock.widget() is body
    finally:
        win.close()


def test_managed_title_close_is_authoritative_hide_path(qtbot):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window.panels import PanelLocation
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        for name, dock, action_text in (
            ("inspection", win.inspection_dock, "Inspection"),
            ("profile", win.profile_dock, "Profile"),
            ("operations", win.operation_dock, "Operations"),
        ):
            action = _view_action(win, action_text)
            if not action.isChecked():
                action.trigger()
                _process_events(qtbot, count=15)
            panel = win.panel_manager.panel_for_dock(dock)
            body = _panel_body(panel)
            _assert_panel_invariants(win, name, PanelLocation.DOCKED)

            close_button = dock.findChild(QtWidgets.QToolButton, "ManagedDockCloseButton")
            assert close_button is not None
            close_button.click()
            _process_events(qtbot, count=15)
            _assert_panel_invariants(win, name, PanelLocation.HIDDEN)
            assert not action.isChecked()

            action.trigger()
            _process_events(qtbot, count=15)
            assert _panel_body(panel) is body
            _assert_panel_invariants(win, name, PanelLocation.DOCKED)
    finally:
        win.close()


def test_operation_dock_view_menu_grows_and_hides_window(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Operations")
        start_width = win.width()
        action.trigger()
        _process_events(qtbot, count=20)
        opened_width = win.width()
        assert win.operation_dock.isVisible()
        assert opened_width > start_width

        action.trigger()
        _process_events(qtbot, count=20)
        assert not win.operation_dock.isVisible()
        assert win.width() < opened_width
    finally:
        win.close()


def test_operation_dock_grows_without_prior_manual_resize(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Operations")
        start_width = win.width()

        action.trigger()
        _process_events(qtbot, count=20)

        assert win.operation_dock.isVisible()
        assert win.width() > start_width
    finally:
        win.close()


def test_hiding_operation_dock_with_inspection_open_shrinks_window_not_inspection(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(12 * 13, dtype=float).reshape(12, 13))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        inspection_action = _view_action(win, "Inspection")
        operation_action = _view_action(win, "Operations")
        inspection_action.trigger()
        _process_events(qtbot, count=20)
        operation_action.trigger()
        _process_events(qtbot, count=20)

        opened_width = win.width()
        inspection_width = win.inspection_dock.width()

        operation_action.trigger()
        _process_events(qtbot, count=25)

        assert not win.operation_dock.isVisible()
        assert win.inspection_dock.isVisible()
        assert win.width() < opened_width
        assert abs(win.inspection_dock.width() - inspection_width) <= 4
    finally:
        win.close()
