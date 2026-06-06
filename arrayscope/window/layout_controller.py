"""Centralized window/dock layout behavior."""

from __future__ import annotations

import pyqtgraph.Qt as Qt


class WindowLayoutManager:
    def __init__(self, window):
        self.window = window

    def restore_window_settings(self):
        win = self.window
        geometry = win._settings.value("geometry")
        if geometry is not None:
            win.restoreGeometry(geometry)
        state = win._settings.value("window_state")
        if state is not None:
            win.restoreState(state)
        if not win.profile_dock.isVisible() and win.data.ndim == 1:
            win.profile_dock.show()
        self.sync_progressive_docks()
        Qt.QtCore.QTimer.singleShot(0, self.resize_default_docks)

    def reset_layout(self):
        win = self.window
        win._operation_dock_user_visible = False
        win._profile_dock_user_visible = False
        win._inspection_dock_user_visible = False
        win.profile_dock.setFloating(False)
        win.profile_dock.hide()
        if hasattr(win, "inspection_dock"):
            win.inspection_dock.setFloating(False)
            win.inspection_dock.hide()
        if win.data.ndim == 1:
            win.profile_dock.show()
        win.operation_dock.setFloating(False)
        win.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, win.operation_dock)
        if win.document.steps:
            win.operation_dock.show()
        else:
            win.operation_dock.hide()
        win.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, win.profile_dock)
        Qt.QtCore.QTimer.singleShot(0, self.resize_default_docks)

    def set_operation_dock_visible_from_user(self, visible):
        win = self.window
        win._operation_dock_user_visible = bool(visible)
        win.operation_dock.setVisible(bool(visible))
        self.schedule_view_geometry_refresh()

    def set_profile_dock_visible_from_user(self, visible):
        win = self.window
        win._profile_dock_user_visible = bool(visible)
        win.profile_dock.setVisible(bool(visible))
        self.schedule_view_geometry_refresh()

    def sync_progressive_docks(self):
        win = self.window
        changed = False
        if hasattr(win, "operation_dock"):
            has_steps = bool(win.document.steps)
            if has_steps:
                changed = changed or not win.operation_dock.isVisible()
                self.set_dock_visible_later(win.operation_dock, True)
            elif not getattr(win, "_operation_dock_user_visible", False):
                changed = changed or win.operation_dock.isVisible()
                self.set_dock_visible_later(win.operation_dock, False)
        if hasattr(win, "profile_dock"):
            live_profile = win.widgets["buttons"]["display"]["live_profile"].isChecked()
            should_show_profile = win.data.ndim == 1 or live_profile or getattr(win, "_profile_dock_user_visible", False)
            if should_show_profile:
                changed = changed or not win.profile_dock.isVisible()
                self.set_dock_visible_later(win.profile_dock, True)
            elif not win.profile_dock.isFloating():
                changed = changed or win.profile_dock.isVisible()
                self.set_dock_visible_later(win.profile_dock, False)
        if hasattr(win, "inspection_dock"):
            should_show_inspection = getattr(win, "_inspection_dock_user_visible", False)
            if should_show_inspection:
                changed = changed or not win.inspection_dock.isVisible()
                self.set_dock_visible_later(win.inspection_dock, True)
            elif not win.inspection_dock.isFloating():
                changed = changed or win.inspection_dock.isVisible()
                self.set_dock_visible_later(win.inspection_dock, False)
        if changed:
            self.schedule_view_geometry_refresh()

    def schedule_view_geometry_refresh(self):
        Qt.QtCore.QTimer.singleShot(0, self.refresh_view_geometry)

    def set_dock_visible_later(self, dock, visible):
        Qt.QtCore.QTimer.singleShot(0, lambda dock=dock, visible=visible: self.apply_queued_dock_visibility(dock, visible))

    def apply_queued_dock_visibility(self, dock, visible):
        win = self.window
        if not visible:
            if dock is getattr(win, "operation_dock", None) and win.document.steps:
                return
            if dock is getattr(win, "profile_dock", None):
                live_profile = win.widgets["buttons"]["display"]["live_profile"].isChecked()
                if win.data.ndim == 1 or live_profile or getattr(win, "_profile_dock_user_visible", False):
                    return
            if dock is getattr(win, "inspection_dock", None):
                if getattr(win, "_inspection_dock_user_visible", False):
                    return
        dock.setVisible(bool(visible))

    def refresh_view_geometry(self):
        win = self.window
        if hasattr(win, "centralWidget") and win.centralWidget() is not None:
            win.centralWidget().updateGeometry()

    def resize_profile_dock_default(self):
        win = self.window
        if not hasattr(win, "profile_dock") or not win.profile_dock.isVisible() or win.profile_dock.isFloating():
            return
        target_height = max(140, int(win.height() * 0.23))
        try:
            win.resizeDocks([win.profile_dock], [target_height], Qt.QtCore.Qt.Orientation.Vertical)
        except Exception:
            pass

    def resize_default_docks(self):
        win = self.window
        try:
            if win.profile_dock.isVisible() and not win.profile_dock.isFloating():
                win.resizeDocks([win.profile_dock], [max(140, int(win.height() * 0.23))], Qt.QtCore.Qt.Orientation.Vertical)
            if win.operation_dock.isVisible() and not win.operation_dock.isFloating():
                win.resizeDocks([win.operation_dock], [max(220, int(win.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
            if hasattr(win, "inspection_dock") and win.inspection_dock.isVisible() and not win.inspection_dock.isFloating():
                win.resizeDocks([win.inspection_dock], [max(240, int(win.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
        except Exception:
            pass
