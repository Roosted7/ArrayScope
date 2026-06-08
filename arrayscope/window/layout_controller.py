"""Centralized window/dock layout behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception


@dataclass
class ManagedDockState:
    user_visible: bool | None = None
    auto_visible: bool = False


class WindowLayoutManager:
    def __init__(self, window):
        self.window = window
        self._dock_states = {}

    def restore_window_settings(self):
        win = self.window
        geometry = win._settings.value("geometry")
        if geometry is not None:
            win.restoreGeometry(geometry)
        state = win._settings.value("window_state")
        if state is not None:
            win.restoreState(state)
        if not win.profile_dock.isVisible() and win.data.ndim == 1:
            self.set_managed_dock_visible(win.profile_dock, True, reason="restore-one-dimensional", preserve_canvas=False)
        self.sync_progressive_docks()
        Qt.QtCore.QTimer.singleShot(0, self.resize_default_docks)

    def reset_layout(self):
        win = self.window
        win._operation_dock_user_visible = None
        win._profile_dock_user_visible = None
        win._inspection_dock_user_visible = None
        for dock in self._managed_docks():
            if dock is not None:
                self._state_for(dock).user_visible = None
        self._apply_dock_floating_raw(win.profile_dock, False)
        self.set_managed_dock_visible(win.profile_dock, False, reason="reset", preserve_canvas=False)
        if hasattr(win, "inspection_dock"):
            self._apply_dock_floating_raw(win.inspection_dock, False)
            self._redock_managed_dock(win.inspection_dock)
            self.set_managed_dock_visible(win.inspection_dock, False, reason="reset", preserve_canvas=False)
        if win.data.ndim == 1:
            self.set_managed_dock_visible(win.profile_dock, True, reason="reset-one-dimensional", preserve_canvas=False)
        self._apply_dock_floating_raw(win.operation_dock, False)
        self._redock_managed_dock(win.operation_dock)
        if win.document.steps:
            self.set_managed_dock_visible(win.operation_dock, True, reason="reset-operations", preserve_canvas=False)
        else:
            self.set_managed_dock_visible(win.operation_dock, False, reason="reset-operations", preserve_canvas=False)
        self._redock_managed_dock(win.profile_dock)
        Qt.QtCore.QTimer.singleShot(0, self.resize_default_docks)

    def set_operation_dock_visible_from_user(self, visible):
        win = self.window
        win._operation_dock_user_visible = bool(visible)
        self._state_for(win.operation_dock).user_visible = bool(visible)
        self.set_managed_dock_visible(win.operation_dock, bool(visible), reason="user-operation")

    def set_profile_dock_visible_from_user(self, visible):
        win = self.window
        win._profile_dock_user_visible = bool(visible)
        self._state_for(win.profile_dock).user_visible = bool(visible)
        self.set_managed_dock_visible(win.profile_dock, bool(visible), reason="user-profile")

    def set_inspection_dock_visible_from_user(self, visible):
        win = self.window
        win._inspection_dock_user_visible = bool(visible)
        self._state_for(win.inspection_dock).user_visible = bool(visible)
        self.set_managed_dock_visible(win.inspection_dock, bool(visible), reason="user-inspection")

    def sync_progressive_docks(self):
        win = self.window
        changed = False
        if hasattr(win, "operation_dock"):
            has_steps = bool(win.document.steps)
            user_visible = getattr(win, "_operation_dock_user_visible", None)
            if has_steps and user_visible is not False:
                changed = changed or not win.operation_dock.isVisible()
                self.set_dock_visible_later(win.operation_dock, True)
            elif not has_steps and user_visible is not True:
                changed = changed or win.operation_dock.isVisible()
                self.set_dock_visible_later(win.operation_dock, False)
        if hasattr(win, "profile_dock"):
            live_profile = win.widgets["buttons"]["display"]["live_profile"].isChecked()
            user_visible = getattr(win, "_profile_dock_user_visible", None)
            should_show_profile = (win.data.ndim == 1 or live_profile or user_visible is True) and user_visible is not False
            if should_show_profile:
                changed = changed or not win.profile_dock.isVisible()
                self.set_dock_visible_later(win.profile_dock, True)
            elif user_visible is not True:
                changed = changed or win.profile_dock.isVisible()
                self.set_dock_visible_later(win.profile_dock, False)
        if changed:
            self.schedule_view_geometry_refresh()

    def schedule_view_geometry_refresh(self):
        Qt.QtCore.QTimer.singleShot(0, self.refresh_view_geometry)

    def set_dock_visible_later(self, dock, visible):
        self._state_for(dock).auto_visible = bool(visible)
        Qt.QtCore.QTimer.singleShot(0, lambda dock=dock, visible=visible: self.apply_queued_dock_visibility(dock, visible))

    def apply_queued_dock_visibility(self, dock, visible):
        win = self.window
        if not visible:
            if dock is getattr(win, "operation_dock", None) and win.document.steps:
                if getattr(win, "_operation_dock_user_visible", None) is not False:
                    return
            if dock is getattr(win, "profile_dock", None):
                live_profile = win.widgets["buttons"]["display"]["live_profile"].isChecked()
                user_visible = getattr(win, "_profile_dock_user_visible", None)
                if (win.data.ndim == 1 or live_profile or user_visible is True) and user_visible is not False:
                    return
            if dock is getattr(win, "inspection_dock", None):
                if getattr(win, "_inspection_dock_user_visible", None) is True:
                    return
        self.set_managed_dock_visible(dock, bool(visible), reason="progressive")

    def set_dock_visible_preserving_canvas(self, dock, visible):
        return self.set_managed_dock_visible(dock, visible, reason="legacy")

    def set_managed_dock_visible(self, dock, visible, *, reason, preserve_canvas=True, raise_dock=True):
        win = self.window
        currently_visible = dock.isVisible() if win.isVisible() else not dock.isHidden()
        if currently_visible == bool(visible):
            if visible and raise_dock:
                dock.raise_()
            return
        before = self._view_snapshot() if preserve_canvas else None
        if visible and not dock.isFloating():
            self._redock_managed_dock(dock)
            if preserve_canvas:
                self._grow_for_opening_dock(dock)
        self._visibility_preserve_active = True
        try:
            self._apply_dock_visibility_raw(dock, bool(visible))
            if visible and raise_dock:
                dock.raise_()
        finally:
            self._visibility_preserve_active = False
        if preserve_canvas:
            Qt.QtCore.QTimer.singleShot(0, lambda before=before: self._restore_view_snapshot_if_reasonable(before, retry=True))
        self.schedule_view_geometry_refresh()

    def make_managed_dock_action(self, text, dock, setter):
        action = Qt.QtGui.QAction(text, self.window)
        action.setCheckable(True)
        action.setChecked(dock.isVisible() if self.window.isVisible() else not dock.isHidden())

        def on_triggered(checked=False):
            setter(bool(checked))

        def sync_checked(visible):
            blocker = Qt.QtCore.QSignalBlocker(action)
            try:
                action.setChecked(bool(visible))
            finally:
                del blocker

        action.triggered.connect(on_triggered)
        dock.visibilityChanged.connect(sync_checked)
        return action

    def close_managed_docks_for_shutdown(self):
        for dock in self._managed_docks():
            if dock is None:
                continue
            self._visibility_preserve_active = True
            try:
                self._apply_dock_visibility_raw(dock, False)
                dock.close()
            finally:
                self._visibility_preserve_active = False

    def _managed_docks(self):
        win = self.window
        return (
            getattr(win, "inspection_dock", None),
            getattr(win, "profile_dock", None),
            getattr(win, "operation_dock", None),
        )

    def _state_for(self, dock):
        return self._dock_states.setdefault(dock, ManagedDockState())

    def set_managed_dock_floating(self, dock, floating, *, reason, preserve_canvas=True):
        before = self._view_snapshot() if preserve_canvas else None
        self._apply_dock_floating_raw(dock, bool(floating))
        if not floating:
            self._redock_managed_dock(dock)
        if preserve_canvas:
            Qt.QtCore.QTimer.singleShot(0, lambda before=before: self._restore_view_snapshot_if_reasonable(before, retry=True))
        self.schedule_view_geometry_refresh()

    def _apply_dock_visibility_raw(self, dock, visible):
        dock.setVisible(bool(visible))

    def _apply_dock_floating_raw(self, dock, floating):
        dock.setFloating(bool(floating))

    def _redock_managed_dock(self, dock):
        win = self.window
        if dock is getattr(win, "inspection_dock", None):
            win.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        elif dock is getattr(win, "profile_dock", None):
            win.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        elif dock is getattr(win, "operation_dock", None):
            win.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def refresh_view_geometry(self):
        win = self.window
        if hasattr(win, "centralWidget") and win.centralWidget() is not None:
            win.centralWidget().updateGeometry()

    def _view_snapshot(self):
        rect = self._view_global_rect()
        view_box = getattr(getattr(self.window, "img_view", None), "getView", lambda: None)()
        view_range = None if view_box is None else view_box.viewRange()
        return None if rect is None else (rect, view_range)

    def _view_global_rect(self):
        win = self.window
        view = getattr(win, "img_view", None)
        if view is None or not view.isVisible():
            return None
        top_left = view.mapToGlobal(Qt.QtCore.QPoint(0, 0))
        return Qt.QtCore.QRect(top_left, view.size())

    def _restore_view_snapshot_if_reasonable(self, before, *, retry=False):
        if before is None:
            return
        before_rect, before_range = before
        win = self.window
        if getattr(win, "_closing", False):
            return
        if win.isMaximized() or win.isFullScreen():
            self._restore_view_range(before_range)
            self.refresh_view_geometry()
            return
        view = getattr(win, "img_view", None)
        if view is None or not view.isVisible():
            return
        after = self._view_global_rect()
        if after is None:
            self.refresh_view_geometry()
            return
        delta_w = int(before_rect.width() - after.width())
        delta_h = int(before_rect.height() - after.height())
        move_delta = before_rect.topLeft() - after.topLeft()
        if delta_w == 0 and delta_h == 0 and move_delta.isNull():
            self.refresh_view_geometry()
            return
        current = win.geometry()
        target = Qt.QtCore.QRect(current)
        minimum = win.minimumSize()
        target.setWidth(max(int(minimum.width()), current.width() + delta_w))
        target.setHeight(max(int(minimum.height()), current.height() + delta_h))
        target.translate(move_delta)
        screen = win.screen()
        available = screen.availableGeometry() if screen is not None else None
        if available is not None:
            if target.width() > available.width():
                target.setWidth(available.width())
            if target.height() > available.height():
                target.setHeight(available.height())
            if target.left() < available.left():
                target.moveLeft(available.left())
            if target.top() < available.top():
                target.moveTop(available.top())
            if target.right() > available.right():
                target.moveRight(available.right())
            if target.bottom() > available.bottom():
                target.moveBottom(available.bottom())
        if target != current:
            win.setGeometry(target)
        self._restore_view_range(before_range)
        Qt.QtCore.QTimer.singleShot(0, lambda view_range=before_range: self._restore_view_range(view_range))
        if retry:
            Qt.QtCore.QTimer.singleShot(50, lambda before=(before_rect, before_range): self._restore_view_snapshot_if_reasonable(before, retry=False))
        self.refresh_view_geometry()

    def _restore_view_range(self, view_range):
        if view_range is None:
            return
        view_box = getattr(getattr(self.window, "img_view", None), "getView", lambda: None)()
        if view_box is not None:
            view_box.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    def _grow_for_opening_dock(self, dock):
        win = self.window
        if win.isMaximized() or win.isFullScreen():
            return
        area = win.dockWidgetArea(dock)
        width_delta, height_delta = self._dock_extent_for_area(dock, area)
        if width_delta <= 0 and height_delta <= 0:
            return
        target = Qt.QtCore.QRect(win.geometry())
        target.setWidth(target.width() + width_delta)
        target.setHeight(target.height() + height_delta)
        if area == Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea:
            target.translate(-width_delta, 0)
        elif area == Qt.QtCore.Qt.DockWidgetArea.TopDockWidgetArea:
            target.translate(0, -height_delta)
        target = self._clamp_to_available_screen(target)
        if target != win.geometry():
            win.setGeometry(target)

    def _dock_extent_for_area(self, dock, area=None):
        if area is None:
            area = self.window.dockWidgetArea(dock)
        dock_extent = dock.size().expandedTo(dock.minimumSizeHint()).expandedTo(dock.minimumSize())
        width_delta = int(dock_extent.width()) if area in (
            Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
        ) else 0
        height_delta = int(dock_extent.height()) if area in (
            Qt.QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
            Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        ) else 0
        return width_delta, height_delta

    def _clamp_to_available_screen(self, target):
        screen = self.window.screen()
        available = screen.availableGeometry() if screen is not None else None
        if available is None:
            return target
        if target.width() > available.width():
            target.setWidth(available.width())
        if target.height() > available.height():
            target.setHeight(available.height())
        if target.left() < available.left():
            target.moveLeft(available.left())
        if target.top() < available.top():
            target.moveTop(available.top())
        if target.right() > available.right():
            target.moveRight(available.right())
        if target.bottom() > available.bottom():
            target.moveBottom(available.bottom())
        return target

    def resize_profile_dock_default(self):
        win = self.window
        if not hasattr(win, "profile_dock") or not win.profile_dock.isVisible() or win.profile_dock.isFloating():
            return
        target_height = max(140, int(win.height() * 0.23))
        try:
            win.resizeDocks([win.profile_dock], [target_height], Qt.QtCore.Qt.Orientation.Vertical)
        except Exception as exc:
            handle_ui_exception("resize profile dock", exc)

    def resize_default_docks(self):
        win = self.window
        try:
            if win.profile_dock.isVisible() and not win.profile_dock.isFloating():
                win.resizeDocks([win.profile_dock], [max(140, int(win.height() * 0.23))], Qt.QtCore.Qt.Orientation.Vertical)
            if win.operation_dock.isVisible() and not win.operation_dock.isFloating():
                win.resizeDocks([win.operation_dock], [max(220, int(win.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
            if hasattr(win, "inspection_dock") and win.inspection_dock.isVisible() and not win.inspection_dock.isFloating():
                win.resizeDocks([win.inspection_dock], [max(240, int(win.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
        except Exception as exc:
            handle_ui_exception("resize default docks", exc)
