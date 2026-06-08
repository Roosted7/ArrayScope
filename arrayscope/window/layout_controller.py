"""Centralized window/dock layout behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.window.panels import PanelLocation


@dataclass
class ManagedDockState:
    user_visible: bool | None = None
    auto_visible: bool = False
    reserved_extent: int | None = None
    active_extent: int | None = None
    last_area: Qt.QtCore.Qt.DockWidgetArea | None = None


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
        self.set_managed_dock_visible(win.profile_dock, False, reason="reset", preserve_canvas=False)
        if hasattr(win, "inspection_dock"):
            self.redock_managed_panel(win.inspection_dock, reason="reset", preserve_canvas=False)
            self.set_managed_dock_visible(win.inspection_dock, False, reason="reset", preserve_canvas=False)
        if win.data.ndim == 1:
            self.set_managed_dock_visible(win.profile_dock, True, reason="reset-one-dimensional", preserve_canvas=False)
        self.redock_managed_panel(win.operation_dock, reason="reset", preserve_canvas=False)
        if win.document.steps:
            self.set_managed_dock_visible(win.operation_dock, True, reason="reset-operations", preserve_canvas=False)
        else:
            self.set_managed_dock_visible(win.operation_dock, False, reason="reset-operations", preserve_canvas=False)
        self._add_dock_to_panel_area(win.profile_dock)
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
        panel_manager = getattr(win, "panel_manager", None)
        panel = None if panel_manager is None else panel_manager.panel_for_dock(dock)
        if panel is not None and panel.location == PanelLocation.DETACHED:
            if visible:
                if panel.dialog is not None:
                    panel.dialog.show()
                    panel.dialog.raise_()
            else:
                panel_manager.hide_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
            self.schedule_view_geometry_refresh()
            return
        if panel is not None and visible and Qt.QtWidgets.QDockWidget.widget(dock) is not panel.body:
            self._apply_panel_visibility_delta(dock, opening=True, preserve_canvas=preserve_canvas)
            panel_manager.redock_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
            self.schedule_view_geometry_refresh()
            return
        currently_visible = dock.isVisible() if win.isVisible() else not dock.isHidden()
        if currently_visible == bool(visible):
            if visible and raise_dock:
                dock.raise_()
            return
        if visible:
            self._add_dock_to_panel_area(dock)
        transition_geometry = Qt.QtCore.QRect(win.geometry())
        if visible:
            self._apply_panel_visibility_delta(dock, opening=True, preserve_canvas=preserve_canvas)
        self._visibility_preserve_active = True
        try:
            self._apply_dock_visibility_raw(dock, bool(visible))
            if panel is not None:
                panel.location = PanelLocation.DOCKED if visible else PanelLocation.HIDDEN
                if not visible:
                    win.removeDockWidget(dock)
            if visible and raise_dock:
                dock.raise_()
        finally:
            self._visibility_preserve_active = False
        if not visible:
            self._activate_main_window_layout()
            self._apply_panel_visibility_delta(
                dock,
                opening=False,
                preserve_canvas=preserve_canvas,
                base_geometry=transition_geometry,
            )
        self.schedule_view_geometry_refresh()

    def make_managed_dock_action(self, text, dock, setter):
        action = Qt.QtGui.QAction(text, self.window)
        action.setCheckable(True)
        action.setChecked(dock.isVisible() if self.window.isVisible() else not dock.isHidden())
        dock._arrayscope_managed_action = action

        def on_triggered(checked=False):
            setter(bool(checked))

        def sync_checked(visible):
            panel_manager = getattr(self.window, "panel_manager", None)
            panel = None if panel_manager is None else panel_manager.panel_for_dock(dock)
            if panel is not None and panel.location == PanelLocation.DETACHED:
                visible = True
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

    def detach_managed_dock(self, dock, *, reason, preserve_canvas=True):
        panel_manager = getattr(self.window, "panel_manager", None)
        panel = None if panel_manager is None else panel_manager.panel_for_dock(dock)
        if panel is None:
            return
        was_docked = panel.location == PanelLocation.DOCKED
        transition_geometry = Qt.QtCore.QRect(self.window.geometry())
        panel_manager.detach_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
        if was_docked:
            self._activate_main_window_layout()
            self._apply_panel_visibility_delta(
                dock,
                opening=False,
                preserve_canvas=preserve_canvas,
                base_geometry=transition_geometry,
            )
        self.schedule_view_geometry_refresh()

    def redock_managed_panel(self, dock, *, reason, preserve_canvas=True):
        panel_manager = getattr(self.window, "panel_manager", None)
        panel = None if panel_manager is None else panel_manager.panel_for_dock(dock)
        if panel is None:
            return
        if panel.location != PanelLocation.DOCKED:
            self._apply_panel_visibility_delta(dock, opening=True, preserve_canvas=preserve_canvas)
        panel_manager.redock_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
        self.schedule_view_geometry_refresh()

    def _apply_dock_visibility_raw(self, dock, visible):
        dock.setVisible(bool(visible))

    def _add_dock_to_panel_area(self, dock):
        win = self.window
        area = self._dock_area_for_transition(dock)
        win.addDockWidget(area, dock)

    def refresh_view_geometry(self):
        win = self.window
        if hasattr(win, "centralWidget") and win.centralWidget() is not None:
            win.centralWidget().updateGeometry()

    def _activate_main_window_layout(self):
        layout = self.window.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.refresh_view_geometry()

    def _apply_panel_visibility_delta(self, dock, *, opening, preserve_canvas, base_geometry=None):
        win = self.window
        if not preserve_canvas or win.isMaximized() or win.isFullScreen():
            return
        area = self._dock_area_for_transition(dock)
        state = self._state_for(dock)
        if opening:
            width_delta, height_delta = self._dock_extent_for_area(dock, area)
            state.active_extent = width_delta if self._is_horizontal_dock_area(area) else height_delta
        elif state.active_extent is not None:
            extent = int(state.active_extent)
            width_delta = extent if self._is_horizontal_dock_area(area) else 0
            height_delta = 0 if self._is_horizontal_dock_area(area) else extent
        else:
            width_delta, height_delta = self._dock_extent_for_area(dock, area)
        if width_delta <= 0 and height_delta <= 0:
            return
        target = Qt.QtCore.QRect(win.geometry() if base_geometry is None else base_geometry)
        minimum = win.minimumSize()
        sign = 1 if opening else -1
        target.setWidth(max(int(minimum.width()), target.width() + sign * width_delta))
        target.setHeight(max(int(minimum.height()), target.height() + sign * height_delta))
        target = self._clamp_to_available_screen(target)
        if target != win.geometry():
            win.setGeometry(target)
        if not opening:
            state.active_extent = None

    def _dock_extent_for_area(self, dock, area=None):
        if area is None:
            area = self._dock_area_for_transition(dock)
        state = self._state_for(dock)
        if dock.isVisible():
            current = dock.width() if self._is_horizontal_dock_area(area) else dock.height()
            if current > 0:
                state.reserved_extent = max(int(state.reserved_extent or 0), int(current))
        fallback_extent = self._default_dock_extent(dock, area)
        panel_extent = max(int(state.reserved_extent or 0), fallback_extent)
        state.reserved_extent = panel_extent
        extent = panel_extent + self._dock_separator_extent()
        state.last_area = area
        width_delta = extent if self._is_horizontal_dock_area(area) else 0
        height_delta = 0 if self._is_horizontal_dock_area(area) else extent
        return width_delta, height_delta

    def _dock_area_for_transition(self, dock):
        panel_manager = getattr(self.window, "panel_manager", None)
        panel = None if panel_manager is None else panel_manager.panel_for_dock(dock)
        if panel is not None:
            return panel.last_dock_area
        state = self._state_for(dock)
        if state.last_area is not None:
            return state.last_area
        return self.window.dockWidgetArea(dock)

    def _default_dock_extent(self, dock, area):
        hint = dock.sizeHint().expandedTo(dock.minimumSizeHint()).expandedTo(dock.minimumSize())
        body = Qt.QtWidgets.QDockWidget.widget(dock)
        if body is not None:
            hint = hint.expandedTo(body.sizeHint()).expandedTo(body.minimumSizeHint()).expandedTo(body.minimumSize())
        if self._is_horizontal_dock_area(area):
            return max(220, int(hint.width()))
        return max(140, int(hint.height()))

    def _dock_separator_extent(self):
        try:
            return int(self.window.style().pixelMetric(Qt.QtWidgets.QStyle.PixelMetric.PM_DockWidgetSeparatorExtent))
        except Exception:
            return 6

    def _is_horizontal_dock_area(self, area):
        return area in (
            Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
        )

    def _clamp_to_available_screen(self, target):
        screen = self.window.screen()
        available = screen.availableGeometry() if screen is not None else None
        if available is None:
            return target
        if target.width() <= available.width() and target.left() < available.left():
            target.moveLeft(available.left())
        if target.height() <= available.height() and target.top() < available.top():
            target.moveTop(available.top())
        if target.width() <= available.width() and target.right() > available.right():
            target.moveRight(available.right())
        if target.height() <= available.height() and target.bottom() > available.bottom():
            target.moveBottom(available.bottom())
        return target

    def resize_profile_dock_default(self):
        win = self.window
        if not hasattr(win, "profile_dock") or not win.profile_dock.isVisible():
            return
        target_height = max(140, int(win.height() * 0.23))
        try:
            win.resizeDocks([win.profile_dock], [target_height], Qt.QtCore.Qt.Orientation.Vertical)
        except Exception as exc:
            handle_ui_exception("resize profile dock", exc)

    def resize_default_docks(self):
        win = self.window
        try:
            if win.profile_dock.isVisible():
                win.resizeDocks([win.profile_dock], [max(140, int(win.height() * 0.23))], Qt.QtCore.Qt.Orientation.Vertical)
            if win.operation_dock.isVisible():
                win.resizeDocks([win.operation_dock], [max(220, int(win.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
            if hasattr(win, "inspection_dock") and win.inspection_dock.isVisible():
                win.resizeDocks([win.inspection_dock], [max(240, int(win.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
        except Exception as exc:
            handle_ui_exception("resize default docks", exc)
