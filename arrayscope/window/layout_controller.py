"""Centralized window/dock layout behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.window.canvas_preserve import CanvasPreserveController
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
        self.canvas_preserver = CanvasPreserveController(self)

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
        self.sync_progressive_docks(preserve_canvas=False)
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

    def sync_progressive_docks(self, *, preserve_canvas=True):
        win = self.window
        preserve_canvas = bool(preserve_canvas) and bool(getattr(win, "_progressive_preserve_enabled", True))
        changed = False
        if hasattr(win, "operation_dock"):
            has_steps = bool(win.document.steps)
            user_visible = getattr(win, "_operation_dock_user_visible", None)
            if has_steps and user_visible is not False:
                changed = changed or not win.operation_dock.isVisible()
                self.set_dock_visible_later(win.operation_dock, True, preserve_canvas=preserve_canvas)
            elif not has_steps and user_visible is not True:
                changed = changed or win.operation_dock.isVisible()
                self.set_dock_visible_later(win.operation_dock, False, preserve_canvas=preserve_canvas)
        if hasattr(win, "profile_dock"):
            live_profile = win.widgets["buttons"]["display"]["live_profile"].isChecked()
            user_visible = getattr(win, "_profile_dock_user_visible", None)
            should_show_profile = (win.data.ndim == 1 or live_profile or user_visible is True) and user_visible is not False
            if should_show_profile:
                changed = changed or not win.profile_dock.isVisible()
                self.set_dock_visible_later(win.profile_dock, True, preserve_canvas=preserve_canvas)
            elif user_visible is not True:
                changed = changed or win.profile_dock.isVisible()
                self.set_dock_visible_later(win.profile_dock, False, preserve_canvas=preserve_canvas)
        if changed:
            self.schedule_view_geometry_refresh()

    def schedule_view_geometry_refresh(self):
        Qt.QtCore.QTimer.singleShot(0, self.refresh_view_geometry)

    def set_dock_visible_later(self, dock, visible, *, preserve_canvas=True):
        self._state_for(dock).auto_visible = bool(visible)
        Qt.QtCore.QTimer.singleShot(
            0,
            lambda dock=dock, visible=visible, preserve_canvas=preserve_canvas: self.apply_queued_dock_visibility(
                dock,
                visible,
                preserve_canvas=preserve_canvas,
            ),
        )

    def apply_queued_dock_visibility(self, dock, visible, *, preserve_canvas=True):
        win = self.window
        if not self._window_alive():
            return
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
        self.set_managed_dock_visible(dock, bool(visible), reason="progressive", preserve_canvas=preserve_canvas)

    def set_dock_visible_preserving_canvas(self, dock, visible):
        return self.set_managed_dock_visible(dock, visible, reason="legacy")

    def set_managed_dock_visible(self, dock, visible, *, reason, preserve_canvas=True, raise_dock=True):
        win = self.window
        if not self._window_alive():
            return
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
            def transition():
                panel_manager.redock_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
                if raise_dock:
                    dock.raise_()

            self.run_panel_transition_preserving_canvas(transition, preserve_canvas=preserve_canvas, transition_name="redock")
            self.schedule_view_geometry_refresh()
            return
        currently_visible = dock.isVisible() if win.isVisible() else not dock.isHidden()
        if currently_visible == bool(visible):
            if visible and raise_dock:
                dock.raise_()
            return

        def transition():
            if visible:
                self._add_dock_to_panel_area(dock)
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

        self.run_panel_transition_preserving_canvas(
            transition,
            preserve_canvas=preserve_canvas,
            transition_name="show-docked" if visible else "hide-docked",
        )
        self.schedule_view_geometry_refresh()

    def _window_alive(self) -> bool:
        win = self.window
        if getattr(win, "_closing", False):
            return False
        try:
            win.isVisible()
            return True
        except RuntimeError:
            return False

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
        self.canvas_preserver.cancel()
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
        if was_docked:
            def transition():
                panel_manager.detach_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
                self._activate_main_window_layout()

            self.run_panel_transition_preserving_canvas(
                transition,
                preserve_canvas=preserve_canvas,
                strong_preserve=False,
                transition_name="detach",
            )
        else:
            panel_manager.detach_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)
        self.schedule_view_geometry_refresh()

    def redock_managed_panel(self, dock, *, reason, preserve_canvas=True):
        panel_manager = getattr(self.window, "panel_manager", None)
        panel = None if panel_manager is None else panel_manager.panel_for_dock(dock)
        if panel is None:
            return
        if panel.location != PanelLocation.DOCKED:
            def transition():
                panel_manager.redock_panel(panel.name, reason=reason, preserve_canvas=preserve_canvas)

            self.run_panel_transition_preserving_canvas(transition, preserve_canvas=preserve_canvas, transition_name="redock")
        else:
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

    def run_panel_transition_preserving_canvas(
        self,
        transition,
        *,
        preserve_canvas: bool,
        strong_preserve: bool = True,
        transition_name: str = "",
    ) -> None:
        self.canvas_preserver.run(
            transition,
            preserve_canvas=preserve_canvas,
            allow_strong=strong_preserve,
            transition_name=transition_name,
        )

    def _visible_managed_dock_extents(self):
        extents = []
        for dock in self._managed_docks():
            if dock is None or not dock.isVisible():
                continue
            area = self._dock_area_for_transition(dock)
            orientation = (
                Qt.QtCore.Qt.Orientation.Horizontal
                if self._is_horizontal_dock_area(area)
                else Qt.QtCore.Qt.Orientation.Vertical
            )
            size = dock.width() if orientation == Qt.QtCore.Qt.Orientation.Horizontal else dock.height()
            if size > 0:
                extents.append((dock, int(size), orientation))
        return tuple(extents)

    def _restore_visible_dock_extents(self, dock_extents) -> None:
        for orientation in (Qt.QtCore.Qt.Orientation.Horizontal, Qt.QtCore.Qt.Orientation.Vertical):
            docks = [dock for dock, _size, item_orientation in dock_extents if item_orientation == orientation and dock.isVisible()]
            sizes = [size for dock, size, item_orientation in dock_extents if item_orientation == orientation and dock.isVisible()]
            if not docks:
                continue
            try:
                self.window.resizeDocks(docks, sizes, orientation)
            except Exception as exc:
                handle_ui_exception("restore managed dock extents", exc)

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
