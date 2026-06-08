"""Managed docked/detached panel state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.ui.icons import set_button_icon


class PanelLocation(Enum):
    HIDDEN = "hidden"
    DOCKED = "docked"
    DETACHED = "detached"


@dataclass
class ManagedPanel:
    name: str
    title: str
    dock: QtWidgets.QDockWidget
    body: QtWidgets.QWidget
    dialog: QtWidgets.QDialog | None
    location: PanelLocation
    last_dock_area: Qt.QtCore.Qt.DockWidgetArea
    placeholder: QtWidgets.QWidget | None = None
    user_visible: bool | None = None
    auto_visible: bool = False


class PanelManager:
    def __init__(self, window):
        self.window = window
        self._panels_by_name = {}
        self._names_by_dock = {}

    def register_panel(self, name, title, dock, body=None, default_area=None):
        body = QtWidgets.QDockWidget.widget(dock) if body is None else body
        if body is None:
            raise ValueError(f"panel {name!r} has no body widget")
        default_area = self.window.dockWidgetArea(dock) if default_area is None else default_area
        features = dock.features()
        try:
            features &= ~QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            dock.setFeatures(features)
        except Exception:
            pass
        panel = ManagedPanel(
            name=str(name),
            title=str(title),
            dock=dock,
            body=body,
            dialog=None,
            location=PanelLocation.DOCKED if dock.isVisible() else PanelLocation.HIDDEN,
            last_dock_area=default_area,
        )
        self._panels_by_name[panel.name] = panel
        self._names_by_dock[dock] = panel.name
        dock.setTitleBarWidget(_DockPanelTitleBar(self.window, panel, self))
        return panel

    def name_for_dock(self, dock):
        return self._names_by_dock.get(dock)

    def panel_for_dock(self, dock):
        name = self.name_for_dock(dock)
        return None if name is None else self._panels_by_name[name]

    def is_visible(self, name):
        panel = self._panels_by_name[str(name)]
        return panel.location in (PanelLocation.DOCKED, PanelLocation.DETACHED)

    def location(self, name):
        return self._panels_by_name[str(name)].location

    def show_docked(self, name, *, reason, preserve_canvas=True):
        del reason, preserve_canvas
        panel = self._panels_by_name[str(name)]
        if panel.location == PanelLocation.DETACHED:
            self.redock_panel(panel.name, reason="show-docked", preserve_canvas=False)
            return
        self._destroy_dialog_and_take_body(panel)
        self._set_body_in_dock(panel)
        self.window.addDockWidget(panel.last_dock_area, panel.dock)
        panel.dock.setVisible(True)
        panel.location = PanelLocation.DOCKED
        self._sync_view_action(panel)

    def hide_panel(self, name, *, reason, preserve_canvas=True):
        del reason, preserve_canvas
        panel = self._panels_by_name[str(name)]
        if panel.dialog is not None:
            self._destroy_dialog_and_take_body(panel)
            self._store_body_in_hidden_dock(panel)
        else:
            self._hide_dock(panel)
        panel.location = PanelLocation.HIDDEN
        self._sync_view_action(panel)

    def detach_panel(self, name, *, reason, preserve_canvas=True):
        del reason, preserve_canvas
        panel = self._panels_by_name[str(name)]
        if panel.location == PanelLocation.DETACHED:
            if panel.dialog is not None:
                panel.dialog.show()
                panel.dialog.raise_()
            return
        if panel.location == PanelLocation.DOCKED:
            panel.last_dock_area = self.window.dockWidgetArea(panel.dock)
        self._create_detached_dialog(panel)
        panel.location = PanelLocation.DETACHED
        self._sync_view_action(panel)

    def redock_panel(self, name, *, reason, preserve_canvas=True):
        del reason, preserve_canvas
        panel = self._panels_by_name[str(name)]
        self._destroy_dialog_and_take_body(panel)
        self._set_body_in_dock(panel)
        self.window.addDockWidget(panel.last_dock_area, panel.dock)
        panel.dock.setVisible(True)
        panel.location = PanelLocation.DOCKED
        self._sync_view_action(panel)

    def toggle_panel(self, name, *, reason):
        if self.is_visible(name):
            self.hide_panel(name, reason=reason)
        else:
            self.show_docked(name, reason=reason)

    def reset_panel_layout(self):
        for name in tuple(self._panels_by_name):
            self.window.layout_manager.redock_managed_panel(self._panels_by_name[name].dock, reason="reset", preserve_canvas=False)
            self.hide_panel(name, reason="reset", preserve_canvas=False)

    def _hide_detached_from_dialog(self, name):
        panel = self._panels_by_name[str(name)]
        self._destroy_dialog_and_take_body(panel)
        self._store_body_in_hidden_dock(panel)
        panel.location = PanelLocation.HIDDEN
        self._sync_view_action(panel)

    def _destroy_dialog_and_take_body(self, panel: ManagedPanel) -> QtWidgets.QWidget | None:
        if panel.dialog is None:
            return panel.body
        dialog = panel.dialog
        body = dialog.take_body()
        dialog._closing_for_redock = True
        dialog.close()
        dialog.deleteLater()
        panel.dialog = None
        if body is not None:
            panel.body = body
        return panel.body

    def _set_body_in_dock(self, panel: ManagedPanel) -> None:
        if panel.body is None:
            raise RuntimeError(f"panel {panel.name!r} has no body widget")
        if panel.placeholder is not None and panel.placeholder is not panel.body:
            panel.placeholder.deleteLater()
            panel.placeholder = None
        if QtWidgets.QDockWidget.widget(panel.dock) is not panel.body:
            panel.body.setParent(None)
            panel.dock.setWidget(panel.body)

    def _store_body_in_hidden_dock(self, panel: ManagedPanel) -> None:
        self._set_body_in_dock(panel)
        self._hide_dock(panel)

    def _hide_dock(self, panel: ManagedPanel) -> None:
        panel.dock.setVisible(False)
        self.window.removeDockWidget(panel.dock)

    def _create_detached_dialog(self, panel: ManagedPanel) -> None:
        self._destroy_dialog_and_take_body(panel)
        body = QtWidgets.QDockWidget.widget(panel.dock)
        if body is not None:
            panel.body = body
        if panel.body is None:
            raise RuntimeError(f"panel {panel.name!r} has no body widget to detach")
        panel.body.setParent(None)
        panel.placeholder = QtWidgets.QWidget(panel.dock)
        panel.dock.setWidget(panel.placeholder)
        self._hide_dock(panel)
        dialog = _DetachedPanelDialog(
            self.window,
            panel.title,
            panel.body,
            on_redock=lambda _checked=False, dock=panel.dock: self.window.layout_manager.redock_managed_panel(dock, reason="dialog"),
        )
        dialog.closedByUser.connect(lambda name=panel.name: self._hide_detached_from_dialog(name))
        panel.dialog = dialog
        dialog.resize(max(320, panel.body.width()), max(220, panel.body.height()))
        dialog.show()

    def _sync_view_action(self, panel):
        action = getattr(panel.dock, "_arrayscope_managed_action", None)
        if action is None:
            return
        blocker = Qt.QtCore.QSignalBlocker(action)
        try:
            action.setChecked(panel.location in (PanelLocation.DOCKED, PanelLocation.DETACHED))
        finally:
            del blocker


class _DetachedPanelDialog(QtWidgets.QDialog):
    closedByUser = Qt.QtCore.Signal()

    def __init__(self, parent, title, body, *, on_redock):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.QtCore.Qt.WindowType.Tool, True)
        self._body = body
        self._on_redock = on_redock
        self._closing_for_redock = False
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_bar = QtWidgets.QWidget(self)
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(6, 4, 6, 4)
        self.move_handle = QtWidgets.QLabel(title)
        self.move_handle.setObjectName("DetachedPanelMoveHandle")
        redock = QtWidgets.QToolButton()
        redock.setObjectName("DetachedPanelRedockButton")
        redock.setText("Dock")
        redock.clicked.connect(self._on_redock)
        title_layout.addWidget(self.move_handle, 1)
        title_layout.addWidget(redock)
        layout.addWidget(title_bar)
        layout.addWidget(body, 1)
        self.move_handle.installEventFilter(self)

    def take_body(self):
        if self._body is not None:
            self.layout().removeWidget(self._body)
            self._body.setParent(None)
        body = self._body
        self._body = None
        return body

    def closeEvent(self, event):
        if not self._closing_for_redock:
            self.closedByUser.emit()
        super().closeEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.move_handle and event.type() == Qt.QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == Qt.QtCore.Qt.MouseButton.LeftButton:
                window = self.window().windowHandle()
                if window is not None:
                    window.startSystemMove()
                return True
        return super().eventFilter(obj, event)


class _DockPanelTitleBar(QtWidgets.QWidget):
    def __init__(self, window, panel, manager):
        super().__init__(panel.dock)
        self.window = window
        self.panel = panel
        self.manager = manager
        self._press_pos = None
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 4, 3)
        layout.setSpacing(4)
        self.label = QtWidgets.QLabel(panel.title)
        self.label.setObjectName("ManagedDockTitleLabel")
        self.detach_button = QtWidgets.QToolButton()
        self.detach_button.setObjectName("ManagedDockDetachButton")
        set_button_icon(self.detach_button, "open_in_new", tooltip="Detach panel")
        self.close_button = QtWidgets.QToolButton()
        self.close_button.setObjectName("ManagedDockCloseButton")
        set_button_icon(self.close_button, "close", tooltip="Hide panel")
        layout.addWidget(self.label, 1)
        layout.addWidget(self.detach_button)
        layout.addWidget(self.close_button)
        self.detach_button.clicked.connect(self._detach)
        self.close_button.clicked.connect(self._hide)

    def _detach(self):
        self.window.layout_manager.detach_managed_dock(self.panel.dock, reason="titlebar-detach")

    def _hide(self):
        if self.panel.name == "inspection":
            self.window.layout_manager.set_inspection_dock_visible_from_user(False)
        elif self.panel.name == "profile":
            self.window.layout_manager.set_profile_dock_visible_from_user(False)
        elif self.panel.name == "operations":
            self.window.layout_manager.set_operation_dock_visible_from_user(False)
        else:
            self.window.layout_manager.set_managed_dock_visible(self.panel.dock, False, reason="titlebar-close")

    def mousePressEvent(self, event):
        if event.button() == Qt.QtCore.Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is None:
            return super().mouseMoveEvent(event)
        distance = (event.position().toPoint() - self._press_pos).manhattanLength()
        if distance >= QtWidgets.QApplication.startDragDistance():
            self._press_pos = None
            self._detach()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        super().mouseReleaseEvent(event)
