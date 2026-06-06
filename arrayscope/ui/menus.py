from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.app.settings_state import AppSettingsState, settings_from_mapping, settings_to_mapping
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.ui.icons import set_action_icon
from arrayscope.ui.toasts import show_status_message


class WindowMenuMixin:
    def _load_app_settings(self):
        return settings_from_mapping(
            {
                "theme": self._settings.value("theme", ThemeChoice.SYSTEM.value),
                "prefetch_nearby_slices": self._settings.value("prefetch_nearby_slices", False),
            }
        )

    def _save_app_settings(self):
        for key, value in settings_to_mapping(self.app_settings).items():
            self._settings.setValue(key, value)

    def _setup_menus(self):
        file_menu = self.menuBar().addMenu("File")
        save_recipe_action = QtGui.QAction("Save Operation Recipe", self)
        set_action_icon(save_recipe_action, "save")
        save_recipe_action.triggered.connect(self.save_operation_recipe)
        load_recipe_action = QtGui.QAction("Load Operation Recipe", self)
        set_action_icon(load_recipe_action, "folder_open")
        load_recipe_action.triggered.connect(self.load_operation_recipe)
        save_view_action = QtGui.QAction("Save View Recipe", self)
        set_action_icon(save_view_action, "view_quilt")
        save_view_action.triggered.connect(self.save_view_recipe)
        load_view_action = QtGui.QAction("Load View Recipe", self)
        set_action_icon(load_view_action, "folder_open")
        load_view_action.triggered.connect(self.load_view_recipe)
        export_derived_action = QtGui.QAction("Export Derived Array", self)
        set_action_icon(export_derived_action, "download")
        export_derived_action.triggered.connect(self.export_derived_array)
        for action in (save_recipe_action, load_recipe_action, save_view_action, load_view_action, export_derived_action):
            file_menu.addAction(action)

        view_menu = self.menuBar().addMenu("View")
        operation_action = self.operation_dock.toggleViewAction()
        operation_action.triggered.connect(lambda visible: self._set_operation_dock_visible_from_user(visible))
        profile_action = self.profile_dock.toggleViewAction()
        profile_action.triggered.connect(lambda visible: self._set_profile_dock_visible_from_user(visible))
        view_menu.addAction(operation_action)
        view_menu.addAction(profile_action)
        command_palette_action = QtGui.QAction("Command Palette", self)
        set_action_icon(command_palette_action, "search")
        command_palette_action.setShortcut(QtGui.QKeySequence("Ctrl+K"))
        command_palette_action.triggered.connect(self.open_command_palette)
        view_menu.addAction(command_palette_action)
        view_menu.addSeparator()
        reset_layout_action = QtGui.QAction("Reset layout", self)
        set_action_icon(reset_layout_action, "reset_wrench")
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        theme_menu = self.menuBar().addMenu("Theme")
        self._theme_actions = {}
        self._theme_action_group = QtGui.QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for choice, label in (
            (ThemeChoice.SYSTEM, "System / Native"),
            (ThemeChoice.NATIVE, "Native"),
            (ThemeChoice.DARK, "Dark"),
            (ThemeChoice.LIGHT, "Light"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._theme_action_group.addAction(action)
            action.triggered.connect(lambda checked=False, choice=choice: self._apply_theme_choice(choice))
            theme_menu.addAction(action)
            self._theme_actions[choice] = action
        self._sync_theme_actions()

    def _sync_theme_actions(self):
        if not hasattr(self, "_theme_actions"):
            return
        for choice, action in self._theme_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.theme == choice)
            action.blockSignals(False)

    def _apply_theme_choice(self, choice, persist=True):
        result = apply_theme_to_qapplication(QtWidgets.QApplication.instance(), choice)
        if result.warning:
            show_status_message(self, f"Theme warning: {result.warning}")
            if persist:
                QtWidgets.QMessageBox.warning(self, "Theme Warning", result.warning)
        theme_to_store = result.requested if result.applied == result.requested else result.applied
        self.applied_theme = result.applied
        self.theme_backend = result.backend
        self.app_settings = AppSettingsState(theme=theme_to_store, prefetch_nearby_slices=getattr(self, "app_settings", AppSettingsState()).prefetch_nearby_slices)
        if persist:
            self._save_app_settings()
        self._sync_theme_actions()

    def _set_prefetch_enabled(self, enabled):
        self.app_settings = AppSettingsState(theme=self.app_settings.theme, prefetch_nearby_slices=bool(enabled))
        self._save_app_settings()

    def _restore_window_settings(self):
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value("window_state")
        if state is not None:
            self.restoreState(state)
        if not self.profile_dock.isVisible() and self.data.ndim == 1:
            self.profile_dock.show()
        self._sync_progressive_docks()
        Qt.QtCore.QTimer.singleShot(0, self._resize_default_docks)

    def reset_layout(self):
        self._operation_dock_user_visible = False
        self._profile_dock_user_visible = False
        self.profile_dock.setFloating(False)
        self.profile_dock.hide()
        if self.data.ndim == 1:
            self.profile_dock.show()
        self.operation_dock.setFloating(False)
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.operation_dock)
        if self.document.steps:
            self.operation_dock.show()
        else:
            self.operation_dock.hide()
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.profile_dock)
        Qt.QtCore.QTimer.singleShot(0, self._resize_default_docks)

    def _set_operation_dock_visible_from_user(self, visible):
        self._operation_dock_user_visible = bool(visible)
        self.operation_dock.setVisible(bool(visible))
        self._schedule_view_geometry_refresh()

    def _set_profile_dock_visible_from_user(self, visible):
        self._profile_dock_user_visible = bool(visible)
        self.profile_dock.setVisible(bool(visible))
        self._schedule_view_geometry_refresh()

    def _resize_default_docks(self):
        try:
            if self.profile_dock.isVisible() and not self.profile_dock.isFloating():
                self.resizeDocks([self.profile_dock], [max(140, int(self.height() * 0.23))], Qt.QtCore.Qt.Orientation.Vertical)
            if self.operation_dock.isVisible() and not self.operation_dock.isFloating():
                self.resizeDocks([self.operation_dock], [max(220, int(self.width() * 0.24))], Qt.QtCore.Qt.Orientation.Horizontal)
        except Exception:
            pass

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("window_state", self.saveState())
        self._save_app_settings()
        super().closeEvent(event)
