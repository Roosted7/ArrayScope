from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.app.settings_state import AppSettingsState, settings_from_mapping, settings_to_mapping
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.operations.registry import operation_entries
from arrayscope.ui.icons import set_action_icon, verify_icon_names
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
        operation_action = self.layout_manager.make_managed_dock_action(
            "Operations", self.operation_dock, self._set_operation_dock_visible_from_user
        )
        profile_action = self.layout_manager.make_managed_dock_action(
            "Profile", self.profile_dock, self._set_profile_dock_visible_from_user
        )
        view_menu.addAction(operation_action)
        view_menu.addAction(profile_action)
        if hasattr(self, "inspection_dock"):
            inspection_action = self.layout_manager.make_managed_dock_action(
                "Inspection", self.inspection_dock, self._set_inspection_dock_visible_from_user
            )
            view_menu.addAction(inspection_action)
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
        verify_icons_action = QtGui.QAction("Verify icons", self)
        set_action_icon(verify_icons_action, "warning")
        verify_icons_action.triggered.connect(self.verify_icons)
        view_menu.addAction(verify_icons_action)

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

    def verify_icons(self):
        names = {
            "save",
            "folder_open",
            "view_quilt",
            "download",
            "search",
            "reset_wrench",
            "warning",
            "show_chart",
            "monitor_heart",
            "data_array",
            "data_object",
            "functions",
            "analytics",
            "aspect_ratio",
            "edit",
            "fit_screen",
            "waves",
            "crop",
            "join_inner",
            "call_split",
            "drag_indicator",
            "inventory_2",
        }
        names.update(entry.id for entry in operation_entries())
        result = verify_icon_names(sorted(names))
        missing = [name for name, ok in result.items() if not ok]
        if missing:
            message = "Missing/null icons: " + ", ".join(missing)
            QtWidgets.QMessageBox.warning(self, "Verify icons", message)
        else:
            show_status_message(self, f"Verified {len(result)} icons")

    def _restore_window_settings(self):
        self.layout_manager.restore_window_settings()

    def reset_layout(self):
        self.layout_manager.reset_layout()

    def _set_operation_dock_visible_from_user(self, visible):
        self.layout_manager.set_operation_dock_visible_from_user(visible)

    def _set_profile_dock_visible_from_user(self, visible):
        self.layout_manager.set_profile_dock_visible_from_user(visible)

    def _set_inspection_dock_visible_from_user(self, visible):
        self.layout_manager.set_inspection_dock_visible_from_user(visible)

    def _resize_default_docks(self):
        self.layout_manager.resize_default_docks()

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("window_state", self.saveState())
        self._save_app_settings()
        self._closing = True
        watcher = getattr(self, "_file_watcher", None)
        if watcher is not None:
            watcher.deleteLater()
            self._file_watcher = None
        for controller_name in ("evaluation_controller", "pixel_evaluation_controller", "profile_evaluation_controller", "roi_evaluation_controller"):
            controller = getattr(self, controller_name, None)
            if controller is not None:
                controller.shutdown_for_close()
        if hasattr(self, "_profile_timer"):
            self._profile_timer.stop()
        if hasattr(self, "layout_manager"):
            self.layout_manager.close_managed_docks_for_shutdown()
        super().closeEvent(event)
        Qt.QtCore.QTimer.singleShot(0, self._quit_if_last_arrayscope_window)

    def _quit_if_last_arrayscope_window(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        visible_arrayscope = [
            widget
            for widget in app.topLevelWidgets()
            if widget is not self and type(widget).__name__ == type(self).__name__ and widget.isVisible()
        ]
        if not visible_arrayscope:
            app.quit()
