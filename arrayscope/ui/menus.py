from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.app.settings_state import (
    AppSettingsState,
    FFTBackendChoice,
    FFTWorkersChoice,
    PanelResizeBehavior,
    settings_from_mapping,
    settings_to_mapping,
)
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.operations import fft_backend
from arrayscope.operations.registry import operation_entries
from arrayscope.ui.icons import set_action_icon, verify_icon_names
from arrayscope.ui.toasts import show_status_message


class WindowMenuMixin:
    def _load_app_settings(self):
        return settings_from_mapping(
            {
                "theme": self._settings.value("theme", ThemeChoice.SYSTEM.value),
                "prefetch_nearby_slices": self._settings.value("prefetch_nearby_slices", False),
                "panel_resize_behavior": self._settings.value("panel_resize_behavior", PanelResizeBehavior.BEST_EFFORT.value),
                "fft_backend": self._settings.value("fft_backend", FFTBackendChoice.AUTO.value),
                "fft_workers": self._settings.value("fft_workers", FFTWorkersChoice.AUTO.value),
                "render_memory_budget_mb": self._settings.value("render_memory_budget_mb", 512),
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
        preserve_canvas_action = QtGui.QAction("Preserve Canvas Size on Panel Changes", self, checkable=True)
        preserve_canvas_action.setChecked(self.app_settings.panel_resize_behavior == PanelResizeBehavior.BEST_EFFORT)
        preserve_canvas_action.triggered.connect(self._set_preserve_canvas_enabled)
        view_menu.addAction(preserve_canvas_action)
        self._preserve_canvas_action = preserve_canvas_action
        reset_layout_action = QtGui.QAction("Reset layout", self)
        set_action_icon(reset_layout_action, "reset_wrench")
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        verify_icons_action = QtGui.QAction("Verify icons", self)
        set_action_icon(verify_icons_action, "warning")
        verify_icons_action.triggered.connect(self.verify_icons)
        view_menu.addAction(verify_icons_action)

        performance_menu = QtWidgets.QMenu("Performance", self)
        self.menuBar().addMenu(performance_menu)
        self._performance_menu = performance_menu
        self._fft_backend_actions = {}
        self._fft_backend_action_group = QtGui.QActionGroup(self)
        self._fft_backend_action_group.setExclusive(True)
        backend_menu = QtWidgets.QMenu("FFT Backend", self)
        performance_menu.addMenu(backend_menu)
        self._fft_backend_menu = backend_menu
        for choice, label in (
            (FFTBackendChoice.AUTO, "Auto"),
            (FFTBackendChoice.SCIPY, "SciPy"),
            (FFTBackendChoice.PYFFTW, "pyFFTW"),
            (FFTBackendChoice.NUMPY, "NumPy"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._fft_backend_action_group.addAction(action)
            action.triggered.connect(lambda checked=False, choice=choice: self._set_fft_backend_choice(choice))
            backend_menu.addAction(action)
            self._fft_backend_actions[choice] = action

        self._fft_workers_actions = {}
        self._fft_workers_action_group = QtGui.QActionGroup(self)
        self._fft_workers_action_group.setExclusive(True)
        workers_menu = QtWidgets.QMenu("FFT Workers", self)
        performance_menu.addMenu(workers_menu)
        self._fft_workers_menu = workers_menu
        for choice, label in (
            (FFTWorkersChoice.AUTO, "Auto"),
            (FFTWorkersChoice.ONE, "1"),
            (FFTWorkersChoice.TWO, "2"),
            (FFTWorkersChoice.FOUR, "4"),
            (FFTWorkersChoice.ALL_MINUS_ONE, "All minus one"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._fft_workers_action_group.addAction(action)
            action.triggered.connect(lambda checked=False, choice=choice: self._set_fft_workers_choice(choice))
            workers_menu.addAction(action)
            self._fft_workers_actions[choice] = action

        self._render_budget_actions = {}
        self._render_budget_action_group = QtGui.QActionGroup(self)
        self._render_budget_action_group.setExclusive(True)
        budget_menu = QtWidgets.QMenu("Render Memory Budget", self)
        performance_menu.addMenu(budget_menu)
        self._render_budget_menu = budget_menu
        for mb in (256, 512, 1024, 2048):
            action = QtGui.QAction(f"{mb} MiB", self, checkable=True)
            self._render_budget_action_group.addAction(action)
            action.triggered.connect(lambda checked=False, mb=mb: self._set_render_memory_budget_mb(mb))
            budget_menu.addAction(action)
            self._render_budget_actions[mb] = action
        self._sync_performance_actions()

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

    def _sync_performance_actions(self):
        if not hasattr(self, "_fft_backend_actions"):
            return
        for choice, action in self._fft_backend_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.fft_backend == choice)
            action.blockSignals(False)
        for choice, action in self._fft_workers_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.fft_workers == choice)
            action.blockSignals(False)
        for mb, action in self._render_budget_actions.items():
            action.blockSignals(True)
            action.setChecked(int(self.app_settings.render_memory_budget_mb) == int(mb))
            action.blockSignals(False)

    def _apply_performance_settings(self, persist=True):
        current = getattr(self, "app_settings", AppSettingsState())
        fft_backend.set_fft_runtime_options(backend=current.fft_backend.value, workers=current.fft_workers.value)
        resolved = fft_backend.resolve_fft_backend(current.fft_backend.value)
        if current.fft_backend == FFTBackendChoice.PYFFTW and resolved.name != "pyfftw":
            show_status_message(self, "pyFFTW is not installed; using SciPy FFT")
        if persist:
            self._save_app_settings()
        self._sync_performance_actions()

    def _set_fft_backend_choice(self, choice):
        self.app_settings = self._updated_app_settings(fft_backend=choice)
        self._apply_performance_settings(persist=True)

    def _set_fft_workers_choice(self, choice):
        self.app_settings = self._updated_app_settings(fft_workers=choice)
        self._apply_performance_settings(persist=True)

    def _set_render_memory_budget_mb(self, mb):
        self.app_settings = self._updated_app_settings(render_memory_budget_mb=int(mb))
        self._apply_performance_settings(persist=True)

    def _updated_app_settings(self, **changes):
        current = getattr(self, "app_settings", AppSettingsState())
        values = {
            "theme": current.theme,
            "prefetch_nearby_slices": current.prefetch_nearby_slices,
            "panel_resize_behavior": current.panel_resize_behavior,
            "fft_backend": current.fft_backend,
            "fft_workers": current.fft_workers,
            "render_memory_budget_mb": current.render_memory_budget_mb,
        }
        values.update(changes)
        return AppSettingsState(**values)

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
        current = getattr(self, "app_settings", AppSettingsState())
        self.app_settings = AppSettingsState(
            theme=theme_to_store,
            prefetch_nearby_slices=current.prefetch_nearby_slices,
            panel_resize_behavior=current.panel_resize_behavior,
            fft_backend=current.fft_backend,
            fft_workers=current.fft_workers,
            render_memory_budget_mb=current.render_memory_budget_mb,
        )
        if persist:
            self._save_app_settings()
        self._sync_theme_actions()

    def _set_prefetch_enabled(self, enabled):
        self.app_settings = AppSettingsState(
            theme=self.app_settings.theme,
            prefetch_nearby_slices=bool(enabled),
            panel_resize_behavior=self.app_settings.panel_resize_behavior,
            fft_backend=self.app_settings.fft_backend,
            fft_workers=self.app_settings.fft_workers,
            render_memory_budget_mb=self.app_settings.render_memory_budget_mb,
        )
        self._save_app_settings()

    def _set_preserve_canvas_enabled(self, enabled):
        behavior = PanelResizeBehavior.BEST_EFFORT if enabled else PanelResizeBehavior.OFF
        self.app_settings = AppSettingsState(
            theme=self.app_settings.theme,
            prefetch_nearby_slices=self.app_settings.prefetch_nearby_slices,
            panel_resize_behavior=behavior,
            fft_backend=self.app_settings.fft_backend,
            fft_workers=self.app_settings.fft_workers,
            render_memory_budget_mb=self.app_settings.render_memory_budget_mb,
        )
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
            "open_in_new",
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
        for controller_name in (
            "evaluation_controller",
            "visible_evaluation_controller",
            "pixel_evaluation_controller",
            "profile_evaluation_controller",
            "roi_evaluation_controller",
            "prefetch_evaluation_controller",
        ):
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
