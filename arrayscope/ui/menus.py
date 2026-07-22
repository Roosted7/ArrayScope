from __future__ import annotations

import dataclasses

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.app.free_threading import FreeThreadingChoice
from arrayscope.app.qt_platform import QtPlatformChoice
from arrayscope.app.settings_state import (
    AppSettingsState,
    FFTBackendChoice,
    FFTWorkersChoice,
    ImageRenderingBackendChoice,
    MemoryProfileChoice,
    MontageQualityPolicyChoice,
    PanelResizeBehavior,
    WgpuPresentMethodChoice,
    settings_from_mapping,
    settings_to_mapping,
)
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.core.memory_budget import format_bytes
from arrayscope.operations import fft_backend
from arrayscope.operations.registry import operation_entries
from arrayscope.ui.diagnostics import DiagnosticsDialog
from arrayscope.ui.icons import set_action_icon, verify_icon_names
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.diagnostics_snapshot import collect_runtime_diagnostics_snapshot


class WindowMenuMixin:
    def _load_app_settings(self):
        return settings_from_mapping(
            {
                "theme": self._settings.value("theme", ThemeChoice.SYSTEM.value),
                "prefetch_nearby_slices": self._settings.value("prefetch_nearby_slices", False),
                "panel_resize_behavior": self._settings.value(
                    "panel_resize_behavior", PanelResizeBehavior.BEST_EFFORT.value
                ),
                "fft_backend": self._settings.value("fft_backend", FFTBackendChoice.AUTO.value),
                "fft_workers": self._settings.value("fft_workers", FFTWorkersChoice.AUTO.value),
                "image_rendering_backend": self._settings.value(
                    "image_rendering_backend", ImageRenderingBackendChoice.AUTO.value
                ),
                "wgpu_present_method": self._settings.value(
                    "wgpu_present_method", WgpuPresentMethodChoice.BITMAP.value
                ),
                "memory_profile": self._settings.value(
                    "memory_profile", MemoryProfileChoice.BALANCED.value
                ),
                "render_memory_budget_mb": self._settings.value("render_memory_budget_mb", 512),
                "montage_quality_policy": self._settings.value(
                    "montage_quality_policy", MontageQualityPolicyChoice.RESIDENT.value
                ),
                "qt_platform": self._settings.value("qt_platform", QtPlatformChoice.AUTO.value),
                "python_free_threading": self._settings.value(
                    "python_free_threading", FreeThreadingChoice.ENABLED.value
                ),
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
        for action in (
            save_recipe_action,
            load_recipe_action,
            save_view_action,
            load_view_action,
            export_derived_action,
        ):
            file_menu.addAction(action)
        recent_menu = QtWidgets.QMenu("Recent Recipes", self)
        recent_menu.setToolTipsVisible(True)
        file_menu.addMenu(recent_menu)
        recent_menu.aboutToShow.connect(
            lambda menu=recent_menu: self.populate_recent_recipes_menu(menu)
        )

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
        compare_action = QtGui.QAction("Compare with…", self)
        set_action_icon(compare_action, "open_in_new")
        compare_action.setToolTip(
            "Open a second window over this array, pre-linked on dimensions, "
            "pan/zoom and window/level, with a linked value cursor."
        )
        compare_action.triggered.connect(lambda _checked=False: self.open_compare_window())
        view_menu.addAction(compare_action)
        difference_action = QtGui.QAction("Open difference (A − B)…", self)
        set_action_icon(difference_action, "open_in_new")
        difference_action.setToolTip(
            "Open a third window over A − B (a derived, region-only source), "
            "linked into the same compare group as A and B."
        )
        difference_action.triggered.connect(lambda _checked=False: self.open_difference_window())
        view_menu.addAction(difference_action)
        view_menu.addSeparator()
        panel_resize_menu = QtWidgets.QMenu("Panel Resize Behavior", self)
        view_menu.addMenu(panel_resize_menu)
        self._panel_resize_actions = {}
        self._panel_resize_action_group = QtGui.QActionGroup(self)
        self._panel_resize_action_group.setExclusive(True)
        for behavior, label in (
            (PanelResizeBehavior.BEST_EFFORT, "Best effort"),
            (PanelResizeBehavior.STRONG_WAYLAND, "Strong Wayland"),
            (PanelResizeBehavior.OFF, "Off"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._panel_resize_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, behavior=behavior: self._set_panel_resize_behavior(behavior)
            )
            panel_resize_menu.addAction(action)
            self._panel_resize_actions[behavior] = action
        self._panel_resize_menu = panel_resize_menu
        self._sync_panel_resize_actions()
        self._setup_qt_platform_menu(view_menu)
        reset_layout_action = QtGui.QAction("Reset layout", self)
        set_action_icon(reset_layout_action, "reset_wrench")
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
        view_menu.addSeparator()
        theme_menu = QtWidgets.QMenu("Theme", self)
        view_menu.addMenu(theme_menu)
        self._theme_actions = {}
        self._theme_action_group = QtGui.QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for choice, label in (
            (ThemeChoice.SYSTEM, "System"),
            (ThemeChoice.DARK, "Dark"),
            (ThemeChoice.LIGHT, "Light"),
            (ThemeChoice.NATIVE, "OS Native"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._theme_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._apply_theme_choice(choice)
            )
            theme_menu.addAction(action)
            self._theme_actions[choice] = action
        self._sync_theme_actions()
        performance_menu = QtWidgets.QMenu("Performance", self)
        self.menuBar().addMenu(performance_menu)
        self._performance_menu = performance_menu
        self._memory_profile_actions = {}
        self._memory_profile_action_group = QtGui.QActionGroup(self)
        self._memory_profile_action_group.setExclusive(True)
        profile_menu = QtWidgets.QMenu("Memory Profile", self)
        performance_menu.addMenu(profile_menu)
        self._memory_profile_menu = profile_menu
        for choice, label in (
            (MemoryProfileChoice.CONSERVATIVE, "Conservative"),
            (MemoryProfileChoice.BALANCED, "Balanced"),
            (MemoryProfileChoice.AGGRESSIVE, "Aggressive"),
            (MemoryProfileChoice.CUSTOM, "Custom"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._memory_profile_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_memory_profile_choice(choice)
            )
            profile_menu.addAction(action)
            self._memory_profile_actions[choice] = action

        # The prefetch handler survived the legacy-path removal but its menu
        # action did not; the setting was unreachable except by editing the
        # config file (2026-07-15 rewiring, together with its scheduler call).
        prefetch_action = QtGui.QAction("Prefetch Nearby Slices", self, checkable=True)
        prefetch_action.setChecked(
            bool(getattr(self.app_settings, "prefetch_nearby_slices", False))
        )
        prefetch_action.toggled.connect(self._set_prefetch_enabled)
        performance_menu.addAction(prefetch_action)
        self._prefetch_nearby_slices_action = prefetch_action

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
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_fft_backend_choice(choice)
            )
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
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_fft_workers_choice(choice)
            )
            workers_menu.addAction(action)
            self._fft_workers_actions[choice] = action

        self._image_rendering_backend_actions = {}
        self._image_rendering_backend_action_group = QtGui.QActionGroup(self)
        self._image_rendering_backend_action_group.setExclusive(True)
        image_backend_menu = QtWidgets.QMenu("Image Rendering Backend", self)
        performance_menu.addMenu(image_backend_menu)
        self._image_rendering_backend_menu = image_backend_menu
        for choice, label in (
            (ImageRenderingBackendChoice.AUTO, "Auto (hardware GL picks VisPy)"),
            (ImageRenderingBackendChoice.PYQTGRAPH, "PyQtGraph stable"),
            (ImageRenderingBackendChoice.VISPY, "VisPy experimental"),
            # Explicit pin only — AUTO never resolves to wgpu (queue row 3d:
            # promotion is an evidence decision, field hours count toward it).
            (ImageRenderingBackendChoice.WGPU, "wgpu experimental (GPU compute)"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            self._image_rendering_backend_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_image_rendering_backend_choice(
                    choice
                )
            )
            image_backend_menu.addAction(action)
            self._image_rendering_backend_actions[choice] = action

        self._wgpu_present_method_actions = {}
        self._wgpu_present_method_action_group = QtGui.QActionGroup(self)
        self._wgpu_present_method_action_group.setExclusive(True)
        wgpu_present_menu = QtWidgets.QMenu("wgpu Presentation", self)
        wgpu_present_menu.setToolTipsVisible(True)
        performance_menu.addMenu(wgpu_present_menu)
        self._wgpu_present_method_menu = wgpu_present_menu
        for choice, label, tooltip in (
            (
                WgpuPresentMethodChoice.AUTO,
                "Auto (screen on native Wayland)",
                "Present through the compositor swapchain wherever the measured "
                "native-Wayland path exists; bitmap everywhere else.",
            ),
            (
                WgpuPresentMethodChoice.BITMAP,
                "Bitmap (readback compositing)",
                "Render offscreen and composite through Qt; works everywhere, "
                "pays a per-frame GPU→CPU readback.",
            ),
            (
                WgpuPresentMethodChoice.SCREEN,
                "Screen (native swapchain pin)",
                "Always request the native-Wayland swapchain; falls back to "
                "bitmap with a status message where it cannot exist.",
            ),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setToolTip(tooltip)
            self._wgpu_present_method_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_wgpu_present_method_choice(choice)
            )
            wgpu_present_menu.addAction(action)
            self._wgpu_present_method_actions[choice] = action

        self._montage_quality_actions = {}
        self._montage_quality_action_group = QtGui.QActionGroup(self)
        self._montage_quality_action_group.setExclusive(True)
        montage_quality_menu = QtWidgets.QMenu("Montage LOD", self)
        montage_quality_menu.setToolTipsVisible(True)
        performance_menu.addMenu(montage_quality_menu)
        self._montage_quality_menu = montage_quality_menu
        for choice, label in (
            (MontageQualityPolicyChoice.RESIDENT, "Resident (multi-resolution)"),
            (MontageQualityPolicyChoice.NATIVE_ONLY, "Native only"),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setToolTip(
                "Resident presents the closest materialized pyramid level for zoomed-out montages (VisPy tiled scenes); "
                "Native only always presents full-resolution textures."
            )
            self._montage_quality_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_montage_quality_policy_choice(choice)
            )
            montage_quality_menu.addAction(action)
            self._montage_quality_actions[choice] = action

        self._render_budget_actions = {}
        self._render_budget_action_group = QtGui.QActionGroup(self)
        self._render_budget_action_group.setExclusive(True)
        budget_menu = QtWidgets.QMenu("Render Memory Budget", self)
        budget_menu.setToolTipsVisible(True)
        performance_menu.addMenu(budget_menu)
        self._render_budget_menu = budget_menu
        for mb in (128, 256, 512, 1024, 2048, 4096, 8192):
            action = QtGui.QAction(f"{mb} MiB", self, checkable=True)
            action.setToolTip("Per-render hard cap for visible images and montage tile allocation.")
            self._render_budget_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, mb=mb: self._set_render_memory_budget_mb(mb)
            )
            budget_menu.addAction(action)
            self._render_budget_actions[mb] = action
        self._setup_free_threading_menu(performance_menu)
        performance_menu.addSeparator()
        less_memory_action = QtGui.QAction("Use Less Memory", self)
        less_memory_action.setToolTip(
            "Switch to Conservative profile and lower the per-render memory budget one step."
        )
        less_memory_action.triggered.connect(self._use_less_memory)
        performance_menu.addAction(less_memory_action)
        more_memory_action = QtGui.QAction("Use More Memory", self)
        more_memory_action.setToolTip(
            "Switch to Aggressive profile and raise the per-render memory budget one step."
        )
        more_memory_action.triggered.connect(self._use_more_memory)
        performance_menu.addAction(more_memory_action)
        decrease_budget_action = QtGui.QAction("Decrease Render Budget", self)
        decrease_budget_action.setToolTip("Lower the per-render memory budget one preset step.")
        decrease_budget_action.triggered.connect(
            lambda checked=False: self._adjust_render_memory_budget(-1)
        )
        performance_menu.addAction(decrease_budget_action)
        increase_budget_action = QtGui.QAction("Increase Render Budget", self)
        increase_budget_action.setToolTip("Raise the per-render memory budget one preset step.")
        increase_budget_action.triggered.connect(
            lambda checked=False: self._adjust_render_memory_budget(1)
        )
        performance_menu.addAction(increase_budget_action)
        self._less_memory_action = less_memory_action
        self._more_memory_action = more_memory_action
        self._decrease_render_budget_action = decrease_budget_action
        self._increase_render_budget_action = increase_budget_action
        self._sync_performance_actions()

        developer_menu = QtWidgets.QMenu("Developer", self)
        self.menuBar().addMenu(developer_menu)
        diagnostics_action = QtGui.QAction("Diagnostics", self)
        set_action_icon(diagnostics_action, "monitor_heart")
        diagnostics_action.triggered.connect(self.open_diagnostics_dialog)
        developer_menu.addAction(diagnostics_action)
        tile_truth_action = QtGui.QAction("Tile truth overlay", self, checkable=True)
        tile_truth_action.setToolTip(
            "Overlay lifecycle target and backend-acknowledged tile identities on the image."
        )
        tile_truth_action.toggled.connect(self._set_tile_truth_overlay_enabled)
        developer_menu.addAction(tile_truth_action)
        verify_icons_action = QtGui.QAction("Verify icons", self)
        set_action_icon(verify_icons_action, "warning")
        verify_icons_action.triggered.connect(self.verify_icons)
        developer_menu.addAction(verify_icons_action)
        self._developer_menu = developer_menu
        self._diagnostics_action = diagnostics_action
        self._tile_truth_overlay_action = tile_truth_action

    def _set_tile_truth_overlay_enabled(self, enabled: bool) -> None:
        setter = getattr(getattr(self, "renderer", None), "set_tile_truth_overlay_enabled", None)
        if callable(setter):
            setter(bool(enabled))

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
        for choice, action in getattr(self, "_image_rendering_backend_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.image_rendering_backend == choice)
            action.blockSignals(False)
        for choice, action in getattr(self, "_wgpu_present_method_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.wgpu_present_method == choice)
            action.blockSignals(False)
        if hasattr(self, "_wgpu_present_method_menu"):
            # Presentation is a wgpu-backend concern; the submenu greys out
            # (choice preserved) while another backend is selected.
            self._wgpu_present_method_menu.setEnabled(
                self.app_settings.image_rendering_backend == ImageRenderingBackendChoice.WGPU
            )
        for choice, action in getattr(self, "_montage_quality_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.montage_quality_policy == choice)
            action.blockSignals(False)
        for choice, action in self._memory_profile_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.memory_profile == choice)
            action.blockSignals(False)
        for mb, action in self._render_budget_actions.items():
            action.blockSignals(True)
            action.setChecked(int(self.app_settings.render_memory_budget_mb) == int(mb))
            action.blockSignals(False)
        budgets = sorted(int(mb) for mb in self._render_budget_actions)
        current_budget = int(self.app_settings.render_memory_budget_mb)
        if hasattr(self, "_decrease_render_budget_action"):
            self._decrease_render_budget_action.setEnabled(current_budget > budgets[0])
            self._increase_render_budget_action.setEnabled(current_budget < budgets[-1])

    def _apply_performance_settings(self, persist=True):
        current = getattr(self, "app_settings", AppSettingsState())
        fft_backend.set_fft_runtime_options(
            backend=current.fft_backend.value, workers=current.fft_workers.value
        )
        resolved = fft_backend.resolve_fft_backend(current.fft_backend.value)
        if current.fft_backend == FFTBackendChoice.PYFFTW and resolved.name != "pyfftw":
            show_status_message(self, "pyFFTW is not installed; using SciPy FFT")
        if persist:
            self._save_app_settings()
        if hasattr(self, "_refresh_memory_policy"):
            policy = self._refresh_memory_policy(active_render=False)
            if hasattr(self, "_apply_compute_policy"):
                self._apply_compute_policy()
            if persist:
                show_status_message(
                    self,
                    (
                        f"Memory profile: {policy.profile.value.title()}, "
                        f"visible budget {format_bytes(policy.visible_render_budget_bytes)}, "
                        f"display cache {format_bytes(policy.display_cache_budget_bytes)}"
                    ),
                    timeout=3000,
                )
        self._sync_performance_actions()

    def _set_fft_backend_choice(self, choice):
        self.app_settings = self._updated_app_settings(fft_backend=choice)
        self._apply_performance_settings(persist=True)

    def _set_fft_workers_choice(self, choice):
        self.app_settings = self._updated_app_settings(fft_workers=choice)
        self._apply_performance_settings(persist=True)

    def _set_image_rendering_backend_choice(self, choice):
        self.app_settings = self._updated_app_settings(image_rendering_backend=choice)
        self._apply_performance_settings(persist=True)
        show_status_message(self, "Image rendering backend changes apply to newly opened windows.")

    def _set_wgpu_present_method_choice(self, choice):
        self.app_settings = self._updated_app_settings(wgpu_present_method=choice)
        self._apply_performance_settings(persist=True)
        show_status_message(self, "wgpu presentation changes apply to newly opened windows.")

    def _set_montage_quality_policy_choice(self, choice):
        if self.app_settings.montage_quality_policy == choice:
            self._sync_performance_actions()
            return
        self.app_settings = self._updated_app_settings(montage_quality_policy=choice)
        self._save_app_settings()
        self._sync_performance_actions()
        self._rerender_montage_for_lod_policy_change()

    def _rerender_montage_for_lod_policy_change(self):
        """Apply the LOD policy on the next montage render without a restart.

        The policy is captured per ``MontageRenderSession``, so a settings
        change must replace the active session.  Scheduling a full montage
        render does that; tile evaluation resolves from the display caches,
        making this a presentation-level refresh rather than a recompute.
        """

        renderer = getattr(self, "renderer", None)
        update = getattr(renderer, "update_image_view", None)
        if callable(update):
            update()

    def _set_memory_profile_choice(self, choice):
        self.app_settings = self._updated_app_settings(memory_profile=choice)
        self._apply_performance_settings(persist=True)

    def _set_render_memory_budget_mb(self, mb):
        self.app_settings = self._updated_app_settings(render_memory_budget_mb=int(mb))
        self._apply_performance_settings(persist=True)

    def _adjust_render_memory_budget(self, direction: int):
        budgets = sorted(int(mb) for mb in self._render_budget_actions)
        current = int(getattr(self.app_settings, "render_memory_budget_mb", 512))
        if int(direction) < 0:
            candidates = [mb for mb in budgets if mb < current]
            target = candidates[-1] if candidates else budgets[0]
        else:
            candidates = [mb for mb in budgets if mb > current]
            target = candidates[0] if candidates else budgets[-1]
        self._set_render_memory_budget_mb(target)

    def _use_less_memory(self):
        self.app_settings = self._updated_app_settings(
            memory_profile=MemoryProfileChoice.CONSERVATIVE
        )
        self._adjust_render_memory_budget(-1)

    def _use_more_memory(self):
        self.app_settings = self._updated_app_settings(
            memory_profile=MemoryProfileChoice.AGGRESSIVE
        )
        self._adjust_render_memory_budget(1)

    def _updated_app_settings(self, **changes):
        current = getattr(self, "app_settings", AppSettingsState())
        return dataclasses.replace(current, **changes)

    def _setup_qt_platform_menu(self, view_menu):
        """Display-server choice (only offered on Linux Wayland sessions).

        The platform is fixed at QApplication creation, so changes apply on
        the next launch (arrayscope.app.qt_platform applies the policy and,
        in "auto", supervises the wayland->xcb crash fallback).
        """
        from arrayscope.app.qt_platform import platform_choice_applies

        if not platform_choice_applies():
            return
        platform_menu = QtWidgets.QMenu("Display Server", self)
        platform_menu.setToolTipsVisible(True)
        view_menu.addMenu(platform_menu)
        self._qt_platform_menu = platform_menu
        active = QtWidgets.QApplication.instance().platformName()
        active_action = QtGui.QAction(f"Active now: {active}", self)
        active_action.setEnabled(False)
        platform_menu.addAction(active_action)
        platform_menu.addSeparator()
        self._qt_platform_actions = {}
        self._qt_platform_action_group = QtGui.QActionGroup(self)
        self._qt_platform_action_group.setExclusive(True)
        for choice, label, tip in (
            (
                QtPlatformChoice.AUTO,
                "Auto (Wayland, X11 on early crash)",
                "Start on native Wayland; if the app crashes shortly after "
                "launch, relaunch once on X11/XWayland.",
            ),
            (
                QtPlatformChoice.WAYLAND,
                "Force Wayland",
                "Always use the native Wayland platform.",
            ),
            (
                QtPlatformChoice.XCB,
                "Force X11 (XWayland)",
                "Always use the X11 platform through XWayland.",
            ),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setToolTip(tip)
            self._qt_platform_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_qt_platform_choice(choice)
            )
            platform_menu.addAction(action)
            self._qt_platform_actions[choice] = action
        self._sync_qt_platform_actions()

    def _sync_qt_platform_actions(self):
        for choice, action in getattr(self, "_qt_platform_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.qt_platform == choice)
            action.blockSignals(False)

    def _set_qt_platform_choice(self, choice):
        if self.app_settings.qt_platform == choice:
            self._sync_qt_platform_actions()
            return
        self.app_settings = self._updated_app_settings(qt_platform=choice)
        self._save_app_settings()
        self._sync_qt_platform_actions()
        show_status_message(self, "Display server changes take effect after restarting ArrayScope.")

    def _setup_free_threading_menu(self, performance_menu):
        """Performance > Python Free-Threading (free-threaded 3.14t builds).

        The GIL state is fixed at interpreter start, so changes apply on the
        next launch (arrayscope.app.free_threading applies the policy and,
        while enabled, supervises the crash-triggered auto-disable).
        """
        from arrayscope.app import free_threading

        menu = QtWidgets.QMenu("Python Free-Threading", self)
        menu.setToolTipsVisible(True)
        performance_menu.addMenu(menu)
        self._free_threading_menu = menu
        if not free_threading.interpreter_is_free_threaded():
            info = QtGui.QAction("Requires a free-threaded Python build (e.g. 3.14t)", self)
            info.setEnabled(False)
            menu.addAction(info)
            return
        state = (
            "GIL enabled"
            if free_threading.gil_currently_enabled()
            else "GIL disabled (free threading)"
        )
        active_action = QtGui.QAction(f"Active now: {state}", self)
        active_action.setEnabled(False)
        menu.addAction(active_action)
        menu.addSeparator()
        self._free_threading_actions = {}
        self._free_threading_action_group = QtGui.QActionGroup(self)
        self._free_threading_action_group.setExclusive(True)
        for choice, label, tip, selectable in (
            (
                FreeThreadingChoice.ENABLED,
                "Enabled (GIL off)",
                "Run without the GIL; if the app crashes shortly after "
                "launch, free threading is auto-disabled and the launch "
                "retried with the GIL enabled.",
                True,
            ),
            (
                FreeThreadingChoice.FORCE_DISABLED,
                "Force-disabled (GIL on)",
                "Always relaunch with the GIL enabled (PYTHON_GIL=1).",
                True,
            ),
            (
                FreeThreadingChoice.AUTO_DISABLED,
                "Auto-disabled (crashed shortly after launch)",
                "Set by the crash supervisor after an early abnormal exit "
                "of a free-threaded session; select Enabled to try again.",
                False,
            ),
        ):
            action = QtGui.QAction(label, self, checkable=True)
            action.setToolTip(tip)
            action.setEnabled(selectable)
            self._free_threading_action_group.addAction(action)
            action.triggered.connect(
                lambda checked=False, choice=choice: self._set_free_threading_choice(choice)
            )
            menu.addAction(action)
            self._free_threading_actions[choice] = action
        self._sync_free_threading_actions()

    def _sync_free_threading_actions(self):
        for choice, action in getattr(self, "_free_threading_actions", {}).items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.python_free_threading == choice)
            action.blockSignals(False)

    def _set_free_threading_choice(self, choice):
        if self.app_settings.python_free_threading == choice:
            self._sync_free_threading_actions()
            return
        self.app_settings = self._updated_app_settings(python_free_threading=choice)
        self._save_app_settings()
        self._sync_free_threading_actions()
        show_status_message(self, "Free-threading changes take effect after restarting ArrayScope.")

    def _retheme_presentation_surfaces(self):
        """Restyle pyqtgraph surfaces in every open ArrayScope window.

        The palette/stylesheet switch is application-wide, but pyqtgraph
        widgets keep their construction-time colors until told otherwise.
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        from arrayscope.ui.icons import refresh_icon_tints

        refresh_icon_tints()
        windows = [w for w in app.topLevelWidgets() if isinstance(w, type(self))]
        if self not in windows:
            windows.append(self)
        for window in windows:
            view = getattr(window, "img_view", None)
            apply_view = getattr(view, "applyThemeTokens", None)
            if callable(apply_view):
                apply_view()
            profile_dock = getattr(window, "profile_dock", None)
            line_plot = getattr(profile_dock, "line_plot", None)
            apply_line = getattr(line_plot, "apply_theme", None)
            if callable(apply_line):
                apply_line()
            inspection_dock = getattr(window, "inspection_dock", None)
            apply_inspection = getattr(inspection_dock, "apply_theme", None)
            if callable(apply_inspection):
                apply_inspection()
            toolbar = getattr(window, "display_toolbar", None)
            refresh_toolbar = getattr(toolbar, "refresh_icons", None)
            if callable(refresh_toolbar):
                refresh_toolbar()

    def _sync_theme_actions(self):
        if not hasattr(self, "_theme_actions"):
            return
        for choice, action in self._theme_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.theme == choice)
            action.blockSignals(False)

    def _apply_theme_choice(self, choice, persist=True):
        result = apply_theme_to_qapplication(QtWidgets.QApplication.instance(), choice)
        self._retheme_presentation_surfaces()
        if result.warning:
            show_status_message(self, f"Theme warning: {result.warning}")
            if persist:
                QtWidgets.QMessageBox.warning(self, "Theme Warning", result.warning)
        theme_to_store = result.requested if result.applied == result.requested else result.applied
        self.applied_theme = result.applied
        self.theme_backend = result.backend
        self.app_settings = self._updated_app_settings(theme=theme_to_store)
        if persist:
            self._save_app_settings()
        self._sync_theme_actions()

    def _set_prefetch_enabled(self, enabled):
        self.app_settings = self._updated_app_settings(prefetch_nearby_slices=bool(enabled))
        self._save_app_settings()

    def _set_preserve_canvas_enabled(self, enabled):
        behavior = PanelResizeBehavior.BEST_EFFORT if enabled else PanelResizeBehavior.OFF
        self._set_panel_resize_behavior(behavior)

    def _set_panel_resize_behavior(self, behavior):
        self.app_settings = self._updated_app_settings(panel_resize_behavior=behavior)
        self._save_app_settings()
        self._sync_panel_resize_actions()
        show_status_message(
            self, f"Panel resize: {_panel_resize_behavior_label(behavior)}", timeout=2500
        )

    def _sync_panel_resize_actions(self):
        if not hasattr(self, "_panel_resize_actions"):
            return
        for behavior, action in self._panel_resize_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.panel_resize_behavior == behavior)
            action.blockSignals(False)

    def collect_runtime_diagnostics(self):
        return collect_runtime_diagnostics_snapshot(self)

    def open_diagnostics_dialog(self):
        dialog = getattr(self, "_diagnostics_dialog", None)
        if dialog is None:
            dialog = DiagnosticsDialog(self, self.collect_runtime_diagnostics, interval_ms=500)
            dialog.destroyed.connect(lambda _obj=None: setattr(self, "_diagnostics_dialog", None))
            self._diagnostics_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        dialog.refresh()

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

    def _restore_window_settings(
        self,
        *,
        initial_viewport=None,
        defer_progressive_docks: bool = False,
    ):
        self.layout_manager.restore_window_settings(
            initial_viewport=initial_viewport,
            defer_progressive_docks=defer_progressive_docks,
        )

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
        save_file_session = getattr(self, "_save_file_view_session_on_close", None)
        if callable(save_file_session):
            save_file_session()
        self.layout_manager.save_window_settings()
        self._save_app_settings()
        self._closing = True
        watcher = getattr(self, "_file_watcher", None)
        if watcher is not None:
            watcher.deleteLater()
            self._file_watcher = None
        coordinator = getattr(self, "render_coordinator", None)
        if coordinator is not None:
            coordinator.cancel_pending()
        for controller_name in (
            "evaluation_controller",
            "visible_evaluation_controller",
            "histogram_evaluation_controller",
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
        # Timer category: UI cosmetic. Qt event-turn barrier. Quitting is checked after close handlers and
        # deleteLater work finish, avoiding shutdown reentrancy.
        Qt.QtCore.QTimer.singleShot(0, self, self._quit_if_last_arrayscope_window)

    def _quit_if_last_arrayscope_window(self):
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        visible_arrayscope = [
            widget
            for widget in app.topLevelWidgets()
            if widget is not self
            and type(widget).__name__ == type(self).__name__
            and widget.isVisible()
        ]
        if not visible_arrayscope:
            app.quit()


def _panel_resize_behavior_label(behavior):
    labels = {
        PanelResizeBehavior.BEST_EFFORT: "Best effort",
        PanelResizeBehavior.STRONG_WAYLAND: "Strong Wayland",
        PanelResizeBehavior.OFF: "Off",
    }
    return labels.get(behavior, str(getattr(behavior, "value", behavior)))
