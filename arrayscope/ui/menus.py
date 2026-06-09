from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.app.settings_state import (
    AppSettingsState,
    FFTBackendChoice,
    FFTWorkersChoice,
    MemoryProfileChoice,
    PanelResizeBehavior,
    settings_from_mapping,
    settings_to_mapping,
)
from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.runtime_diagnostics import (
    CanvasPreserveRuntimeDiagnostics,
    MontageTimingDiagnostics,
    MontageRuntimeDiagnostics,
    RenderCoalescerDiagnostics,
    RenderRuntimeDiagnostics,
    RenderTimingDiagnostics,
    WindowRuntimeDiagnostics,
)
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.regions import region_text
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.operations import fft_backend
from arrayscope.operations.registry import operation_entries
from arrayscope.ui.diagnostics import DiagnosticsDialog
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
                "memory_profile": self._settings.value("memory_profile", MemoryProfileChoice.BALANCED.value),
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
            action.triggered.connect(lambda checked=False, behavior=behavior: self._set_panel_resize_behavior(behavior))
            panel_resize_menu.addAction(action)
            self._panel_resize_actions[behavior] = action
        self._panel_resize_menu = panel_resize_menu
        self._sync_panel_resize_actions()
        reset_layout_action = QtGui.QAction("Reset layout", self)
        set_action_icon(reset_layout_action, "reset_wrench")
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)
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
            action.triggered.connect(lambda checked=False, choice=choice: self._set_memory_profile_choice(choice))
            profile_menu.addAction(action)
            self._memory_profile_actions[choice] = action

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
        budget_menu.setToolTipsVisible(True)
        performance_menu.addMenu(budget_menu)
        self._render_budget_menu = budget_menu
        for mb in (128, 256, 512, 1024, 2048, 4096, 8192):
            action = QtGui.QAction(f"{mb} MiB", self, checkable=True)
            action.setToolTip("Per-render hard cap for visible images and montage canvas/tile allocation.")
            self._render_budget_action_group.addAction(action)
            action.triggered.connect(lambda checked=False, mb=mb: self._set_render_memory_budget_mb(mb))
            budget_menu.addAction(action)
            self._render_budget_actions[mb] = action
        performance_menu.addSeparator()
        less_memory_action = QtGui.QAction("Use Less Memory", self)
        less_memory_action.setToolTip("Switch to Conservative profile and lower the per-render memory budget one step.")
        less_memory_action.triggered.connect(self._use_less_memory)
        performance_menu.addAction(less_memory_action)
        more_memory_action = QtGui.QAction("Use More Memory", self)
        more_memory_action.setToolTip("Switch to Aggressive profile and raise the per-render memory budget one step.")
        more_memory_action.triggered.connect(self._use_more_memory)
        performance_menu.addAction(more_memory_action)
        decrease_budget_action = QtGui.QAction("Decrease Render Budget", self)
        decrease_budget_action.setToolTip("Lower the per-render memory budget one preset step.")
        decrease_budget_action.triggered.connect(lambda checked=False: self._adjust_render_memory_budget(-1))
        performance_menu.addAction(decrease_budget_action)
        increase_budget_action = QtGui.QAction("Increase Render Budget", self)
        increase_budget_action.setToolTip("Raise the per-render memory budget one preset step.")
        increase_budget_action.triggered.connect(lambda checked=False: self._adjust_render_memory_budget(1))
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
        verify_icons_action = QtGui.QAction("Verify icons", self)
        set_action_icon(verify_icons_action, "warning")
        verify_icons_action.triggered.connect(self.verify_icons)
        developer_menu.addAction(verify_icons_action)
        self._developer_menu = developer_menu
        self._diagnostics_action = diagnostics_action

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
        fft_backend.set_fft_runtime_options(backend=current.fft_backend.value, workers=current.fft_workers.value)
        resolved = fft_backend.resolve_fft_backend(current.fft_backend.value)
        if current.fft_backend == FFTBackendChoice.PYFFTW and resolved.name != "pyfftw":
            show_status_message(self, "pyFFTW is not installed; using SciPy FFT")
        if persist:
            self._save_app_settings()
        if hasattr(self, "_refresh_memory_policy"):
            policy = self._refresh_memory_policy(active_render=False)
            if persist:
                show_status_message(
                    self,
                    (
                        f"Memory profile: {policy.profile.value.title()}, "
                        f"visible budget {format_bytes(policy.visible_render_budget_bytes)}, "
                        f"tile cache {format_bytes(policy.tile_cache_budget_bytes)}"
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
        self.app_settings = self._updated_app_settings(memory_profile=MemoryProfileChoice.CONSERVATIVE)
        self._adjust_render_memory_budget(-1)

    def _use_more_memory(self):
        self.app_settings = self._updated_app_settings(memory_profile=MemoryProfileChoice.AGGRESSIVE)
        self._adjust_render_memory_budget(1)

    def _updated_app_settings(self, **changes):
        current = getattr(self, "app_settings", AppSettingsState())
        values = {
            "theme": current.theme,
            "prefetch_nearby_slices": current.prefetch_nearby_slices,
            "panel_resize_behavior": current.panel_resize_behavior,
            "fft_backend": current.fft_backend,
            "fft_workers": current.fft_workers,
            "memory_profile": current.memory_profile,
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
            memory_profile=current.memory_profile,
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
            memory_profile=self.app_settings.memory_profile,
            render_memory_budget_mb=self.app_settings.render_memory_budget_mb,
        )
        self._save_app_settings()

    def _set_preserve_canvas_enabled(self, enabled):
        behavior = PanelResizeBehavior.BEST_EFFORT if enabled else PanelResizeBehavior.OFF
        self._set_panel_resize_behavior(behavior)

    def _set_panel_resize_behavior(self, behavior):
        self.app_settings = AppSettingsState(
            theme=self.app_settings.theme,
            prefetch_nearby_slices=self.app_settings.prefetch_nearby_slices,
            panel_resize_behavior=behavior,
            fft_backend=self.app_settings.fft_backend,
            fft_workers=self.app_settings.fft_workers,
            memory_profile=self.app_settings.memory_profile,
            render_memory_budget_mb=self.app_settings.render_memory_budget_mb,
        )
        self._save_app_settings()
        self._sync_panel_resize_actions()
        show_status_message(self, f"Panel resize: {_panel_resize_behavior_label(behavior)}", timeout=2500)

    def _sync_panel_resize_actions(self):
        if not hasattr(self, "_panel_resize_actions"):
            return
        for behavior, action in self._panel_resize_actions.items():
            action.blockSignals(True)
            action.setChecked(self.app_settings.panel_resize_behavior == behavior)
            action.blockSignals(False)

    def collect_runtime_diagnostics(self):
        policy = self._refresh_memory_policy(active_render=getattr(self, "visible_evaluation_controller", None).is_busy() if hasattr(self, "visible_evaluation_controller") else False)
        schedulers = []
        for name in (
            "visible_evaluation_controller",
            "pixel_evaluation_controller",
            "profile_evaluation_controller",
            "roi_evaluation_controller",
            "prefetch_evaluation_controller",
        ):
            controller = getattr(self, name, None)
            if controller is not None:
                schedulers.append(controller.diagnostics())
        session = getattr(self, "_montage_session", None)
        canvas = getattr(self, "_current_montage_canvas", None)
        canvas_bytes = None
        if canvas is not None:
            canvas_bytes = int(getattr(canvas.data, "nbytes", 0))
            histogram = getattr(canvas, "histogram_data", None)
            if histogram is not None:
                canvas_bytes += int(getattr(histogram, "nbytes", 0))
        montage = MontageRuntimeDiagnostics(
            active=session is not None,
            session_id=None if session is None else int(session.session_id),
            canvas_rect=None if canvas is None else (int(canvas.origin_x), int(canvas.origin_y), int(canvas.origin_x + canvas.display_shape[1]), int(canvas.origin_y + canvas.display_shape[0])),
            canvas_shape=None if canvas is None else tuple(int(size) for size in canvas.display_shape),
            canvas_bytes=canvas_bytes,
            loaded_tiles=0 if session is None else len(session.rendered_tiles),
            loading_tiles=0 if session is None else len(session.loading_tiles),
            pending_tiles=0 if session is None else len(session.pending_tiles),
            skipped_tiles=0 if session is None else len(session.skipped_tiles),
            visible_tiles=0 if session is None else len(session.visible_tiles),
            show_loading_overlays=False if session is None else bool(session.show_loading_overlays),
        )
        decision = getattr(self, "_last_render_decision", None)
        context = getattr(self, "_last_render_context", None)
        render = RenderRuntimeDiagnostics(
            last_decision_kind="" if decision is None else str(getattr(decision.kind, "value", decision.kind)),
            last_decision_reason="" if decision is None else str(getattr(decision, "reason", "")),
            last_context_summary="" if context is None else str(context),
            last_request_key=str(getattr(self, "_last_render_request_key", "") or ""),
            last_error=str(getattr(self, "_last_render_error", "") or ""),
            estimated_display_bytes=None if context is None else int(getattr(context, "estimated_display_bytes", 0)),
            render_budget_bytes=None if context is None else int(getattr(context, "render_budget_bytes", 0)),
        )
        coalescer = getattr(self, "render_coordinator", None)
        backend_choice, workers_choice = fft_backend.get_fft_runtime_options()
        resolved = fft_backend.resolve_fft_backend(backend_choice.value)
        cost = estimate_pipeline_cost(
            self.base_data.shape,
            getattr(self.base_data, "dtype", None),
            tuple(self.document.enabled_operations),
        )
        region_plan = self.operation_evaluator.planner_diagnostics()
        capability_stage_count = None if region_plan is None else len(tuple(getattr(region_plan, "stages", ())))
        candidates = () if region_plan is None else tuple(getattr(region_plan, "cache_candidates", ()))
        transitions = () if region_plan is None else tuple(getattr(region_plan, "transitions", ()))
        expanded_axes = tuple(
            sorted({int(axis) for transition in transitions for axis in getattr(transition, "expanded_axes", ())})
        )
        return WindowRuntimeDiagnostics(
            memory_policy=policy,
            image_cache=self.operation_evaluator.image_cache_diagnostics(),
            tile_cache=self.operation_evaluator.tile_cache_diagnostics(),
            profile_cache=self.operation_evaluator.profile_cache_diagnostics(),
            stage_cache=self.operation_evaluator.stage_cache_diagnostics(),
            schedulers=tuple(schedulers),
            render=render,
            montage=montage,
            canvas_preserve=(
                self.layout_manager.canvas_preserver.diagnostics()
                if hasattr(getattr(self, "layout_manager", None), "canvas_preserver")
                else CanvasPreserveRuntimeDiagnostics()
            ),
            render_timing=RenderTimingDiagnostics(
                last_render_sync_ms=getattr(self, "_last_render_sync_ms", None),
                last_control_sync_ms=getattr(self, "_last_control_sync_ms", None),
                last_planning_ms=getattr(self, "_last_planning_ms", None),
                last_worker_queue_wait_ms=getattr(self, "_last_worker_queue_wait_ms", None),
                last_evaluation_ms=getattr(self, "_last_render_completed_ms", None),
                last_display_commit_ms=getattr(self, "_last_display_commit_ms", None),
                last_set_image_ms=getattr(self, "_last_set_image_ms", None),
                last_levels_histogram_ms=getattr(self, "_last_levels_histogram_ms", None),
                last_operation_dock_ms=getattr(self, "_last_operation_dock_ms", None),
                last_inspection_refresh_ms=getattr(self, "_last_inspection_refresh_ms", None),
            ),
            montage_timing=MontageTimingDiagnostics(
                last_tile_eval_ms=getattr(self, "_last_montage_tile_eval_ms", None),
                last_canvas_compose_ms=getattr(self, "_last_montage_canvas_compose_ms", None),
                last_canvas_commit_ms=getattr(self, "_last_montage_canvas_commit_ms", None),
                last_overlay_update_ms=getattr(self, "_last_montage_overlay_update_ms", None),
                cached_tiles_last_session=int(getattr(self, "_montage_cached_tiles_last_session", 0) or 0),
                missing_tiles_last_session=int(getattr(self, "_montage_missing_tiles_last_session", 0) or 0),
            ),
            render_coalescer=RenderCoalescerDiagnostics(
                pending=False if coalescer is None else bool(coalescer.has_pending_render),
                interactive_active=False if coalescer is None else bool(coalescer.interactive_active),
                requested=0 if coalescer is None else int(coalescer.requested),
                flushed=0 if coalescer is None else int(coalescer.flushed),
                coalesced=0 if coalescer is None else int(coalescer.coalesced),
                deferred_side_panel_refreshes=0 if coalescer is None else int(coalescer.deferred_side_panel_refreshes),
            ),
            fft_backend_choice=backend_choice.value,
            fft_backend_resolved=resolved.name,
            fft_workers_choice=workers_choice.value,
            fft_workers_resolved=int(fft_backend.runtime_fft_workers()),
            operation_count=len(tuple(self.document.enabled_operations)),
            derived_shape=tuple(int(size) for size in self.document.current_shape),
            derived_dtype=str(self.data.dtype),
            pipeline_peak_bytes=cost.estimated_peak_bytes,
            pipeline_warnings=tuple(cost.warnings),
            optimized_operation_count=(
                getattr(region_plan, "optimized_operation_count", None)
                if region_plan is not None
                else cost.optimized_operation_count
            ),
            operation_optimization_summaries=(
                tuple(getattr(region_plan, "optimization_steps", ()))
                if region_plan is not None
                else tuple(cost.optimization_steps)
            ),
            capability_stage_count=capability_stage_count,
            stage_cache_candidate_count=None if region_plan is None else len(candidates),
            stage_cache_candidate_summaries=tuple(_stage_cache_candidate_summary(candidate) for candidate in candidates),
            operation_final_region="" if region_plan is None else region_text(region_plan.final_region),
            operation_required_input_region="" if region_plan is None else region_text(region_plan.required_input_region),
            operation_expanded_axes=expanded_axes,
            operation_transition_summaries=tuple(_region_transition_summary(transition) for transition in transitions),
        )

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
        coordinator = getattr(self, "render_coordinator", None)
        if coordinator is not None:
            coordinator.cancel_pending()
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


def _panel_resize_behavior_label(behavior):
    labels = {
        PanelResizeBehavior.BEST_EFFORT: "Best effort",
        PanelResizeBehavior.STRONG_WAYLAND: "Strong Wayland",
        PanelResizeBehavior.OFF: "Off",
    }
    return labels.get(behavior, str(getattr(behavior, "value", behavior)))


def _stage_cache_candidate_summary(candidate):
    region = getattr(candidate, "region", None)
    axes = "n/a" if region is None else ",".join(str(getattr(axis.kind, "value", axis.kind)) for axis in region.axes)
    nbytes = getattr(candidate, "estimated_nbytes", None)
    size = "unknown" if nbytes is None else format_bytes(int(nbytes))
    return (
        f"stage {getattr(candidate, 'stage_index', '?')} "
        f"{getattr(candidate, 'priority', 'n/a')} {size}, axes={axes}, "
        f"retain={'yes' if getattr(candidate, 'retain', True) else 'no'} "
        f"{getattr(candidate, 'retain_reason', '')}, "
        f"{getattr(candidate, 'reason', '')}"
    )


def _region_transition_summary(transition):
    expanded = tuple(getattr(transition, "expanded_axes", ()))
    expanded_text = "n/a" if not expanded else ",".join(str(int(axis)) for axis in expanded)
    return (
        f"stage {getattr(transition, 'stage_index', '?')} {type(getattr(transition, 'operation', object())).__name__} "
        f"output={region_text(transition.output_region)} "
        f"input={region_text(transition.required_input_region)} "
        f"expanded={expanded_text}"
    )
