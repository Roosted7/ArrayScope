import numpy as np
from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets
import platform
from arrayscope.app.errors import handle_ui_exception
from arrayscope.operations.coordinator import OperationCoordinator
from arrayscope.profiles.coordinator import ProfileCoordinator
from arrayscope.core.array_metadata import derived_info_for
from arrayscope.core.compute_policy import ComputeLane, EvaluationContext, compute_policy_from_settings
from arrayscope.core.gui_callback_budget import default_gui_callback_budget_decision
from arrayscope.core.gui_gc import configure_gui_gc_latency
from arrayscope.core.resource_governor import ResourceGovernor, SchedulerBusyState
from arrayscope.core.resource_telemetry import sample_resource_snapshot
from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.core.roi_store import RoiStore
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.kernel import Kernel, Lane, Priority, ThreadWorkerBackend
from arrayscope.kernel.eval_adapter import KernelEvaluationController
from arrayscope.kernel.qt_bridge import QtKernelBridge
from arrayscope.export.workflow import ExportWorkflowMixin
from arrayscope.ui.dimension_controls import DimensionControlMixin
from arrayscope.ui.display_controls import DisplayControlBuildMixin
from arrayscope.ui.menus import WindowMenuMixin
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.domain import Domain
from arrayscope.window.file_view_session import FileViewSessionMixin
from arrayscope.window.file_reload import FileReloadMixin
from arrayscope.window.inspection import InspectionWorkflowMixin
from arrayscope.window.interaction_mode import InteractionMode
from arrayscope.window.operation_actions import OperationActionsMixin
from arrayscope.window.render import RenderOrchestrator
from arrayscope.window.render_coordinator import RenderCoordinator
from arrayscope.window.state_sync import StateSyncMixin


class ArrayScopeWindow(
    WindowMenuMixin,
    DisplayControlBuildMixin,
    StateSyncMixin,
    OperationActionsMixin,
    FileViewSessionMixin,
    InspectionWorkflowMixin,
    DimensionControlMixin,
    ExportWorkflowMixin,
    FileReloadMixin,
    QtWidgets.QMainWindow,
):
    # Styling constants — use pt (point) units so font sizes are DPI-independent
    DIMENSION_LABEL_STYLE = "QLabel { font-size: 9pt; padding: 1px; margin: 2px; }"
    FLIP_ICON_STYLE = "QLabel { font-size: 15pt; padding: 0px; margin: 0px; color: palette(text); }"
    SHIFT_LABEL_STYLE = "QLabel { font-size: 8pt; padding: 1px 2px; margin: 0px; color: palette(mid); }"
    SHIFT_LABEL_ACTIVE_STYLE = "QLabel { font-size: 8pt; padding: 1px 2px; margin: 0px; font-weight: bold; color: darkMagenta; }"
    BUTTON_STYLE = "QPushButton { font-size: 9pt; padding: 2px; margin: 2px; } QPushButton:disabled { color: palette(mid); }"
    SPINBOX_STYLE = "QSpinBox { font-size: 9pt; } QSpinBox:disabled { color: palette(mid); }"
    RADIO_BUTTON_STYLE = "QRadioButton { font-size: 9pt; }"
    GROUPBOX_BASE_STYLE = "QGroupBox { font-size: 9pt; font-weight: bold; border: 1px solid palette(mid); border-radius: 3px; margin-top: 1.4ex; padding-top: 3pt; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }"
    
    @staticmethod
    def _set_emoji_font(widget):
        if platform.system() == 'Darwin':
            font = widget.font()
            font.setFamily('Apple Color Emoji')
            widget.setFont(font)

    def __init__(self, data, complex_dim=None, filepath=None, dataset_path=None, selector_class_name=None, axes=None):
        configure_gui_gc_latency()
        super(ArrayScopeWindow, self).__init__()
        self.renderer = RenderOrchestrator(self)
        self.resize(600,800)
        self._settings = Qt.QtCore.QSettings()
        self.app_settings = self._load_app_settings()
        self._apply_theme_choice(self.app_settings.theme, persist=False)
        self._apply_performance_settings(persist=False)
        self.compute_policy = compute_policy_from_settings(self.app_settings)

        self.operation_coordinator = OperationCoordinator(data, axes=axes)
        self.profile_coordinator = ProfileCoordinator()
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.renderer._refresh_memory_policy(active_render=False)
        self.resource_governor = ResourceGovernor(
            self.compute_policy,
            profile=self.app_settings.memory_profile,
        )
        self.latency_feedback = self.resource_governor.latency_feedback
        self.resource_governor.update_telemetry(
            sample_resource_snapshot(),
            self.renderer._memory_policy(),
        )
        self._init_compare_document(data)
        self.kernel = Kernel(ThreadWorkerBackend(), handler_error_hook=handle_ui_exception)
        self.kernel_bridge = QtKernelBridge(self.kernel, self)
        self.visible_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.visible_workers,
            name="visible",
            lane_default=Lane.VISIBLE_MATERIALIZATION,
            priority_default=Priority.VISIBLE_IMAGE,
            apply_lane_quota=False,
        )
        self.evaluation_controller = self.visible_evaluation_controller
        self.montage_tile_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.montage_tile_workers,
            name="montage_tile",
            lane_default=Lane.VISIBLE_MATERIALIZATION,
            priority_default=Priority.VISIBLE_IMAGE,
            max_callback_dispatch_per_drain=8,
            apply_lane_quota=False,
        )
        self.stage_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.stage_workers,
            name="stage",
            lane_default=Lane.STAGE_MATERIALIZATION,
            priority_default=Priority.VISIBLE_IMAGE,
            apply_lane_quota=False,
        )
        self.histogram_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.histogram_workers,
            name="histogram",
            lane_default=Lane.HISTOGRAM_REFINEMENT,
            priority_default=Priority.HISTOGRAM,
            apply_lane_quota=False,
        )
        self.pixel_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.pixel_workers,
            name="pixel",
            lane_default=Lane.PROFILE_ROI_HOVER,
            priority_default=Priority.HOVER,
            apply_lane_quota=False,
        )
        self.profile_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.profile_workers,
            name="profile",
            lane_default=Lane.PROFILE_ROI_HOVER,
            priority_default=Priority.LIVE_PROFILE,
            apply_lane_quota=False,
        )
        self.roi_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.roi_workers,
            name="roi",
            lane_default=Lane.PROFILE_ROI_HOVER,
            priority_default=Priority.SELECTED_ROI,
            apply_lane_quota=False,
        )
        self.prefetch_evaluation_controller = KernelEvaluationController(
            self.kernel,
            self.kernel_bridge,
            self,
            max_workers=self.compute_policy.prefetch_workers,
            name="prefetch",
            lane_default=Lane.SPECULATIVE_RESIDENCY,
            priority_default=Priority.PREFETCH,
            apply_lane_quota=False,
        )
        self._apply_resource_governor_decisions(refresh_telemetry=False)
        self.render_coordinator = RenderCoordinator(self)
        self._deferred_side_panel_refresh_pending = False
        self.data = derived_info_for(self.document)
        self.singleton = [e == 1 for e in list(self.data.shape)]
        initial_channel = ChannelMode.COMPLEX if np.issubdtype(self.data.dtype, np.complexfloating) else ChannelMode.REAL
        self.view_state = ViewState.from_shape(self.data.shape).with_channel(initial_channel)
        self._channel_user_selected = False
        self.current_colormap = None
        self._colormap_user_selected = False
        self._force_autolevel = False
        self._filepath = filepath
        self._dataset_path = dataset_path
        self._selector_class_name = selector_class_name
        self._operation_dock_user_visible = None
        self._profile_dock_user_visible = None
        self._inspection_dock_user_visible = None
        self._suspend_progressive_dock_sync = True
        self._progressive_preserve_enabled = False
        self._last_operation_axis = None
        self._focused_dimension_axis = None
        self._active_slice_axis = None
        self.statusBar()
        
        # If data is real-valued and has size-2 dimensions, arrayscope can combine them as complex (ISMRMD uses this for real/imag parts)
        if np.iscomplexobj(data):
            self.can_combine_as_complex = [False] * data.ndim
        else:
            self.can_combine_as_complex = [data.shape[i] == 2 for i in range(data.ndim)]
        self.combined_as_complex = [np.iscomplexobj(data) and data.shape[i] == 1 for i in range(data.ndim)]
        
        # Store complex_dim for later use (after widgets are created)
        self._initial_complex_dim = complex_dim
        
        # Line plot mode uses a single selected dimension
        self.line_plot_dimension = self.view_state.line_axis or 0
        self.profile_axes = (self.line_plot_dimension,)
        self.roi_store = RoiStore()
        self.interaction_mode = InteractionMode.CURSOR
                
        self._build_window_ui(data, filepath)
        self._update_array_info_label()
        from arrayscope.sync.controller import WindowSyncController

        self.sync_controller = WindowSyncController(self)
        self._apply_channel_colormap()
        try:
            restored_file_view_session = self._restore_file_view_session_if_available()
        finally:
            self._suspend_progressive_dock_sync = False
        
        if complex_dim is not None: # user requested combining as complex
            if complex_dim < 0 or complex_dim >= data.ndim:
                show_status_message(self, f"complex_dim={complex_dim} is out of range for {data.ndim}D array. Ignoring.")
            elif np.iscomplexobj(data):
                show_status_message(self, f"Data is already complex. Ignoring complex_dim={complex_dim}.")
            elif data.shape[complex_dim] != 2:
                show_status_message(self, f"Dimension {complex_dim} has shape {data.shape[complex_dim]}, not 2. Cannot combine as complex.")
            else:
                self.combineAsComplex(complex_dim) # valid
        
        # Initialize dimension controls based on the authoritative view state.
        initial_viewport = None
        if restored_file_view_session:
            tx = self._viewport_continuity_transaction()
            initial_viewport = None if tx is None else tx.viewport
        self._restore_window_settings(
            initial_viewport=initial_viewport,
            defer_progressive_docks=bool(restored_file_view_session),
        )
        if restored_file_view_session:
            self._apply_file_session_layout_intent()
        if restored_file_view_session:
            self.render(
                reason="file-view-session-restore",
                force_autolevel=self._pending_display_levels_for_render() is None,
                defer_side_panels=True,
            )
            self.show()
            self._progressive_preserve_enabled = True
            self._run_deferred_side_panel_refresh(reason="file-view-session-restore")
        else:
            self.render(reason="initial", force_autolevel=True)
            self.show()
        if restored_file_view_session:
            self._restore_viewport_continuity_shape_after_layout()

            def finish_restored_file_session_viewport():
                apply_restored_viewport = getattr(self, "_apply_viewport_continuity_when_ready", None)
                if callable(apply_restored_viewport):
                    apply_restored_viewport()

            # Timer category: UI cosmetic. Qt event-turn barrier. The callback rechecks restore readiness
            # after the first show/layout pass.
            Qt.QtCore.QTimer.singleShot(0, self, finish_restored_file_session_viewport)
        # Timer category: UI cosmetic. Qt event-turn barrier. Progressive dock preservation starts after
        # startup layout has settled.
        Qt.QtCore.QTimer.singleShot(0, self, lambda: setattr(self, "_progressive_preserve_enabled", True))

        def show_first_run_hints():
            if getattr(self, "_closing", False):
                return
            from arrayscope.ui.hints import maybe_show_first_run_hints

            maybe_show_first_run_hints(self)

        # Timer category: UI cosmetic. One-time discoverability hints appear
        # after the first layout pass and never gate data flow.
        Qt.QtCore.QTimer.singleShot(0, self, show_first_run_hints)

        # Set up file watcher if a filepath was provided (QFileSystemWatcher uses
        # OS-native events: inotify on Linux, FSEvents on macOS, ReadDirectoryChanges on Windows)
        self._file_watcher = None
        if filepath is not None:
            self._file_watcher = Qt.QtCore.QFileSystemWatcher([str(filepath)])
            self._file_watcher.fileChanged.connect(self._on_file_changed)

    def _evaluation_context(self, lane, token=None):
        lane = ComputeLane(lane)
        return EvaluationContext(
            lane=lane,
            cancellation_token=token,
            fft_workers=self.compute_policy.fft_workers_for_lane(lane),
            memory_policy=self.renderer._memory_policy(),
        )

    def _apply_compute_policy(self) -> None:
        self.compute_policy = compute_policy_from_settings(self.app_settings)
        governor = getattr(self, "resource_governor", None)
        if governor is not None:
            governor.update_policy(self.compute_policy, profile=getattr(self.app_settings, "memory_profile", None))
            self._apply_resource_governor_decisions()
            return
        quota_by_lane: dict[Lane, int] = {}
        for lane, controller in self._evaluation_controllers_by_lane().items():
            workers = self.compute_policy.workers_for_lane(lane)
            if controller is not None:
                controller.set_reported_max_workers(workers)
            kernel_lane = self._kernel_lane_for_compute_lane(lane)
            quota_by_lane[kernel_lane] = max(int(workers), int(quota_by_lane.get(kernel_lane, 0)))
        kernel = getattr(self, "kernel", None)
        if kernel is not None:
            for lane, workers in quota_by_lane.items():
                kernel.set_lane_quota(lane, workers)

    def _evaluation_controllers_by_lane(self):
        return {
            ComputeLane.VISIBLE: getattr(self, "visible_evaluation_controller", None),
            ComputeLane.MONTAGE_TILE: getattr(self, "montage_tile_evaluation_controller", None),
            ComputeLane.STAGE: getattr(self, "stage_evaluation_controller", None),
            ComputeLane.HISTOGRAM: getattr(self, "histogram_evaluation_controller", None),
            ComputeLane.PREFETCH: getattr(self, "prefetch_evaluation_controller", None),
            ComputeLane.PROFILE: getattr(self, "profile_evaluation_controller", None),
            ComputeLane.ROI: getattr(self, "roi_evaluation_controller", None),
            ComputeLane.PIXEL: getattr(self, "pixel_evaluation_controller", None),
        }

    def _kernel_lane_for_compute_lane(self, lane: ComputeLane) -> Lane:
        return {
            ComputeLane.VISIBLE: Lane.VISIBLE_MATERIALIZATION,
            ComputeLane.MONTAGE_TILE: Lane.VISIBLE_MATERIALIZATION,
            ComputeLane.STAGE: Lane.STAGE_MATERIALIZATION,
            ComputeLane.HISTOGRAM: Lane.HISTOGRAM_REFINEMENT,
            ComputeLane.PREFETCH: Lane.SPECULATIVE_RESIDENCY,
            ComputeLane.PROFILE: Lane.PROFILE_ROI_HOVER,
            ComputeLane.ROI: Lane.PROFILE_ROI_HOVER,
            ComputeLane.PIXEL: Lane.PROFILE_ROI_HOVER,
        }[ComputeLane(lane)]

    def _resource_governor_work_active(self) -> bool:
        for controller in self._evaluation_controllers_by_lane().values():
            if controller is not None and controller.is_busy():
                return True
        coordinator = getattr(self, "render_coordinator", None)
        if bool(coordinator is not None and getattr(coordinator, "has_pending_render", False)):
            return True
        view = getattr(self, "img_view", None)
        draw_pending = getattr(view, "presentationDrawPending", None)
        if callable(draw_pending):
            try:
                if bool(draw_pending()):
                    return True
            except Exception:
                pass
        session = getattr(self, "_frame_session", None)
        if session is None:
            return False
        semantic_progress = getattr(session, "semantic_level_evidence_progress", None)
        return bool(
            getattr(session, "pending_tiles", None)
            or getattr(session, "loading_tiles", None)
            or getattr(session, "active_tile_requests", None)
            or getattr(session, "pending_level_tiles", None)
            or int(getattr(session, "level_scan_remaining_tiles", 0) or 0) > 0
            or bool(getattr(session, "histogram_aggregate_inflight", False))
            or (
                semantic_progress is not None
                and (
                    semantic_progress.inflight_generation is not None
                    or int(semantic_progress.pending_batches) > 0
                )
            )
            or getattr(session, "pending_payload_upserts", None)
            or getattr(session, "pending_removals", None)
            or getattr(session, "dirty_payloads", None)
            or bool(getattr(session, "final_commit_pending", False))
            or bool(getattr(session, "flush_pending", False))
            or getattr(session.stage_fan_in, "attached_requests", None)
        )

    def _scheduler_busy_state(self) -> SchedulerBusyState:
        session = getattr(self, "_frame_session", None)
        stage_ready = False
        backlog = 0
        semantic_evidence_blocking = False
        first_pixel_pending = False
        if session is not None:
            stage_ready = bool(
                session.stage_fan_in.values
                or session.stage_fan_in.active_requests
                or session.stage_fan_in.attached_requests
            )
            backlog = (
                len(getattr(session, "pending_payload_upserts", ()) or ())
                + len(getattr(session, "pending_removals", ()) or ())
            )
            semantic_progress = getattr(session, "semantic_level_evidence_progress", None)
            capabilities = image_view_backend_capabilities(self.img_view)
            semantic_evidence_blocking = bool(
                semantic_progress is not None
                and not bool(capabilities.shader_windowing)
                and not bool(getattr(session, "display_committed", False))
                and (
                    semantic_progress.inflight_generation is not None
                    or int(semantic_progress.pending_batches) > 0
                )
            )
            first_pixel_pending = not bool(session.visible_first_pixels_presented())
        kernel_visible_busy = bool(self.kernel.diagnostics().visible_backlog)
        return SchedulerBusyState(
            visible_busy=bool(
                first_pixel_pending
                or kernel_visible_busy
                or getattr(
                    getattr(self, "visible_evaluation_controller", None),
                    "is_busy",
                    lambda: False,
                )()
            ),
            montage_busy=bool(
                first_pixel_pending
                or getattr(
                    getattr(self, "montage_tile_evaluation_controller", None),
                    "is_busy",
                    lambda: False,
                )()
            ),
            stage_busy=getattr(getattr(self, "stage_evaluation_controller", None), "is_busy", lambda: False)(),
            prefetch_busy=getattr(getattr(self, "prefetch_evaluation_controller", None), "is_busy", lambda: False)(),
            result_backlog=backlog,
            stage_ready_or_in_flight=stage_ready,
            semantic_evidence_blocking=semantic_evidence_blocking,
        )

    def _interaction_active_now(self) -> bool:
        return bool(
            getattr(getattr(self, "render_coordinator", None), "interactive_active", False)
            or getattr(self, "_viewport_interaction_active", False)
        )

    def _note_interaction_state_changed(self) -> None:
        """Apply interaction quotas on the edge and wake deferred quality."""
        if getattr(self, "_closing", False):
            return
        active = self._interaction_active_now()
        previous_active = getattr(self, "_last_interaction_active_state", None)
        self._last_interaction_active_state = active
        if previous_active is None or bool(previous_active) != bool(active):
            self._apply_resource_governor_decisions(refresh_telemetry=False)
        if previous_active is True and not active:
            renderer = getattr(self, "renderer", None)
            replan = getattr(renderer, "replan_deferred_interactive_native_quality", None)
            if callable(replan):
                replan()

    def _note_kernel_completion_drain(self) -> None:
        if getattr(self, "_closing", False):
            return
        self._apply_resource_governor_decisions(refresh_telemetry=False)

    def _apply_resource_governor_decisions(self, *, refresh_telemetry: bool = True) -> None:
        governor = getattr(self, "resource_governor", None)
        if governor is None:
            return
        if refresh_telemetry:
            policy = self._refresh_memory_policy(active_render=self._resource_governor_work_active())
            governor.update_telemetry(sample_resource_snapshot(), policy)
        interactive = self._interaction_active_now()
        self._governor_interactive_applied = interactive
        busy = self._scheduler_busy_state()
        quota_by_lane: dict[Lane, int] = {}
        montage_worker_target = None
        for lane, controller in self._evaluation_controllers_by_lane().items():
            if controller is None:
                continue
            decision = governor.decide_lane_workers(lane, interactive=interactive, busy_state=busy)
            controller.set_reported_max_workers(decision.target_workers)
            if lane == ComputeLane.MONTAGE_TILE:
                montage_worker_target = int(decision.target_workers)
            kernel_lane = self._kernel_lane_for_compute_lane(lane)
            quota_by_lane[kernel_lane] = max(
                int(decision.target_workers),
                int(quota_by_lane.get(kernel_lane, 0)),
            )
        for lane, workers in quota_by_lane.items():
            self.kernel.set_lane_quota(lane, workers)
        if montage_worker_target is not None:
            preview_target = min(1, montage_worker_target) if interactive else montage_worker_target
            self.kernel.set_lane_quota(Lane.DISPLAY_PREVIEW, preview_target)
            self.kernel.set_lane_quota(
                Lane.DISPLAY_PREPARATION,
                0 if interactive else montage_worker_target,
            )
        # R4: completions drain through one QtKernelBridge; the governor owns
        # only this drain knob plus commit batch bounds and kernel lane quotas.
        bridge = getattr(self, "kernel_bridge", None)
        if bridge is not None:
            decision = governor.decide_bridge_drain(interactive=interactive)
            bridge.set_max_items_per_drain(decision.batch_limit)
            bridge.set_budget_ms(decision.budget_ms)
        # Montage prefetch asks the governor for a per-run tile budget in
        # `window.montage_prefetch`. Do not project that local decision onto
        # the shared prefetch controller: slice/profile prefetch have their
        # own policy depth and admission gates.

    def _record_ui_work(
        self,
        channel: str,
        elapsed_ms: float,
        *,
        count: int = 1,
        byte_count: int = 0,
        work_class: str = "",
        backend: str = "",
        details: tuple[str, ...] = (),
    ) -> None:
        governor = getattr(self, "resource_governor", None)
        if governor is not None:
            governor.record_ui_observation(
                channel,
                elapsed_ms,
                item_count=count,
                byte_count=byte_count,
                work_class=work_class,
                backend=backend,
                details=tuple(details),
            )
            return
        feedback = getattr(self, "latency_feedback", None)
        if feedback is not None:
            feedback.observe(channel, elapsed_ms, count=count, byte_count=byte_count)

    def _gui_callback_budget_decision(self, channel: str, *, interactive: bool = False):
        return default_gui_callback_budget_decision(channel, interactive=interactive)

    def _note_viewport_interaction(self, reason: str = "viewport") -> None:
        if str(reason) == "range-programmatic":
            return
        if str(reason) == "range-pointer":
            self._release_viewport_continuity()
        self._viewport_interaction_active = True
        timer = getattr(self, "_viewport_interaction_quiet_timer", None)
        if timer is None:
            # Timer category: UI cosmetic. User-interaction quiet detector for side-panel refresh and
            # in-progress viewport continuations. Montage tile correctness is
            # submitted through the kernel by the active retarget path.
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._on_viewport_interaction_quiet)
            self._viewport_interaction_quiet_timer = timer
        timer.start(120)
        self._note_interaction_state_changed()

    def _on_viewport_interaction_quiet(self) -> None:
        self._viewport_interaction_active = False
        self._note_interaction_state_changed()
        if getattr(self, "_montage_viewport_update_pending", False) and getattr(self.view_state, "montage_axis", None) is not None:
            self._montage_viewport_update_pending = False
            self.retarget_montage_viewport()
    def resizeEvent(self, event):
        # A top-level resize after the saved shape has settled is new user
        # intent. Release it before QMainWindow relays the resize into child
        # viewports; otherwise the child viewport callback can reopen the
        # continuity transaction and force the outer window back to the saved
        # shape. Programmatic continuity resizes enter with shape_settled=False.
        tx = self._viewport_continuity_transaction()
        if tx is not None and not tx.released and tx.shape_settled:
            self._release_viewport_continuity()
        super().resizeEvent(event)
        self._note_viewport_interaction("resize")
        if hasattr(self, "dimension_strip"):
            # Synchronous: viewport-continuity restore measures chrome right
            # after resizing, so the dims-area height must be settled (the
            # relayout early-returns when geometry is unchanged).
            self.dimension_strip._run_scheduled_relayout()
        renderer = getattr(self, "renderer", None)
        update_title = getattr(renderer, "_update_display_group_title", None)
        if callable(update_title):
            update_title()

    def _on_sync_facet_toggled(self, facet, enabled):
        controller = getattr(self, "sync_controller", None)
        if controller is not None:
            controller.set_facet_enabled(facet, bool(enabled))

    def closeEvent(self, event):
        self._closing = True
        controller = getattr(self, "sync_controller", None)
        if controller is not None:
            controller.shutdown()
        layout_manager = getattr(self, "layout_manager", None)
        if layout_manager is not None:
            layout_manager.canvas_preserver.cancel()
        coordinator = getattr(self, "render_coordinator", None)
        if coordinator is not None:
            coordinator.cancel_pending()
        for name in (
            "visible_evaluation_controller",
            "montage_tile_evaluation_controller",
            "stage_evaluation_controller",
            "histogram_evaluation_controller",
            "pixel_evaluation_controller",
            "profile_evaluation_controller",
            "roi_evaluation_controller",
            "prefetch_evaluation_controller",
        ):
            controller = getattr(self, name, None)
            if controller is not None:
                controller.shutdown_for_close()
        bridge = getattr(self, "kernel_bridge", None)
        if bridge is not None:
            bridge.close()
        kernel = getattr(self, "kernel", None)
        if kernel is not None:
            kernel.shutdown()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Rendering API — the window's public rendering surface.
    #
    # Rendering orchestration state and scheduling live on
    # ``self.renderer`` (RenderOrchestrator). The window exposes only the
    # semantic entry points below; siblings and widgets call these instead
    # of reaching into orchestration internals.
    # ------------------------------------------------------------------

    @property
    def _frame_session(self):
        return getattr(self.renderer, "_frame_session", None)

    @property
    def display_geometry(self):
        return getattr(self.renderer, "display_geometry", None)

    def render(self, *args, **kwargs):
        return self.renderer.render(*args, **kwargs)

    def request_render(self, *args, **kwargs):
        return self.renderer.request_render(*args, **kwargs)

    def update_image_view(self, *args, **kwargs):
        return self.renderer.update_image_view(*args, **kwargs)

    def update_line_plot(self, *args, **kwargs):
        return self.renderer.update_line_plot(*args, **kwargs)

    def update_display_mode(self, *args, **kwargs):
        return self.renderer.update_display_mode(*args, **kwargs)

    def auto_window_levels(self, *args, **kwargs):
        return self.renderer.auto_window_levels(*args, **kwargs)

    def fit_image_to_view(self, *args, **kwargs):
        return self.renderer.fit_image_to_view(*args, **kwargs)

    def one_to_one_image(self, *args, **kwargs):
        return self.renderer.one_to_one_image(*args, **kwargs)

    def is_line_plot_mode(self, *args, **kwargs):
        return self.renderer.is_line_plot_mode(*args, **kwargs)

    def toggle_profile_dock(self, *args, **kwargs):
        return self.renderer.toggle_profile_dock(*args, **kwargs)

    def on_tab_changed(self, *args, **kwargs):
        return self.renderer.on_tab_changed(*args, **kwargs)

    def getPixel(self, *args, **kwargs):
        return self.renderer.getPixel(*args, **kwargs)

    def _on_view_range_changed(self, *args, **kwargs):
        return self.renderer._on_view_range_changed(*args, **kwargs)

    def _on_display_levels_changed(self, *args, **kwargs):
        return self.renderer._on_display_levels_changed(*args, **kwargs)

    def _on_level_presentation_changed(self, *args, **kwargs):
        return self.renderer._on_level_presentation_changed(*args, **kwargs)

    def _on_image_mouse_moved(self, *args, **kwargs):
        return self.renderer._on_image_mouse_moved(*args, **kwargs)

    def _on_image_viewport_resized(self, *args, **kwargs):
        return self.renderer._on_image_viewport_resized(*args, **kwargs)

    def _on_inspection_dock_visibility_changed(self, *args, **kwargs):
        return self.renderer._on_inspection_dock_visibility_changed(*args, **kwargs)

    def _on_operation_dock_visibility_changed(self, *args, **kwargs):
        return self.renderer._on_operation_dock_visibility_changed(*args, **kwargs)

    def _on_profile_dock_visibility_changed(self, *args, **kwargs):
        return self.renderer._on_profile_dock_visibility_changed(*args, **kwargs)

    def _on_live_profile_toggled(self, *args, **kwargs):
        return self.renderer._on_live_profile_toggled(*args, **kwargs)

    def _on_profile_marker_moved(self, *args, **kwargs):
        return self.renderer._on_profile_marker_moved(*args, **kwargs)

    def _on_window_mode_changed(self, *args, **kwargs):
        return self.renderer._on_window_mode_changed(*args, **kwargs)

    def _update_live_profile_from_pending_pos(self, *args, **kwargs):
        return self.renderer._update_live_profile_from_pending_pos(*args, **kwargs)

    def _processing_pressed(self, *args, **kwargs):
        return self.renderer._processing_pressed(*args, **kwargs)

    def _apply_channel_colormap(self, *args, **kwargs):
        return self.renderer._apply_channel_colormap(*args, **kwargs)

    def _set_display_colormap(self, *args, **kwargs):
        return self.renderer._set_display_colormap(*args, **kwargs)

    def _phase_colormap(self, *args, **kwargs):
        return self.renderer._phase_colormap(*args, **kwargs)

    def _clear_image_hover_state(self, *args, **kwargs):
        return self.renderer._clear_image_hover_state(*args, **kwargs)

    def _queue_display_levels(self, *args, **kwargs):
        return self.renderer._queue_display_levels(*args, **kwargs)

    def _pending_display_levels_for_render(self, *args, **kwargs):
        return self.renderer._pending_display_levels_for_render(*args, **kwargs)

    def _current_window_mode(self, *args, **kwargs):
        return self.renderer._current_window_mode(*args, **kwargs)

    def _refresh_memory_policy(self, *args, **kwargs):
        return self.renderer._refresh_memory_policy(*args, **kwargs)

    def _run_deferred_side_panel_refresh(self, *args, **kwargs):
        return self.renderer._run_deferred_side_panel_refresh(*args, **kwargs)

    def retarget_montage_viewport(self):
        return self.renderer.retarget_montage_viewport()

    def _visible_render_budget_bytes(self, *args, **kwargs):
        return self.renderer._visible_render_budget_bytes(*args, **kwargs)

    def _current_montage_resize_focus(self, *args, **kwargs):
        return self.renderer._current_montage_resize_focus(*args, **kwargs)

    @property
    def _current_montage_plan(self):
        return getattr(self.renderer, "_current_montage_plan", None)

    @property
    def _current_montage_geometry(self):
        return getattr(self.renderer, "_current_montage_geometry", None)

    def _is_committed_display_frame_current(self, *args, **kwargs):
        return self.renderer._is_committed_display_frame_current(*args, **kwargs)

    def _interactive_frame_cache_hit(self, *args, **kwargs):
        return self.renderer._interactive_frame_cache_hit(*args, **kwargs)

    def _interactive_render_supersedes_presentation(self, *args, **kwargs):
        return self.renderer._interactive_render_supersedes_presentation(*args, **kwargs)

    def _cancel_render_dependent_work_for_interactive_change(self, *args, **kwargs):
        return self.renderer._cancel_render_dependent_work_for_interactive_change(*args, **kwargs)
