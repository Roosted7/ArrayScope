import numpy as np
from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets
import platform
from arrayscope.operations.coordinator import OperationCoordinator
from arrayscope.profiles.coordinator import ProfileCoordinator
from arrayscope.core.array_metadata import derived_info_for
from arrayscope.core.compute_policy import ComputeLane, EvaluationContext, compute_policy_from_settings
from arrayscope.core.resource_governor import ResourceGovernor, SchedulerBusyState
from arrayscope.core.resource_telemetry import sample_resource_snapshot
from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.core.roi_store import RoiStore
from arrayscope.export.workflow import ExportWorkflowMixin
from arrayscope.ui.dimension_controls import DimensionControlMixin
from arrayscope.ui.display_controls import DisplayControlBuildMixin
from arrayscope.ui.menus import WindowMenuMixin
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.domain import Domain
from arrayscope.window.evaluation_controller import EvaluationController
from arrayscope.window.file_reload import FileReloadMixin
from arrayscope.window.inspection import InspectionWorkflowMixin
from arrayscope.window.interaction_mode import InteractionMode
from arrayscope.window.operation_actions import OperationActionsMixin
from arrayscope.window.render import RenderMixin
from arrayscope.window.render_coordinator import RenderCoordinator
from arrayscope.window.render_generation import RenderGeneration
from arrayscope.window.state_sync import StateSyncMixin


class ArrayScopeWindow(
    WindowMenuMixin,
    DisplayControlBuildMixin,
    StateSyncMixin,
    OperationActionsMixin,
    InspectionWorkflowMixin,
    DimensionControlMixin,
    RenderMixin,
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

    def __init__(self, data, complex_dim=None, filepath=None, dataset_path=None, selector_class_name=None):
        super(ArrayScopeWindow, self).__init__()
        self.resize(600,800)
        self._settings = Qt.QtCore.QSettings()
        self.app_settings = self._load_app_settings()
        self._apply_theme_choice(self.app_settings.theme, persist=False)
        self._apply_performance_settings(persist=False)
        self.compute_policy = compute_policy_from_settings(self.app_settings)

        self.operation_coordinator = OperationCoordinator(data)
        self.profile_coordinator = ProfileCoordinator()
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self._refresh_memory_policy(active_render=False)
        self.resource_governor = ResourceGovernor(
            self.compute_policy,
            profile=self.app_settings.memory_profile,
        )
        self.latency_feedback = self.resource_governor.latency_feedback
        self.resource_governor.update_telemetry(
            sample_resource_snapshot(),
            self._memory_policy(),
        )
        self._init_compare_document(data)
        self._render_generation = RenderGeneration()
        self.visible_evaluation_controller = EvaluationController(self, max_workers=self.compute_policy.visible_workers, name="visible")
        self.evaluation_controller = self.visible_evaluation_controller
        self.montage_tile_evaluation_controller = EvaluationController(
            self,
            max_workers=self.compute_policy.montage_tile_workers,
            name="montage",
            max_callback_dispatch_per_drain=8,
        )
        self.stage_evaluation_controller = EvaluationController(self, max_workers=self.compute_policy.stage_workers, name="stage")
        self.pixel_evaluation_controller = EvaluationController(self, max_workers=self.compute_policy.pixel_workers, name="pixel")
        self.profile_evaluation_controller = EvaluationController(self, max_workers=self.compute_policy.profile_workers, name="profile")
        self.roi_evaluation_controller = EvaluationController(self, max_workers=self.compute_policy.roi_workers, name="roi")
        self.prefetch_evaluation_controller = EvaluationController(self, max_workers=self.compute_policy.prefetch_workers, name="prefetch")
        self._ensure_resource_governor_timer()
        self.render_coordinator = RenderCoordinator(self)
        self._deferred_side_panel_refresh_pending = False
        self.data = derived_info_for(self.document)
        self.singleton = [e == 1 for e in list(self.data.shape)]
        initial_channel = ChannelMode.COMPLEX if np.issubdtype(self.data.dtype, np.complexfloating) else ChannelMode.REAL
        self.view_state = ViewState.from_shape(self.data.shape).with_channel(initial_channel)
        self._channel_user_selected = False
        self._force_autolevel = False
        self._filepath = filepath
        self._dataset_path = dataset_path
        self._selector_class_name = selector_class_name
        self._operation_dock_user_visible = None
        self._profile_dock_user_visible = None
        self._inspection_dock_user_visible = None
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
        self.render(reason="initial", force_autolevel=True)
        self.show()
        Qt.QtCore.QTimer.singleShot(0, lambda: setattr(self, "_progressive_preserve_enabled", True))

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
            memory_policy=self._memory_policy(),
        )

    def _apply_compute_policy(self) -> None:
        self.compute_policy = compute_policy_from_settings(self.app_settings)
        governor = getattr(self, "resource_governor", None)
        if governor is not None:
            governor.update_policy(self.compute_policy, profile=getattr(self.app_settings, "memory_profile", None))
            self._apply_resource_governor_decisions()
            return
        for lane, controller in self._evaluation_controllers_by_lane().items():
            if controller is not None:
                controller.set_max_workers(self.compute_policy.workers_for_lane(lane))

    def _evaluation_controllers_by_lane(self):
        return {
            ComputeLane.VISIBLE: getattr(self, "visible_evaluation_controller", None),
            ComputeLane.MONTAGE_TILE: getattr(self, "montage_tile_evaluation_controller", None),
            ComputeLane.STAGE: getattr(self, "stage_evaluation_controller", None),
            ComputeLane.PREFETCH: getattr(self, "prefetch_evaluation_controller", None),
            ComputeLane.PROFILE: getattr(self, "profile_evaluation_controller", None),
            ComputeLane.ROI: getattr(self, "roi_evaluation_controller", None),
            ComputeLane.PIXEL: getattr(self, "pixel_evaluation_controller", None),
        }

    def _ensure_resource_governor_timer(self):
        timer = getattr(self, "_resource_governor_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.timeout.connect(self._on_resource_governor_timer)
            self._resource_governor_timer = timer
        if not timer.isActive():
            timer.start(250)
        return timer

    def _on_resource_governor_timer(self) -> None:
        if getattr(self, "_closing", False):
            return
        active = self._resource_governor_work_active()
        self._apply_resource_governor_decisions()
        timer = getattr(self, "_resource_governor_timer", None)
        if timer is not None:
            timer.start(250 if active else 1000)

    def _resource_governor_work_active(self) -> bool:
        for controller in self._evaluation_controllers_by_lane().values():
            if controller is not None and controller.is_busy():
                return True
        coordinator = getattr(self, "render_coordinator", None)
        return bool(coordinator is not None and getattr(coordinator, "has_pending_render", False))

    def _scheduler_busy_state(self) -> SchedulerBusyState:
        session = getattr(self, "_montage_session", None)
        stage_ready = False
        backlog = 0
        if session is not None:
            stage_ready = bool(getattr(session, "stage_values", None) or getattr(session, "active_stage_requests", None) or getattr(session, "attached_stage_requests", None))
            backlog = len(getattr(session, "pending_completed_tiles", ()) or ())
        return SchedulerBusyState(
            visible_busy=getattr(getattr(self, "visible_evaluation_controller", None), "is_busy", lambda: False)(),
            montage_busy=getattr(getattr(self, "montage_tile_evaluation_controller", None), "is_busy", lambda: False)(),
            stage_busy=getattr(getattr(self, "stage_evaluation_controller", None), "is_busy", lambda: False)(),
            prefetch_busy=getattr(getattr(self, "prefetch_evaluation_controller", None), "is_busy", lambda: False)(),
            result_backlog=backlog,
            stage_ready_or_in_flight=stage_ready,
        )

    def _apply_resource_governor_decisions(self) -> None:
        governor = getattr(self, "resource_governor", None)
        if governor is None:
            return
        policy = self._refresh_memory_policy(active_render=self._resource_governor_work_active())
        governor.update_telemetry(sample_resource_snapshot(), policy)
        interactive = bool(getattr(getattr(self, "render_coordinator", None), "interactive_active", False))
        busy = self._scheduler_busy_state()
        for lane, controller in self._evaluation_controllers_by_lane().items():
            if controller is None:
                continue
            decision = governor.decide_lane_workers(lane, interactive=interactive, busy_state=busy)
            controller.set_max_workers(decision.target_workers)
        for channel, controller in (
            ("visible_callback", getattr(self, "visible_evaluation_controller", None)),
            ("montage_tile_result", getattr(self, "montage_tile_evaluation_controller", None)),
            ("stage_callback", getattr(self, "stage_evaluation_controller", None)),
            ("prefetch_callback", getattr(self, "prefetch_evaluation_controller", None)),
            ("profile_update", getattr(self, "profile_evaluation_controller", None)),
            ("roi_refresh", getattr(self, "roi_evaluation_controller", None)),
            ("pixel_hover", getattr(self, "pixel_evaluation_controller", None)),
        ):
            if controller is None:
                continue
            decision = governor.decide_ui_work(channel, interactive=interactive)
            controller.set_max_callback_dispatch_per_drain(decision.batch_limit)
        histogram_decision = governor.decide_ui_work("histogram_preview", interactive=interactive)
        img_view = getattr(self, "img_view", None)
        if img_view is not None and hasattr(img_view, "setHistogramPreviewInterval"):
            img_view.setHistogramPreviewInterval(histogram_decision.interval_ms)
        profile_decision = governor.decide_ui_work("profile_update", interactive=interactive)
        profile_timer = getattr(self, "_profile_timer", None)
        if profile_timer is not None:
            profile_timer.setInterval(max(1, int(profile_decision.interval_ms)))
        prefetch_decision = governor.decide_montage_prefetch(
            stage_ready_or_in_flight=busy.stage_ready_or_in_flight,
            visible_busy=busy.visible_busy or busy.montage_busy or busy.stage_busy,
        )
        prefetch = getattr(self, "prefetch_evaluation_controller", None)
        if prefetch is not None:
            prefetch.set_max_prefetch(max(1, prefetch_decision.max_items if prefetch_decision.allowed else 1))

    def _record_ui_work(self, channel: str, elapsed_ms: float, *, count: int = 1) -> None:
        governor = getattr(self, "resource_governor", None)
        if governor is not None:
            governor.record_ui_observation(channel, elapsed_ms, item_count=count)
            return
        feedback = getattr(self, "latency_feedback", None)
        if feedback is not None:
            feedback.observe(channel, elapsed_ms, count=count)

    def _ui_work_decision(self, channel: str, *, interactive: bool = False):
        governor = getattr(self, "resource_governor", None)
        if governor is not None:
            return governor.decide_ui_work(channel, interactive=interactive)
        return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "dimension_strip"):
            self.dimension_strip._schedule_relayout()

    def closeEvent(self, event):
        self._closing = True
        coordinator = getattr(self, "render_coordinator", None)
        if coordinator is not None:
            coordinator.cancel_pending()
        for name in (
            "visible_evaluation_controller",
            "montage_tile_evaluation_controller",
            "stage_evaluation_controller",
            "pixel_evaluation_controller",
            "profile_evaluation_controller",
            "roi_evaluation_controller",
            "prefetch_evaluation_controller",
        ):
            controller = getattr(self, name, None)
            if controller is not None:
                controller.shutdown_for_close()
        super().closeEvent(event)
