import numpy as np
from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets
import platform
from arrayscope.operations.coordinator import OperationCoordinator
from arrayscope.profiles.coordinator import ProfileCoordinator
from arrayscope.core.array_metadata import derived_info_for
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
        self._settings = Qt.QtCore.QSettings("ArrayScope", "ArrayScope")
        self.app_settings = self._load_app_settings()
        self._apply_theme_choice(self.app_settings.theme, persist=False)
        self._apply_performance_settings(persist=False)

        self.operation_coordinator = OperationCoordinator(data)
        self.profile_coordinator = ProfileCoordinator()
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self._refresh_memory_policy(active_render=False)
        self._init_compare_document(data)
        self.visible_evaluation_controller = EvaluationController(self, max_workers=1, name="visible")
        self.evaluation_controller = self.visible_evaluation_controller
        self.pixel_evaluation_controller = EvaluationController(self, max_workers=1, name="pixel")
        self.profile_evaluation_controller = EvaluationController(self, max_workers=1, name="profile")
        self.roi_evaluation_controller = EvaluationController(self, max_workers=1, name="roi")
        self.prefetch_evaluation_controller = EvaluationController(self, max_workers=1, name="prefetch")
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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "dimension_strip"):
            self.dimension_strip._schedule_relayout()
