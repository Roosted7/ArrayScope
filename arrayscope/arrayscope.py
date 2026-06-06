import numpy as np
from .qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets, QtGui
import os
import math
import platform
from enum import Enum
from pathlib import Path
from .colormaps import gray_colormap, named_colormap, phase_colormap
from .dialogs import SaveRangeDialog
from .dimension_roles import DimensionRoles
from .imageview2d import ImageView2D
from .operation_dock import OperationStackDock
from .operation_coordinator import OperationCoordinator
from .operation_evaluator import OperationEvaluator
from .operation_pipeline import ArrayDocument, evaluate_shape
from .operation_recipes import load_recipe, save_recipe
from .operation_registry import operation_entries
from .operation_stack import delete_operation, move_operation, reorder_operations
from .profile import clamp_marker_position, image_hover_indices, profile_state_from_image_hover, profile_y_range
from .profile_coordinator import ProfileCoordinator
from .profile_dock import ProfileDock
from .settings_state import AppSettingsState, settings_from_mapping, settings_to_mapping
from .slice_engine import apply_channel
from .theme import ThemeChoice, apply_theme_to_qapplication
from .video_export import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
from .view_state import ChannelMode, ScaleMode, ViewState
from .window_levels import choose_window_levels

def getNumberOfDecimalPlaces(number):
    if isinstance(number, (int, np.integer)):
        return int(0)
    else:
        return int(max(1, (number.as_integer_ratio()[1]).bit_length()))

class Domain(Enum):
    INV_FOURIER=-1
    NATIVE=0
    FOURIER=1


class ArrayScopeWindow(QtWidgets.QMainWindow):
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
        self.resize(800,800)
        self._settings = Qt.QtCore.QSettings("ArrayScope", "ArrayScope")
        self.app_settings = self._load_app_settings()
        self._apply_theme_choice(self.app_settings.theme, persist=False)

        self.operation_coordinator = OperationCoordinator(data)
        self.profile_coordinator = ProfileCoordinator()
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.data = self.operation_evaluator.current_data()
        self.singleton = [e == 1 for e in list(data.shape)]
        initial_channel = ChannelMode.COMPLEX if np.iscomplexobj(self.data) else ChannelMode.REAL
        self.view_state = ViewState.from_shape(self.data.shape).with_channel(initial_channel)
        self._force_autolevel = False
        self._filepath = filepath
        self._dataset_path = dataset_path
        self._selector_class_name = selector_class_name
        
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
                
        self.domain = [Domain.NATIVE for _ in range(data.ndim)]
        self.widgets = {
            'buttons': {
                'primary': [QtWidgets.QPushButton('Y', checkable=True) for i in range(data.ndim)],
                'secondary': [QtWidgets.QPushButton('X', checkable=True) for i in range(data.ndim)],
                'profile': [QtWidgets.QPushButton('P', checkable=True) for i in range(data.ndim)],
                'channel': {
                    'complex': QtWidgets.QRadioButton('complex', enabled=np.iscomplexobj(self.data)),
                    'real': QtWidgets.QRadioButton('real', enabled=np.iscomplexobj(self.data)),
                    'imag': QtWidgets.QRadioButton('imag', enabled=np.iscomplexobj(self.data)),
                    'abs': QtWidgets.QRadioButton('abs', enabled=np.iscomplexobj(self.data)),
                    'angle': QtWidgets.QRadioButton('angle', enabled=np.iscomplexobj(self.data)),
                },
                'processing': {
                    #'log': QtWidgets.QRadioButton('log', checkable=True),
                    'linear': QtWidgets.QRadioButton('linear', checkable=True, checked=True),
                    'symlog': QtWidgets.QRadioButton('symlog', checkable=True),
                    #'asinh': QtWidgets.QRadioButton('asinh', checkable=True)
                    
                },
                'display': {
                    'square_pixels': QtWidgets.QRadioButton('Square pixels', checkable=True, checked=True),
                    'square_fov': QtWidgets.QRadioButton('Square FOV', checkable=True),
                    'fit': QtWidgets.QRadioButton('Fit', checkable=True),
                    'window_relative': QtWidgets.QRadioButton('Relative', checkable=True, checked=True),
                    'window_absolute': QtWidgets.QRadioButton('Absolute', checkable=True),
                    'live_profile': QtWidgets.QCheckBox('Live profile'),
                }
            },
            'labels': {
                'dims': [QtWidgets.QLabel('[' + str(data.shape[i]) + ']', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'flip': [QtWidgets.QLabel('', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'shift': [QtWidgets.QLabel('', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'complex': [QtWidgets.QLabel('', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'primary': QtWidgets.QLabel('Y'),
                'secondary': QtWidgets.QLabel('X'),
                'slice': QtWidgets.QLabel('Slice'),
                'dimensions': QtWidgets.QLabel('Dimensions'),
                'pixelValue': QtWidgets.QLabel(''),
                'arrayInfo': QtWidgets.QLabel('')
            },
            'spins': {
                'slice_indices': [QtWidgets.QSpinBox(minimum=0, maximum=data.shape[i]-1) for i in range(data.ndim)]
            }
        }
        
        # Create a button group for the channel radio buttons
        self.channel_button_group = QtWidgets.QButtonGroup()
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['complex'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['real'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['imag'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['abs'])
        self.channel_button_group.addButton(self.widgets['buttons']['channel']['angle'])
        self._update_channel_controls()
            
        self.scale_button_group = QtWidgets.QButtonGroup()
        #self.scale_button_group.addButton(self.widgets['buttons']['processing']['log'])
        self.scale_button_group.addButton(self.widgets['buttons']['processing']['linear'])
        self.scale_button_group.addButton(self.widgets['buttons']['processing']['symlog'])
        #self.scale_button_group.addButton(self.widgets['buttons']['processing']['asinh'])
        
        
        self.display_button_group = QtWidgets.QButtonGroup()
        self.display_button_group.addButton(self.widgets['buttons']['display']['square_pixels'])
        self.display_button_group.addButton(self.widgets['buttons']['display']['square_fov'])
        self.display_button_group.addButton(self.widgets['buttons']['display']['fit'])

        self.window_button_group = QtWidgets.QButtonGroup()
        self.window_button_group.addButton(self.widgets['buttons']['display']['window_relative'])
        self.window_button_group.addButton(self.widgets['buttons']['display']['window_absolute'])
        self._pending_profile_pos = None
        self._pending_profile_point = None
        self._profile_timer = Qt.QtCore.QTimer(self)
        self._profile_timer.setSingleShot(True)
        self._profile_timer.setInterval(40)
        self._profile_timer.timeout.connect(self._update_live_profile_from_pending_pos)
        
        self.layouts = {
            'main': QtWidgets.QVBoxLayout(),
            'top': QtWidgets.QVBoxLayout(),
            'topUp': QtWidgets.QHBoxLayout(),
            'topDown': QtWidgets.QHBoxLayout(),
            'bot': QtWidgets.QHBoxLayout(),
            'botLeft': QtWidgets.QVBoxLayout(),
            'botRight': QtWidgets.QVBoxLayout(),
            'dims': QtWidgets.QHBoxLayout(),
            'primary': QtWidgets.QHBoxLayout(),
            'secondary': QtWidgets.QHBoxLayout(),
            'slice': QtWidgets.QHBoxLayout(),
            'hover': QtWidgets.QHBoxLayout()
        }
        
        for i, label in enumerate(self.widgets['labels']['dims']):
            label.mousePressEvent = lambda event, i=i, l=label: self.dimClicked(event, l, i)
            label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
            label.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            label.customContextMenuRequested.connect(lambda pos, label=label, dim=i: self._show_operation_context_menu(pos, label, dim))
            label.setToolTip(f"Left click: centered FFT along dim {i}. Right click: operations.")

        
        # Set up flip labels with click handlers
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            flip_label.mousePressEvent = lambda event, i=i: self.flipAxisClicked(event, i)
            flip_label.setStyleSheet(self.FLIP_ICON_STYLE)
            flip_label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignLeft | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._set_emoji_font(flip_label)

        for shift_label in self.widgets['labels']['shift']:
            shift_label.setVisible(False)
        
        # Set up complex indicator labels with click handlers
        for i, complex_label in enumerate(self.widgets['labels']['complex']):
            complex_label.mousePressEvent = lambda event, i=i: self.complexOrRealClicked(event, i)
            complex_label.setStyleSheet(self.FLIP_ICON_STYLE)
            complex_label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignRight | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._set_emoji_font(complex_label)
        
        # Apply compact styling to dimension control widgets
        for label in self.widgets['labels']['dims']:
            label.setStyleSheet(self.DIMENSION_LABEL_STYLE)
            label.setMinimumHeight(24)
            label.setMinimumWidth(30)
        
        for btn in self.widgets['buttons']['primary'] + self.widgets['buttons']['secondary'] + self.widgets['buttons']['profile']:
            btn.setStyleSheet(self.BUTTON_STYLE)
            btn.setFixedWidth(28)
            
        for spin in self.widgets['spins']['slice_indices']:
            spin.setStyleSheet(self.SPINBOX_STYLE)
        
        # Set minimal spacing for all control layouts
        self.layouts['dims'].setSpacing(2)  # Extra tight for dimensions row
        self.layouts['dims'].setContentsMargins(0, 0, 0, 0)  # Remove all margins
        self.layouts['primary'].setSpacing(2)  # Tighter spacing
        self.layouts['primary'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        self.layouts['secondary'].setSpacing(2)  # Tighter spacing
        self.layouts['secondary'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        self.layouts['slice'].setSpacing(2)  # Tighter spacing
        self.layouts['slice'].setContentsMargins(0, 1, 0, 1)  # Minimal margins
        
        # For each dimension, create a QVBoxLayout with flip icon + dim label above primary button
        dim_containers = []
        for i in range(data.ndim):
            # Create a container widget for this dimension column
            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout()
            container_layout.setSpacing(0)
            container_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create horizontal layout for flip icon + dimension label
            label_row = QtWidgets.QWidget()
            label_layout = QtWidgets.QHBoxLayout()
            label_layout.setSpacing(0)
            label_layout.setContentsMargins(0, 0, 0, 0)
            
            # Add flip icon (left-aligned)
            label_layout.addWidget(self.widgets['labels']['flip'][i])
            # Add dimension label (centered, takes remaining space)
            self.widgets['labels']['dims'][i].setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
            label_layout.addWidget(self.widgets['labels']['dims'][i], 1)
            # Add complex indicator (ℝ/ℂ)
            label_layout.addWidget(self.widgets['labels']['complex'][i])
            
            label_row.setLayout(label_layout)
            container_layout.addWidget(label_row)
            
            container.setLayout(container_layout)
            dim_containers.append(container)
            self.layouts['dims'].addWidget(container)
        
        # Store containers for later access
        self.dim_containers = dim_containers
        
        # Add all buttons and spinboxes to their vertical containers
        for i in range(data.ndim):
            w = self.widgets['buttons']['primary'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.setToolTip(f"Use dim {i} as image Y axis")
            w.clicked.connect(lambda checked, i=i : self.set_dimension_role("y", i))
            
            w = self.widgets['buttons']['secondary'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.setToolTip(f"Use dim {i} as image X axis")
            w.clicked.connect(lambda checked, i=i: self.set_dimension_role("x", i))

            w = self.widgets['buttons']['profile'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.setToolTip(f"Use dim {i} as profile axis")
            w.clicked.connect(lambda checked, i=i: self.set_dimension_role("p", i))
            
            w = self.widgets['spins']['slice_indices'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.valueChanged.connect(lambda value, i=i: self._on_slice_index_changed(i, value))
        
        self._setup_export_context_menus()
        self._save_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Save, self)
        self._save_shortcut.activated.connect(self._save_current_numpy_file)
        
        # Create a single compact control panel with all radio buttons
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout()
        
        # Channel controls - horizontal layout to save space
        channel_group = QtWidgets.QGroupBox("Channel")
        channel_group.setStyleSheet(self.GROUPBOX_BASE_STYLE)
        channel_layout = QtWidgets.QHBoxLayout()
        channel_layout.setSpacing(5)
        channel_layout.setContentsMargins(3, 3, 3, 3)
        
        buttons = list(self.widgets['buttons']['channel'].values())
        for btn in buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            self.channel_button_group.addButton(btn)
            channel_layout.addWidget(btn)
            btn.clicked.connect(lambda checked=False, name=btn.text(): self._on_channel_clicked(name))
        
        channel_group.setLayout(channel_layout)
        controls_layout.addWidget(channel_group)
        
        # Processing controls - horizontal layout
        processing_group = QtWidgets.QGroupBox("Scale")
        processing_group.setStyleSheet(self.GROUPBOX_BASE_STYLE)
        processing_layout = QtWidgets.QHBoxLayout()
        processing_layout.setSpacing(5)
        processing_layout.setContentsMargins(3, 3, 3, 3)
        
        proc_buttons = list(self.widgets['buttons']['processing'].values())
        for btn in proc_buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            processing_layout.addWidget(btn)
            # When a processing button is pressed while already active, force auto-level
            btn.pressed.connect(lambda b=btn: self._processing_pressed(b))
            btn.clicked.connect(lambda checked=False, b=btn: self._on_scale_clicked("symlog" if b is self.widgets['buttons']['processing']['symlog'] else "linear"))
        
        processing_group.setLayout(processing_layout)
        controls_layout.addWidget(processing_group)
        
        # Display controls - horizontal layout
        self.display_group = QtWidgets.QGroupBox("Display")
        self.display_group.setStyleSheet(self.GROUPBOX_BASE_STYLE)
        display_layout = QtWidgets.QHBoxLayout()
        display_layout.setSpacing(5)
        display_layout.setContentsMargins(3, 3, 3, 3)
        
        disp_buttons = [
            self.widgets['buttons']['display']['square_pixels'],
            self.widgets['buttons']['display']['square_fov'],
            self.widgets['buttons']['display']['fit'],
        ]
        for btn in disp_buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            display_layout.addWidget(btn)
            btn.clicked.connect(self.update_display_mode)
        window_relative = self.widgets['buttons']['display']['window_relative']
        window_absolute = self.widgets['buttons']['display']['window_absolute']
        window_relative.setStyleSheet(self.RADIO_BUTTON_STYLE)
        window_absolute.setStyleSheet(self.RADIO_BUTTON_STYLE)
        window_relative.setToolTip("Preserve the selected relative range within each new histogram")
        window_absolute.setToolTip("Keep numeric histogram/window limits fixed while changing views")
        display_layout.addWidget(window_relative)
        display_layout.addWidget(window_absolute)
        live_profile = self.widgets['buttons']['display']['live_profile']
        live_profile.setStyleSheet(self.RADIO_BUTTON_STYLE)
        live_profile.setToolTip("Update the line plot from the mouse position in the image view")
        live_profile.toggled.connect(self._on_live_profile_toggled)
        display_layout.addWidget(live_profile)
        
        # Optional: tooltip hints
        self.widgets['buttons']['display']['square_pixels'].setToolTip('Lock to 1:1 pixel aspect')
        self.widgets['buttons']['display']['square_fov'].setToolTip('Lock aspect to image width/height ratio')
        self.widgets['buttons']['display']['fit'].setToolTip('Always fit entire image in viewport')
        
        self.display_group.setLayout(display_layout)
        controls_layout.addWidget(self.display_group)
        
        # Add stretch to push everything to the top
        controls_layout.addStretch()
        
        controls_widget.setLayout(controls_layout)
        self.layouts['botRight'].addWidget(controls_widget)
        
        # Create tab widget for switching between image and line plot views
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Create image view tab
        self.image_tab = QtWidgets.QWidget()
        self.image_tab_layout = QtWidgets.QVBoxLayout()
        
        self.img_view = ImageView2D()
        self.image_tab_layout.addWidget(self.img_view)
        self.image_tab.setLayout(self.image_tab_layout)
        self.img_view.getView().scene().sigMouseMoved.connect(lambda pos: self._on_image_mouse_moved(pos))
        self.img_view.set_profile_marker_callback(self._on_profile_marker_moved)
        
        # Connect to view range changes to update aspect ratio in fit mode
        self.img_view.getView().sigRangeChanged.connect(self._on_view_range_changed)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.image_tab, "Image View")
        
        # Connect tab change handler
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        self.tab_widget.tabBar().installEventFilter(self)
        
        # Add tab widget to the main layout
        self.layouts['topDown'].addWidget(self.tab_widget)

        # Reload / file-changed button (⟳ by default, ⚠️ when file changes on disk)
        self._reload_btn = QtWidgets.QPushButton("⟳")
        self._reload_btn.setStyleSheet("QPushButton { font-size: 18pt; padding: 1px 2px; margin: 0px; border: none; background: transparent; }")
        self._reload_btn.setToolTip("Reload file")
        self._reload_btn.setFlat(True)
        self._reload_btn.setFixedSize(28, 20)
        self._reload_btn.clicked.connect(self._reload_file)
        self._reload_btn.setVisible(filepath is not None)
        self._set_emoji_font(self._reload_btn)
        self.layouts['topUp'].addWidget(self._reload_btn)
        self.layouts['topUp'].addWidget(self.widgets['labels']['pixelValue'])
        self.layouts['topUp'].addWidget(self.widgets['labels']['arrayInfo'])

        self.layouts['botLeft'].addLayout(self.layouts['dims'])
        
        
        # Create container widgets for proper alignment
        left_container = QtWidgets.QWidget()
        left_container.setLayout(self.layouts['botLeft'])
        
        right_container = QtWidgets.QWidget()
        right_container.setLayout(self.layouts['botRight'])
        
        # Set both containers to expand vertically the same way
        left_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        right_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        
        self.layouts['top'].addLayout(self.layouts['topUp'],1)
        self.layouts['top'].addLayout(self.layouts['topDown'],20)  # Give viewport much more space
        
        self.layouts['bot'].addWidget(left_container, 12)
        self.layouts['bot'].addWidget(right_container, 1)

        # Give the viewport (top) much more weight so it expands when window grows
        self.layouts['main'].addLayout(self.layouts['top'], 10)  # Viewport gets most space
        self.layouts['main'].addLayout(self.layouts['bot'], 1)   # Controls get minimal fixed space
        tmp = QtWidgets.QWidget()
        tmp.setLayout(self.layouts['main'])
        self.setCentralWidget(tmp)
        self.profile_dock = ProfileDock(self, on_axis_changed=self.set_profile_axis)
        self.line_plot = self.profile_dock.line_plot
        self.plot_widget = self.profile_dock.widget
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.profile_dock)
        self.profile_dock.set_axes(self.data.shape, self.line_plot_dimension)
        self.profile_dock.visibilityChanged.connect(self._on_profile_dock_visibility_changed)
        self.profile_dock.hide()
        self.operation_dock = OperationStackDock(
            self,
            on_undo=self.undo_last_operation,
            on_clear=self.clear_operations,
            on_save_recipe=self.save_operation_recipe,
            on_load_recipe=self.load_operation_recipe,
            on_materialize=self.materialize_current_array,
            on_delete_selected=self.delete_selected_operation,
            on_move_selected_up=lambda index: self.move_selected_operation(index, -1),
            on_move_selected_down=lambda index: self.move_selected_operation(index, 1),
            on_reorder=self.reorder_operations,
        )
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.operation_dock)
        self._update_operation_dock()
        self._setup_menus()
        self._restore_window_settings()
        
        # Initialize complex indicators for size-2 real dimensions
        self.update_complex_indicators()
        self.update_shift_indicators()
        
        if complex_dim is not None: # user requested combining as complex
            if complex_dim < 0 or complex_dim >= data.ndim:
                print(f"Warning: complex_dim={complex_dim} is out of range for {data.ndim}D array. Ignoring.")
            elif np.iscomplexobj(data):
                print(f"Warning: Data is already complex. Ignoring complex_dim={complex_dim}.")
            elif data.shape[complex_dim] != 2:
                print(f"Warning: Dimension {complex_dim} has shape {data.shape[complex_dim]}, not 2. Cannot combine as complex. Ignoring.")
            else:
                self.combineAsComplex(complex_dim) # valid
        
        # Initialize dimension controls based on the authoritative view state.
        self.render(reason="initial", force_autolevel=True)
        self.show()

        # Set up file watcher if a filepath was provided (QFileSystemWatcher uses
        # OS-native events: inotify on Linux, FSEvents on macOS, ReadDirectoryChanges on Windows)
        self._file_watcher = None
        if filepath is not None:
            self._file_watcher = Qt.QtCore.QFileSystemWatcher([str(filepath)])
            self._file_watcher.fileChanged.connect(self._on_file_changed)

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
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.operation_dock.toggleViewAction())
        view_menu.addAction(self.profile_dock.toggleViewAction())
        view_menu.addSeparator()
        reset_layout_action = QtGui.QAction("Reset layout", self)
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
            print(f"Theme warning: {result.warning}")
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
        Qt.QtCore.QTimer.singleShot(0, self._resize_default_docks)

    def reset_layout(self):
        self.profile_dock.setFloating(False)
        self.profile_dock.hide()
        if self.data.ndim == 1:
            self.profile_dock.show()
        self.operation_dock.setFloating(False)
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.operation_dock)
        self.operation_dock.show()
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.profile_dock)
        Qt.QtCore.QTimer.singleShot(0, self._resize_default_docks)

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

    def _set_view_state(self, state):
        self.view_state = state.for_shape(self.data.shape)
        self.line_plot_dimension = self.view_state.line_axis if self.view_state.line_axis is not None else 0
        self.profile_axes = tuple(axis for axis in getattr(self, "profile_axes", (self.line_plot_dimension,)) if axis < self.view_state.ndim)
        if not self.profile_axes and self.view_state.line_axis is not None:
            self.profile_axes = (self.view_state.line_axis,)
        return self.view_state

    def _image_axes(self):
        return self.view_state.image_axes or ()

    def _axis_flipped(self, axis):
        return bool(self.view_state.axis_flipped[int(axis)])

    def _sync_controls_from_view_state(self):
        if not hasattr(self, "widgets"):
            return
        for axis, spinbox in enumerate(self.widgets['spins']['slice_indices'][: self.data.ndim]):
            spinbox.blockSignals(True)
            try:
                spinbox.setMaximum(self.data.shape[axis] - 1)
                spinbox.setValue(self.view_state.slice_indices[axis])
            finally:
                spinbox.blockSignals(False)

        channel_buttons = self.widgets['buttons']['channel']
        if self.view_state.channel.value in channel_buttons:
            channel_buttons[self.view_state.channel.value].setChecked(True)
        self.widgets['buttons']['processing']['linear'].setChecked(self.view_state.scale == ScaleMode.LINEAR)
        self.widgets['buttons']['processing']['symlog'].setChecked(self.view_state.scale == ScaleMode.SYMLOG)

    def _on_slice_index_changed(self, axis, value):
        if axis >= self.view_state.ndim:
            return
        self._set_view_state(self.view_state.with_slice(axis, value))
        self.render(reason="slice")

    def _on_channel_clicked(self, name):
        self._set_view_state(self.view_state.with_channel(name))
        self._force_autolevel = True
        self._apply_channel_colormap()
        self.render(reason="channel", force_autolevel=True)

    def _on_scale_clicked(self, scale):
        self._set_view_state(self.view_state.with_scale(ScaleMode.SYMLOG if scale == "symlog" else ScaleMode.LINEAR))
        self._force_autolevel = True
        self.render(reason="scale", force_autolevel=True)

    def _set_document(self, document):
        self.operation_coordinator.set_document(document)
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        data = self.operation_evaluator.current_data()
        self.data = data
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._sync_controls_to_current_data()
        self._force_autolevel = True
        self._update_channel_controls()
        self._update_operation_dock()

    def _sync_controls_to_current_data(self):
        ndim = self.data.ndim
        self.singleton = [size == 1 for size in self.data.shape]
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self.domain = [Domain.NATIVE for _ in range(ndim)]

        if np.iscomplexobj(self.data):
            self.can_combine_as_complex = [False] * ndim
        else:
            self.can_combine_as_complex = [self.data.shape[i] == 2 for i in range(ndim)]
        self.combined_as_complex = [np.iscomplexobj(self.data) and self.data.shape[i] == 1 for i in range(ndim)]

        valid_dims = [i for i in range(ndim) if not self.singleton[i]]
        if ndim >= 1 and (self.line_plot_dimension >= ndim or self.singleton[self.line_plot_dimension]):
            self.line_plot_dimension = valid_dims[0] if valid_dims else 0
        self.profile_axes = tuple(axis for axis in getattr(self, "profile_axes", ()) if axis < ndim)
        if not self.profile_axes and ndim >= 1:
            self.profile_axes = (self.line_plot_dimension,)
        if self.profile_axes:
            self.line_plot_dimension = self.profile_axes[0]

        for i, container in enumerate(getattr(self, "dim_containers", [])):
            visible = i < ndim
            container.setVisible(visible)
            self.widgets['buttons']['primary'][i].setVisible(visible)
            self.widgets['buttons']['secondary'][i].setVisible(visible)
            self.widgets['buttons']['profile'][i].setVisible(visible)
            self.widgets['spins']['slice_indices'][i].setVisible(visible)
            if visible:
                self.widgets['labels']['dims'][i].setText(f'[{self.data.shape[i]}]')
                self.widgets['spins']['slice_indices'][i].setMaximum(self.data.shape[i] - 1)
                self.widgets['spins']['slice_indices'][i].setValue(
                    min(self.widgets['spins']['slice_indices'][i].value(), self.data.shape[i] - 1)
                )

        self.tab_widget.setTabEnabled(0, ndim >= 2)
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.line_plot_dimension)
            if ndim == 1:
                self.profile_dock.show()
                self.profile_dock.raise_()

        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_dimension_controls()
        self._sync_controls_from_view_state()

    def _update_operation_dock(self):
        if hasattr(self, "operation_dock"):
            self.operation_dock.set_operations(
                self.document.operations,
                output_shape=self.document.current_shape,
                cache_status=self.operation_evaluator.last_status,
                operation_shapes=self._operation_shapes(),
            )

    def _operation_shapes(self):
        return self.operation_coordinator.operation_shapes()

    def _replace_base_data(self, data):
        self.operation_coordinator.replace_base_data(data)
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.data = self.operation_evaluator.current_data()
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._sync_controls_to_current_data()
        self._update_channel_controls()
        self._update_operation_dock()

    def dimClicked(self, event, label, dim):
        if dim >= self.data.ndim or self.singleton[dim]:
            return
        if event.button() == Qt.QtCore.Qt.MouseButton.RightButton:
            return
    
        p = QtGui.QPalette()
        
        # If already transformed, any click returns to native
        if self.domain[dim] == Domain.FOURIER:
            # From FFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_ifft(dim)  # Undo the FFT by applying IFFT
        elif self.domain[dim] == Domain.INV_FOURIER:
            # From IFFT domain, go back to native (undo)
            self.domain[dim] = Domain.NATIVE
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('black'))
            label.setStyleSheet("font-weight: normal;")
            self._apply_fft(dim)  # Undo the IFFT by applying FFT
        elif event.button() == Qt.QtCore.Qt.MouseButton.RightButton:
            # Right click from native: apply IFFT
            self.domain[dim] = Domain.INV_FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('green'))
            label.setStyleSheet("font-weight: bold; color: green;")
            self._apply_ifft(dim)
        else:
            # Left click from native: apply FFT
            self.domain[dim] = Domain.FOURIER
            p.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor('blue'))
            label.setStyleSheet("font-weight: bold; color: blue;")
            self._apply_fft(dim)

        label.setPalette(p)
        self.update_image_view()
        self.update_line_plot()
        
    def _apply_fft(self, dim):
        """Apply forward FFT along specified dimension."""
        self._append_operation("centered_fft", dim)
        
    def _apply_ifft(self, dim):
        """Apply inverse FFT along specified dimension."""
        self._append_operation("centered_ifft", dim)

    def _show_operation_context_menu(self, pos, widget, dim):
        if dim >= self.data.ndim:
            return

        menu = QtWidgets.QMenu(self)
        for entry in operation_entries():
            action = menu.addAction(entry.label)
            action.setData(entry.id)
            action.setEnabled(self._operation_entry_enabled(entry, dim))
            action.triggered.connect(lambda checked=False, operation_id=entry.id: self._append_operation(operation_id, dim))

        menu.exec(widget.mapToGlobal(pos))

    def _operation_entry_enabled(self, entry, dim):
        if dim >= self.data.ndim:
            return False
        if entry.id in {"mean", "rss", "sum", "max", "min"} and self.data.ndim <= 1:
            return False
        if entry.id == "combine_real_imag":
            return (not np.iscomplexobj(self.data)) and self.data.shape[dim] == 2
        if entry.id == "split_complex":
            return np.iscomplexobj(self.data) and self.data.shape[dim] == 1
        return True

    def _append_operation(self, operation_id, dim=None):
        try:
            parameters = self._collect_operation_parameters(operation_id, dim)
            if parameters is None:
                return
            self.operation_coordinator.append_operation(operation_id, axis=dim, parameters=parameters)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Failed to apply operation:\n{e}")
            return

        self.render(reason="operation", force_autolevel=True)

    def _collect_operation_parameters(self, operation_id, dim):
        if operation_id != "crop":
            return {}

        axis_size = self.data.shape[dim]
        start, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Crop Axis",
            f"Start index for dim {dim}",
            0,
            0,
            axis_size,
            1,
        )
        if not ok:
            return None

        stop, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Crop Axis",
            f"Stop index for dim {dim} (exclusive)",
            axis_size,
            start,
            axis_size,
            1,
        )
        if not ok:
            return None

        return {"start": start, "stop": stop}

    def undo_last_operation(self):
        self.operation_coordinator.undo()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="operation-undo", force_autolevel=True)

    def delete_selected_operation(self, index):
        if index is None:
            return
        try:
            self.operation_coordinator.delete(index)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot delete operation:\n{e}")
            return
        self.render(reason="operation-delete", force_autolevel=True)

    def move_selected_operation(self, index, direction):
        if index is None:
            return
        try:
            self.operation_coordinator.move(index, direction)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot reorder operation:\n{e}")
            return
        self.render(reason="operation-move", force_autolevel=True)

    def reorder_operations(self, order):
        try:
            self.operation_coordinator.reorder(order)
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Cannot reorder operation stack:\n{e}")
            self._update_operation_dock()
            return False
        self.render(reason="operation-reorder", force_autolevel=True)
        return True

    def clear_operations(self):
        self.operation_coordinator.clear()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="operation-clear", force_autolevel=True)

    def save_operation_recipe(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save operation recipe",
            "arrayscope-recipe.json",
            "JSON files (*.json)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".json"):
            file_path += ".json"
        try:
            save_recipe(file_path, self.document.operations)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Save Error", f"Failed to save recipe:\n{e}")

    def load_operation_recipe(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load operation recipe",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not file_path:
            return
        try:
            operations = load_recipe(file_path, self.base_data.shape)
            self.operation_coordinator.load_operations(operations)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        try:
            self._set_document(self.operation_coordinator.document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        self.render(reason="recipe-load", force_autolevel=True)

    def materialize_current_array(self):
        self.operation_coordinator.materialize()
        self._set_document(self.operation_coordinator.document)
        self.render(reason="materialize", force_autolevel=True)

    def update_shift_indicators(self):
        for i, shift_label in enumerate(self.widgets['labels']['shift']):
            if i >= self.data.ndim:
                shift_label.setText('')
                shift_label.setToolTip('')
                shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
                continue
            if self.singleton[i]:
                shift_label.setText('')
                shift_label.setToolTip('')
                shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
                continue

            shift_label.setText('')
            shift_label.setToolTip('')
            shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
    
    def flipAxisClicked(self, event, dim):
        """Handle click on flip axis icon"""
        image_axes = self._image_axes()
        if dim not in image_axes and dim != self.view_state.line_axis:
            return
        self._set_view_state(self.view_state.with_axis_flipped(dim, not self._axis_flipped(dim)))
        self.update_flip_icons()
        self.apply_axis_flips()
        
    def update_flip_icons(self):
        image_axes = self._image_axes()
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            if i in image_axes:
                # In line plot mode, only show horizontal flip icon for the plot dimension
                if self.is_line_plot_mode():
                    if i == self.view_state.line_axis:
                        flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                        flip_label.setToolTip("Flip X axis")
                        if self._axis_flipped(i):
                            flip_label.setText('⬅️')    
                        else:
                            flip_label.setText('➡️')
                    else:
                        flip_label.setText('')  # Hide flip icons for non-plot dimensions
                        flip_label.setToolTip('')
                # In image view mode, show vertical flip for primary, horizontal for secondary
                elif self.view_state.image_axes is not None and i == self.view_state.image_axes[0]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeVerCursor))
                    flip_label.setToolTip("Flip Y")
                    if self._axis_flipped(i):
                        flip_label.setText('⬇️')
                    else:
                        flip_label.setText('⬆️')
                elif self.view_state.image_axes is not None and i == self.view_state.image_axes[1]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                    flip_label.setToolTip("Flip X")
                    if self._axis_flipped(i):
                        flip_label.setText('⬅️')
                    else:
                        flip_label.setText('➡️')
                else:
                    flip_label.setText('')  # Clear for dimensions not in primary/secondary
                    flip_label.setToolTip('')
            else:
                flip_label.setText('')  # Clear for unselected dimensions
                flip_label.setToolTip('')
    
    def apply_axis_flips(self):
        if self.is_line_plot_mode():
            plot_view = self.plot_widget.getViewBox()
            plot_dim = self.view_state.line_axis
            if plot_dim is not None:
                plot_view.invertX(self._axis_flipped(plot_dim))
        else:
            if self.view_state.image_axes is None:
                return
            
            view = self.img_view.getView()
            y_dim, x_dim = self.view_state.image_axes
            view.invertY(self._axis_flipped(y_dim))
            view.invertX(self._axis_flipped(x_dim))

    def update_complex_indicators(self):
        """Initialize or update ℝ/ℂ indicators for dimensions that can be combined as complex"""
        for i in range(self.data.ndim):
            indicator = self.widgets['labels']['complex'][i]
            
            if self.combined_as_complex[i]:
                indicator.setText('ℂ')
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Split to real')
            elif self.can_combine_as_complex[i]:
                indicator.setText('ℝ')
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Combine as complex')
            else:
                # No indicator, already-complex data or non-size-2 dimensions
                indicator.setText('')
                indicator.setToolTip('')
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))

    def _update_channel_controls(self):
        """Keep channel options in sync with the current array dtype."""
        is_complex = np.iscomplexobj(self.data)
        channel_buttons = self.widgets['buttons']['channel']
        enabled_channels = {
            'complex': is_complex,
            'real': True,
            'abs': True,
            'imag': is_complex,
            'angle': is_complex,
        }

        for name, button in channel_buttons.items():
            button.setEnabled(enabled_channels[name])

        checked_channel = self.view_state.channel.value
        if not enabled_channels.get(checked_channel, False):
            checked_channel = 'complex' if is_complex else 'real'
            self._set_view_state(self.view_state.with_channel(checked_channel))

        channel_buttons[checked_channel].setChecked(True)
    
    def complexOrRealClicked(self, event, dim):
        if self.can_combine_as_complex[dim] and not self.combined_as_complex[dim]:
            self._append_operation("combine_real_imag", dim)
        elif self.combined_as_complex[dim]:
            self._append_operation("split_complex", dim)
    
    def combineAsComplex(self, dim):
        """Combine a size-2 real dimension into complex as an operation."""
        if not self.can_combine_as_complex[dim] or self.combined_as_complex[dim]:
            return

        self._append_operation("combine_real_imag", dim)
    
    def splitToReal(self, dim):
        """Split a singleton complex dimension back to real/imag as an operation."""
        if not self.combined_as_complex[dim]:
            return

        self._append_operation("split_complex", dim)
    
    def getPixel(self, pos):
        img = self.img_view.image
        if img is None or self.view_state.image_axes is None:
            return
        container = self.img_view.getView()
        if container.sceneBoundingRect().contains(pos): 
            mousePoint = container.mapSceneToView(pos) 
            hover = image_hover_indices(self.view_state, math.floor(mousePoint.x()), math.floor(mousePoint.y()))
            if hover is not None:
                x_i, y_i = hover
                primary_axis, secondary_axis = self.view_state.image_axes
                index = list(self.view_state.slice_indices)
                index[primary_axis] = y_i
                index[secondary_axis] = x_i
                value = apply_channel(self.operation_evaluator.current_data()[tuple(index)], self.view_state.channel)
                decimal_places = getNumberOfDecimalPlaces(abs(value))
                if decimal_places > 5:
                    self.widgets['labels']['pixelValue'].setText("({}, {}) = {:.3e}".format (x_i, y_i, value))
                else:
                    self.widgets['labels']['pixelValue'].setText("({}, {}) = {:.{}f}".format (x_i, y_i, value, decimal_places))

    def _on_image_mouse_moved(self, pos):
        self.getPixel(pos)
    
    def _on_profile_marker_moved(self, image_x, image_y):
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            return
        if self.view_state.image_axes is None:
            return
        clamped = clamp_marker_position(self.view_state.shape, self.view_state.image_axes, image_x, image_y)
        if (float(clamped[0]), float(clamped[1])) != (float(image_x), float(image_y)):
            self.img_view.setProfileMarker(clamped[0], clamped[1], visible=True)
        self._pending_profile_point = (float(clamped[0]), float(clamped[1]))
        if not self._profile_timer.isActive():
            self._profile_timer.start()

    def _update_live_profile_from_pending_pos(self):
        point = self._pending_profile_point
        pos = self._pending_profile_pos
        self._pending_profile_point = None
        self._pending_profile_pos = None
        if point is None and pos is None:
            return
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            return
        if self.is_line_plot_mode():
            return

        if point is None:
            view = self.img_view.getView()
            if not view.sceneBoundingRect().contains(pos):
                self._clear_live_profile_marker()
                return
            mouse_point = view.mapSceneToView(pos)
            point = (mouse_point.x(), mouse_point.y())

        try:
            profile_render = self.profile_coordinator.render_from_marker(
                self.operation_evaluator,
                self.view_state,
                point[0],
                point[1],
                line_axis=self.view_state.line_axis,
                y_range_mode=self.profile_dock.y_range_mode(),
                image_levels=self.img_view.getLevels(),
            )
            if profile_render is None:
                self._clear_live_profile_marker()
                return
            self.profile_dock.update_line_result(profile_render.line_result, profile_render.view_state, y_range=profile_render.y_range)
            self._update_operation_dock()
            self.profile_dock.show()
            self.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
        except Exception as e:
            print(f"Live profile update failed: {e}")
            self._clear_live_profile_marker()

    def _clear_live_profile_marker(self):
        if hasattr(self, "img_view"):
            self.img_view.hideProfileMarker()

    def _on_live_profile_toggled(self, enabled):
        if enabled and hasattr(self, "profile_dock"):
            if not self.profile_dock.isVisible():
                self.profile_dock.setFloating(True)
                self.profile_dock.resize(560, 260)
            self.profile_dock.show()
            self.profile_dock.raise_()
            self.img_view.getView().setCursor(Qt.QtCore.Qt.CursorShape.CrossCursor)
            self._ensure_profile_marker()
        if not enabled:
            self._pending_profile_pos = None
            self._pending_profile_point = None
            self._profile_timer.stop()
            self._clear_live_profile_marker()
            self.img_view.getView().unsetCursor()
            self.update_line_plot()

    def _on_profile_dock_visibility_changed(self, visible):
        if not visible and self.widgets['buttons']['display']['live_profile'].isChecked():
            self.widgets['buttons']['display']['live_profile'].setChecked(False)
        elif visible and not self.profile_dock.isFloating():
            Qt.QtCore.QTimer.singleShot(0, self._resize_profile_dock_default)

    def _resize_profile_dock_default(self):
        if not hasattr(self, "profile_dock") or not self.profile_dock.isVisible() or self.profile_dock.isFloating():
            return
        target_height = max(140, int(self.height() * 0.23))
        try:
            self.resizeDocks([self.profile_dock], [target_height], Qt.QtCore.Qt.Orientation.Vertical)
        except Exception:
            pass

    def _ensure_profile_marker(self):
        position = self.img_view.profileMarkerPosition()
        if position is None:
            x, y = self._default_profile_marker_position()
            self.img_view.setProfileMarker(x, y, visible=True)
            self._on_profile_marker_moved(x, y)
        else:
            self._on_profile_marker_moved(*position)

    def _default_profile_marker_position(self):
        if self.view_state.image_axes is None:
            return (0, 0)
        primary_axis, secondary_axis = self.view_state.image_axes
        x = (self.view_state.shape[secondary_axis] - 1) / 2.0
        y = (self.view_state.shape[primary_axis] - 1) / 2.0
        return (round(x), round(y))

    def _current_profile_y_range(self):
        if not hasattr(self, "profile_dock"):
            return None
        try:
            image_levels = self.img_view.getLevels()
        except Exception:
            image_levels = None
        return profile_y_range(self.profile_dock.y_range_mode(), image_levels)

    def _phase_colormap(self):
        return phase_colormap()

    def _apply_channel_colormap(self):
        if self.view_state.channel in (ChannelMode.COMPLEX, ChannelMode.ANGLE):
            self.img_view.setColorMap(self._phase_colormap())
        else:
            self.img_view.setColorMap(gray_colormap())

    def update_image_view(self, *, force_autolevel: bool = False):
        if self.view_state.image_axes is None: # No image view for 1D data
            return
            
        al = True
        force_auto = force_autolevel or getattr(self, '_force_autolevel', False)
        window_mode = self._current_window_mode()
        # reset the one-shot flag after using it
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        
        previous_levels = None
        previous_bounds = None
        if not force_auto and getattr(self.img_view, "image", None) is not None:
            try:
                previous_levels = self.img_view.getLevels()
                previous_bounds = self.img_view.getHistogramDataBounds()
            except Exception:
                previous_levels = None
                previous_bounds = None

        
        try:
            colormap_lut = None
            if self.view_state.channel == ChannelMode.COMPLEX:
                colormap_lut = self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
            display_image = self.operation_evaluator.image(self.view_state, colormap_lut=colormap_lut)
            current_bounds = self._display_histogram_bounds(display_image)
            level_decision = choose_window_levels(
                mode=window_mode,
                previous_levels=previous_levels,
                previous_bounds=previous_bounds,
                current_bounds=current_bounds,
                default_levels=display_image.default_levels,
                force_auto=force_auto,
            )
            al = level_decision.auto_levels
            levels = level_decision.levels

            if display_image.histogram_data is not None:
                self.img_view.setImage(
                    display_image.data,
                    autoLevels=al,
                    levels=levels,
                    histogramData=display_image.histogram_data,
                )
            else:
                self.img_view.setImage(display_image.data, autoLevels=al, levels=levels)
            self._update_operation_dock()
            
            # Apply axis flips after setting the image
            self.apply_axis_flips()
            
        except Exception as e:
            print(f'Image update failed: {e}')

    def _current_window_mode(self):
        if self.widgets['buttons']['display']['window_absolute'].isChecked():
            return "absolute"
        return "relative"

    def _display_histogram_bounds(self, display_image):
        data = display_image.histogram_data
        if data is None:
            data = display_image.data
        try:
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                return (float(np.min(finite_data)), float(np.max(finite_data)))
        except Exception:
            return None
        return None
    
    def update_display_mode(self):
        """Update the display mode for the image view"""
        if self.widgets['buttons']['display']['square_pixels'].isChecked():
            self.img_view.setDisplayMode('square_pixels')
        elif self.widgets['buttons']['display']['square_fov'].isChecked():
            self.img_view.setDisplayMode('square_fov')
        elif self.widgets['buttons']['display']['fit'].isChecked():
            self.img_view.setDisplayMode('fit')

    def _processing_pressed(self, btn):
        """Called on processing button press; if the button is already checked
        the user is re-clicking it and we should force an auto-level on next update."""
        try:
            if btn.isChecked():
                self._force_autolevel = True
            else:
                self._force_autolevel = False
        except Exception:
            self._force_autolevel = False

        # Update the display group title
        self._update_display_group_title()

        # Force update to ensure view changes immediately
        self.render(reason="processing-pressed", force_autolevel=True)

    def _update_display_group_title(self):
        """Update the display group title with aspect ratio information."""
        mode = self.img_view.displayMode
        aspect_str = ''
        
        if mode == 'square_pixels': # Simple
            self.display_group.setTitle('Display (1:1)')
            return
        
        if mode == 'fit': #use the viewport aspect ratio
            aspect_str = ''
            try:
                if hasattr(self.img_view, 'image') and self.img_view.image is not None:
                    view = self.img_view.getView()
                    
                    img_height, img_width = self.img_view.image.shape[:2]
                    widget_ratio = view.size().width() / view.size().height()
                    img_ratio = img_width / img_height
                    ratio = img_ratio * widget_ratio
                    
                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
            finally:
                self.display_group.setTitle(f'Display {aspect_str}')
            
        elif mode == 'square_fov':
            # For square FOV, use the image aspect ratio
            if hasattr(self.img_view, 'image') and self.img_view.image is not None:
                shape = self.img_view.image.shape
                if len(shape) >= 2:
                    height, width = shape[:2]
                    ratio = width / height
                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
                else:
                    aspect_str = ''
            self.display_group.setTitle(f'Display {aspect_str}')
        else:
            self.display_group.setTitle('Display')
    
    def update_line_plot(self):
        if not hasattr(self, "profile_dock") or not self.profile_dock.isVisible():
            return
        if self.widgets['buttons']['display']['live_profile'].isChecked():
            position = self.img_view.profileMarkerPosition()
            if position is not None:
                self._on_profile_marker_moved(*position)
                return
        line_result = self.operation_evaluator.line(self.view_state)
        self.profile_dock.update_line_result(line_result, self.view_state, y_range=self._current_profile_y_range())
        self._update_operation_dock()

    def _on_view_range_changed(self):
        """Update display group title when view range changes (for fit mode)."""
        self._update_display_group_title()

    def on_tab_changed(self, index):
        """Handle central image tab changes."""
        self.update_dimension_controls()
        self.update()
        self.line_plot.hide_crosshair()
    
    def is_line_plot_mode(self):
        """The historical line-plot tab is no longer the primary plot surface."""
        return False

    def set_profile_axis(self, axis):
        self._set_view_state(self.view_state.with_line_axis(int(axis)))
        self.profile_axes = (self.view_state.line_axis,)
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        self.render(reason="profile-axis")

    def set_dimension_role(self, role, axis):
        if axis >= self.data.ndim:
            return
        if role == "p":
            self._set_view_state(self.view_state.with_line_axis(int(axis)))
            self.profile_axes = (self.view_state.line_axis,)
            if hasattr(self, "profile_dock"):
                self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        elif role in ("y", "x"):
            if self.view_state.image_axes is None:
                return
            self._set_view_state(self.view_state.with_image_axis(role, axis))
        self.render(reason=f"dimension-{role}")

    def transposeView(self, event):
        if self.view_state.image_axes is None:
            return
        self._set_view_state(self.view_state.transposed_image_axes())
        self.render(reason="transpose")

    def render(self, *, reason: str = "state", force_autolevel: bool = False):
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._sync_controls_from_view_state()
        self._update_channel_controls()
        self.update_dimension_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_image_view(force_autolevel=force_autolevel)
        self.update_line_plot()
        self._update_operation_dock()

    def update(self):
        self.render(reason="legacy-update")

    def update_dimension_controls(self):
        """Update button and spinbox states based on current mode"""
        if self.is_line_plot_mode():
            # Line plot mode: single dimension selection, all other spinboxes enabled
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                bProfile = self.widgets['buttons']['profile'][i]
                if i >= self.data.ndim:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(False)
                    continue
                
                if self.singleton[i]:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(False)
                elif i == self.line_plot_dimension:
                    # This is the dimension we're plotting along
                    w.setEnabled(False)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(True)
                    bPrim.setChecked(True)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(i in self.profile_axes)
                else:
                    # All other dimensions: enable spinbox to select slice
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(True)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(i in self.profile_axes)
        else:
            image_axes = self._image_axes()
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                bProfile = self.widgets['buttons']['profile'][i]
                if i >= self.data.ndim:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(False)
                    continue
                if self.singleton[i] == True:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bProfile.setChecked(False)
                elif i in image_axes:
                    w.setEnabled(False)
                    if self.view_state.image_axes is not None and i == self.view_state.image_axes[0]:
                        bPrim.setChecked(True)
                        bSecondary.setChecked(False)
                    else:
                        bPrim.setChecked(False)
                        bSecondary.setChecked(True)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(True)
                    bProfile.setChecked(i in self.profile_axes)
                else:
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(True)
                    bProfile.setEnabled(True)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(i in self.profile_axes)
                    
            if self.view_state.image_axes is not None:
                self.widgets['buttons']['primary'][self.view_state.image_axes[0]].setChecked(True)
                self.widgets['buttons']['secondary'][self.view_state.image_axes[1]].setChecked(True)
        
        self.update_flip_icons()
        self.update_shift_indicators()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Check for 'T' key to transpose view (swap X and Y dimensions)
        if key == Qt.QtCore.Qt.Key.Key_T and modifiers == Qt.QtCore.Qt.KeyboardModifier.NoModifier:
            if not self.is_line_plot_mode() and self.view_state.image_axes is not None:
                self.transposeView(event)
                event.accept()
                return
        
        # Check CTRL+number for colormap changes
        if modifiers == Qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            if key == Qt.QtCore.Qt.Key.Key_1:
                self.setColormap('gray')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_2:
                self.setColormap('viridis')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_3:
                self.setColormap('plasma')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_4:
                self.setColormap('PAL-relaxed')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_5:
                self.setColormap('cividis')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_6:
                self.setColormap('CET-CBL1')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_7:
                self.setColormap('d3-cool')
                event.accept()
                return
            elif key == Qt.QtCore.Qt.Key.Key_8:
                self.setColormap('d3-warm')
                event.accept()
                return
                
        # Pass event to parent if not handled
        super().keyPressEvent(event)
    
    def setColormap(self, colormap_name):
        """Set the colormap for the image view"""
        try:
            colormap = named_colormap(colormap_name)
            if colormap is None:
                print(f"Unknown colormap: {colormap_name}")
                return

            # Apply colormap to the image view
            self.img_view.setColorMap(colormap)
            #self.current_colormap = colormap_name
            
        except Exception as e:
            print(f"Failed to set colormap {colormap_name}: {e}")
    
    def eventFilter(self, obj, event):
        if obj == self.tab_widget.tabBar():
            if event.type() == Qt.QtCore.QEvent.Type.MouseButtonDblClick:
                self.profile_dock.toggle_style()
                event.accept()
                return True
        return super().eventFilter(obj, event)
    
    def _setup_export_context_menus(self):
        for i in range(self.data.ndim):
            prim_btn = self.widgets['buttons']['primary'][i]
            sec_btn = self.widgets['buttons']['secondary'][i]
            
            prim_btn.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            sec_btn.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            
            prim_btn.customContextMenuRequested.connect(lambda pos, btn=prim_btn, dim=i: self._show_export_context_menu(pos, btn, dim))
            sec_btn.customContextMenuRequested.connect( lambda pos, btn=sec_btn,  dim=i: self._show_export_context_menu(pos, btn, dim))
    
    def _show_export_context_menu(self, pos, btn, dim):
        menu = QtWidgets.QMenu(self)

        operations_menu = menu.addMenu("Operations")
        for entry in operation_entries():
            action = operations_menu.addAction(entry.label)
            action.setEnabled(self._operation_entry_enabled(entry, dim))
            action.triggered.connect(lambda checked=False, operation_id=entry.id: self._append_operation(operation_id, dim))

        menu.addSeparator()
        
        export_action = menu.addAction("Export along this dimension...")
        export_action.triggered.connect(lambda: self._start_export(dim))
        
        # Show menu at cursor position
        menu.exec(btn.mapToGlobal(pos))
    
    def _start_export(self, export_dim):
        """Initiate video export workflow"""
        if self.singleton[export_dim]:
            QtWidgets.QMessageBox.warning(self, "Cannot Export", 
                f"Dimension {export_dim} has size 1 and cannot be exported.")
            return
        
        # Get export settings from dialog
        settings_dialog = VideoExportSettingsDialog(parent=self, export_dim=export_dim, data_shape=self.data.shape)
        if settings_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        
        settings = settings_dialog.get_settings()
        
        # Get file save path (for PNG frames we ask for a directory)
        file_path = None
        if settings['format'] == 'png':
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Export frames to directory", os.path.expanduser("~")
            )
            if not dir_path:
                return
            file_path = dir_path
        else:
            file_filter = f"{settings['format'].upper()} files (*.{settings['format']})"
            file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, f"Export Video as {settings['format'].upper()}", 
                f"export.{settings['format']}", file_filter
            )
            if not file_path:
                return
        
        # Capture display mode and widget aspect ratio
        display_mode = getattr(self.img_view, 'displayMode', 'square_pixels')
        view = self.img_view.getView()
        widget_ratio = view.size().width() / view.size().height() if view.size().height() != 0 else 1.0
        
        # Capture current display levels for consistent frame scaling
        levels = None
        try:
            levels = tuple(self.img_view.getLevels())
        except Exception:
            levels = None

        # Capture current color map LUT (256x3) to apply in export
        lut = None
        try:
            if hasattr(self.img_view, 'histogram') and hasattr(self.img_view.histogram, 'gradient'):
                cm = self.img_view.histogram.gradient.colorMap()
                if cm is not None:
                    lut = cm.getLookupTable(0.0, 1.0, 256, alpha=False)
        except Exception:
            lut = None
        
        # Create worker thread
        worker = VideoExportWorker(
            data=self.data,
            view_state=self.view_state,
            export_dim=export_dim,
            output_path=file_path,
            fps=settings['fps'],
            format_type=settings['format'],
            window_level_mode=settings.get('window_level', 'displayed'),
            levels=levels,
            pixel_ratio_mode=settings.get('pixel_ratio', 'square_pixels'),
            display_mode=display_mode,
            widget_ratio=widget_ratio,
            colormap_lut=lut
        )
        
        # Show progress dialog
        progress_dialog = VideoExportDialog(self)
        progress_dialog.start_export(worker, self.data.shape[export_dim])

    def _save_current_numpy_file(self):
        """Save the currently displayed array state to a NumPy .npy file."""
        range_dialog = SaveRangeDialog(self, self.data.shape)
        if range_dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        ranges = range_dialog.get_ranges()
        sliced_data = self.data[tuple(slice(start, stop) for start, stop in ranges)]
        output_data = np.squeeze(sliced_data) if range_dialog.should_squeeze() else sliced_data

        default_name = 'arrayscope.npy'
        if self._filepath is not None:
            source_path = Path(self._filepath)
            source_name = source_path.name
            if source_name.lower().endswith('.nii.gz'):
                default_name = source_name[:-7] + '.npy'
            else:
                default_name = f"{source_path.stem}.npy"

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current array as NumPy file",
            default_name,
            "NumPy files (*.npy)"
        )
        if not file_path:
            return

        if not file_path.lower().endswith('.npy'):
            file_path += '.npy'

        try:
            np.save(file_path, output_data)
            print(f"Saved array {list(output_data.shape)} to {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Failed to save NumPy file:\n{e}")

    def _on_file_changed(self, path):
        """Called by QFileSystemWatcher when the source file changes on disk."""
        self._reload_btn.setText("⚠️")
        self._reload_btn.setToolTip("File changed — click to reload")
        # Re-add the path: handles atomic replacement where the original inode disappears
        if self._file_watcher:
            self._file_watcher.addPath(path)

    def _reload_file(self):
        """Reload data from the source file, preserving slice positions where possible."""
        if self._filepath is None:
            return
        try:
            new_data = None
            new_dataset_path = self._dataset_path

            if self._selector_class_name is None:
                from .file_interpreters import load_file
                new_data = load_file(self._filepath)
            else:
                from .selectors import H5DatasetSelector, NpzDatasetSelector, MatDatasetSelector
                selector_map = {
                    'H5DatasetSelector': H5DatasetSelector,
                    'NpzDatasetSelector': NpzDatasetSelector,
                    'MatDatasetSelector': MatDatasetSelector,
                }
                selector_cls = selector_map.get(self._selector_class_name)
                if selector_cls is None:
                    return
                selector = selector_cls(self._filepath)
                compatible_keys = {d[0] for d in selector.compatible_datasets}
                if self._dataset_path is not None and self._dataset_path in compatible_keys:
                    new_data = selector.load_data(self._dataset_path)
                    selector.close()
                elif selector.requires_gui():
                    selected = selector.show()
                    if selected is None:
                        selector.close()
                        return  # User cancelled — keep ⚠️ visible
                    new_data = selector.load_data(selected)
                    new_dataset_path = selected
                    selector.close()
                else:
                    result = selector.get_single_data()
                    selector.close()
                    if result is None:
                        return
                    new_dataset_path, new_data = result

            if new_data is None:
                return

            self._dataset_path = new_dataset_path
            self._reset_data(new_data)
            self._reload_btn.setText("⟳")
            self._reload_btn.setToolTip("Reload file")
            if self._file_watcher and self._filepath:
                self._file_watcher.addPath(str(self._filepath))
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Reload Error", f"Failed to reload:\n{e}")

    def _reset_data(self, new_data):
        """Replace the displayed data, clamping slice positions to the new shape."""
        old_ndim = self.data.ndim
        new_ndim = new_data.ndim

        if new_ndim != old_ndim:
            # Per-dimension widgets were built for old_ndim and cannot be rebuilt in-place.
            # Open a fresh window with the new data and close this one.
            win = ArrayScopeWindow(new_data,
                                filepath=self._filepath,
                                dataset_path=self._dataset_path,
                                selector_class_name=self._selector_class_name)
            win.setWindowTitle(self.windowTitle())
            win.show()
            self.close()
            return

        self._replace_base_data(new_data)
        self.singleton = [e == 1 for e in new_data.shape]

        if np.iscomplexobj(new_data):
            self.can_combine_as_complex = [False] * new_ndim
        else:
            self.can_combine_as_complex = [new_data.shape[i] == 2 for i in range(new_ndim)]
        self.combined_as_complex = [False] * new_ndim

        # Reset FFT domain state and dim label styling
        self.domain = [Domain.NATIVE for _ in range(new_ndim)]
        for i, label in enumerate(self.widgets['labels']['dims']):
            label.setStyleSheet(self.DIMENSION_LABEL_STYLE)
            label.setText(f'[{new_data.shape[i]}]')

        # Update spinbox maximums (auto-clamps current value)
        for i in range(new_ndim):
            self.widgets['spins']['slice_indices'][i].setMaximum(new_data.shape[i] - 1)

        self._set_view_state(self.view_state.for_shape(new_data.shape, preserve_flags=True))
        self._update_channel_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_dimension_controls()
        self._force_autolevel = True
        self.render(reason="reload", force_autolevel=True)
