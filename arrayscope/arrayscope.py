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
from .operation_evaluator import OperationEvaluator
from .operation_pipeline import ArrayDocument
from .operation_recipes import load_recipe, save_recipe
from .operation_registry import create_operation, operation_entries
from .profile import clamp_marker_position, profile_state_from_image_hover, profile_y_range
from .profile_dock import ProfileDock
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

        self.base_data = data
        self.document = ArrayDocument(self.base_data)
        self.operation_evaluator = OperationEvaluator(self.document)
        self.data = self.operation_evaluator.current_data()
        self.singleton = [e == 1 for e in list(data.shape)]
        self.selected_indices = []
        self.channel = None
        self.scale = None
        self._force_autolevel = False
        self._filepath = filepath
        self._dataset_path = dataset_path
        self._selector_class_name = selector_class_name
        
        self.axis_flipped = [False] * data.ndim  # Track flip state so that one can toggle dims and come back to the same flip state
        self.fftshifted = [False] * data.ndim
        
        # If data is real-valued and has size-2 dimensions, arrayscope can combine them as complex (ISMRMD uses this for real/imag parts)
        if np.iscomplexobj(data):
            self.can_combine_as_complex = [False] * data.ndim
        else:
            self.can_combine_as_complex = [data.shape[i] == 2 for i in range(data.ndim)]
        self.combined_as_complex = [np.iscomplexobj(data) and data.shape[i] == 1 for i in range(data.ndim)]
        
        # Store complex_dim for later use (after widgets are created)
        self._initial_complex_dim = complex_dim
        
        for dim in range(0,data.ndim):
            if self.singleton[dim] is False and len(self.selected_indices) < 2:
                self.selected_indices.append(dim)
        # For 1D arrays, we only need one dimension; for 2D+, ensure we have two
        if len(self.selected_indices) < 2 and data.ndim >= 2:
            self.selected_indices = [0, 1]
        elif len(self.selected_indices) == 0:
            # Edge case: if all dimensions are singleton, pick first one
            self.selected_indices = [0]
        
        # Line plot mode uses a single selected dimension
        self.line_plot_dimension = 0  # Default to first non-singleton dimension
        for dim in range(data.ndim):
            if not self.singleton[dim]:
                self.line_plot_dimension = dim
                break
        self.profile_axes = (self.line_plot_dimension,)

        self.view_state = self._make_view_state(slice_indices=[0] * data.ndim)
                
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
            w.valueChanged.connect(self.update)
        
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
            btn.clicked.connect(self.update)
        
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
            btn.clicked.connect(self.update)
        
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
        self.img_view.setProfileMarkerCallback(self._on_profile_marker_moved)
        
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
        )
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.operation_dock)
        self._update_operation_dock()
        
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
        
        # Initialize dimension controls based on data dimensions
        if len(self.selected_indices) >= 1:
            self.changedIndex(True, 0, self.selected_indices[0], update=False)
        if len(self.selected_indices) >= 2:
            self.changedIndex(True, 1, self.selected_indices[1], update=False)
        self.update_dimension_controls()  # Initialize dimension controls properly
        self.update()
        self.show()

        # Set up file watcher if a filepath was provided (QFileSystemWatcher uses
        # OS-native events: inotify on Linux, FSEvents on macOS, ReadDirectoryChanges on Windows)
        self._file_watcher = None
        if filepath is not None:
            self._file_watcher = Qt.QtCore.QFileSystemWatcher([str(filepath)])
            self._file_watcher.fileChanged.connect(self._on_file_changed)




    def _make_view_state(self, slice_indices=None):
        if slice_indices is None:
            slice_indices = self._current_slice_indices()
        else:
            slice_indices = tuple(
                max(0, min(int(index), self.data.shape[axis] - 1))
                for axis, index in enumerate(slice_indices)
            )

        image_axes = None
        if self.data.ndim >= 2 and len(self.selected_indices) >= 2:
            image_axes = (self.selected_indices[0], self.selected_indices[1])

        return ViewState(
            ndim=self.data.ndim,
            shape=tuple(self.data.shape),
            image_axes=image_axes,
            line_axis=self.line_plot_dimension if self.data.ndim >= 1 else None,
            slice_indices=slice_indices,
            channel=self._current_channel_mode(),
            scale=self._current_scale_mode(),
            axis_flipped=tuple(self.axis_flipped),
            axis_fftshifted=tuple(self.fftshifted),
        )

    def _set_document(self, document):
        evaluator = OperationEvaluator(document)
        data = evaluator.current_data()
        self.document = document
        self.operation_evaluator = evaluator
        self.data = data
        self._sync_controls_to_current_data()
        self._force_autolevel = True
        self._update_channel_controls()
        self._update_operation_dock()

    def _sync_controls_to_current_data(self):
        ndim = self.data.ndim
        self.singleton = [size == 1 for size in self.data.shape]
        if len(self.axis_flipped) != ndim:
            self.axis_flipped = [False] * ndim
        if len(self.fftshifted) != ndim:
            self.fftshifted = [False] * ndim
        self.domain = [Domain.NATIVE for _ in range(ndim)]

        if np.iscomplexobj(self.data):
            self.can_combine_as_complex = [False] * ndim
        else:
            self.can_combine_as_complex = [self.data.shape[i] == 2 for i in range(ndim)]
        self.combined_as_complex = [np.iscomplexobj(self.data) and self.data.shape[i] == 1 for i in range(ndim)]

        valid_dims = [i for i in range(ndim) if not self.singleton[i]]
        if ndim == 1:
            self.selected_indices = [0]
        elif ndim >= 2:
            selected = [i for i in self.selected_indices if i < ndim and not self.singleton[i]]
            for dim in valid_dims + list(range(ndim)):
                if len(selected) >= 2:
                    break
                if dim not in selected:
                    selected.append(dim)
            self.selected_indices = selected[:2]
        else:
            self.selected_indices = []

        if ndim >= 1:
            if self.line_plot_dimension >= ndim or self.singleton[self.line_plot_dimension]:
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

    def _update_operation_dock(self):
        if hasattr(self, "operation_dock"):
            self.operation_dock.set_operations(self.document.operations)

    def _replace_base_data(self, data):
        self.base_data = data
        self.document = ArrayDocument(self.base_data)
        self.operation_evaluator = OperationEvaluator(self.document)
        self.data = self.operation_evaluator.current_data()
        self._sync_controls_to_current_data()
        self._update_channel_controls()
        self._update_operation_dock()

    def _sync_view_state_from_window(self):
        self.view_state = self._make_view_state()
        return self.view_state

    def _current_slice_indices(self):
        if not hasattr(self, 'widgets'):
            return (0,) * self.data.ndim

        indices = []
        for axis, spinbox in enumerate(self.widgets['spins']['slice_indices'][: self.data.ndim]):
            indices.append(max(0, min(spinbox.value(), self.data.shape[axis] - 1)))
        return tuple(indices)

    def _current_channel_mode(self):
        if hasattr(self, 'widgets'):
            channel_buttons = self.widgets['buttons']['channel']
            for name, button in channel_buttons.items():
                if button.isChecked():
                    return ChannelMode(name)

        if self.channel is not None:
            return ChannelMode(self.channel)
        return ChannelMode.REAL

    def _current_scale_mode(self):
        if hasattr(self, 'widgets') and self.widgets['buttons']['processing']['symlog'].isChecked():
            return ScaleMode.SYMLOG
        if self.scale is not None:
            return ScaleMode(self.scale)
        return ScaleMode.LINEAR

    
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
            operation = create_operation(operation_id, axis=dim, parameters=parameters)
            new_document = self.document.with_operation(operation)
            if len(new_document.current_shape) < 1:
                raise ValueError("operation would produce a scalar, which this viewer cannot display yet")
            self._set_document(new_document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Operation Error", f"Failed to apply operation:\n{e}")
            return

        self.update()

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
        self._set_document(self.document.without_last_operation())
        self.update()

    def clear_operations(self):
        self._set_document(ArrayDocument(self.base_data))
        self.update()

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
            document = ArrayDocument(self.base_data, operations=operations)
            if len(document.current_shape) < 1:
                raise ValueError("recipe produces a scalar, which this viewer cannot display yet")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        try:
            self._set_document(document)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Recipe Load Error", f"Failed to load recipe:\n{e}")
            return
        self.update()

    def materialize_current_array(self):
        self.base_data = np.array(self.operation_evaluator.current_data(), copy=True)
        self._set_document(ArrayDocument(self.base_data))
        self.update()

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
        # Only respond if this dimension is currently selected
        if dim not in self.selected_indices:
            return
        
        # Toggle flip
        self.axis_flipped[dim] = not self.axis_flipped[dim]
        
        self.update_flip_icons()
        self._sync_view_state_from_window()
        self.apply_axis_flips()
        
    def update_flip_icons(self):
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            if i in self.selected_indices:
                # In line plot mode, only show horizontal flip icon for the plot dimension
                if self.is_line_plot_mode():
                    if i == self.line_plot_dimension:
                        flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                        flip_label.setToolTip("Flip X axis")
                        if self.axis_flipped[i]:
                            flip_label.setText('⬅️')    
                        else:
                            flip_label.setText('➡️')
                    else:
                        flip_label.setText('')  # Hide flip icons for non-plot dimensions
                        flip_label.setToolTip('')
                # In image view mode, show vertical flip for primary, horizontal for secondary
                elif i == self.selected_indices[0]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeVerCursor))
                    flip_label.setToolTip("Flip Y")
                    if self.axis_flipped[i]:
                        flip_label.setText('⬇️')
                    else:
                        flip_label.setText('⬆️')
                elif len(self.selected_indices) > 1 and i == self.selected_indices[1]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                    flip_label.setToolTip("Flip X")
                    if self.axis_flipped[i]:
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
            plot_dim = self.line_plot_dimension
            plot_view.invertX(self.axis_flipped[plot_dim])
        else:
            if self.data.ndim == 1: # Shouldn't ever be in view mode for 1D data
                return
            
            view = self.img_view.getView()
            
            y_dim = self.selected_indices[0]
            view.invertY(self.axis_flipped[y_dim])

            if len(self.selected_indices) > 1:
                x_dim = self.selected_indices[1]
                view.invertX(self.axis_flipped[x_dim])

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

        checked_channel = next(
            (name for name, button in channel_buttons.items() if button.isChecked()),
            None,
        )

        for name, button in channel_buttons.items():
            button.setEnabled(enabled_channels[name])

        if checked_channel not in enabled_channels or not enabled_channels[checked_channel]:
            checked_channel = 'complex' if is_complex else 'real'

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
        container = self.img_view.getView()
        if container.sceneBoundingRect().contains(pos): 
            mousePoint = container.mapSceneToView(pos) 
            x_i = math.floor(mousePoint.x()) 
            y_i = math.floor(mousePoint.y()) 
            if x_i >= 0 and x_i < img.shape [ 0 ] and y_i >= 0 and y_i < img.shape[1]:
                if img.ndim == 3 and img.shape[-1] in (3, 4):
                    self.widgets['labels']['pixelValue'].setText(f"({x_i}, {y_i}) RGB = {img[x_i, y_i, :3].tolist()}")
                    return
                decimal_places = getNumberOfDecimalPlaces(abs(img[x_i ,y_i]))
                if decimal_places > 5:
                    self.widgets['labels']['pixelValue'].setText("({}, {}) = {:.3e}".format (x_i, y_i, img[x_i ,y_i]))
                else:
                    self.widgets['labels']['pixelValue'].setText("({}, {}) = {:.{}f}".format (x_i, y_i, img[x_i ,y_i], decimal_places))

    def _on_image_mouse_moved(self, pos):
        self.getPixel(pos)
    
    def _on_profile_marker_moved(self, image_x, image_y):
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            return
        self._sync_view_state_from_window()
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
            self._sync_view_state_from_window()
            profile_state = profile_state_from_image_hover(
                self.view_state,
                point[0],
                point[1],
                line_axis=self.line_plot_dimension,
            )
            if profile_state is None:
                self._clear_live_profile_marker()
                return

            line_result = self.operation_evaluator.line(profile_state)
            self.profile_dock.update_line_result(line_result, profile_state, y_range=self._current_profile_y_range())
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
        self._sync_view_state_from_window()
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
        if self.channel in ('complex', 'angle'):
            self.img_view.setColorMap(self._phase_colormap())
        else:
            self.img_view.setColorMap(gray_colormap())

    def update_image_view(self):
        if self.data.ndim == 1: # No image view for 1D data
            return
            
        al = True
        old_channel = self.channel
        oldscale = self.scale
        
        if self.widgets['buttons']['channel']['complex'].isChecked():
            self.channel = 'complex'
        elif self.widgets['buttons']['channel']['abs'].isChecked():
            self.channel = 'abs'
        elif self.widgets['buttons']['channel']['angle'].isChecked():
            self.channel = 'angle'
        elif self.widgets['buttons']['channel']['real'].isChecked():
            self.channel = 'real'
        elif self.widgets['buttons']['channel']['imag'].isChecked():
            self.channel = 'imag'
        
        if self.widgets['buttons']['processing']['symlog'].isChecked():
            self.scale = 'symlog'
        else:
            self.scale = None
        
        changed_channel = old_channel != self.channel
        changed_scale = oldscale != self.scale
        if changed_channel:
            self._apply_channel_colormap()
        force_auto = changed_scale or changed_channel or getattr(self, '_force_autolevel', False)
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
            self._sync_view_state_from_window()
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
        self.update_image_view()

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
        self._sync_view_state_from_window()
        line_result = self.operation_evaluator.line(self.view_state)
        self.profile_dock.update_line_result(line_result, self.view_state, y_range=self._current_profile_y_range())

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
        self.line_plot_dimension = int(axis)
        self.profile_axes = (self.line_plot_dimension,)
        self._sync_view_state_from_window()
        self.update_dimension_controls()
        self.update_line_plot()

    def set_dimension_role(self, role, axis):
        if axis >= self.data.ndim:
            return
        if role == "p":
            self.profile_axes = (int(axis),)
            self.line_plot_dimension = self.profile_axes[0]
            if hasattr(self, "profile_dock"):
                self.profile_dock.set_axes(self.data.shape, self.line_plot_dimension)
        elif role in ("y", "x"):
            if len(self.selected_indices) < 2:
                return
            roles = DimensionRoles.from_axes(tuple(self.selected_indices[:2]), self.profile_axes)
            roles = roles.with_image_axis(role, axis)
            self.selected_indices = list(roles.image_axes)
        self.update_dimension_controls()
        self._sync_view_state_from_window()
        self.update()

    def transposeView(self, event):
        old_primary = self.selected_indices[0]
        old_secondary = self.selected_indices[1]

        # Swap the flip states along with the dimensions (prevents the axes from appearing flipped after transpose)
        self.axis_flipped[old_primary], self.axis_flipped[old_secondary] = self.axis_flipped[old_secondary], self.axis_flipped[old_primary]
        
        self.changedIndex(event, 0, old_secondary, update=False)
        self.changedIndex(event, 1, old_primary, update=True)

    def update(self):
        self.update_slice()
        self.update_image_view()
        self.update_line_plot()
        self._sync_view_state_from_window()

    def update_slice(self):
        """Update slice for image view mode"""
        self.slice = [slice(None)] * self.data.ndim
        for dim in range(0, self.data.ndim):
            if dim in self.selected_indices:
                self.slice[dim] = slice(None) # pass?
            else:
                val = self.widgets['spins']['slice_indices'][dim].value()
                self.slice[dim] = slice(val, val+1)

    def changedIndex(self, checked, which_one, idx, update=True):
        if self.is_line_plot_mode():
            # In line plot mode, only use primary button to select the dimension to plot along
            if which_one == 0:  # Primary button clicked
                self.line_plot_dimension = idx
        else:
            # In image view mode, use the original two-dimension selection
            # For 1D arrays, we don't have a second dimension
            if which_one < len(self.selected_indices):
                self.selected_indices[which_one] = idx
        
        self.update_dimension_controls()
        self._sync_view_state_from_window()
        if update is True:
            self.update()
    
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
            # Image view mode: original two-dimension selection behavior
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
                elif i in self.selected_indices:
                    w.setEnabled(False)
                    if i == self.selected_indices[0]:
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
                    
            # Only set buttons if we have at least 2 selected indices
            if len(self.selected_indices) >= 1:
                self.widgets['buttons']['primary'][self.selected_indices[0]].setChecked(True)
            if len(self.selected_indices) >= 2:
                self.widgets['buttons']['secondary'][self.selected_indices[1]].setChecked(True)
        
        self.update_flip_icons()
        self.update_shift_indicators()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Check for 'T' key to transpose view (swap X and Y dimensions)
        if key == Qt.QtCore.Qt.Key.Key_T and modifiers == Qt.QtCore.Qt.KeyboardModifier.NoModifier:
            if not self.is_line_plot_mode() and len(self.selected_indices) >= 2:
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
        self._sync_view_state_from_window()
        worker = VideoExportWorker(
            data=self.data,
            view_state=self.view_state,
            export_dim=export_dim,
            output_path=file_path,
            fps=settings['fps'],
            format_type=settings['format'],
            window_level_mode=settings.get('window_level', 'displayed'),
            selected_indices=self.selected_indices,
            singleton=self.singleton,
            levels=levels,
            pixel_ratio_mode=settings.get('pixel_ratio', 'square_pixels'),
            display_mode=display_mode,
            widget_ratio=widget_ratio,
            axis_flipped=self.axis_flipped,
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

        # Clamp selected_indices: keep existing choices where still valid, fill from valid dims
        valid_dims = [i for i in range(new_ndim) if not self.singleton[i]]
        clamped = [i for i in self.selected_indices if i < new_ndim and not self.singleton[i]]
        for d in valid_dims:
            if len(clamped) >= 2:
                break
            if d not in clamped:
                clamped.append(d)
        if not clamped and valid_dims:
            clamped = [valid_dims[0]]
        self.selected_indices = clamped[:2]

        # Clamp line_plot_dimension
        if self.line_plot_dimension >= new_ndim or self.singleton[self.line_plot_dimension]:
            for d in range(new_ndim):
                if not self.singleton[d]:
                    self.line_plot_dimension = d
                    break

        self.axis_flipped = [False] * new_ndim
        self.fftshifted = [False] * new_ndim

        self._update_channel_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        if len(self.selected_indices) >= 1:
            self.changedIndex(True, 0, self.selected_indices[0], update=False)
        if len(self.selected_indices) >= 2:
            self.changedIndex(True, 1, self.selected_indices[1], update=False)
        self.update_dimension_controls()
        self._force_autolevel = True
        self.update()
