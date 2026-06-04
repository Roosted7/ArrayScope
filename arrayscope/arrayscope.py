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
from .dim_ops import (
    apply_fftshift,
    centered_fft,
    centered_ifft,
    combine_real_imag_axis,
    split_complex_axis,
    undo_fftshift,
)
from .imageview2d import ImageView2D
from .line_plot import LinePlotController
from .slice_engine import make_image
from .video_export import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
from .view_state import ChannelMode, ScaleMode, ViewState

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

        self.data = data
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
        self.combined_as_complex = [False] * data.ndim
        
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

        self.view_state = self._make_view_state(slice_indices=[0] * data.ndim)
                
        self.domain = [Domain.NATIVE for _ in range(data.ndim)]
        self.widgets = {
            'buttons': {
                'primary': [QtWidgets.QPushButton(str(i), checkable=True) for i in range(data.ndim)],
                'secondary': [QtWidgets.QPushButton(str(i), checkable=True) for i in range(data.ndim)],
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
                    'fit': QtWidgets.QRadioButton('Fit', checkable=True)
                }
            },
            'labels': {
                'dims': [QtWidgets.QLabel('[' + str(data.shape[i]) + ']', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'flip': [QtWidgets.QLabel('', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
                'shift': [QtWidgets.QLabel('sh', alignment=Qt.QtCore.Qt.AlignmentFlag.AlignCenter) for i in range(data.ndim)],
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
            label.setToolTip(f"Apply centered FFT along dim {i}")

        
        # Set up flip labels with click handlers
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            flip_label.mousePressEvent = lambda event, i=i: self.flipAxisClicked(event, i)
            flip_label.setStyleSheet(self.FLIP_ICON_STYLE)
            flip_label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignLeft | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._set_emoji_font(flip_label)

        for i, shift_label in enumerate(self.widgets['labels']['shift']):
            shift_label.mousePressEvent = lambda event, i=i: self.fftshiftClicked(event, i)
            shift_label.setStyleSheet(self.SHIFT_LABEL_STYLE)
            shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
            shift_label.setToolTip(f"Toggle fftshift along dim {i}")
        
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
        
        for btn in self.widgets['buttons']['primary'] + self.widgets['buttons']['secondary']:
            btn.setStyleSheet(self.BUTTON_STYLE)
            
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
            label_layout.addWidget(self.widgets['labels']['shift'][i])
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
            w.clicked.connect(lambda checked, i=i : self.changedIndex(checked, 0, i))
            
            w = self.widgets['buttons']['secondary'][i]
            self.dim_containers[i].layout().addWidget(w)
            w.clicked.connect(lambda checked, i=i: self.changedIndex(checked, 1, i))
            
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
        
        disp_buttons = list(self.widgets['buttons']['display'].values())
        for btn in disp_buttons:
            btn.setStyleSheet(self.RADIO_BUTTON_STYLE)
            display_layout.addWidget(btn)
            btn.clicked.connect(self.update_display_mode)
        
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
        self.img_view.getView().scene().sigMouseMoved.connect(lambda pos: self.getPixel(pos))
        
        # Connect to view range changes to update aspect ratio in fit mode
        self.img_view.getView().sigRangeChanged.connect(self._on_view_range_changed)
        
        # Create line plot tab
        self.plot_tab = QtWidgets.QWidget()
        self.plot_tab_layout = QtWidgets.QVBoxLayout()
        
        self.line_plot = LinePlotController(self)
        self.plot_widget = self.line_plot.widget
        self.plot_tab_layout.addWidget(self.plot_widget)
        self.plot_tab.setLayout(self.plot_tab_layout)

        # Add tabs to tab widget
        self.tab_widget.addTab(self.image_tab, "Image View")
        self.tab_widget.addTab(self.plot_tab, "Line Plot")

        if data.ndim == 1:
            self.tab_widget.setTabEnabled(0, False)  # Disable Image View tab
            self.tab_widget.setCurrentIndex(1)  # Set line plot as default
        
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

    def _sync_view_state_from_window(self):
        self.view_state = self._make_view_state()
        return self.view_state

    def _current_slice_indices(self):
        if not hasattr(self, 'widgets'):
            return (0,) * self.data.ndim

        indices = []
        for axis, spinbox in enumerate(self.widgets['spins']['slice_indices']):
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
        if self.singleton[dim]:
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
        """Apply forward FFT along specified dimension"""
        self.data = centered_fft(self.data, dim)
        self._update_channel_controls()
        
    def _apply_ifft(self, dim):
        """Apply inverse FFT along specified dimension"""
        self.data = centered_ifft(self.data, dim)
        self._update_channel_controls()

    def fftshiftClicked(self, event, dim):
        """Toggle fftshift along one array dimension without applying an FFT."""
        if self.singleton[dim]:
            return

        if self.fftshifted[dim]:
            self.data = undo_fftshift(self.data, dim)
            self.fftshifted[dim] = False
        else:
            self.data = apply_fftshift(self.data, dim)
            self.fftshifted[dim] = True

        self.update_shift_indicators()
        self.update()

    def update_shift_indicators(self):
        for i, shift_label in enumerate(self.widgets['labels']['shift']):
            if self.singleton[i]:
                shift_label.setText('')
                shift_label.setToolTip('')
                shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
                continue

            shift_label.setText('sh')
            shift_label.setToolTip(
                f"{'Undo fftshift' if self.fftshifted[i] else 'Apply fftshift'} along dim {i}"
            )
            shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
            if self.fftshifted[i]:
                shift_label.setStyleSheet(self.SHIFT_LABEL_ACTIVE_STYLE)
            else:
                shift_label.setStyleSheet(self.SHIFT_LABEL_STYLE)
    
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
            # ℝ clicked - combine to complex
            self.combineAsComplex(dim)
        elif self.combined_as_complex[dim]:
            # ℂ clicked - split back to real
            self.splitToReal(dim)
    
    def combineAsComplex(self, dim):
        """Combine a size-2 real dimension into complex (real+imag), keeping singleton dimension (makes indexing easier)"""
        if not self.can_combine_as_complex[dim] or self.combined_as_complex[dim]:
            return

        if self.fftshifted[dim]:
            self.data = undo_fftshift(self.data, dim)
            self.fftshifted[dim] = False
        
        self.data = combine_real_imag_axis(self.data, dim)

        # Update state
        self.combined_as_complex[dim] = True
        self.can_combine_as_complex[dim] = False
        self.singleton[dim] = True  # Now it's a singleton dimension
        
        # Once data is complex, no other dimension can be combined (all other size-2 dims become invalid)
        for i in range(self.data.ndim):
            if i != dim:
                self.can_combine_as_complex[i] = False
        
        # If the converted dimension was in selected_indices or line_plot_dimension, we need to find a new valid selection
        if dim in self.selected_indices:
            # Find a new non-singleton dimension to replace it
            for new_dim in range(self.data.ndim):
                if not self.singleton[new_dim] and new_dim not in self.selected_indices:
                    idx = self.selected_indices.index(dim)
                    self.selected_indices[idx] = new_dim
                    break
        
        # Same problem can happen with line plot.
        if self.line_plot_dimension == dim:
            for new_dim in range(self.data.ndim):
                if not self.singleton[new_dim]:
                    self.line_plot_dimension = new_dim
                    break
        
        self._update_channel_controls()
        
        # Update UI
        self.update_complex_indicators()
        self.update_dimension_controls()
        self.update()
    
    def splitToReal(self, dim):
        """Split a complex dimension back to real (real+imag as separate slices)"""
        if not self.combined_as_complex[dim]:
            return
        
        self.data = split_complex_axis(self.data, dim)
        
        # Update state
        self.combined_as_complex[dim] = False
        self.can_combine_as_complex[dim] = True
        self.singleton[dim] = False
        
        # Data is now real - re-enable combining for all size-2 dimensions
        for i in range(self.data.ndim):
            if self.data.shape[i] == 2:
                self.can_combine_as_complex[i] = True
        
        # Update max value for the spinbox for this dimension
        self.widgets['spins']['slice_indices'][dim].setMaximum(self.data.shape[dim] - 1)
        
        self._update_channel_controls()
        
        # Update UI
        self.update_complex_indicators()
        self.update_dimension_controls()
        self.update()
    
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
            
        prev_levels = None
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
        al = changed_scale or changed_channel or getattr(self, '_force_autolevel', False)
        # reset the one-shot flag after using it
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        
        prev_levels = None
        if not al:
            prev_levels = self.img_view.imageItem.levels

        
        try:
            self._sync_view_state_from_window()
            colormap_lut = None
            if self.view_state.channel == ChannelMode.COMPLEX:
                colormap_lut = self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
            display_image = make_image(self.data, self.view_state, colormap_lut=colormap_lut)
            if display_image.default_levels is not None and al:
                prev_levels = display_image.default_levels

            if display_image.histogram_data is not None:
                self.img_view.setImage(
                    display_image.data,
                    autoLevels=al,
                    histogramData=display_image.histogram_data,
                )
            else:
                self.img_view.setImage(display_image.data, autoLevels=al, levels=prev_levels)
            
            # Apply axis flips after setting the image
            self.apply_axis_flips()
            
        except Exception as e:
            print(f'Image update failed: {e}')
    
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
        self._sync_view_state_from_window()
        self.line_plot.update(self.data, self.view_state)

    def _on_view_range_changed(self):
        """Update display group title when view range changes (for fit mode)."""
        self._update_display_group_title()

    def on_tab_changed(self, index):
        """Handle tab change between Image View and Line Plot"""
        self.update_dimension_controls()
        self.update()
        if not self.is_line_plot_mode():
            self.line_plot.hide_crosshair()
    
    def is_line_plot_mode(self):
        """Check if currently in line plot mode"""
        return self.tab_widget.currentIndex() == 1

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
                
                if self.singleton[i]:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                elif i == self.line_plot_dimension:
                    # This is the dimension we're plotting along
                    w.setEnabled(False)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bPrim.setChecked(True)
                    bSecondary.setChecked(False)
                else:
                    # All other dimensions: enable spinbox to select slice
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
        else:
            # Image view mode: original two-dimension selection behavior
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                if self.singleton[i] == True:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
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
                else:
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(True)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    
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
                # which tab was double-clicked
                tab_bar = self.tab_widget.tabBar()
                clicked_index = tab_bar.tabAt(event.pos())
                
                # Check if line/bar tab (index 1)
                if clicked_index == 1:
                    self.line_plot.toggle_style()
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
        menu = QtWidgets.QMenu()
        
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
        if hasattr(self.img_view, 'imageItem') and self.img_view.imageItem.levels is not None:
            levels = tuple(self.img_view.imageItem.levels)

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

        self.data = new_data
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
