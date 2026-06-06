from __future__ import annotations

import numpy as np

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.display.imageview2d import ImageView2D
from arrayscope.ui.docks.operations import OperationStackDock
from arrayscope.ui.docks.profiles import ProfileDock
from arrayscope.ui.dimension_strip import DimensionStrip
from arrayscope.ui.display_toolbar import DisplayToolbar
from arrayscope.ui.hud import PixelHud
from arrayscope.ui.icons import set_button_icon
from arrayscope.ui.status_label import PixelStatusLabel
from arrayscope.window.domain import Domain


class DisplayControlBuildMixin:
    def _build_window_ui(self, data, filepath):
        self._create_widget_registry(data)
        self._create_button_groups_and_profile_timer()
        self._create_layout_registry()
        self._build_dimension_role_bar(data)
        self._build_display_controls_panel()
        self._build_main_canvas()
        self._build_header_bar(filepath)
        self._compose_central_layout()
        self._build_docks_and_restore_layout()

    def _create_widget_registry(self, data):
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
                'pixelValue': PixelStatusLabel(),
                'arrayInfo': QtWidgets.QLabel('')
            },
            'spins': {
                'slice_indices': [QtWidgets.QSpinBox(minimum=0, maximum=data.shape[i]-1) for i in range(data.ndim)]
            }
        }

    def _create_button_groups_and_profile_timer(self):
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

    def _create_layout_registry(self):
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

    def _build_dimension_role_bar(self, data):
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
            # Add complex-state indicator.
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
        self.dimension_strip = DimensionStrip(data.ndim, self)
        self.dimension_strip.roleChanged.connect(self.set_dimension_role)
        self.dimension_strip.sliceChanged.connect(self._on_slice_index_changed)
        self.dimension_strip.operationRequested.connect(lambda axis: self._show_operation_context_menu_for_axis(axis))
        self.dimension_strip.focusedAxisChanged.connect(lambda axis: setattr(self, "_focused_dimension_axis", int(axis)))
        for container in self.dim_containers:
            container.hide()
        self.layouts['dims'].addWidget(self.dimension_strip, 1)

    def _build_display_controls_panel(self):
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
        self.display_controls_widget = controls_widget
        controls_widget.hide()
        self.layouts['botRight'].addWidget(controls_widget)

    def _build_main_canvas(self):
        # Create tab widget for switching between image and line plot views
        self.tab_widget = QtWidgets.QTabWidget()
        
        # Create image view tab
        self.image_tab = QtWidgets.QWidget()
        self.image_tab_layout = QtWidgets.QVBoxLayout()
        
        self.img_view = ImageView2D()
        self.pixel_hud = PixelHud()
        self.img_view.setHudWidget(self.pixel_hud)
        self.image_tab_layout.addWidget(self.img_view)
        self.image_tab.setLayout(self.image_tab_layout)
        self.img_view.getView().scene().sigMouseMoved.connect(lambda pos: self._on_image_mouse_moved(pos))
        self.img_view.set_profile_marker_callback(self._on_profile_marker_moved)
        
        # Connect to view range changes to update aspect ratio in fit mode
        self.img_view.getView().sigRangeChanged.connect(self._on_view_range_changed)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.image_tab, "Image View")
        self.tab_widget.tabBar().hide()
        
        # Connect tab change handler
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        self.tab_widget.tabBar().installEventFilter(self)
        
        # Add tab widget to the main layout
        self.layouts['topDown'].addWidget(self.tab_widget)

    def _build_header_bar(self, filepath):
        self._reload_btn = QtWidgets.QPushButton()
        self._reload_btn.setStyleSheet("QPushButton { padding: 1px 2px; margin: 0px; border: none; background: transparent; }")
        set_button_icon(self._reload_btn, "refresh", tooltip="Reload file")
        self._reload_btn.setFlat(True)
        self._reload_btn.setFixedSize(28, 20)
        self._reload_btn.clicked.connect(self._reload_file)
        self._reload_btn.setVisible(filepath is not None)
        self.layouts['topUp'].addWidget(self._reload_btn)
        self.display_toolbar = DisplayToolbar(self)
        self.display_toolbar.channelChanged.connect(self._on_channel_clicked)
        self.display_toolbar.scaleChanged.connect(self._on_scale_clicked)
        self.display_toolbar.aspectChanged.connect(self._on_aspect_toolbar_changed)
        self.display_toolbar.windowModeChanged.connect(self._on_window_mode_changed)
        self.display_toolbar.autoWindowRequested.connect(self.auto_window_levels)
        self.display_toolbar.liveProfileToggled.connect(self._set_live_profile_checked)
        self.layouts['topUp'].addWidget(self.display_toolbar)
        self.layouts['topUp'].addWidget(self.widgets['labels']['pixelValue'])
        self.layouts['topUp'].addWidget(self.widgets['labels']['arrayInfo'])

    def _compose_central_layout(self):
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

    def _build_docks_and_restore_layout(self):
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
            on_add_operation=self.open_operation_adder,
            on_export_derived=self.export_derived_array,
            on_save_view_recipe=self.save_view_recipe,
            on_load_view_recipe=self.load_view_recipe,
            on_enabled_changed=self.set_operation_enabled,
            on_edit_operation=self.edit_operation,
        )
        self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.operation_dock)
        self._update_operation_dock()
        self._setup_menus()
        self._restore_window_settings()
        
        # Initialize complex indicators for size-2 real dimensions
        self.update_complex_indicators()
        self.update_shift_indicators()
