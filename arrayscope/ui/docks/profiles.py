"""Dockable profile plot panel for ArrayScope."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.display.line_plot import LinePlotController


class ProfileDock(QtWidgets.QDockWidget):
    def __init__(self, parent, on_axis_changed):
        super().__init__("Profile", parent)
        self.setObjectName("ProfileDock")
        self._on_axis_changed = on_axis_changed
        self._updating_axis = False

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title_row = QtWidgets.QHBoxLayout()
        title_row.addWidget(QtWidgets.QLabel("Profile"))
        title_row.addStretch()
        self.float_button = QtWidgets.QPushButton("Float")
        self.close_button = QtWidgets.QPushButton("Close")
        title_row.addWidget(self.float_button)
        title_row.addWidget(self.close_button)
        layout.addLayout(title_row)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Axis:"))
        self.axis_combo = QtWidgets.QComboBox()
        controls.addWidget(self.axis_combo)
        controls.addWidget(QtWidgets.QLabel("Y:"))
        self.y_scale_combo = QtWidgets.QComboBox()
        self.y_scale_combo.addItem("Match image window", "match_image")
        self.y_scale_combo.addItem("Auto", "auto")
        controls.addWidget(self.y_scale_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.line_plot = LinePlotController(parent)
        layout.addWidget(self.line_plot.widget, 1)

        body.setLayout(layout)
        self.setWidget(body)
        self.setAllowedAreas(
            Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.setWindowTitle("Profile")
        self.resize(560, 260)

        self.axis_combo.currentIndexChanged.connect(self._axis_index_changed)
        self.float_button.clicked.connect(self._toggle_floating)
        self.close_button.clicked.connect(self.hide)
        self.topLevelChanged.connect(self._top_level_changed)

    @property
    def widget(self):
        return self.line_plot.widget

    def set_axes(self, shape, current_axis):
        self._updating_axis = True
        try:
            self.axis_combo.clear()
            for axis, size in enumerate(shape):
                self.axis_combo.addItem(f"{axis} [{size}]", axis)
            if shape:
                current_axis = max(0, min(int(current_axis), len(shape) - 1))
                self.axis_combo.setCurrentIndex(current_axis)
        finally:
            self._updating_axis = False

    def y_range_mode(self):
        return self.y_scale_combo.currentData() or "match_image"

    def update_line_result(self, line_result, view_state, y_range=None):
        self.line_plot.update_line_result(line_result, view_state, y_range=y_range)

    def hide_crosshair(self):
        self.line_plot.hide_crosshair()

    def toggle_style(self):
        self.line_plot.toggle_style()

    def _toggle_floating(self):
        self.setFloating(not self.isFloating())

    def _top_level_changed(self, floating):
        self.float_button.setText("Dock" if floating else "Float")
        if floating:
            flags = self.windowFlags()
            flags |= Qt.QtCore.Qt.WindowType.Window
            flags &= ~Qt.QtCore.Qt.WindowType.FramelessWindowHint
            self.setWindowFlags(flags)
            self.show()

    def _axis_index_changed(self, index):
        if self._updating_axis or index < 0:
            return
        axis = self.axis_combo.itemData(index)
        if axis is not None:
            self._on_axis_changed(int(axis))
