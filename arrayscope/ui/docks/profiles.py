"""Dockable profile plot panel for ArrayScope."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets
import numpy as np

from arrayscope.display.line_plot import LinePlotController
from arrayscope.ui.file_dialogs import get_save_file_name
from arrayscope.ui.icons import set_button_icon


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

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Axis:"))
        self.axis_combo = QtWidgets.QComboBox()
        controls.addWidget(self.axis_combo)
        controls.addWidget(QtWidgets.QLabel("Y:"))
        self.y_scale_combo = QtWidgets.QComboBox()
        self.y_scale_combo.addItem("Match image window", "match_image")
        self.y_scale_combo.addItem("Auto", "auto")
        controls.addWidget(self.y_scale_combo)
        controls.addWidget(QtWidgets.QLabel("Mode:"))
        self.profile_mode_combo = QtWidgets.QComboBox()
        self.profile_mode_combo.addItem("Magnitude", "abs")
        self.profile_mode_combo.addItem("Magnitude + phase", "abs_phase")
        self.profile_mode_combo.addItem("Phase", "angle")
        self.profile_mode_combo.addItem("Real", "real")
        self.profile_mode_combo.addItem("Imaginary", "imag")
        controls.addWidget(self.profile_mode_combo)
        self.export_button = QtWidgets.QPushButton("Export")
        set_button_icon(self.export_button, "download")
        controls.addWidget(self.export_button)
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
        self.profile_mode_combo.currentIndexChanged.connect(lambda _index: parent.update_line_plot())
        self.export_button.clicked.connect(self.export_profile)

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
        line_result = self._line_result_for_mode(line_result)
        self.line_plot.update_line_result(line_result, view_state, y_range=y_range)

    def update_line_results(self, line_entries, y_range=None):
        converted = []
        phase_strip = None
        for line_result, view_state, label in line_entries:
            if phase_strip is None and self.profile_mode() == "abs_phase" and np.iscomplexobj(line_result.data):
                phase_strip = np.angle(line_result.data)
            converted.append((self._line_result_for_mode(line_result), view_state, label))
        self.line_plot.update_line_results(tuple(converted), y_range=y_range, phase_strip_data=phase_strip)

    def profile_mode(self):
        return self.profile_mode_combo.currentData() or "abs"

    def export_profile(self):
        data = self.line_plot.current_line_data
        if data is None:
            return None
        file_path, _ = get_save_file_name(
            self,
            "Export profile",
            "arrayscope-profile.csv",
            "CSV files (*.csv);;NumPy files (*.npy)",
        )
        if not file_path:
            return None
        x = np.arange(len(data))
        if file_path.lower().endswith(".npy"):
            payload = np.column_stack([x, data])
            np.save(file_path, payload)
        else:
            if not file_path.lower().endswith(".csv"):
                file_path += ".csv"
            payload = np.column_stack([x, data])
            np.savetxt(file_path, payload, delimiter=",", header="index,value", comments="")
        return file_path

    def hide_crosshair(self):
        self.line_plot.hide_crosshair()

    def toggle_style(self):
        self.line_plot.toggle_style()

    def _axis_index_changed(self, index):
        if self._updating_axis or index < 0:
            return
        axis = self.axis_combo.itemData(index)
        if axis is not None:
            self._on_axis_changed(int(axis))

    def _line_result_for_mode(self, line_result):
        data = line_result.data
        if np.iscomplexobj(data):
            mode = self.profile_mode()
            if mode == "angle":
                data = np.angle(data)
            elif mode == "real":
                data = np.real(data)
            elif mode == "imag":
                data = np.imag(data)
            else:
                data = np.abs(data)
            return type(line_result)(data=data, axis=line_result.axis)
        return line_result
