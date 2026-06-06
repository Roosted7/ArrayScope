"""Dockable ROI inspection panel."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import numpy as np
import pyqtgraph as pg
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.roi import RoiKind
from arrayscope.ui.icons import set_button_icon


class InspectionDock(QtWidgets.QDockWidget):
    def __init__(self, parent, *, on_tool_changed, on_add_roi, on_delete_roi, on_clear_rois):
        super().__init__("Inspection", parent)
        self.setObjectName("InspectionDock")
        self._on_tool_changed = on_tool_changed
        self._on_add_roi = on_add_roi
        self._on_delete_roi = on_delete_roi
        self._on_clear_rois = on_clear_rois
        self._roi_ids = []
        self._stats_by_roi = {}
        self._updating = False

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.drag_handle = _DockDragHandle(self)
        layout.addWidget(self.drag_handle)

        controls = QtWidgets.QHBoxLayout()
        self.tool_combo = QtWidgets.QComboBox()
        for label, tool in (
            ("Cursor", "cursor"),
            ("Profile", "profile"),
            ("Line", "roi_line"),
            ("Rectangle", "roi_rectangle"),
            ("Polyline", "roi_polyline"),
            ("Freehand", "roi_freehand"),
        ):
            self.tool_combo.addItem(label, tool)
        controls.addWidget(self.tool_combo)

        self.add_button = QtWidgets.QToolButton()
        set_button_icon(self.add_button, "add", tooltip="Add ROI for the selected tool")
        controls.addWidget(self.add_button)
        self.delete_button = QtWidgets.QToolButton()
        set_button_icon(self.delete_button, "delete", tooltip="Delete selected ROI")
        controls.addWidget(self.delete_button)
        self.clear_button = QtWidgets.QToolButton()
        set_button_icon(self.clear_button, "delete_sweep", tooltip="Clear all ROIs")
        controls.addWidget(self.clear_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setMaximumHeight(96)
        layout.addWidget(self.roi_list)

        self.stats_table = QtWidgets.QTableWidget(0, 7)
        self.stats_table.setHorizontalHeaderLabels(("ROI", "Kind", "Count", "Mean", "Std", "Min", "Max"))
        self.stats_table.verticalHeader().hide()
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.stats_table)

        self.histogram_plot = pg.PlotWidget()
        self.histogram_plot.showGrid(x=True, y=True, alpha=0.25)
        self.histogram_plot.setMinimumHeight(120)
        layout.addWidget(self.histogram_plot, 1)

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

        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        self.add_button.clicked.connect(self._add_clicked)
        self.delete_button.clicked.connect(self._delete_clicked)
        self.clear_button.clicked.connect(self._clear_clicked)

    def current_tool(self):
        return self.tool_combo.currentData() or "cursor"

    def set_current_tool(self, tool):
        index = self.tool_combo.findData(tool)
        if index >= 0:
            self.tool_combo.setCurrentIndex(index)

    def set_rois(self, selections):
        current = self.current_roi_id()
        self._updating = True
        try:
            self.roi_list.clear()
            self._roi_ids = []
            for selection in selections:
                self._roi_ids.append(selection.id)
                item = QtWidgets.QListWidgetItem(_roi_item_text(selection))
                item.setData(Qt.QtCore.Qt.ItemDataRole.UserRole, selection.id)
                item.setCheckState(
                    Qt.QtCore.Qt.CheckState.Checked
                    if selection.enabled
                    else Qt.QtCore.Qt.CheckState.Unchecked
                )
                self.roi_list.addItem(item)
                if selection.id == current:
                    self.roi_list.setCurrentItem(item)
        finally:
            self._updating = False

    def current_roi_id(self):
        item = self.roi_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.QtCore.Qt.ItemDataRole.UserRole)

    def set_statistics(self, stats_by_roi):
        self._stats_by_roi = dict(stats_by_roi)
        self.stats_table.setRowCount(len(self._stats_by_roi))
        for row, (roi_id, payload) in enumerate(self._stats_by_roi.items()):
            selection, stats = payload
            values = (
                selection.label,
                selection.geometry.kind.value.replace("_", " "),
                str(stats.finite_count),
                _fmt(stats.mean),
                _fmt(stats.std),
                _fmt(stats.minimum),
                _fmt(stats.maximum),
            )
            for column, value in enumerate(values):
                self.stats_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.stats_table.resizeColumnsToContents()

    def set_histograms(self, histogram_results):
        self.histogram_plot.clear()
        colors = ((230, 60, 30), (40, 120, 210), (40, 150, 90), (180, 90, 210), (220, 150, 40))
        for index, result in enumerate(histogram_results):
            if result.counts.size == 0:
                continue
            centers = (result.edges[:-1] + result.edges[1:]) * 0.5
            pen = pg.mkPen(colors[index % len(colors)], width=2)
            self.histogram_plot.plot(centers, result.counts, pen=pen, name=result.name)

    def _tool_changed(self, _index):
        if not self._updating:
            self._on_tool_changed(self.current_tool())

    def _add_clicked(self):
        self._on_add_roi(self.current_tool())

    def _delete_clicked(self):
        roi_id = self.current_roi_id()
        if roi_id is not None:
            self._on_delete_roi(roi_id)

    def _clear_clicked(self):
        self._on_clear_rois()


def _roi_item_text(selection):
    geometry = selection.geometry
    if geometry.kind == RoiKind.RECTANGLE and geometry.rect is not None:
        detail = "rect"
    else:
        detail = f"{len(geometry.points)} pts"
    return f"{selection.label}  {geometry.kind.value.replace('_', ' ')}  {detail}"


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


class _DockDragHandle(QtWidgets.QFrame):
    def __init__(self, dock):
        super().__init__(dock)
        self._dock = dock
        self._drag_pos = None
        self.setObjectName("InspectionDockDragHandle")
        self.setCursor(Qt.QtCore.Qt.CursorShape.SizeAllCursor)
        self.setStyleSheet(
            "QFrame#InspectionDockDragHandle { background: palette(midlight); border-radius: 3px; }"
            "QLabel { font-weight: bold; padding: 3px 6px; }"
        )
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        label = QtWidgets.QLabel("Inspection")
        layout.addWidget(label)
        layout.addStretch()
        self.setLayout(layout)

    def mousePressEvent(self, event):
        if event.button() == Qt.QtCore.Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self._dock.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_pos is None:
            return super().mouseMoveEvent(event)
        if not self._dock.isFloating():
            self._dock.setFloating(True)
        self._dock.move(event.globalPosition().toPoint() - self._drag_pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
