"""Dockable ROI inspection panel."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import numpy as np
import pyqtgraph as pg
import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.core.roi import RoiKind
from arrayscope.ui.docks.common import StandardDockWidget, add_size_grip, configure_standard_dock
from arrayscope.ui.icons import set_button_icon
from arrayscope.ui.roi_model import RoiTableModel


class InspectionDock(StandardDockWidget):
    def __init__(self, parent, *, on_tool_changed, on_add_roi, on_delete_roi, on_clear_rois, on_select_roi=None):
        super().__init__("Inspection", parent)
        self.setObjectName("InspectionDock")
        self._on_tool_changed = on_tool_changed
        self._on_add_roi = on_add_roi
        self._on_delete_roi = on_delete_roi
        self._on_clear_rois = on_clear_rois
        self._on_select_roi = on_select_roi
        self._roi_ids = []
        self._stats_by_roi = {}
        self._updating = False

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        self.tool_combo = QtWidgets.QComboBox()
        for label, tool in (
            ("Cursor", "cursor"),
            ("Profile", "profile"),
            ("Line", "roi_line"),
            ("Rectangle", "roi_rectangle"),
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
        self.roi_model = RoiTableModel(self)
        self.stats_table = QtWidgets.QTableView()
        self.stats_table.setModel(self.roi_model)
        self.stats_table.verticalHeader().hide()
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.stats_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.stats_table)

        self.histogram_plot = pg.PlotWidget()
        self.histogram_plot.showGrid(x=True, y=True, alpha=0.25)
        self.histogram_plot.setMinimumHeight(120)
        layout.addWidget(self.histogram_plot, 1)
        add_size_grip(layout)

        body.setLayout(layout)
        self.setWidget(body)
        self.setAllowedAreas(
            Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        configure_standard_dock(self, min_size=(360, 260))

        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        self.add_button.clicked.connect(self._add_clicked)
        self.delete_button.clicked.connect(self._delete_clicked)
        self.clear_button.clicked.connect(self._clear_clicked)
        self.stats_table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.add_button.setEnabled(self.current_tool() in {"roi_line", "roi_rectangle"})

    def current_tool(self):
        return self.tool_combo.currentData() or "cursor"

    def set_current_tool(self, tool):
        index = self.tool_combo.findData(tool)
        if index >= 0:
            self.tool_combo.setCurrentIndex(index)

    def set_rois(self, selections):
        self._roi_ids = [selection.id for selection in selections]

    def current_roi_id(self):
        index = self.stats_table.currentIndex()
        if not index.isValid():
            return None
        return self.roi_model.roi_id_for_row(index.row())

    def set_statistics(self, stats_by_roi):
        self._stats_by_roi = dict(stats_by_roi)
        rows = []
        for roi_id, payload in self._stats_by_roi.items():
            selection, stats = payload
            values = (
                selection.label,
                selection.geometry.kind.value.replace("_", " "),
                str(stats.finite_count),
                _fmt(stats.mean),
                _fmt(stats.std),
                _fmt(stats.minimum),
                _fmt(stats.maximum),
                "",
            )
            rows.append({"id": roi_id, "values": values, "enabled": selection.enabled, "color": selection.color})
        self.roi_model.set_rows(rows)
        self.stats_table.resizeColumnsToContents()

    def set_histograms(self, histogram_results):
        self.histogram_plot.clear()
        for index, result in enumerate(histogram_results):
            if result.counts.size == 0:
                continue
            centers = (result.edges[:-1] + result.edges[1:]) * 0.5
            color = getattr(result, "color", None) or _color_for_histogram_name(result.name, self._stats_by_roi)
            pen = pg.mkPen(color, width=2)
            self.histogram_plot.plot(centers, result.counts, pen=pen, name=result.name)

    def _tool_changed(self, _index):
        self.add_button.setEnabled(self.current_tool() in {"roi_line", "roi_rectangle"})
        if not self._updating:
            self._on_tool_changed(self.current_tool())

    def _add_clicked(self):
        tool = self.current_tool()
        if tool in {"roi_line", "roi_rectangle"}:
            self._on_add_roi(tool)

    def _delete_clicked(self):
        roi_id = self.current_roi_id()
        if roi_id is not None:
            self._on_delete_roi(roi_id)

    def _clear_clicked(self):
        self._on_clear_rois()

    def _selection_changed(self, *_args):
        if self._updating or self._on_select_roi is None:
            return
        roi_id = self.current_roi_id()
        if roi_id is not None:
            self._on_select_roi(roi_id)


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


def _color_for_histogram_name(name, stats_by_roi):
    for _roi_id, (selection, _stats) in stats_by_roi.items():
        if str(name).startswith(selection.label):
            return selection.color
    return (230, 60, 30)
