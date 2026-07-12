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
    def __init__(self, parent, *, on_tool_changed, on_add_roi, on_delete_roi, on_clear_rois, on_select_roi=None, on_sync_toggled=None, on_change_color=None):
        super().__init__("Inspection", parent)
        self.setObjectName("InspectionDock")
        self._on_sync_toggled = on_sync_toggled
        self._on_tool_changed = on_tool_changed
        self._on_add_roi = on_add_roi
        self._on_delete_roi = on_delete_roi
        self._on_clear_rois = on_clear_rois
        self._on_select_roi = on_select_roi
        self._on_change_color = on_change_color
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
        self.sync_button = QtWidgets.QToolButton()
        self.sync_button.setCheckable(True)
        set_button_icon(
            self.sync_button,
            "link",
            tooltip="Sync ROIs with other linked ArrayScope windows (also from separately started sessions)",
        )
        self.sync_button.toggled.connect(
            lambda checked: self._on_sync_toggled(bool(checked)) if self._on_sync_toggled is not None else None
        )
        controls.addWidget(self.sync_button)
        layout.addLayout(controls)
        self.roi_model = RoiTableModel(self)
        self.stats_table = QtWidgets.QTableView()
        self.stats_table.setModel(self.roi_model)
        self.stats_table.verticalHeader().hide()
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        self.stats_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.stats_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        self.histogram_plot = pg.PlotWidget()
        self.histogram_plot.showGrid(x=True, y=True, alpha=0.25)
        self.histogram_plot.setMinimumHeight(120)
        self.histogram_plot.getPlotItem().setLabel("bottom", "Value")
        self.histogram_plot.getPlotItem().setLabel("left", "Count")

        self.splitter = QtWidgets.QSplitter(Qt.QtCore.Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.stats_table)
        self.splitter.addWidget(self.histogram_plot)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        # Auto split: the table gets exactly its rows plus half an empty row,
        # the histogram takes the rest (clamped to 25–75% of the range) —
        # until the user drags the handle, which takes over for good.
        self._splitter_user_override = False
        self._applying_auto_split = False
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        layout.addWidget(self.splitter, 1)
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
        self.stats_table.clicked.connect(self._cell_clicked)
        self.add_button.setEnabled(self.current_tool() in {"roi_line", "roi_rectangle"})

    def _cell_clicked(self, index):
        """Single-click editing: color swatch opens the picker bubble, the
        name cell goes straight into inline rename."""
        if not index.isValid():
            return
        if index.column() == RoiTableModel.COLOR_COLUMN:
            roi_id = self.roi_model.roi_id_for_row(index.row())
            if roi_id is not None and self._on_change_color is not None:
                self._on_change_color(roi_id)
        elif index.column() == RoiTableModel.NAME_COLUMN:
            self.stats_table.edit(index)

    def _on_splitter_moved(self, _pos, _index):
        if not self._applying_auto_split:
            self._splitter_user_override = True

    def _apply_auto_split(self):
        if self._splitter_user_override:
            return
        total = self.splitter.height()
        if total <= 60:
            return
        rows = max(1, self.roi_model.rowCount())
        header_height = self.stats_table.horizontalHeader().height()
        row_height = self.stats_table.verticalHeader().defaultSectionSize()
        if rows and self.stats_table.rowHeight(0) > 0:
            row_height = self.stats_table.rowHeight(0)
        row_height = max(16, int(row_height or 24))
        table_needed = int(header_height + (rows + 0.5) * row_height + 2 * self.stats_table.frameWidth())
        graph = total - table_needed
        graph = max(int(total * 0.25), min(int(total * 0.75), graph))
        self._applying_auto_split = True
        try:
            self.splitter.setSizes([max(0, total - graph), graph])
        finally:
            self._applying_auto_split = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_auto_split()

    def apply_theme(self, tokens=None):
        """Restyle the histogram plot from the active theme tokens."""
        from arrayscope.app.theme import current_theme_tokens

        tokens = tokens or current_theme_tokens()
        try:
            self.histogram_plot.setBackground(tokens.canvas)
            axis_pen = pg.mkPen(tokens.plot_text)
            for name in ("left", "bottom"):
                axis = self.histogram_plot.getPlotItem().getAxis(name)
                axis.setPen(axis_pen)
                axis.setTextPen(axis_pen)
        except Exception:
            pass

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
        self._apply_auto_split()

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
