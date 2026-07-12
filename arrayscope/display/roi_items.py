"""ROI graphics-item helpers for ImageView2D."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.core.roi import RoiGeometry, RoiKind


def default_roi_label(kind, index) -> str:
    # The kind is displayed alongside the name everywhere (table, overlay,
    # HUD), so default names are just numbers.
    return str(int(index))


def point_distance(a, b) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def item_for_roi(selection):
    geometry = selection.geometry
    pen = pg.mkPen(selection.color + (220,), width=2)
    hover_pen = pg.mkPen(selection.color + (255,), width=3)
    if geometry.kind == RoiKind.LINE:
        return pg.LineSegmentROI(geometry.points[:2], pen=pen, hoverPen=hover_pen, movable=False)
    if geometry.kind == RoiKind.RECTANGLE:
        x, y, width, height = geometry.rect
        return pg.RectROI((x, y), (width, height), pen=pen, hoverPen=hover_pen, movable=False)
    if geometry.kind in (RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
        return pg.PolyLineROI(
            geometry.points,
            closed=geometry.kind == RoiKind.FREEHAND_POLYGON,
            pen=pen,
            hoverPen=hover_pen,
            movable=False,
        )
    raise ValueError(f"unsupported ROI kind: {geometry.kind}")


def make_item_passive(item) -> None:
    """Prevent a graphics item from owning pointer semantics."""

    for name, value in (
        ("setAcceptedMouseButtons", QtCore.Qt.MouseButton.NoButton),
        ("setAcceptHoverEvents", False),
    ):
        method = getattr(item, name, None)
        if callable(method):
            try:
                method(value)
            except Exception:
                pass


def sync_item_to_roi_geometry(item, geometry: RoiGeometry) -> None:
    """Mirror semantic ROI geometry into a PyQtGraph ROI item."""

    if geometry.kind == RoiKind.RECTANGLE and geometry.rect is not None:
        x, y, width, height = geometry.rect
        item.setPos(float(x), float(y))
        if hasattr(item, "setSize"):
            item.setSize((float(width), float(height)))
        return
    points = tuple((float(x), float(y)) for x, y in geometry.points)
    if geometry.kind == RoiKind.FREEHAND_POLYGON and len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if hasattr(item, "setPoints"):
        item.setPoints(points)
        item.setPos(0.0, 0.0)
        return
    if hasattr(item, "setState"):
        state = dict(item.saveState()) if hasattr(item, "saveState") else {}
        state.update({"pos": (0.0, 0.0), "points": points})
        state.setdefault("size", (1.0, 1.0))
        state.setdefault("angle", 0.0)
        item.setState(state)


class MovableInfoPanel(QtWidgets.QFrame):
    """Draggable ROI summary overlay.

    Structured mode (`set_rows`) renders one line per ROI: bold name +
    italic kind on the left, then a single continuous vertical divider,
    then the value columns right-aligned so they line up across rows.
    `setText` remains for plain-text callers (VisPy path, tests).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_offset = None
        self.setObjectName("RoiInfoPanel")
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(8, 5, 8, 5)
        self._grid.setHorizontalSpacing(8)
        self._grid.setVerticalSpacing(2)
        self._row_widgets = []
        self._divider = QtWidgets.QFrame(self)
        self._divider.setObjectName("RoiInfoDivider")
        self._divider.setFixedWidth(1)
        self._divider.hide()
        self._plain_label = QtWidgets.QLabel(self)
        self._plain_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._grid.addWidget(self._plain_label, 0, 0, 1, 4)
        self._plain_label.hide()
        # Styling comes from the application stylesheet (QFrame#RoiInfoPanel).

    def setText(self, text):
        self._plain_label.setText(str(text))
        self._plain_label.setVisible(bool(str(text)))
        for widgets in self._row_widgets:
            for widget in widgets:
                widget.setVisible(False)
        self._divider.hide()

    def text(self):
        if self._plain_label.isVisible() or not self._row_widgets:
            return self._plain_label.text()
        import re

        lines = []
        for widgets in self._row_widgets:
            if not widgets[0].isVisible():
                continue
            name = re.sub("<[^>]+>", "", widgets[0].text()).replace("&nbsp;", " ")
            values = " ".join(w.text() for w in widgets[1:] if w.isVisible() and w.text())
            lines.append(f"{name}: {values}".strip())
        return "\n".join(lines)

    def set_rows(self, rows):
        """rows: sequence of (name, kind, *value_columns)."""
        self._plain_label.hide()
        rows = tuple(rows)
        value_columns = max((len(row) - 2 for row in rows), default=0)
        while len(self._row_widgets) < len(rows):
            index = len(self._row_widgets) + 1  # row 0 is the plain label
            name_label = QtWidgets.QLabel(self)
            name_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            widgets = [name_label]
            self._grid.addWidget(name_label, index, 0)
            for column in range(2):
                value_label = QtWidgets.QLabel(self)
                value_label.setAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
                )
                self._grid.addWidget(value_label, index, 2 + column)
                widgets.append(value_label)
            self._row_widgets.append(tuple(widgets))
        from html import escape

        for index, widgets in enumerate(self._row_widgets):
            if index >= len(rows):
                for widget in widgets:
                    widget.setVisible(False)
                continue
            name, kind, *values = rows[index]
            widgets[0].setText(f"<b>{escape(str(name))}</b>&nbsp;<i>{escape(str(kind))}</i>")
            widgets[0].setVisible(True)
            for column, value_label in enumerate(widgets[1:]):
                value_label.setText(str(values[column]) if column < len(values) else "")
                value_label.setVisible(column < max(1, value_columns))
        if rows:
            self._grid.addWidget(self._divider, 1, 1, len(rows), 1)
            self._divider.show()
        else:
            self._divider.hide()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._drag_offset = event.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is None:
            return super().mouseMoveEvent(event)
        next_pos = self.mapToParent(event.pos() - self._drag_offset)
        parent = self.parentWidget()
        if parent is not None:
            next_pos.setX(max(0, min(next_pos.x(), parent.width() - self.width())))
            next_pos.setY(max(0, min(next_pos.y(), parent.height() - self.height())))
        self.move(next_pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        event.accept()
