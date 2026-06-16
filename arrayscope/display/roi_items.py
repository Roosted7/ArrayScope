"""ROI graphics-item helpers for ImageView2D."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.core.roi import RoiGeometry, RoiKind


def default_roi_label(kind, index) -> str:
    labels = {
        RoiKind.LINE: "Line",
        RoiKind.RECTANGLE: "Rectangle",
        RoiKind.POLYLINE: "Polyline",
        RoiKind.FREEHAND_POLYGON: "Freehand",
    }
    return f"{labels.get(kind, 'ROI')} {int(index)}"


def point_distance(a, b) -> float:
    return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def item_for_roi(selection):
    geometry = selection.geometry
    pen = pg.mkPen(selection.color + (220,), width=2)
    hover_pen = pg.mkPen(selection.color + (255,), width=3)
    if geometry.kind == RoiKind.LINE:
        return pg.LineSegmentROI(geometry.points[:2], pen=pen, hoverPen=hover_pen, movable=True)
    if geometry.kind == RoiKind.RECTANGLE:
        x, y, width, height = geometry.rect
        return pg.RectROI((x, y), (width, height), pen=pen, hoverPen=hover_pen, movable=True)
    if geometry.kind in (RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
        return pg.PolyLineROI(
            geometry.points,
            closed=geometry.kind == RoiKind.FREEHAND_POLYGON,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,
        )
    raise ValueError(f"unsupported ROI kind: {geometry.kind}")


def geometry_from_item(item, previous):
    state = item.getState()
    if previous.kind == RoiKind.RECTANGLE:
        pos = state["pos"]
        size = state["size"]
        return RoiGeometry(previous.kind, rect=(float(pos.x()), float(pos.y()), float(size.x()), float(size.y())))
    if previous.kind in (RoiKind.LINE, RoiKind.POLYLINE, RoiKind.FREEHAND_POLYGON):
        pos = state.get("pos")
        dx = 0.0 if pos is None else float(pos.x())
        dy = 0.0 if pos is None else float(pos.y())
        points = tuple((float(point.x()) + dx, float(point.y()) + dy) for point in state.get("points", ()))
        if previous.kind == RoiKind.FREEHAND_POLYGON:
            from arrayscope.core.roi import close_polygon

            points = close_polygon(points)
        return RoiGeometry(previous.kind, points=points, line_width=previous.line_width, closed=previous.closed)
    return previous


class MovableInfoPanel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_offset = None
        self.setObjectName("RoiInfoPanel")
        self.setWordWrap(False)
        self.setStyleSheet(
            "QLabel#RoiInfoPanel { background: rgba(20, 20, 20, 175); color: white; "
            "border: 1px solid rgba(255, 255, 255, 90); border-radius: 4px; padding: 6px 8px; }"
        )

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

