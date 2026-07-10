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


class MovableInfoPanel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_offset = None
        self.setObjectName("RoiInfoPanel")
        self.setWordWrap(False)
        # Styling comes from the application stylesheet (QLabel#RoiInfoPanel).

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
