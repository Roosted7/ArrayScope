"""Display overlay graphics owned by ImageView2D."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class MontageTileOverlay:
    x: int
    y: int
    width: int
    height: int
    state: str
    text: str


class MontageTileOverlayItem(QtWidgets.QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.setAcceptHoverEvents(False)
        self._overlays = ()

    @property
    def overlay_count(self) -> int:
        return len(self._overlays)

    def setOverlays(self, overlays: tuple[MontageTileOverlay, ...]) -> None:
        self.prepareGeometryChange()
        self._overlays = tuple(overlays or ())
        self.setVisible(bool(self._overlays))
        self.update()

    def boundingRect(self):
        if not self._overlays:
            return QtCore.QRectF()
        x0 = min(float(overlay.x) for overlay in self._overlays)
        y0 = min(float(overlay.y) for overlay in self._overlays)
        x1 = max(float(overlay.x + overlay.width) for overlay in self._overlays)
        y1 = max(float(overlay.y + overlay.height) for overlay in self._overlays)
        return QtCore.QRectF(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))

    def paint(self, painter, option, widget=None):
        del option, widget
        for overlay in self._overlays:
            rect = QtCore.QRectF(float(overlay.x), float(overlay.y), float(overlay.width), float(overlay.height))
            if str(overlay.state) == "skipped":
                brush = QtGui.QColor(130, 70, 20, 95)
                pen = QtGui.QColor(210, 130, 60, 180)
            else:
                brush = QtGui.QColor(35, 35, 35, 95)
                pen = QtGui.QColor(170, 170, 170, 140)
            painter.setBrush(QtGui.QBrush(brush))
            painter.setPen(QtGui.QPen(pen))
            painter.drawRect(rect)
            self._paint_status_mark(painter, rect, str(overlay.state))

    def _paint_status_mark(self, painter, rect, state: str) -> None:
        tile_extent = min(float(rect.width()), float(rect.height()))
        size = max(tile_extent * 0.08, min(tile_extent * 0.18, 0.35))
        cx = float(rect.center().x())
        cy = float(rect.center().y())
        icon_rect = QtCore.QRectF(cx - size / 2.0, cy - size / 2.0, size, size)
        pen = QtGui.QPen(QtGui.QColor(245, 245, 245, 230))
        pen.setWidthF(max(tile_extent * 0.006, size * 0.12))
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        if state == "skipped":
            painter.drawLine(icon_rect.topLeft(), icon_rect.bottomRight())
            painter.drawLine(icon_rect.bottomLeft(), icon_rect.topRight())
            return
        painter.drawEllipse(icon_rect)
        painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx, cy - size * 0.32))
        painter.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx + size * 0.28, cy))
