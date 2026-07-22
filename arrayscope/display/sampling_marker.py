"""Shared screen-space marker for a fixed hover sample."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class SamplingMarkerOverlay(QtWidgets.QWidget):
    """Small high-contrast X painted identically on every display backend."""

    SIZE = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # The wgpu floating-widget compositor uses this attribute to preserve
        # the transparent custom-painted background when rasterizing.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

    def paintEvent(self, _event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        path = QtGui.QPainterPath()
        inset = 3.0
        end = float(self.SIZE - 1) - inset
        path.moveTo(inset, inset)
        path.lineTo(end, end)
        path.moveTo(inset, end)
        path.lineTo(end, inset)
        painter.setPen(
            QtGui.QPen(
                QtGui.QColor(15, 25, 32, 230),
                4.0,
                QtCore.Qt.PenStyle.SolidLine,
                QtCore.Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawPath(path)
        painter.setPen(
            QtGui.QPen(
                QtGui.QColor(80, 210, 255, 255),
                2.0,
                QtCore.Qt.PenStyle.SolidLine,
                QtCore.Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawPath(path)
