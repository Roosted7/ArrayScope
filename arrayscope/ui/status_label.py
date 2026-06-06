"""Compact eliding status label for the main toolbar row."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class PixelStatusLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__("", parent)
        self._value_text = ""
        self._slice_text = ""
        self.setMinimumWidth(0)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)

    def set_pixel_status(self, value_text, slice_text=""):
        self._value_text = str(value_text)
        self._slice_text = str(slice_text)
        full = self._combined_text(self._slice_text)
        self.setToolTip(full)
        self.setText(full)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        text = self._text_for_width(max(8, self.width()))
        painter.drawText(self.rect(), self.alignment() | QtCore.Qt.AlignmentFlag.AlignVCenter, text)

    def _text_for_width(self, width):
        metrics = self.fontMetrics()
        full = self._combined_text(self._slice_text)
        if metrics.horizontalAdvance(full) <= width:
            return full
        if self._slice_text:
            for limit in range(len(self._slice_text), 0, -1):
                candidate = self._combined_text(self._slice_text[: max(0, limit - 3)] + "...")
                if metrics.horizontalAdvance(candidate) <= width:
                    return candidate
        if metrics.horizontalAdvance(self._value_text) <= width:
            return self._value_text
        if "=" in self._value_text:
            value_only = self._value_text.split("=", 1)[1].strip()
            if metrics.horizontalAdvance(value_only) <= width:
                return value_only
            return metrics.elidedText(value_only, QtCore.Qt.TextElideMode.ElideRight, width)
        return metrics.elidedText(self._value_text, QtCore.Qt.TextElideMode.ElideRight, width)

    def _combined_text(self, slice_text):
        if slice_text:
            return f"{self._value_text} | {slice_text}"
        return self._value_text
