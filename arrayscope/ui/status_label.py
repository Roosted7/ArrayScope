"""Compact eliding status labels for the main toolbar row."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


class ElideLabel(QtWidgets.QLabel):
    """QLabel that paints elided text and never forces the layout wider.

    Regular QLabels demand their full text width as a minimum; inside the
    toolbar's centered status section that would block shrinking. An
    `Ignored` policy is no good either — next to stretch items it collapses
    to zero width. Preferred policy + a zero minimum hint gives natural
    width when there is room and graceful eliding when there is not.
    """

    def __init__(self, text="", parent=None):
        super().__init__(str(text), parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QtCore.QSize(0, hint.height())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        metrics = self.fontMetrics()
        text = metrics.elidedText(self.text(), QtCore.Qt.TextElideMode.ElideRight, max(8, self.width()))
        painter.drawText(self.rect(), self.alignment() | QtCore.Qt.AlignmentFlag.AlignVCenter, text)


class PixelStatusLabel(QtWidgets.QLabel):
    statusChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__("", parent)
        self._value_text = ""
        self._slice_text = ""
        self.setMinimumWidth(0)
        self.setMaximumWidth(420)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QtCore.QSize(0, hint.height())

    def set_pixel_status(self, value_text, slice_text=""):
        self._value_text = str(value_text)
        self._slice_text = str(slice_text)
        full = self._combined_text(self._slice_text)
        self.setToolTip(full)
        self.setText(full)
        self.update()
        self.statusChanged.emit(full)

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
