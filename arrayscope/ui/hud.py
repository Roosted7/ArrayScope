"""Small on-canvas HUD widgets."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets


class PixelHud(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PixelHud")
        self.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignLeft | Qt.QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setStyleSheet(
            "QLabel#PixelHud { background: rgba(20, 20, 20, 175); color: white; "
            "padding: 4px 6px; border-radius: 4px; }"
        )
        self.setAttribute(Qt.QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

    def show_text_near(self, text, pos):
        self.setText(text)
        self.adjustSize()
        parent = self.parentWidget()
        if parent is None:
            self.move(8, 8)
        else:
            x = int(pos.x()) + 14
            y = int(pos.y()) + 14
            x = max(4, min(x, parent.width() - self.width() - 4))
            y = max(4, min(y, parent.height() - self.height() - 4))
            self.move(x, y)
        self.show()
        self.raise_()
