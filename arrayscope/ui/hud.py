"""Small on-canvas HUD widgets."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.ui.icons import material_icon


class PixelHud(QtWidgets.QFrame):
    """Cursor-following info chip.

    Shows the pixel readout, optionally preceded by context rows (hovered ROI
    or profile marker details), each with a small leading icon.
    """

    _ICON_SIZE = 13

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PixelHud")
        self.setAttribute(Qt.QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(7, 5, 8, 5)
        self._grid.setHorizontalSpacing(5)
        self._grid.setVerticalSpacing(2)
        self._row_widgets: list[tuple[QtWidgets.QLabel, QtWidgets.QLabel]] = []
        self._rows_key = None
        self.hide()

    # ------------------------------------------------------------------

    SEPARATOR = ("separator", "—")

    def set_rows(self, rows) -> None:
        """rows: sequence of (icon_name | None, text); PixelHud.SEPARATOR
        renders a thin horizontal rule instead of a text row."""
        rows = tuple(
            (icon, str(text))
            for icon, text in rows
            if (icon, text) == self.SEPARATOR or str(text)
        )
        if rows == self._rows_key:
            return
        self._rows_key = rows
        while len(self._row_widgets) < len(rows):
            index = len(self._row_widgets)
            icon_label = QtWidgets.QLabel(self)
            icon_label.setFixedSize(self._ICON_SIZE + 2, self._ICON_SIZE + 2)
            text_label = QtWidgets.QLabel(self)
            separator = QtWidgets.QFrame(self)
            separator.setObjectName("PixelHudSeparator")
            separator.setFixedHeight(1)
            self._grid.addWidget(icon_label, index, 0)
            self._grid.addWidget(text_label, index, 1)
            self._grid.addWidget(separator, index, 0, 1, 2)
            self._row_widgets.append((icon_label, text_label, separator))
        for index, (icon_label, text_label, separator) in enumerate(self._row_widgets):
            if index >= len(rows):
                icon_label.setVisible(False)
                text_label.setVisible(False)
                separator.setVisible(False)
                continue
            icon_name, text = rows[index]
            if (icon_name, text) == self.SEPARATOR:
                icon_label.setVisible(False)
                text_label.setVisible(False)
                separator.setVisible(True)
                continue
            separator.setVisible(False)
            if icon_name:
                icon_label.setPixmap(material_icon(icon_name).pixmap(self._ICON_SIZE, self._ICON_SIZE))
            else:
                icon_label.clear()
            text_label.setText(text)
            icon_label.setVisible(True)
            text_label.setVisible(True)

    def setText(self, text) -> None:  # QLabel-compatible convenience
        self.set_rows(((None, str(text)),))

    def text(self) -> str:
        return "\n".join(label.text() for _icon, label, _sep in self._row_widgets if label.isVisible())

    def show_rows_near(self, rows, pos) -> None:
        self.set_rows(rows)
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

    def show_text_near(self, text, pos) -> None:
        self.show_rows_near(((None, str(text)),), pos)
