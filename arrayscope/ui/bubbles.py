"""Compact popup "bubbles" — nearby, non-modal edit controls.

Modeled on the histogram level-edit bubble: a small frameless popup with an
icon, a compact control, and a confirm button. Popups close on outside
clicks, so no modal dialogs interrupt the flow.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.icons import material_icon, set_button_icon


class EditBubble(QtWidgets.QWidget):
    """Base: frameless popup with an icon and a horizontal content row."""

    def __init__(self, parent=None, *, icon_name=None):
        super().__init__(
            parent,
            QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QtWidgets.QFrame(self)
        frame.setObjectName("EditBubble")
        outer.addWidget(frame)
        self.content_layout = QtWidgets.QHBoxLayout(frame)
        self.content_layout.setContentsMargins(8, 6, 6, 6)
        self.content_layout.setSpacing(6)
        if icon_name:
            icon_label = QtWidgets.QLabel(frame)
            icon_label.setPixmap(material_icon(icon_name).pixmap(14, 14))
            self.content_layout.addWidget(icon_label)

    def add_widget(self, widget, stretch=0):
        self.content_layout.addWidget(widget, stretch)
        return widget

    def add_confirm(self, callback):
        confirm = QtWidgets.QToolButton(self)
        set_button_icon(confirm, "done", tooltip="Apply")

        def _apply(_checked=False):
            callback()
            self.close()

        confirm.clicked.connect(_apply)
        self.content_layout.addWidget(confirm)
        return confirm

    def open_at(self, global_pos, focus_widget=None, *, place="above"):
        """Show the bubble anchored at ``global_pos``.

        ``place="above"`` (the default) floats the bubble just above the anchor
        point -- the histogram/edit-bubble convention. ``place="below"`` drops
        it just beneath the anchor, which is what a control that lives at the
        *top* of a dock (e.g. an "Add operation" button) wants.
        """

        self.adjustSize()
        screen = QtWidgets.QApplication.screenAt(global_pos)
        geometry = None if screen is None else screen.availableGeometry()
        width = self.sizeHint().width()
        height = self.sizeHint().height()
        x = int(global_pos.x()) - 12
        if place == "below":  # noqa: SIM108 - explicit above/below reads clearer
            y = int(global_pos.y()) + 4
        else:
            y = int(global_pos.y()) - height - 8
        if geometry is not None:
            x = max(geometry.left() + 4, min(x, geometry.right() - width - 4))
            y = max(geometry.top() + 4, min(y, geometry.bottom() - height - 4))
        self.move(x, y)
        self.show()
        if focus_widget is not None:
            focus_widget.setFocus()


class LineEditBubble(EditBubble):
    """Single-line text edit with confirm; Enter applies, outside click cancels."""

    def __init__(self, parent=None, *, icon_name="edit", label="", initial="", on_accept=None):
        super().__init__(parent, icon_name=icon_name)
        if label:
            self.content_layout.addWidget(QtWidgets.QLabel(str(label)))
        self.edit = QtWidgets.QLineEdit(str(initial), self)
        self.edit.setMinimumWidth(140)
        self.edit.selectAll()
        self.add_widget(self.edit, 1)
        self._on_accept = on_accept

        def _apply():
            text = self.edit.text().strip()
            if text and self._on_accept is not None:
                self._on_accept(text)

        self.add_confirm(_apply)
        self.edit.returnPressed.connect(lambda: (_apply(), self.close()))


class ColorSwatchBubble(EditBubble):
    """Compact swatch row; the trailing palette button opens the full picker."""

    def __init__(self, parent=None, *, colors=(), current=None, on_accept=None):
        super().__init__(parent, icon_name="colorize")
        self._on_accept = on_accept
        current = None if current is None else tuple(int(v) for v in current[:3])
        for color in colors:
            color = tuple(int(v) for v in color[:3])
            swatch = QtWidgets.QToolButton(self)
            swatch.setObjectName("ColorSwatchButton")
            swatch.setFixedSize(20, 20)
            swatch.setCheckable(True)
            swatch.setChecked(color == current)
            swatch.setStyleSheet(
                f"QToolButton#ColorSwatchButton {{ background: rgb({color[0]}, {color[1]}, {color[2]});"
                " border-radius: 4px; border: 2px solid transparent; }"
                "QToolButton#ColorSwatchButton:checked { border-color: palette(highlighted-text); }"
                "QToolButton#ColorSwatchButton:hover { border-color: palette(highlight); }"
            )
            swatch.clicked.connect(lambda _checked=False, color=color: self._pick(color))
            self.content_layout.addWidget(swatch)
        more = QtWidgets.QToolButton(self)
        set_button_icon(more, "edit", tooltip="More colors…")
        more.clicked.connect(self._open_full_picker)
        self.content_layout.addWidget(more)
        self._current = current

    def _pick(self, color):
        if self._on_accept is not None:
            self._on_accept(color)
        self.close()

    def _open_full_picker(self, _checked=False):
        initial = QtGui.QColor(*(self._current or (230, 60, 30)))
        self.hide()
        color = QtWidgets.QColorDialog.getColor(initial, self.parentWidget(), "ROI color")
        if color.isValid() and self._on_accept is not None:
            self._on_accept((color.red(), color.green(), color.blue()))
        self.close()
