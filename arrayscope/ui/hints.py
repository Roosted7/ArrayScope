"""One-time discoverability hints shown over the canvas."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.ui.icons import glyph_icon, material_icon

_HINTS = (
    ("glyph:0", "Click a dimension's number badge to plot a profile along it"),
    ("glyph::", "Type : into a dimension's index box to montage that axis"),
    ("icon:menu", "Right-click the image for ROIs, live profile and export"),
)


class FirstRunHints(QtWidgets.QFrame):
    """Dismissible hint chip for the three learn-by-accident power features."""

    SETTINGS_KEY = "first_run_hints_dismissed"

    def __init__(self, parent, *, on_dismiss=None):
        super().__init__(parent)
        self.setObjectName("FirstRunHints")
        self._on_dismiss = on_dismiss
        # The widget hugs this widget's top-right corner even if a surface
        # hook later reparents the chip (wgpu screen present moves overlays
        # to the top-level so they composite above the swapchain).
        self._anchor = parent
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(7)
        layout.setVerticalSpacing(4)
        for row, (icon_spec, text) in enumerate(_HINTS):
            icon_label = QtWidgets.QLabel(self)
            kind, _, name = icon_spec.partition(":")
            icon = glyph_icon(name) if kind == "glyph" else material_icon(name)
            icon_label.setPixmap(icon.pixmap(14, 14))
            layout.addWidget(icon_label, row, 0)
            layout.addWidget(QtWidgets.QLabel(text, self), row, 1)
        dismiss = QtWidgets.QPushButton("Got it", self)
        dismiss.setObjectName("FirstRunHintsDismiss")
        dismiss.clicked.connect(self._dismiss)
        layout.addWidget(dismiss, len(_HINTS), 0, 1, 2, QtCore.Qt.AlignmentFlag.AlignRight)
        if parent is not None:
            parent.installEventFilter(self)
        self._reposition()

    def _dismiss(self, _checked=False):
        if self._on_dismiss is not None:
            self._on_dismiss()
        self.hide()
        self.deleteLater()

    def _reposition(self):
        anchor = self._anchor
        parent = self.parentWidget()
        if parent is None or anchor is None:
            return
        self.adjustSize()
        pos = QtCore.QPoint(max(8, anchor.width() - self.width() - 16), 14)
        if parent is not anchor:
            pos = anchor.mapTo(parent, pos)
        self.move(pos)
        self.raise_()

    def eventFilter(self, obj, event):
        if obj is self._anchor and event.type() == QtCore.QEvent.Type.Resize:
            self._reposition()
        return super().eventFilter(obj, event)


def maybe_show_first_run_hints(window) -> FirstRunHints | None:
    """Show hints once per machine; dismissal is persisted in QSettings."""
    settings = getattr(window, "_settings", None)
    if settings is None:
        return None
    if str(settings.value(FirstRunHints.SETTINGS_KEY, "false")).lower() in ("true", "1"):
        return None
    view = getattr(window, "img_view", None)
    if view is None or not view.isVisible():
        return None

    def _persist_dismiss():
        settings.setValue(FirstRunHints.SETTINGS_KEY, True)

    hints = FirstRunHints(view, on_dismiss=_persist_dismiss)
    prepare = getattr(view, "_prepare_display_overlay_widget", None)
    if callable(prepare):
        prepare(hints)
    hints.show()
    hints._reposition()
    return hints
