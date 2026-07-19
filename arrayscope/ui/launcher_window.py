"""Empty-state launcher window: open files via dialog or drag-and-drop.

Shown when ArrayScope starts without file arguments (e.g. from a desktop
shell launcher). On macOS this window also keeps the app alive to receive
Finder open-document events.
"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets


class LauncherWindow(QtWidgets.QWidget):
    """Minimal landing window. ``open_callback(Path)`` opens one path."""

    def __init__(self, open_callback, *, supported_suffixes, name_filter, parent=None):
        super().__init__(parent)
        self._open_callback = open_callback
        self._supported_suffixes = tuple(s.lower() for s in supported_suffixes)
        self._name_filter = name_filter
        self.setWindowTitle("ArrayScope")
        self.setAcceptDrops(True)
        self.setMinimumSize(420, 300)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.addStretch(1)

        title = QtWidgets.QLabel("ArrayScope")
        font = title.font()
        font.setPointSize(font.pointSize() + 6)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("Drop array files here, or")
        subtitle.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        open_btn = QtWidgets.QPushButton("Open Files…")
        open_btn.setDefault(True)
        open_btn.clicked.connect(self._open_dialog)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(open_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        formats = QtWidgets.QLabel(
            "Supported: " + "  ".join(sorted(self._supported_suffixes)) + "  DICOM folders"
        )
        formats.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        formats.setWordWrap(True)
        formats.setStyleSheet("color: palette(mid);")
        layout.addWidget(formats)
        layout.addStretch(2)

    def _open_dialog(self):
        dialog = QtWidgets.QFileDialog(self, "Open array files", "", self._name_filter)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        opened_any = False
        for name in dialog.selectedFiles():
            opened_any = bool(self._open_callback(Path(name))) or opened_any
        if opened_any:
            self.close()

    def _droppable_paths(self, mime):
        if not mime.hasUrls():
            return []
        paths = [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile()]
        return [
            p
            for p in paths
            if p.is_dir() or any(str(p).lower().endswith(s) for s in self._supported_suffixes)
        ]

    def dragEnterEvent(self, event):
        if self._droppable_paths(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = self._droppable_paths(event.mimeData())
        if not paths:
            return
        event.acceptProposedAction()
        opened_any = False
        for path in paths:
            opened_any = bool(self._open_callback(path)) or opened_any
        if opened_any:
            self.close()
