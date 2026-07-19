"""Small standalone window shown while a file is loading.

Appears immediately at startup — before any file I/O — so the user always
sees feedback for slow files. Determinate byte/slice progress when the
reader can know its budget, an indeterminate bar otherwise. Closing the
window (or pressing Cancel) requests cancellation of the load.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtWidgets


def _human_bytes(count):
    count = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if count < 1024 or unit == "TB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024


_STAGE_TEXT = {
    "starting": "Starting…",
    "probing": "Reading file header…",
    "reading": "Reading data…",
    "converting": "Converting…",
    "finalizing": "Finishing…",
}


class LoadingWindow(QtWidgets.QWidget):
    """Progress feedback for one file being opened."""

    cancel_requested = QtCore.Signal()

    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._closing_programmatically = False
        name = Path(filepath).name or str(filepath)
        self.setWindowTitle(f"Opening {name} — ArrayScope")
        self.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(8)

        title = QtWidgets.QLabel(name)
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        title.setFont(font)
        title.setTextInteractionFlags(Qt.QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)

        self._stage_label = QtWidgets.QLabel("Starting…")
        layout.addWidget(self._stage_label)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate until a fraction arrives
        self._bar.setTextVisible(True)
        layout.addWidget(self._bar)

        self._bytes_label = QtWidgets.QLabel("")
        self._bytes_label.setStyleSheet("color: palette(mid);")
        layout.addWidget(self._bytes_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._request_cancel)
        button_row.addWidget(self._cancel_btn)
        layout.addLayout(button_row)

    @property
    def cancelled(self):
        return self._cancelled

    def _request_cancel(self):
        self._cancelled = True
        self._cancel_btn.setEnabled(False)
        self._stage_label.setText("Cancelling…")
        self.cancel_requested.emit()

    def apply_progress(self, event):
        """Update from a LoadProgress observation (GUI thread only)."""
        self._stage_label.setText(event.message or _STAGE_TEXT.get(event.stage, event.stage))
        if event.fraction is None:
            self._bar.setRange(0, 0)
        else:
            self._bar.setRange(0, 1000)
            self._bar.setValue(round(1000 * max(0.0, min(1.0, event.fraction))))
        if event.bytes_total:
            self._bytes_label.setText(
                f"{_human_bytes(event.bytes_done or 0)} of {_human_bytes(event.bytes_total)}"
            )

    def show_error(self, message):
        """Switch into a terminal error state."""
        self._stage_label.setText("Failed to load file")
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bytes_label.setText(str(message))
        self._bytes_label.setStyleSheet("color: #c04040;")
        self._bytes_label.setWordWrap(True)
        self._cancel_btn.setText("Close")
        self._cancel_btn.setEnabled(True)
        with contextlib.suppress(TypeError, RuntimeError):
            self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.close_quietly)

    def close_quietly(self):
        """Close without treating it as a user cancellation."""
        self._closing_programmatically = True
        self.close()

    def closeEvent(self, event):
        if not self._closing_programmatically and not self._cancelled:
            self._cancelled = True
            self.cancel_requested.emit()
        super().closeEvent(event)
