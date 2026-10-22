"""Nonmodal progress and cancellation feedback for one selected-array export."""

from __future__ import annotations

from math import prod
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.core.memory_budget import format_bytes
from arrayscope.ui.status_label import ElideLabel

_PHASE_TEXT = {
    "starting": "Preparing export…",
    "preparing": "Preparing export…",
    "evaluating": "Evaluating selected data…",
    "materializing": "Evaluating selected data…",
    "writing": "Writing NumPy file…",
    "finalizing": "Finishing export…",
    "committing": "Finishing export…",
}


class ExportProgressDialog(QtWidgets.QDialog):
    """Present worker observations without taking ownership of the worker.

    Cancel, Escape, and the window close control request cancellation once.
    They keep feedback visible until the controller reports completion with
    ``close_quietly`` or a terminal failure with ``set_error``.
    """

    cancel_requested = QtCore.Signal()

    def __init__(self, parent, path, output_shape, dtype=None):
        super().__init__(parent)
        self._working = True
        self._cancelling = False
        path = Path(path)
        name = path.name or str(path)
        self.setObjectName("ExportProgressDialog")
        self.setWindowTitle(f"Exporting {name} — ArrayScope")
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.resize(480, 240)
        self.setMinimumWidth(380)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        self.filename_label = ElideLabel(name)
        self.filename_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.filename_label.setAccessibleName(f"Export file: {name}")
        self.filename_label.setToolTip(str(path))
        font = self.filename_label.font()
        font.setBold(True)
        self.filename_label.setFont(font)
        layout.addWidget(self.filename_label)

        self.summary_label = self._label("", "Selected output summary")
        self.set_output_summary(output_shape, dtype)
        layout.addWidget(self.summary_label)
        self.status_label = self._label("Preparing export…", "Export status")
        layout.addWidget(self.status_label)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setAccessibleName("Array export progress")
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        self.progress_detail_label = self._label("", "Export progress details")
        layout.addWidget(self.progress_detail_label)

        self.error_label = self._label("", "Export error")
        self.error_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            | QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("&Cancel")
        self.cancel_button.setAccessibleName("Cancel array export")
        self.cancel_button.setToolTip("Stop exporting after the current operation finishes.")
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self._activate_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def _label(self, text, accessible_name):
        label = QtWidgets.QLabel(text)
        label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        # A label's default accessible name is its current text. Replacing it
        # with a fixed role name would hide changing status/error details.
        label.setAccessibleDescription(accessible_name)
        label.setWordWrap(True)
        return label

    def set_output_summary(self, output_shape, dtype=None):
        """Refresh metadata once worker preflight has resolved the output dtype."""
        shape = tuple(int(size) for size in output_shape)
        shape_text = "Scalar" if not shape else "[" + " × ".join(map(str, shape)) + "]"
        summary = f"Selected output: {shape_text}"
        if dtype is None:
            summary += " · dtype determined during evaluation"
        else:
            dtype = np.dtype(dtype)
            summary += f" · {dtype} · {format_bytes(prod(shape) * dtype.itemsize)} of array data"
        self.summary_label.setText(summary)

    def apply_progress(self, event):
        """Apply an export observation on the GUI thread."""
        if not self._working:
            return
        phase = getattr(event.phase, "value", event.phase)
        if not self._cancelling:
            self.status_label.setText(
                event.message or _PHASE_TEXT.get(phase, "Exporting selected array…")
            )
        if phase == "writing" and event.total_bytes is not None and event.total_bytes >= 0:
            total = int(event.total_bytes)
            completed = max(0, min(total, int(event.completed_bytes or 0)))
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(round(completed * 1000 / total) if total else 1000)
            if not self._cancelling:
                self.progress_detail_label.setText(
                    f"{format_bytes(completed)} of {format_bytes(total)}"
                )
        else:
            self.progress_bar.setRange(0, 0)
            if not self._cancelling:
                self.progress_detail_label.clear()

    def set_cancelling(self, reason=""):
        """Keep feedback visible while the controller stops the current work."""
        if not self._working:
            return
        self._cancelling = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("Cancelling…")
        self.status_label.setText("Cancelling export…")
        explanation = (
            "Waiting for the current operation to finish. This window will close when export stops."
        )
        self.progress_detail_label.setText(f"{reason}\n{explanation}" if reason else explanation)

    def set_error(self, message):
        """Show a terminal, closable failure without requesting cancellation."""
        self._working = False
        self.status_label.setText("Export failed")
        self.progress_bar.hide()
        self.progress_detail_label.hide()
        self.error_label.setText(str(message))
        self.error_label.show()
        self.cancel_button.setText("&Close")
        self.cancel_button.setAccessibleName("Close export error")
        self.cancel_button.setToolTip("Close this export error.")
        self.cancel_button.setEnabled(True)
        self.cancel_button.setDefault(True)
        self.show()

    def close_quietly(self):
        """Let the controller close after the worker has finished."""
        self._working = False
        self.close()

    def _activate_button(self):
        if not self._working:
            self.close_quietly()
        elif not self._cancelling:
            self.set_cancelling()
            self.cancel_requested.emit()

    def reject(self):
        # QDialog handles Escape through reject(), bypassing closeEvent().
        if self._working:
            self._activate_button()
        else:
            super().reject()

    def closeEvent(self, event):
        if self._working:
            self._activate_button()
            event.ignore()
            return
        super().closeEvent(event)
