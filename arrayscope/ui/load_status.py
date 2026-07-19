"""Status-bar widget showing how much of a streaming file is available.

Shown in the viewer while a file keeps loading in the background (the
viewer opens as soon as the destination array is allocated). The bar
reports the fraction of the file actually read so far.
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets


class LoadStatusWidget(QtWidgets.QWidget):
    """Compact progress readout for a background file load."""

    cancel_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._label = QtWidgets.QLabel("Loading… 0% available")
        layout.addWidget(self._label)

        self._bar = QtWidgets.QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedWidth(120)
        self._bar.setFixedHeight(12)
        layout.addWidget(self._bar)

        self._cancel_btn = QtWidgets.QToolButton()
        self._cancel_btn.setText("✕")
        self._cancel_btn.setAutoRaise(True)
        self._cancel_btn.setToolTip("Stop loading (keeps the data read so far)")
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self._cancel_btn)

    def apply_progress(self, event):
        if event.fraction is None:
            self._bar.setRange(0, 0)
            self._label.setText(event.message or "Loading…")
            return
        fraction = max(0.0, min(1.0, event.fraction))
        self._bar.setRange(0, 1000)
        self._bar.setValue(round(1000 * fraction))
        self._label.setText(f"Loading… {fraction:.0%} available")
