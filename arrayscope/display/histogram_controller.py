"""Histogram interaction helpers for ImageView2D."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore


class HistogramLevelPreviewController(QtCore.QObject):
    def __init__(self, owner, *, interval_ms: int = 33):
        super().__init__(owner)
        self.owner = owner
        self.interval_ms = int(interval_ms)
        self.pending_levels = None
        self.timer = QtCore.QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.flush_preview)

    def schedule_from_widget(self) -> None:
        self.pending_levels = self._widget_levels()
        if self.pending_levels is None:
            return
        if not self.timer.isActive():
            self.timer.start(max(1, int(self.interval_ms)))

    def finish_from_widget(self) -> None:
        levels = self._widget_levels()
        if levels is not None:
            self.pending_levels = levels
        self.flush_preview()
        self.owner.userLevelsChanged.emit()

    def flush_preview(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        levels = self.pending_levels
        self.pending_levels = None
        if levels is None:
            return
        self.owner._apply_histogram_preview_levels(levels)

    def cancel(self) -> None:
        self.pending_levels = None
        if self.timer.isActive():
            self.timer.stop()

    def _widget_levels(self):
        try:
            levels = self.owner.histogram.getLevels()
        except Exception:
            return None
        if levels is None:
            return None
        low, high = levels
        return (float(low), float(high))
