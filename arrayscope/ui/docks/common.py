"""Shared dock-widget chrome helpers."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets


class StandardDockWidget(QtWidgets.QDockWidget):
    def close(self):
        snapshot, layout_manager = self._preserve_snapshot()
        result = super().close()
        self._restore_snapshot(snapshot, layout_manager)
        return result

    def setVisible(self, visible):
        snapshot, layout_manager = (None, None)
        if not bool(visible) and self.isVisible():
            snapshot, layout_manager = self._preserve_snapshot()
        super().setVisible(bool(visible))
        self._restore_snapshot(snapshot, layout_manager)

    def closeEvent(self, event):
        snapshot, layout_manager = self._preserve_snapshot()
        super().closeEvent(event)
        self._restore_snapshot(snapshot, layout_manager)

    def _preserve_snapshot(self):
        parent = self.parentWidget()
        layout_manager = getattr(parent, "layout_manager", None)
        if layout_manager is None or self.isFloating() or getattr(layout_manager, "_visibility_preserve_active", False):
            return None, layout_manager
        return layout_manager._view_snapshot(), layout_manager

    def _restore_snapshot(self, snapshot, layout_manager):
        if layout_manager is not None and snapshot is not None:
            Qt.QtCore.QTimer.singleShot(0, lambda snapshot=snapshot: layout_manager._restore_view_snapshot_if_reasonable(snapshot))


def configure_standard_dock(dock: QtWidgets.QDockWidget, *, min_size=(280, 180)) -> None:
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )
    dock.setMinimumSize(int(min_size[0]), int(min_size[1]))


def add_size_grip(layout: QtWidgets.QBoxLayout) -> QtWidgets.QSizeGrip:
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    grip = QtWidgets.QSizeGrip(layout.parentWidget())
    grip.setToolTip("Resize")
    row.addWidget(grip, 0)
    layout.addLayout(row)
    return grip
