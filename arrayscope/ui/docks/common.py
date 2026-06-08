"""Shared dock-widget chrome helpers."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets


class StandardDockWidget(QtWidgets.QDockWidget):
    pass


def configure_standard_dock(dock: QtWidgets.QDockWidget, *, min_size=(280, 180)) -> None:
    dock.setFeatures(
        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
    )
    body = QtWidgets.QDockWidget.widget(dock)
    if body is not None:
        body.setMinimumSize(int(min_size[0]), int(min_size[1]))


def add_size_grip(layout: QtWidgets.QBoxLayout) -> QtWidgets.QSizeGrip:
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    grip = QtWidgets.QSizeGrip(layout.parentWidget())
    grip.setToolTip("Resize")
    row.addWidget(grip, 0)
    layout.addLayout(row)
    return grip
