"""Shared dock-widget chrome helpers."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets


class StandardDockWidget(QtWidgets.QDockWidget):
    def closeEvent(self, event):
        parent = self.parent()
        layout_manager = getattr(parent, "layout_manager", None)
        if layout_manager is not None and not getattr(layout_manager, "_visibility_preserve_active", False):
            if self is getattr(parent, "inspection_dock", None):
                layout_manager.set_inspection_dock_visible_from_user(False)
            elif self is getattr(parent, "profile_dock", None):
                layout_manager.set_profile_dock_visible_from_user(False)
            elif self is getattr(parent, "operation_dock", None):
                layout_manager.set_operation_dock_visible_from_user(False)
            else:
                layout_manager.set_managed_dock_visible(self, False, reason="dock-close-button")
            event.ignore()
            return
        super().closeEvent(event)


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
