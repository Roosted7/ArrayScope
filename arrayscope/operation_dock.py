"""Qt operation-stack dock for ArrayScope."""

from __future__ import annotations

from .qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from .operation_registry import describe_operation


class OperationStackDock(QtWidgets.QDockWidget):
    def __init__(
        self,
        parent,
        on_undo,
        on_clear,
        on_save_recipe,
        on_load_recipe,
        on_materialize,
    ):
        super().__init__("Operations", parent)
        self._on_undo = on_undo
        self._on_clear = on_clear
        self._on_save_recipe = on_save_recipe
        self._on_load_recipe = on_load_recipe
        self._on_materialize = on_materialize

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.operation_list = QtWidgets.QListWidget()
        self.operation_list.setAlternatingRowColors(True)
        layout.addWidget(self.operation_list, 1)

        button_layout = QtWidgets.QGridLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.save_button = QtWidgets.QPushButton("Save Recipe")
        self.load_button = QtWidgets.QPushButton("Load Recipe")
        self.materialize_button = QtWidgets.QPushButton("Materialize")

        button_layout.addWidget(self.undo_button, 0, 0)
        button_layout.addWidget(self.clear_button, 0, 1)
        button_layout.addWidget(self.save_button, 1, 0)
        button_layout.addWidget(self.load_button, 1, 1)
        button_layout.addWidget(self.materialize_button, 2, 0, 1, 2)
        layout.addLayout(button_layout)

        body.setLayout(layout)
        self.setWidget(body)

        self.undo_button.clicked.connect(self._on_undo)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save_recipe)
        self.load_button.clicked.connect(self._on_load_recipe)
        self.materialize_button.clicked.connect(self._on_materialize)

        self.setAllowedAreas(
            Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

    def set_operations(self, operations):
        self.operation_list.clear()
        for index, operation in enumerate(operations, start=1):
            self.operation_list.addItem(f"{index}. {describe_operation(operation)}")

        has_operations = bool(operations)
        self.undo_button.setEnabled(has_operations)
        self.clear_button.setEnabled(has_operations)
        self.save_button.setEnabled(has_operations)
