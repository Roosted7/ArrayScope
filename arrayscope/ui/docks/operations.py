"""Qt operation-stack dock for ArrayScope."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.operations.registry import describe_operation


class OperationListWidget(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._on_reorder = None

    def set_reorder_callback(self, callback):
        self._on_reorder = callback

    def dropEvent(self, event):
        before = [self.item(row).data(Qt.QtCore.Qt.ItemDataRole.UserRole) for row in range(self.count())]
        super().dropEvent(event)
        after = [self.item(row).data(Qt.QtCore.Qt.ItemDataRole.UserRole) for row in range(self.count())]
        if before != after and self._on_reorder is not None:
            accepted = self._on_reorder(tuple(after))
            if not accepted:
                event.ignore()


class OperationStackDock(QtWidgets.QDockWidget):
    def __init__(
        self,
        parent,
        on_undo,
        on_clear,
        on_save_recipe,
        on_load_recipe,
        on_materialize,
        on_delete_selected,
        on_move_selected_up,
        on_move_selected_down,
        on_reorder,
    ):
        super().__init__("Operations", parent)
        self.setObjectName("OperationsDock")
        self._on_undo = on_undo
        self._on_clear = on_clear
        self._on_save_recipe = on_save_recipe
        self._on_load_recipe = on_load_recipe
        self._on_materialize = on_materialize
        self._on_delete_selected = on_delete_selected
        self._on_move_selected_up = on_move_selected_up
        self._on_move_selected_down = on_move_selected_down
        self._on_reorder = on_reorder
        self._operations = ()
        self._operation_shapes = ()
        self._output_shape = None
        self._cache_status = None

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.operation_list = OperationListWidget()
        self.operation_list.setAlternatingRowColors(True)
        self.operation_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.operation_list.setDefaultDropAction(Qt.QtCore.Qt.DropAction.MoveAction)
        self.operation_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.operation_list.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.operation_list.setSpacing(3)
        self.operation_list.set_reorder_callback(self._handle_reorder)
        layout.addWidget(self.operation_list, 1)

        self.shape_label = QtWidgets.QLabel("Output shape: -")
        self.cache_status_label = QtWidgets.QLabel("Cache: Cold")
        layout.addWidget(self.shape_label)
        layout.addWidget(self.cache_status_label)

        button_layout = QtWidgets.QGridLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.save_button = QtWidgets.QPushButton("Save Recipe")
        self.load_button = QtWidgets.QPushButton("Load Recipe")
        self.materialize_button = QtWidgets.QPushButton("Materialize")

        button_layout.addWidget(self.undo_button, 0, 0)
        button_layout.addWidget(self.clear_button, 0, 1)
        button_layout.addWidget(self.delete_button, 1, 0)
        button_layout.addWidget(self.save_button, 1, 1)
        button_layout.addWidget(self.load_button, 2, 0)
        button_layout.addWidget(self.materialize_button, 2, 1)
        layout.addLayout(button_layout)

        body.setLayout(layout)
        self.setWidget(body)

        self.undo_button.clicked.connect(self._on_undo)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save_recipe)
        self.load_button.clicked.connect(self._on_load_recipe)
        self.materialize_button.clicked.connect(self._on_materialize)
        self.delete_button.clicked.connect(lambda: self._on_delete_selected(self.current_operation_index()))
        self.operation_list.currentRowChanged.connect(lambda _row: self._update_button_state())
        self.operation_list.customContextMenuRequested.connect(self._show_context_menu)

        self.setAllowedAreas(
            Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

    def set_operations(self, operations, output_shape=None, cache_status=None, operation_shapes=None):
        self._operations = tuple(operations)
        self._operation_shapes = tuple(operation_shapes or ())
        self._output_shape = output_shape
        self._cache_status = cache_status
        previous_row = self.operation_list.currentRow()
        self.operation_list.clear()
        if operations:
            for index, operation in enumerate(operations, start=1):
                item = QtWidgets.QListWidgetItem(self._operation_text(index, operation))
                item.setData(Qt.QtCore.Qt.ItemDataRole.UserRole, index - 1)
                item.setSizeHint(Qt.QtCore.QSize(180, 46))
                item.setToolTip("Drag to reorder. Right-click for operation actions.")
                flags = item.flags()
                flags |= Qt.QtCore.Qt.ItemFlag.ItemIsDragEnabled | Qt.QtCore.Qt.ItemFlag.ItemIsDropEnabled
                item.setFlags(flags)
                self.operation_list.addItem(item)
            if 0 <= previous_row < len(operations):
                self.operation_list.setCurrentRow(previous_row)
        else:
            self.operation_list.addItem("No operations")
            self.operation_list.item(0).setFlags(Qt.QtCore.Qt.ItemFlag.NoItemFlags)

        has_operations = bool(operations)
        self.undo_button.setEnabled(has_operations)
        self.clear_button.setEnabled(has_operations)
        self.save_button.setEnabled(has_operations)
        self.shape_label.setText(f"Output shape: {tuple(output_shape) if output_shape is not None else '-'}")
        if cache_status is not None:
            self.cache_status_label.setText(f"Cache: {cache_status.status.value}")
            self.cache_status_label.setToolTip(cache_status.message)
            self.cache_status_label.setStyleSheet(_cache_status_style(cache_status.status.value))
        self._update_button_state()

    def _operation_text(self, index, operation):
        text = f"{index}. {describe_operation(operation)}"
        if index - 1 < len(self._operation_shapes):
            text += f"\n   -> shape {tuple(self._operation_shapes[index - 1])}"
        return text

    def current_operation_index(self):
        row = self.operation_list.currentRow()
        if row < 0:
            return None
        item = self.operation_list.item(row)
        if item is None or not (item.flags() & Qt.QtCore.Qt.ItemFlag.ItemIsSelectable):
            return None
        return row

    def _update_button_state(self):
        index = self.current_operation_index()
        has_selection = index is not None
        self.delete_button.setEnabled(has_selection)

    def _show_context_menu(self, pos):
        item = self.operation_list.itemAt(pos)
        if item is not None:
            self.operation_list.setCurrentItem(item)
        index = self.current_operation_index()
        if index is None:
            return

        menu = QtWidgets.QMenu(self.operation_list)
        delete_action = menu.addAction("Delete operation")
        move_up_action = menu.addAction("Move up")
        move_down_action = menu.addAction("Move down")
        move_up_action.setEnabled(index > 0)
        move_down_action.setEnabled(index < len(self._operations) - 1)
        action = menu.exec(self.operation_list.mapToGlobal(pos))
        if action == delete_action:
            self._on_delete_selected(index)
        elif action == move_up_action:
            self._on_move_selected_up(index)
        elif action == move_down_action:
            self._on_move_selected_down(index)

    def _handle_reorder(self, order):
        if len(order) != len(self._operations):
            self.set_operations(
                self._operations,
                output_shape=self._output_shape,
                cache_status=self._cache_status,
                operation_shapes=self._operation_shapes,
            )
            return False
        accepted = self._on_reorder(order)
        if not accepted:
            self.set_operations(
                self._operations,
                output_shape=self._output_shape,
                cache_status=self._cache_status,
                operation_shapes=self._operation_shapes,
            )
            return False
        return True


def _cache_status_style(status):
    if status == "Error":
        return "QLabel { background: rgba(180, 40, 40, 55); padding: 2px 4px; border-radius: 3px; }"
    if status in {"Cached", "Ready"}:
        return "QLabel { background: rgba(40, 140, 80, 45); padding: 2px 4px; border-radius: 3px; }"
    if status == "Computing":
        return "QLabel { background: rgba(180, 140, 40, 50); padding: 2px 4px; border-radius: 3px; }"
    return "QLabel { background: rgba(128, 128, 128, 35); padding: 2px 4px; border-radius: 3px; }"
