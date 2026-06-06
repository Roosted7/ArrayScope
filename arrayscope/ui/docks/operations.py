"""Qt operation-stack dock for ArrayScope."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations.registry import describe_operation
from arrayscope.ui.icons import material_icon, set_button_icon


class ElidedLabel(QtWidgets.QLabel):
    def __init__(self, text="", parent=None):
        super().__init__("", parent)
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)

    def setFullText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        text = self.fontMetrics().elidedText(self._full_text, QtCore.Qt.TextElideMode.ElideRight, max(8, self.width()))
        painter.drawText(self.rect(), self.alignment() | QtCore.Qt.AlignmentFlag.AlignVCenter, text)


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
        on_add_operation=None,
        on_export_derived=None,
        on_save_view_recipe=None,
        on_load_view_recipe=None,
        on_enabled_changed=None,
        on_edit_operation=None,
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
        self._on_add_operation = on_add_operation
        self._on_export_derived = on_export_derived
        self._on_save_view_recipe = on_save_view_recipe
        self._on_load_view_recipe = on_load_view_recipe
        self._on_enabled_changed = on_enabled_changed
        self._on_edit_operation = on_edit_operation
        self._operations = ()
        self._steps = ()
        self._operation_shapes = ()
        self._operation_dtypes = ()
        self._output_shape = None
        self._cache_status = None
        self._image_cache_status = None
        self._profile_cache_status = None
        self._derived_estimate = None

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add")
        self.palette_button = QtWidgets.QPushButton("Search")
        set_button_icon(self.add_button, "add")
        set_button_icon(self.palette_button, "search")
        header.addWidget(self.add_button)
        header.addWidget(self.palette_button)
        header.addStretch(1)
        layout.addLayout(header)

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
        self.cache_status_label = QtWidgets.QLabel("View cache: Cold")
        self.profile_cache_status_label = QtWidgets.QLabel("Profile/pixel cache: Cold")
        self.derived_estimate_label = QtWidgets.QLabel("Full derived: -")
        layout.addWidget(self.shape_label)
        layout.addWidget(self.cache_status_label)
        layout.addWidget(self.profile_cache_status_label)
        layout.addWidget(self.derived_estimate_label)

        button_layout = QtWidgets.QGridLayout()
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.save_button = QtWidgets.QPushButton("Save Recipe")
        self.load_button = QtWidgets.QPushButton("Load Recipe")
        self.materialize_button = QtWidgets.QPushButton("Materialize")
        self.export_button = QtWidgets.QPushButton("Export Derived")
        self.save_view_button = QtWidgets.QPushButton("Save View")
        self.load_view_button = QtWidgets.QPushButton("Load View")
        for button, icon_name in (
            (self.undo_button, "undo"),
            (self.clear_button, "delete_sweep"),
            (self.delete_button, "delete"),
            (self.save_button, "save"),
            (self.load_button, "folder_open"),
            (self.materialize_button, "inventory_2"),
            (self.export_button, "download"),
            (self.save_view_button, "view_quilt"),
            (self.load_view_button, "folder_open"),
        ):
            set_button_icon(button, icon_name)

        button_layout.addWidget(self.undo_button, 0, 0)
        button_layout.addWidget(self.clear_button, 0, 1)
        button_layout.addWidget(self.delete_button, 1, 0)
        button_layout.addWidget(self.save_button, 1, 1)
        button_layout.addWidget(self.load_button, 2, 0)
        button_layout.addWidget(self.materialize_button, 2, 1)
        button_layout.addWidget(self.export_button, 3, 0)
        button_layout.addWidget(self.save_view_button, 3, 1)
        button_layout.addWidget(self.load_view_button, 4, 0, 1, 2)
        layout.addLayout(button_layout)

        body.setLayout(layout)
        self.setWidget(body)

        self.undo_button.clicked.connect(self._on_undo)
        self.clear_button.clicked.connect(self._on_clear)
        self.save_button.clicked.connect(self._on_save_recipe)
        self.load_button.clicked.connect(self._on_load_recipe)
        self.materialize_button.clicked.connect(self._on_materialize)
        self.delete_button.clicked.connect(lambda: self._on_delete_selected(self.current_operation_index()))
        self.add_button.clicked.connect(lambda: self._on_add_operation() if self._on_add_operation is not None else None)
        self.palette_button.clicked.connect(lambda: self._on_add_operation(search=True) if self._on_add_operation is not None else None)
        self.export_button.clicked.connect(lambda: self._on_export_derived() if self._on_export_derived is not None else None)
        self.save_view_button.clicked.connect(lambda: self._on_save_view_recipe() if self._on_save_view_recipe is not None else None)
        self.load_view_button.clicked.connect(lambda: self._on_load_view_recipe() if self._on_load_view_recipe is not None else None)
        self.operation_list.currentRowChanged.connect(lambda _row: self._update_button_state())
        self.operation_list.customContextMenuRequested.connect(self._show_context_menu)

        self.setAllowedAreas(
            Qt.QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )

    def set_operations(
        self,
        operations,
        output_shape=None,
        cache_status=None,
        image_cache_status=None,
        profile_cache_status=None,
        derived_estimate=None,
        operation_shapes=None,
        steps=None,
        operation_dtypes=None,
    ):
        self._operations = tuple(operations)
        self._steps = tuple(steps or ())
        self._operation_shapes = tuple(operation_shapes or ())
        self._operation_dtypes = tuple(operation_dtypes or ())
        self._output_shape = output_shape
        self._cache_status = cache_status
        self._image_cache_status = image_cache_status
        self._profile_cache_status = profile_cache_status
        self._derived_estimate = derived_estimate
        previous_row = self.operation_list.currentRow()
        self.operation_list.clear()
        row_count = len(self._steps) if self._steps else len(operations)
        if row_count:
            for row in range(row_count):
                operation = self._steps[row].operation if self._steps else operations[row]
                item = QtWidgets.QListWidgetItem()
                item.setData(Qt.QtCore.Qt.ItemDataRole.UserRole, row)
                item.setSizeHint(Qt.QtCore.QSize(220, 58))
                item.setToolTip("Drag to reorder. Right-click for operation actions.")
                flags = item.flags()
                flags |= Qt.QtCore.Qt.ItemFlag.ItemIsDragEnabled | Qt.QtCore.Qt.ItemFlag.ItemIsDropEnabled
                item.setFlags(flags)
                self.operation_list.addItem(item)
                self.operation_list.setItemWidget(item, self._row_widget(row, operation))
            if 0 <= previous_row < row_count:
                self.operation_list.setCurrentRow(previous_row)
        else:
            self.operation_list.addItem("No operations")
            self.operation_list.item(0).setFlags(Qt.QtCore.Qt.ItemFlag.NoItemFlags)

        has_operations = bool(row_count)
        self.undo_button.setEnabled(has_operations)
        self.clear_button.setEnabled(has_operations)
        self.save_button.setEnabled(has_operations)
        self.export_button.setEnabled(True)
        self.save_view_button.setEnabled(True)
        self.shape_label.setText(f"Output shape: {tuple(output_shape) if output_shape is not None else '-'}")
        if image_cache_status is None:
            image_cache_status = cache_status
        if image_cache_status is not None:
            self.cache_status_label.setText(f"View cache: {_cache_status_summary(image_cache_status)}")
            self.cache_status_label.setToolTip(_cache_status_tooltip(image_cache_status))
            self.cache_status_label.setStyleSheet(_cache_status_style(image_cache_status.status.value))
        if profile_cache_status is not None:
            self.profile_cache_status_label.setText(f"Profile/pixel cache: {_cache_status_summary(profile_cache_status)}")
            self.profile_cache_status_label.setToolTip(_cache_status_tooltip(profile_cache_status))
            self.profile_cache_status_label.setStyleSheet(_cache_status_style(profile_cache_status.status.value))
        if derived_estimate is not None:
            shape, dtype, nbytes = derived_estimate
            self.derived_estimate_label.setText(f"Full derived: {tuple(shape)} {dtype} {_format_nbytes(nbytes)}")
            self.derived_estimate_label.setToolTip(
                f"Estimated full materialized derived array\nshape: {tuple(shape)}\ndtype: {dtype}\nsize: {_format_nbytes(nbytes)}"
            )
        self._update_button_state()

    def _operation_text(self, index, operation):
        text = f"{index}. {describe_operation(operation)}"
        if index - 1 < len(self._operation_shapes):
            text += f"\n   -> shape {tuple(self._operation_shapes[index - 1])}"
        return text

    def _row_widget(self, index, operation):
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(5)
        drag = QtWidgets.QLabel()
        drag.setPixmap(material_icon("drag_indicator").pixmap(18, 18))
        drag.setToolTip("Drag to reorder")
        layout.addWidget(drag)
        enabled = QtWidgets.QCheckBox()
        enabled.setChecked(self._steps[index].enabled if self._steps else True)
        enabled.setToolTip("Enable operation")
        enabled.toggled.connect(lambda checked, index=index: self._on_enabled_changed(index, checked) if self._on_enabled_changed is not None else None)
        layout.addWidget(enabled)
        text_col = QtWidgets.QVBoxLayout()
        title_text = describe_operation(operation)
        title = ElidedLabel(title_text)
        text_col.addWidget(title)
        full_meta = self._operation_meta_text(index, compact=False)
        compact_meta = self._operation_meta_text(index, compact=True)
        meta = ElidedLabel(compact_meta)
        meta.setToolTip(full_meta)
        meta.setStyleSheet("QLabel { color: palette(mid); font-size: 8pt; }")
        text_col.addWidget(meta)
        layout.addLayout(text_col, 1)
        edit = QtWidgets.QToolButton()
        set_button_icon(edit, "edit", tooltip="Edit operation")
        edit.setFixedSize(24, 24)
        edit.setEnabled(type(operation).__name__ == "Crop")
        edit.clicked.connect(lambda _checked=False, index=index: self._on_edit_operation(index) if self._on_edit_operation is not None else None)
        layout.addWidget(edit)
        delete = QtWidgets.QToolButton()
        set_button_icon(delete, "delete", tooltip="Delete operation")
        delete.setFixedSize(24, 24)
        delete.clicked.connect(lambda _checked=False, index=index: self._on_delete_selected(index))
        layout.addWidget(delete)
        row.setLayout(layout)
        return row

    def _operation_meta_text(self, index, *, compact=False):
        parts = []
        if index < len(self._operation_shapes):
            shape = tuple(self._operation_shapes[index])
            parts.append(str(shape) if compact else f"shape {shape}")
            dtype = self._operation_dtypes[index] if index < len(self._operation_dtypes) else None
            if dtype is not None:
                parts.append(str(dtype) if compact else f"dtype {dtype}")
                parts.append(_format_nbytes(_estimate_nbytes(shape, dtype)))
        return " | ".join(parts)

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
        row_count = len(self._steps) if self._steps else len(self._operations)
        move_down_action.setEnabled(index < row_count - 1)
        action = menu.exec(self.operation_list.mapToGlobal(pos))
        if action == delete_action:
            self._on_delete_selected(index)
        elif action == move_up_action:
            self._on_move_selected_up(index)
        elif action == move_down_action:
            self._on_move_selected_down(index)

    def _handle_reorder(self, order):
        row_count = len(self._steps) if self._steps else len(self._operations)
        if len(order) != row_count:
            self.set_operations(
                self._operations,
                output_shape=self._output_shape,
                cache_status=self._cache_status,
                image_cache_status=self._image_cache_status,
                profile_cache_status=self._profile_cache_status,
                derived_estimate=self._derived_estimate,
                operation_shapes=self._operation_shapes,
            )
            return False
        accepted = self._on_reorder(order)
        if not accepted:
            self.set_operations(
                self._operations,
                output_shape=self._output_shape,
                cache_status=self._cache_status,
                image_cache_status=self._image_cache_status,
                profile_cache_status=self._profile_cache_status,
                derived_estimate=self._derived_estimate,
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


def _cache_status_summary(cache_status):
    text = cache_status.status.value
    last_eval_ms = getattr(cache_status, "last_eval_ms", None)
    if last_eval_ms is not None:
        text += f", {last_eval_ms:.0f} ms"
    bytes_used = getattr(cache_status, "bytes_used", None)
    max_bytes = getattr(cache_status, "max_bytes", None)
    if bytes_used is not None and max_bytes:
        text += f", {_format_nbytes(bytes_used)}/{_format_nbytes(max_bytes)}"
    return text


def _cache_status_tooltip(cache_status):
    parts = [getattr(cache_status, "message", "")]
    for label, attr in (
        ("Entries", "entries"),
        ("Hits", "hits"),
        ("Misses", "misses"),
        ("Evictions", "evictions"),
    ):
        if hasattr(cache_status, attr):
            parts.append(f"{label}: {getattr(cache_status, attr)}")
    if getattr(cache_status, "last_eval_ms", None) is not None:
        parts.append(f"Last evaluation: {cache_status.last_eval_ms:.1f} ms")
    if hasattr(cache_status, "bytes_used"):
        parts.append(f"Memory: {_format_nbytes(cache_status.bytes_used)} / {_format_nbytes(cache_status.max_bytes)}")
    return "\n".join(part for part in parts if part)


def _estimate_nbytes(shape, dtype):
    try:
        import numpy as np

        return int(np.prod(tuple(shape), dtype=np.int64)) * np.dtype(dtype).itemsize
    except Exception:
        return 0


def _format_nbytes(nbytes):
    nbytes = int(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
