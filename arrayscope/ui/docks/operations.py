"""Qt operation-stack dock for ArrayScope."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations.registry import describe_operation, get_operation_entry, operation_id_for
from arrayscope.ui.docks.common import StandardDockWidget, add_size_grip, configure_standard_dock
from arrayscope.ui.icons import material_icon, set_button_icon


def _operation_name(operation) -> str:
    """Short display name: the registry label without axis phrasing."""
    try:
        entry = get_operation_entry(operation_id_for(operation))
    except Exception:
        return type(operation).__name__
    label = entry.label.rstrip(".")
    for suffix in (" over axis", " axis"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label


def _operation_params_text(operation) -> str:
    """Editable parameters line: axis plus registered parameter values."""
    try:
        entry = get_operation_entry(operation_id_for(operation))
    except Exception:
        return ""
    parts = []
    if entry.requires_axis:
        parts.append(f"axis {getattr(operation, 'axis', '?')}")
    for parameter in entry.parameters:
        parts.append(f"{parameter.name}={getattr(operation, parameter.name, '?')}")
    return " · ".join(parts)


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


class OperationStackDock(StandardDockWidget):
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
        on_sync_toggled=None,
        on_change_axis=None,
        axis_choices_provider=None,
    ):
        super().__init__("Operations", parent)
        self.setObjectName("OperationsDock")
        self._on_sync_toggled = on_sync_toggled
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
        self._on_change_axis = on_change_axis
        self._axis_choices_provider = axis_choices_provider
        self._operations = ()
        self._steps = ()
        self._operation_shapes = ()
        self._operation_dtypes = ()
        self._output_shape = None
        self._derived_estimate = None
        self._row_snapshot_key = None

        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("Add operation")
        self.palette_button = QtWidgets.QToolButton()
        set_button_icon(self.add_button, "add")
        set_button_icon(self.palette_button, "search", tooltip="Search operations and commands (Ctrl+K)")
        header.addWidget(self.add_button)
        header.addWidget(self.palette_button)
        header.addStretch(1)
        self.sync_button = QtWidgets.QToolButton()
        self.sync_button.setCheckable(True)
        set_button_icon(
            self.sync_button,
            "link",
            tooltip="Sync operations with other linked ArrayScope windows (also from separately started sessions)",
        )
        self.sync_button.toggled.connect(
            lambda checked: self._on_sync_toggled(bool(checked)) if self._on_sync_toggled is not None else None
        )
        header.addWidget(self.sync_button)
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
        self.derived_estimate_label = QtWidgets.QLabel("Full derived: -")
        for label in (self.shape_label, self.derived_estimate_label):
            label.setObjectName("OperationsMetaLabel")
        layout.addWidget(self.shape_label)
        layout.addWidget(self.derived_estimate_label)

        # Frequent actions get a compact row; recipe/view/export management is
        # tucked into a single "More" menu so rare actions don't crowd the dock.
        self.undo_button = QtWidgets.QToolButton()
        self.delete_button = QtWidgets.QToolButton()
        self.clear_button = QtWidgets.QToolButton()
        self.materialize_button = QtWidgets.QPushButton("Materialize")
        set_button_icon(self.undo_button, "undo", tooltip="Undo last operation")
        set_button_icon(self.delete_button, "delete", tooltip="Delete selected operation")
        set_button_icon(self.clear_button, "delete_sweep", tooltip="Clear all operations")
        set_button_icon(self.materialize_button, "inventory_2")
        self.materialize_button.setToolTip("Evaluate the full derived array and keep it resident")

        # Rare actions stay as real buttons (stable API, signal wiring) but are
        # presented as menu entries rather than a wall of buttons.
        self.save_button = QtWidgets.QPushButton("Save Recipe")
        self.load_button = QtWidgets.QPushButton("Load Recipe")
        self.export_button = QtWidgets.QPushButton("Export Derived")
        self.save_view_button = QtWidgets.QPushButton("Save View")
        self.load_view_button = QtWidgets.QPushButton("Load View")
        for button in (self.save_button, self.load_button, self.export_button, self.save_view_button, self.load_view_button):
            button.setVisible(False)

        self.more_button = QtWidgets.QToolButton()
        self.more_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_button.setText("More")
        set_button_icon(self.more_button, "folder_open", tooltip="Recipes, views and export")
        more_menu = QtWidgets.QMenu(self.more_button)
        for entry in (
            ("Save Operation Recipe…", "save", self.save_button),
            ("Load Operation Recipe…", "folder_open", self.load_button),
            None,
            ("Save View Recipe…", "view_quilt", self.save_view_button),
            ("Load View Recipe…", "folder_open", self.load_view_button),
            None,
            ("Export Derived Array…", "download", self.export_button),
        ):
            if entry is None:
                more_menu.addSeparator()
                continue
            label, icon_name, proxy = entry
            action = more_menu.addAction(material_icon(icon_name), label)
            action.triggered.connect(lambda _checked=False, button=proxy: button.click())
        self.more_button.setMenu(more_menu)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(4)
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.materialize_button)
        button_layout.addWidget(self.more_button)
        layout.addLayout(button_layout)
        add_size_grip(layout)

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
        configure_standard_dock(self, min_size=(300, 260))
        self._footer_compact = False

    def _sync_footer_compaction(self):
        """Collapse footer button text to icons when the dock gets narrow."""
        compact = self.width() < 360
        if compact == self._footer_compact:
            return
        self._footer_compact = compact
        self.materialize_button.setText("" if compact else "Materialize")
        self.more_button.setText("" if compact else "More")
        self.more_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
            if compact
            else QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_footer_compaction()

    def set_operations(
        self,
        operations,
        output_shape=None,
        cache_status=None,
        display_cache_status=None,
        profile_cache_status=None,
        derived_estimate=None,
        operation_shapes=None,
        steps=None,
        operation_dtypes=None,
    ):
        operations = tuple(operations)
        steps = tuple(steps or ())
        operation_shapes = tuple(operation_shapes or ())
        operation_dtypes = tuple(operation_dtypes or ())
        row_snapshot_key = _operation_row_snapshot_key(
            operations,
            steps=steps,
            operation_shapes=operation_shapes,
            operation_dtypes=operation_dtypes,
        )
        rows_unchanged = row_snapshot_key == self._row_snapshot_key
        self._operations = operations
        self._steps = steps
        self._operation_shapes = operation_shapes
        self._operation_dtypes = operation_dtypes
        self._output_shape = output_shape
        self._cache_status = cache_status
        self._display_cache_status = display_cache_status
        self._profile_cache_status = profile_cache_status
        self._derived_estimate = derived_estimate
        previous_row = self.operation_list.currentRow()
        row_count = len(self._steps) if self._steps else len(operations)
        if not rows_unchanged:
            self.operation_list.clear()
        if rows_unchanged:
            pass
        elif row_count:
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
        self._row_snapshot_key = row_snapshot_key

        has_operations = bool(row_count)
        self.undo_button.setEnabled(has_operations)
        self.clear_button.setEnabled(has_operations)
        self.save_button.setEnabled(has_operations)
        self.export_button.setEnabled(True)
        self.save_view_button.setEnabled(True)
        self.shape_label.setText(f"Output shape: {tuple(output_shape) if output_shape is not None else '-'}")
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

        left_col = QtWidgets.QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(1)
        axis = getattr(operation, "axis", None)
        axis_button = QtWidgets.QToolButton()
        axis_button.setObjectName("OperationAxisButton")
        axis_button.setText("" if axis is None else f"d{int(axis)}")
        axis_button.setToolTip("Change the dimension this operation applies to")
        axis_button.setVisible(axis is not None)
        if axis is not None:
            axis_button.clicked.connect(
                lambda _checked=False, index=index: self._show_axis_menu(index)
            )
        left_col.addWidget(axis_button, 0, Qt.QtCore.Qt.AlignmentFlag.AlignHCenter)
        enabled = QtWidgets.QCheckBox()
        enabled.setChecked(self._steps[index].enabled if self._steps else True)
        enabled.setToolTip("Enable operation")
        enabled.toggled.connect(lambda checked, index=index: self._on_enabled_changed(index, checked) if self._on_enabled_changed is not None else None)
        left_col.addWidget(enabled, 0, Qt.QtCore.Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(left_col)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setSpacing(0)
        title = ElidedLabel(_operation_name(operation))
        title.setToolTip(describe_operation(operation))
        text_col.addWidget(title)
        params_text = _operation_params_text(operation)
        if params_text:
            params = ElidedLabel(params_text)
            params.setToolTip(params_text)
            params.setStyleSheet("QLabel { font-size: 8pt; }")
            text_col.addWidget(params)
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
        # Only operations with editable parameters get the button at all.
        edit.setVisible(type(operation).__name__ == "Crop")
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

    def _operation_at(self, index):
        if self._steps and 0 <= index < len(self._steps):
            return self._steps[index].operation
        if 0 <= index < len(self._operations):
            return self._operations[index]
        return None

    def _populate_axis_menu(self, menu, index):
        provider = self._axis_choices_provider
        choices = provider() if callable(provider) else ()
        operation = self._operation_at(index)
        current_axis = getattr(operation, "axis", None)
        for label, axis in choices:
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current_axis is not None and int(axis) == int(current_axis))
            action.triggered.connect(
                lambda _checked=False, index=index, axis=axis: self._on_change_axis(index, int(axis))
                if self._on_change_axis is not None
                else None
            )
        return bool(choices)

    def _show_axis_menu(self, index):
        menu = QtWidgets.QMenu(self)
        if self._populate_axis_menu(menu, index):
            menu.exec(QtGui.QCursor.pos())

    def _show_context_menu(self, pos):
        item = self.operation_list.itemAt(pos)
        if item is not None:
            self.operation_list.setCurrentItem(item)
        index = self.current_operation_index()
        if index is None:
            return

        operation = self._operation_at(index)
        step_enabled = self._steps[index].enabled if self._steps and index < len(self._steps) else True
        menu = QtWidgets.QMenu(self.operation_list)
        toggle_action = menu.addAction(
            material_icon("close" if step_enabled else "done"),
            "Disable operation" if step_enabled else "Enable operation",
        )
        axis_menu = menu.addMenu(material_icon("call_split"), "Change dimension")
        if getattr(operation, "axis", None) is None or not self._populate_axis_menu(axis_menu, index):
            axis_menu.menuAction().setEnabled(False)
        edit_action = menu.addAction(material_icon("edit"), "Edit parameters…")
        edit_action.setEnabled(type(operation).__name__ == "Crop")
        menu.addSeparator()
        move_up_action = menu.addAction(material_icon("arrow_upward"), "Move up")
        move_down_action = menu.addAction(material_icon("arrow_downward"), "Move down")
        move_up_action.setEnabled(index > 0)
        row_count = len(self._steps) if self._steps else len(self._operations)
        move_down_action.setEnabled(index < row_count - 1)
        menu.addSeparator()
        delete_action = menu.addAction(material_icon("delete"), "Delete operation")
        action = menu.exec(self.operation_list.mapToGlobal(pos))
        if action == delete_action:
            self._on_delete_selected(index)
        elif action == toggle_action:
            if self._on_enabled_changed is not None:
                self._on_enabled_changed(index, not step_enabled)
        elif action == edit_action:
            if self._on_edit_operation is not None:
                self._on_edit_operation(index)
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
                derived_estimate=self._derived_estimate,
                operation_shapes=self._operation_shapes,
            )
            return False
        accepted = self._on_reorder(order)
        if not accepted:
            self.set_operations(
                self._operations,
                output_shape=self._output_shape,
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
        ("Hit rate", "hit_rate"),
        ("Evictions", "evictions"),
        ("Chunked renders", "chunked_evaluations"),
        ("Degraded previews", "degraded_evaluations"),
        ("Refused renders", "refused_evaluations"),
        ("Cancelled renders", "cancelled_evaluations"),
        ("Scheduler pending", "scheduler_pending"),
        ("Scheduler running", "scheduler_running"),
        ("Scheduler cancelled", "scheduler_cancelled"),
        ("Scheduler stale", "scheduler_stale"),
    ):
        if hasattr(cache_status, attr):
            value = getattr(cache_status, attr)
            if attr == "hit_rate" and value is not None:
                value = f"{100.0 * float(value):.1f}%"
            parts.append(f"{label}: {value}")
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


def _operation_row_snapshot_key(operations, *, steps, operation_shapes, operation_dtypes):
    if steps:
        operation_key = tuple(
            (
                type(step.operation),
                step.operation,
                bool(getattr(step, "enabled", True)),
            )
            for step in steps
        )
    else:
        operation_key = tuple((type(operation), operation) for operation in operations)
    return (
        operation_key,
        tuple(tuple(shape) for shape in operation_shapes),
        tuple(None if dtype is None else str(dtype) for dtype in operation_dtypes),
    )


def _format_nbytes(nbytes):
    nbytes = int(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or unit == "GiB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
