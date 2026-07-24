"""Operation manager: browse, arrange, hide, import and edit operations.

This is the operations analogue of :mod:`arrayscope.ui.colormap_designer`. It is
the flagship UI of the custom-operations feature and deliberately mirrors the
colormap designer's interaction model:

- The library tree (left) is drag-reorderable: ops move within and between
  groups and groups reorder; the arrangement persists through
  :func:`arrayscope.operations.library.apply_library_layout` and drives every
  UI surface that lists operations.
- Built-in / pack ("system") ops are read-only definitions: they can only be
  hidden (hide-not-delete, restorable) or moved to another group. User ops are
  fully editable and are removed outright.
- Edits save automatically (on edit-finish / toggle), matching the colormap
  designer -- no explicit Save button, no silent loss.
- The right column is a live editor for the selected op; the "Add" button walks
  the "connect up a custom function" import flow.

The dialog registers a library listener so an external mutation (a recipe load,
another window's edit) rebuilds the tree in place, preserving selection by id.
"""

from __future__ import annotations

import os

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations import library
from arrayscope.operations.registry import get_operation_entry
from arrayscope.ui.file_dialogs import get_open_file_name
from arrayscope.ui.icons import material_icon, set_button_icon, set_label_icon
from arrayscope.ui.toasts import show_status_message

_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole
_GROUP_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_VIRTUAL_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2

_PROBLEMS_GROUP = "Problems"
_KIND_CHOICES = ("int", "float")


def _is_user_op(operation_id: str) -> bool:
    return str(operation_id).startswith("user:")


# TODO(integration): a small ``library.user_operation_source_path(id)`` helper
# would remove this local wrapper scan (the library already has a private
# ``_wrapper_path_for_id``). Kept here so this chunk does not edit library.py.
def _user_op_source_path(operation_id: str) -> str | None:
    """Absolute path to the ``.py`` backing a user op, or ``None``."""

    import json

    directory = library.user_operations_directory()
    if not os.path.isdir(directory):
        return None
    for file_name in sorted(os.listdir(directory)):
        if not file_name.endswith(".json") or file_name in (
            "layout.json",
            "hidden-operations.json",
        ):
            continue
        path = os.path.join(directory, file_name)
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if str(payload.get("id") or "") != str(operation_id):
            continue
        source = payload.get("source") or {}
        rel_or_abs = str(source.get("path") or "")
        if not rel_or_abs:
            return None
        return rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(directory, rel_or_abs)
    return None


class _OperationTree(QtWidgets.QTreeWidget):
    """Groups as parents, operations as draggable children."""

    orderEdited = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setExpandsOnDoubleClick(False)

    def dropEvent(self, event):
        super().dropEvent(event)
        self._normalize()
        self.orderEdited.emit()

    def _normalize(self):
        """Keep the two-level shape: groups top-level, ops under groups."""

        index = 0
        while index < self.topLevelItemCount():
            item = self.topLevelItem(index)
            if item.data(0, _ID_ROLE):
                # An op dropped at the top level slides into the nearest group.
                self.takeTopLevelItem(index)
                target = self.topLevelItem(max(0, index - 1))
                if target is not None and not target.data(0, _VIRTUAL_ROLE):
                    target.addChild(item)
                    target.setExpanded(True)
                continue
            child_index = 0
            while child_index < item.childCount():
                child = item.child(child_index)
                if child.data(0, _ID_ROLE):
                    child_index += 1
                    continue
                # A group nested under a group moves back to the top level.
                item.takeChild(child_index)
                self.insertTopLevelItem(self.indexOfTopLevelItem(item) + 1, child)
            index += 1


class OperationManagerDialog(QtWidgets.QDialog):
    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self.setWindowTitle("Operations")
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.resize(780, 500)
        self._loaded_id = None
        self._suppress = False
        self._reloading = False

        root = QtWidgets.QHBoxLayout(self)

        # -- left: library tree -------------------------------------------
        left = QtWidgets.QVBoxLayout()
        self.tree = _OperationTree(self)
        self.tree.setMinimumWidth(300)
        left.addWidget(self.tree, 1)
        hint = QtWidgets.QLabel("Drag to rearrange operations and groups")
        hint.setObjectName("OperationsMetaLabel")
        left.addWidget(hint)

        tree_buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QToolButton(self)
        set_button_icon(self.add_button, "add", tooltip="Import a custom operation…")
        self.remove_button = QtWidgets.QToolButton(self)
        set_button_icon(self.remove_button, "delete", tooltip="Hide")
        self.unhide_button = QtWidgets.QToolButton(self)
        set_button_icon(self.unhide_button, "reset_wrench", tooltip="Restore / unhide")
        self.open_file_button = QtWidgets.QToolButton(self)
        set_button_icon(self.open_file_button, "edit", tooltip="Open the code file")
        self.open_folder_button = QtWidgets.QToolButton(self)
        set_button_icon(
            self.open_folder_button, "folder_open", tooltip="Open the operations folder"
        )
        self.reset_all_button = QtWidgets.QToolButton(self)
        set_button_icon(self.reset_all_button, "refresh", tooltip="Reset layout and unhide all")
        for button in (
            self.add_button,
            self.remove_button,
            self.unhide_button,
            self.open_file_button,
            self.open_folder_button,
        ):
            tree_buttons.addWidget(button)
        tree_buttons.addStretch(1)
        tree_buttons.addWidget(self.reset_all_button)
        left.addLayout(tree_buttons)
        root.addLayout(left)

        # -- right: editor ------------------------------------------------
        right = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.label_edit = QtWidgets.QLineEdit(self)
        form.addRow("Label", self.label_edit)
        self.id_label = QtWidgets.QLabel("", self)
        self.id_label.setObjectName("OperationsMetaLabel")
        self.id_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Id", self.id_label)
        self.group_combo = QtWidgets.QComboBox(self)
        self.group_combo.setEditable(True)
        form.addRow("Group", self.group_combo)
        self.description_edit = QtWidgets.QLineEdit(self)
        form.addRow("Description", self.description_edit)
        icon_row = QtWidgets.QHBoxLayout()
        self.icon_edit = QtWidgets.QLineEdit(self)
        self.icon_edit.setPlaceholderText("material icon name, e.g. extension")
        self.icon_preview = QtWidgets.QLabel(self)
        icon_row.addWidget(self.icon_edit, 1)
        icon_row.addWidget(self.icon_preview)
        self.icon_row_widget = QtWidgets.QWidget(self)
        self.icon_row_widget.setLayout(icon_row)
        icon_row.setContentsMargins(0, 0, 0, 0)
        self.icon_form_label = QtWidgets.QLabel("Icon", self)
        form.addRow(self.icon_form_label, self.icon_row_widget)
        self.requires_axis_check = QtWidgets.QCheckBox("Requires an axis", self)
        form.addRow("", self.requires_axis_check)
        self.common_check = QtWidgets.QCheckBox("Show in the Common (pinned) section", self)
        form.addRow("", self.common_check)
        right.addLayout(form)

        right.addWidget(QtWidgets.QLabel("Parameters"))
        self.params_table = QtWidgets.QTableWidget(0, 6, self)
        self.params_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Default", "Min", "Max", "Step"]
        )
        header = self.params_table.horizontalHeader()
        # Fit all six columns at the default dialog width (no horizontal
        # scrollbar, no truncated header): the name column absorbs the slack,
        # Kind hugs its combo, and the four numeric columns share the rest.
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for _column in range(2, 6):
            header.setSectionResizeMode(_column, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.params_table.verticalHeader().setVisible(False)
        right.addWidget(self.params_table, 1)
        params_buttons = QtWidgets.QHBoxLayout()
        self.add_param_button = QtWidgets.QToolButton(self)
        set_button_icon(self.add_param_button, "add", tooltip="Add a parameter")
        self.remove_param_button = QtWidgets.QToolButton(self)
        set_button_icon(self.remove_param_button, "delete", tooltip="Remove the selected parameter")
        params_buttons.addWidget(self.add_param_button)
        params_buttons.addWidget(self.remove_param_button)
        params_buttons.addStretch(1)
        right.addLayout(params_buttons)

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setObjectName("OperationsMetaLabel")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.done_button = QtWidgets.QPushButton("Done", self)
        set_button_icon(self.done_button, "done")
        actions.addWidget(self.done_button)
        right.addLayout(actions)
        root.addLayout(right, 1)

        # -- wiring -------------------------------------------------------
        self.tree.currentItemChanged.connect(lambda *_: self._load_editor())
        self.tree.orderEdited.connect(self._persist_layout)
        self.add_button.clicked.connect(self._import_operation)
        self.remove_button.clicked.connect(self._remove_or_hide)
        self.unhide_button.clicked.connect(self._unhide)
        self.open_file_button.clicked.connect(self._open_code_file)
        self.open_folder_button.clicked.connect(self._open_folder)
        self.reset_all_button.clicked.connect(self._reset_all)
        self.done_button.clicked.connect(self.accept)

        self.label_edit.editingFinished.connect(self._apply_user_edits)
        self.description_edit.editingFinished.connect(self._apply_user_edits)
        self.icon_edit.editingFinished.connect(self._apply_user_edits)
        self.icon_edit.textChanged.connect(self._update_icon_preview)
        self.requires_axis_check.toggled.connect(self._on_requires_axis_toggled)
        self.group_combo.currentTextChanged.connect(self._on_group_changed)
        self.common_check.toggled.connect(self._on_common_toggled)
        self.params_table.cellChanged.connect(lambda *_: self._apply_user_edits())
        self.add_param_button.clicked.connect(self._add_parameter_row)
        self.remove_param_button.clicked.connect(self._remove_parameter_row)

        library.add_library_listener(self._refresh_tree)
        self._refresh_tree()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        library.remove_library_listener(self._refresh_tree)
        super().closeEvent(event)

    def reject(self):
        library.remove_library_listener(self._refresh_tree)
        super().reject()

    def accept(self):
        library.remove_library_listener(self._refresh_tree)
        super().accept()

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _refresh_tree(self):
        selected = self.selected_operation_id()
        self._reloading = True
        try:
            self.tree.clear()
            hidden = library.hidden_operations()
            for group, entries in library.grouped_operations(include_hidden=True):
                group_item = self._group_item(group)
                self.tree.addTopLevelItem(group_item)
                for entry in entries:
                    group_item.addChild(self._operation_item(entry, entry.id in hidden))
                group_item.setExpanded(True)
            self._append_problems_group()
        finally:
            self._reloading = False
        if selected is None or not self.select_operation(selected):
            self._select_first_operation()
        self._load_editor()

    def _group_item(self, group: str) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([group])
        item.setData(0, _GROUP_ROLE, group)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        return item

    def _operation_item(self, entry, hidden: bool) -> QtWidgets.QTreeWidgetItem:
        markers = []
        if hidden:
            markers.append("(hidden)")
        if _is_user_op(entry.id):
            markers.append("(user)")
        suffix = ("  " + " ".join(markers)) if markers else ""
        item = QtWidgets.QTreeWidgetItem([f"{entry.label}{suffix}"])
        item.setIcon(0, material_icon(entry.icon or "data_array"))
        item.setData(0, _ID_ROLE, entry.id)
        item.setToolTip(0, entry.description or entry.id)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        if hidden:
            item.setForeground(
                0,
                self.palette().brush(
                    QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text
                ),
            )
            font = item.font(0)
            font.setItalic(True)
            item.setFont(0, font)
        return item

    def _append_problems_group(self):
        problems = library.user_operation_problems()
        if not problems:
            return
        group_item = self._group_item(_PROBLEMS_GROUP)
        group_item.setData(0, _VIRTUAL_ROLE, True)
        group_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        self.tree.addTopLevelItem(group_item)
        for path, message in problems:
            child = QtWidgets.QTreeWidgetItem([os.path.basename(path)])
            child.setData(0, _VIRTUAL_ROLE, True)
            child.setIcon(0, material_icon("warning"))
            child.setToolTip(0, message)
            child.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            group_item.addChild(child)
        group_item.setExpanded(True)

    def select_operation(self, operation_id: str) -> bool:
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                if child.data(0, _ID_ROLE) == str(operation_id):
                    self.tree.setCurrentItem(child)
                    return True
        return False

    def _select_first_operation(self):
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            if group_item.data(0, _VIRTUAL_ROLE):
                continue
            if group_item.childCount():
                self.tree.setCurrentItem(group_item.child(0))
                return

    def selected_operation_id(self):
        item = self.tree.currentItem()
        if item is None or item.data(0, _VIRTUAL_ROLE):
            return None
        value = item.data(0, _ID_ROLE)
        return None if value is None else str(value)

    # ------------------------------------------------------------------
    # Layout persistence (drag & drop)
    # ------------------------------------------------------------------

    def _tree_layout(self, override: dict | None = None):
        group_order = []
        op_groups = {}
        op_order = {}
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            if group_item.data(0, _VIRTUAL_ROLE):
                continue
            group = str(group_item.data(0, _GROUP_ROLE))
            group_order.append(group)
            for child_index in range(group_item.childCount()):
                operation_id = str(group_item.child(child_index).data(0, _ID_ROLE))
                op_groups[operation_id] = group
                op_order[operation_id] = child_index
        if override:
            op_groups.update(override)
        return group_order, op_groups, op_order

    def _persist_layout(self, override: dict | None = None):
        group_order, op_groups, op_order = self._tree_layout(override)
        library.apply_library_layout(
            group_order=group_order, op_groups=op_groups, op_order=op_order
        )

    # ------------------------------------------------------------------
    # Editor
    # ------------------------------------------------------------------

    def _load_editor(self):
        if self._reloading:
            return
        operation_id = self.selected_operation_id()
        self._loaded_id = operation_id
        if operation_id is None:
            self._set_editor_enabled(False)
            self._suppress = True
            try:
                self.label_edit.clear()
                self.id_label.clear()
                self.description_edit.clear()
                self.params_table.setRowCount(0)
            finally:
                self._suppress = False
            self.status_label.setText("Select an operation to edit it.")
            self._sync_button_states()
            return
        try:
            entry = get_operation_entry(operation_id)
        except Exception:
            self._sync_button_states()
            return
        is_user = _is_user_op(operation_id)
        hidden = operation_id in library.hidden_operations()

        self._suppress = True
        try:
            self.label_edit.setText(entry.label)
            self.id_label.setText(operation_id)
            self._reload_group_combo(library.effective_group(entry))
            self.description_edit.setText(entry.description)
            self.icon_edit.setText(entry.icon or "")
            self._update_icon_preview()
            self.requires_axis_check.setChecked(entry.requires_axis)
            self.common_check.setChecked(operation_id in library.effective_common_ids())
            self._fill_parameters(entry.parameters, editable=is_user)
        finally:
            self._suppress = False

        self._set_icon_row_visible(is_user)
        self._apply_editor_editability(is_user)
        if is_user:
            self.status_label.setText(
                "User operation — edits save automatically. Remove deletes it and its files."
            )
        elif hidden:
            self.status_label.setText(
                "Hidden system operation — Restore returns it to the listing."
            )
        else:
            self.status_label.setText(
                "System operation — its definition is read-only; hide it or move it to another group."
            )
        self._sync_button_states()

    def _reload_group_combo(self, current_group: str):
        groups = [group for group, _entries in library.grouped_operations(include_hidden=True)]
        if current_group not in groups:
            groups.append(current_group)
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self.group_combo.addItems(groups)
        self.group_combo.setCurrentText(current_group)
        self.group_combo.blockSignals(False)

    def _set_editor_enabled(self, enabled: bool):
        """Enable/disable every editor widget (used when nothing is selected)."""

        for widget in (
            self.label_edit,
            self.group_combo,
            self.description_edit,
            self.icon_edit,
            self.requires_axis_check,
            self.common_check,
            self.params_table,
            self.add_param_button,
            self.remove_param_button,
        ):
            widget.setEnabled(enabled)

    def _apply_editor_editability(self, is_user: bool):
        """Reflect the selected op's edit permissions in the widget states.

        A system op's definition is read-only: its Label, Description, Icon and
        parameters must LOOK disabled (greyed + read-only), not merely swallow
        edits. The two controls that genuinely act on a system op -- Group and
        the Common toggle -- stay enabled. A user op is fully editable.
        """

        for widget in (self.label_edit, self.description_edit, self.icon_edit):
            widget.setReadOnly(not is_user)
            widget.setEnabled(is_user)
        self.requires_axis_check.setEnabled(is_user)
        self.params_table.setEnabled(is_user)
        self.add_param_button.setEnabled(is_user)
        self.remove_param_button.setEnabled(is_user)
        # These act on system ops too, so they stay live regardless.
        self.group_combo.setEnabled(True)
        self.common_check.setEnabled(True)

    def _set_icon_row_visible(self, visible: bool):
        self.icon_form_label.setVisible(visible)
        self.icon_row_widget.setVisible(visible)

    def _update_icon_preview(self, *_args):
        name = self.icon_edit.text().strip() or "extension"
        set_label_icon(self.icon_preview, name, icon_size=18)

    def _fill_parameters(self, parameters, *, editable: bool):
        self.params_table.blockSignals(True)
        try:
            self.params_table.setRowCount(0)
            for parameter in parameters:
                self._append_parameter_row(parameter, editable=editable)
            self.params_table.setEditTriggers(
                QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
                | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
                if editable
                else QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
            )
        finally:
            self.params_table.blockSignals(False)

    def _append_parameter_row(self, parameter, *, editable: bool):
        row = self.params_table.rowCount()
        self.params_table.insertRow(row)

        def _text(value):
            return "" if value is None else str(value)

        name = getattr(parameter, "name", "") if parameter is not None else ""
        kind = getattr(parameter, "kind", "float") if parameter is not None else "float"
        self.params_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
        kind_combo = QtWidgets.QComboBox()
        kind_combo.addItems(_KIND_CHOICES)
        kind_combo.setCurrentText(kind if kind in _KIND_CHOICES else "float")
        kind_combo.setEnabled(editable)
        kind_combo.currentIndexChanged.connect(lambda *_: self._apply_user_edits())
        self.params_table.setCellWidget(row, 1, kind_combo)
        for column, attribute in ((2, "default"), (3, "minimum"), (4, "maximum"), (5, "step")):
            value = getattr(parameter, attribute, None) if parameter is not None else None
            self.params_table.setItem(row, column, QtWidgets.QTableWidgetItem(_text(value)))

    def _add_parameter_row(self):
        if not _is_user_op(self._loaded_id or ""):
            return
        self._append_parameter_row(None, editable=True)
        self._apply_user_edits()

    def _remove_parameter_row(self):
        row = self.params_table.currentRow()
        if row < 0:
            return
        self.params_table.removeRow(row)
        self._apply_user_edits()

    def _table_parameters(self):
        parameters = []
        for row in range(self.params_table.rowCount()):
            name_item = self.params_table.item(row, 0)
            name = name_item.text().strip() if name_item is not None else ""
            if not name:
                continue
            kind_widget = self.params_table.cellWidget(row, 1)
            kind = kind_widget.currentText() if kind_widget is not None else "float"
            payload = {
                "name": name,
                "label": name.replace("_", " ").title(),
                "kind": kind,
            }
            for column, key in ((2, "default"), (3, "minimum"), (4, "maximum"), (5, "step")):
                value = self._parse_number(self.params_table.item(row, column), kind)
                if value is not None:
                    payload[key] = value
            parameters.append(payload)
        return parameters

    @staticmethod
    def _parse_number(item, kind):
        if item is None:
            return None
        text = item.text().strip()
        if not text:
            return None
        try:
            return int(text) if kind == "int" else float(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return None

    # ------------------------------------------------------------------
    # Edit application (autosave)
    # ------------------------------------------------------------------

    def _apply_user_edits(self):
        if self._suppress or self._reloading:
            return
        operation_id = self._loaded_id
        if operation_id is None or not _is_user_op(operation_id):
            return
        library.update_user_operation(
            operation_id,
            label=self.label_edit.text().strip() or operation_id,
            description=self.description_edit.text().strip(),
            icon=self.icon_edit.text().strip() or "extension",
            requires_axis=self.requires_axis_check.isChecked(),
            parameters=self._table_parameters(),
        )

    def _on_requires_axis_toggled(self, _checked):
        self._apply_user_edits()

    def _on_group_changed(self, new_group):
        if self._suppress or self._reloading:
            return
        operation_id = self._loaded_id
        new_group = str(new_group).strip()
        if operation_id is None or not new_group:
            return
        if _is_user_op(operation_id):
            library.update_user_operation(operation_id, group=new_group)
        self._persist_layout(override={operation_id: new_group})

    def _on_common_toggled(self, checked):
        if self._suppress or self._reloading:
            return
        operation_id = self._loaded_id
        if operation_id is None:
            return
        common = list(library.effective_common_ids())
        if checked and operation_id not in common:
            common.append(operation_id)
        elif not checked and operation_id in common:
            common.remove(operation_id)
        else:
            return
        library.apply_library_layout(common_ids=common)

    # ------------------------------------------------------------------
    # Tree actions
    # ------------------------------------------------------------------

    def _sync_button_states(self):
        operation_id = self.selected_operation_id()
        if operation_id is None:
            for button in (self.remove_button, self.unhide_button, self.open_file_button):
                button.setEnabled(False)
            return
        is_user = _is_user_op(operation_id)
        hidden = operation_id in library.hidden_operations()
        self.remove_button.setEnabled(True)
        set_button_icon(
            self.remove_button,
            "delete",
            tooltip="Remove this user operation" if is_user else "Hide",
        )
        self.unhide_button.setEnabled(hidden)
        self.open_file_button.setEnabled(is_user)

    def _remove_or_hide(self):
        operation_id = self.selected_operation_id()
        if operation_id is None:
            return
        if _is_user_op(operation_id):
            entry_label = self._current_label(operation_id)
            confirm = QtWidgets.QMessageBox.question(
                self,
                "Remove operation",
                f"Remove the user operation “{entry_label}” and delete its files?",
            )
            if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            library.remove_user_operation(operation_id)
            show_status_message(self._window, f"Removed operation “{entry_label}”.", timeout=2500)
        else:
            library.set_operation_hidden(operation_id, True)
            show_status_message(
                self._window,
                f"Hid “{self._current_label(operation_id)}” — select it and press Restore.",
                timeout=3000,
            )

    def _unhide(self):
        operation_id = self.selected_operation_id()
        if operation_id is None:
            return
        if library.reset_operation(operation_id):
            show_status_message(
                self._window, f"Restored “{self._current_label(operation_id)}”.", timeout=2500
            )

    def _current_label(self, operation_id):
        try:
            return get_operation_entry(operation_id).label
        except Exception:
            return operation_id

    def _open_code_file(self):
        operation_id = self.selected_operation_id()
        if operation_id is None or not _is_user_op(operation_id):
            return
        path = _user_op_source_path(operation_id)
        if not path or not os.path.exists(path):
            show_status_message(self._window, "Could not locate the code file.", timeout=3000)
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _open_folder(self):
        directory = library.user_operations_directory()
        os.makedirs(directory, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(directory))

    def _reset_all(self):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Reset operations",
            "Reset the layout to defaults and unhide every hidden operation?",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        for operation_id in tuple(library.hidden_operations()):
            library.set_operation_hidden(operation_id, False)
        library.reset_layout()
        show_status_message(self._window, "Reset operations to defaults.", timeout=2500)

    # ------------------------------------------------------------------
    # Import flow
    # ------------------------------------------------------------------

    def _import_operation(self):
        path, _ = get_open_file_name(
            self, "Import a custom operation", "", "Python (*.py);;All files (*)"
        )
        if not path:
            return
        try:
            infos = library.introspect_python_source(path)
        except Exception as exc:
            show_status_message(self._window, f"Could not read {path}: {exc}", timeout=4000)
            return
        if not infos:
            show_status_message(
                self._window, "That file has no top-level functions to import.", timeout=4000
            )
            return
        existing_groups = [
            group for group, _entries in library.grouped_operations(include_hidden=True)
        ]
        dialog = _OperationImportDialog(self, path, infos, existing_groups)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        result = dialog.result_payload()
        try:
            operation_id = library.import_custom_operation(
                path,
                result["callable"],
                link=result["link"],
                label=result["label"],
                description=result["description"],
                group=result["group"],
                icon=result["icon"],
            )
        except Exception as exc:
            show_status_message(self._window, f"Import failed: {exc}", timeout=4000)
            return
        # Follow with the edited parameters / requires_axis when the user changed
        # them beyond what auto-fill produced.
        if result["params_edited"] or result["requires_axis_edited"]:
            library.update_user_operation(
                operation_id,
                requires_axis=result["requires_axis"],
                parameters=result["parameters"],
            )
        self.select_operation(operation_id)
        show_status_message(self._window, f"Imported “{result['label']}”.", timeout=2500)


class _OperationImportDialog(QtWidgets.QDialog):
    """Compact "connect up a custom function" panel (auto-filled, editable)."""

    def __init__(self, parent, path, infos, existing_groups):
        super().__init__(parent)
        self.setWindowTitle("Import a custom operation")
        self.resize(480, 460)
        self._path = path
        self._infos = {info.name: info for info in infos}

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.function_combo = QtWidgets.QComboBox(self)
        self.function_combo.addItems([info.name for info in infos])
        form.addRow("Function", self.function_combo)
        self.label_edit = QtWidgets.QLineEdit(self)
        form.addRow("Label", self.label_edit)
        self.description_edit = QtWidgets.QLineEdit(self)
        form.addRow("Description", self.description_edit)
        self.group_combo = QtWidgets.QComboBox(self)
        self.group_combo.setEditable(True)
        self.group_combo.addItems(existing_groups or ["User"])
        self.group_combo.setCurrentText("User")
        form.addRow("Group", self.group_combo)
        icon_row = QtWidgets.QHBoxLayout()
        self.icon_edit = QtWidgets.QLineEdit(self)
        self.icon_edit.setText("extension")
        self.icon_preview = QtWidgets.QLabel(self)
        icon_row.addWidget(self.icon_edit, 1)
        icon_row.addWidget(self.icon_preview)
        icon_holder = QtWidgets.QWidget(self)
        icon_holder.setLayout(icon_row)
        icon_row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Icon", icon_holder)
        self.requires_axis_check = QtWidgets.QCheckBox("Requires an axis", self)
        form.addRow("", self.requires_axis_check)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Detected parameters"))
        self.params_table = QtWidgets.QTableWidget(0, 4, self)
        self.params_table.setHorizontalHeaderLabels(["Name", "Kind", "Default", "Description"])
        self.params_table.horizontalHeader().setStretchLastSection(True)
        self.params_table.verticalHeader().setVisible(False)
        layout.addWidget(self.params_table, 1)

        mode_box = QtWidgets.QGroupBox("How should the code be stored?", self)
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        self.copy_radio = QtWidgets.QRadioButton("Import a copy (recommended)", self)
        self.copy_radio.setChecked(True)
        copy_hint = QtWidgets.QLabel(
            "Copies the file into ArrayScope — the operation keeps working if you move or edit the original."
        )
        copy_hint.setObjectName("OperationsMetaLabel")
        copy_hint.setWordWrap(True)
        self.link_radio = QtWidgets.QRadioButton("Link to the file", self)
        link_hint = QtWidgets.QLabel(
            "Keeps a live link to the original — edits you make to it are picked up automatically."
        )
        link_hint.setObjectName("OperationsMetaLabel")
        link_hint.setWordWrap(True)
        for widget in (self.copy_radio, copy_hint, self.link_radio, link_hint):
            mode_layout.addWidget(widget)
        layout.addWidget(mode_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.function_combo.currentTextChanged.connect(self._load_function)
        self.icon_edit.textChanged.connect(self._update_icon_preview)
        self._auto_filled_params = ()
        self._auto_requires_axis = False
        self._update_icon_preview()
        self._load_function(self.function_combo.currentText())

    def _update_icon_preview(self, *_args):
        set_label_icon(
            self.icon_preview, self.icon_edit.text().strip() or "extension", icon_size=18
        )

    def _load_function(self, name):
        info = self._infos.get(name)
        if info is None:
            return
        self.label_edit.setText(name.replace("_", " ").title())
        self.description_edit.setText(info.doc)
        self.requires_axis_check.setChecked(info.has_axis)
        self._auto_requires_axis = info.has_axis
        self._auto_filled_params = info.params
        self.params_table.setRowCount(0)
        for parameter in info.params:
            row = self.params_table.rowCount()
            self.params_table.insertRow(row)
            self.params_table.setItem(row, 0, QtWidgets.QTableWidgetItem(parameter.name))
            kind_combo = QtWidgets.QComboBox()
            kind_combo.addItems(_KIND_CHOICES)
            kind_combo.setCurrentText(
                parameter.kind if parameter.kind in _KIND_CHOICES else "float"
            )
            self.params_table.setCellWidget(row, 1, kind_combo)
            default = "" if parameter.default is None else str(parameter.default)
            self.params_table.setItem(row, 2, QtWidgets.QTableWidgetItem(default))
            self.params_table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))

    def _table_parameters(self):
        parameters = []
        for row in range(self.params_table.rowCount()):
            name_item = self.params_table.item(row, 0)
            name = name_item.text().strip() if name_item is not None else ""
            if not name:
                continue
            kind_widget = self.params_table.cellWidget(row, 1)
            kind = kind_widget.currentText() if kind_widget is not None else "float"
            payload = {"name": name, "label": name.replace("_", " ").title(), "kind": kind}
            default_item = self.params_table.item(row, 2)
            default_text = default_item.text().strip() if default_item is not None else ""
            if default_text:
                try:
                    payload["default"] = int(default_text) if kind == "int" else float(default_text)
                except ValueError:
                    payload["default"] = default_text
            description_item = self.params_table.item(row, 3)
            if description_item is not None and description_item.text().strip():
                payload["description"] = description_item.text().strip()
            parameters.append(payload)
        return parameters

    def _params_edited(self, parameters):
        auto = []
        for parameter in self._auto_filled_params:
            payload = {
                "name": parameter.name,
                "label": parameter.name.replace("_", " ").title(),
                "kind": parameter.kind,
            }
            if parameter.default is not None:
                payload["default"] = parameter.default
            auto.append(payload)
        return parameters != auto

    def result_payload(self):
        parameters = self._table_parameters()
        return {
            "callable": self.function_combo.currentText(),
            "label": self.label_edit.text().strip() or self.function_combo.currentText(),
            "description": self.description_edit.text().strip(),
            "group": self.group_combo.currentText().strip() or "User",
            "icon": self.icon_edit.text().strip() or "extension",
            "link": self.link_radio.isChecked(),
            "requires_axis": self.requires_axis_check.isChecked(),
            "requires_axis_edited": self.requires_axis_check.isChecked()
            != self._auto_requires_axis,
            "parameters": parameters,
            "params_edited": self._params_edited(parameters),
        }
