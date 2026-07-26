"""Operation manager: browse, arrange, duplicate and edit operations.

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
- The right column is the product's one operation editor. ``New`` creates an
  unfinished entry in place; source/callable selection and copy/link storage
  live in this same editor rather than a second modal editor.

The dialog registers a library listener so an external mutation (a recipe load,
another window's edit) rebuilds the tree in place, preserving selection by id.
"""

from __future__ import annotations

import os

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations import library
from arrayscope.operations.operation_definitions import export_operation_definition
from arrayscope.operations.registry import get_operation_entry
from arrayscope.ui.file_dialogs import get_open_file_name
from arrayscope.ui.icons import material_icon, set_button_icon, set_label_icon
from arrayscope.ui.toasts import show_status_message

_ID_ROLE = QtCore.Qt.ItemDataRole.UserRole
_GROUP_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_VIRTUAL_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2

_PROBLEMS_GROUP = "Problems"
_KIND_CHOICES = ("int", "float")
_PARAM_LABEL_ROLE = QtCore.Qt.ItemDataRole.UserRole
_RUNTIME_CHOICES = (
    ("Python", "python"),
    ("Command", "command"),
    ("Julia", "julia"),
    ("MATLAB", "matlab"),
)


def _is_user_op(operation_id: str) -> bool:
    return str(operation_id).startswith("user:")


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
        self.resize(780, 720)
        self.setMinimumSize(720, 620)
        self.setSizeGripEnabled(True)
        self._loaded_id = None
        self._suppress = False
        self._reloading = False
        self._source_infos = {}
        self._editing_environment_id = ""

        root = QtWidgets.QHBoxLayout(self)

        # -- left: library tree -------------------------------------------
        left = QtWidgets.QVBoxLayout()
        self.tree = _OperationTree(self)
        self.tree.setMinimumWidth(210)
        left.addWidget(self.tree, 1)
        hint = QtWidgets.QLabel("Drag to rearrange operations and groups")
        hint.setObjectName("OperationsMetaLabel")
        left.addWidget(hint)

        self.new_button = QtWidgets.QToolButton(self)
        set_button_icon(self.new_button, "add", tooltip="New operation")
        self.new_button.setText("New")
        self.new_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Compatibility for callers of the original manager API; this is now
        # an in-manager creation action, never a modal import editor.
        self.add_button = self.new_button
        self.duplicate_button = QtWidgets.QToolButton(self)
        set_button_icon(self.duplicate_button, "data_object", tooltip="Duplicate selected to edit")
        self.duplicate_button.setText("Duplicate")
        self.duplicate_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        primary_buttons = QtWidgets.QHBoxLayout()
        primary_buttons.addWidget(self.new_button)
        primary_buttons.addWidget(self.duplicate_button)
        primary_buttons.addStretch(1)
        left.addLayout(primary_buttons)

        tree_buttons = QtWidgets.QHBoxLayout()
        self.remove_button = QtWidgets.QToolButton(self)
        set_button_icon(self.remove_button, "delete", tooltip="Hide")
        self.unhide_button = QtWidgets.QToolButton(self)
        set_button_icon(self.unhide_button, "visibility", tooltip="Restore / unhide")
        self.open_file_button = QtWidgets.QToolButton(self)
        set_button_icon(self.open_file_button, "edit", tooltip="Open the code file")
        self.open_folder_button = QtWidgets.QToolButton(self)
        set_button_icon(
            self.open_folder_button, "folder_open", tooltip="Open the operations folder"
        )
        self.reset_all_button = QtWidgets.QToolButton(self)
        set_button_icon(self.reset_all_button, "refresh", tooltip="Reset layout and unhide all")
        for button in (
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
        right_widget = QtWidgets.QWidget(self)
        right = QtWidgets.QVBoxLayout(right_widget)
        self.editor_scroll = QtWidgets.QScrollArea(self)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.editor_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.editor_scroll.setWidget(right_widget)
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
        self.changes_shape_label = QtWidgets.QLabel("", self)
        self.changes_shape_label.setObjectName("OperationsMetaLabel")
        form.addRow("Output", self.changes_shape_label)
        self.runtime_combo = QtWidgets.QComboBox(self)
        for label, value in _RUNTIME_CHOICES:
            self.runtime_combo.addItem(label, value)
        form.addRow("Runtime", self.runtime_combo)
        self.common_check = QtWidgets.QCheckBox("Show in the Common (pinned) section", self)
        form.addRow("", self.common_check)
        right.addLayout(form)

        self.source_box = QtWidgets.QGroupBox("Implementation", self)
        source_layout = QtWidgets.QVBoxLayout(self.source_box)
        source_form = QtWidgets.QFormLayout()
        source_row = QtWidgets.QHBoxLayout()
        self.source_path_edit = QtWidgets.QLineEdit(self)
        self.source_path_edit.setPlaceholderText("Choose a Python source file")
        self.source_browse_button = QtWidgets.QToolButton(self)
        set_button_icon(
            self.source_browse_button, "folder_open", tooltip="Choose Python source file"
        )
        source_row.addWidget(self.source_path_edit, 1)
        source_row.addWidget(self.source_browse_button)
        source_holder = QtWidgets.QWidget(self)
        source_holder.setLayout(source_row)
        source_row.setContentsMargins(0, 0, 0, 0)
        source_form.addRow("Source file", source_holder)
        self.callable_combo = QtWidgets.QComboBox(self)
        source_form.addRow("Callable", self.callable_combo)
        source_layout.addLayout(source_form)
        self.callable_hint = QtWidgets.QLabel("", self)
        self.callable_hint.setObjectName("OperationsMetaLabel")
        source_layout.addWidget(self.callable_hint)

        storage_row = QtWidgets.QHBoxLayout()
        self.copy_radio = QtWidgets.QRadioButton("Copy into ArrayScope", self)
        self.link_radio = QtWidgets.QRadioButton("Link to original", self)
        storage_row.addWidget(self.copy_radio)
        storage_row.addWidget(self.link_radio)
        storage_row.addStretch(1)
        source_layout.addLayout(storage_row)
        self.storage_hint = QtWidgets.QLabel("", self)
        self.storage_hint.setObjectName("OperationsMetaLabel")
        self.storage_hint.setWordWrap(True)
        source_layout.addWidget(self.storage_hint)
        right.addWidget(self.source_box)

        self.command_box = QtWidgets.QGroupBox("Command template", self)
        command_layout = QtWidgets.QFormLayout(self.command_box)
        self.command_template_edit = QtWidgets.QLineEdit(self)
        self.command_template_edit.setPlaceholderText("tool --option {parameter} {in} {out}")
        command_layout.addRow("Arguments", self.command_template_edit)
        command_hint = QtWidgets.QLabel(
            "Use {in}, {out}, and each declared parameter. Values remain literal argv tokens.",
            self,
        )
        command_hint.setObjectName("OperationsMetaLabel")
        command_hint.setWordWrap(True)
        command_layout.addRow("", command_hint)
        right.addWidget(self.command_box)

        self.advanced_button = QtWidgets.QToolButton(self)
        self.advanced_button.setText("Advanced")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        right.addWidget(self.advanced_button)

        self.advanced_panel = QtWidgets.QWidget(self)
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        runtime_advanced = QtWidgets.QGroupBox("Runtime settings", self)
        runtime_form = QtWidgets.QFormLayout(runtime_advanced)
        self.environment_combo = QtWidgets.QComboBox(self)
        runtime_form.addRow("Environment", self.environment_combo)
        self.handoff_combo = QtWidgets.QComboBox(self)
        self.handoff_combo.addItems(["npy", "cfl"])
        runtime_form.addRow("Array handoff", self.handoff_combo)
        self.timeout_spin = QtWidgets.QDoubleSpinBox(self)
        self.timeout_spin.setRange(0.0, 86_400.0)
        self.timeout_spin.setDecimals(1)
        self.timeout_spin.setSpecialValueText("No timeout")
        self.timeout_spin.setSuffix(" s")
        runtime_form.addRow("Timeout", self.timeout_spin)
        self.shell_check = QtWidgets.QCheckBox(
            "Run through the system shell (unsafe; explicit opt-in)", self
        )
        runtime_form.addRow("", self.shell_check)
        self.review_button = QtWidgets.QPushButton("Mark imported command reviewed", self)
        self.review_button.setVisible(False)
        runtime_form.addRow("", self.review_button)
        advanced_layout.addWidget(runtime_advanced)

        self.environments_box = QtWidgets.QGroupBox("Named environments", self)
        self.environments_box.setMinimumHeight(350)
        environment_layout = QtWidgets.QVBoxLayout(self.environments_box)
        environment_form = QtWidgets.QFormLayout()
        self.environment_editor_combo = QtWidgets.QComboBox(self)
        environment_form.addRow("Edit", self.environment_editor_combo)
        self.environment_id_edit = QtWidgets.QLineEdit(self)
        self.environment_id_edit.setPlaceholderText("e.g. recon")
        environment_form.addRow("Id", self.environment_id_edit)
        self.environment_name_edit = QtWidgets.QLineEdit(self)
        environment_form.addRow("Name", self.environment_name_edit)
        self.environment_kind_combo = QtWidgets.QComboBox(self)
        self.environment_kind_combo.addItem("Variables / working directory only", "")
        self.environment_kind_combo.addItem("Interpreter path", "interpreter")
        self.environment_kind_combo.addItem("Conda environment", "conda_env")
        self.environment_kind_combo.addItem("Virtualenv path", "venv_path")
        environment_form.addRow("Locator", self.environment_kind_combo)
        self.environment_locator_edit = QtWidgets.QLineEdit(self)
        environment_form.addRow("Value", self.environment_locator_edit)
        self.environment_cwd_edit = QtWidgets.QLineEdit(self)
        environment_form.addRow("Working directory", self.environment_cwd_edit)
        self.environment_variables_edit = QtWidgets.QPlainTextEdit(self)
        self.environment_variables_edit.setPlaceholderText(
            "One NAME=value per line, e.g. BART_TOOLBOX_PATH=/opt/bart"
        )
        self.environment_variables_edit.setMaximumHeight(72)
        self.environment_variables_edit.setMinimumHeight(60)
        environment_form.addRow("Environment variables", self.environment_variables_edit)
        environment_layout.addLayout(environment_form)
        environment_buttons = QtWidgets.QHBoxLayout()
        self.environment_new_button = QtWidgets.QPushButton("New", self)
        self.environment_save_button = QtWidgets.QPushButton("Save environment", self)
        self.environment_remove_button = QtWidgets.QPushButton("Remove", self)
        environment_buttons.addWidget(self.environment_new_button)
        environment_buttons.addWidget(self.environment_save_button)
        environment_buttons.addWidget(self.environment_remove_button)
        environment_buttons.addStretch(1)
        environment_layout.addLayout(environment_buttons)
        advanced_layout.addWidget(self.environments_box)
        self.advanced_panel.setMinimumHeight(520)
        self.advanced_panel.setVisible(False)
        right.addWidget(self.advanced_panel)

        right.addWidget(QtWidgets.QLabel("Parameters"))
        self.params_table = QtWidgets.QTableWidget(0, 7, self)
        self.params_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Default", "Min", "Max", "Step", "Description"]
        )
        header = self.params_table.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(6):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for column, width in enumerate((70, 54, 58, 38, 38, 42)):
            header.resizeSection(column, width)
        self.params_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.params_table.setMinimumHeight(215)
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
        root.addWidget(self.editor_scroll, 1)

        # -- wiring -------------------------------------------------------
        self.tree.currentItemChanged.connect(lambda *_: self._load_editor())
        self.tree.orderEdited.connect(self._persist_layout)
        self.new_button.clicked.connect(self._new_operation)
        self.duplicate_button.clicked.connect(self._duplicate_operation)
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
        self.runtime_combo.currentIndexChanged.connect(self._on_runtime_changed)
        self.group_combo.currentTextChanged.connect(self._on_group_changed)
        self.common_check.toggled.connect(self._on_common_toggled)
        self.params_table.cellChanged.connect(lambda *_: self._apply_user_edits())
        self.add_param_button.clicked.connect(self._add_parameter_row)
        self.remove_param_button.clicked.connect(self._remove_parameter_row)
        self.source_browse_button.clicked.connect(self._choose_source_file)
        self.source_path_edit.editingFinished.connect(self._retarget_source)
        self.callable_combo.currentTextChanged.connect(self._on_callable_changed)
        self.copy_radio.toggled.connect(self._update_storage_hint)
        self.link_radio.toggled.connect(self._on_storage_mode_changed)
        self.command_template_edit.editingFinished.connect(self._apply_runtime_edits)
        self.environment_combo.currentIndexChanged.connect(self._apply_runtime_edits)
        self.handoff_combo.currentTextChanged.connect(self._apply_runtime_edits)
        self.timeout_spin.editingFinished.connect(self._apply_runtime_edits)
        self.shell_check.toggled.connect(self._apply_runtime_edits)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        self.review_button.clicked.connect(self._review_imported_command)
        self.environment_editor_combo.currentIndexChanged.connect(self._load_environment_editor)
        self.environment_new_button.clicked.connect(self._new_environment)
        self.environment_save_button.clicked.connect(self._save_environment)
        self.environment_remove_button.clicked.connect(self._remove_environment)

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
        if entry.unavailable_reason:
            markers.append("(unavailable)")
        suffix = ("  " + " ".join(markers)) if markers else ""
        item = QtWidgets.QTreeWidgetItem([f"{entry.label}{suffix}"])
        item.setIcon(0, material_icon(entry.icon or "data_array"))
        item.setData(0, _ID_ROLE, entry.id)
        item.setToolTip(0, entry.unavailable_reason or entry.description or entry.id)
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
                self.group_combo.clear()
                self.description_edit.clear()
                self.icon_edit.clear()
                self.icon_preview.clear()
                self.requires_axis_check.setChecked(False)
                self.changes_shape_label.clear()
                self.runtime_combo.setCurrentIndex(0)
                self.common_check.setChecked(False)
                self.source_path_edit.clear()
                self.callable_combo.clear()
                self.callable_hint.clear()
                self.copy_radio.setChecked(False)
                self.link_radio.setChecked(False)
                self.copy_radio.setVisible(False)
                self.link_radio.setVisible(False)
                self.storage_hint.clear()
                self.command_template_edit.clear()
                self.params_table.setRowCount(0)
            finally:
                self._suppress = False
            item = self.tree.currentItem()
            problem = item.toolTip(0) if item is not None and item.data(0, _VIRTUAL_ROLE) else ""
            self.status_label.setText(
                f"Operation problem — {problem}" if problem else "Select an operation to edit it."
            )
            self._sync_button_states()
            return
        try:
            entry = get_operation_entry(operation_id)
        except Exception:
            self._sync_button_states()
            return
        is_user = _is_user_op(operation_id)
        hidden = operation_id in library.hidden_operations()
        definition = export_operation_definition(entry)

        self._suppress = True
        try:
            self.label_edit.setText(entry.label)
            self.id_label.setText(operation_id)
            self._reload_group_combo(library.effective_group(entry))
            self.description_edit.setText(entry.description)
            self.icon_edit.setText(entry.icon or "")
            self._update_icon_preview()
            self.requires_axis_check.setChecked(entry.requires_axis)
            self.changes_shape_label.setText(
                "Changes shape" if entry.changes_shape else "Preserves shape"
            )
            self.common_check.setChecked(operation_id in library.effective_common_ids())
            runtime = str(definition.get("runtime") or "python")
            runtime_index = self.runtime_combo.findData(runtime)
            self.runtime_combo.setCurrentIndex(max(0, runtime_index))
            self._load_source_editor(definition, is_user=is_user)
            self.command_template_edit.setText(str(definition.get("command_template") or ""))
            self.handoff_combo.setCurrentText(str(definition.get("handoff") or "npy"))
            timeout = definition.get("timeout_s", 600.0)
            self.timeout_spin.setValue(0.0 if timeout is None else float(timeout))
            self.shell_check.setChecked(bool(definition.get("shell", False)))
            self._reload_environment_combos(str(definition.get("environment") or ""))
            self._fill_parameters(entry.parameters, editable=is_user)
        finally:
            self._suppress = False

        self._set_icon_row_visible(True)
        self._sync_runtime_visibility()
        self._apply_editor_editability(is_user)
        wrapper = library.user_operation_wrapper(operation_id) or {} if is_user else {}
        review = wrapper.get("review") if isinstance(wrapper, dict) else None
        self.review_button.setVisible(
            bool(is_user and isinstance(review, dict) and review.get("required"))
        )
        if entry.unavailable_reason:
            self.status_label.setText(f"Unavailable — {entry.unavailable_reason}")
        elif is_user:
            template = wrapper.get("template")
            if isinstance(template, dict) and template.get("message"):
                self.status_label.setText(str(template["message"]))
            else:
                self.status_label.setText(
                    "User operation — every shown value is editable and saves automatically."
                )
        elif hidden:
            self.status_label.setText(
                "Hidden system operation — Restore returns it; Duplicate makes an editable copy."
            )
        else:
            self.status_label.setText(
                "System definition — read-only here. Duplicate makes an editable user copy."
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
            self.runtime_combo,
            self.source_box,
            self.source_path_edit,
            self.source_browse_button,
            self.callable_combo,
            self.copy_radio,
            self.link_radio,
            self.common_check,
            self.params_table,
            self.add_param_button,
            self.remove_param_button,
            self.command_box,
            self.command_template_edit,
            self.environment_combo,
            self.handoff_combo,
            self.timeout_spin,
            self.shell_check,
            self.advanced_button,
        ):
            widget.setEnabled(enabled)

    def _apply_editor_editability(self, is_user: bool):
        """Reflect the selected op's edit permissions in the widget states.

        A system op's definition is read-only: its Label, Description, Icon and
        parameters must LOOK disabled (greyed + read-only), not merely swallow
        edits. The two controls that genuinely act on a system op -- Group and
        the Common toggle -- stay enabled. A user op is fully editable.
        """

        self.source_box.setEnabled(True)
        for widget in (self.label_edit, self.description_edit, self.icon_edit):
            widget.setReadOnly(not is_user)
            widget.setEnabled(is_user)
        self.requires_axis_check.setEnabled(is_user)
        self.runtime_combo.setEnabled(is_user)
        self.source_path_edit.setReadOnly(not is_user)
        self.source_path_edit.setEnabled(is_user)
        self.source_browse_button.setEnabled(is_user)
        self.callable_combo.setEnabled(is_user)
        self.copy_radio.setEnabled(is_user)
        self.link_radio.setEnabled(is_user)
        self.params_table.setEnabled(is_user)
        self.add_param_button.setEnabled(is_user)
        self.remove_param_button.setEnabled(is_user)
        self.command_template_edit.setReadOnly(not is_user)
        self.command_template_edit.setEnabled(is_user)
        self.environment_combo.setEnabled(is_user)
        self.handoff_combo.setEnabled(is_user)
        self.timeout_spin.setEnabled(is_user)
        self.shell_check.setEnabled(is_user)
        # Environment records are library-wide and remain editable regardless
        # of whether the selected operation is a system definition.
        self.advanced_button.setEnabled(True)
        self.environments_box.setEnabled(True)
        # These act on system ops too, so they stay live regardless.
        self.group_combo.setEnabled(True)
        self.common_check.setEnabled(True)

    def _sync_runtime_visibility(self):
        runtime = str(self.runtime_combo.currentData() or "python")
        self.source_box.setVisible(runtime == "python")
        self.command_box.setVisible(runtime != "python")
        self.shell_check.setEnabled(
            bool(_is_user_op(self._loaded_id or "") and runtime == "command")
        )
        if runtime != "command":
            self.shell_check.setChecked(False)

    def _on_runtime_changed(self, _index):
        self._sync_runtime_visibility()
        self._apply_runtime_edits()

    def _apply_runtime_edits(self, *_args):
        if self._suppress or self._reloading:
            return
        operation_id = self._loaded_id
        if operation_id is None or not _is_user_op(operation_id):
            return
        runtime = str(self.runtime_combo.currentData() or "python")
        fields = {
            "runtime": runtime,
            "environment": str(self.environment_combo.currentData() or ""),
            "handoff": self.handoff_combo.currentText(),
            "timeout_s": None if self.timeout_spin.value() <= 0 else self.timeout_spin.value(),
            "shell": bool(self.shell_check.isChecked() and runtime == "command"),
        }
        if runtime != "python":
            fields["command_template"] = self.command_template_edit.text().strip()
        wrapper = library.user_operation_wrapper(operation_id) or {}
        template = wrapper.get("template")
        if isinstance(template, dict) and template.get("kind") == "empty":
            fields["template"] = None
        library.update_user_operation(operation_id, **fields)

    def _toggle_advanced(self, expanded):
        self.advanced_panel.setVisible(bool(expanded))
        self.advanced_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow
        )
        if expanded:
            self.adjustSize()

    def _reload_environment_combos(self, selected: str = ""):
        try:
            records = library.execution_environments()
        except Exception:
            records = ()
        self.environment_combo.blockSignals(True)
        self.environment_editor_combo.blockSignals(True)
        try:
            self.environment_combo.clear()
            self.environment_combo.addItem("Current process / none", "")
            self.environment_editor_combo.clear()
            self.environment_editor_combo.addItem("New environment…", "")
            for record in records:
                self.environment_combo.addItem(record.name, record.id)
                self.environment_editor_combo.addItem(record.name, record.id)
            index = self.environment_combo.findData(selected)
            self.environment_combo.setCurrentIndex(max(0, index))
            editor_index = self.environment_editor_combo.findData(self._editing_environment_id)
            self.environment_editor_combo.setCurrentIndex(max(0, editor_index))
        finally:
            self.environment_combo.blockSignals(False)
            self.environment_editor_combo.blockSignals(False)
        self._load_environment_editor(self.environment_editor_combo.currentIndex())

    def _load_environment_editor(self, _index):
        environment_id = str(self.environment_editor_combo.currentData() or "")
        self._editing_environment_id = environment_id
        record = None
        if environment_id:
            try:
                record = next(
                    item for item in library.execution_environments() if item.id == environment_id
                )
            except (StopIteration, OSError, ValueError):
                record = None
        if record is None:
            self.environment_id_edit.clear()
            self.environment_name_edit.clear()
            self.environment_kind_combo.setCurrentIndex(0)
            self.environment_locator_edit.clear()
            self.environment_cwd_edit.clear()
            self.environment_variables_edit.clear()
            self.environment_remove_button.setEnabled(False)
            return
        self.environment_id_edit.setText(record.id)
        self.environment_name_edit.setText(record.name)
        locator_kind = ""
        locator_value = ""
        for field in ("interpreter", "conda_env", "venv_path"):
            value = getattr(record, field)
            if value:
                locator_kind, locator_value = field, value
                break
        index = self.environment_kind_combo.findData(locator_kind)
        self.environment_kind_combo.setCurrentIndex(max(0, index))
        self.environment_locator_edit.setText(locator_value)
        self.environment_cwd_edit.setText(record.working_directory)
        self.environment_variables_edit.setPlainText(
            "\n".join(f"{key}={value}" for key, value in record.variables)
        )
        self.environment_remove_button.setEnabled(True)

    def _new_environment(self):
        self.environment_editor_combo.setCurrentIndex(0)
        self.environment_id_edit.setFocus()

    def _environment_variables(self):
        variables = {}
        for line in self.environment_variables_edit.toPlainText().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "=" not in stripped:
                raise ValueError(f"environment variable must be NAME=value: {stripped}")
            key, value = stripped.split("=", 1)
            if not key.strip():
                raise ValueError("environment variable name cannot be empty")
            variables[key.strip()] = value
        return variables

    def _save_environment(self):
        environment_id = self.environment_id_edit.text().strip()
        if not environment_id:
            show_status_message(self._window, "Environment id is required.", timeout=3000)
            return
        locator_kind = str(self.environment_kind_combo.currentData() or "")
        fields = {
            "id": environment_id,
            "name": self.environment_name_edit.text().strip() or environment_id,
            "interpreter": "",
            "conda_env": "",
            "venv_path": "",
            "working_directory": self.environment_cwd_edit.text().strip(),
        }
        if locator_kind:
            fields[locator_kind] = self.environment_locator_edit.text().strip()
        try:
            fields["variables"] = self._environment_variables()
            if self._editing_environment_id and self._editing_environment_id != environment_id:
                library.remove_execution_environment(self._editing_environment_id)
            library.update_execution_environment(**fields)
        except Exception as exc:
            show_status_message(self._window, f"Could not save environment: {exc}", timeout=4000)
            return
        self._editing_environment_id = environment_id
        self._reload_environment_combos(str(self.environment_combo.currentData() or ""))
        show_status_message(self._window, f"Saved environment “{fields['name']}”.", timeout=2500)

    def _remove_environment(self):
        environment_id = self._editing_environment_id
        if not environment_id:
            return
        if library.remove_execution_environment(environment_id):
            self._editing_environment_id = ""
            self._reload_environment_combos()
            show_status_message(self._window, "Removed execution environment.", timeout=2500)

    def _review_imported_command(self):
        operation_id = self._loaded_id
        if operation_id and library.review_user_operation(operation_id):
            self.select_operation(operation_id)
            show_status_message(
                self._window,
                "Command definition marked reviewed; runtime availability was rechecked.",
                timeout=3500,
            )

    def _load_source_editor(self, definition, *, is_user: bool):
        source = definition.get("source") or {}
        self.callable_combo.clear()
        self._source_infos = {}
        if is_user:
            operation_id = str(definition.get("id") or "")
            path = library.user_operation_source_path(operation_id) or ""
            self.source_path_edit.setText(path)
            self.source_path_edit.setToolTip(path)
            self.source_path_edit.setCursorPosition(len(path))
            if path:
                try:
                    infos = library.introspect_python_source(path)
                except Exception:
                    infos = []
                self._source_infos = {info.name: info for info in infos}
                self.callable_combo.addItems([info.name for info in infos])
            callable_name = str(source.get("callable") or "")
            if callable_name and self.callable_combo.findText(callable_name) < 0:
                self.callable_combo.addItem(callable_name)
            self.callable_combo.setCurrentText(callable_name)
            count = len(self._source_infos)
            self.callable_hint.setText(
                f"{count} top-level callable{'s' if count != 1 else ''} found (AST only)."
            )
            is_link = str(source.get("mode") or "import") == "link"
            self.copy_radio.setVisible(True)
            self.link_radio.setVisible(True)
            self.link_radio.setChecked(is_link)
            self.copy_radio.setChecked(not is_link)
        else:
            source_text = self._system_source_text(source)
            self.source_path_edit.setText(source_text)
            self.source_path_edit.setToolTip(source_text)
            self.source_path_edit.setCursorPosition(0)
            callable_name = str(source.get("class") or source.get("name") or source.get("id") or "")
            if callable_name:
                self.callable_combo.addItem(callable_name)
            self.callable_hint.setText("Registered system implementation.")
            self.copy_radio.setVisible(False)
            self.link_radio.setVisible(False)
            self.copy_radio.setChecked(False)
            self.link_radio.setChecked(False)
        self._update_storage_hint()

    @staticmethod
    def _system_source_text(source):
        mode = str(source.get("mode") or "")
        if mode == "native":
            return f"{source.get('module', '')}.{source.get('class', '')}".strip(".")
        if mode == "pack":
            return f"Pack: {source.get('provider') or source.get('id')}"
        if mode == "entry-point":
            provider = source.get("distribution") or source.get("value")
            return f"Entry point: {provider}"
        return str(source.get("path") or "")

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
        name_item = QtWidgets.QTableWidgetItem(name)
        label = (
            getattr(parameter, "label", name.replace("_", " ").title())
            if parameter is not None
            else ""
        )
        name_item.setData(_PARAM_LABEL_ROLE, label)
        self.params_table.setItem(row, 0, name_item)
        kind_combo = QtWidgets.QComboBox()
        kind_combo.addItems(_KIND_CHOICES)
        kind_combo.setCurrentText(kind if kind in _KIND_CHOICES else "float")
        kind_combo.setEnabled(editable)
        kind_combo.currentIndexChanged.connect(lambda *_: self._apply_user_edits())
        self.params_table.setCellWidget(row, 1, kind_combo)
        for column, attribute in ((2, "default"), (3, "minimum"), (4, "maximum"), (5, "step")):
            value = getattr(parameter, attribute, None) if parameter is not None else None
            self.params_table.setItem(row, column, QtWidgets.QTableWidgetItem(_text(value)))
        description = getattr(parameter, "description", "") if parameter is not None else ""
        self.params_table.setItem(row, 6, QtWidgets.QTableWidgetItem(description))

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
                "label": (name_item.data(_PARAM_LABEL_ROLE) or name.replace("_", " ").title()),
                "kind": kind,
            }
            for column, key in ((2, "default"), (3, "minimum"), (4, "maximum"), (5, "step")):
                value = self._parse_number(self.params_table.item(row, column), kind)
                if value is not None:
                    payload[key] = value
            description_item = self.params_table.item(row, 6)
            if description_item is not None and description_item.text().strip():
                payload["description"] = description_item.text().strip()
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
            for button in (
                self.duplicate_button,
                self.remove_button,
                self.unhide_button,
                self.open_file_button,
            ):
                button.setEnabled(False)
            return
        is_user = _is_user_op(operation_id)
        hidden = operation_id in library.hidden_operations()
        self.duplicate_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        # The icon follows the action, not just the tooltip: hiding a system op
        # is reversible (the Restore button brings it back), so a trash can
        # overstates it. Deleting a user op really does remove its files.
        set_button_icon(
            self.remove_button,
            "delete" if is_user else "visibility_off",
            tooltip="Remove this user operation" if is_user else "Hide this operation",
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
        path = library.user_operation_source_path(operation_id)
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

    def _new_operation(self):
        try:
            operation_id = library.create_empty_user_operation()
        except Exception as exc:
            show_status_message(self._window, f"Could not create operation: {exc}", timeout=4000)
            return
        self.select_operation(operation_id)
        show_status_message(
            self._window,
            "Created an empty operation — choose a source file or open its code.",
            timeout=3500,
        )

    def _duplicate_operation(self):
        source_id = self.selected_operation_id()
        if source_id is None:
            return
        try:
            operation_id = library.duplicate_operation(source_id)
        except Exception as exc:
            show_status_message(self._window, f"Duplicate failed: {exc}", timeout=4000)
            return
        self.select_operation(operation_id)
        wrapper = library.user_operation_wrapper(operation_id) or {}
        template = wrapper.get("template") or {}
        message = str(template.get("message") or "Created an editable copy.")
        show_status_message(self._window, message, timeout=4500)

    # ------------------------------------------------------------------
    # Source / callable editor
    # ------------------------------------------------------------------

    def _choose_source_file(self):
        if not _is_user_op(self._loaded_id or ""):
            return
        path, _ = get_open_file_name(
            self, "Choose operation source", "", "Python (*.py);;All files (*)"
        )
        if not path:
            return
        self._populate_source_file(path)

    def _populate_source_file(self, path):
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
        self._suppress = True
        try:
            self.source_path_edit.setText(path)
            self._source_infos = {info.name: info for info in infos}
            self.callable_combo.clear()
            self.callable_combo.addItems([info.name for info in infos])
            self.callable_combo.setCurrentIndex(0)
        finally:
            self._suppress = False
        self._retarget_source(infer=True)

    def _on_callable_changed(self, _name):
        if self._suppress or self._reloading:
            return
        self._retarget_source(infer=True)

    def _on_storage_mode_changed(self, checked):
        self._update_storage_hint()
        if not checked or self._suppress or self._reloading:
            return
        self._retarget_source(infer=False)

    def _update_storage_hint(self, *_args):
        if self.link_radio.isChecked():
            self.storage_hint.setText(
                "Linked: edits to the original file are picked up automatically."
            )
        elif self.copy_radio.isChecked():
            self.storage_hint.setText(
                "Copied: ArrayScope keeps its own file, independent of the original."
            )
        else:
            self.storage_hint.setText(
                "System implementation. Duplicate it to choose editable source storage."
            )

    def _retarget_source(self, *, infer=True):
        if self._suppress or self._reloading:
            return
        operation_id = self._loaded_id
        if operation_id is None or not _is_user_op(operation_id):
            return
        path = self.source_path_edit.text().strip()
        callable_name = self.callable_combo.currentText().strip()
        if not path or not callable_name:
            return
        try:
            library.update_user_operation_source(
                operation_id,
                path,
                callable_name,
                link=self.link_radio.isChecked(),
                infer=infer,
            )
        except Exception as exc:
            show_status_message(self._window, f"Could not update source: {exc}", timeout=4000)
            return
        self.select_operation(operation_id)
