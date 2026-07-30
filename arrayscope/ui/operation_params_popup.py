"""Anchored popup for editing an operation's parameters.

Stage 2 of the two-stage add flow (and the pencil-edit flow): given an
:class:`~arrayscope.operations.registry.OperationEntry` and a built
:class:`~arrayscope.operations.parameter_forms.ParameterForm`, render one editor
per field, the form's read-only derived lines, and a confirm button.

One widget per parameter kind: a spinbox for ``int``/``float``, a combo box for
``enum``, a check box for ``bool``, a line edit for ``string``, and a line edit
plus a browse button for ``path``.

An operation that takes an axis SET renders one checkbox per dimension above
its parameters -- the add popup's inline combo cannot express a set, so
axis-set operations are routed here even when they have no parameters at all.

Editing any field routes through :meth:`ParameterForm.set_value`, which may
adjust *other* fields (crop ``start`` nudging ``stop``); every widget is then
re-synced from the form so the popup always mirrors the model. Validation
failures surface inline and disable confirm.
"""

from __future__ import annotations

from collections.abc import Callable

from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.operations.input_slots import SLOT_FILE, SLOT_SAVED_ARRAY, SlotBinding
from arrayscope.operations.parameter_forms import ParameterForm
from arrayscope.operations.registry import OperationEntry
from arrayscope.ui.bubbles import EditBubble
from arrayscope.ui.file_dialogs import get_open_file_name
from arrayscope.ui.icons import material_icon, set_button_icon

# Generous fallbacks for an unbounded side of a field, so a spinbox never
# silently clamps a legitimate value while still refusing absurd input.
_INT_LIMIT = 2_147_483_647
_FLOAT_LIMIT = 1.0e12


def _decimals_for(step: float | int | None) -> int:
    """Pick a sensible decimal count from a float step (0.01 -> 2, 1 -> 0)."""

    if not step:
        return 3
    text = repr(float(step))
    if "e" in text or "E" in text:
        return 6
    if "." not in text:
        return 0
    return min(6, len(text.split(".", 1)[1].rstrip("0")) or 1)


class OperationParamsPopup(EditBubble):
    """Popup rendering a :class:`ParameterForm` for one operation."""

    def __init__(
        self,
        entry: OperationEntry,
        form: ParameterForm,
        on_accept: Callable[[dict, dict], None],
        parent=None,
    ) -> None:
        super().__init__(parent, icon_name=None)
        # A Qt.Popup auto-closes on focus loss; with WA_DeleteOnClose that would
        # delete the C++ object out from under callers still holding a Python
        # reference (and, headless, it auto-closes the instant it is shown over
        # an active window). Keep the object alive on close -- the owning window
        # is the natural parent and reaps it, and callers deleteLater the prior
        # popup when opening a new one.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._entry = entry
        self._form = form
        self._on_accept = on_accept
        self._editors: dict[str, QtWidgets.QWidget] = {}
        self._slot_combos: dict[str, QtWidgets.QComboBox] = {}
        self._derived_labels: list[QtWidgets.QLabel] = []
        self._syncing = False

        container = QtWidgets.QWidget(self)
        column = QtWidgets.QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        column.addLayout(self._build_header())

        self._fields_layout = QtWidgets.QFormLayout()
        self._fields_layout.setContentsMargins(0, 0, 0, 0)
        self._fields_layout.setSpacing(4)
        self._axis_checks: dict[int, QtWidgets.QCheckBox] = {}
        if form.axes_field is not None:
            self._fields_layout.addRow(form.axes_field.label, self._build_axes_row(form.axes_field))
        for slot in form.slot_fields:
            row = QtWidgets.QWidget(self)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            combo = QtWidgets.QComboBox(self)
            combo.addItem("Choose input…", SlotBinding(""))
            for option in slot.options:
                combo.addItem(option.label, option.binding)
                index = combo.count() - 1
                combo.setItemData(index, option.description, QtCore.Qt.ItemDataRole.ToolTipRole)
                if not option.available:
                    item = combo.model().item(index)
                    if item is not None:
                        item.setEnabled(False)
                    combo.setItemData(
                        index,
                        option.unavailable_reason,
                        QtCore.Qt.ItemDataRole.ToolTipRole,
                    )
            if slot.binding.is_bound and combo.findData(slot.binding) < 0:
                combo.addItem(slot.binding.label or slot.binding.kind, slot.binding)
            combo.setCurrentIndex(max(0, combo.findData(slot.binding)))
            combo.setToolTip(slot.description)
            combo.currentIndexChanged.connect(
                lambda _index, name=slot.name: self._on_slot_changed(name)
            )
            self._slot_combos[slot.name] = combo
            row_layout.addWidget(combo, 1)
            for kind, prompt, pattern in (
                (SLOT_SAVED_ARRAY, "Choose a saved array", "NumPy arrays (*.npy)"),
                (SLOT_FILE, "Choose a file", "All files (*)"),
            ):
                if kind not in slot.accepts:
                    continue
                browse = QtWidgets.QToolButton(self)
                set_button_icon(browse, "folder_open", tooltip=prompt)
                browse.clicked.connect(
                    lambda _checked=False, name=slot.name, kind=kind, prompt=prompt, pattern=pattern: (
                        self._choose_slot_file(name, kind, prompt, pattern)
                    )
                )
                row_layout.addWidget(browse)
            self._fields_layout.addRow(slot.label, row)
        for field in form.fields:
            editor, row = self._make_editor(field)
            self._editors[field.name] = editor
            self._fields_layout.addRow(field.label, row)
        column.addLayout(self._fields_layout)

        self._derived_layout = QtWidgets.QVBoxLayout()
        self._derived_layout.setContentsMargins(0, 0, 0, 0)
        self._derived_layout.setSpacing(0)
        column.addLayout(self._derived_layout)

        self._warning = QtWidgets.QLabel()
        self._warning.setObjectName("OperationParamsWarning")
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("QLabel { color: #d9534f; font-size: 8pt; }")
        self._warning.setVisible(False)
        column.addWidget(self._warning)

        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addStretch(1)
        self._confirm = QtWidgets.QToolButton(self)
        set_button_icon(self._confirm, "done", tooltip="Apply")
        self._confirm.clicked.connect(self._accept)
        bottom.addWidget(self._confirm)
        column.addLayout(bottom)

        self.add_widget(container, 1)
        self._sync_from_form()

    # -- construction helpers -------------------------------------------------

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        header = QtWidgets.QVBoxLayout()
        header.setSpacing(0)
        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(6)
        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(material_icon(self._entry.icon).pixmap(16, 16))
        title_row.addWidget(icon_label)
        title = QtWidgets.QLabel(self._entry.label.rstrip("."))
        title.setStyleSheet("QLabel { font-weight: 600; }")
        title_row.addWidget(title)
        title_row.addStretch(1)
        header.addLayout(title_row)
        if self._entry.description:
            subtitle = QtWidgets.QLabel(self._entry.description)
            subtitle.setWordWrap(True)
            subtitle.setStyleSheet("QLabel { color: palette(mid); font-size: 8pt; }")
            header.addWidget(subtitle)
        return header

    def _build_axes_row(self, axes_field) -> QtWidgets.QWidget:
        """One checkbox per dimension: the axis-SET counterpart of the axis combo.

        A row of checkboxes rather than a checkable combo box, deliberately.
        Every axis and its state is visible at once -- which matters when the
        thing being chosen is "which two of these five dimensions" -- and it
        avoids styling ``QComboBox`` internals, which is how this codebase last
        lost the native drop-down arrow application-wide.
        """

        row = QtWidgets.QWidget(self)
        row.setObjectName("OperationAxesRow")
        layout = QtWidgets.QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        selected = set(axes_field.selected)
        for label, axis in axes_field.choices:
            check = QtWidgets.QCheckBox(str(label), row)
            check.setChecked(int(axis) in selected)
            if axes_field.description:
                check.setToolTip(axes_field.description)
            check.toggled.connect(lambda _checked, axis=int(axis): self._on_axis_toggled(axis))
            self._axis_checks[int(axis)] = check
            layout.addWidget(check)
        return row

    def _on_axis_toggled(self, axis: int) -> None:
        if self._syncing:
            return
        self._form.set_selected_axes(
            [value for value, check in self._axis_checks.items() if check.isChecked()]
        )
        self._refresh_derived()
        self._refresh_validation()

    def _make_editor(self, field) -> tuple[QtWidgets.QWidget, QtWidgets.QWidget]:
        """The value widget for one field, plus the widget to put in the row.

        They differ only for ``path``, where the row also carries a browse
        button; every other kind edits in place.
        """

        if field.kind == "enum":
            combo = QtWidgets.QComboBox(self)
            combo.addItems([str(choice) for choice in field.choices])
            combo.setCurrentIndex(max(0, combo.findText(str(field.value))))
            combo.setEnabled(not field.read_only)
            combo.currentIndexChanged.connect(lambda _index, name=field.name: self._on_edit(name))
            self._finish_editor(combo, field)
            return combo, combo
        if field.kind == "bool":
            check = QtWidgets.QCheckBox(self)
            check.setChecked(bool(field.value))
            check.setEnabled(not field.read_only)
            check.toggled.connect(lambda _checked, name=field.name: self._on_edit(name))
            self._finish_editor(check, field)
            return check, check
        if field.kind in ("string", "path"):
            line = QtWidgets.QLineEdit(self)
            line.setText(str(field.value))
            line.setReadOnly(bool(field.read_only))
            line.editingFinished.connect(lambda name=field.name: self._on_edit(name))
            self._finish_editor(line, field)
            if field.kind != "path":
                return line, line
            row = QtWidgets.QWidget(self)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)
            row_layout.addWidget(line, 1)
            browse = QtWidgets.QToolButton(self)
            set_button_icon(browse, "folder_open", tooltip="Choose a file")
            browse.clicked.connect(lambda _checked=False, name=field.name: self._choose_path(name))
            row_layout.addWidget(browse)
            return line, row
        spin = self._make_spin(field)
        return spin, spin

    def _finish_editor(self, widget: QtWidgets.QWidget, field) -> None:
        if field.description:
            widget.setToolTip(field.description)

    def _make_spin(self, field) -> QtWidgets.QAbstractSpinBox:
        if field.kind == "float":
            spin = QtWidgets.QDoubleSpinBox(self)
            low = float(field.minimum) if field.minimum is not None else -_FLOAT_LIMIT
            high = float(field.maximum) if field.maximum is not None else _FLOAT_LIMIT
            spin.setDecimals(_decimals_for(field.step))
            spin.setRange(low, high)
            spin.setSingleStep(float(field.step) if field.step else 0.1)
            spin.setValue(float(field.value))
        else:
            spin = QtWidgets.QSpinBox(self)
            low = int(field.minimum) if field.minimum is not None else -_INT_LIMIT
            high = int(field.maximum) if field.maximum is not None else _INT_LIMIT
            spin.setRange(low, high)
            spin.setSingleStep(int(field.step) if field.step else 1)
            spin.setValue(int(field.value))
        if field.description:
            spin.setToolTip(field.description)
        spin.setReadOnly(bool(field.read_only))
        spin.valueChanged.connect(lambda _value, name=field.name: self._on_edit(name))
        return spin

    # -- interaction ----------------------------------------------------------

    def _editor_value(self, name: str):
        editor = self._editors[name]
        if isinstance(editor, QtWidgets.QComboBox):
            return editor.currentText()
        if isinstance(editor, QtWidgets.QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QtWidgets.QLineEdit):
            return editor.text()
        return editor.value()

    def _on_edit(self, name: str) -> None:
        if self._syncing:
            return
        try:
            self._form.set_value(name, self._editor_value(name))
        except ValueError as exc:
            # A free-text field can hold something the kind refuses (an enum the
            # user typed, a malformed boolean). Report it and keep the model on
            # its last good value rather than crashing the popup.
            self._warning.setText(str(exc))
            self._warning.setVisible(True)
            self._confirm.setEnabled(False)
            return
        self._sync_from_form()

    def _choose_path(self, name: str) -> None:
        path, _selected_filter = get_open_file_name(self, "Choose file", "", "All files (*)")
        if not path:
            return
        self._form.set_value(name, path)
        self._sync_from_form()

    def _on_slot_changed(self, name: str) -> None:
        combo = self._slot_combos[name]
        binding = combo.currentData()
        self._form.set_binding(
            name,
            binding if isinstance(binding, SlotBinding) else SlotBinding(""),
        )
        self._refresh_validation()

    def _choose_slot_file(self, name: str, kind: str, prompt: str, pattern: str) -> None:
        path, _selected_filter = get_open_file_name(self, prompt, "", pattern)
        if not path:
            return
        prefix = "Saved array" if kind == SLOT_SAVED_ARRAY else "File"
        binding = SlotBinding(
            kind,
            path=path,
            label=f"{prefix} — {path.rsplit('/', 1)[-1]}",
        )
        combo = self._slot_combos[name]
        index = combo.findData(binding)
        if index < 0:
            combo.addItem(binding.label, binding)
            index = combo.count() - 1
        combo.setCurrentIndex(index)

    def _sync_from_form(self) -> None:
        """Push every field's current model value back into its widget.

        Runs after every edit because :meth:`ParameterForm.set_value` may have
        adjusted a *different* field than the one the user touched.
        """

        self._syncing = True
        try:
            selected = set(self._form.selected_axes())
            for axis, check in self._axis_checks.items():
                check.blockSignals(True)
                check.setChecked(axis in selected)
                check.blockSignals(False)
            for field in self._form.fields:
                editor = self._editors[field.name]
                editor.blockSignals(True)
                if isinstance(editor, QtWidgets.QComboBox):
                    editor.setCurrentIndex(max(0, editor.findText(str(field.value))))
                elif isinstance(editor, QtWidgets.QCheckBox):
                    editor.setChecked(bool(field.value))
                elif isinstance(editor, QtWidgets.QLineEdit):
                    editor.setText(str(field.value))
                elif field.kind == "float":
                    editor.setValue(float(field.value))
                else:
                    editor.setValue(int(field.value))
                editor.blockSignals(False)
        finally:
            self._syncing = False
        self._refresh_derived()
        self._refresh_validation()

    def _refresh_derived(self) -> None:
        for label in self._derived_labels:
            self._derived_layout.removeWidget(label)
            label.deleteLater()
        self._derived_labels = []
        for derived in self._form.derived():
            label = QtWidgets.QLabel(f"{derived.label}: {derived.text}")
            label.setStyleSheet("QLabel { color: palette(mid); font-size: 8pt; }")
            self._derived_layout.addWidget(label)
            self._derived_labels.append(label)

    def _refresh_validation(self) -> None:
        message = self._form.validate()
        if message:
            self._warning.setText(message)
            self._warning.setVisible(True)
            self._confirm.setEnabled(False)
        else:
            self._warning.setVisible(False)
            self._confirm.setEnabled(True)

    def _accept(self) -> None:
        if self._form.validate() is not None:
            return
        values = dict(self._form.values())
        bindings = dict(self._form.bindings())
        if self._on_accept is not None:
            self._on_accept(values, bindings)
        self.close()
