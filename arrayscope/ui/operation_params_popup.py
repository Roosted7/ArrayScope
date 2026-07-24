"""Anchored popup for editing an operation's parameters.

Stage 2 of the two-stage add flow (and the pencil-edit flow): given an
:class:`~arrayscope.operations.registry.OperationEntry` and a built
:class:`~arrayscope.operations.parameter_forms.ParameterForm`, render one
spinbox per field, the form's read-only derived lines, and a confirm button.

Editing any field routes through :meth:`ParameterForm.set_value`, which may
adjust *other* fields (crop ``start`` nudging ``stop``); every widget is then
re-synced from the form so the popup always mirrors the model. Validation
failures surface inline and disable confirm.
"""

from __future__ import annotations

from collections.abc import Callable

from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.operations.parameter_forms import ParameterForm
from arrayscope.operations.registry import OperationEntry
from arrayscope.ui.bubbles import EditBubble
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
        on_accept: Callable[[dict], None],
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
        self._spins: dict[str, QtWidgets.QAbstractSpinBox] = {}
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
        for field in form.fields:
            spin = self._make_spin(field)
            self._spins[field.name] = spin
            self._fields_layout.addRow(field.label, spin)
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

    def _on_edit(self, name: str) -> None:
        if self._syncing:
            return
        self._form.set_value(name, self._spins[name].value())
        self._sync_from_form()

    def _sync_from_form(self) -> None:
        """Push every field's current model value back into its widget.

        Runs after every edit because :meth:`ParameterForm.set_value` may have
        adjusted a *different* field than the one the user touched.
        """

        self._syncing = True
        try:
            for field in self._form.fields:
                spin = self._spins[field.name]
                spin.blockSignals(True)
                if field.kind == "float":
                    spin.setValue(float(field.value))
                else:
                    spin.setValue(int(field.value))
                spin.blockSignals(False)
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
        if self._on_accept is not None:
            self._on_accept(values)
        self.close()
