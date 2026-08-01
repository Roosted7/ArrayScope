"""Tests for the operation parameter popup (stage 2 of the add flow)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from arrayscope.operations.input_slots import (
    SLOT_DIMENSION_SET,
    SLOT_FILE,
    SLOT_SAVED_ARRAY,
    OperationInputSlot,
    SlotBinding,
    SlotSourceOption,
)
from arrayscope.operations.parameter_forms import build_parameter_form
from arrayscope.operations.registry import (
    OperationEntry,
    OperationParameter,
    create_operation,
    get_operation_entry,
)
from arrayscope.ui.operation_params_popup import OperationParamsPopup


def _crop_popup(qtbot, on_accept):
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    popup = OperationParamsPopup(entry, form, on_accept)
    qtbot.addWidget(popup)
    return popup, form


def test_crop_form_renders_two_spinboxes_and_derived_line(qtbot):
    popup, _form = _crop_popup(qtbot, on_accept=lambda values, bindings: None)
    spins = popup.findChildren(QtWidgets.QSpinBox)
    assert len(spins) == 2
    # The derived "Output length" line is present and reflects the full axis.
    derived_texts = [label.text() for label in popup._derived_labels]
    assert any("Output length: 10" in text for text in derived_texts)


def test_editing_start_past_stop_nudges_stop(qtbot):
    popup, form = _crop_popup(qtbot, on_accept=lambda values, bindings: None)
    # Full axis is [0, 10). Drag start up to 10 -> stop is nudged to stay above.
    popup._editors["start"].setValue(10)
    assert form.field("start").value < form.field("stop").value
    # Widgets re-synced from the interdependence-adjusted model.
    assert popup._editors["stop"].value() == form.field("stop").value


def test_invalid_state_disables_confirm(qtbot):
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    popup = OperationParamsPopup(entry, form, lambda values, bindings: None)
    qtbot.addWidget(popup)
    # Force a bound violation directly (start below its minimum of 0) so
    # form.validate() returns a message; the popup must reflect it.
    form.field("start").value = -3
    popup._refresh_validation()
    assert not popup._confirm.isEnabled()
    assert popup._warning.isVisibleTo(popup)
    # Recovering to a valid value re-enables confirm and clears the warning.
    form.field("start").value = 1
    popup._refresh_validation()
    assert popup._confirm.isEnabled()
    assert not popup._warning.isVisibleTo(popup)


def test_accept_delivers_values_dict(qtbot):
    captured = {}
    popup, _form = _crop_popup(qtbot, on_accept=lambda values, bindings: captured.update(values))
    popup._editors["start"].setValue(2)
    popup._editors["stop"].setValue(7)
    popup._accept()
    assert captured == {"start": 2, "stop": 7}


def test_slot_renders_beside_parameters_and_accepts_binding(qtbot):
    slot = OperationInputSlot(
        "reference",
        "Reference array",
        "Choose an auxiliary array.",
        accepts=(SLOT_DIMENSION_SET,),
    )
    entry = OperationEntry(
        id="test:slot-popup",
        label="Slot popup",
        operation_type=object,
        input_slots=(slot,),
    )
    binding = SlotBinding(
        SLOT_DIMENSION_SET,
        source_id="document",
        indices=(None, None),
        label="Current dimension set",
    )
    form = build_parameter_form(
        entry,
        shape=(4, 5),
        slot_options={
            "reference": (SlotSourceOption(binding, binding.label, "Current displayed plane."),)
        },
    )
    captured = {}
    popup = OperationParamsPopup(
        entry,
        form,
        lambda values, bindings: captured.update(
            values=values,
            bindings=bindings,
        ),
    )
    qtbot.addWidget(popup)

    assert popup._confirm.isEnabled() is False
    assert popup._slot_combos["reference"].count() == 2
    popup._slot_combos["reference"].setCurrentIndex(1)
    popup._accept()

    assert captured == {"values": {}, "bindings": {"reference": binding}}


@pytest.mark.skipif(
    build_parameter_form(get_operation_entry("mean"), shape=(4, 5), axis=0) is not None,
    reason="mean is expected to be parameterless",
)
def test_float_op_uses_double_spinbox(qtbot):
    entry = get_operation_entry("soft_threshold")
    form = build_parameter_form(entry, shape=(4, 5), axis=0)
    popup = OperationParamsPopup(entry, form, lambda values, bindings: None)
    qtbot.addWidget(popup)
    assert popup.findChildren(QtWidgets.QDoubleSpinBox)
    popup._editors["threshold"].setValue(0.5)
    assert form.field("threshold").value == pytest.approx(0.5)


def _kinds_popup(qtbot, on_accept=lambda values, bindings: None):
    entry = OperationEntry(
        id="test:kinds-popup",
        label="Kinds",
        operation_type=object,
        requires_axis=False,
        parameters=(
            OperationParameter(
                "mode", "Mode", kind="enum", choices=("reflect", "edge"), default="reflect"
            ),
            OperationParameter("verbose", "Verbose", kind="bool", default=False),
            OperationParameter("wavelet", "Wavelet", kind="string", default="db4"),
            OperationParameter("model", "Model", kind="path"),
        ),
    )
    form = build_parameter_form(entry, shape=(4, 5), axis=0)
    popup = OperationParamsPopup(entry, form, on_accept)
    qtbot.addWidget(popup)
    return popup, form


def test_each_parameter_kind_renders_its_own_widget(qtbot):
    popup, form = _kinds_popup(qtbot)

    assert isinstance(popup._editors["mode"], QtWidgets.QComboBox)
    assert isinstance(popup._editors["verbose"], QtWidgets.QCheckBox)
    assert isinstance(popup._editors["wavelet"], QtWidgets.QLineEdit)
    assert isinstance(popup._editors["model"], QtWidgets.QLineEdit)
    # Only the path field carries a browse button beside its line edit.
    assert popup._editors["mode"].count() == 2

    popup._editors["mode"].setCurrentIndex(1)
    popup._editors["verbose"].setChecked(True)
    popup._editors["wavelet"].setText("sym8")
    popup._editors["wavelet"].editingFinished.emit()

    assert form.values() == {
        "mode": "edge",
        "verbose": True,
        "wavelet": "sym8",
        "model": "",
    }


def test_path_field_gets_a_browse_button_beside_its_line_edit(qtbot, monkeypatch):
    popup, form = _kinds_popup(qtbot)

    row = popup._editors["model"].parentWidget()
    assert row.layout().count() == 2  # line edit + browse
    assert popup._editors["wavelet"].parentWidget() is not row

    monkeypatch.setattr(
        "arrayscope.ui.operation_params_popup.get_open_file_name",
        lambda *_args, **_kwargs: ("/tmp/weights.pt", ""),
    )
    popup._choose_path("model")

    assert popup._editors["model"].text() == "/tmp/weights.pt"
    assert form.values()["model"] == "/tmp/weights.pt"

    # A cancelled dialog leaves the field alone.
    monkeypatch.setattr(
        "arrayscope.ui.operation_params_popup.get_open_file_name",
        lambda *_args, **_kwargs: ("", ""),
    )
    popup._choose_path("model")
    assert form.values()["model"] == "/tmp/weights.pt"


@pytest.mark.parametrize(
    ("kind", "prompt", "pattern", "expected_label"),
    [
        (SLOT_SAVED_ARRAY, "Choose a saved array", "NumPy arrays (*.npy)", "Saved array — r.npy"),
        (SLOT_FILE, "Choose a file", "All files (*)", "File — r.npy"),
    ],
)
def test_browsing_binds_a_file_backed_slot(
    qtbot, monkeypatch, kind, prompt, pattern, expected_label
):
    slot = OperationInputSlot("reference", "Reference", accepts=(kind,))
    entry = OperationEntry(
        id=f"test:browse-{kind}",
        label="Browse",
        operation_type=object,
        requires_axis=False,
        input_slots=(slot,),
    )
    form = build_parameter_form(entry, shape=(4, 5))
    popup = OperationParamsPopup(entry, form, lambda values, bindings: None)
    qtbot.addWidget(popup)
    assert form.validate() is not None  # unbound to start with

    monkeypatch.setattr(
        "arrayscope.ui.operation_params_popup.get_open_file_name",
        lambda *_args, **_kwargs: ("/tmp/r.npy", ""),
    )
    popup._choose_slot_file("reference", kind, prompt, pattern)

    binding = form.bindings()["reference"]
    assert binding.kind == kind
    assert binding.path == "/tmp/r.npy"
    assert binding.label == expected_label
    assert popup._slot_combos["reference"].currentText() == expected_label


# --- axis-set selector -------------------------------------------------------


def _axes_popup(qtbot, operation_id, on_accept, **form_kwargs):
    entry = get_operation_entry(operation_id)
    form = build_parameter_form(
        entry,
        shape=(4, 10, 6),
        axis_choices=(("dim 0 [4]", 0), ("dim 1 [10] Y", 1), ("dim 2 [6] X", 2)),
        display_axes=(1, 2),
        **form_kwargs,
    )
    popup = OperationParamsPopup(entry, form, on_accept)
    qtbot.addWidget(popup)
    return popup, form


def test_axis_set_operation_renders_one_checkbox_per_dimension(qtbot):
    popup, form = _axes_popup(qtbot, "centered_fftn", on_accept=lambda values, bindings: None)

    checks = popup.findChildren(QtWidgets.QCheckBox)
    assert [check.text() for check in checks] == [
        "dim 0 [4]",
        "dim 1 [10] Y",
        "dim 2 [6] X",
    ]
    # Defaults to the displayed image, labelled the same way the single-axis
    # combo labels its entries.
    assert [check.isChecked() for check in checks] == [False, True, True]
    assert form.selected_axes() == (1, 2)


def test_toggling_an_axis_checkbox_updates_the_form(qtbot):
    popup, form = _axes_popup(qtbot, "centered_fftn", on_accept=lambda values, bindings: None)

    popup._axis_checks[0].setChecked(True)
    popup._axis_checks[2].setChecked(False)

    assert form.selected_axes() == (0, 1)


def test_clearing_every_axis_disables_confirm(qtbot):
    popup, _form = _axes_popup(qtbot, "mean_axes", on_accept=lambda values, bindings: None)

    for check in popup._axis_checks.values():
        check.setChecked(False)

    assert popup._confirm.isEnabled() is False
    assert "at least 1" in popup._warning.text()

    popup._axis_checks[1].setChecked(True)
    assert popup._confirm.isEnabled() is True


def test_confirming_carries_the_axis_set_into_the_committed_values(qtbot):
    committed = []
    popup, _form = _axes_popup(
        qtbot,
        "mean_axes",
        on_accept=lambda values, bindings: committed.append(values),
    )

    popup._axis_checks[0].setChecked(True)
    popup._axis_checks[2].setChecked(False)
    popup._accept()

    # The committed mapping is exactly what ``create_operation`` takes, axis set
    # included -- no separate channel for the selection.
    assert committed == [{"axes": (0, 1)}]
    operation = create_operation("mean_axes", parameters=committed[0], ndim=3)
    assert operation.axes == (0, 1)
