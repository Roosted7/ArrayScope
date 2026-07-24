"""Tests for the operation parameter popup (stage 2 of the add flow)."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from arrayscope.operations.parameter_forms import build_parameter_form
from arrayscope.operations.registry import get_operation_entry
from arrayscope.ui.operation_params_popup import OperationParamsPopup
from tests.ui.helpers import process_events


def _crop_popup(qtbot, on_accept):
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    popup = OperationParamsPopup(entry, form, on_accept)
    qtbot.addWidget(popup)
    return popup, form


def test_crop_form_renders_two_spinboxes_and_derived_line(qtbot):
    popup, _form = _crop_popup(qtbot, on_accept=lambda values: None)
    spins = popup.findChildren(QtWidgets.QSpinBox)
    assert len(spins) == 2
    # The derived "Output length" line is present and reflects the full axis.
    derived_texts = [label.text() for label in popup._derived_labels]
    assert any("Output length: 10" in text for text in derived_texts)


def test_editing_start_past_stop_nudges_stop(qtbot):
    popup, form = _crop_popup(qtbot, on_accept=lambda values: None)
    # Full axis is [0, 10). Drag start up to 10 -> stop is nudged to stay above.
    popup._spins["start"].setValue(10)
    process_events(qtbot, count=2)
    assert form.field("start").value < form.field("stop").value
    # Widgets re-synced from the interdependence-adjusted model.
    assert popup._spins["stop"].value() == form.field("stop").value


def test_invalid_state_disables_confirm(qtbot):
    entry = get_operation_entry("crop")
    form = build_parameter_form(entry, shape=(4, 10, 6), axis=1)
    popup = OperationParamsPopup(entry, form, lambda values: None)
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
    popup, _form = _crop_popup(qtbot, on_accept=lambda values: captured.update(values))
    popup._spins["start"].setValue(2)
    popup._spins["stop"].setValue(7)
    process_events(qtbot, count=2)
    popup._accept()
    assert captured == {"start": 2, "stop": 7}


@pytest.mark.skipif(
    build_parameter_form(get_operation_entry("mean"), shape=(4, 5), axis=0) is not None,
    reason="mean is expected to be parameterless",
)
def test_float_op_uses_double_spinbox(qtbot):
    from arrayscope.operations.packs.sigpy_pack import sigpy_available

    if not sigpy_available():
        pytest.skip("sigpy not installed")
    entry = get_operation_entry("sigpy:soft_thresh")
    form = build_parameter_form(entry, shape=(4, 5), axis=0)
    popup = OperationParamsPopup(entry, form, lambda values: None)
    qtbot.addWidget(popup)
    assert popup.findChildren(QtWidgets.QDoubleSpinBox)
    popup._spins["lamda"].setValue(0.5)
    process_events(qtbot, count=2)
    assert form.field("lamda").value == pytest.approx(0.5)
