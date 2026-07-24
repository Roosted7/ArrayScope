"""Window-level tests for the callback-style operation add / edit flow.

These replace the old modal-dialog flow (``_crop_parameters_dialog`` /
``QInputDialog``): request_operation now opens a non-modal params popup and the
append happens in its accept callback. The sigpy regression test pins the
original "requires parameter lamda" bug -- a parameterized plugin op must reach
the document once its popup is confirmed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from arrayscope.operations.registry import operation_id_for
from tests.ui.helpers import clear_arrayscope_settings, process_events


def _window(qtbot, shape=(4, 8, 6)):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    data = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    win.show()
    process_events(qtbot)
    return win


def _op_ids(win):
    return [operation_id_for(step.operation) for step in win.document.steps]


def test_parameterless_op_appends_immediately(qtbot):
    win = _window(qtbot)
    assert win.request_operation("centered_fft", 0) is True
    assert _op_ids(win) == ["centered_fft"]


def test_crop_opens_params_popup_and_commits_on_accept(qtbot):
    win = _window(qtbot)
    # No nested event loop: request_operation returns while the popup is open.
    win.request_operation("crop", 1)
    popup = win._operation_params_popup
    assert popup is not None
    popup._spins["start"].setValue(2)
    popup._spins["stop"].setValue(6)
    process_events(qtbot, count=2)
    popup._accept()
    process_events(qtbot)
    assert _op_ids(win) == ["crop"]
    crop = win.document.steps[0].operation
    assert (int(crop.start), int(crop.stop)) == (2, 6)


def test_add_popup_parameterless_op_lands(qtbot):
    win = _window(qtbot)
    win.open_operation_adder()
    add_popup = win._operation_add_popup
    assert add_popup.select_operation("mean")
    add_popup.activate_current()
    process_events(qtbot)
    assert _op_ids(win) == ["mean"]


def test_add_popup_parameterized_op_chains_to_params_popup(qtbot):
    win = _window(qtbot)
    win.open_operation_adder()
    add_popup = win._operation_add_popup
    assert add_popup.select_operation("crop")
    add_popup.activate_current()
    process_events(qtbot)
    params_popup = win._operation_params_popup
    assert params_popup is not None
    params_popup._spins["stop"].setValue(3)
    process_events(qtbot, count=2)
    params_popup._accept()
    process_events(qtbot)
    assert _op_ids(win) == ["crop"]


def test_edit_operation_seeds_and_replaces_generic_op(qtbot):
    win = _window(qtbot)
    # Land a crop, then re-edit it through the generic pencil flow.
    win.operation_coordinator.append_operation("crop", axis=1, parameters={"start": 1, "stop": 5})
    win._set_document(win.operation_coordinator.document)
    win.edit_operation(0)
    popup = win._operation_params_popup
    assert popup is not None
    # Seeded with the operation's current values.
    assert popup._spins["start"].value() == 1
    assert popup._spins["stop"].value() == 5
    popup._spins["stop"].setValue(4)
    process_events(qtbot, count=2)
    popup._accept()
    process_events(qtbot)
    crop = win.document.steps[0].operation
    assert (int(crop.start), int(crop.stop)) == (1, 4)


def test_sigpy_soft_thresh_regression(qtbot):
    """The original bug: a sigpy op raised 'requires parameter lamda'.

    Driving the params popup's accept with lamda=0.5 must land the op.
    """
    from arrayscope.operations.packs.sigpy_pack import sigpy_available

    if not sigpy_available():
        pytest.skip("sigpy not installed")
    win = _window(qtbot)
    win.request_operation("sigpy:soft_thresh")  # requires_axis=False
    popup = win._operation_params_popup
    assert popup is not None
    popup._spins["lamda"].setValue(0.5)
    process_events(qtbot, count=2)
    popup._accept()
    process_events(qtbot)
    assert _op_ids(win) == ["sigpy:soft_thresh"]
    op = win.document.steps[0].operation
    assert dict(op.params)["lamda"] == pytest.approx(0.5)
