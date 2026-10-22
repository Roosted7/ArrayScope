"""Export selections must preview the scientific array they will actually save."""

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.dialogs import SaveRangeDialog


@pytest.mark.parametrize(
    ("shape", "squeeze"),
    [((1, 1), True), ((), True), ((0, 3), True), ((2, 1, 4), False)],
)
def test_shape_preview_agrees_with_numpy_for_scalar_empty_and_unsqueezed_outputs(
    qtbot, shape, squeeze
):
    dialog = SaveRangeDialog(None, shape)
    qtbot.addWidget(dialog)
    dialog._squeeze_checkbox.setChecked(squeeze)
    source = np.zeros(shape)
    result = source[tuple(slice(start, stop) for start, stop in dialog.get_ranges())]
    if squeeze:
        result = np.squeeze(result)
    assert str(result.shape) in dialog._output_shape_label.text()


def test_slider_changes_replace_stale_spinbox_limits_after_manual_edits(qtbot):
    dialog = SaveRangeDialog(None, (10,))
    qtbot.addWidget(dialog)
    slider, start, end = dialog._controls[0]
    start.setValue(5)
    slider.setValues(0, 1)
    assert (start.value(), end.value()) == slider.values() == (0, 1)
    assert dialog.get_ranges() == [(0, 2)]
    end.setValue(0)
    slider.setValues(8, 9)
    assert (start.value(), end.value()) == slider.values() == (8, 9)
    assert dialog.get_ranges() == [(8, 10)]


def test_typed_preview_and_keyboard_fields_describe_selected_values_before_export(qtbot):
    from arrayscope.core.axis_info import AxisInfo

    dialog = SaveRangeDialog(
        None,
        (4, 5),
        dtype=np.complex64,
        axes=(AxisInfo("y", "Row", 4), AxisInfo("x", "Column", 5)),
        action_text="Next…",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    _slider, start, _end = dialog._controls[0]
    assert "Row" in start.accessibleName()
    assert "inclusive" in dialog._controls[0][2].accessibleDescription()
    start.setFocus()
    qtbot.keyClick(start, QtCore.Qt.Key.Key_Up)
    assert dialog.get_ranges() == [(1, 4), (0, 5)]
    assert "(3, 5)" in dialog._output_shape_label.text()
    assert "complex64" in dialog._output_dtype_label.text()
    assert "120 bytes" in dialog._output_size_label.text()
    assert dialog.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).text() == "Next…"


def test_unknown_dtype_is_honest_and_cancel_is_available_from_the_keyboard(qtbot):
    dialog = SaveRangeDialog(None, (8, 8))
    qtbot.addWidget(dialog)
    dialog.show()
    assert "determined during export" in dialog._output_dtype_label.text()
    assert "determined during export" in dialog._output_size_label.text()
    qtbot.keyClick(dialog, QtCore.Qt.Key.Key_Escape)
    assert dialog.result() == QtWidgets.QDialog.DialogCode.Rejected


def test_large_axis_range_opens_without_allocating_array_data(qtbot):
    length = 2**31 + 7
    dialog = SaveRangeDialog(None, (length,), dtype=np.uint8)
    qtbot.addWidget(dialog)
    slider, start, end = dialog._controls[0]
    assert end.value() == length - 1
    start.setValue(length - 4)
    assert slider.values() == (length - 4, length - 1)
    assert dialog.get_ranges() == [(length - 4, length)]
    assert "4 bytes" in dialog._output_size_label.text()


def test_indices_above_float_exactness_roundtrip_through_keyboard_and_slider(qtbot):
    length = 2**53 + 100
    target = 2**53 + 17
    dialog = SaveRangeDialog(None, (length,), dtype=np.uint8)
    qtbot.addWidget(dialog)
    dialog.show()
    slider, start, end = dialog._controls[0]
    changes = []
    start.valueChanged.connect(changes.append)
    dialog.activateWindow()
    start.setFocus()
    qtbot.waitUntil(start.hasFocus)
    start.lineEdit().selectAll()
    qtbot.keyClicks(start.lineEdit(), str(target))
    assert start.value() == target
    assert start.lineEdit().text() == str(target)
    accessible = QtGui.QAccessible.queryAccessibleInterface(start)
    assert accessible.text(QtGui.QAccessible.Text.Value) == str(target)
    assert changes[-1] == target
    qtbot.keyClick(start, QtCore.Qt.Key.Key_Up)
    assert start.value() == target + 1
    assert slider.values() == (target + 1, length - 1)
    slider.setValues(target + 5, target + 9)
    assert start.value() == target + 5
    assert end.value() == target + 9
    assert dialog.get_ranges() == [(target + 5, target + 10)]
    assert "5 bytes" in dialog._output_size_label.text()
    start.lineEdit().selectAll()
    start.lineEdit().insert("1.5")
    assert start.value() == target + 5
    assert start.lineEdit().text() == str(target + 5)
    start.lineEdit().clear()
    qtbot.keyClick(start, QtCore.Qt.Key.Key_Tab)
    assert start.value() == target + 5
    assert start.lineEdit().text() == str(target + 5)
