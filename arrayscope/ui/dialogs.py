"""Standalone dialogs used by ArrayScope."""

from __future__ import annotations

from math import prod

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.core.memory_budget import format_bytes
from arrayscope.ui.widgets import RangeSlider

SPINBOX_STYLE = (
    "QAbstractSpinBox { font-size: 9pt; } QAbstractSpinBox:disabled { color: palette(mid); }"
)


class _LargeIndexSpinBox(QtWidgets.QAbstractSpinBox):
    """An exact integer editor for indices beyond QSpinBox's signed-32-bit range."""

    valueChanged = QtCore.Signal(object)

    def __init__(self):
        super().__init__()
        self._minimum, self._maximum, self._value = 0, 99, 0
        self.lineEdit().setText("0")
        self.lineEdit().textEdited.connect(self._text_edited)
        self.editingFinished.connect(self._finish_edit)

    def value(self):
        return self._value

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def setRange(self, minimum, maximum):
        self._minimum = int(minimum)
        self._maximum = max(self._minimum, int(maximum))
        self.setValue(self._value)

    def setMinimum(self, minimum):
        self.setRange(minimum, max(int(minimum), self._maximum))

    def setMaximum(self, maximum):
        self.setRange(min(self._minimum, int(maximum)), maximum)

    def setValue(self, value):
        value = max(self._minimum, min(int(value), self._maximum))
        if value != self._value:
            self._value = value
            self.lineEdit().setText(str(value))
            self.valueChanged.emit(value)

    def validate(self, text, position):
        state = QtGui.QValidator.State
        if not text:
            return state.Intermediate, text, position
        if not text.isdecimal():
            return state.Invalid, text, position
        try:
            value = int(text)
        except ValueError:
            return state.Invalid, text, position
        if value > self._maximum:
            return state.Invalid, text, position
        return (state.Acceptable if value >= self._minimum else state.Intermediate), text, position

    def _text_edited(self, text):
        if (
            self.keyboardTracking()
            and self.validate(text, 0)[0] == QtGui.QValidator.State.Acceptable
        ):
            value = int(text)
            if value != self._value:
                self._value = value
                self.valueChanged.emit(value)

    def _finish_edit(self):
        text = self.lineEdit().text()
        if self.validate(text, 0)[0] == QtGui.QValidator.State.Acceptable:
            self.setValue(int(text))
        self.lineEdit().setText(str(self._value))

    def fixup(self, _text):
        return str(self._value)

    def stepBy(self, steps):
        self._finish_edit()
        self.setValue(self._value + int(steps))

    def stepEnabled(self):
        flags = self.StepEnabledFlag.StepNone
        if self._value < self._maximum:
            flags |= self.StepEnabledFlag.StepUpEnabled
        if self._value > self._minimum:
            flags |= self.StepEnabledFlag.StepDownEnabled
        return flags


class SaveRangeDialog(QtWidgets.QDialog):
    """Dialog for selecting an inclusive index range along each dimension."""

    def __init__(self, parent, data_shape, *, dtype=None, axes=None, action_text="Save"):
        super().__init__(parent)
        self.setWindowTitle("Export as NumPy")
        self._controls = []
        self._data_shape = tuple(int(size) for size in data_shape)
        self._dtype = None if dtype is None else np.dtype(dtype)
        axes = tuple(axes) if axes is not None else ()
        if any(size < 0 for size in self._data_shape):
            raise ValueError("Dimension sizes cannot be negative")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        explanation = QtWidgets.QLabel(
            "Choose the values to export. Start and end indices are inclusive.\n"
            "Scientific values are saved in array axis order; display colors and scaling are not applied."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(10)

        for dim, size in enumerate(self._data_shape):
            max_index = max(0, size - 1)
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QGridLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setHorizontalSpacing(8)
            row_layout.setVerticalSpacing(4)

            axis_name = axes[dim].label if len(axes) == len(self._data_shape) else f"Dim {dim}"
            dimension_label = QtWidgets.QLabel(f"{axis_name} [{size:,}]")
            dimension_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            dimension_label.setWordWrap(True)
            dimension_label.setStyleSheet("QLabel { font-size: 9pt; font-weight: bold; }")
            slider = RangeSlider(row_widget, 0, max_index)
            slider.setAccessibleName(f"{axis_name} range")
            slider.setAccessibleDescription("Use the Start and End fields to select exact indices.")

            spinbox_class = _LargeIndexSpinBox if max_index > 2**31 - 1 else QtWidgets.QSpinBox
            start_spinbox = spinbox_class()
            end_spinbox = spinbox_class()
            for spinbox in (start_spinbox, end_spinbox):
                spinbox.setRange(0, max_index)
                spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                spinbox.setStyleSheet(SPINBOX_STYLE)
                if spinbox_class is _LargeIndexSpinBox:
                    metrics = spinbox.fontMetrics()
                    # Fit 64-bit indices in full. Still longer integers remain
                    # exact and editable through native horizontal caret scrolling.
                    text_width = min(
                        metrics.horizontalAdvance(str(max_index)),
                        metrics.horizontalAdvance("9" * 20),
                    )
                    spinbox.setFixedWidth(max(70, text_width + 20))
                else:
                    spinbox.setFixedWidth(70)

            if spinbox_class is _LargeIndexSpinBox:
                slider.setToolTip(
                    "Drag for approximate bounds; use Start and End for exact large indices."
                )

            end_spinbox.setValue(max_index)
            for position, spinbox in (("start", start_spinbox), ("end", end_spinbox)):
                spinbox.setAccessibleName(f"{axis_name} {position} index")
                spinbox.setAccessibleDescription(
                    f"Zero-based inclusive index, from 0 to {max_index}."
                    if size
                    else "This dimension is empty; there are no indices to select."
                )
                if size == 0:
                    spinbox.setSpecialValueText("Empty")
                    spinbox.setEnabled(False)
            slider.setEnabled(size > 0)
            start_spinbox.valueChanged.connect(
                lambda value, end_box=end_spinbox: end_box.setMinimum(value)
            )
            end_spinbox.valueChanged.connect(
                lambda value, start_box=start_spinbox: start_box.setMaximum(value)
            )
            start_spinbox.valueChanged.connect(
                lambda value, slider_widget=slider, end_box=end_spinbox: slider_widget.setValues(
                    value, end_box.value()
                )
            )
            end_spinbox.valueChanged.connect(
                lambda value, slider_widget=slider, start_box=start_spinbox: (
                    slider_widget.setValues(start_box.value(), value)
                )
            )
            start_spinbox.valueChanged.connect(lambda _value: self._update_output_shape_label())
            end_spinbox.valueChanged.connect(lambda _value: self._update_output_shape_label())
            slider.valuesChanged.connect(
                lambda lower, upper, start_box=start_spinbox, end_box=end_spinbox: (
                    self._sync_spinboxes(
                        start_box,
                        end_box,
                        lower,
                        upper,
                    )
                )
            )

            slider.setValues(0, max_index)

            row_layout.addWidget(dimension_label, 0, 0)
            row_layout.addWidget(slider, 0, 1, 1, 4)
            start_label = QtWidgets.QLabel("Start")
            start_label.setBuddy(start_spinbox)
            row_layout.addWidget(start_label, 1, 0)
            row_layout.addWidget(start_spinbox, 1, 1)
            row_layout.addItem(
                QtWidgets.QSpacerItem(
                    24,
                    1,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Minimum,
                ),
                1,
                2,
            )
            end_label = QtWidgets.QLabel("End (inclusive)")
            end_label.setBuddy(end_spinbox)
            row_layout.addWidget(end_label, 1, 3)
            row_layout.addWidget(end_spinbox, 1, 4)

            content_layout.addWidget(row_widget)
            row_layout.setColumnStretch(1, 0)
            row_layout.setColumnStretch(2, 1)
            row_layout.setColumnStretch(4, 0)
            self._controls.append((slider, start_spinbox, end_spinbox))

        if not self._data_shape:
            content_layout.addWidget(QtWidgets.QLabel("This array contains one scalar value."))
        content_layout.addStretch()
        scroll_area.setWidget(content)
        scroll_area.setMaximumHeight(min(440, 84 * max(len(data_shape), 1)))
        layout.addWidget(scroll_area)

        self._squeeze_checkbox = QtWidgets.QCheckBox("Remove dimensions of length 1 (squeeze)")
        self._squeeze_checkbox.setToolTip(
            "Equivalent to numpy.squeeze: removes selected dimensions of length 1. "
            "Selecting one value in every dimension produces a scalar with shape ()."
        )
        self._squeeze_checkbox.setChecked(True)
        self._squeeze_checkbox.toggled.connect(self._update_output_shape_label)
        layout.addWidget(self._squeeze_checkbox)

        preview = QtWidgets.QGroupBox("Export preview")
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setSpacing(4)
        self._output_shape_label = QtWidgets.QLabel()
        self._output_dtype_label = QtWidgets.QLabel()
        self._output_size_label = QtWidgets.QLabel()
        for label in (self._output_shape_label, self._output_dtype_label, self._output_size_label):
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            preview_layout.addWidget(label)
        self._output_size_label.setToolTip(
            "Logical array data bytes, excluding file headers and compression. "
            "Working memory for operations is reported separately."
        )
        layout.addWidget(preview)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.buttons = buttons
        next_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save)
        next_button.setText(action_text)
        next_button.setDefault(True)
        layout.addWidget(buttons)

        self._update_output_shape_label()
        self.resize(640, min(640, 260 + 84 * max(len(data_shape), 1)))

    def _sync_spinboxes(self, start_spinbox, end_spinbox, lower_value, upper_value):
        with QtCore.QSignalBlocker(start_spinbox), QtCore.QSignalBlocker(end_spinbox):
            # A previous manual edit tightened the opposite field's bounds.
            # Replace those bounds before applying the slider's complete pair.
            start_spinbox.setMaximum(end_spinbox.maximum())
            end_spinbox.setMinimum(0)
            start_spinbox.setValue(lower_value)
            end_spinbox.setValue(upper_value)
            start_spinbox.setMaximum(upper_value)
            end_spinbox.setMinimum(lower_value)
        self._update_output_shape_label()

    def _selected_shape(self):
        selected_shape = tuple(stop - start for start, stop in self.get_ranges())
        if self.should_squeeze():
            return tuple(size for size in selected_shape if size != 1)
        return selected_shape

    def _update_output_shape_label(self):
        shape = self._selected_shape()
        self._output_shape_label.setText(
            f"Output shape: {shape}" + (" — scalar" if not shape else "")
        )
        self._output_dtype_label.setText(
            "Data type: determined during export"
            if self._dtype is None
            else f"Data type: {self._dtype}"
        )
        if self._dtype is None or self._dtype.hasobject:
            size_text = "Array data size: determined during export"
        else:
            nbytes = prod(shape) * self._dtype.itemsize
            size_text = f"Array data: {nbytes:,} bytes"
            if nbytes >= 1024:
                size_text += f" ({format_bytes(nbytes)})"
        self._output_size_label.setText(size_text)

    def get_ranges(self):
        """Return Python slice bounds as (start, stop) tuples."""
        return [
            (min(start.value(), size), min(end.value() + 1, size))
            for size, (_, start, end) in zip(self._data_shape, self._controls, strict=True)
        ]

    def should_squeeze(self):
        return self._squeeze_checkbox.isChecked()
