"""Standalone dialogs used by ArrayScope."""

from __future__ import annotations

from .qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtWidgets

from .widgets import RangeSlider


SPINBOX_STYLE = "QSpinBox { font-size: 9pt; } QSpinBox:disabled { color: palette(mid); }"


class SaveRangeDialog(QtWidgets.QDialog):
    """Dialog for selecting an inclusive index range along each dimension."""

    def __init__(self, parent, data_shape):
        super().__init__(parent)
        self.setWindowTitle("Save as NumPy")
        self._controls = []
        self._data_shape = tuple(data_shape)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)
        content_layout.setSpacing(10)

        for dim, size in enumerate(data_shape):
            max_index = max(0, size - 1)
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QGridLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setHorizontalSpacing(8)
            row_layout.setVerticalSpacing(4)

            dimension_label = QtWidgets.QLabel(f"Dim {dim}")
            dimension_label.setStyleSheet("QLabel { font-size: 9pt; font-weight: bold; }")
            slider = RangeSlider(row_widget, 0, max_index)

            start_spinbox = QtWidgets.QSpinBox()
            end_spinbox = QtWidgets.QSpinBox()
            for spinbox in (start_spinbox, end_spinbox):
                spinbox.setRange(0, max_index)
                spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                spinbox.setStyleSheet(SPINBOX_STYLE)
                spinbox.setFixedWidth(70)

            end_spinbox.setValue(max_index)
            start_spinbox.valueChanged.connect(lambda value, end_box=end_spinbox: end_box.setMinimum(value))
            end_spinbox.valueChanged.connect(lambda value, start_box=start_spinbox: start_box.setMaximum(value))
            start_spinbox.valueChanged.connect(
                lambda value, slider_widget=slider, end_box=end_spinbox: slider_widget.setValues(value, end_box.value())
            )
            end_spinbox.valueChanged.connect(
                lambda value, slider_widget=slider, start_box=start_spinbox: slider_widget.setValues(start_box.value(), value)
            )
            start_spinbox.valueChanged.connect(lambda _value: self._update_output_shape_label())
            end_spinbox.valueChanged.connect(lambda _value: self._update_output_shape_label())
            slider.valuesChanged.connect(
                lambda lower, upper, start_box=start_spinbox, end_box=end_spinbox: self._sync_spinboxes(
                    start_box,
                    end_box,
                    lower,
                    upper,
                )
            )

            slider.setValues(0, max_index)

            row_layout.addWidget(dimension_label, 0, 0)
            row_layout.addWidget(slider, 0, 1, 1, 4)
            row_layout.addWidget(QtWidgets.QLabel("start"), 1, 0)
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
            row_layout.addWidget(QtWidgets.QLabel("end"), 1, 3)
            row_layout.addWidget(end_spinbox, 1, 4)

            content_layout.addWidget(row_widget)
            row_layout.setColumnStretch(1, 0)
            row_layout.setColumnStretch(2, 1)
            row_layout.setColumnStretch(4, 0)
            self._controls.append((slider, start_spinbox, end_spinbox))

        content_layout.addStretch()
        scroll_area.setWidget(content)
        scroll_area.setMaximumHeight(min(440, 84 * max(len(data_shape), 1)))
        layout.addWidget(scroll_area)

        footer_layout = QtWidgets.QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)

        self._squeeze_checkbox = QtWidgets.QCheckBox("Squeeze singleton dimensions")
        self._squeeze_checkbox.setChecked(True)
        self._squeeze_checkbox.toggled.connect(self._update_output_shape_label)
        footer_layout.addWidget(self._squeeze_checkbox)
        self._output_shape_label = QtWidgets.QLabel()
        self._output_shape_label.setStyleSheet("QLabel { font-size: 9pt; color: palette(windowText); }")
        footer_layout.addWidget(self._output_shape_label)
        footer_layout.addStretch()
        layout.addLayout(footer_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_output_shape_label()
        self.resize(640, min(560, 150 + 84 * len(data_shape)))

    def _sync_spinboxes(self, start_spinbox, end_spinbox, lower_value, upper_value):
        start_spinbox.blockSignals(True)
        end_spinbox.blockSignals(True)
        start_spinbox.setValue(lower_value)
        end_spinbox.setValue(upper_value)
        start_spinbox.blockSignals(False)
        end_spinbox.blockSignals(False)
        self._update_output_shape_label()

    def _selected_shape(self):
        selected_shape = [max(end.value() - start.value() + 1, 0) for _, start, end in self._controls]
        if self.should_squeeze():
            squeezed_shape = [size for size in selected_shape if size != 1]
            return squeezed_shape or [1]
        return selected_shape

    def _update_output_shape_label(self):
        self._output_shape_label.setText(f"Output shape: {self._selected_shape()}")

    def get_ranges(self):
        """Return Python slice bounds as (start, stop) tuples."""
        return [(start.value(), end.value() + 1) for _, start, end in self._controls]

    def should_squeeze(self):
        return self._squeeze_checkbox.isChecked()
