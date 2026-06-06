"""Compact dimension role strip widgets."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.ui.icons import set_button_icon


class DimensionChip(QtWidgets.QFrame):
    roleChanged = Qt.QtCore.Signal(str, int)
    sliceChanged = Qt.QtCore.Signal(int, int)
    operationRequested = Qt.QtCore.Signal(int)
    focused = Qt.QtCore.Signal(int)

    def __init__(self, axis, parent=None):
        super().__init__(parent)
        if parent is not None:
            parent.installEventFilter(self)
        self.axis = int(axis)
        self.setObjectName(f"DimensionChip{axis}")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFocusPolicy(Qt.QtCore.Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda _pos: self.operationRequested.emit(self.axis))

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 3, 5, 3)
        layout.setSpacing(3)

        self.axis_label = QtWidgets.QLabel()
        self.axis_label.setMinimumWidth(52)
        layout.addWidget(self.axis_label)

        self.y_button = QtWidgets.QToolButton(checkable=True)
        self.x_button = QtWidgets.QToolButton(checkable=True)
        for role, button in (("y", self.y_button), ("x", self.x_button)):
            button.setFixedSize(24, 22)
            button.clicked.connect(lambda _checked=False, role=role: self.roleChanged.emit(role, self.axis))
            layout.addWidget(button)

        self.slice_spin = QtWidgets.QSpinBox()
        self.slice_spin.setMinimum(0)
        self.slice_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.slice_spin.setFixedWidth(58)
        self.slice_spin.valueChanged.connect(lambda value: self.sliceChanged.emit(self.axis, int(value)))
        layout.addWidget(self.slice_spin)

        self.ops_button = QtWidgets.QToolButton()
        set_button_icon(self.ops_button, "add", tooltip="Add operation on this dimension")
        self.ops_button.clicked.connect(lambda: self.operationRequested.emit(self.axis))
        layout.addWidget(self.ops_button)
        self.setLayout(layout)
        self.setMinimumWidth(190)
        self.setMaximumWidth(204)

    def update_state(self, shape, view_state, profile_axes=()):
        size = int(shape[self.axis])
        self.axis_label.setText(f"{self.axis} [{size}]")
        image_axes = view_state.image_axes or ()
        is_y = len(image_axes) > 0 and image_axes[0] == self.axis
        is_x = len(image_axes) > 1 and image_axes[1] == self.axis
        self.y_button.setChecked(is_y)
        self.x_button.setChecked(is_x)
        set_button_icon(self.y_button, "arrow_downward" if is_y and view_state.axis_flipped[self.axis] else "arrow_upward")
        set_button_icon(self.x_button, "arrow_forward" if is_x and view_state.axis_flipped[self.axis] else "arrow_back")
        self.y_button.setToolTip("Flip Y direction" if is_y else f"Use dim {self.axis} as image Y axis")
        self.x_button.setToolTip("Flip X direction" if is_x else f"Use dim {self.axis} as image X axis")
        is_display_axis = self.axis in image_axes
        is_singleton = size == 1
        self.y_button.setEnabled(not is_singleton and view_state.image_axes is not None)
        self.x_button.setEnabled(not is_singleton and view_state.image_axes is not None)
        self.slice_spin.blockSignals(True)
        try:
            self.slice_spin.setMaximum(max(0, size - 1))
            self.slice_spin.setValue(view_state.slice_indices[self.axis])
            self.slice_spin.setEnabled(not is_display_axis and not is_singleton)
            self.slice_spin.setVisible(not is_display_axis)
        finally:
            self.slice_spin.blockSignals(False)

    def focusInEvent(self, event):
        self.focused.emit(self.axis)
        super().focusInEvent(event)


class DimensionStrip(QtWidgets.QWidget):
    roleChanged = Qt.QtCore.Signal(str, int)
    sliceChanged = Qt.QtCore.Signal(int, int)
    operationRequested = Qt.QtCore.Signal(int)
    focusedAxisChanged = Qt.QtCore.Signal(int)

    def __init__(self, ndim, parent=None):
        super().__init__(parent)
        self.chips = []
        self._columns = 0
        self._relayout_pending = False
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for axis in range(int(ndim)):
            chip = DimensionChip(axis)
            chip.roleChanged.connect(self.roleChanged)
            chip.sliceChanged.connect(self.sliceChanged)
            chip.operationRequested.connect(self.operationRequested)
            chip.focused.connect(self.focusedAxisChanged)
            self.chips.append(chip)
        self.setLayout(layout)
        self._relayout()
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)

    def update_shape(self, shape):
        for axis, chip in enumerate(self.chips):
            chip.setVisible(axis < len(shape))
        self._relayout()

    def update_state(self, shape, view_state, profile_axes=()):
        self.update_shape(shape)
        for axis, chip in enumerate(self.chips):
            if axis < len(shape):
                chip.update_state(shape, view_state, profile_axes)

    def chip(self, axis):
        return self.chips[int(axis)]

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = self._column_count()
        if columns != self._columns:
            self._schedule_relayout()

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QtCore.QEvent.Type.Resize:
            self._schedule_relayout()
        return super().eventFilter(obj, event)

    def _schedule_relayout(self):
        if self._relayout_pending:
            return
        self._relayout_pending = True
        QtCore.QTimer.singleShot(0, self._run_scheduled_relayout)

    def _run_scheduled_relayout(self):
        self._relayout_pending = False
        self._relayout()

    def _column_count(self):
        visible = [chip for chip in self.chips if chip.isVisible()]
        if not visible:
            visible = self.chips
        parent = self.parentWidget()
        available_width = max(1, parent.width() if parent is not None else self.width())
        chip_width = 208
        columns = max(1, available_width // chip_width)
        columns = max(3, columns)
        return min(max(1, len(visible)), columns)

    def _relayout(self, columns=None):
        if self._relayout_pending:
            self._relayout_pending = False
        visible = [chip for chip in self.chips if chip.isVisible()]
        if not visible:
            visible = self.chips
        columns = self._column_count() if columns is None else columns
        if columns == self._columns and self.layout().count() == len(self.chips):
            return
        self._columns = columns
        self.setMaximumWidth(columns * 208)
        self.setMinimumWidth(min(max(1, len(visible)), 3) * 194)
        layout = self.layout()
        for chip in self.chips:
            layout.removeWidget(chip)
        for visible_index, chip in enumerate(visible):
            row = visible_index // columns
            col = visible_index % columns
            layout.addWidget(chip, row, col, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        for col in range(columns):
            layout.setColumnStretch(col, 0)
