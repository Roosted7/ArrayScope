"""Compact dimension role strip widgets."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.ui.icons import set_button_icon


class SliceIndexEdit(QtWidgets.QAbstractSpinBox):
    stepRequested = Qt.QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)

    def text(self):
        return self.lineEdit().text()

    def setText(self, text):
        self.lineEdit().setText(str(text))

    def setAlignment(self, alignment):
        self.lineEdit().setAlignment(alignment)

    def stepBy(self, steps):
        self.stepRequested.emit(int(steps))

    def stepEnabled(self):
        return (
            QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepUpEnabled
            | QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
        )

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        step = 1 if delta > 0 else -1
        if event.modifiers() & Qt.QtCore.Qt.KeyboardModifier.ShiftModifier:
            step *= 10
        self.stepRequested.emit(step)
        event.accept()


class DimensionChip(QtWidgets.QFrame):
    roleChanged = Qt.QtCore.Signal(str, int)
    sliceChanged = Qt.QtCore.Signal(int, int)
    sliceTextChanged = Qt.QtCore.Signal(int, str)
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
        self.p_button = QtWidgets.QToolButton(checkable=True)
        for role, button in (("y", self.y_button), ("x", self.x_button), ("p", self.p_button)):
            button.setFixedSize(24, 22)
            button.setText(role.upper())
            button.clicked.connect(lambda _checked=False, role=role: self.roleChanged.emit(role, self.axis))
            layout.addWidget(button)

        self._axis_size = 1
        self.slice_edit = SliceIndexEdit()
        self.slice_edit.setFixedWidth(68)
        self.slice_edit.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        self.slice_edit.editingFinished.connect(self._slice_edit_finished)
        self.slice_edit.stepRequested.connect(self._slice_edit_stepped)
        layout.addWidget(self.slice_edit)

        self.ops_button = QtWidgets.QToolButton()
        set_button_icon(self.ops_button, "add", tooltip="Add operation on this dimension")
        self.ops_button.clicked.connect(lambda: self.operationRequested.emit(self.axis))
        layout.addWidget(self.ops_button)
        self.setLayout(layout)
        self.setMinimumWidth(220)
        self.setMaximumWidth(238)

    def update_state(self, shape, view_state, profile_axes=()):
        size = int(shape[self.axis])
        self._axis_size = size
        self.axis_label.setText(f"{self.axis} [{size}]")
        image_axes = view_state.image_axes or ()
        is_y = len(image_axes) > 0 and image_axes[0] == self.axis
        is_x = len(image_axes) > 1 and image_axes[1] == self.axis
        is_p = self.axis in tuple(profile_axes or ())
        is_m = getattr(view_state, "montage_axis", None) == self.axis
        self.y_button.setChecked(is_y)
        self.x_button.setChecked(is_x)
        self.p_button.setChecked(is_p)
        set_button_icon(self.y_button, "arrow_downward" if is_y and view_state.axis_flipped[self.axis] else "arrow_upward")
        set_button_icon(self.x_button, "arrow_forward" if is_x and view_state.axis_flipped[self.axis] else "arrow_back")
        tiled_tooltip = "Tiled dimension cannot also be image X/Y. Clear the range first."
        self.y_button.setToolTip(tiled_tooltip if is_m else ("Flip Y direction" if is_y else f"Use dim {self.axis} as image Y axis"))
        self.x_button.setToolTip(tiled_tooltip if is_m else ("Flip X direction" if is_x else f"Use dim {self.axis} as image X axis"))
        self.p_button.setToolTip(f"Toggle dim {self.axis} as profile axis")
        is_display_axis = self.axis in image_axes or is_m
        is_singleton = size == 1
        can_use_as_image = not is_singleton and not is_m and view_state.image_axes is not None
        self.y_button.setEnabled(can_use_as_image)
        self.x_button.setEnabled(can_use_as_image)
        self.p_button.setEnabled(not is_singleton)
        self.slice_edit.blockSignals(True)
        try:
            axis_text = None
            if getattr(view_state, "axis_range_text", None):
                axis_text = view_state.axis_range_text[self.axis]
            if axis_text is not None:
                self.slice_edit.setText(str(axis_text))
            elif is_m and getattr(view_state, "montage_text", None):
                self.slice_edit.setText(str(view_state.montage_text))
            elif self.axis in image_axes:
                self.slice_edit.setText(":")
            else:
                self.slice_edit.setText(str(view_state.slice_indices[self.axis]))
            self.slice_edit.setEnabled(not is_singleton)
            self.slice_edit.setVisible(True)
            self.slice_edit.setToolTip("Slice index or range, e.g. ':' or '0:2:100'")
        finally:
            self.slice_edit.blockSignals(False)

    def _slice_edit_finished(self):
        text = self.slice_edit.text().strip()
        if ":" in text:
            self.sliceTextChanged.emit(self.axis, text)
            return
        try:
            self.sliceChanged.emit(self.axis, int(text))
        except ValueError:
            self.sliceTextChanged.emit(self.axis, text)

    def _slice_edit_stepped(self, delta):
        text = self.slice_edit.text().strip()
        if ":" in text:
            shifted = _shift_slice_text(text, delta, self._axis_size)
            self.slice_edit.setText(shifted)
            self.sliceTextChanged.emit(self.axis, shifted)
            return
        try:
            value = int(text)
        except ValueError:
            value = 0
        value = max(0, min(self._axis_size - 1, value + int(delta)))
        self.slice_edit.setText(str(value))
        self.sliceChanged.emit(self.axis, value)

    def focusInEvent(self, event):
        self.focused.emit(self.axis)
        super().focusInEvent(event)


class DimensionStrip(QtWidgets.QWidget):
    roleChanged = Qt.QtCore.Signal(str, int)
    sliceChanged = Qt.QtCore.Signal(int, int)
    sliceTextChanged = Qt.QtCore.Signal(int, str)
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
            chip.sliceTextChanged.connect(self.sliceTextChanged)
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
        chip_width = 242
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
        self.setMaximumWidth(columns * 242)
        self.setMinimumWidth(min(max(1, len(visible)), 3) * 220)
        layout = self.layout()
        for chip in self.chips:
            layout.removeWidget(chip)
        for visible_index, chip in enumerate(visible):
            row = visible_index // columns
            col = visible_index % columns
            layout.addWidget(chip, row, col, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        for col in range(columns):
            layout.setColumnStretch(col, 0)


def _shift_slice_text(text, delta, axis_size):
    parts = str(text).split(":")
    while len(parts) < 3:
        parts.append("")
    start_text, step_text, stop_text = parts[:3]
    try:
        step = int(step_text) if step_text else 1
    except ValueError:
        step = 1
    shift = int(delta) * (abs(step) if step != 0 else 1)

    def shifted(value_text):
        if value_text == "":
            return ""
        try:
            return str(max(0, min(int(axis_size), int(value_text) + shift)))
        except ValueError:
            return value_text

    start = shifted(start_text)
    stop = shifted(stop_text)
    if step_text:
        return f"{start}:{step_text}:{stop}"
    return f"{start}:{stop}"
