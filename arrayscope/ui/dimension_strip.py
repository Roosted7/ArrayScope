"""Compact dimension role strip widgets."""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.core.axis_info import axis_display_name, axis_metadata_summary
from arrayscope.core.slice_selection import (
    parse_slice_selection,
    selection_text_is_allowed,
    shift_slice_selection_text,
)
from arrayscope.ui.icons import set_button_icon


_QT_WIDGET_MAX_SIZE = 16_777_215


class _SliceSelectionValidator(QtGui.QValidator):
    def validate(self, text, pos):
        state = self.State.Acceptable if selection_text_is_allowed(text) else self.State.Invalid
        return state, text, pos


class SliceIndexEdit(QtWidgets.QAbstractSpinBox):
    stepRequested = Qt.QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.lineEdit().setValidator(_SliceSelectionValidator(self))

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
        self._button_icon_state: dict[QtWidgets.QAbstractButton, tuple[str, str | None]] = {}

    def update_state(self, shape, view_state, profile_axes=(), axes=None):
        size = int(shape[self.axis])
        self._axis_size = size
        axis_info = _axis_info_for(axes, shape, self.axis)
        _set_text_if_changed(self.axis_label, f"{_elide(axis_display_name(axis_info, self.axis))} [{size}]")
        _set_tooltip_if_changed(self.axis_label, "" if axis_info is None else axis_metadata_summary(axis_info))
        image_axes = view_state.image_axes or ()
        is_y = len(image_axes) > 0 and image_axes[0] == self.axis
        is_x = len(image_axes) > 1 and image_axes[1] == self.axis
        is_p = self.axis in tuple(profile_axes or ())
        is_m = getattr(view_state, "montage_axis", None) == self.axis
        _set_checked_if_changed(self.y_button, is_y)
        _set_checked_if_changed(self.x_button, is_x)
        _set_checked_if_changed(self.p_button, is_p)
        tiled_tooltip = "Use this range as an image-axis crop"
        y_tooltip = tiled_tooltip if is_m else ("Flip Y direction" if is_y else f"Use dim {self.axis} as image Y axis")
        x_tooltip = tiled_tooltip if is_m else ("Flip X direction" if is_x else f"Use dim {self.axis} as image X axis")
        self._set_button_icon_if_changed(
            self.y_button,
            "arrow_downward" if is_y and view_state.axis_flipped[self.axis] else "arrow_upward",
            tooltip=y_tooltip,
        )
        self._set_button_icon_if_changed(
            self.x_button,
            "arrow_forward" if is_x and view_state.axis_flipped[self.axis] else "arrow_back",
            tooltip=x_tooltip,
        )
        _set_tooltip_if_changed(self.p_button, f"Toggle dim {self.axis} as profile axis")
        is_singleton = size == 1
        can_use_as_image = not is_singleton and view_state.image_axes is not None
        _set_enabled_if_changed(self.y_button, can_use_as_image)
        _set_enabled_if_changed(self.x_button, can_use_as_image)
        _set_enabled_if_changed(self.p_button, not is_singleton)
        self.slice_edit.blockSignals(True)
        try:
            axis_text = None
            if getattr(view_state, "axis_range_text", None):
                axis_text = view_state.axis_range_text[self.axis]
            if axis_text is not None:
                self._set_slice_text_if_changed(str(axis_text))
            elif is_m and getattr(view_state, "montage_text", None):
                self._set_slice_text_if_changed(str(view_state.montage_text))
            elif self.axis in image_axes:
                self._set_slice_text_if_changed(":")
            else:
                self._set_slice_text_if_changed(str(view_state.slice_indices[self.axis]))
            _set_enabled_if_changed(self.slice_edit, not is_singleton)
            _set_visible_if_changed(self.slice_edit, True)
            _set_tooltip_if_changed(
                self.slice_edit,
                "Slice index or range. Python default: 0:100:2. "
                "MATLAB fallback: 0:2:100. Lists: 0 5 8. Repair: 0-100 -> 0:100.",
            )
        finally:
            self.slice_edit.blockSignals(False)

    def _set_button_icon_if_changed(self, button, name: str, *, tooltip: str | None = None) -> None:
        key = (str(name), tooltip)
        if self._button_icon_state.get(button) == key:
            return
        set_button_icon(button, name, tooltip=tooltip)
        self._button_icon_state[button] = key

    def _set_slice_text_if_changed(self, text: str) -> None:
        if self.slice_edit.text() != text:
            self.slice_edit.setText(text)

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
        try:
            shifted = _shift_slice_text(text, delta, self._axis_size)
        except ValueError:
            try:
                value = int(text)
            except ValueError:
                value = 0
            value = max(0, min(self._axis_size - 1, value + int(delta)))
            self.slice_edit.setText(str(value))
            self.sliceChanged.emit(self.axis, value)
            return
        selection = parse_slice_selection(shifted, self._axis_size)
        self.slice_edit.setText(shifted)
        if selection.kind == "scalar":
            self.sliceChanged.emit(self.axis, selection.indices[0])
        else:
            self.sliceTextChanged.emit(self.axis, shifted)

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

    def update_state(self, shape, view_state, profile_axes=(), axes=None):
        self.update_shape(shape)
        for axis, chip in enumerate(self.chips):
            if axis < len(shape):
                chip.update_state(shape, view_state, profile_axes, axes=axes)

    def update_axis_state(self, axis: int, shape, view_state, profile_axes=(), axes=None) -> None:
        self.chip(axis).update_state(shape, view_state, profile_axes, axes=axes)

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
        # Qt event-turn barrier. Resize events can arrive before the final
        # contents rect is stable; `_relayout_pending` guards the latest pass.
        QtCore.QTimer.singleShot(0, self, self._run_scheduled_relayout)

    def _run_scheduled_relayout(self):
        self._relayout_pending = False
        self._relayout()

    def _column_count(self):
        visible = [chip for chip in self.chips if chip.isVisible()]
        if not visible:
            visible = self.chips
        parent = self.parentWidget()
        parent_width = 0 if parent is None else parent.contentsRect().width()
        available_width = max(1, parent_width or self.contentsRect().width() or self.width())
        chip_width = 242
        columns = max(1, available_width // chip_width)
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
        self.setMaximumWidth(max(220, columns * 242))
        self.setMinimumWidth(min(max(1, columns), len(visible)) * 220)
        layout = self.layout()
        for chip in self.chips:
            layout.removeWidget(chip)
        for visible_index, chip in enumerate(visible):
            row = visible_index // columns
            col = visible_index % columns
            layout.addWidget(chip, row, col, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)
        for col in range(columns):
            layout.setColumnStretch(col, 0)
        layout.invalidate()
        self.updateGeometry()


def _shift_slice_text(text, delta, axis_size):
    return shift_slice_selection_text(text, delta, axis_size)


def _axis_info_for(axes, shape, axis):
    """Return the AxisInfo for `axis` when metadata aligns with the shape."""
    if axes is None or len(axes) != len(shape) or axis >= len(axes):
        return None
    return axes[axis]


def _elide(name: str, limit: int = 12) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _set_text_if_changed(widget, text: str) -> None:
    if widget.text() != text:
        widget.setText(text)


def _set_checked_if_changed(button, checked: bool) -> None:
    checked = bool(checked)
    if button.isChecked() != checked:
        button.setChecked(checked)


def _set_enabled_if_changed(widget, enabled: bool) -> None:
    enabled = bool(enabled)
    if widget.isEnabled() != enabled:
        widget.setEnabled(enabled)


def _set_visible_if_changed(widget, visible: bool) -> None:
    visible = bool(visible)
    if widget.isVisible() != visible:
        widget.setVisible(visible)


def _set_tooltip_if_changed(widget, tooltip: str) -> None:
    if widget.toolTip() != tooltip:
        widget.setToolTip(tooltip)
