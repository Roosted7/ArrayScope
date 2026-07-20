"""Compact dimension role strip widgets."""

from __future__ import annotations

import contextlib

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


def _make_chip_separator(parent) -> QtWidgets.QFrame:
    separator = QtWidgets.QFrame(parent)
    separator.setObjectName("DimChipSeparator")
    separator.setFixedWidth(1)
    return separator


class SliceIndexEdit(QtWidgets.QAbstractSpinBox):
    stepRequested = Qt.QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Native-style arrows render cleanly in every theme (the PlusMinus
        # glyphs looked rough, especially on light palettes).
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.UpDownArrows)
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
        self.axis = int(axis)
        self.setObjectName(f"DimensionChip{axis}")
        self.setProperty("dimensionChip", True)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFocusPolicy(Qt.QtCore.Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda _pos: self.operationRequested.emit(self.axis)
        )

        layout = QtWidgets.QHBoxLayout()
        # The index badge sits flush against the chip's left edge.
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)

        # The badge doubles as the profile toggle (replaces the old P button).
        self.index_badge = QtWidgets.QToolButton()
        self.index_badge.setObjectName("DimChipBadge")
        self.index_badge.setCheckable(True)
        self.index_badge.setText(str(self.axis))
        self.index_badge.clicked.connect(
            lambda _checked=False: self.roleChanged.emit("p", self.axis)
        )
        layout.addWidget(self.index_badge)

        metrics = self.fontMetrics()
        char = max(6, metrics.averageCharWidth())

        self.axis_label = QtWidgets.QLabel()
        self.axis_label.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        # Size text: 5 characters by default, may shrink to 3 under pressure.
        self.axis_label.setMinimumWidth(3 * char)
        self.axis_label.setMaximumWidth(5 * char + 4)
        self.axis_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.axis_label)
        layout.addWidget(_make_chip_separator(self))

        self.y_button = QtWidgets.QToolButton(checkable=True)
        self.x_button = QtWidgets.QToolButton(checkable=True)
        for role, button in (("y", self.y_button), ("x", self.x_button)):
            button.setFixedSize(24, 22)
            button.setText(role.upper())
            button.clicked.connect(
                lambda _checked=False, role=role: self.roleChanged.emit(role, self.axis)
            )
            layout.addWidget(button)
        layout.addWidget(_make_chip_separator(self))

        self._axis_size = 1
        self.slice_edit = SliceIndexEdit()
        # Defaults to fitting "100:2:200"; shrinks to ~3.5 characters before
        # anything else in the chip gives way.
        stepper_allowance = 24
        self.slice_edit.setMinimumWidth(int(3.5 * char) + stepper_allowance)
        self._slice_edit_preferred = metrics.horizontalAdvance("100:2:200") + stepper_allowance + 10
        self.slice_edit.setMaximumWidth(max(self._slice_edit_preferred, 96) + 40)
        self.slice_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.slice_edit.setAlignment(Qt.QtCore.Qt.AlignmentFlag.AlignCenter)
        self.slice_edit.editingFinished.connect(self._slice_edit_finished)
        self.slice_edit.stepRequested.connect(self._slice_edit_stepped)
        layout.addWidget(self.slice_edit, 1)

        self.ops_button = QtWidgets.QToolButton()
        set_button_icon(self.ops_button, "add", tooltip="Add operation on this dimension")
        self.ops_button.clicked.connect(lambda: self.operationRequested.emit(self.axis))
        layout.addWidget(self.ops_button)
        self.setLayout(layout)
        self._button_icon_state: dict[QtWidgets.QAbstractButton, tuple[str, str | None]] = {}
        self._profile_available = True

    def update_state(self, shape, view_state, profile_axes=(), axes=None):
        size = int(shape[self.axis])
        self._axis_size = size
        axis_info = _axis_info_for(axes, shape, self.axis)
        display_name = _elide(axis_display_name(axis_info, self.axis))
        # The chip badge already shows the dimension index; the label carries
        # the size (plus the metadata name when one exists).
        label_text = str(size) if display_name == str(self.axis) else f"{display_name} · {size}"
        _set_text_if_changed(self.axis_label, label_text)
        _set_tooltip_if_changed(
            self.axis_label,
            f"{size} elements" if axis_info is None else axis_metadata_summary(axis_info),
        )
        image_axes = view_state.image_axes or ()
        is_y = len(image_axes) > 0 and image_axes[0] == self.axis
        is_x = len(image_axes) > 1 and image_axes[1] == self.axis
        is_p = self.axis in tuple(profile_axes or ())
        is_m = getattr(view_state, "montage_axis", None) == self.axis
        _set_checked_if_changed(self.y_button, is_y)
        _set_checked_if_changed(self.x_button, is_x)
        profile_available = bool(getattr(self, "_profile_available", True))
        _set_checked_if_changed(self.index_badge, is_p and profile_available)
        if profile_available:
            badge_tooltip = f"Dimension {self.axis} — click to plot a profile along it"
        else:
            badge_tooltip = (
                f"Dimension {self.axis} — click to open the profile dock and plot along it"
            )
        _set_tooltip_if_changed(self.index_badge, badge_tooltip)
        tiled_tooltip = "Use this range as an image-axis crop"
        y_tooltip = (
            tiled_tooltip
            if is_m
            else ("Flip Y direction" if is_y else f"Show dim {self.axis} on the image Y axis")
        )
        x_tooltip = (
            tiled_tooltip
            if is_m
            else ("Flip X direction" if is_x else f"Show dim {self.axis} on the image X axis")
        )
        flipped = bool(view_state.axis_flipped[self.axis])
        _set_text_if_changed(self.y_button, ("Y↓" if flipped else "Y↑") if is_y else "Y")
        _set_tooltip_if_changed(self.y_button, y_tooltip)
        _set_text_if_changed(self.x_button, ("X→" if flipped else "X←") if is_x else "X")
        _set_tooltip_if_changed(self.x_button, x_tooltip)
        self._set_montage_state_if_changed(is_m)
        is_singleton = size == 1
        can_use_as_image = not is_singleton and view_state.image_axes is not None
        _set_enabled_if_changed(self.y_button, can_use_as_image)
        _set_enabled_if_changed(self.x_button, can_use_as_image)
        _set_enabled_if_changed(self.index_badge, not is_singleton)
        self.slice_edit.blockSignals(True)
        try:
            axis_text = None
            if getattr(view_state, "axis_range_text", None):
                axis_text = view_state.axis_range_text[self.axis]
            if axis_text is not None:
                self._set_slice_text_if_changed(str(axis_text))
            elif is_m and getattr(view_state, "montage_text", None):
                self._set_slice_text_if_changed(str(view_state.montage_text))
            elif is_m:
                # Canonical full coverage: montage_indices/montage_text
                # normalize to None when every index is selected, and the
                # honest spelling of "all montage indices" is ":" — not the
                # scalar slice index the axis would have outside montage mode.
                self._set_slice_text_if_changed(":")
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

    def _set_montage_state_if_changed(self, is_montage: bool) -> None:
        is_montage = bool(is_montage)
        if self.property("montageAxis") == is_montage:
            return
        self.setProperty("montageAxis", is_montage)
        style = self.style()
        style.unpolish(self)
        style.polish(self)

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
    layoutChanged = Qt.QtCore.Signal()

    def __init__(self, ndim, parent=None):
        super().__init__(parent)
        self.chips = []
        self._columns = 0
        self._relayout_pending = False
        self._profile_available = True
        self._badge_width_count = None
        self._chip_geometry = None
        self._watched_parent = None
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
        self._sync_badge_widths(tuple(range(int(ndim))))
        self._relayout()
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum
        )

    def set_profile_available(self, available: bool) -> None:
        """Profile toggles highlight only while the profile dock is visible."""
        available = bool(available)
        if getattr(self, "_profile_available", None) == available:
            return
        self._profile_available = available
        for chip in self.chips:
            chip._profile_available = available

    def _sync_badge_widths(self, shape) -> None:
        # Every badge shares one width, sized to the largest visible index.
        visible_count = len(shape)
        if visible_count == getattr(self, "_badge_width_count", None):
            return
        self._badge_width_count = visible_count
        widest = str(max(0, visible_count - 1))
        metrics = self.fontMetrics()
        width = metrics.horizontalAdvance(widest) + 16
        for chip in self.chips:
            chip.index_badge.setFixedWidth(width)

    def update_shape(self, shape):
        for axis, chip in enumerate(self.chips):
            chip.setVisible(axis < len(shape))
        self._sync_badge_widths(shape)
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

    def row_metrics(self):
        """(row_count, row_height, vertical_spacing) for the current width.

        Columns are recomputed fresh so callers get correct heights even
        before a deferred relayout lands.
        """
        visible = [chip for chip in self.chips if chip.isVisible()] or self.chips
        columns = max(1, self._column_count())
        rows = (len(visible) + columns - 1) // columns
        row_height = max(28, self.chips[0].sizeHint().height() if self.chips else 28)
        return rows, row_height, self.layout().verticalSpacing()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = self._column_count()
        if columns != self._columns:
            self._schedule_relayout()

    def event(self, event):
        if event.type() == QtCore.QEvent.Type.ParentChange:
            self._watch_parent_resizes()
        return super().event(event)

    def _watch_parent_resizes(self):
        # Column invalidation must come from the parent's resize stream: the
        # strip's own maximumWidth is sized to the current grid, so after a
        # transient narrowing (dock transition, scrollbar flicker) the parent
        # can re-widen without the strip receiving a resizeEvent, leaving the
        # chips wrapped onto an extra row that no longer matches the width.
        parent = self.parentWidget()
        previous = getattr(self, "_watched_parent", None)
        if parent is previous:
            return
        if previous is not None:
            with contextlib.suppress(RuntimeError):
                previous.removeEventFilter(self)
        self._watched_parent = parent
        if parent is not None:
            parent.installEventFilter(self)

    def eventFilter(self, obj, event):
        if (
            obj is getattr(self, "_watched_parent", None)
            and event.type() == QtCore.QEvent.Type.Resize
        ):
            self._schedule_relayout()
        return super().eventFilter(obj, event)

    def _schedule_relayout(self):
        if self._column_count() != self._columns:
            # Row-count changes move the surrounding chrome height, which
            # viewport-continuity measures synchronously — do not defer.
            self._run_scheduled_relayout()
            return
        if self._relayout_pending:
            return
        self._relayout_pending = True
        # Timer category: UI cosmetic. Qt event-turn barrier. Resize events can arrive before the final
        # contents rect is stable; `_relayout_pending` guards the latest pass.
        QtCore.QTimer.singleShot(0, self, self._run_scheduled_relayout)

    def _run_scheduled_relayout(self):
        self._relayout_pending = False
        self._relayout()
        # Height consumers re-sync even when the grid itself was unchanged
        # (the surrounding scroll area may have been sized from stale rows).
        self.layoutChanged.emit()

    # Chips stretch between these bounds. Width policy: give every chip the
    # same width. Under pressure the inter-chip spacing shrinks first
    # (PREFERRED -> MIN), then the chips themselves (the slice input absorbs
    # the change), and only then rows wrap. Chips never squash below
    # MIN_CHIP_WIDTH, and the last, partially filled row keeps the same chip
    # width and spacing as full rows.
    MIN_CHIP_WIDTH = 214
    MAX_CHIP_WIDTH = 248
    # Hard floor: chips never sit closer than this (the old preferred gap).
    MIN_CHIP_SPACING = 12
    PREFERRED_CHIP_SPACING = 16

    # The dims row also holds the right-aligned sync-link button; reserve its
    # footprint (button + fixed gaps) so the last column never clips into it.
    SIBLING_RESERVE = 56

    def _available_width(self):
        parent = self.parentWidget()
        if parent is not None and parent.contentsRect().width() > 0:
            return max(1, parent.contentsRect().width() - self.SIBLING_RESERVE)
        return max(1, self.contentsRect().width() or self.width())

    def _column_count(self):
        visible = [chip for chip in self.chips if chip.isVisible()]
        if not visible:
            visible = self.chips
        available = self._available_width()
        columns = max(
            1,
            (available + self.MIN_CHIP_SPACING) // (self.MIN_CHIP_WIDTH + self.MIN_CHIP_SPACING),
        )
        return min(max(1, len(visible)), columns)

    def _geometry_for(self, columns):
        """Pick (chip_width, spacing): spacing shrinks before chips do."""
        available = self._available_width()
        gaps = max(0, columns - 1)
        if columns * self.MAX_CHIP_WIDTH + gaps * self.PREFERRED_CHIP_SPACING <= available:
            return self.MAX_CHIP_WIDTH, self.PREFERRED_CHIP_SPACING
        if columns * self.MAX_CHIP_WIDTH + gaps * self.MIN_CHIP_SPACING <= available:
            spacing = self.PREFERRED_CHIP_SPACING
            if gaps:
                spacing = (available - columns * self.MAX_CHIP_WIDTH) // gaps
                spacing = max(self.MIN_CHIP_SPACING, min(self.PREFERRED_CHIP_SPACING, spacing))
            return self.MAX_CHIP_WIDTH, spacing
        share = (available - gaps * self.MIN_CHIP_SPACING) // max(1, columns)
        width = int(max(self.MIN_CHIP_WIDTH, min(self.MAX_CHIP_WIDTH, share)))
        return width, self.MIN_CHIP_SPACING

    def _relayout(self, columns=None):
        if self._relayout_pending:
            self._relayout_pending = False
        visible = [chip for chip in self.chips if chip.isVisible()]
        if not visible:
            visible = self.chips
        columns = self._column_count() if columns is None else columns
        chip_width, spacing = self._geometry_for(columns)
        if (
            columns == self._columns
            and (chip_width, spacing) == getattr(self, "_chip_geometry", None)
            and self.layout().count() == len(self.chips)
        ):
            return
        self._columns = columns
        self._chip_geometry = (chip_width, spacing)
        for chip in self.chips:
            chip.setFixedWidth(chip_width)
        layout = self.layout()
        layout.setHorizontalSpacing(spacing)
        row_width = columns * chip_width + spacing * (columns - 1)
        self.setMaximumWidth(max(self.MIN_CHIP_WIDTH, row_width))
        self.setMinimumWidth(min(max(1, columns), len(visible)) * self.MIN_CHIP_WIDTH)
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
        self.layoutChanged.emit()


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
