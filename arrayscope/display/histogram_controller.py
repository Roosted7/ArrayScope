"""Histogram interaction helpers for ImageView2D."""

from __future__ import annotations

from math import ceil

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.icons import set_button_icon


MIN_HISTOGRAM_BIN_SCREEN_PX = 5
DEFAULT_HISTOGRAM_BIN_CAP = 500
MIN_LEVEL_SPAN_FRACTION = 1e-12


class HistogramLevelPreviewController(QtCore.QObject):
    def __init__(self, owner, *, interval_ms: int = 33):
        super().__init__(owner)
        self.owner = owner
        self.interval_ms = int(interval_ms)
        self.pending_levels = None
        self.timer = QtCore.QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.flush_preview)

    def schedule_from_widget(self) -> None:
        self.pending_levels = self._widget_levels()
        if self.pending_levels is None:
            return
        if not self.timer.isActive():
            self.timer.start(max(1, int(self.interval_ms)))

    def finish_from_widget(self) -> None:
        levels = self._widget_levels()
        if levels is not None:
            self.pending_levels = levels
        self.flush_preview()
        self.owner.userLevelsChanged.emit()

    def flush_preview(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        levels = self.pending_levels
        self.pending_levels = None
        if levels is None:
            return
        self.owner._apply_histogram_preview_levels(levels)

    def cancel(self) -> None:
        self.pending_levels = None
        if self.timer.isActive():
            self.timer.stop()

    def _widget_levels(self):
        try:
            levels = self.owner.histogram.getLevels()
        except Exception:
            return None
        if levels is None:
            return None
        low, high = levels
        return (float(low), float(high))


class HistogramDisplayController(QtCore.QObject):
    """Own adaptive histogram plotting and manual level editing."""

    def __init__(self, owner, *, min_bin_screen_px: int = MIN_HISTOGRAM_BIN_SCREEN_PX):
        super().__init__(owner)
        self.owner = owner
        self.min_bin_screen_px = max(1, int(min_bin_screen_px))
        self._refresh_pending = False
        self._manual_popup: HistogramLevelEditPopup | None = None
        self._manual_start_levels: tuple[float, float] | None = None
        self._manual_region_mouse_click_event = None
        self._manual_region_installed = False

    def install(self) -> None:
        item = self._histogram_item()
        if item is None:
            return
        vb = getattr(item, "vb", None)
        if vb is not None:
            vb.sigRangeChanged.connect(lambda *_args: self.schedule_refresh())
            scene = vb.scene()
            if scene is not None and hasattr(scene, "sigMouseClicked"):
                scene.sigMouseClicked.connect(self._on_scene_mouse_clicked)
        self._install_manual_clicks()

    def schedule_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QtCore.QTimer.singleShot(0, self._refresh_from_timer)

    def _refresh_from_timer(self) -> None:
        self._refresh_pending = False
        self.refresh_histogram_plot(auto_level=False)

    def refresh_histogram_plot(self, *, auto_level: bool = False) -> bool:
        item = self._histogram_item()
        if item is None or item.imageItem() is None:
            return False
        histogram = adaptive_histogram_for_view(
            item.imageItem(),
            item,
            histogram_bounds=self.owner.getHistogramDataBounds(),
            min_bin_screen_px=self.min_bin_screen_px,
        )
        if histogram is None:
            return False
        x, y = histogram
        if x is None or y is None:
            return False

        item.plot.setData(x, y)
        region = getattr(item, "region", None)
        if region is not None:
            if auto_level and len(x) > 0:
                levels = self.owner.getHistogramDataBounds() or _plot_bounds(x)
            else:
                levels = item.imageItem().getLevels()
            if levels is not None:
                if auto_level:
                    self.owner._apply_display_levels(float(levels[0]), float(levels[1]), emit_user=False)
                else:
                    with QtCore.QSignalBlocker(region):
                        region.setRegion((float(levels[0]), float(levels[1])))
        return True

    def begin_limit_edit(self, which: str, scene_pos=None) -> None:
        levels = self._owner_levels()
        if levels is None:
            return
        which = "lower" if str(which) == "lower" else "upper"
        value = levels[0] if which == "lower" else levels[1]
        self._show_popup(
            label="Low" if which == "lower" else "High",
            value=float(value),
            apply=lambda new_value, which=which: self._apply_limit(which, float(new_value)),
            scene_pos=scene_pos,
        )

    def begin_span_edit(self, scene_pos=None) -> None:
        levels = self._owner_levels()
        if levels is None:
            return
        low, high = levels
        span = high - low
        if span <= 0.0:
            return
        anchor = self._value_for_scene_pos(scene_pos)
        if anchor is None:
            anchor = (low + high) * 0.5
        fraction = (float(anchor) - low) / span
        fraction = max(0.0, min(1.0, float(fraction)))
        self._show_popup(
            label="Span",
            value=float(span),
            apply=lambda new_value, fraction=fraction: self._apply_span(float(new_value), fraction),
            scene_pos=scene_pos,
        )

    def active_popup(self):
        return self._manual_popup

    def close_popup(self) -> None:
        if self._manual_popup is not None:
            self._manual_popup.close()

    def _install_manual_clicks(self) -> None:
        item = self._histogram_item()
        region = None if item is None else getattr(item, "region", None)
        if region is None or self._manual_region_installed:
            return
        lines = list(getattr(region, "lines", ()) or ())
        for index, line in enumerate(lines[:2]):
            line.sigClicked.connect(
                lambda _line, event, index=index: self._on_limit_line_clicked(index, event)
            )
        self._manual_region_mouse_click_event = region.mouseClickEvent

        def mouse_click_event(event, _region=region):
            if (
                event.button() == QtCore.Qt.MouseButton.LeftButton
                and event.double()
                and not self._line_click_in_progress(event)
            ):
                event.accept()
                self.request_auto_window()
                return
            if (
                event.button() == QtCore.Qt.MouseButton.LeftButton
                and not event.double()
                and not self._line_click_in_progress(event)
            ):
                event.accept()
                self.begin_span_edit(event.scenePos())
                return
            self._manual_region_mouse_click_event(event)

        region.mouseClickEvent = mouse_click_event
        self._manual_region_installed = True

    def _on_scene_mouse_clicked(self, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not event.double():
            return
        if event.isAccepted():
            return
        item = self._histogram_item()
        vb = None if item is None else getattr(item, "vb", None)
        if vb is None:
            return
        try:
            if not vb.sceneBoundingRect().contains(event.scenePos()):
                return
        except Exception:
            return
        if self._line_click_in_progress(event):
            return
        event.accept()
        self.request_auto_window()

    def request_auto_window(self) -> None:
        self.close_popup()
        signal = getattr(self.owner, "autoWindowRequested", None)
        if signal is not None:
            signal.emit()

    def _on_limit_line_clicked(self, index: int, event) -> None:
        if event.button() != QtCore.Qt.MouseButton.LeftButton or event.double():
            return
        event.accept()
        self.begin_limit_edit("lower" if int(index) == 0 else "upper", event.scenePos())

    def _line_click_in_progress(self, event) -> bool:
        item = self._histogram_item()
        region = None if item is None else getattr(item, "region", None)
        if region is None:
            return False
        value = self._value_for_scene_pos(event.scenePos())
        if value is None:
            return False
        tolerance = self._value_click_tolerance()
        for line in list(getattr(region, "lines", ()) or ()):
            try:
                if abs(float(line.value()) - float(value)) <= tolerance:
                    return True
            except Exception:
                continue
        return False

    def _show_popup(self, *, label: str, value: float, apply, scene_pos=None) -> None:
        self.close_popup()
        levels = self._owner_levels()
        if levels is None:
            return
        self._manual_start_levels = levels
        step = self._step_for_levels(levels)
        popup = HistogramLevelEditPopup(
            self.owner,
            label=label,
            value=float(value),
            step=step,
            apply_callback=apply,
            accept_callback=self._accept_popup,
            reject_callback=self._reject_popup,
        )
        self._manual_popup = popup
        popup.destroyed.connect(lambda *_args, popup=popup: self._on_popup_destroyed(popup))
        popup.show_near(self._global_point_for_scene_pos(scene_pos))

    def _accept_popup(self) -> None:
        self._manual_start_levels = None
        self._manual_popup = None
        self.owner.userLevelsChanged.emit()

    def _reject_popup(self) -> None:
        levels = self._manual_start_levels
        self._manual_start_levels = None
        self._manual_popup = None
        if levels is not None:
            self.owner._apply_display_levels(levels[0], levels[1], emit_user=False)

    def _on_popup_destroyed(self, popup) -> None:
        if self._manual_popup is popup:
            self._manual_popup = None

    def _apply_limit(self, which: str, value: float) -> None:
        levels = self._owner_levels()
        if levels is None or not np.isfinite(value):
            return
        low, high = levels
        min_span = self._minimum_level_span()
        if which == "lower":
            low = min(float(value), high - min_span)
        else:
            high = max(float(value), low + min_span)
        self.owner._apply_display_levels(low, high, emit_user=False)

    def _apply_span(self, span: float, anchor_fraction: float) -> None:
        levels = self._owner_levels()
        if levels is None or not np.isfinite(span):
            return
        low, high = levels
        old_span = high - low
        if old_span <= 0.0:
            return
        span = max(float(span), self._minimum_level_span())
        anchor_value = low + max(0.0, min(1.0, float(anchor_fraction))) * old_span
        new_low = anchor_value - anchor_fraction * span
        new_high = new_low + span
        self.owner._apply_display_levels(new_low, new_high, emit_user=False)

    def _owner_levels(self) -> tuple[float, float] | None:
        try:
            low, high = self.owner.getLevels()
        except Exception:
            return None
        low = float(low)
        high = float(high)
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return None
        return (low, high)

    def _minimum_level_span(self) -> float:
        levels = self._owner_levels()
        data_bounds = self.owner.getHistogramDataBounds()
        spans = []
        if levels is not None:
            spans.append(abs(levels[1] - levels[0]))
        if data_bounds is not None:
            spans.append(abs(float(data_bounds[1]) - float(data_bounds[0])))
        baseline = max([span for span in spans if span > 0.0] or [1.0])
        return max(np.finfo(float).eps, baseline * MIN_LEVEL_SPAN_FRACTION)

    def _step_for_levels(self, levels: tuple[float, float]) -> float:
        span = max(abs(float(levels[1]) - float(levels[0])), self._minimum_level_span())
        return max(span / 100.0, self._minimum_level_span())

    def _value_click_tolerance(self) -> float:
        item = self._histogram_item()
        vb = None if item is None else getattr(item, "vb", None)
        if vb is not None:
            try:
                return max(abs(float(vb.viewPixelSize()[1])) * 4.0, self._minimum_level_span())
            except Exception:
                pass
        return self._minimum_level_span()

    def _value_for_scene_pos(self, scene_pos) -> float | None:
        if scene_pos is None:
            return None
        item = self._histogram_item()
        vb = None if item is None else getattr(item, "vb", None)
        if vb is None:
            return None
        try:
            value_point = vb.mapSceneToView(scene_pos)
            return float(value_point.y())
        except Exception:
            return None

    def _global_point_for_scene_pos(self, scene_pos):
        item = self._histogram_item()
        vb = None if item is None else getattr(item, "vb", None)
        widget = self.owner.histogram
        if scene_pos is not None and vb is not None:
            view_widget = vb.getViewWidget()
            if view_widget is not None:
                point = view_widget.mapFromScene(scene_pos)
                if hasattr(point, "toPoint"):
                    point = point.toPoint()
                return view_widget.mapToGlobal(point)
        return widget.mapToGlobal(widget.rect().center())

    def _histogram_item(self):
        return getattr(getattr(self.owner, "histogram", None), "item", None)


class HistogramNumberEdit(QtWidgets.QAbstractSpinBox):
    valueEdited = QtCore.Signal(float)

    def __init__(self, *, value: float, step: float, parent=None):
        super().__init__(parent)
        self._value = float(value)
        self._step = max(float(step), np.finfo(float).eps)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self.setKeyboardTracking(True)
        self.lineEdit().setValidator(_float_validator(self))
        self.lineEdit().textEdited.connect(self._text_edited)
        self.setValue(value, emit=False)

    def value(self) -> float:
        return float(self._value)

    def setValue(self, value: float, *, emit: bool = True) -> None:
        value = float(value)
        if not np.isfinite(value):
            return
        changed = value != self._value
        self._value = value
        self.lineEdit().setText(_format_level(value))
        if emit and changed:
            self.valueEdited.emit(self._value)

    def stepBy(self, steps: int) -> None:
        self.setValue(self._value + int(steps) * self._step)
        self.selectAll()

    def stepEnabled(self):
        return (
            QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepUpEnabled
            | QtWidgets.QAbstractSpinBox.StepEnabledFlag.StepDownEnabled
        )

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        steps = 1 if delta > 0 else -1
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
            steps *= 10
        self.stepBy(steps)
        event.accept()

    def _text_edited(self, text: str) -> None:
        try:
            value = float(str(text).strip())
        except ValueError:
            return
        if not np.isfinite(value):
            return
        self._value = value
        self.valueEdited.emit(value)


class HistogramLevelEditPopup(QtWidgets.QWidget):
    def __init__(
        self,
        parent,
        *,
        label: str,
        value: float,
        step: float,
        apply_callback,
        accept_callback,
        reject_callback,
    ):
        flags = QtCore.Qt.WindowType.Popup | QtCore.Qt.WindowType.FramelessWindowHint
        super().__init__(parent, flags)
        self._apply_callback = apply_callback
        self._accept_callback = accept_callback
        self._reject_callback = reject_callback
        self._closing_mode: str | None = None
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        frame = QtWidgets.QFrame(self)
        frame.setObjectName("HistogramLevelEditBubble")
        frame.setStyleSheet(
            "QFrame#HistogramLevelEditBubble {"
            "background: palette(window);"
            "border: 1px solid palette(mid);"
            "border-radius: 6px;"
            "}"
            "QLabel { font-size: 9pt; color: palette(windowText); }"
            "QAbstractSpinBox { font-size: 9pt; padding: 1px 2px; }"
        )
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(8, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(QtWidgets.QLabel(str(label)))
        self.edit = HistogramNumberEdit(value=value, step=step, parent=frame)
        self.edit.setFixedWidth(92)
        self.edit.valueEdited.connect(self._apply_value)
        layout.addWidget(self.edit)
        self.accept_button = QtWidgets.QToolButton(frame)
        set_button_icon(self.accept_button, "done", icon_size=16, tooltip="Apply", text_beside_icon=False)
        self.accept_button.setFixedSize(24, 22)
        self.accept_button.clicked.connect(self.accept)
        layout.addWidget(self.accept_button)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 8)
        outer.addWidget(frame)
        self._triangle_height = 8
        self.adjustSize()

    def show_near(self, global_point) -> None:
        self.adjustSize()
        x = int(global_point.x() - self.width() * 0.5)
        y = int(global_point.y() - self.height() - 6)
        screen = QtGui.QGuiApplication.screenAt(global_point)
        if screen is not None:
            rect = screen.availableGeometry()
            x = max(rect.left(), min(x, rect.right() - self.width()))
            y = max(rect.top(), min(y, rect.bottom() - self.height()))
        self.move(x, y)
        self.show()
        self.raise_()
        self.edit.setFocus(QtCore.Qt.FocusReason.PopupFocusReason)
        self.edit.selectAll()

    def accept(self) -> None:
        if self._closing_mode is None:
            self._apply_value(self.edit.value())
            self._closing_mode = "accept"
            self._accept_callback()
        self.close()

    def reject(self) -> None:
        if self._closing_mode is None:
            self._closing_mode = "reject"
            self._reject_callback()
        self.close()

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self.accept()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.reject()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self._closing_mode is None:
            self._closing_mode = "accept"
            self._accept_callback()
        super().closeEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        palette = self.palette()
        base = palette.color(QtGui.QPalette.ColorRole.Window)
        border = palette.color(QtGui.QPalette.ColorRole.Mid)
        center = self.width() * 0.5
        bottom = self.height() - 1
        path = QtGui.QPainterPath()
        path.moveTo(center - 8, bottom - self._triangle_height)
        path.lineTo(center + 8, bottom - self._triangle_height)
        path.lineTo(center, bottom)
        path.closeSubpath()
        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(QtGui.QBrush(base))
        painter.drawPath(path)

    def _apply_value(self, value: float) -> None:
        self._apply_callback(float(value))


def adaptive_histogram_for_view(
    image_item,
    histogram_item,
    *,
    histogram_bounds=None,
    min_bin_screen_px: int = MIN_HISTOGRAM_BIN_SCREEN_PX,
    bin_cap: int = DEFAULT_HISTOGRAM_BIN_CAP,
):
    data = getattr(image_item, "image", None)
    if data is None:
        return None
    data = np.asarray(data)
    if data.size == 0:
        return None
    sampled = _sample_histogram_data(data)
    sampled = sampled[np.isfinite(sampled)]
    if sampled.size == 0:
        return None

    bounds = _finite_increasing_pair(histogram_bounds)
    if bounds is None:
        low = float(np.nanmin(sampled))
        high = float(np.nanmax(sampled))
        if not np.isfinite(low) or not np.isfinite(high):
            return None
        if high == low:
            high = low + 1.0
        bounds = (low, high)
    low, high = bounds
    span = high - low
    if span <= 0.0:
        return None

    visible_span = _visible_value_span(histogram_item)
    if visible_span is None or visible_span <= 0.0:
        visible_span = span
    visible_span = max(min(float(visible_span), span), np.finfo(float).eps)

    pixel_height = _histogram_value_pixel_height(histogram_item)
    max_visible_bins = max(2, int(pixel_height / max(1, int(min_bin_screen_px))))
    visible_bin_width = visible_span / max_visible_bins
    requested_bins = max(2, int(ceil(span / max(visible_bin_width, np.finfo(float).eps))))
    cap = max(2, min(int(bin_cap), int(sampled.size)))
    bins = max(2, min(requested_bins, cap))
    counts, edges = np.histogram(sampled, bins=bins, range=(low, high))
    return edges[:-1], counts


def _sample_histogram_data(data: np.ndarray, *, target_image_size: int = 200) -> np.ndarray:
    if data.ndim < 2:
        return data.reshape(-1)
    step0 = max(1, int(ceil(data.shape[0] / float(target_image_size))))
    step1 = max(1, int(ceil(data.shape[1] / float(target_image_size))))
    return data[::step0, ::step1].reshape(-1)


def _visible_value_span(histogram_item) -> float | None:
    vb = getattr(histogram_item, "vb", None)
    if vb is None:
        return None
    try:
        if getattr(histogram_item, "orientation", "vertical") == "vertical":
            low, high = vb.viewRange()[1]
        else:
            low, high = vb.viewRange()[0]
        span = abs(float(high) - float(low))
    except Exception:
        return None
    if not np.isfinite(span) or span <= 0.0:
        return None
    return span


def _histogram_value_pixel_height(histogram_item) -> float:
    vb = getattr(histogram_item, "vb", None)
    if vb is None:
        return 200.0
    for getter in (getattr(vb, "screenGeometry", None), getattr(vb, "sceneBoundingRect", None)):
        if getter is None:
            continue
        try:
            rect = getter()
            if rect is None:
                continue
            value = rect.height() if getattr(histogram_item, "orientation", "vertical") == "vertical" else rect.width()
            if value and np.isfinite(float(value)) and float(value) > 1.0:
                return float(value)
        except Exception:
            continue
    return 200.0


def _finite_increasing_pair(values) -> tuple[float, float] | None:
    if values is None:
        return None
    try:
        low, high = values
        low = float(low)
        high = float(high)
    except Exception:
        return None
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return (low, high)


def _float_validator(parent):
    validator = QtGui.QDoubleValidator(parent)
    validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
    return validator


def _format_level(value: float) -> str:
    return f"{float(value):.8g}"


def _plot_bounds(x) -> tuple[float, float] | None:
    try:
        values = np.asarray(x, dtype=float)
    except Exception:
        return None
    if values.size < 1:
        return None
    low = float(values[0])
    high = float(values[-1])
    if values.size > 1:
        high += abs(float(values[-1]) - float(values[-2]))
    elif high == low:
        high = low + 1.0
    return _finite_increasing_pair((low, high))
