"""Histogram interaction helpers for ImageView2D."""

from __future__ import annotations

import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from time import perf_counter

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.display.histogram_plot import (
    DEFAULT_HISTOGRAM_BIN_CAP,
    MIN_HISTOGRAM_BIN_SCREEN_PX,
    HistogramPlotRequest,
    HistogramPlotResult,
    compute_histogram_plot,
    finite_increasing_pair,
    sample_histogram_data,
)
from arrayscope.ui.icons import material_icon, set_button_icon

MIN_LEVEL_SPAN_FRACTION = 1e-12
ASYNC_HISTOGRAM_SOURCE_SIZE = 256 * 256


_HISTOGRAM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arrayscope-histogram")


class HistogramLevelPreviewController(QtCore.QObject):
    def __init__(self, owner, *, interval_ms: int = 33):
        super().__init__(owner)
        self.owner = owner
        self.interval_ms = int(interval_ms)
        self.pending_levels = None
        self.last_applied_levels = None
        # Timer category: UI cosmetic. Bounded preview coalescer. Manual level drags can fire faster than
        # the backend can redraw; final release still flushes immediately.
        self.timer = QtCore.QTimer(owner)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.flush_preview)

    def schedule_from_widget(self) -> None:
        self.pending_levels = self._widget_levels()
        if self.pending_levels is None:
            return
        immediate = getattr(self.owner, "_histogram_preview_immediate", None)
        if callable(immediate) and bool(immediate()):
            self.flush_preview(final=False)
            return
        if not self.timer.isActive():
            self.timer.start(max(1, int(self.interval_ms)))

    def finish_from_widget(self) -> None:
        levels = self._widget_levels()
        if levels is not None:
            self.pending_levels = levels
        if levels is not None and levels == self.last_applied_levels:
            if self.timer.isActive():
                self.timer.stop()
            self.pending_levels = None
            finalize = getattr(self.owner, "_finish_histogram_preview_levels", None)
            if callable(finalize):
                finalize(levels)
        else:
            self.flush_preview(final=True)
        self.owner.userLevelsChanged.emit()

    def flush_preview(self, *, final: bool = False) -> None:
        if self.timer.isActive():
            self.timer.stop()
        levels = self.pending_levels
        self.pending_levels = None
        if levels is None:
            return
        self.owner._apply_histogram_preview_levels(levels, final=bool(final))
        self.last_applied_levels = levels

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

    _histogram_ready = QtCore.Signal(object)

    def __init__(self, owner, *, min_bin_screen_px: int = MIN_HISTOGRAM_BIN_SCREEN_PX):
        super().__init__(owner)
        self.owner = owner
        self.min_bin_screen_px = max(1, int(min_bin_screen_px))
        self._refresh_pending = False
        # Timer category: UI cosmetic. Qt event-turn barrier. PyQtGraph emits multiple range/image signals
        # during one interaction; this collapses them to one latest refresh.
        self._refresh_timer = QtCore.QTimer(owner)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._refresh_from_timer)
        self._generation = 0
        self._closed = False
        self._active_future = None
        self._active_request_signature = None
        self._running_request_signature = None
        self._pending_request: HistogramPlotRequest | None = None
        self._result_cache = OrderedDict()
        self._manual_popup: HistogramLevelEditPopup | None = None
        self._manual_start_levels: tuple[float, float] | None = None
        self._pending_span_edit_scene_pos: QtCore.QPointF | None = None
        # Timer category: UI cosmetic. User-input timeout matching Qt's double-click interval. It prevents
        # a single-click edit from stealing a double-click auto-window gesture.
        self._pending_span_edit_timer = QtCore.QTimer(owner)
        self._pending_span_edit_timer.setSingleShot(True)
        self._pending_span_edit_timer.timeout.connect(self._flush_pending_span_edit)
        self._manual_region_mouse_click_event = None
        self._filtered_histogram_widgets = []
        self._last_histogram_release: tuple[float, QtCore.QPointF] | None = None
        self._last_histogram_auto_reset: tuple[float, QtCore.QPointF] | None = None
        self._manual_region_installed = False
        self._histogram_ready.connect(self._handle_histogram_ready)

    def install(self) -> None:
        item = self._histogram_item()
        if item is None:
            return
        widget = getattr(self.owner, "histogram", None)
        self._install_histogram_event_filter(getattr(widget, "viewport", lambda: None)())
        vb = getattr(item, "vb", None)
        if vb is not None:
            vb.sigRangeChanged.connect(lambda *_args: self._on_histogram_view_range_changed())
            scene = vb.scene()
            if scene is not None:
                for view in tuple(scene.views() or ()):
                    self._install_histogram_event_filter(getattr(view, "viewport", lambda: None)())
        self._install_manual_clicks()

    def _on_histogram_view_range_changed(self) -> None:
        # A range change that did NOT originate from a programmatic
        # presentation apply is the user zooming/panning the value axis; latch
        # it so index-driven refreshes stop resetting their view.
        if not getattr(self.owner, "_applying_presentation", False):
            self.owner._user_histogram_view_dirty = True
        self.schedule_refresh()

    def eventFilter(self, obj, event):
        if obj in self._filtered_histogram_widgets and self._handle_native_histogram_double_click(
            obj, event
        ):
            return True
        return super().eventFilter(obj, event)

    def _install_histogram_event_filter(self, widget) -> None:
        if widget is None or widget in self._filtered_histogram_widgets:
            return
        widget.installEventFilter(self)
        self._filtered_histogram_widgets.append(widget)

    def schedule_refresh(self) -> None:
        if self._refresh_pending:
            return
        self._refresh_pending = True
        self._refresh_timer.start(0)

    def _refresh_from_timer(self) -> None:
        if not self._refresh_pending:
            return
        self._refresh_pending = False
        self.refresh_histogram_plot(auto_level=False)

    def cancel(self) -> None:
        """Cancel queued refresh work before the owning widget is destroyed."""

        self._closed = True
        self._generation += 1
        self._refresh_pending = False
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
        self._cancel_pending_span_edit()
        self._active_request_signature = None
        self._running_request_signature = None
        self._pending_request = None
        self._result_cache.clear()
        self.close_popup()

    def refresh_histogram_plot(self, *, auto_level: bool = False) -> bool:
        item = self._histogram_item()
        if item is None or item.imageItem() is None:
            return False
        request = histogram_plot_request_for_view(
            item.imageItem(),
            item,
            histogram_bounds=self.owner.getHistogramDataBounds(),
            min_bin_screen_px=self.min_bin_screen_px,
            generation=self._generation + 1,
        )
        if request is None:
            return False
        self._generation = int(request.generation)
        request_signature = _request_signature(request)
        if auto_level:
            auto_bounds = _finite_increasing_pair(self.owner.getHistogramDataBounds())
            if auto_bounds is not None:
                self.owner._apply_display_levels(
                    float(auto_bounds[0]), float(auto_bounds[1]), emit_user=False
                )
            if np.asarray(request.data).size > ASYNC_HISTOGRAM_SOURCE_SIZE:
                self._schedule_histogram_job(request)
                return True
            self._active_request_signature = request_signature
            result = compute_histogram_plot(request)
            return self._apply_histogram_result(result, auto_level=False)
        if np.asarray(request.data).size <= ASYNC_HISTOGRAM_SOURCE_SIZE:
            self._active_request_signature = request_signature
            result = compute_histogram_plot(request)
            return self._apply_histogram_result(result, auto_level=auto_level)
        self._schedule_histogram_job(request)
        return True

    def _schedule_histogram_job(self, request: HistogramPlotRequest) -> None:
        self._closed = False
        signature = _request_signature(request)
        self._active_request_signature = signature
        cached = self._cached_histogram_result(request)
        if cached is not None:
            self._histogram_ready.emit(cached)
            return
        if self._active_request_signature == signature:
            running_signature = self._running_request_signature
            if running_signature == signature:
                active_future = self._active_future
                if active_future is None or not active_future.done():
                    return
                self._running_request_signature = None
                self._active_future = None
        if self._running_request_signature is not None:
            self._pending_request = request
            previous_future = self._active_future
            if (
                previous_future is not None
                and not previous_future.done()
                and previous_future.cancel()
            ):
                self._running_request_signature = None
                self._active_future = None
            else:
                return
        previous_future = self._active_future
        if previous_future is not None and not previous_future.done():
            previous_future.cancel()
        self._pending_request = None
        self._running_request_signature = signature
        submit = getattr(self.owner, "_submit_background_task", None)
        if callable(submit):
            started = submit(
                lambda request=request: compute_histogram_plot(request),
                on_done=self._histogram_ready.emit,
                key=("histogram_plot", signature),
            )
            if getattr(started, "scheduled", False):
                self._active_future = None
                return
            if (
                str(getattr(started, "reason", "")) in {"limited", "idle", "cost"}
                and not self._closed
            ):
                self._running_request_signature = None
                self._refresh_pending = True
                if not self._refresh_timer.isActive():
                    self._refresh_timer.start(
                        max(8, int(getattr(self.owner, "_histogram_retry_interval_ms", 33)))
                    )
                return
            # Terminal decline (owner closed for shutdown, or no controller
            # behind the submitter): drop the refresh. The module executor
            # below is non-daemon and joined at interpreter exit, so work
            # rescheduled there would outlive kernel shutdown.
            self._running_request_signature = None
            return
        self_ref = weakref.ref(self)

        def done(future):
            controller = self_ref()
            if controller is None:
                return
            try:
                result = future.result()
            except Exception:
                result = HistogramPlotResult(
                    generation=int(request.generation),
                    source_identity=request.source_identity,
                    view_signature=request.view_signature,
                    x=None,
                    y=None,
                    cancelled=True,
                )
            controller._histogram_ready.emit(result)

        future = _HISTOGRAM_EXECUTOR.submit(compute_histogram_plot, request)
        self._active_future = future
        future.add_done_callback(done)

    def _handle_histogram_ready(self, result) -> None:
        try:
            self._remember_histogram_result(result)
            self._apply_histogram_result(result, auto_level=False)
        finally:
            self._finish_histogram_job(result)

    def _remember_histogram_result(self, result) -> None:
        if not isinstance(result, HistogramPlotResult) or not result.has_data:
            return
        signature = _result_signature(result)
        self._result_cache[signature] = result
        self._result_cache.move_to_end(signature)
        while len(self._result_cache) > 32:
            self._result_cache.popitem(last=False)

    def _cached_histogram_result(self, request: HistogramPlotRequest) -> HistogramPlotResult | None:
        signature = _request_signature(request)
        result = self._result_cache.get(signature)
        if result is None:
            return None
        self._result_cache.move_to_end(signature)
        return replace(result, generation=int(request.generation))

    def _finish_histogram_job(self, result) -> None:
        if not isinstance(result, HistogramPlotResult):
            return
        result_signature = _result_signature(result)
        if self._running_request_signature == result_signature:
            self._running_request_signature = None
            self._active_future = None
        if self._closed:
            self._pending_request = None
            return
        pending = self._pending_request
        self._pending_request = None
        if pending is not None and _request_signature(pending) != self._active_request_signature:
            self._active_request_signature = _request_signature(pending)
        if pending is not None:
            self._schedule_histogram_job(pending)

    def _apply_histogram_result(self, result, *, auto_level: bool = False) -> bool:
        if self._closed or not isinstance(result, HistogramPlotResult):
            return False
        if int(result.generation) != int(self._generation):
            return False
        result_signature = _result_signature(result)
        if (
            result_signature != self._active_request_signature
            and self._active_request_signature is not None
        ):
            return False
        item = self._histogram_item()
        if item is None or item.imageItem() is None:
            return False
        current = histogram_plot_request_for_view(
            item.imageItem(),
            item,
            histogram_bounds=self.owner.getHistogramDataBounds(),
            min_bin_screen_px=self.min_bin_screen_px,
            generation=self._generation,
        )
        if current is None or _request_signature(current) != result_signature:
            return False
        if not result.has_data:
            return False
        x = result.x
        y = result.y
        budget_getter = getattr(self.owner, "_gui_callback_budget", None)
        budget = (
            budget_getter("histogram_refresh", interactive=False, work_class="histogram_plot")
            if callable(budget_getter)
            else None
        )
        apply_start = perf_counter()
        item.plot.setData(x, y)
        region = getattr(item, "region", None)
        if region is not None:
            if auto_level and len(x) > 0:
                levels = self.owner.getHistogramDataBounds() or _plot_bounds(x)
            else:
                levels = item.imageItem().getLevels()
            if levels is not None:
                if auto_level:
                    self.owner._apply_display_levels(
                        float(levels[0]), float(levels[1]), emit_user=False
                    )
                else:
                    with QtCore.QSignalBlocker(region):
                        region.setRegion((float(levels[0]), float(levels[1])))
        if budget is not None:
            budget.record_item(item_count=1)
            recorder = getattr(self.owner, "_record_gui_budget", None)
            if callable(recorder):
                recorder(budget)
        else:
            recorder = getattr(self.owner, "_record_gui_callback_observation", None)
            if callable(recorder):
                recorder(
                    channel="histogram_refresh",
                    work_class="histogram_plot",
                    elapsed_ms=(perf_counter() - apply_start) * 1000.0,
                    item_count=1,
                    byte_count=int(getattr(np.asarray(y), "nbytes", 0) if y is not None else 0),
                )
        if self._active_request_signature == result_signature:
            self._active_request_signature = None
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

    def cancel_manual_edit(self) -> None:
        self._cancel_pending_span_edit()
        popup = self._manual_popup
        if popup is not None:
            popup.reject()
            return
        self._manual_start_levels = None

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
                and not event.double()
                and not self._line_click_in_progress(event)
            ):
                event.accept()
                self._schedule_span_edit(event.scenePos())
                return
            self._manual_region_mouse_click_event(event)

        region.mouseClickEvent = mouse_click_event
        self._manual_region_installed = True

    def _handle_native_histogram_double_click(self, viewport, event) -> bool:
        event_type = event.type()
        if event_type not in {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QEvent.Type.MouseButtonDblClick,
        }:
            return False
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return False
        if event_type == QtCore.QEvent.Type.MouseButtonRelease:
            if not self._histogram_release_completes_double_click(event):
                return False
            event.accept()
            self._request_auto_window_from_histogram_event(event)
            return True
        if event_type != QtCore.QEvent.Type.MouseButtonDblClick:
            return False
        self._last_histogram_release = None
        if self._histogram_event_was_recently_handled(event):
            return False
        event.accept()
        self._request_auto_window_from_histogram_event(event)
        return True

    def _histogram_release_completes_double_click(self, event) -> bool:
        position = self._histogram_global_position(event)
        if position is None:
            self._last_histogram_release = None
            return False
        now = perf_counter()
        if self._histogram_position_was_recently_handled(now, position):
            self._last_histogram_release = None
            return False
        previous = self._last_histogram_release
        self._last_histogram_release = (now, position)
        if previous is None:
            return False
        previous_time, previous_position = previous
        interval_s = max(1, QtWidgets.QApplication.doubleClickInterval()) / 1000.0
        if now - previous_time > interval_s:
            return False
        max_distance = max(1, QtWidgets.QApplication.startDragDistance())
        delta = position - previous_position
        if delta.x() * delta.x() + delta.y() * delta.y() > max_distance * max_distance:
            return False
        self._last_histogram_release = None
        return True

    def _histogram_event_was_recently_handled(self, event) -> bool:
        position = self._histogram_global_position(event)
        if position is None:
            return False
        return self._histogram_position_was_recently_handled(perf_counter(), position)

    def _histogram_position_was_recently_handled(
        self, now: float, position: QtCore.QPointF
    ) -> bool:
        previous = self._last_histogram_auto_reset
        if previous is None:
            return False
        previous_time, previous_position = previous
        if now - previous_time > 0.25:
            return False
        return self._histogram_positions_match(position, previous_position)

    def _request_auto_window_from_histogram_event(self, event) -> None:
        position = self._histogram_global_position(event)
        if position is not None:
            self._last_histogram_auto_reset = (perf_counter(), position)
        self.request_auto_window()

    @staticmethod
    def _histogram_positions_match(first: QtCore.QPointF, second: QtCore.QPointF) -> bool:
        delta = first - second
        return delta.x() * delta.x() + delta.y() * delta.y() <= 1.0

    @staticmethod
    def _histogram_global_position(event) -> QtCore.QPointF | None:
        try:
            if hasattr(event, "globalPosition"):
                return QtCore.QPointF(event.globalPosition())
            if hasattr(event, "globalPos"):
                return QtCore.QPointF(event.globalPos())
        except Exception:
            return None
        return None

    def request_auto_window(self) -> None:
        self.cancel_manual_edit()
        signal = getattr(self.owner, "autoWindowRequested", None)
        if signal is not None:
            signal.emit()

    def _schedule_span_edit(self, scene_pos) -> None:
        self._pending_span_edit_scene_pos = QtCore.QPointF(scene_pos)
        self._pending_span_edit_timer.start(max(1, QtWidgets.QApplication.doubleClickInterval()))

    def _flush_pending_span_edit(self) -> None:
        scene_pos = self._pending_span_edit_scene_pos
        self._pending_span_edit_scene_pos = None
        if scene_pos is not None:
            self.begin_span_edit(scene_pos)

    def _cancel_pending_span_edit(self) -> None:
        if self._pending_span_edit_timer.isActive():
            self._pending_span_edit_timer.stop()
        self._pending_span_edit_scene_pos = None

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
        icon_label = QtWidgets.QLabel(frame)
        icon_label.setPixmap(material_icon("tonality").pixmap(14, 14))
        layout.addWidget(icon_label)
        layout.addWidget(QtWidgets.QLabel(str(label)))
        self.edit = HistogramNumberEdit(value=value, step=step, parent=frame)
        self.edit.setFixedWidth(92)
        self.edit.valueEdited.connect(self._apply_value)
        layout.addWidget(self.edit)
        self.accept_button = QtWidgets.QToolButton(frame)
        set_button_icon(
            self.accept_button, "done", icon_size=16, tooltip="Apply", text_beside_icon=False
        )
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
    request = histogram_plot_request_for_view(
        image_item,
        histogram_item,
        histogram_bounds=histogram_bounds,
        min_bin_screen_px=min_bin_screen_px,
        bin_cap=bin_cap,
    )
    if request is None:
        return None
    result = compute_histogram_plot(request)
    if not result.has_data:
        return None
    return result.x, result.y


def _sample_histogram_data(data: np.ndarray, *, target_image_size: int = 200) -> np.ndarray:
    return sample_histogram_data(data, target_image_size=target_image_size)


def histogram_plot_request_for_view(
    image_item,
    histogram_item,
    *,
    histogram_bounds=None,
    min_bin_screen_px: int = MIN_HISTOGRAM_BIN_SCREEN_PX,
    bin_cap: int = DEFAULT_HISTOGRAM_BIN_CAP,
    generation: int = 0,
) -> HistogramPlotRequest | None:
    data = getattr(image_item, "image", None)
    if data is None:
        return None
    data = np.asarray(data)
    if data.size == 0:
        return None
    visible_span = _visible_value_span(histogram_item)
    pixel_extent = _histogram_value_pixel_height(histogram_item)
    view_signature = (
        _finite_increasing_pair(histogram_bounds),
        None if visible_span is None else round(float(visible_span), 9),
        round(float(pixel_extent), 3),
        int(min_bin_screen_px),
        int(bin_cap),
    )
    return HistogramPlotRequest(
        data=data,
        source_identity=(id(data), tuple(data.shape), str(data.dtype)),
        histogram_bounds=_finite_increasing_pair(histogram_bounds),
        visible_value_span=visible_span,
        pixel_extent=pixel_extent,
        bin_cap=int(bin_cap),
        min_bin_screen_px=int(min_bin_screen_px),
        generation=int(generation),
        view_signature=view_signature,
    )


def _request_signature(request: HistogramPlotRequest):
    return (request.source_identity, request.view_signature)


def _result_signature(result: HistogramPlotResult):
    return (result.source_identity, result.view_signature)


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
            value = (
                rect.height()
                if getattr(histogram_item, "orientation", "vertical") == "vertical"
                else rect.width()
            )
            if value and np.isfinite(float(value)) and float(value) > 1.0:
                return float(value)
        except Exception:
            continue
    return 200.0


def _finite_increasing_pair(values) -> tuple[float, float] | None:
    return finite_increasing_pair(values)


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
