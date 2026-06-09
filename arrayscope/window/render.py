from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.display.colormaps import gray_colormap, phase_colormap
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.levels import finite_bounds
from arrayscope.display.imageview2d import MontageTileOverlay
from arrayscope.display.montage import (
    MontageTileState,
    make_montage_plan,
    make_montage_viewport_canvas,
    montage_rect_for_viewport,
    optimal_montage_columns,
)
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportIntent, ViewportPolicy
from arrayscope.core.memory_budget import (
    estimate_display_image_bytes,
    estimate_montage_bytes,
    format_bytes,
)
from arrayscope.core.memory_policy import (
    apply_policy_hysteresis,
    compute_memory_policy,
    input_nbytes_for,
)
from arrayscope.operations.evaluator import _document_key, stage_document_key
from arrayscope.operations.chunked import evaluate_image_snapshot_chunked
from arrayscope.profiles.model import profile_y_range
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.operations.evaluator import evaluate_image_snapshot, evaluate_line_snapshot, evaluate_scalar_snapshot
from arrayscope.operations.render_plan import (
    MAX_IDLE_PREFETCH_SLICES,
    PREFETCH_IDLE_DELAY_MS,
    RenderDecisionKind,
    choose_visible_render_decision,
    degraded_view_state,
    estimate_visible_render_context,
)
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.interaction_mode import InteractionMode
from arrayscope.window.montage_session import MontageRenderSession


MONTAGE_COMMIT_INTERVAL_MS = 33


@dataclass(frozen=True)
class RenderedView:
    view_state: object
    document_key: tuple
    display_image: DisplayImage
    geometry: DisplayGeometry


def getNumberOfDecimalPlaces(number):
    if isinstance(number, (int, np.integer)):
        return int(0)
    else:
        return int(max(1, (number.as_integer_ratio()[1]).bit_length()))


class RenderMixin:
    def getPixel(self, pos):
        source = getattr(self.img_view, "histogramSource", None)
        if source is None:
            source = getattr(self.img_view, "image", None)
        if source is None or self.view_state.image_axes is None:
            label = self.widgets['labels']['pixelValue']
            if hasattr(label, "set_pixel_status"):
                label.set_pixel_status("", self._slice_context_text())
            else:
                label.setText("")
            if hasattr(self, "img_view"):
                self.img_view.hideHud()
            return
        container = self.img_view.getView()
        mousePoint = container.mapSceneToView(pos)
        geometry = getattr(self, "display_geometry", None)
        status = None if geometry is None else geometry.montage_status_for_display_point(mousePoint.x(), mousePoint.y())
        if status is not None and status.kind != "loaded":
            text_pair = self._montage_status_value_text(status)
            if text_pair is None:
                if hasattr(self, "img_view"):
                    self.img_view.hideHud()
                label = self.widgets['labels']['pixelValue']
                if hasattr(label, "set_pixel_status"):
                    label.set_pixel_status("", self._slice_context_text())
                else:
                    label.setText("")
                return
            value_text, context = text_pair
            label = self.widgets['labels']['pixelValue']
            if hasattr(label, "set_pixel_status"):
                label.set_pixel_status(value_text, context)
            else:
                label.setText(value_text if not context else f"{value_text} | {context}")
            if hasattr(self, "img_view"):
                self.img_view.showHudText(value_text if not context else f"{value_text} | {context}", pos)
            return
        point_context = None if geometry is None else geometry.context_for_display_point(mousePoint.x(), mousePoint.y())
        if point_context is not None:
            mapping = point_context.mapping
            x_i, y_i = mapping.local_x, mapping.local_y
            context = point_context.context_text
            value = self._hover_value_from_display(mapping)
            if value is not None:
                self._commit_pixel_value(value, x_i, y_i, context, pos)
            return
        if hasattr(self, "img_view"):
            self.img_view.hideHud()

    def _montage_status_value_text(self, status):
        if status.kind in {"gap", "outside"}:
            return None
        context = ""
        axis = self.view_state.montage_axis
        if axis is not None and status.source_index is not None:
            context = f"d{axis}={status.source_index}"
        if status.kind == "skipped":
            return "tile skipped by memory budget", context
        return "tile loading...", context

    def _hover_value_from_display(self, mapping):
        if hasattr(self.img_view, "valueAtDisplayMapping"):
            value = self.img_view.valueAtDisplayMapping(mapping)
            if value is not None:
                if isinstance(value, np.ndarray):
                    return tuple(value.tolist())
                if np.isscalar(value):
                    try:
                        return value.item()
                    except AttributeError:
                        return value
                return value
        source = getattr(self.img_view, "histogramSource", None)
        if source is None:
            source = getattr(self.img_view, "image", None)
        if source is None:
            return None
        data = np.asarray(source)
        y_i = int(mapping.local_y)
        x_i = int(mapping.local_x)
        if y_i < 0 or x_i < 0 or y_i >= data.shape[0] or x_i >= data.shape[1]:
            return None
        value = data[y_i, x_i]
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if np.isscalar(value):
            try:
                return value.item()
            except AttributeError:
                return value
        return value

    def _request_pixel_value(self, index, x_i, y_i, context, pos):
        view_state = self.view_state
        document = self.document
        self._pixel_request_id = getattr(self, "_pixel_request_id", 0) + 1
        request_id = self._pixel_request_id

        def done(value):
            if request_id != getattr(self, "_pixel_request_id", 0):
                return
            self._commit_pixel_value(value, x_i, y_i, context, pos)

        cached = self.operation_evaluator.cached_scalar(view_state, index)
        if cached is not None:
            done(cached)
            return

        def evaluate():
            return evaluate_scalar_snapshot(
                document,
                view_state,
                index,
                stage_cache=self.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(document),
            )

        request_key = self.operation_evaluator.scalar_key(view_state, index, document=document)

        def done_result(result):
            if request_id != getattr(self, "_pixel_request_id", 0):
                return
            if request_key != self.operation_evaluator.scalar_key(view_state, index):
                return
            done(self.operation_evaluator.store_scalar_result(view_state, index, result))

        self.pixel_evaluation_controller.start(evaluate, on_done=done_result, on_error=lambda _exc: None, slow_ms=0)

    def _commit_pixel_value(self, value, x_i, y_i, context, pos):
        try:
            if isinstance(value, (tuple, list)):
                values = []
                for entry in value:
                    try:
                        values.append(f"{float(entry):.4g}")
                    except Exception:
                        values.append(str(entry))
                value_text = f"({x_i}, {y_i}) = ({', '.join(values)})"
            else:
                decimal_places = getNumberOfDecimalPlaces(abs(value))
                if decimal_places > 5:
                    value_text = "({}, {}) = {:.3e}".format(x_i, y_i, value)
                else:
                    value_text = "({}, {}) = {:.{}f}".format(x_i, y_i, value, decimal_places)
        except Exception:
            value_text = f"({x_i}, {y_i}) = {value}"
        text = value_text if not context else f"{value_text} | {context}"
        label = self.widgets['labels']['pixelValue']
        if hasattr(label, "set_pixel_status"):
            label.set_pixel_status(value_text, context)
        else:
            label.setText(text)
        if hasattr(self, "img_view"):
            self.img_view.showHudText(text, pos)

    def _slice_context_text(self):
        axes = self.view_state.non_display_axes()
        if not axes:
            return ""
        return " ".join(f"d{axis}={self.view_state.slice_indices[axis]}" for axis in axes)

    def _on_image_mouse_moved(self, pos):
        self.getPixel(pos)
    
    def _on_profile_marker_moved(self, image_x, image_y):
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            return
        if self.view_state.image_axes is None:
            return
        clamped = self._clamp_profile_marker_point(image_x, image_y)
        if clamped is None:
            self._clear_live_profile_marker()
            return
        if (float(clamped[0]), float(clamped[1])) != (float(image_x), float(image_y)):
            self.img_view.setProfileMarker(clamped[0], clamped[1], visible=True)
        self._pending_profile_point = (float(clamped[0]), float(clamped[1]))
        if not self._profile_timer.isActive():
            self._profile_timer.start()

    def _update_live_profile_from_pending_pos(self):
        point = self._pending_profile_point
        pos = self._pending_profile_pos
        self._pending_profile_point = None
        self._pending_profile_pos = None
        if point is None and pos is None:
            return
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            return
        if self.is_line_plot_mode():
            return

        if point is None:
            view = self.img_view.getView()
            if not view.sceneBoundingRect().contains(pos):
                self._clear_live_profile_marker()
                return
            mouse_point = view.mapSceneToView(pos)
            point = (mouse_point.x(), mouse_point.y())

        view_state = self.view_state
        document = self.document
        self._profile_request_id = getattr(self, "_profile_request_id", 0) + 1
        request_id = self._profile_request_id
        image_levels = self.img_view.getLevels()
        y_range_mode = self.profile_dock.y_range_mode()
        profile_axes = tuple(getattr(self, "profile_axes", ()) or ((self.view_state.line_axis,) if self.view_state.line_axis is not None else ()))
        clamped = self._clamp_profile_marker_point(point[0], point[1])
        if clamped is None:
            self._clear_live_profile_marker()
            return
        geometry = getattr(self, "display_geometry", None)
        status = None if geometry is None else geometry.montage_status_for_display_point(clamped[0], clamped[1])
        if status is not None and status.kind != "loaded":
            if status.kind in {"loading", "unloaded"}:
                self.profile_dock.update_line_results((), y_range=None)
                self.img_view.setProfileMarker(round(clamped[0]), round(clamped[1]), visible=True)
                show_status_message(self, "Montage tile loading; profile available when tile finishes.", timeout=2000)
                self._schedule_loading_montage_profile_retry(float(clamped[0]), float(clamped[1]))
            elif status.kind == "skipped":
                self.profile_dock.update_line_results((), y_range=None)
                self._clear_live_profile_marker()
                show_status_message(self, "Montage tile skipped by memory budget; zoom in or increase render budget.", timeout=3000)
            else:
                self._clear_live_profile_marker()
            return
        profile_states, profile_label_suffix = self._profile_states_for_display_point(view_state, clamped[0], clamped[1], profile_axes)
        if not profile_states:
            self._clear_live_profile_marker()
            return
        y_range = profile_y_range(y_range_mode, image_levels)
        cached_entries = []
        for profile_state in profile_states:
            cached = self.operation_evaluator.cached_line(profile_state)
            if cached is None:
                cached_entries = []
                break
            cached_entries.append((cached, profile_state, f"dim {profile_state.line_axis}{profile_label_suffix}"))
        if cached_entries:
            if request_id != getattr(self, "_profile_request_id", 0):
                return
            self.profile_dock.update_line_results(tuple(cached_entries), y_range=y_range)
            self._update_operation_dock()
            self.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
            return

        def evaluate():
            return tuple(
                (
                    profile_state,
                    evaluate_line_snapshot(
                        document,
                        profile_state,
                        stage_cache=self.operation_evaluator.stage_cache,
                        stage_document_key=stage_document_key(document),
                    ),
                )
                for profile_state in profile_states
            )

        request_keys = {profile_state: self.operation_evaluator.line_key(profile_state, document=document) for profile_state in profile_states}

        def done(results):
            if request_id != getattr(self, "_profile_request_id", 0):
                return
            entries = []
            for profile_state, result in results:
                if request_keys[profile_state] != self.operation_evaluator.line_key(profile_state):
                    return
                line_result = self.operation_evaluator.store_line_result(profile_state, result)
                entries.append((line_result, profile_state, f"dim {profile_state.line_axis}{profile_label_suffix}"))
            self.profile_dock.update_line_results(tuple(entries), y_range=y_range)
            self._update_operation_dock()
            self.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
            if view_state.montage_axis is None:
                for axis in profile_axes:
                    self._prefetch_profiles_near_marker(view_state, point[0], point[1], line_axis=axis)

        def error(exc):
            show_status_message(self, f"Live profile update failed: {exc}")
            self._clear_live_profile_marker()

        self.profile_evaluation_controller.start_latest(
            evaluate,
            key=tuple(request_keys.values()),
            priority=EvalPriority.LIVE_PROFILE,
            replace_group="live-profile",
            on_done=done,
            on_error=error,
        )

    def _clamp_profile_marker_point(self, image_x, image_y):
        geometry = getattr(self, "display_geometry", None)
        if geometry is None:
            return None
        return geometry.clamp_display_point(image_x, image_y)

    def _profile_states_for_display_point(self, view_state, image_x, image_y, profile_axes):
        geometry = getattr(self, "display_geometry", None)
        if geometry is None:
            return (), ""
        mapping = geometry.display_point_to_array_index(image_x, image_y)
        if mapping is None:
            return (), ""
        states = geometry.display_point_to_profile_states(image_x, image_y, profile_axes)
        suffix = "" if mapping.montage_axis is None or mapping.montage_index is None else f" d{mapping.montage_axis}={mapping.montage_index}"
        return states, suffix

    def _clear_live_profile_marker(self):
        if hasattr(self, "img_view"):
            self.img_view.hideProfileMarker()

    def _on_live_profile_toggled(self, enabled):
        if hasattr(self, "display_toolbar"):
            self.display_toolbar.set_current(live_profile=enabled)
        if enabled and hasattr(self, "profile_dock"):
            self.interaction_mode = InteractionMode.LIVE_PROFILE
            self._profile_dock_user_visible = None
            if hasattr(self, "img_view"):
                self.img_view.cancelPendingRoiDrawing()
                self.img_view.setInspectionTool("profile")
            self.layout_manager.set_managed_dock_visible(self.profile_dock, True, reason="live-profile")
            self._schedule_view_geometry_refresh()
            self.img_view.getView().setCursor(Qt.QtCore.Qt.CursorShape.CrossCursor)
            self._ensure_profile_marker()
        if not enabled:
            if getattr(self, "interaction_mode", None) == InteractionMode.LIVE_PROFILE:
                self.interaction_mode = InteractionMode.CURSOR
            if getattr(self, "_profile_dock_user_visible", None) is not True:
                self._profile_dock_user_visible = None
            self._pending_profile_pos = None
            self._pending_profile_point = None
            self._profile_timer.stop()
            self._clear_live_profile_marker()
            self.img_view.getView().unsetCursor()
            self.update_line_plot()

    def _on_profile_dock_visibility_changed(self, visible):
        if getattr(self, "_closing", False):
            return
        if getattr(self.layout_manager, "_visibility_preserve_active", False):
            return
        if not visible:
            self._profile_dock_user_visible = False
        if not visible and self.widgets['buttons']['display']['live_profile'].isChecked():
            self.widgets['buttons']['display']['live_profile'].setChecked(False)

    def _on_inspection_dock_visibility_changed(self, visible):
        if getattr(self, "_closing", False):
            return
        if getattr(self.layout_manager, "_visibility_preserve_active", False):
            return
        if not visible:
            self._inspection_dock_user_visible = False
        if visible and getattr(self, "_inspection_stale", False):
            self._refresh_inspection_dock_now()

    def _on_operation_dock_visibility_changed(self, visible):
        if getattr(self, "_closing", False):
            return
        if getattr(self.layout_manager, "_visibility_preserve_active", False):
            return
        if not visible:
            self._operation_dock_user_visible = False

    def _resize_profile_dock_default(self):
        self.layout_manager.resize_profile_dock_default()

    def _ensure_profile_marker(self):
        position = self.img_view.profileMarkerPosition()
        if position is None:
            x, y = self._default_profile_marker_position()
            self.img_view.setProfileMarker(x, y, visible=True)
            self._on_profile_marker_moved(x, y)
        else:
            self._on_profile_marker_moved(*position)

    def _default_profile_marker_position(self):
        geometry = getattr(self, "display_geometry", None)
        if geometry is None:
            return (0, 0)
        height, width = geometry.display_shape
        x = (width - 1) / 2.0
        y = (height - 1) / 2.0
        return (round(x), round(y))

    def _current_profile_y_range(self):
        if not hasattr(self, "profile_dock"):
            return None
        try:
            image_levels = self.img_view.getLevels()
        except Exception as exc:
            handle_ui_exception("profile y range levels", exc)
            image_levels = None
        return profile_y_range(self.profile_dock.y_range_mode(), image_levels)

    def _phase_colormap(self):
        return phase_colormap()

    def _apply_channel_colormap(self):
        if self.view_state.channel in (ChannelMode.COMPLEX, ChannelMode.ANGLE):
            self.img_view.setColorMap(self._phase_colormap())
        else:
            self.img_view.setColorMap(gray_colormap())

    def update_image_view(self, *, force_autolevel: bool = False):
        if self.view_state.image_axes is None: # No image view for 1D data
            return
        self._refresh_memory_policy(active_render=self.visible_evaluation_controller.is_busy())
        if self.view_state.montage_axis is not None:
            return self.update_montage_view(force_autolevel=force_autolevel)
        self._montage_session = None
        self._stop_montage_session_slow_overlay()
        self._current_montage_geometry = None
        self._current_montage_plan = None
        self._current_montage_canvas = None
        if hasattr(self.img_view, "clearMontageTileOverlays"):
            self.img_view.clearMontageTileOverlays()
            
        al = True
        force_auto = force_autolevel or getattr(self, '_force_autolevel', False)
        window_mode = self._current_window_mode()
        # reset the one-shot flag after using it
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        
        previous_levels = None
        previous_bounds = None
        if not force_auto and getattr(self.img_view, "image", None) is not None:
            try:
                previous_levels = self.img_view.getLevels()
                previous_bounds = self.img_view.getHistogramDataBounds()
            except Exception as exc:
                handle_ui_exception("previous image levels", exc)
                previous_levels = None
                previous_bounds = None

        colormap_lut = None
        if self.view_state.channel == ChannelMode.COMPLEX:
            colormap_lut = self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
        view_state = self.view_state
        document = self.document
        cached = self.operation_evaluator.cached_image(view_state, colormap_lut=colormap_lut)
        if cached is not None:
            self._last_render_was_degraded = False
            geometry = DisplayGeometry(view_state=view_state, display_shape=cached.data.shape[:2])
            self._apply_display_image(
                cached,
                geometry=geometry,
                window_mode=window_mode,
                previous_levels=previous_levels,
                previous_bounds=previous_bounds,
                force_auto=force_auto,
            )
            return
        estimated_bytes = self._estimated_image_display_bytes(view_state)
        context = estimate_visible_render_context(
            document,
            view_state,
            display_bytes=estimated_bytes,
            render_budget_bytes=self._visible_render_budget_bytes(),
        )
        decision = choose_visible_render_decision(context)
        self._last_render_context = context
        self._last_render_decision = decision
        self._last_render_request_key = None
        self._last_render_error = ""
        self._last_render_completed_ms = None
        if decision.kind == RenderDecisionKind.REFUSE:
            self.operation_evaluator.note_render_refused(decision.reason)
            show_status_message(
                self,
                decision.status_text or f"Image view would allocate {format_bytes(estimated_bytes)}. Reduce image-axis ranges or switch axes.",
                timeout=6000,
            )
            if getattr(self.img_view, "image", None) is not None:
                self.img_view.setImageStale(True)
                self.img_view.setEvaluationOverlay(True, decision.overlay_text or "Exact view over budget")
            self._update_operation_dock()
            return

        request_key = self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut, document=document)
        self._last_render_request_key = str(request_key)
        self.prefetch_evaluation_controller.cancel_prefetch()

        if decision.kind == RenderDecisionKind.DEGRADED_PREVIEW:
            preview_state = degraded_view_state(view_state, factor=decision.degraded_factor)
            preview_key = ("degraded_preview", request_key, decision.degraded_factor)

            def evaluate_preview(token):
                return evaluate_image_snapshot(
                    document,
                    preview_state,
                    colormap_lut=colormap_lut,
                    cancellation_token=token,
                    degraded=True,
                    stage_cache=self.operation_evaluator.stage_cache,
                    stage_document_key=stage_document_key(document),
                )

            def done_preview(result):
                if request_key != self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut):
                    return
                self.operation_evaluator.note_render_degraded()
                display_image = result.value
                self._degraded_rendered_view = display_image
                self._last_render_was_degraded = True
                geometry = DisplayGeometry(view_state=preview_state, display_shape=display_image.data.shape[:2])
                self._apply_display_image(
                    display_image,
                    geometry=geometry,
                    window_mode=window_mode,
                    previous_levels=previous_levels,
                    previous_bounds=previous_bounds,
                    force_auto=force_auto,
                )
                self.img_view.setImageStale(True)
                self.img_view.setEvaluationOverlay(True, decision.overlay_text)
                show_status_message(self, decision.status_text, timeout=6000)

            self.visible_evaluation_controller.start_latest(
                evaluate_preview,
                key=preview_key,
                priority=EvalPriority.VISIBLE_IMAGE,
                replace_group="visible-image",
                on_done=done_preview,
                on_error=lambda exc: show_status_message(self, f"Preview update failed: {exc}"),
                on_stale=lambda: None,
                on_slow=lambda: self.img_view.setEvaluationOverlay(True, "Updating preview..."),
                pass_token=True,
            )
            return

        def evaluate(token):
            if decision.kind == RenderDecisionKind.ASYNC_CHUNKED:
                return evaluate_image_snapshot_chunked(
                    document,
                    view_state,
                    chunk_axis=decision.chunk_axis,
                    chunk_size=decision.chunk_size,
                    colormap_lut=colormap_lut,
                    cancellation_token=token,
                    stage_cache=self.operation_evaluator.stage_cache,
                    stage_document_key=stage_document_key(document),
                )
            return evaluate_image_snapshot(
                document,
                view_state,
                colormap_lut=colormap_lut,
                cancellation_token=token,
                stage_cache=self.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(document),
            )

        def slow():
            self.img_view.setImageStale(True)
            text = "Updating view in chunks..." if decision.kind == RenderDecisionKind.ASYNC_CHUNKED else "Updating view..."
            self.img_view.setEvaluationOverlay(True, text)
            self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating image view")
            self._update_operation_dock()

        def done(result):
            if request_key != self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut):
                return
            self._last_render_completed_ms = float(getattr(result, "eval_ms", 0.0) or 0.0)
            self._last_render_was_degraded = False
            self._degraded_rendered_view = None
            display_image = self.operation_evaluator.store_image_result(view_state, colormap_lut, result)
            geometry = DisplayGeometry(view_state=view_state, display_shape=display_image.data.shape[:2])
            self._apply_display_image(
                display_image,
                geometry=geometry,
                window_mode=window_mode,
                previous_levels=previous_levels,
                previous_bounds=previous_bounds,
                force_auto=force_auto,
            )
            self._schedule_prefetch_nearby_slices(view_state, colormap_lut)

        def error(exc):
            self._last_render_error = str(exc)
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            show_status_message(self, f"Image update failed: {exc}")

        self.visible_evaluation_controller.start_latest(
            evaluate,
            key=request_key,
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible-image",
            on_done=done,
            on_error=error,
            on_stale=lambda: self.operation_evaluator.note_render_cancelled(),
            on_slow=slow,
            pass_token=True,
        )

    def update_montage_view(self, *, force_autolevel: bool = False):
        axis = self.view_state.montage_axis
        if axis is None or self.view_state.image_axes is None or axis in self.view_state.image_axes:
            return
        policy = self._refresh_memory_policy(active_render=self._montage_render_active())
        force_auto = force_autolevel or getattr(self, '_force_autolevel', False)
        if getattr(self, '_force_autolevel', False):
            self._force_autolevel = False
        window_mode = self._current_window_mode()
        previous_levels = None
        previous_bounds = None
        if not force_auto and getattr(self.img_view, "image", None) is not None:
            try:
                previous_levels = self.img_view.getLevels()
                previous_bounds = self.img_view.getHistogramDataBounds()
            except Exception as exc:
                handle_ui_exception("previous montage levels", exc)
                previous_levels = None
                previous_bounds = None

        colormap_lut = None
        if self.view_state.channel == ChannelMode.COMPLEX:
            colormap_lut = self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
        view_state = self.view_state
        document = self.document
        all_indices = tuple(view_state.montage_indices or tuple(range(int(view_state.shape[axis]))))
        viewport_size = self.img_view.graphicsView.viewport().size()
        viewport_shape = (max(1, viewport_size.height()), max(1, viewport_size.width()))
        tile_shape = self._montage_tile_shape(view_state)
        columns = view_state.montage_columns
        if columns is None and all_indices:
            columns = optimal_montage_columns(len(all_indices), tile_shape, viewport_shape)
        estimate = estimate_montage_bytes(
            tile_shape,
            len(all_indices),
            getattr(document.base_data, "dtype", np.dtype(float)),
            rgb=view_state.channel == ChannelMode.COMPLEX,
            histogram=True,
            columns=columns,
        )
        if estimate > policy.montage_canvas_budget_bytes:
            show_status_message(
                self,
                f"Montage would allocate {format_bytes(estimate)}. Showing visible tiles only; reduce tile count or zoom in for detail.",
                timeout=6000,
            )
        plan = make_montage_plan(
            view_state,
            axis=axis,
            indices=all_indices,
            tile_shape=tile_shape,
            columns=columns,
            viewport_shape=viewport_shape,
        )
        current_range = self._current_montage_global_view_range() if getattr(self.img_view, "image", None) is not None else None
        canvas_rect = montage_rect_for_viewport(plan, view_range=current_range, viewport_shape=viewport_shape)
        candidate_tiles = plan.tiles_intersecting(((canvas_rect[0], canvas_rect[2]), (canvas_rect[1], canvas_rect[3])), margin_tiles=0)
        output_dtype = np.uint8 if view_state.channel == ChannelMode.COMPLEX else getattr(document.base_data, "dtype", np.dtype(float))
        single_estimate = estimate_display_image_bytes(
            tile_shape,
            output_dtype,
            rgb=view_state.channel == ChannelMode.COMPLEX,
            histogram=True,
        )
        if single_estimate > policy.single_tile_budget_bytes:
            visible_tiles = ()
            skipped_tiles = tuple(candidate_tiles)
            skipped_count = len(skipped_tiles)
        else:
            visible_tiles = tuple(candidate_tiles)
            skipped_tiles = ()
            skipped_count = 0
        if not visible_tiles and not skipped_tiles:
            show_status_message(
                self,
                f"Montage tile would allocate {format_bytes(single_estimate)}. Zoom out less or reduce tile size/range.",
                timeout=6000,
            )
            return
        if skipped_count:
            self._warn_montage_tiles_skipped(
                skipped_count=skipped_count,
                tile_bytes=single_estimate,
                budget_bytes=policy.single_tile_budget_bytes,
                tile_shape=tile_shape,
            )
        cached_tiles = []
        missing_tiles = []
        for tile in visible_tiles:
            cached = self.operation_evaluator.cached_montage_tile(
                tile.view_state,
                montage_axis=axis,
                source_index=tile.source_index,
                colormap_lut=colormap_lut,
            )
            if cached is None:
                missing_tiles.append(tile)
            else:
                cached_tiles.append(cached.bind(tile) if hasattr(cached, "bind") else cached.payload().bind(tile))
        session_key = (
            "montage_tiles",
            _document_key(document),
            view_state,
            tuple(tile.source_index for tile in candidate_tiles),
            colormap_lut.tobytes() if colormap_lut is not None else None,
            viewport_shape if view_state.montage_columns is None else None,
        )
        session_id = int(getattr(self, "_montage_session_id", 0)) + 1
        self._montage_session_id = session_id
        session = MontageRenderSession(
            session_id=session_id,
            key=session_key,
            plan=plan,
            view_state=view_state,
            document=document,
            montage_axis=axis,
            colormap_lut=colormap_lut,
            viewport_shape=viewport_shape,
            view_range=current_range,
            output_dtype=np.dtype(output_dtype),
            rgb=view_state.channel == ChannelMode.COMPLEX,
            window_mode=window_mode,
            previous_levels=previous_levels,
            previous_bounds=previous_bounds,
            force_auto=force_auto,
            visible_tiles=tuple(visible_tiles),
            rendered_tiles={int(rendered.tile.montage_index): rendered for rendered in cached_tiles},
            loading_tiles={int(tile.montage_index) for tile in missing_tiles},
            skipped_tiles={int(tile.montage_index) for tile in skipped_tiles},
            pending_tiles=list(missing_tiles),
        )
        self._montage_session = session
        try:
            self._commit_montage_session_canvas(session, force=True)
        except MemoryError as exc:
            show_status_message(self, str(exc), timeout=6000)
            return
        if not session.pending_tiles and not session.loading_tiles:
            self._stop_montage_session_slow_overlay()
            self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.READY, "Montage view ready")
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            self._update_operation_dock()
            return
        self.prefetch_evaluation_controller.cancel_prefetch()
        self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating montage view")
        self._update_operation_dock()
        self._schedule_montage_session_slow_overlay(session)
        self._schedule_next_montage_tile(session)

    def _warn_montage_tiles_skipped(self, *, skipped_count: int, tile_bytes: int, budget_bytes: int, tile_shape) -> None:
        message = (
            f"Montage skipped {int(skipped_count)} tile(s) because each tile would allocate "
            f"{format_bytes(int(tile_bytes))}, over the visible render budget of {format_bytes(int(budget_bytes))}. "
            f"Tile shape is {tuple(int(size) for size in tile_shape)}. Zoom in, crop/range the image axes, "
            "or increase Performance > Render Memory Budget."
        )
        show_status_message(self, message, timeout=8000)
        warning_key = (int(skipped_count), int(tile_bytes), int(budget_bytes), tuple(int(size) for size in tile_shape))
        if getattr(self, "_last_montage_skip_warning_key", None) == warning_key:
            return
        self._last_montage_skip_warning_key = warning_key
        try:
            Qt.QtWidgets.QMessageBox.warning(self, "Montage tiles skipped", message)
        except Exception as exc:
            handle_ui_exception("montage skipped warning", exc)

    def _schedule_next_montage_tile(self, session: MontageRenderSession) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        tile = session.next_tile()
        if tile is None:
            self._schedule_montage_canvas_commit(session, force=True)
            if session.pending_tiles:
                self._schedule_next_montage_tile(session)
                return
            if self._is_current_montage_session(session.session_id, session.key):
                self._stop_montage_session_slow_overlay()
                self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.READY, "Montage view ready")
                self.img_view.setImageStale(False)
                self.img_view.setEvaluationOverlay(False)
                self._update_operation_dock()
            return

        def evaluate():
            return self._evaluate_montage_tile_snapshot(session, tile)

        self.visible_evaluation_controller.start_latest(
            evaluate,
            key=("montage_tile", session.key, int(tile.montage_index)),
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="visible-montage",
            on_done=lambda result, session_id=session.session_id, tile=tile: self._on_montage_tile_done(session_id, tile, result),
            on_error=lambda exc, session_id=session.session_id, tile=tile: self._on_montage_tile_error(session_id, tile, exc),
            on_stale=lambda: None,
            on_slow=lambda: self._on_montage_tile_slow(session.session_id),
            slow_ms=100,
        )

    def _evaluate_montage_tile_snapshot(self, session, tile):
        return evaluate_image_snapshot(
            session.document,
            tile.view_state,
            colormap_lut=session.colormap_lut,
            stage_cache=self.operation_evaluator.stage_cache,
            stage_document_key=stage_document_key(session.document),
        )

    def _on_montage_tile_slow(self, session_id):
        session = getattr(self, "_montage_session", None)
        if session is None or int(session.session_id) != int(session_id):
            return
        self._show_montage_session_loading_overlay(session)

    def _schedule_montage_session_slow_overlay(self, session):
        timer = self._ensure_montage_session_slow_timer()
        self._montage_session_slow_key = (int(session.session_id), session.key)
        timer.start(100)

    def _ensure_montage_session_slow_timer(self):
        timer = getattr(self, "_montage_session_slow_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._on_montage_session_slow_timer)
            self._montage_session_slow_timer = timer
        return timer

    def _stop_montage_session_slow_overlay(self):
        timer = getattr(self, "_montage_session_slow_timer", None)
        if timer is not None:
            timer.stop()
        self._montage_session_slow_key = None

    def _on_montage_session_slow_timer(self):
        key = getattr(self, "_montage_session_slow_key", None)
        if key is None:
            return
        session_id, session_key = key
        self._show_montage_session_loading_overlay_if_current(session_id, session_key)

    def _show_montage_session_loading_overlay_if_current(self, session_id, key):
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, key):
            return
        if not session.pending_tiles and not session.loading_tiles:
            return
        self._show_montage_session_loading_overlay(session)

    def _show_montage_session_loading_overlay(self, session):
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        session.show_loading_overlays = True
        self._schedule_montage_canvas_commit(session, force=True)
        self.img_view.setImageStale(True)
        self.img_view.setEvaluationOverlay(True, "Updating montage...")
        self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating montage view")
        self._update_operation_dock()

    def _on_montage_tile_done(self, session_id, tile, result) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        rendered = self.operation_evaluator.store_montage_tile_result(
            tile,
            montage_axis=session.montage_axis,
            colormap_lut=session.colormap_lut,
            result=result,
        )
        session.mark_loaded(rendered)
        self._schedule_montage_canvas_commit(session, force=not session.pending_tiles)
        self._schedule_next_montage_tile(session)

    def _on_montage_tile_error(self, session_id, tile, exc) -> None:
        session = getattr(self, "_montage_session", None)
        if session is None or not self._is_current_montage_session(session_id, session.key):
            return
        session.mark_skipped(tile)
        show_status_message(self, f"Montage tile update failed: {exc}", timeout=4000)
        self._schedule_montage_canvas_commit(session, force=True)
        self._schedule_next_montage_tile(session)

    def _schedule_montage_canvas_commit(self, session, *, force=False) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        elapsed_ms = (monotonic() - float(session.last_commit_monotonic or 0.0)) * 1000.0
        if force or session.canvas is None or elapsed_ms >= MONTAGE_COMMIT_INTERVAL_MS:
            self._commit_montage_session_canvas(session, force=force)
            return
        session.final_commit_pending = True
        timer = getattr(self, "_montage_commit_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_montage_canvas_commit)
            self._montage_commit_timer = timer
        if not timer.isActive():
            timer.start(max(1, int(MONTAGE_COMMIT_INTERVAL_MS - elapsed_ms)))

    def _flush_montage_canvas_commit(self):
        session = getattr(self, "_montage_session", None)
        if session is None or not session.final_commit_pending:
            return
        self._commit_montage_session_canvas(session, force=False)

    def _commit_montage_session_canvas(self, session, *, force=False) -> None:
        if not self._is_current_montage_session(session.session_id, session.key):
            return
        self._classify_canvas_tiles(session)
        previous_canvas = getattr(self, "_current_montage_canvas", None)
        previous_global_range = self._current_montage_global_view_range()
        canvas = make_montage_viewport_canvas(
            session.plan,
            session.rendered_tuple(),
            view_range=session.view_range,
            viewport_shape=session.viewport_shape,
            budget_bytes=self._montage_canvas_budget_bytes(),
            dtype=session.output_dtype,
            rgb=session.rgb,
            include_histogram=True,
            loading_tiles=session.loading_tile_tuple(),
            skipped_tiles=session.skipped_tile_tuple(),
        )
        session.canvas = canvas
        rendered_geometry = DisplayGeometry(
            view_state=session.view_state,
            display_shape=canvas.data.shape[:2],
            montage=session.plan.geometry,
            montage_origin_x=canvas.origin_x,
            montage_origin_y=canvas.origin_y,
            montage_tile_states=canvas.tile_states,
        )
        self._current_montage_geometry = session.plan.geometry
        self._current_montage_plan = session.plan
        self._current_montage_canvas = canvas
        self._next_viewport_policy = ViewportPolicy.PRESERVE
        self._montage_canvas_commit_active = True
        try:
            self._apply_display_image(
                DisplayImage(data=canvas.data, histogram_data=canvas.histogram_data),
                geometry=rendered_geometry,
                window_mode=session.window_mode,
                previous_levels=session.previous_levels,
                previous_bounds=session.previous_bounds,
                force_auto=session.force_auto,
            )
            self._update_montage_tile_overlays(canvas)
            if previous_canvas is not None and previous_global_range is not None:
                local_range = (
                    (
                        float(previous_global_range[0][0]) - float(canvas.origin_x),
                        float(previous_global_range[0][1]) - float(canvas.origin_x),
                    ),
                    (
                        float(previous_global_range[1][0]) - float(canvas.origin_y),
                        float(previous_global_range[1][1]) - float(canvas.origin_y),
                    ),
                )
                self.img_view.getView().setRange(xRange=local_range[0], yRange=local_range[1], padding=0)
        finally:
            self._montage_canvas_commit_active = False
        session.note_committed()
        self._retry_live_profile_after_montage_tile()

    def _classify_canvas_tiles(self, session) -> None:
        rect = montage_rect_for_viewport(session.plan, view_range=session.view_range, viewport_shape=session.viewport_shape)
        pending = {int(tile.montage_index) for tile in session.pending_tiles}
        for tile in session.plan.tiles:
            index = int(tile.montage_index)
            intersects = tile.x0 < rect[2] and tile.x0 + tile.width > rect[0] and tile.y0 < rect[3] and tile.y0 + tile.height > rect[1]
            if not intersects:
                continue
            if index in session.rendered_tiles or index in session.loading_tiles or index in session.skipped_tiles:
                continue
            if index in pending:
                session.mark_loading(tile)
            else:
                session.pending_tiles.append(tile)
                session.mark_loading(tile)

    def _update_montage_tile_overlays(self, canvas) -> None:
        if not hasattr(self.img_view, "setMontageTileOverlays"):
            return
        overlays = []
        for tile in canvas.full_plan.tiles:
            state = canvas.tile_states[int(tile.montage_index)] if int(tile.montage_index) < len(canvas.tile_states) else MontageTileState.UNLOADED
            if state == MontageTileState.LOADING and not bool(getattr(getattr(self, "_montage_session", None), "show_loading_overlays", False)):
                continue
            if state not in {MontageTileState.LOADING, MontageTileState.SKIPPED}:
                continue
            x = int(tile.x0) - int(canvas.origin_x)
            y = int(tile.y0) - int(canvas.origin_y)
            if x + int(tile.width) <= 0 or y + int(tile.height) <= 0 or x >= canvas.display_shape[1] or y >= canvas.display_shape[0]:
                continue
            overlays.append(
                MontageTileOverlay(
                    x=max(0, x),
                    y=max(0, y),
                    width=max(1, min(int(tile.width), canvas.display_shape[1] - max(0, x))),
                    height=max(1, min(int(tile.height), canvas.display_shape[0] - max(0, y))),
                    state=state.value,
                    text="Skipped" if state == MontageTileState.SKIPPED else "Loading",
                )
            )
        self.img_view.setMontageTileOverlays(tuple(overlays))

    def _is_current_montage_session(self, session_id, key) -> bool:
        session = getattr(self, "_montage_session", None)
        if session is None:
            return False
        return int(session.session_id) == int(session_id) and session.key == key and self.view_state.montage_axis is not None

    def _retry_live_profile_after_montage_tile(self) -> None:
        try:
            if not self.widgets['buttons']['display']['live_profile'].isChecked():
                return
            position = self.img_view.profileMarkerPosition()
            if position is None:
                return
            self._on_profile_marker_moved(*position)
            self._update_live_profile_from_pending_pos()
        except Exception as exc:
            handle_ui_exception("montage live profile retry", exc)

    def _schedule_loading_montage_profile_retry(self, x, y) -> None:
        timer = getattr(self, "_montage_profile_retry_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._retry_loading_montage_profile)
            self._montage_profile_retry_timer = timer
        self._pending_montage_profile_retry = (float(x), float(y))
        if not timer.isActive():
            timer.start(80)

    def _retry_loading_montage_profile(self) -> None:
        point = getattr(self, "_pending_montage_profile_retry", None)
        self._pending_montage_profile_retry = None
        if point is None or self.view_state.montage_axis is None:
            return
        if not self.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.profile_dock.isVisible():
            self._schedule_loading_montage_profile_retry(float(point[0]), float(point[1]))
            return
        self._pending_profile_point = (float(point[0]), float(point[1]))
        self._pending_profile_pos = None
        self._update_live_profile_from_pending_pos()

    def _schedule_montage_viewport_update(self) -> None:
        timer = getattr(self, "_montage_viewport_update_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_montage_viewport_update)
            self._montage_viewport_update_timer = timer
        timer.start(60)

    def _run_montage_viewport_update(self) -> None:
        if getattr(self, "_closing", False):
            return
        if self.view_state.montage_axis is None:
            return
        self.update_montage_view()

    def _montage_tile_shape(self, view_state):
        primary_axis, secondary_axis = view_state.image_axes
        primary_indices = view_state.axis_range_indices[primary_axis]
        secondary_indices = view_state.axis_range_indices[secondary_axis]
        return (
            len(primary_indices) if primary_indices is not None else int(view_state.shape[primary_axis]),
            len(secondary_indices) if secondary_indices is not None else int(view_state.shape[secondary_axis]),
        )

    def _current_montage_global_view_range(self):
        try:
            local_range = self.img_view.getView().viewRange()
        except Exception:
            return None
        canvas = getattr(self, "_current_montage_canvas", None)
        origin_x = 0 if canvas is None else int(canvas.origin_x)
        origin_y = 0 if canvas is None else int(canvas.origin_y)
        return (
            (float(local_range[0][0]) + origin_x, float(local_range[0][1]) + origin_x),
            (float(local_range[1][0]) + origin_y, float(local_range[1][1]) + origin_y),
        )

    def _apply_display_image(self, display_image, *, geometry, window_mode, previous_levels, previous_bounds, force_auto):
        try:
            viewport_policy = self._viewport_policy_for_display_shape(display_image.data.shape[:2])
            current_bounds = self._display_histogram_bounds(display_image)
            level_decision = choose_window_levels(
                mode=window_mode,
                previous_levels=previous_levels,
                previous_bounds=previous_bounds,
                current_bounds=current_bounds,
                default_levels=display_image.default_levels,
                force_auto=force_auto,
            )
            al = level_decision.auto_levels
            levels = level_decision.levels

            if display_image.histogram_data is not None:
                self.img_view.setImage(
                    display_image.data,
                    autoLevels=al,
                    levels=levels,
                    histogramData=display_image.histogram_data,
                    viewport_policy=viewport_policy,
                )
            else:
                self.img_view.setImage(display_image.data, autoLevels=al, levels=levels, viewport_policy=viewport_policy)
            self.display_geometry = geometry
            self._update_operation_dock()
            
            # Apply axis flips after setting the image
            self.apply_axis_flips()
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            self._refresh_inspection_dock()
            
        except Exception as e:
            handle_ui_exception("image update", e)
            show_status_message(self, f"Image update failed: {e}")

    def _viewport_policy_for_display_shape(self, display_shape):
        display_shape = tuple(int(size) for size in display_shape)
        requested = getattr(self, "_next_viewport_policy", None)
        if requested is not None:
            self._next_viewport_policy = None
            self._last_display_shape = display_shape
            return requested
        previous_shape = getattr(self, "_last_display_shape", None)
        self._last_display_shape = display_shape
        if previous_shape is None or tuple(previous_shape) != display_shape:
            return ViewportPolicy.RESET_FOR_NEW_SHAPE
        return ViewportPolicy.PRESERVE

    def _ensure_prefetch_idle_timer(self):
        timer = getattr(self, "_prefetch_idle_timer", None)
        if timer is None:
            timer = Qt.QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(PREFETCH_IDLE_DELAY_MS)
            timer.timeout.connect(self._run_pending_prefetch)
            self._prefetch_idle_timer = timer
        return timer

    def _schedule_prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.app_settings, "prefetch_nearby_slices", False):
            self.operation_evaluator.note_prefetch_skipped()
            return
        self._pending_prefetch_request = (view_state, colormap_lut)
        timer = self._ensure_prefetch_idle_timer()
        timer.start(PREFETCH_IDLE_DELAY_MS)

    def _run_pending_prefetch(self):
        request = getattr(self, "_pending_prefetch_request", None)
        self._pending_prefetch_request = None
        if request is None:
            return
        view_state, colormap_lut = request
        self._refresh_memory_policy(active_render=self.visible_evaluation_controller.is_busy())
        if self.visible_evaluation_controller.is_busy():
            self.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="visible_busy")
            self.operation_evaluator.note_prefetch_skipped()
            return
        self._prefetch_nearby_slices(view_state, colormap_lut)

    def _prefetch_nearby_slices(self, view_state, colormap_lut):
        if not getattr(self.app_settings, "prefetch_nearby_slices", False):
            self.operation_evaluator.note_prefetch_skipped()
            return
        if view_state.montage_axis is not None:
            self.operation_evaluator.note_prefetch_skipped()
            return
        if self.visible_evaluation_controller.is_busy():
            self.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="visible_busy")
            self.operation_evaluator.note_prefetch_skipped()
            return
        policy = self._memory_policy()
        if self.operation_evaluator._image_cache.bytes_used > int(self.operation_evaluator._image_cache.max_bytes * policy.cache_prefetch_skip_fraction):
            self.operation_evaluator.note_prefetch_skipped()
            return
        if self._estimated_image_display_bytes(view_state) > policy.prefetch_budget_bytes:
            self.operation_evaluator.note_prefetch_skipped()
            return
        if not self._prefetch_cost_allowed(view_state):
            self.prefetch_evaluation_controller.start_prefetch(lambda: None, blocked_reason="cost")
            self.operation_evaluator.note_prefetch_skipped()
            return
        axis = getattr(self, "_active_slice_axis", None)
        if axis is None or view_state.image_axes is None or axis in view_state.image_axes:
            return
        document = self.document
        size = view_state.shape[axis]
        current = view_state.slice_indices[axis]
        last = getattr(self, "_last_prefetch_slice_index", None)
        direction = 0 if last is None else (1 if current >= last else -1)
        self._last_prefetch_slice_index = current
        deltas = self._prefetch_deltas(direction, max_radius=min(2, max(1, size - 1)))
        scheduled = 0
        for delta in deltas:
            limit = MAX_IDLE_PREFETCH_SLICES if direction != 0 else 1
            if scheduled >= limit:
                break
            index = current + delta
            if 0 <= index < size:
                prefetch_state = view_state.with_slice(axis, index)
                prefetch_key = self.operation_evaluator.image_key(
                    prefetch_state,
                    colormap_lut=colormap_lut,
                    document=document,
                )
                started = self.prefetch_evaluation_controller.start_prefetch(
                    lambda prefetch_state=prefetch_state, document=document: self.operation_evaluator.prefetch_image_snapshot(
                        document,
                        prefetch_state,
                        colormap_lut=colormap_lut,
                    ),
                    on_done=lambda result, prefetch_state=prefetch_state, document=document, prefetch_key=prefetch_key: self._store_prefetch_image_if_current(
                        document,
                        prefetch_key,
                        prefetch_state,
                        colormap_lut,
                        result,
                    ),
                    key=prefetch_key,
                    memory_budget_bytes=policy.prefetch_budget_bytes,
                )
                self._note_prefetch_start(started)
                if started.scheduled:
                    scheduled += 1

    def _prefetch_cost_allowed(self, view_state):
        operations = tuple(self.document.enabled_operations)
        if not operations:
            return True
        image_axes = set(view_state.image_axes or ())
        for operation in operations:
            if type(operation).__name__ in {"Mean", "Sum", "Maximum", "Minimum", "RootSumSquares"} and int(operation.axis) in image_axes:
                return False
        cost = estimate_pipeline_cost(self.base_data.shape, getattr(self.base_data, "dtype", None), operations)
        peak = cost.estimated_peak_bytes or 0
        policy = self._memory_policy()
        if peak > policy.operation_prefetch_peak_budget_bytes:
            return False
        if any(type(operation).__name__ in {"CenteredFFT", "CenteredIFFT"} for operation in operations) and peak > policy.fft_prefetch_peak_budget_bytes:
            return False
        return True

    def _prefetch_deltas(self, direction, *, max_radius):
        radii = range(1, int(max_radius) + 1)
        if direction > 0:
            return tuple(delta for radius in radii for delta in (radius, -radius))
        if direction < 0:
            return tuple(delta for radius in radii for delta in (-radius, radius))
        return tuple(delta for radius in radii for delta in (-radius, radius))

    def _prefetch_profiles_near_marker(self, view_state, image_x, image_y, *, line_axis=None):
        if view_state.image_axes is None or line_axis is None:
            return
        document = self.document
        primary_axis, secondary_axis = view_state.image_axes
        cx = int(round(image_x))
        cy = int(round(image_y))
        max_radius = 4
        scheduled = 0
        request_key_cache = {}
        for radius in range(0, max_radius + 1):
            points = []
            if radius == 0:
                points.append((cx, cy))
            else:
                for dx in (-radius, radius):
                    points.append((cx + dx, cy))
                for dy in (-radius, radius):
                    points.append((cx, cy + dy))
            for x, y in points:
                if scheduled >= 24:
                    return
                if not (0 <= x < view_state.shape[secondary_axis] and 0 <= y < view_state.shape[primary_axis]):
                    continue
                profile_state = self.profile_coordinator.state_from_marker(view_state, x, y, line_axis=line_axis)
                if profile_state is None:
                    continue
                request_key_cache[profile_state] = self.operation_evaluator.line_key(profile_state, document=document)
                started = self.profile_evaluation_controller.start_prefetch(
                    lambda profile_state=profile_state, document=document: self.operation_evaluator.prefetch_line_snapshot(document, profile_state),
                    on_done=lambda result, profile_state=profile_state, document=document, key=request_key_cache[profile_state]: self._store_prefetch_profile_if_current(
                        document,
                        key,
                        profile_state,
                        result,
                    ),
                    key=request_key_cache[profile_state],
                    memory_budget_bytes=self._prefetch_budget_bytes(),
                )
                self._note_prefetch_start(started)
                if started.scheduled:
                    scheduled += 1

    def _store_prefetch_profile_if_current(self, document, request_key, profile_state, result):
        if request_key != self.operation_evaluator.line_key(profile_state):
            self.operation_evaluator.note_prefetch_stale()
            return False
        return self.operation_evaluator.store_prefetch_line_result(document, profile_state, result)

    def _store_prefetch_image_if_current(self, document, request_key, view_state, colormap_lut, result):
        current_key = self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut)
        if request_key != current_key:
            self.operation_evaluator.note_prefetch_stale()
            return False
        return self.operation_evaluator.store_prefetch_image_result(document, view_state, colormap_lut, result)

    def _note_prefetch_start(self, started):
        if started.scheduled:
            self.operation_evaluator.note_prefetch_scheduled()
        elif started.reason == "deduped":
            self.operation_evaluator.note_prefetch_deduped()
        elif started.reason == "limited":
            self.operation_evaluator.note_prefetch_limited()

    def _current_window_mode(self):
        if self.widgets['buttons']['display']['window_absolute'].isChecked():
            return "absolute"
        return "relative"

    def _display_histogram_bounds(self, display_image):
        data = display_image.histogram_data
        if data is None:
            data = display_image.data
        try:
            return finite_bounds(data)
        except Exception as exc:
            handle_ui_exception("display histogram bounds", exc)
            return None

    def _estimated_image_display_bytes(self, view_state):
        if view_state.image_axes is None:
            return 0
        shape = []
        for axis in view_state.image_axes:
            indices = view_state.axis_range_indices[axis]
            shape.append(len(indices) if indices is not None else view_state.shape[axis])
        dtypes = self.operation_coordinator.operation_dtype_estimates()
        dtype = dtypes[-1] if dtypes else getattr(self.document.base_data, "dtype", np.dtype(float))
        rgb = view_state.channel == ChannelMode.COMPLEX
        return estimate_display_image_bytes(tuple(shape), dtype, rgb=rgb, histogram=rgb)

    def _visible_render_budget_bytes(self) -> int:
        return int(self._memory_policy().visible_render_budget_bytes)

    def _montage_canvas_budget_bytes(self) -> int:
        return int(self._memory_policy().montage_canvas_budget_bytes)

    def _single_montage_tile_budget_bytes(self) -> int:
        return int(self._memory_policy().single_tile_budget_bytes)

    def _prefetch_budget_bytes(self) -> int:
        return int(self._memory_policy().prefetch_budget_bytes)

    def _memory_policy(self):
        policy = getattr(self, "_current_memory_policy", None)
        if policy is None:
            policy = self._refresh_memory_policy()
        return policy

    def _refresh_memory_policy(self, *, active_render: bool = False):
        current = compute_memory_policy(
            profile=getattr(getattr(self, "app_settings", None), "memory_profile", "balanced"),
            render_cap_mb=getattr(getattr(self, "app_settings", None), "render_memory_budget_mb", 512),
            input_nbytes=input_nbytes_for(getattr(self, "base_data", None)),
        )
        policy = apply_policy_hysteresis(
            getattr(self, "_current_memory_policy", None),
            current,
            active_render=bool(active_render),
        )
        self._current_memory_policy = policy
        self._apply_memory_policy_to_caches(policy)
        return policy

    def _apply_memory_policy_to_caches(self, policy) -> None:
        evaluator = getattr(self, "operation_evaluator", None)
        if evaluator is not None and hasattr(evaluator, "apply_memory_policy"):
            evaluator.apply_memory_policy(policy)

    def _montage_render_active(self) -> bool:
        session = getattr(self, "_montage_session", None)
        return bool(session is not None and (session.pending_tiles or session.loading_tiles))
    
    def update_display_mode(self):
        """Update the display mode for the image view"""
        if self.widgets['buttons']['display']['square_pixels'].isChecked():
            self.img_view.setDisplayMode('square_pixels')
        elif self.widgets['buttons']['display']['fit'].isChecked():
            self.img_view.setDisplayMode('fit')

    def _on_aspect_toolbar_changed(self, mode):
        if mode == "one_to_one":
            self.one_to_one_image()
            return
        self.fit_image_to_view()

    def _on_window_mode_changed(self, mode):
        self.widgets['buttons']['display']['window_relative'].setChecked(mode != "absolute")
        self.widgets['buttons']['display']['window_absolute'].setChecked(mode == "absolute")
        self.render(reason="window-mode")

    def _set_live_profile_checked(self, enabled):
        self.widgets['buttons']['display']['live_profile'].setChecked(bool(enabled))

    def fit_image_to_view(self, enabled=True):
        self.widgets['buttons']['display']['fit'].setChecked(bool(enabled))
        if hasattr(self, "display_toolbar"):
            blocker = Qt.QtCore.QSignalBlocker(self.display_toolbar.fit_action)
            try:
                self.display_toolbar.fit_action.setChecked(bool(enabled))
            finally:
                del blocker
        self.img_view.setFitLocked(bool(enabled))
        self._sync_controls_from_view_state()

    def one_to_one_image(self):
        self.widgets['buttons']['display']['square_pixels'].setChecked(True)
        if hasattr(self, "display_toolbar"):
            blocker = Qt.QtCore.QSignalBlocker(self.display_toolbar.fit_action)
            try:
                self.display_toolbar.fit_action.setChecked(False)
            finally:
                del blocker
        self.img_view.oneToOne()
        self._sync_controls_from_view_state()

    def auto_window_levels(self):
        self._force_autolevel = True
        self.render(reason="auto-window", force_autolevel=True)

    def toggle_profile_dock(self):
        visible = not self.profile_dock.isVisible()
        self.layout_manager.set_profile_dock_visible_from_user(visible)

    def _processing_pressed(self, btn):
        """Called on processing button press; if the button is already checked
        the user is re-clicking it and we should force an auto-level on next update."""
        try:
            if btn.isChecked():
                self._force_autolevel = True
            else:
                self._force_autolevel = False
        except Exception as exc:
            handle_ui_exception("processing button", exc)
            self._force_autolevel = False

        # Update the display group title
        self._update_display_group_title()

        # Force update to ensure view changes immediately
        self.render(reason="processing-pressed", force_autolevel=True)

    def _update_display_group_title(self):
        """Update the display group title with aspect ratio information."""
        mode = self.img_view.displayMode
        aspect_str = ''
        
        if mode == 'square_pixels': # Simple
            self.display_group.setTitle('Display (1:1)')
            return
        
        if mode == 'fit': #use the viewport aspect ratio
            aspect_str = ''
            try:
                if hasattr(self.img_view, 'image') and self.img_view.image is not None:
                    view = self.img_view.getView()
                    
                    img_height, img_width = self.img_view.image.shape[:2]
                    widget_ratio = view.size().width() / view.size().height()
                    img_ratio = img_width / img_height
                    ratio = img_ratio * widget_ratio
                    
                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
            finally:
                self.display_group.setTitle(f'Display {aspect_str}')
            
        else:
            self.display_group.setTitle('Display')
    
    def update_line_plot(self):
        if not hasattr(self, "profile_dock") or not self.profile_dock.isVisible():
            return
        if self.widgets['buttons']['display']['live_profile'].isChecked():
            position = self.img_view.profileMarkerPosition()
            if position is not None:
                self._on_profile_marker_moved(*position)
                return
        view_state = self.view_state
        y_range = self._current_profile_y_range()
        document = self.document
        profile_axes = tuple(getattr(self, "profile_axes", ()) or ((view_state.line_axis,) if view_state.line_axis is not None else ()))
        profile_states = tuple(view_state.with_line_axis(axis) for axis in profile_axes)
        cached_entries = []
        for profile_state in profile_states:
            cached = self.operation_evaluator.cached_line(profile_state)
            if cached is None:
                cached_entries = []
                break
            cached_entries.append((cached, profile_state, f"dim {profile_state.line_axis}"))
        if cached_entries:
            self.profile_dock.update_line_results(tuple(cached_entries), y_range=y_range)
            self._update_operation_dock()
            return

        def evaluate():
            return tuple(
                (
                    profile_state,
                    evaluate_line_snapshot(
                        document,
                        profile_state,
                        stage_cache=self.operation_evaluator.stage_cache,
                        stage_document_key=stage_document_key(document),
                    ),
                )
                for profile_state in profile_states
            )

        request_keys = {profile_state: self.operation_evaluator.line_key(profile_state, document=document) for profile_state in profile_states}

        def done(results):
            entries = []
            for profile_state, result in results:
                if request_keys[profile_state] != self.operation_evaluator.line_key(profile_state):
                    return
                line_result = self.operation_evaluator.store_line_result(profile_state, result)
                entries.append((line_result, profile_state, f"dim {profile_state.line_axis}"))
            self.profile_dock.update_line_results(tuple(entries), y_range=y_range)
            self._update_operation_dock()

        self.profile_evaluation_controller.start_latest(
            evaluate,
            key=tuple(request_keys.values()),
            priority=EvalPriority.LIVE_PROFILE,
            replace_group="profile-plot",
            on_done=done,
            on_error=lambda exc: show_status_message(self, f"Profile update failed: {exc}"),
        )

    def _on_view_range_changed(self):
        """Update display group title when view range changes (for fit mode)."""
        self._update_display_group_title()
        if self.view_state.montage_axis is not None and not getattr(self, "_montage_canvas_commit_active", False):
            self._schedule_montage_viewport_update()

    def on_tab_changed(self, index):
        """Handle central image tab changes."""
        self.update_dimension_controls()
        self.update()
        self.line_plot.hide_crosshair()
    
    def is_line_plot_mode(self):
        """The historical line-plot tab is no longer the primary plot surface."""
        return False

    def render(self, *, reason: str = "state", force_autolevel: bool = False):
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._coerce_channel_for_current_dtype()
        self._sync_controls_from_view_state()
        if hasattr(self, "tab_widget"):
            self.tab_widget.setVisible(self.data.ndim >= 2)
        self._update_channel_controls()
        self.update_dimension_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_image_view(force_autolevel=force_autolevel)
        if self.profile_dock.isVisible() or self.widgets['buttons']['display']['live_profile'].isChecked():
            self.update_line_plot()
        self._update_operation_dock()
        self._sync_progressive_docks()
