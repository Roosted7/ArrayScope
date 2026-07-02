from __future__ import annotations

from time import perf_counter

import numpy as np
import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.view_state import ChannelMode
from arrayscope.core.work_graph import WorkItem, WorkLane
from arrayscope.core.window_levels import LevelSourceRank
from arrayscope.display.colormaps import named_colormap, phase_colormap
from arrayscope.display.colormap_policy import resolved_colormap_name
from arrayscope.display.planning import normalize_bounds
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.operations.evaluator import (
    _document_key,
    evaluate_line_snapshot,
    evaluate_scalar_snapshot,
    stage_document_key,
)
from arrayscope.profiles.model import profile_y_range
from arrayscope.ui.toasts import show_revert_action, show_status_message
from arrayscope.display.model.frame import CommittedDisplayFrame, TiledValueSource
from arrayscope.display.geometry import display_geometry_coordinates_equal
from arrayscope.window.display_presenter import DisplayPresentationMixin
from arrayscope.window.evaluation_controller import EvalPriority
from arrayscope.window.interaction_mode import InteractionMode
from arrayscope.window.frame_renderer import FrameRenderMixin
from arrayscope.window.render_prefetch import RenderPrefetchMixin
from arrayscope.window.render_resources import RenderResourceMixin


def getNumberOfDecimalPlaces(number):
    if isinstance(number, (int, np.integer)):
        return int(0)
    else:
        return int(max(1, (number.as_integer_ratio()[1]).bit_length()))


class RenderOrchestrator(
    DisplayPresentationMixin,
    FrameRenderMixin,
    RenderPrefetchMixin,
    RenderResourceMixin,
    Qt.QtCore.QObject,
):
    """Owner of rendering orchestration state and scheduling.

    The window composes exactly one orchestrator (``window.renderer``). All
    frame-planning, montage-session, presentation-commit, prefetch, and
    resource-budget state lives here rather than on the window; window-level
    semantic state and services (``view_state``, ``document``, widgets,
    evaluation controllers) are reached through ``self.win``.

    Being a ``QObject`` child of the window ties every orchestrator timer to
    the window's lifetime.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win

    def _active_display_colormap_lut(self):
        view = getattr(self.win, "img_view", None)
        getter = getattr(view, "displayColorMapLookupTable", None)
        if callable(getter):
            return getter()
        return self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)

    def _evaluation_colormap_lut(self, view_state=None, *, shader_display: bool | None = None):
        """Return a LUT only when CPU materialization genuinely depends on it."""

        state = self.win.view_state if view_state is None else view_state
        if state.channel != ChannelMode.COMPLEX:
            return None
        if shader_display is None:
            from arrayscope.display.backend_contract import image_view_backend_capabilities

            shader_display = bool(image_view_backend_capabilities(self.win.img_view).shader_windowing)
        if shader_display:
            return None
        return self._active_display_colormap_lut()

    def getPixel(self, pos):
        source = getattr(self.win.img_view, "histogramSource", None)
        if source is None:
            source = getattr(self.win.img_view, "image", None)
        if source is None or self.win.view_state.image_axes is None:
            label = self.win.widgets['labels']['pixelValue']
            if hasattr(label, "set_pixel_status"):
                label.set_pixel_status("", self._slice_context_text())
            else:
                label.setText("")
            if hasattr(self.win, "img_view"):
                self.win.img_view.hideHud()
            return
        container = self.win.img_view.getView()
        mousePoint = container.mapSceneToView(pos)
        geometry = getattr(self, "display_geometry", None)
        status = None if geometry is None else geometry.view_point_to_tile_point(mousePoint.x(), mousePoint.y(), require_loaded=True)
        if status is not None and status.kind != "loaded":
            text_pair = self._montage_status_value_text(status)
            if text_pair is None:
                if hasattr(self.win, "img_view"):
                    self.win.img_view.hideHud()
                label = self.win.widgets['labels']['pixelValue']
                if hasattr(label, "set_pixel_status"):
                    label.set_pixel_status("", self._slice_context_text())
                else:
                    label.setText("")
                return
            value_text, context = text_pair
            label = self.win.widgets['labels']['pixelValue']
            if hasattr(label, "set_pixel_status"):
                label.set_pixel_status(value_text, context)
            else:
                label.setText(value_text if not context else f"{value_text} | {context}")
            if hasattr(self.win, "img_view"):
                self.win.img_view.showHudText(value_text if not context else f"{value_text} | {context}", pos)
            return
        point_context = None if geometry is None else geometry.context_for_view_point(mousePoint.x(), mousePoint.y())
        if point_context is not None:
            mapping = point_context.mapping
            if mapping.montage_axis is None:
                x_i, y_i = mapping.view_x, mapping.view_y
            else:
                x_i, y_i = mapping.local_x, mapping.local_y
            context = point_context.context_text
            value = self._hover_value_from_display(mapping)
            if value is not None:
                self._commit_pixel_value(value, x_i, y_i, context, pos)
            return
        if hasattr(self.win, "img_view"):
            self.win.img_view.hideHud()

    def _montage_status_value_text(self, status):
        if status.kind in {"gap", "outside"}:
            return None
        context = ""
        axis = self.win.view_state.montage_axis
        if axis is not None and status.source_index is not None:
            context = f"d{axis}={status.source_index}"
        if status.kind == "skipped":
            return "tile skipped by memory budget", context
        return "tile loading...", context

    def _hover_value_from_display(self, mapping):
        frame = getattr(self.win, "_committed_display_frame", None)
        if frame is None or not self._is_committed_display_frame_current(frame):
            return None
        value_source = getattr(frame, "value_source", None)
        if value_source is None:
            return None
        return value_source.value_at(mapping)

    def _is_committed_display_frame_current(self, frame: CommittedDisplayFrame) -> bool:
        if not self._is_current_render_generation(int(frame.key.render_generation)):
            return False
        if frame.key.document_key != _document_key(self.win.document):
            return False
        display_geometry = getattr(self, "display_geometry", None)
        if frame.geometry != display_geometry and not (
            isinstance(frame.value_source, TiledValueSource)
            and display_geometry_coordinates_equal(frame.geometry, display_geometry)
        ):
            return False
        if frame.data is None:
            if not isinstance(frame.value_source, TiledValueSource):
                return False
        elif tuple(np.shape(frame.data)[:2]) != tuple(frame.geometry.display_shape):
            return False
        if frame.histogram_data is not None and tuple(np.shape(frame.histogram_data)[:2]) != tuple(frame.geometry.display_shape):
            return False
        if frame.key.request_key != getattr(self, "_committed_display_request_key", None):
            return False
        return True

    def _request_pixel_value(self, index, x_i, y_i, context, pos):
        view_state = self.win.view_state
        document = self.win.document
        self._pixel_request_id = getattr(self, "_pixel_request_id", 0) + 1
        request_id = self._pixel_request_id

        def done(value):
            if request_id != getattr(self, "_pixel_request_id", 0):
                return
            self._commit_pixel_value(value, x_i, y_i, context, pos)

        cached = self.win.operation_evaluator.cached_scalar(view_state, index)
        if cached is not None:
            done(cached)
            return

        def evaluate():
            eval_context = self.win._evaluation_context(ComputeLane.PIXEL, None)
            return evaluate_scalar_snapshot(
                document,
                view_state,
                index,
                stage_cache=self.win.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(document),
                evaluation_context=eval_context,
            )

        request_key = self.win.operation_evaluator.scalar_key(view_state, index, document=document)
        frame_target = FrameTarget(
            semantic_key=request_key,
            viewport_key=("pixel", tuple(int(value) for value in index)),
            presentation_key=("pixel-hover",),
            quality="exact-visible",
        )

        def done_result(result):
            if request_id != getattr(self, "_pixel_request_id", 0):
                return
            if request_key != self.win.operation_evaluator.scalar_key(view_state, index):
                return
            done(self.win.operation_evaluator.store_scalar_result(view_state, index, result))

        self.win.pixel_evaluation_controller.start_latest(
            evaluate,
            key=request_key,
            priority=EvalPriority.HOVER,
            replace_group="pixel-hover",
            frame_target=frame_target,
            supersession_key="pixel-hover",
            supersession_value=(request_id, request_key),
            work_item=WorkItem(
                key=("pixel_hover", request_key),
                lane=WorkLane.PROFILE_ROI_HOVER,
                frame_target=frame_target,
                supersession_key="pixel-hover",
                supersession_value=(request_id, request_key),
                estimated_bytes=8,
                expected_value=1.0,
                reusable_output=True,
            ),
            on_done=done_result,
            on_error=lambda _exc: None,
            slow_ms=0,
        )

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
        label = self.win.widgets['labels']['pixelValue']
        if hasattr(label, "set_pixel_status"):
            label.set_pixel_status(value_text, context)
        else:
            label.setText(text)
        if hasattr(self.win, "img_view"):
            self.win.img_view.showHudText(text, pos)

    def _slice_context_text(self):
        axes = self.win.view_state.non_display_axes()
        if not axes:
            return ""
        return " ".join(f"d{axis}={self.win.view_state.slice_indices[axis]}" for axis in axes)

    def _on_image_mouse_moved(self, pos):
        self._last_image_mouse_scene_pos = pos
        self._last_image_hover_focus = self._image_hover_focus_from_scene_pos(pos)
        self._last_image_hover_focus_frame_key = getattr(getattr(self.win, "_committed_display_frame", None), "key", None)
        self.getPixel(pos)
        schedule_priority = getattr(self, "_schedule_montage_priority_retarget_from_hover", None)
        if callable(schedule_priority):
            schedule_priority()

    def _clear_image_hover_state(self) -> None:
        self._last_image_mouse_scene_pos = None
        self._last_image_hover_focus = None
        self._last_image_hover_focus_frame_key = None
        label = None
        widgets = getattr(self.win, "widgets", None)
        if isinstance(widgets, dict):
            labels = widgets.get("labels", {})
            if isinstance(labels, dict):
                label = labels.get("pixelValue")
        if label is not None:
            if hasattr(label, "set_pixel_status"):
                label.set_pixel_status("", self._slice_context_text())
            elif hasattr(label, "setText"):
                label.setText("")
        view = getattr(self.win, "img_view", None)
        hide_hud = getattr(view, "hideHud", None)
        if callable(hide_hud):
            hide_hud()

    def _refresh_hover_after_display_commit(self) -> None:
        pos = getattr(self, "_last_image_mouse_scene_pos", None)
        if pos is not None:
            self._last_image_hover_focus = self._image_hover_focus_from_scene_pos(pos)
            self._last_image_hover_focus_frame_key = getattr(getattr(self.win, "_committed_display_frame", None), "key", None)
            self.getPixel(pos)

    def _image_hover_focus_from_scene_pos(self, pos):
        try:
            view_point = self.win.img_view.getView().mapSceneToView(pos)
            x = float(view_point.x())
            y = float(view_point.y())
            geometry = getattr(self, "display_geometry", None)
            status = None if geometry is None else geometry.view_point_to_tile_point(x, y, require_loaded=False)
            if status is None or status.kind in {"outside", "gap"}:
                return None
            return (x, y)
        except Exception:
            return None

    def _on_profile_marker_moved(self, image_x, image_y):
        if not self.win.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.win.profile_dock.isVisible():
            return
        if self.win.view_state.image_axes is None:
            return
        clamped = self._clamp_profile_marker_point(image_x, image_y)
        if clamped is None:
            self._clear_live_profile_marker()
            return
        if (float(clamped[0]), float(clamped[1])) != (float(image_x), float(image_y)):
            self.win.img_view.setProfileMarker(clamped[0], clamped[1], visible=True)
        self.win._pending_profile_point = (float(clamped[0]), float(clamped[1]))
        if not self.win._profile_timer.isActive():
            self.win._profile_timer.start()

    def _update_live_profile_from_pending_pos(self):
        from time import perf_counter

        profile_update_start = perf_counter()
        point = self.win._pending_profile_point
        pos = self.win._pending_profile_pos
        self.win._pending_profile_point = None
        self.win._pending_profile_pos = None
        if point is None and pos is None:
            return
        if not self.win.widgets['buttons']['display']['live_profile'].isChecked():
            return
        if not self.win.profile_dock.isVisible():
            return
        if self.is_line_plot_mode():
            return

        if point is None:
            view = self.win.img_view.getView()
            if not view.sceneBoundingRect().contains(pos):
                self._clear_live_profile_marker()
                return
            mouse_point = view.mapSceneToView(pos)
            point = (mouse_point.x(), mouse_point.y())

        view_state = self.win.view_state
        document = self.win.document
        self._profile_request_id = getattr(self, "_profile_request_id", 0) + 1
        request_id = self._profile_request_id
        image_levels = self.win.img_view.getLevels()
        y_range_mode = self.win.profile_dock.y_range_mode()
        profile_axes = tuple(getattr(self.win, "profile_axes", ()) or ((self.win.view_state.line_axis,) if self.win.view_state.line_axis is not None else ()))
        clamped = self._clamp_profile_marker_point(point[0], point[1])
        if clamped is None:
            self._clear_live_profile_marker()
            return
        geometry = getattr(self, "display_geometry", None)
        status = None if geometry is None else geometry.view_point_to_tile_point(clamped[0], clamped[1], require_loaded=False)
        if status is not None and status.kind in {"gap", "outside"}:
            self._clear_live_profile_marker()
            return
        profile_states, profile_label_suffix = self._profile_states_for_display_point(view_state, clamped[0], clamped[1], profile_axes)
        if not profile_states:
            self._clear_live_profile_marker()
            return
        y_range = profile_y_range(y_range_mode, image_levels)
        cached_entries = []
        for profile_state in profile_states:
            cached = self.win.operation_evaluator.cached_line(profile_state)
            if cached is None:
                cached_entries = []
                break
            cached_entries.append((cached, profile_state, f"dim {profile_state.line_axis}{profile_label_suffix}"))
        if cached_entries:
            if request_id != getattr(self, "_profile_request_id", 0):
                return
            self.win.profile_dock.update_line_results(tuple(cached_entries), y_range=y_range)
            self.win._update_operation_dock()
            self.win.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
            if hasattr(self.win, "_record_ui_work"):
                self.win._record_ui_work("profile_update", (perf_counter() - profile_update_start) * 1000.0)
            return

        def evaluate():
            eval_context = self.win._evaluation_context(ComputeLane.PROFILE, None)
            return tuple(
                (
                    profile_state,
                    evaluate_line_snapshot(
                        document,
                        profile_state,
                        stage_cache=self.win.operation_evaluator.stage_cache,
                        stage_document_key=stage_document_key(document),
                        evaluation_context=eval_context,
                    ),
                )
                for profile_state in profile_states
            )

        request_keys = {profile_state: self.win.operation_evaluator.line_key(profile_state, document=document) for profile_state in profile_states}

        def done(results):
            if request_id != getattr(self, "_profile_request_id", 0):
                return
            entries = []
            for profile_state, result in results:
                if request_keys[profile_state] != self.win.operation_evaluator.line_key(profile_state):
                    return
                line_result = self.win.operation_evaluator.store_line_result(profile_state, result)
                entries.append((line_result, profile_state, f"dim {profile_state.line_axis}{profile_label_suffix}"))
            self.win.profile_dock.update_line_results(tuple(entries), y_range=y_range)
            self.win._update_operation_dock()
            self.win.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
            if hasattr(self.win, "_record_ui_work"):
                self.win._record_ui_work("profile_update", (perf_counter() - profile_update_start) * 1000.0)
            if view_state.montage_axis is None:
                for axis in profile_axes:
                    self._prefetch_profiles_near_marker(view_state, point[0], point[1], line_axis=axis)

        def error(exc):
            show_status_message(self.win, f"Live profile update failed: {exc}")
            self._clear_live_profile_marker()

        self.win.profile_evaluation_controller.start_latest(
            evaluate,
            key=tuple(request_keys.values()),
            priority=EvalPriority.LIVE_PROFILE,
            replace_group="live-profile",
            frame_target=FrameTarget(
                semantic_key=tuple(request_keys.values()),
                viewport_key=("profile", tuple(profile_axes), tuple(point)),
                presentation_key=("live-profile",),
                quality="exact-visible",
            ),
            supersession_key="live-profile",
            supersession_value=tuple(request_keys.values()),
            work_item=WorkItem(
                key=("live_profile", tuple(request_keys.values())),
                lane=WorkLane.PROFILE_ROI_HOVER,
                frame_target=FrameTarget(
                    semantic_key=tuple(request_keys.values()),
                    viewport_key=("profile", tuple(profile_axes), tuple(point)),
                    presentation_key=("live-profile",),
                    quality="exact-visible",
                ),
                supersession_key="live-profile",
                supersession_value=tuple(request_keys.values()),
                estimated_bytes=int(self._prefetch_budget_bytes()),
                expected_value=2.0,
                reusable_output=True,
            ),
            on_done=done,
            on_error=error,
        )

    def _clamp_profile_marker_point(self, image_x, image_y):
        geometry = getattr(self, "display_geometry", None)
        if geometry is None:
            return None
        return geometry.clamp_view_point(image_x, image_y)

    def _profile_states_for_display_point(self, view_state, image_x, image_y, profile_axes):
        geometry = getattr(self, "display_geometry", None)
        if geometry is None:
            return (), ""
        mapping = geometry.view_point_to_array_index(image_x, image_y, require_loaded=False)
        if mapping is None:
            return (), ""
        states = geometry.view_point_to_profile_states(image_x, image_y, profile_axes, require_loaded=False)
        suffix = "" if mapping.montage_axis is None or mapping.montage_index is None else f" d{mapping.montage_axis}={mapping.montage_index}"
        return states, suffix

    def _clear_live_profile_marker(self):
        if hasattr(self.win, "img_view"):
            self.win.img_view.hideProfileMarker()

    def _on_live_profile_toggled(self, enabled):
        if enabled and hasattr(self.win, "profile_dock"):
            self.win.interaction_mode = InteractionMode.LIVE_PROFILE
            self.win._profile_dock_user_visible = None
            if hasattr(self.win, "img_view"):
                self.win.img_view.cancelPendingRoiDrawing()
                self.win.img_view.setInspectionTool("profile")
            self.win.layout_manager.set_managed_dock_visible(self.win.profile_dock, True, reason="live-profile")
            self.win._schedule_view_geometry_refresh()
            self.win.img_view.getView().setCursor(Qt.QtCore.Qt.CursorShape.CrossCursor)
            self._ensure_profile_marker()
        if not enabled:
            if getattr(self.win, "interaction_mode", None) == InteractionMode.LIVE_PROFILE:
                self.win.interaction_mode = InteractionMode.CURSOR
            if getattr(self.win, "_profile_dock_user_visible", None) is not True:
                self.win._profile_dock_user_visible = None
            self.win._pending_profile_pos = None
            self.win._pending_profile_point = None
            self.win._profile_timer.stop()
            self._clear_live_profile_marker()
            self.win.img_view.getView().unsetCursor()
            self.update_line_plot()

    def _on_profile_dock_visibility_changed(self, visible):
        if getattr(self.win, "_closing", False):
            return
        if getattr(self.win.layout_manager, "_visibility_preserve_active", False):
            return
        if not visible:
            self.win._profile_dock_user_visible = False
        if not visible and self.win.widgets['buttons']['display']['live_profile'].isChecked():
            self.win.widgets['buttons']['display']['live_profile'].setChecked(False)
        if visible:
            retry_profile = getattr(self, "_retry_loading_montage_profile", None)
            if callable(retry_profile):
                retry_profile()

    def _on_inspection_dock_visibility_changed(self, visible):
        if getattr(self.win, "_closing", False):
            return
        if getattr(self.win.layout_manager, "_visibility_preserve_active", False):
            return
        if not visible:
            self.win._inspection_dock_user_visible = False
        if visible and getattr(self.win, "_inspection_stale", False):
            self.win._refresh_inspection_dock_now()

    def _on_operation_dock_visibility_changed(self, visible):
        if getattr(self.win, "_closing", False):
            return
        if getattr(self.win.layout_manager, "_visibility_preserve_active", False):
            return
        if not visible:
            self.win._operation_dock_user_visible = False

    def _resize_profile_dock_default(self):
        self.win.layout_manager.resize_profile_dock_default()

    def _ensure_profile_marker(self):
        position = self.win.img_view.profileMarkerPosition()
        if position is None:
            x, y = self._default_profile_marker_position()
            self.win.img_view.setProfileMarker(x, y, visible=True)
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
        if not hasattr(self.win, "profile_dock"):
            return None
        try:
            image_levels = self.win.img_view.getLevels()
        except Exception as exc:
            handle_ui_exception("profile y range levels", exc)
            image_levels = None
        return profile_y_range(self.win.profile_dock.y_range_mode(), image_levels)

    def _phase_colormap(self):
        return phase_colormap()

    def _apply_channel_colormap(self):
        name = resolved_colormap_name(
            self.win.view_state.channel,
            getattr(self.win, "current_colormap", None),
            user_selected=bool(getattr(self.win, "_colormap_user_selected", False)),
        )
        self._set_display_colormap(
            name,
            user_selected=bool(getattr(self.win, "_colormap_user_selected", False)),
            request_render=False,
        )

    def _set_display_colormap(self, name, *, user_selected: bool, request_render: bool) -> str:
        """Apply one named LUT to the colorbar and every rendering strategy."""

        colormap = named_colormap(str(name))
        if colormap is None:
            raise ValueError(f"unknown colormap: {name}")
        key_getter = getattr(self.win.img_view, "displayColorMapKey", None)
        previous_key = key_getter() if callable(key_getter) else None
        self.win.img_view.setColorMap(colormap)
        self.win.current_colormap = str(name)
        self.win._colormap_user_selected = bool(user_selected)
        current_key = key_getter() if callable(key_getter) else None

        # Shader-backed paths update uniforms in setColorMap.  The CPU complex
        # path bakes the LUT into RGB pixels and therefore needs a new
        # materialization/cache key.
        if request_render and previous_key != current_key and self._evaluation_colormap_lut() is not None:
            self.render(reason="colormap")
        return self.win.current_colormap

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

    def _current_window_mode(self):
        if self.win.widgets['buttons']['display']['window_absolute'].isChecked():
            return "absolute"
        return "relative"

    def update_display_mode(self):
        """Update the display mode for the image view"""
        if self.win.widgets['buttons']['display']['square_pixels'].isChecked():
            self.win.img_view.setDisplayMode('square_pixels')
        elif self.win.widgets['buttons']['display']['fit'].isChecked():
            self.win.img_view.setDisplayMode('fit')

    def _on_aspect_toolbar_changed(self, mode):
        if mode == "one_to_one":
            self.one_to_one_image()
            return
        self.fit_image_to_view()

    def _on_window_mode_changed(self, mode):
        self.win.widgets['buttons']['display']['window_relative'].setChecked(mode != "absolute")
        self.win.widgets['buttons']['display']['window_absolute'].setChecked(mode == "absolute")
        self.render(reason="window-mode")

    def _set_live_profile_checked(self, enabled):
        self.win.widgets['buttons']['display']['live_profile'].setChecked(bool(enabled))

    def fit_image_to_view(self, enabled=True):
        self.win.widgets['buttons']['display']['fit'].setChecked(bool(enabled))
        if hasattr(self.win, "display_toolbar"):
            blocker = Qt.QtCore.QSignalBlocker(self.win.display_toolbar.fit_action)
            try:
                self.win.display_toolbar.fit_action.setChecked(bool(enabled))
            finally:
                blocker.unblock()
        self.win.img_view.setFitLocked(bool(enabled))
        self.win._sync_controls_from_view_state()

    def one_to_one_image(self):
        self.win.widgets['buttons']['display']['square_pixels'].setChecked(True)
        if hasattr(self.win, "display_toolbar"):
            blocker = Qt.QtCore.QSignalBlocker(self.win.display_toolbar.fit_action)
            try:
                self.win.display_toolbar.fit_action.setChecked(False)
            finally:
                blocker.unblock()
        self.win.img_view.oneToOne()
        self.win._sync_controls_from_view_state()

    def auto_window_levels(self):
        cancel_level_interaction = getattr(self.win.img_view, "cancelHistogramLevelInteraction", None)
        if callable(cancel_level_interaction):
            cancel_level_interaction()
        previous_levels = normalize_bounds(self.win.img_view.getLevels())
        auto_bounds = normalize_bounds(self.win.img_view.getHistogramDataBounds())
        has_committed_target = bool(
            getattr(self.win, "_committed_display_frame", None) is not None
            or bool(getattr(getattr(self, "_montage_session", None), "display_committed", False))
        )
        auto_source = self._apply_display_level_override(
            auto_bounds,
            histogram_range=auto_bounds,
            emit_user=False,
            source_rank=LevelSourceRank.PREVIOUS_COMMITTED,
        )
        if has_committed_target and auto_source is not None:
            self._pending_auto_level_source = None
            self.win._force_autolevel = False
        else:
            self._pending_auto_level_source = auto_source
            self.win._force_autolevel = True
            self.render(reason="auto-window", force_autolevel=True)
            if getattr(self, "_pending_auto_level_source", None) is auto_source:
                self._pending_auto_level_source = None
        if previous_levels is not None:
            show_revert_action(
                self.win,
                "Auto window levels applied.",
                lambda levels=previous_levels: self._revert_auto_window_levels(levels),
                timeout=5000,
            )

    def _revert_auto_window_levels(self, levels) -> None:
        levels = normalize_bounds(levels)
        if levels is None:
            return
        cancel_level_interaction = getattr(self.win.img_view, "cancelHistogramLevelInteraction", None)
        if callable(cancel_level_interaction):
            cancel_level_interaction()
        self.win._force_autolevel = False
        self._pending_auto_level_source = None
        session = getattr(self, "_montage_session", None)
        has_committed_target = bool(
            getattr(self.win, "_committed_display_frame", None) is not None
            or bool(getattr(session, "display_committed", False))
        )
        if has_committed_target:
            self._apply_display_level_override(
                levels,
                emit_user=True,
                source_rank=LevelSourceRank.EXPLICIT_USER,
            )
            if session is not None:
                session.force_auto = False
            return
        self._queue_display_levels(levels)
        if session is not None:
            session.force_auto = False
            session.user_levels_override = levels
        self.win.img_view.setLevels(levels[0], levels[1])
        self.render(reason="auto-window-revert", force_autolevel=False)

    def toggle_profile_dock(self):
        visible = not self.win.profile_dock.isVisible()
        self.win.layout_manager.set_profile_dock_visible_from_user(visible)

    def _processing_pressed(self, btn):
        """Called on processing button press; if the button is already checked
        the user is re-clicking it and we should force an auto-level on next update."""
        try:
            if btn.isChecked():
                self.win._force_autolevel = True
            else:
                self.win._force_autolevel = False
        except Exception as exc:
            handle_ui_exception("processing button", exc)
            self.win._force_autolevel = False

        # Update the display group title
        self._update_display_group_title()

        # Force update to ensure view changes immediately
        self.render(reason="processing-pressed", force_autolevel=True)

    def _update_display_group_title(self):
        """Update the display group title with aspect ratio information."""
        mode = self.win.img_view.displayMode
        aspect_str = ''

        if mode == 'square_pixels': # Simple
            self.win.display_group.setTitle('Display (1:1)')
            return

        if mode == 'fit': #use the viewport aspect ratio
            aspect_str = ''
            try:
                if hasattr(self.win.img_view, 'image') and self.win.img_view.image is not None:
                    view = self.win.img_view.getView()

                    img_height, img_width = self.win.img_view.image.shape[:2]
                    widget_ratio = view.size().width() / view.size().height()
                    img_ratio = img_width / img_height
                    ratio = img_ratio * widget_ratio

                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
            finally:
                self.win.display_group.setTitle(f'Display {aspect_str}')

        else:
            self.win.display_group.setTitle('Display')

    def update_line_plot(self):
        if not hasattr(self.win, "profile_dock") or not self.win.profile_dock.isVisible():
            return
        if self.win.widgets['buttons']['display']['live_profile'].isChecked():
            position = self.win.img_view.profileMarkerPosition()
            if position is not None:
                self._on_profile_marker_moved(*position)
                return
        view_state = self.win.view_state
        y_range = self._current_profile_y_range()
        document = self.win.document
        profile_axes = tuple(getattr(self.win, "profile_axes", ()) or ((view_state.line_axis,) if view_state.line_axis is not None else ()))
        profile_states = tuple(view_state.with_line_axis(axis) for axis in profile_axes)
        cached_entries = []
        for profile_state in profile_states:
            cached = self.win.operation_evaluator.cached_line(profile_state)
            if cached is None:
                cached_entries = []
                break
            cached_entries.append((cached, profile_state, f"dim {profile_state.line_axis}"))
        if cached_entries:
            self.win.profile_dock.update_line_results(tuple(cached_entries), y_range=y_range)
            self.win._update_operation_dock()
            return

        def evaluate():
            eval_context = self.win._evaluation_context(ComputeLane.PROFILE, None)
            return tuple(
                (
                    profile_state,
                    evaluate_line_snapshot(
                        document,
                        profile_state,
                        stage_cache=self.win.operation_evaluator.stage_cache,
                        stage_document_key=stage_document_key(document),
                        evaluation_context=eval_context,
                    ),
                )
                for profile_state in profile_states
            )

        request_keys = {profile_state: self.win.operation_evaluator.line_key(profile_state, document=document) for profile_state in profile_states}

        def done(results):
            entries = []
            for profile_state, result in results:
                if request_keys[profile_state] != self.win.operation_evaluator.line_key(profile_state):
                    return
                line_result = self.win.operation_evaluator.store_line_result(profile_state, result)
                entries.append((line_result, profile_state, f"dim {profile_state.line_axis}"))
            self.win.profile_dock.update_line_results(tuple(entries), y_range=y_range)
            self.win._update_operation_dock()

        self.win.profile_evaluation_controller.start_latest(
            evaluate,
            key=tuple(request_keys.values()),
            priority=EvalPriority.LIVE_PROFILE,
            replace_group="profile-plot",
            frame_target=FrameTarget(
                semantic_key=tuple(request_keys.values()),
                viewport_key=("profile-panel", tuple(profile_axes)),
                presentation_key=("profile-plot",),
                quality="exact-visible",
            ),
            supersession_key="profile-plot",
            supersession_value=tuple(request_keys.values()),
            work_item=WorkItem(
                key=("profile_plot", tuple(request_keys.values())),
                lane=WorkLane.PROFILE_ROI_HOVER,
                frame_target=FrameTarget(
                    semantic_key=tuple(request_keys.values()),
                    viewport_key=("profile-panel", tuple(profile_axes)),
                    presentation_key=("profile-plot",),
                    quality="exact-visible",
                ),
                supersession_key="profile-plot",
                supersession_value=tuple(request_keys.values()),
                estimated_bytes=int(self._prefetch_budget_bytes()),
                expected_value=1.5,
                reusable_output=True,
            ),
            on_done=done,
            on_error=lambda exc: show_status_message(self.win, f"Profile update failed: {exc}"),
        )

    def _on_view_range_changed(self, *_args):
        """Update display group title when view range changes (for fit mode)."""
        self._viewport_bridge().on_view_range_changed()

    def on_tab_changed(self, index):
        """Handle central image tab changes."""
        self.win.update_dimension_controls()
        self.win.update()
        self.win.line_plot.hide_crosshair()

    def is_line_plot_mode(self):
        """Line-plot mode is disabled while the 2D viewer owns the primary surface."""
        return False

    def request_render(self, *, reason: str, force_autolevel: bool = False, interactive: bool = False) -> None:
        self._advance_render_generation(f"request:{reason}")
        coordinator = getattr(self.win, "render_coordinator", None)
        if coordinator is None:
            self.render(reason=reason, force_autolevel=force_autolevel)
            return
        coordinator.request(reason=reason, force_autolevel=force_autolevel, interactive=interactive)

    def _advance_render_generation(self, reason: str) -> int:
        generation = getattr(self.win, "_render_generation", None)
        if generation is None:
            return 0
        return generation.advance(reason)

    def _capture_render_generation(self) -> int:
        generation = getattr(self.win, "_render_generation", None)
        return 0 if generation is None else generation.capture()

    def _is_current_render_generation(self, generation: int) -> bool:
        guard = getattr(self.win, "_render_generation", None)
        return (guard is None or guard.is_current(generation)) and not getattr(self.win, "_closing", False)

    def _cancel_render_dependent_work_for_interactive_change(self) -> None:
        for controller_name, groups in (
            ("histogram_evaluation_controller", ("histogram-plot",)),
            ("profile_evaluation_controller", ("profile-plot", "live-profile")),
            ("roi_evaluation_controller", ("roi-inspection",)),
            ("pixel_evaluation_controller", ("pixel",)),
        ):
            controller = getattr(self.win, controller_name, None)
            if controller is None:
                continue
            for group in groups:
                controller.clear_group(group)
        prefetch = getattr(self.win, "prefetch_evaluation_controller", None)
        if prefetch is not None:
            prefetch.cancel_prefetch()

    def _run_deferred_side_panel_refresh(self, *, reason: str) -> None:
        self._last_deferred_side_panel_refresh_reason = str(reason)
        if getattr(self.win, "_closing", False) or not hasattr(self.win, "widgets"):
            return
        try:
            if self.win.profile_dock.isVisible() or self.win.widgets['buttons']['display']['live_profile'].isChecked():
                self.update_line_plot()
            self.win._update_operation_dock()
            self.win._sync_progressive_docks()
            self.win._refresh_inspection_dock()
            restore_viewport_shape = getattr(self.win, "_restore_viewport_continuity_shape_after_layout", None)
            restore_shape = getattr(self.win, "_viewport_continuity_shape_target", lambda: None)
            if callable(restore_viewport_shape) and callable(restore_shape) and restore_shape() is not None:
                restore_viewport_shape()
        finally:
            self.win._deferred_side_panel_refresh_pending = False

    def render(self, *, reason: str = "state", force_autolevel: bool = False, defer_side_panels: bool = False):
        render_start = perf_counter()
        self._cancel_render_dependent_work_for_interactive_change()
        self._advance_render_generation(f"render:{reason}")
        self.win._set_view_state(self.win.view_state.for_shape(self.win.data.shape, preserve_flags=True))
        self.win._coerce_channel_for_current_dtype()
        control_start = perf_counter()
        if self._interactive_slice_controls_are_current(reason=reason, defer_side_panels=defer_side_panels):
            if hasattr(self.win, "tab_widget"):
                self.win.tab_widget.setVisible(self.win.data.ndim >= 2)
        else:
            self.win._sync_controls_from_view_state()
            if hasattr(self.win, "tab_widget"):
                self.win.tab_widget.setVisible(self.win.data.ndim >= 2)
            self.win._update_channel_controls()
            self.win.update_dimension_controls()
            self.win.update_complex_indicators()
            self.win.update_shift_indicators()
            self.win._interactive_slice_controls_synced_state = None
        self._last_control_sync_ms = (perf_counter() - control_start) * 1000.0
        self.update_image_view(force_autolevel=force_autolevel, defer_side_panels=defer_side_panels)
        if defer_side_panels:
            self.win._deferred_side_panel_refresh_pending = True
        else:
            if self.win.profile_dock.isVisible() or self.win.widgets['buttons']['display']['live_profile'].isChecked():
                self.update_line_plot()
            self.win._update_operation_dock()
            self.win._sync_progressive_docks()
            self.win._deferred_side_panel_refresh_pending = False
        self._last_render_sync_ms = (perf_counter() - render_start) * 1000.0

    def _interactive_slice_controls_are_current(self, *, reason: str, defer_side_panels: bool) -> bool:
        if not bool(defer_side_panels):
            return False
        if str(reason) != "slice":
            return False
        synced_state = getattr(self.win, "_interactive_slice_controls_synced_state", None)
        return synced_state == self.win.view_state
