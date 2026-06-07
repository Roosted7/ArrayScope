from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import pyqtgraph.Qt as Qt

from arrayscope.app.errors import handle_ui_exception
from arrayscope.display.colormaps import gray_colormap, phase_colormap
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.montage import make_montage, optimal_montage_columns
from arrayscope.display.slice_engine import DisplayImage
from arrayscope.display.viewport import ViewportPolicy
from arrayscope.operations.evaluator import _document_key
from arrayscope.profiles.model import profile_y_range
from arrayscope.core.cache_status import CacheStatus, CacheStatusSnapshot
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.operations.evaluator import evaluate_image_snapshot, evaluate_line_snapshot, evaluate_scalar_snapshot
from arrayscope.ui.toasts import show_status_message


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
        img = self.img_view.image
        if img is None or self.view_state.image_axes is None:
            label = self.widgets['labels']['pixelValue']
            if hasattr(label, "set_pixel_status"):
                label.set_pixel_status("Pixel value updating...", self._slice_context_text())
            else:
                label.setText("Pixel value updating...")
            if hasattr(self, "img_view"):
                self.img_view.showHudText("Pixel value updating...", pos)
            return
        container = self.img_view.getView()
        if container.sceneBoundingRect().contains(pos): 
            mousePoint = container.mapSceneToView(pos) 
            geometry = getattr(self, "display_geometry", None)
            mapping = None if geometry is None else geometry.display_point_to_array_index(math.floor(mousePoint.x()), math.floor(mousePoint.y()))
            if mapping is not None:
                x_i, y_i = mapping.local_x, mapping.local_y
                index = mapping.array_index
                context = self._slice_context_text()
                if mapping.montage_axis is not None and mapping.montage_index is not None:
                    context = f"{context} d{mapping.montage_axis}={mapping.montage_index}".strip()
                value_text = f"({x_i}, {y_i}) = updating..."
                text = value_text
                if context:
                    text = f"{value_text} | {context}"
                label = self.widgets['labels']['pixelValue']
                if hasattr(label, "set_pixel_status"):
                    label.set_pixel_status(value_text, context)
                else:
                    label.setText(text)
                if hasattr(self, "img_view"):
                    self.img_view.showHudText(text, pos)
                show_status_message(self, text, timeout=1500)
                self._request_pixel_value(tuple(index), x_i, y_i, context, pos)
                return
        if hasattr(self, "img_view"):
            self.img_view.hideHud()

    def _request_pixel_value(self, index, x_i, y_i, context, pos):
        view_state = self.view_state
        document = self.document

        def done(value):
            try:
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

        cached = self.operation_evaluator.cached_scalar(view_state, index)
        if cached is not None:
            done(cached)
            return

        def evaluate():
            return evaluate_scalar_snapshot(document, view_state, index)

        document_key = self.operation_evaluator.scalar_key(view_state, index, document=document)[1]

        def done_result(result):
            if document_key != self.operation_evaluator.scalar_key(view_state, index)[1]:
                return
            done(self.operation_evaluator.store_scalar_result(view_state, index, result))

        self.pixel_evaluation_controller.start(evaluate, on_done=done_result, on_error=lambda _exc: None, slow_ms=0)

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
        image_levels = self.img_view.getLevels()
        y_range_mode = self.profile_dock.y_range_mode()
        profile_axes = tuple(getattr(self, "profile_axes", ()) or ((self.view_state.line_axis,) if self.view_state.line_axis is not None else ()))
        clamped = self._clamp_profile_marker_point(point[0], point[1])
        if clamped is None:
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
            self.profile_dock.update_line_results(tuple(cached_entries), y_range=y_range)
            self._update_operation_dock()
            self.profile_dock.show()
            self.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
            return

        def evaluate():
            return tuple((profile_state, evaluate_line_snapshot(document, profile_state)) for profile_state in profile_states)

        document_keys = {profile_state: self.operation_evaluator.line_key(profile_state, document=document)[1] for profile_state in profile_states}

        def done(results):
            entries = []
            for profile_state, result in results:
                if document_keys[profile_state] != self.operation_evaluator.line_key(profile_state)[1]:
                    return
                line_result = self.operation_evaluator.store_line_result(profile_state, result)
                entries.append((line_result, profile_state, f"dim {profile_state.line_axis}{profile_label_suffix}"))
            self.profile_dock.update_line_results(tuple(entries), y_range=y_range)
            self._update_operation_dock()
            self.profile_dock.show()
            self.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
            if view_state.montage_axis is None:
                for axis in profile_axes:
                    self._prefetch_profiles_near_marker(view_state, point[0], point[1], line_axis=axis)

        def error(exc):
            show_status_message(self, f"Live profile update failed: {exc}")
            self._clear_live_profile_marker()

        self.profile_evaluation_controller.start(evaluate, on_done=done, on_error=error)

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
            self._profile_dock_user_visible = True
            if not self.profile_dock.isVisible():
                if not self.profile_dock.isFloating():
                    self.addDockWidget(Qt.QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.profile_dock)
                else:
                    self.profile_dock.resize(560, 260)
            self.profile_dock.show()
            self.profile_dock.raise_()
            self._schedule_view_geometry_refresh()
            self.img_view.getView().setCursor(Qt.QtCore.Qt.CursorShape.CrossCursor)
            self._ensure_profile_marker()
        if not enabled:
            self._profile_dock_user_visible = False
            self._pending_profile_pos = None
            self._pending_profile_point = None
            self._profile_timer.stop()
            self._clear_live_profile_marker()
            self.img_view.getView().unsetCursor()
            self.update_line_plot()

    def _on_profile_dock_visibility_changed(self, visible):
        self._profile_dock_user_visible = bool(visible)
        if not visible and self.widgets['buttons']['display']['live_profile'].isChecked():
            self.widgets['buttons']['display']['live_profile'].setChecked(False)
        elif visible and not self.profile_dock.isFloating():
            Qt.QtCore.QTimer.singleShot(0, self._resize_profile_dock_default)

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
        if self.view_state.montage_axis is not None:
            return self.update_montage_view(force_autolevel=force_autolevel)
        self._current_montage_geometry = None
            
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

        def evaluate():
            return evaluate_image_snapshot(document, view_state, colormap_lut=colormap_lut)

        def slow():
            self.img_view.setImageStale(True)
            self.img_view.setEvaluationOverlay(True, "Updating view...")
            self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating image view")
            self._update_operation_dock()

        document_key = self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut, document=document)[1]

        def done(result):
            if document_key != self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut)[1]:
                self.img_view.setImageStale(False)
                self.img_view.setEvaluationOverlay(False)
                return
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
            self._prefetch_nearby_slices(view_state, colormap_lut)

        def error(exc):
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            show_status_message(self, f"Image update failed: {exc}")

        self.evaluation_controller.start(
            evaluate,
            on_done=done,
            on_error=error,
            on_stale=lambda: (self.img_view.setImageStale(False), self.img_view.setEvaluationOverlay(False)),
            on_slow=slow,
        )

    def update_montage_view(self, *, force_autolevel: bool = False):
        axis = self.view_state.montage_axis
        if axis is None or self.view_state.image_axes is None or axis in self.view_state.image_axes:
            return
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
        indices = tuple(view_state.montage_indices or tuple(range(int(view_state.shape[axis]))))
        indices = indices[:256]
        viewport_size = self.img_view.graphicsView.viewport().size()
        viewport_shape = (max(1, viewport_size.height()), max(1, viewport_size.width()))

        def evaluate():
            images = []
            histograms = []
            any_histogram = False
            for index in indices:
                tile_state = view_state.with_slice(axis, index).with_montage_axis(None)
                result = evaluate_image_snapshot(document, tile_state, colormap_lut=colormap_lut)
                images.append(result.value.data)
                histograms.append(result.value.histogram_data)
                any_histogram = any_histogram or result.value.histogram_data is not None
            columns = view_state.montage_columns
            if columns is None and images:
                columns = optimal_montage_columns(
                    len(images),
                    images[0].shape[:2],
                    viewport_shape,
                )
            rendered = make_montage(
                images,
                histogram_images=histograms if any_histogram else images,
                columns=columns,
                indices=indices,
            )
            geometry = DisplayGeometry(view_state=view_state, display_shape=rendered.image.data.shape[:2], montage=rendered.geometry)
            return RenderedView(view_state=view_state, document_key=_document_key(document), display_image=rendered.image, geometry=geometry)

        def slow():
            self.img_view.setImageStale(True)
            self.img_view.setEvaluationOverlay(True, "Updating montage...")
            self.operation_evaluator.last_status = CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating montage view")
            self._update_operation_dock()

        generation_key = (_document_key(document), view_state)

        def done(rendered_view):
            if generation_key != (_document_key(self.document), self.view_state):
                self.img_view.setImageStale(False)
                self.img_view.setEvaluationOverlay(False)
                return
            self._current_montage_geometry = rendered_view.geometry.montage
            self._apply_display_image(
                rendered_view.display_image,
                geometry=rendered_view.geometry,
                window_mode=window_mode,
                previous_levels=previous_levels,
                previous_bounds=previous_bounds,
                force_auto=force_auto,
            )

        def error(exc):
            self.img_view.setImageStale(False)
            self.img_view.setEvaluationOverlay(False)
            show_status_message(self, f"Montage update failed: {exc}")

        self.evaluation_controller.start(evaluate, on_done=done, on_error=error, on_slow=slow)

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

    def _prefetch_nearby_slices(self, view_state, colormap_lut):
        axis = getattr(self, "_active_slice_axis", None)
        if axis is None or view_state.image_axes is None or axis in view_state.image_axes:
            return
        document = self.document
        document_key = self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut, document=document)[1]
        size = view_state.shape[axis]
        current = view_state.slice_indices[axis]
        last = getattr(self, "_last_prefetch_slice_index", None)
        direction = 0 if last is None else (1 if current >= last else -1)
        self._last_prefetch_slice_index = current
        deltas = self._prefetch_deltas(direction, max_radius=min(12, max(2, size - 1)))
        scheduled = 0
        for delta in deltas:
            if scheduled >= 16:
                break
            index = current + delta
            if 0 <= index < size:
                prefetch_state = view_state.with_slice(axis, index)
                self.evaluation_controller.start_prefetch(
                    lambda prefetch_state=prefetch_state, document=document: self.operation_evaluator.prefetch_image_snapshot(
                        document,
                        prefetch_state,
                        colormap_lut=colormap_lut,
                    ),
                    on_done=lambda result, prefetch_state=prefetch_state, document=document, document_key=document_key: self._store_prefetch_image_if_current(
                        document,
                        document_key,
                        prefetch_state,
                        colormap_lut,
                        result,
                    ),
                )
                scheduled += 1

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
        document_key_cache = {}
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
                document_key_cache[profile_state] = self.operation_evaluator.line_key(profile_state, document=document)[1]
                self.profile_evaluation_controller.start_prefetch(
                    lambda profile_state=profile_state, document=document: self.operation_evaluator.prefetch_line_snapshot(document, profile_state),
                    on_done=lambda result, profile_state=profile_state, document=document, key=document_key_cache[profile_state]: self._store_prefetch_profile_if_current(
                        document,
                        key,
                        profile_state,
                        result,
                    ),
                )
                scheduled += 1

    def _store_prefetch_profile_if_current(self, document, document_key, profile_state, result):
        if document_key != self.operation_evaluator.line_key(profile_state)[1]:
            return False
        return self.operation_evaluator.store_prefetch_line_result(document, profile_state, result)

    def _store_prefetch_image_if_current(self, document, document_key, view_state, colormap_lut, result):
        current_key = self.operation_evaluator.image_key(view_state, colormap_lut=colormap_lut)[1]
        if document_key != current_key:
            return False
        return self.operation_evaluator.store_prefetch_image_result(document, view_state, colormap_lut, result)

    def _current_window_mode(self):
        if self.widgets['buttons']['display']['window_absolute'].isChecked():
            return "absolute"
        return "relative"

    def _display_histogram_bounds(self, display_image):
        data = display_image.histogram_data
        if data is None:
            data = display_image.data
        try:
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                return (float(np.min(finite_data)), float(np.max(finite_data)))
        except Exception as exc:
            handle_ui_exception("display histogram bounds", exc)
            return None
        return None
    
    def update_display_mode(self):
        """Update the display mode for the image view"""
        if self.widgets['buttons']['display']['square_pixels'].isChecked():
            self.img_view.setDisplayMode('square_pixels')
        elif self.widgets['buttons']['display']['square_fov'].isChecked():
            self.img_view.setDisplayMode('square_fov')
        elif self.widgets['buttons']['display']['fit'].isChecked():
            self.img_view.setDisplayMode('fit')

    def _on_aspect_toolbar_changed(self, mode):
        for name, checked in (
            ("square_pixels", mode == "square_pixels"),
            ("square_fov", mode == "square_fov"),
            ("fit", mode == "fit"),
        ):
            self.widgets['buttons']['display'][name].setChecked(checked)
        self.update_display_mode()
        self.render(reason="aspect")

    def _on_window_mode_changed(self, mode):
        self.widgets['buttons']['display']['window_relative'].setChecked(mode != "absolute")
        self.widgets['buttons']['display']['window_absolute'].setChecked(mode == "absolute")
        self.render(reason="window-mode")

    def _set_live_profile_checked(self, enabled):
        self.widgets['buttons']['display']['live_profile'].setChecked(bool(enabled))

    def fit_image_to_view(self):
        self.widgets['buttons']['display']['fit'].setChecked(True)
        self.img_view.fitToView()
        self._sync_controls_from_view_state()

    def one_to_one_image(self):
        self.widgets['buttons']['display']['square_pixels'].setChecked(True)
        self.img_view.oneToOne()
        self._sync_controls_from_view_state()

    def auto_window_levels(self):
        self._force_autolevel = True
        self.render(reason="auto-window", force_autolevel=True)

    def toggle_profile_dock(self):
        visible = not self.profile_dock.isVisible()
        self._profile_dock_user_visible = visible
        self.profile_dock.setVisible(visible)
        self._sync_progressive_docks()

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
            
        elif mode == 'square_fov':
            # For square FOV, use the image aspect ratio
            if hasattr(self.img_view, 'image') and self.img_view.image is not None:
                shape = self.img_view.image.shape
                if len(shape) >= 2:
                    height, width = shape[:2]
                    ratio = width / height
                    if abs(ratio - 1.0) < 1e-2:
                        aspect_str = '(1:1)'
                    else:
                        aspect_str = f'({ratio:.2f}:1)'
                else:
                    aspect_str = ''
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
            return tuple((profile_state, evaluate_line_snapshot(document, profile_state)) for profile_state in profile_states)

        document_keys = {profile_state: self.operation_evaluator.line_key(profile_state, document=document)[1] for profile_state in profile_states}

        def done(results):
            entries = []
            for profile_state, result in results:
                if document_keys[profile_state] != self.operation_evaluator.line_key(profile_state)[1]:
                    return
                line_result = self.operation_evaluator.store_line_result(profile_state, result)
                entries.append((line_result, profile_state, f"dim {profile_state.line_axis}"))
            self.profile_dock.update_line_results(tuple(entries), y_range=y_range)
            self._update_operation_dock()

        self.profile_evaluation_controller.start(
            evaluate,
            on_done=done,
            on_error=lambda exc: show_status_message(self, f"Profile update failed: {exc}"),
        )

    def _on_view_range_changed(self):
        """Update display group title when view range changes (for fit mode)."""
        self._update_display_group_title()

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
        self._sync_controls_from_view_state()
        if hasattr(self, "tab_widget"):
            self.tab_widget.setVisible(self.data.ndim >= 2)
        self._update_channel_controls()
        self.update_dimension_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_image_view(force_autolevel=force_autolevel)
        self.update_line_plot()
        self._refresh_inspection_dock()
        self._update_operation_dock()
        self._sync_progressive_docks()
