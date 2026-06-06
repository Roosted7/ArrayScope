from __future__ import annotations

import math

import numpy as np

import pyqtgraph.Qt as Qt

from arrayscope.display.colormaps import gray_colormap, phase_colormap
from arrayscope.profiles.model import clamp_marker_position, image_hover_indices, profile_y_range
from arrayscope.display.slice_engine import apply_channel
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.ui.toasts import show_status_message


def getNumberOfDecimalPlaces(number):
    if isinstance(number, (int, np.integer)):
        return int(0)
    else:
        return int(max(1, (number.as_integer_ratio()[1]).bit_length()))


class RenderMixin:
    def getPixel(self, pos):
        img = self.img_view.image
        if img is None or self.view_state.image_axes is None:
            return
        container = self.img_view.getView()
        if container.sceneBoundingRect().contains(pos): 
            mousePoint = container.mapSceneToView(pos) 
            hover = image_hover_indices(self.view_state, math.floor(mousePoint.x()), math.floor(mousePoint.y()))
            if hover is not None:
                x_i, y_i = hover
                primary_axis, secondary_axis = self.view_state.image_axes
                index = list(self.view_state.slice_indices)
                index[primary_axis] = y_i
                index[secondary_axis] = x_i
                value = apply_channel(self.operation_evaluator.current_data()[tuple(index)], self.view_state.channel)
                decimal_places = getNumberOfDecimalPlaces(abs(value))
                if decimal_places > 5:
                    value_text = "({}, {}) = {:.3e}".format (x_i, y_i, value)
                else:
                    value_text = "({}, {}) = {:.{}f}".format (x_i, y_i, value, decimal_places)
                context = self._slice_context_text()
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
                return
        if hasattr(self, "img_view"):
            self.img_view.hideHud()

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
        clamped = clamp_marker_position(self.view_state.shape, self.view_state.image_axes, image_x, image_y)
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

        try:
            profile_render = self.profile_coordinator.render_from_marker(
                self.operation_evaluator,
                self.view_state,
                point[0],
                point[1],
                line_axis=self.view_state.line_axis,
                y_range_mode=self.profile_dock.y_range_mode(),
                image_levels=self.img_view.getLevels(),
            )
            if profile_render is None:
                self._clear_live_profile_marker()
                return
            self.profile_dock.update_line_result(profile_render.line_result, profile_render.view_state, y_range=profile_render.y_range)
            self._update_operation_dock()
            self.profile_dock.show()
            self.img_view.setProfileMarker(round(point[0]), round(point[1]), visible=True)
        except Exception as e:
            show_status_message(self, f"Live profile update failed: {e}")
            self._clear_live_profile_marker()

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
        if not hasattr(self, "profile_dock") or not self.profile_dock.isVisible() or self.profile_dock.isFloating():
            return
        target_height = max(140, int(self.height() * 0.23))
        try:
            self.resizeDocks([self.profile_dock], [target_height], Qt.QtCore.Qt.Orientation.Vertical)
        except Exception:
            pass

    def _ensure_profile_marker(self):
        position = self.img_view.profileMarkerPosition()
        if position is None:
            x, y = self._default_profile_marker_position()
            self.img_view.setProfileMarker(x, y, visible=True)
            self._on_profile_marker_moved(x, y)
        else:
            self._on_profile_marker_moved(*position)

    def _default_profile_marker_position(self):
        if self.view_state.image_axes is None:
            return (0, 0)
        primary_axis, secondary_axis = self.view_state.image_axes
        x = (self.view_state.shape[secondary_axis] - 1) / 2.0
        y = (self.view_state.shape[primary_axis] - 1) / 2.0
        return (round(x), round(y))

    def _current_profile_y_range(self):
        if not hasattr(self, "profile_dock"):
            return None
        try:
            image_levels = self.img_view.getLevels()
        except Exception:
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
            except Exception:
                previous_levels = None
                previous_bounds = None

        
        try:
            colormap_lut = None
            if self.view_state.channel == ChannelMode.COMPLEX:
                colormap_lut = self._phase_colormap().getLookupTable(0.0, 1.0, 256, alpha=False)
            display_image = self.operation_evaluator.image(self.view_state, colormap_lut=colormap_lut)
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
                )
            else:
                self.img_view.setImage(display_image.data, autoLevels=al, levels=levels)
            self._update_operation_dock()
            
            # Apply axis flips after setting the image
            self.apply_axis_flips()
            
        except Exception as e:
            show_status_message(self, f"Image update failed: {e}")

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
        except Exception:
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
        except Exception:
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
        line_result = self.operation_evaluator.line(self.view_state)
        self.profile_dock.update_line_result(line_result, self.view_state, y_range=self._current_profile_y_range())
        self._update_operation_dock()

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
        self._update_operation_dock()
        self._sync_progressive_docks()
