from __future__ import annotations

from dataclasses import replace

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.slice_selection import center_index
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.model.frame import CanvasValueSource
from arrayscope.ui.icons import clear_label_icon, set_label_icon
from arrayscope.ui.shortcuts import colormap_name_for_key
from arrayscope.ui.toasts import show_status_message


class DimensionControlMixin:
    def update_shift_indicators(self):
        for i, shift_label in enumerate(self.widgets['labels']['shift']):
            if i >= self.data.ndim:
                shift_label.setText('')
                shift_label.setToolTip('')
                shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
                continue
            if self.singleton[i]:
                shift_label.setText('')
                shift_label.setToolTip('')
                shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
                continue

            shift_label.setText('')
            shift_label.setToolTip('')
            shift_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))
    
    def flipAxisClicked(self, event, dim):
        """Handle click on flip axis icon"""
        image_axes = self._image_axes()
        if dim not in image_axes and dim != self.view_state.line_axis:
            return
        self._set_view_state(self.view_state.with_axis_flipped(dim, not self._axis_flipped(dim)))
        self.update_flip_icons()
        self.apply_axis_flips()
        
    def update_flip_icons(self):
        image_axes = self._image_axes()
        for i, flip_label in enumerate(self.widgets['labels']['flip']):
            if i in image_axes:
                # In line plot mode, only show horizontal flip icon for the plot dimension
                if self.is_line_plot_mode():
                    if i == self.view_state.line_axis:
                        flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                        flip_label.setToolTip("Flip X axis")
                        if self._axis_flipped(i):
                            set_label_icon(flip_label, "arrow_back")
                        else:
                            set_label_icon(flip_label, "arrow_forward")
                    else:
                        clear_label_icon(flip_label)  # Hide flip icons for non-plot dimensions
                        flip_label.setToolTip('')
                # In image view mode, show vertical flip for primary, horizontal for secondary
                elif self.view_state.image_axes is not None and i == self.view_state.image_axes[0]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeVerCursor))
                    flip_label.setToolTip("Flip Y")
                    if self._axis_flipped(i):
                        set_label_icon(flip_label, "arrow_downward")
                    else:
                        set_label_icon(flip_label, "arrow_upward")
                elif self.view_state.image_axes is not None and i == self.view_state.image_axes[1]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                    flip_label.setToolTip("Flip X")
                    if self._axis_flipped(i):
                        set_label_icon(flip_label, "arrow_back")
                    else:
                        set_label_icon(flip_label, "arrow_forward")
                else:
                    clear_label_icon(flip_label)  # Clear for dimensions not in primary/secondary
                    flip_label.setToolTip('')
            else:
                clear_label_icon(flip_label)  # Clear for unselected dimensions
                flip_label.setToolTip('')
    
    def apply_axis_flips(self):
        if self.is_line_plot_mode():
            plot_view = self.plot_widget.getViewBox()
            plot_dim = self.view_state.line_axis
            if plot_dim is not None:
                plot_view.invertX(self._axis_flipped(plot_dim))
        else:
            if self.view_state.image_axes is None:
                return
            
            view = self.img_view.getView()
            y_dim, x_dim = self.view_state.image_axes
            view.invertY(not self._axis_flipped(y_dim))
            view.invertX(self._axis_flipped(x_dim))
        self._update_committed_frame_for_axis_flip()

    def _update_committed_frame_for_axis_flip(self):
        frame = getattr(self, "_committed_display_frame", None)
        if frame is None:
            return
        geometry = frame.geometry
        if not _same_view_except_axis_flips(geometry.view_state, self.view_state):
            return
        updated_geometry = DisplayGeometry(
            view_state=self.view_state,
            display_shape=geometry.display_shape,
            montage=geometry.montage,
            montage_origin_x=geometry.montage_origin_x,
            montage_origin_y=geometry.montage_origin_y,
            montage_tile_states=geometry.montage_tile_states,
        )
        value_source = frame.value_source
        if isinstance(value_source, CanvasValueSource):
            value_source = CanvasValueSource(
                data=value_source.data,
                histogram_data=value_source.histogram_data,
                geometry=updated_geometry,
            )
        self._committed_display_frame = replace(frame, geometry=updated_geometry, value_source=value_source, scene=None)

    def update_complex_indicators(self):
        """Initialize or update indicators for dimensions that can be combined as complex."""
        for i in range(self.data.ndim):
            indicator = self.widgets['labels']['complex'][i]
            
            if self.combined_as_complex[i]:
                set_label_icon(indicator, "functions")
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Split to real')
            elif self.can_combine_as_complex[i]:
                set_label_icon(indicator, "data_object")
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Combine as complex')
            else:
                # No indicator, already-complex data or non-size-2 dimensions
                clear_label_icon(indicator)
                indicator.setToolTip('')
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))

    def _update_channel_controls(self):
        """Keep channel options in sync with the current array dtype."""
        is_complex = self._current_is_complex()
        channel_buttons = self.widgets['buttons']['channel']
        enabled_channels = {
            'complex': is_complex,
            'real': True,
            'abs': True,
            'imag': is_complex,
            'angle': is_complex,
        }

        for name, button in channel_buttons.items():
            button.setEnabled(enabled_channels[name])
        if hasattr(self, "display_toolbar"):
            self.display_toolbar.set_channel_options(enabled_channels)

        checked_channel = self.view_state.channel.value
        if checked_channel in channel_buttons:
            channel_buttons[checked_channel].setChecked(True)
        if hasattr(self, "display_toolbar"):
            self.display_toolbar.set_current(channel=checked_channel)
    
    def complexOrRealClicked(self, event, dim):
        if self.can_combine_as_complex[dim] and not self.combined_as_complex[dim]:
            self._append_operation("combine_real_imag", dim)
        elif self.combined_as_complex[dim]:
            self._append_operation("split_complex", dim)
    
    def combineAsComplex(self, dim):
        """Combine a size-2 real dimension into complex as an operation."""
        if not self.can_combine_as_complex[dim] or self.combined_as_complex[dim]:
            return

        self._append_operation("combine_real_imag", dim)
    
    def splitToReal(self, dim):
        """Split a singleton complex dimension back to real/imag as an operation."""
        if not self.combined_as_complex[dim]:
            return

        self._append_operation("split_complex", dim)
    
    def set_profile_axis(self, axis):
        self.set_profile_axes_exactly((axis,))
        self.render(reason="profile-axis")

    def set_profile_axes_exactly(self, axes):
        axes = tuple(int(axis) for axis in axes)
        axes = tuple(axis for axis in axes if 0 <= axis < self.data.ndim and not self.singleton[axis])
        if not axes and self.view_state.line_axis is not None:
            axes = (self.view_state.line_axis,)
        if not axes:
            return
        self.profile_axes = axes
        self._set_view_state(self.view_state.with_line_axis(axes[0]))
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        return self.view_state

    def set_dimension_role(self, role, axis):
        if axis >= self.data.ndim:
            return
        if role == "p":
            current = list(getattr(self, "profile_axes", ()))
            axis = int(axis)
            if axis in current and len(current) > 1:
                current.remove(axis)
            elif axis not in current:
                current.append(axis)
            else:
                current = [axis]
            self.profile_axes = tuple(current)
            self._set_view_state(self.view_state.with_line_axis(self.profile_axes[0]))
            if hasattr(self, "profile_dock"):
                self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        elif role in ("y", "x"):
            if self.view_state.image_axes is None:
                return
            axis = int(axis)
            role_index = 0 if role == "y" else 1
            if self.view_state.image_axes[role_index] == axis:
                self._set_view_state(self.view_state.with_axis_flipped(axis, not self._axis_flipped(axis)))
                self.update_flip_icons()
                if hasattr(self, "dimension_strip"):
                    self.dimension_strip.update_axis_state(axis, self.data.shape, self.view_state, self.profile_axes)
                self.apply_axis_flips()
                return
            previous = self.view_state
            state = previous
            if previous.montage_axis == axis:
                state = state.with_montage_axis(None)
                if previous.montage_indices:
                    state = state.with_axis_range(axis, indices=previous.montage_indices, text=previous.montage_text)
            state = state.with_image_axis(role, axis)
            new_image_axes = set(state.image_axes or ())
            demoted_axes = tuple(old_axis for old_axis in tuple(previous.image_axes or ()) if old_axis not in new_image_axes)
            for demoted_axis in demoted_axes:
                range_indices = previous.axis_range_indices[demoted_axis]
                range_text = previous.axis_range_text[demoted_axis]
                state = state.with_axis_range(demoted_axis, None)
                if range_indices is not None and state.montage_axis is None:
                    state = state.with_montage_axis(demoted_axis, indices=range_indices, text=range_text)
                else:
                    state = state.with_slice(demoted_axis, center_index(self.data.shape[demoted_axis]))
                    if range_indices is not None:
                        show_status_message(self, "Could not preserve the previous image-axis crop as a montage.", timeout=2000)
            self._set_view_state(state)
        elif role == "m":
            if self.view_state.image_axes is None or int(axis) in self.view_state.image_axes:
                return
            if self.view_state.montage_axis == int(axis):
                self._set_view_state(self.view_state.with_montage_axis(None))
            else:
                self._set_view_state(self.view_state.with_montage_axis(int(axis)))
        self.render(reason=f"dimension-{role}")

    def transposeView(self, event):
        if self.view_state.image_axes is None:
            return
        self._set_view_state(self.view_state.transposed_image_axes())
        self.render(reason="transpose")

    def update(self):
        self.render(reason="legacy-update")

    def update_dimension_controls(self):
        """Update button and spinbox states based on current mode"""
        if self.is_line_plot_mode():
            # Line plot mode: single dimension selection, all other spinboxes enabled
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                bProfile = self.widgets['buttons']['profile'][i]
                if i >= self.data.ndim:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(False)
                    continue
                
                if self.singleton[i]:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(False)
                elif i == self.line_plot_dimension:
                    # This is the dimension we're plotting along
                    w.setEnabled(False)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(True)
                    bPrim.setChecked(True)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(i in self.profile_axes)
                else:
                    # All other dimensions: enable spinbox to select slice
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(True)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(i in self.profile_axes)
        else:
            image_axes = self._image_axes()
            montage_axis = self.view_state.montage_axis
            for i, w in enumerate(self.widgets['spins']['slice_indices']):
                bPrim = self.widgets['buttons']['primary'][i]
                bSecondary = self.widgets['buttons']['secondary'][i]
                bProfile = self.widgets['buttons']['profile'][i]
                if i >= self.data.ndim:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(False)
                    continue
                if self.singleton[i] == True:
                    w.setEnabled(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(False)
                    bProfile.setChecked(False)
                elif i in image_axes or i == montage_axis:
                    w.setEnabled(False)
                    if self.view_state.image_axes is not None and i == self.view_state.image_axes[0]:
                        bPrim.setChecked(True)
                        bSecondary.setChecked(False)
                    elif self.view_state.image_axes is not None and i == self.view_state.image_axes[1]:
                        bPrim.setChecked(False)
                        bSecondary.setChecked(True)
                    else:
                        bPrim.setChecked(False)
                        bSecondary.setChecked(False)
                    bPrim.setEnabled(False)
                    bSecondary.setEnabled(False)
                    bProfile.setEnabled(True)
                    bProfile.setChecked(i in self.profile_axes)
                else:
                    w.setEnabled(True)
                    bPrim.setEnabled(True)
                    bSecondary.setEnabled(True)
                    bProfile.setEnabled(True)
                    bPrim.setChecked(False)
                    bSecondary.setChecked(False)
                    bProfile.setChecked(i in self.profile_axes)
                    
            if self.view_state.image_axes is not None:
                self.widgets['buttons']['primary'][self.view_state.image_axes[0]].setChecked(True)
                self.widgets['buttons']['secondary'][self.view_state.image_axes[1]].setChecked(True)
        
        self.update_flip_icons()
        self.update_shift_indicators()
        if hasattr(self, "dimension_strip"):
            self.dimension_strip.update_state(self.data.shape, self.view_state, self.profile_axes)
            for container in getattr(self, "dim_containers", []):
                container.hide()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        key = event.key()
        modifiers = event.modifiers()
        
        # Check for 'T' key to transpose view (swap X and Y dimensions)
        if key == Qt.QtCore.Qt.Key.Key_T and modifiers == Qt.QtCore.Qt.KeyboardModifier.NoModifier:
            if not self.is_line_plot_mode() and self.view_state.image_axes is not None:
                self.transposeView(event)
                event.accept()
                return

        if key == Qt.QtCore.Qt.Key.Key_K and modifiers == Qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            self.open_command_palette()
            event.accept()
            return

        if modifiers == Qt.QtCore.Qt.KeyboardModifier.NoModifier:
            if key == Qt.QtCore.Qt.Key.Key_F:
                self.fit_image_to_view()
                event.accept()
                return
            if key == Qt.QtCore.Qt.Key.Key_1:
                self.one_to_one_image()
                event.accept()
                return
            if key == Qt.QtCore.Qt.Key.Key_A:
                self.auto_window_levels()
                event.accept()
                return
            if key == Qt.QtCore.Qt.Key.Key_P:
                self.toggle_profile_dock()
                event.accept()
                return
            if key == Qt.QtCore.Qt.Key.Key_L:
                live = self.widgets['buttons']['display']['live_profile']
                live.setChecked(not live.isChecked())
                event.accept()
                return
            if key in (Qt.QtCore.Qt.Key.Key_BracketLeft, Qt.QtCore.Qt.Key.Key_BracketRight):
                self.step_active_slice(-1 if key == Qt.QtCore.Qt.Key.Key_BracketLeft else 1)
                event.accept()
                return

        if modifiers == Qt.QtCore.Qt.KeyboardModifier.ShiftModifier:
            if key in (Qt.QtCore.Qt.Key.Key_BracketLeft, Qt.QtCore.Qt.Key.Key_BracketRight):
                self.step_active_slice(-10 if key == Qt.QtCore.Qt.Key.Key_BracketLeft else 10)
                event.accept()
                return
        
        # Check CTRL+number for colormap changes
        if modifiers == Qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            colormap_name = colormap_name_for_key(key)
            if colormap_name is not None:
                self.setColormap(colormap_name)
                event.accept()
                return
                
        # Pass event to parent if not handled
        super().keyPressEvent(event)

    def step_active_slice(self, delta):
        axis = getattr(self, "_active_slice_axis", None)
        if axis is None or axis in self.view_state.display_axes() or self.data.shape[axis] == 1:
            for candidate in self.view_state.non_display_axes():
                if self.data.shape[candidate] > 1:
                    axis = candidate
                    break
        if axis is None and self.view_state.line_axis is not None:
            axis = self.view_state.line_axis
        if axis is None or axis >= self.data.ndim or self.data.shape[axis] <= 1:
            self.statusBar().showMessage("No slice axis available", 2500)
            return
        current = self.view_state.slice_indices[axis]
        new_value = max(0, min(self.data.shape[axis] - 1, current + int(delta)))
        self._on_slice_index_changed(axis, new_value)
    
    def setColormap(self, colormap_name):
        """Set the colormap for the image view"""
        try:
            self._set_display_colormap(colormap_name, user_selected=True, request_render=True)
        except Exception as e:
            handle_ui_exception("set colormap", e)
            self.statusBar().showMessage(f"Failed to set colormap {colormap_name}: {e}", 3000)
    
    def eventFilter(self, obj, event):
        if (
            hasattr(self, "img_view")
            and obj is self.img_view.graphicsView.viewport()
            and event.type() == Qt.QtCore.QEvent.Type.Leave
        ):
            self._clear_image_hover_state()
        if obj == self.tab_widget.tabBar():
            if event.type() == Qt.QtCore.QEvent.Type.MouseButtonDblClick:
                self.profile_dock.toggle_style()
                event.accept()
                return True
        return super().eventFilter(obj, event)


def _same_view_except_axis_flips(previous, current) -> bool:
    if previous is None or current is None:
        return False
    return (
        getattr(previous, "ndim", None) == getattr(current, "ndim", None)
        and tuple(getattr(previous, "shape", ())) == tuple(getattr(current, "shape", ()))
        and getattr(previous, "image_axes", None) == getattr(current, "image_axes", None)
        and getattr(previous, "line_axis", None) == getattr(current, "line_axis", None)
        and tuple(getattr(previous, "slice_indices", ())) == tuple(getattr(current, "slice_indices", ()))
        and getattr(previous, "channel", None) == getattr(current, "channel", None)
        and getattr(previous, "scale", None) == getattr(current, "scale", None)
        and tuple(getattr(previous, "axis_fftshifted", ())) == tuple(getattr(current, "axis_fftshifted", ()))
        and getattr(previous, "montage_axis", None) == getattr(current, "montage_axis", None)
        and getattr(previous, "montage_columns", None) == getattr(current, "montage_columns", None)
        and getattr(previous, "montage_indices", None) == getattr(current, "montage_indices", None)
        and getattr(previous, "montage_text", None) == getattr(current, "montage_text", None)
        and tuple(getattr(previous, "axis_range_indices", ())) == tuple(getattr(current, "axis_range_indices", ()))
        and tuple(getattr(previous, "axis_range_text", ())) == tuple(getattr(current, "axis_range_text", ()))
        and tuple(getattr(previous, "axis_flipped", ())) != tuple(getattr(current, "axis_flipped", ()))
    )
