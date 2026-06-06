from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtGui, QtWidgets

from arrayscope.display.colormaps import gray_colormap, named_colormap, phase_colormap
from arrayscope.ui.dialogs import SaveRangeDialog
from arrayscope.operations.recipes import load_recipe, save_recipe
from arrayscope.operations.registry import operation_entries
from arrayscope.profiles.model import clamp_marker_position, image_hover_indices, profile_y_range
from arrayscope.display.slice_engine import apply_channel
from arrayscope.app.settings_state import AppSettingsState, settings_from_mapping, settings_to_mapping
from arrayscope.app.theme import ThemeChoice, apply_theme_to_qapplication
from arrayscope.export.video import VideoExportWorker, VideoExportDialog, VideoExportSettingsDialog
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.core.window_levels import choose_window_levels
from arrayscope.ui.shortcuts import colormap_name_for_key
from arrayscope.window.domain import Domain


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
                            flip_label.setText('⬅️')    
                        else:
                            flip_label.setText('➡️')
                    else:
                        flip_label.setText('')  # Hide flip icons for non-plot dimensions
                        flip_label.setToolTip('')
                # In image view mode, show vertical flip for primary, horizontal for secondary
                elif self.view_state.image_axes is not None and i == self.view_state.image_axes[0]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeVerCursor))
                    flip_label.setToolTip("Flip Y")
                    if self._axis_flipped(i):
                        flip_label.setText('⬇️')
                    else:
                        flip_label.setText('⬆️')
                elif self.view_state.image_axes is not None and i == self.view_state.image_axes[1]:
                    flip_label.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.SizeHorCursor))
                    flip_label.setToolTip("Flip X")
                    if self._axis_flipped(i):
                        flip_label.setText('⬅️')
                    else:
                        flip_label.setText('➡️')
                else:
                    flip_label.setText('')  # Clear for dimensions not in primary/secondary
                    flip_label.setToolTip('')
            else:
                flip_label.setText('')  # Clear for unselected dimensions
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
            view.invertY(self._axis_flipped(y_dim))
            view.invertX(self._axis_flipped(x_dim))

    def update_complex_indicators(self):
        """Initialize or update ℝ/ℂ indicators for dimensions that can be combined as complex"""
        for i in range(self.data.ndim):
            indicator = self.widgets['labels']['complex'][i]
            
            if self.combined_as_complex[i]:
                indicator.setText('ℂ')
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Split to real')
            elif self.can_combine_as_complex[i]:
                indicator.setText('ℝ')
                indicator.setStyleSheet(self.FLIP_ICON_STYLE + " QLabel {font-weight: bold; }")
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.PointingHandCursor))
                indicator.setToolTip(f'Combine as complex')
            else:
                # No indicator, already-complex data or non-size-2 dimensions
                indicator.setText('')
                indicator.setToolTip('')
                indicator.setCursor(QtGui.QCursor(Qt.QtCore.Qt.CursorShape.ArrowCursor))

    def _update_channel_controls(self):
        """Keep channel options in sync with the current array dtype."""
        is_complex = np.iscomplexobj(self.data)
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

        checked_channel = self.view_state.channel.value
        if not enabled_channels.get(checked_channel, False):
            checked_channel = 'complex' if is_complex else 'real'
            self._set_view_state(self.view_state.with_channel(checked_channel))

        channel_buttons[checked_channel].setChecked(True)
    
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
        self._set_view_state(self.view_state.with_line_axis(int(axis)))
        self.profile_axes = (self.view_state.line_axis,)
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        self.render(reason="profile-axis")

    def set_dimension_role(self, role, axis):
        if axis >= self.data.ndim:
            return
        if role == "p":
            self._set_view_state(self.view_state.with_line_axis(int(axis)))
            self.profile_axes = (self.view_state.line_axis,)
            if hasattr(self, "profile_dock"):
                self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        elif role in ("y", "x"):
            if self.view_state.image_axes is None:
                return
            self._set_view_state(self.view_state.with_image_axis(role, axis))
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
                elif i in image_axes:
                    w.setEnabled(False)
                    if self.view_state.image_axes is not None and i == self.view_state.image_axes[0]:
                        bPrim.setChecked(True)
                        bSecondary.setChecked(False)
                    else:
                        bPrim.setChecked(False)
                        bSecondary.setChecked(True)
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
        
        # Check CTRL+number for colormap changes
        if modifiers == Qt.QtCore.Qt.KeyboardModifier.ControlModifier:
            colormap_name = colormap_name_for_key(key)
            if colormap_name is not None:
                self.setColormap(colormap_name)
                event.accept()
                return
                
        # Pass event to parent if not handled
        super().keyPressEvent(event)
    
    def setColormap(self, colormap_name):
        """Set the colormap for the image view"""
        try:
            colormap = named_colormap(colormap_name)
            if colormap is None:
                print(f"Unknown colormap: {colormap_name}")
                return

            # Apply colormap to the image view
            self.img_view.setColorMap(colormap)
            #self.current_colormap = colormap_name
            
        except Exception as e:
            print(f"Failed to set colormap {colormap_name}: {e}")
    
    def eventFilter(self, obj, event):
        if obj == self.tab_widget.tabBar():
            if event.type() == Qt.QtCore.QEvent.Type.MouseButtonDblClick:
                self.profile_dock.toggle_style()
                event.accept()
                return True
        return super().eventFilter(obj, event)
    
