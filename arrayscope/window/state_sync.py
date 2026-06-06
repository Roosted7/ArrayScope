from __future__ import annotations

import numpy as np

from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.window.domain import Domain


class StateSyncMixin:
    def _set_view_state(self, state):
        self.view_state = state.for_shape(self.data.shape)
        self.line_plot_dimension = self.view_state.line_axis if self.view_state.line_axis is not None else 0
        self.profile_axes = tuple(axis for axis in getattr(self, "profile_axes", (self.line_plot_dimension,)) if axis < self.view_state.ndim)
        if not self.profile_axes and self.view_state.line_axis is not None:
            self.profile_axes = (self.view_state.line_axis,)
        return self.view_state

    def _image_axes(self):
        return self.view_state.image_axes or ()

    def _axis_flipped(self, axis):
        return bool(self.view_state.axis_flipped[int(axis)])

    def _sync_controls_from_view_state(self):
        if not hasattr(self, "widgets"):
            return
        for axis, spinbox in enumerate(self.widgets['spins']['slice_indices'][: self.data.ndim]):
            spinbox.blockSignals(True)
            try:
                spinbox.setMaximum(self.data.shape[axis] - 1)
                spinbox.setValue(self.view_state.slice_indices[axis])
            finally:
                spinbox.blockSignals(False)

        channel_buttons = self.widgets['buttons']['channel']
        if self.view_state.channel.value in channel_buttons:
            channel_buttons[self.view_state.channel.value].setChecked(True)
        self.widgets['buttons']['processing']['linear'].setChecked(self.view_state.scale == ScaleMode.LINEAR)
        self.widgets['buttons']['processing']['symlog'].setChecked(self.view_state.scale == ScaleMode.SYMLOG)
        if hasattr(self, "dimension_strip"):
            self.dimension_strip.update_state(self.data.shape, self.view_state, self.profile_axes)
        if hasattr(self, "display_toolbar"):
            self.display_toolbar.set_current(
                channel=self.view_state.channel.value,
                scale=self.view_state.scale.value,
                aspect=getattr(self.img_view, "displayMode", "square_pixels") if hasattr(self, "img_view") else "square_pixels",
                window_mode="absolute" if self.widgets['buttons']['display']['window_absolute'].isChecked() else "relative",
                live_profile=self.widgets['buttons']['display']['live_profile'].isChecked(),
            )

    def _on_slice_index_changed(self, axis, value):
        if axis >= self.view_state.ndim:
            return
        self._active_slice_axis = int(axis)
        self._set_view_state(self.view_state.with_slice(axis, value))
        self.render(reason="slice")

    def _on_channel_clicked(self, name):
        self._set_view_state(self.view_state.with_channel(name))
        self._force_autolevel = True
        self._apply_channel_colormap()
        self.render(reason="channel", force_autolevel=True)

    def _on_scale_clicked(self, scale):
        self._set_view_state(self.view_state.with_scale(ScaleMode.SYMLOG if scale == "symlog" else ScaleMode.LINEAR))
        self._force_autolevel = True
        self.render(reason="scale", force_autolevel=True)

    def _set_document(self, document):
        self.operation_coordinator.set_document(document)
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        data = self.operation_evaluator.current_data()
        self.data = data
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._sync_controls_to_current_data()
        self._force_autolevel = True
        self._update_channel_controls()
        self._update_operation_dock()

    def _sync_controls_to_current_data(self):
        ndim = self.data.ndim
        self.singleton = [size == 1 for size in self.data.shape]
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self.domain = [Domain.NATIVE for _ in range(ndim)]

        if np.iscomplexobj(self.data):
            self.can_combine_as_complex = [False] * ndim
        else:
            self.can_combine_as_complex = [self.data.shape[i] == 2 for i in range(ndim)]
        self.combined_as_complex = [np.iscomplexobj(self.data) and self.data.shape[i] == 1 for i in range(ndim)]

        valid_dims = [i for i in range(ndim) if not self.singleton[i]]
        if ndim >= 1 and (self.line_plot_dimension >= ndim or self.singleton[self.line_plot_dimension]):
            self.line_plot_dimension = valid_dims[0] if valid_dims else 0
        self.profile_axes = tuple(axis for axis in getattr(self, "profile_axes", ()) if axis < ndim)
        if not self.profile_axes and ndim >= 1:
            self.profile_axes = (self.line_plot_dimension,)
        if self.profile_axes:
            self.line_plot_dimension = self.profile_axes[0]

        for i, container in enumerate(getattr(self, "dim_containers", [])):
            visible = i < ndim
            container.setVisible(visible)
            self.widgets['buttons']['primary'][i].setVisible(visible)
            self.widgets['buttons']['secondary'][i].setVisible(visible)
            self.widgets['buttons']['profile'][i].setVisible(visible)
            self.widgets['spins']['slice_indices'][i].setVisible(visible)
            if visible:
                self.widgets['labels']['dims'][i].setText(f'[{self.data.shape[i]}]')
                self.widgets['spins']['slice_indices'][i].setMaximum(self.data.shape[i] - 1)
                self.widgets['spins']['slice_indices'][i].setValue(
                    min(self.widgets['spins']['slice_indices'][i].value(), self.data.shape[i] - 1)
                )

        self.tab_widget.setTabEnabled(0, ndim >= 2)
        self.tab_widget.setVisible(ndim >= 2)
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.line_plot_dimension)
            if ndim == 1:
                self.profile_dock.show()
                self.profile_dock.raise_()
        if hasattr(self, "dimension_strip"):
            self.dimension_strip.update_state(self.data.shape, self.view_state, self.profile_axes)

        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_dimension_controls()
        self._sync_controls_from_view_state()

    def _update_operation_dock(self):
        if hasattr(self, "operation_dock"):
            self.operation_dock.set_operations(
                self.document.operations,
                output_shape=self.document.current_shape,
                cache_status=self.operation_evaluator.last_status,
                operation_shapes=self._operation_shapes(),
                steps=self.document.steps,
                operation_dtypes=self._operation_dtypes(),
            )
            self._sync_progressive_docks()

    def _operation_shapes(self):
        return self.operation_coordinator.operation_shapes()

    def _operation_dtypes(self):
        return self.operation_coordinator.operation_dtypes()

    def _sync_progressive_docks(self):
        changed = False
        if hasattr(self, "operation_dock"):
            has_steps = bool(self.document.steps)
            if has_steps:
                changed = changed or not self.operation_dock.isVisible()
                self._set_dock_visible_later(self.operation_dock, True)
            elif not getattr(self, "_operation_dock_user_visible", False):
                changed = changed or self.operation_dock.isVisible()
                self._set_dock_visible_later(self.operation_dock, False)
        if hasattr(self, "profile_dock"):
            live_profile = self.widgets["buttons"]["display"]["live_profile"].isChecked()
            should_show_profile = self.data.ndim == 1 or live_profile or getattr(self, "_profile_dock_user_visible", False)
            if should_show_profile:
                changed = changed or not self.profile_dock.isVisible()
                self._set_dock_visible_later(self.profile_dock, True)
            elif not self.profile_dock.isFloating():
                changed = changed or self.profile_dock.isVisible()
                self._set_dock_visible_later(self.profile_dock, False)
        if changed:
            self._schedule_view_geometry_refresh()

    def _schedule_view_geometry_refresh(self):
        import pyqtgraph.Qt as Qt

        Qt.QtCore.QTimer.singleShot(0, self._refresh_view_geometry)

    def _set_dock_visible_later(self, dock, visible):
        import pyqtgraph.Qt as Qt

        Qt.QtCore.QTimer.singleShot(0, lambda dock=dock, visible=visible: self._apply_queued_dock_visibility(dock, visible))

    def _apply_queued_dock_visibility(self, dock, visible):
        if not visible:
            if dock is getattr(self, "operation_dock", None) and self.document.steps:
                return
            if dock is getattr(self, "profile_dock", None):
                live_profile = self.widgets["buttons"]["display"]["live_profile"].isChecked()
                if self.data.ndim == 1 or live_profile or getattr(self, "_profile_dock_user_visible", False):
                    return
        dock.setVisible(bool(visible))

    def _refresh_view_geometry(self):
        if hasattr(self, "centralWidget") and self.centralWidget() is not None:
            self.centralWidget().updateGeometry()
        hint = self.sizeHint()
        if hint.isValid():
            self.resize(max(self.width(), hint.width()), max(self.height(), hint.height()))
        if hasattr(self, "img_view") and self.data.ndim >= 2:
            self.img_view.getView().autoRange()

    def _replace_base_data(self, data):
        self.operation_coordinator.replace_base_data(data)
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.data = self.operation_evaluator.current_data()
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._sync_controls_to_current_data()
        self._update_channel_controls()
        self._update_operation_dock()
