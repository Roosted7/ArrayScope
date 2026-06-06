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

    def _on_slice_index_changed(self, axis, value):
        if axis >= self.view_state.ndim:
            return
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
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.line_plot_dimension)
            if ndim == 1:
                self.profile_dock.show()
                self.profile_dock.raise_()

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
            )

    def _operation_shapes(self):
        return self.operation_coordinator.operation_shapes()

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
