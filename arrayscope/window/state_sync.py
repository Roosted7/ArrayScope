from __future__ import annotations

import numpy as np

from arrayscope.core.array_metadata import derived_info_for
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
                window_mode="absolute" if self.widgets['buttons']['display']['window_absolute'].isChecked() else "relative",
                live_profile=self.widgets['buttons']['display']['live_profile'].isChecked(),
            )

    def _on_slice_index_changed(self, axis, value):
        if axis >= self.view_state.ndim:
            return
        self._active_slice_axis = int(axis)
        state = self.view_state.with_slice(axis, value).with_axis_range(axis, None)
        if state.montage_axis == int(axis):
            state = state.with_montage_axis(None)
        self._set_view_state(state)
        self.render(reason="slice")

    def _on_slice_text_changed(self, axis, text):
        axis = int(axis)
        text = str(text).strip()
        if axis >= self.view_state.ndim:
            return
        if text == "":
            midpoint = max(0, int(self.data.shape[axis]) // 2)
            state = self.view_state.with_slice(axis, midpoint).with_axis_range(axis, None)
            if state.montage_axis == axis:
                state = state.with_montage_axis(None)
            self._active_slice_axis = axis
            self._set_view_state(state)
            self.render(reason="slice-empty-midpoint")
            return
        if ":" not in text:
            try:
                self._on_slice_index_changed(axis, int(text))
            except ValueError:
                self._sync_controls_from_view_state()
            return
        try:
            indices = _indices_from_slice_text(text, self.data.shape[axis])
        except ValueError:
            self._sync_controls_from_view_state()
            return
        if not indices:
            self._sync_controls_from_view_state()
            return
        self._active_slice_axis = axis
        if self.view_state.image_axes is not None and axis in self.view_state.image_axes:
            self._set_view_state(self.view_state.with_axis_range(axis, indices=indices, text=text))
        else:
            self._set_view_state(self.view_state.with_montage_axis(axis, indices=indices, text=text))
        self.render(reason="slice-range")

    def _on_channel_clicked(self, name):
        self._set_channel(name, user_selected=True)
        self.render(reason="channel", force_autolevel=True)

    def _set_channel(self, channel, *, user_selected: bool, force_autolevel: bool = True):
        self._channel_user_selected = bool(user_selected)
        self._set_view_state(self.view_state.with_channel(channel))
        self._force_autolevel = True
        self._apply_channel_colormap()
        self._update_channel_controls()
        return self.view_state.channel

    def _coerce_channel_for_current_dtype(self):
        channel = self.view_state.channel
        is_complex = self._current_is_complex()
        complex_only = {ChannelMode.COMPLEX, ChannelMode.IMAG, ChannelMode.ANGLE}
        target = None
        if not is_complex and channel in complex_only:
            target = ChannelMode.REAL
        elif is_complex and not getattr(self, "_channel_user_selected", False) and channel == ChannelMode.REAL:
            target = ChannelMode.COMPLEX
        if target is None or target == channel:
            return False
        self._set_view_state(self.view_state.with_channel(target))
        self._apply_channel_colormap()
        return True

    def _on_scale_clicked(self, scale):
        self._set_view_state(self.view_state.with_scale(ScaleMode.SYMLOG if scale == "symlog" else ScaleMode.LINEAR))
        self._force_autolevel = True
        self.render(reason="scale", force_autolevel=True)

    def _set_document(self, document):
        self.operation_coordinator.set_document(document)
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.data = self._derived_info()
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._coerce_channel_for_current_dtype()
        self._sync_controls_to_current_data()
        self._force_autolevel = True
        self._update_channel_controls()
        self._update_operation_dock()

    def _sync_controls_to_current_data(self):
        ndim = self.data.ndim
        self.singleton = [size == 1 for size in self.data.shape]
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self.domain = [Domain.NATIVE for _ in range(ndim)]

        if self._current_is_complex():
            self.can_combine_as_complex = [False] * ndim
        else:
            self.can_combine_as_complex = [self.data.shape[i] == 2 for i in range(ndim)]
        self.combined_as_complex = [self._current_is_complex() and self.data.shape[i] == 1 for i in range(ndim)]

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
                self.layout_manager.set_managed_dock_visible(self.profile_dock, True, reason="one-dimensional")
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
                cache_status=self.operation_evaluator.cache_diagnostics(),
                image_cache_status=self.operation_evaluator.image_cache_diagnostics(),
                profile_cache_status=self.operation_evaluator.profile_cache_diagnostics(),
                derived_estimate=self.operation_evaluator.derived_estimate(),
                operation_shapes=self._operation_shapes(),
                steps=self.document.steps,
                operation_dtypes=self._operation_dtypes(),
            )
            self._sync_progressive_docks()

    def _operation_shapes(self):
        return self.operation_coordinator.operation_shapes()

    def _operation_dtypes(self):
        return self.operation_coordinator.operation_dtype_estimates()

    def _sync_progressive_docks(self):
        self.layout_manager.sync_progressive_docks()

    def _schedule_view_geometry_refresh(self):
        self.layout_manager.schedule_view_geometry_refresh()

    def _set_dock_visible_later(self, dock, visible):
        self.layout_manager.set_dock_visible_later(dock, visible)

    def _apply_queued_dock_visibility(self, dock, visible):
        self.layout_manager.apply_queued_dock_visibility(dock, visible)

    def _refresh_view_geometry(self):
        self.layout_manager.refresh_view_geometry()

    def _replace_base_data(self, data):
        self.operation_coordinator.replace_base_and_clear_steps(data)
        self._sync_after_document_data_change()

    def _reload_base_data(self, data, *, preserve_steps=True):
        self.operation_coordinator.reload_base_data(data, preserve_steps=preserve_steps)
        self._sync_after_document_data_change()

    def _sync_after_document_data_change(self):
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.data = self._derived_info()
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._coerce_channel_for_current_dtype()
        self._sync_controls_to_current_data()
        self._update_channel_controls()
        self._update_operation_dock()

    def notify_data_changed(self):
        self.operation_coordinator.mark_base_data_changed()
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        self.data = self._derived_info()
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._coerce_channel_for_current_dtype()
        self._sync_controls_to_current_data()
        self._force_autolevel = True
        self.render(reason="data-changed", force_autolevel=True)
        return self.document.revision

    def _derived_info(self):
        dtypes = self.operation_coordinator.operation_dtype_estimates()
        dtype = dtypes[-1] if dtypes else getattr(self.base_data, "dtype", np.dtype(float))
        return derived_info_for(self.document, dtype=dtype)

    def _current_is_complex(self):
        return np.issubdtype(np.dtype(self.data.dtype), np.complexfloating)


def _indices_from_slice_text(text, axis_size):
    parts = str(text).split(":")
    if len(parts) > 3:
        raise ValueError("slice range must have at most start:stop:step")

    def parse(part):
        part = part.strip()
        return None if part == "" else int(part)

    while len(parts) < 3:
        parts.append("")
    if len(str(text).split(":")) == 3:
        start, step, stop = (parse(part) for part in parts[:3])
        if step is None:
            step = 1
        if start is None:
            start = 0 if step > 0 else int(axis_size) - 1
        if stop is None:
            stop = int(axis_size) - 1 if step > 0 else 0
        if step == 0:
            raise ValueError("slice step cannot be zero")
        end = min(int(axis_size) - 1, stop) if step > 0 else max(0, stop)
        values = []
        current = max(0, min(int(axis_size) - 1, start))
        while (current <= end if step > 0 else current >= end):
            values.append(current)
            current += step
        return tuple(values)
    start, stop, step = (parse(part) for part in parts[:3])
    if step == 0:
        raise ValueError("slice step cannot be zero")
    return tuple(range(*slice(start, stop, step).indices(int(axis_size))))
