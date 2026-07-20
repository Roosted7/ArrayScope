from __future__ import annotations

import contextlib

import numpy as np

from arrayscope.core.array_metadata import derived_info_for
from arrayscope.core.slice_selection import center_index, parse_slice_selection
from arrayscope.core.view_state import ChannelMode, ScaleMode
from arrayscope.ui.toasts import show_status_message
from arrayscope.window.domain import Domain


class StateSyncMixin:
    def _notify_sync(self, facet):
        """Tell the linked-window controller this facet's state changed.

        Safe before the controller exists (startup/session restore); the
        controller itself drops notifications for disabled facets, echoes of
        remote applies, and unchanged payloads.
        """

        controller = getattr(self, "sync_controller", None)
        if controller is not None:
            controller.schedule_publish(facet)

    def _set_view_state(self, state):
        self.view_state = state.for_shape(self.data.shape)
        self.line_plot_dimension = (
            self.view_state.line_axis if self.view_state.line_axis is not None else 0
        )
        self.profile_axes = tuple(
            axis
            for axis in getattr(self, "profile_axes", (self.line_plot_dimension,))
            if axis < self.view_state.ndim
        )
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
        self.state_binder.sync(self)

    def _reset_controls_to_view_state(self):
        """Re-apply bindings even if state is unchanged (widget-side drift)."""

        if not hasattr(self, "widgets"):
            return
        self.state_binder.forget()
        self.state_binder.sync(self)

    def _sync_slice_controls_immediately(self, axis: int) -> None:
        axis = int(axis)
        if hasattr(self, "widgets"):
            self.state_binder.sync(self, names=(f"slice-axis-{axis}", f"strip-axis-{axis}"))
        self._interactive_slice_controls_synced_state = self.view_state

    def _apply_slice_state(
        self, axis: int, state, *, reason: str, interactive: bool, immediate_axis_only: bool
    ) -> None:
        axis = int(axis)
        self._active_slice_axis = axis
        renderer = getattr(self, "renderer", None)
        observe_montage = getattr(renderer, "_observe_montage_prefetch_momentum", None)
        if state.montage_axis is not None and callable(observe_montage):
            observe_montage(self.view_state, state)
        if interactive:
            # Dimension scrubbing IS user interaction: without this note the
            # interaction gates (commit coalescing, speculation yielding,
            # deferred stage planning) are blind to index-window bursts, and
            # every scrub step pays the full synchronous planning pipeline —
            # the few-Hz-scroll field report of 2026-07-05.  The quiet timer
            # releases the flag ~120 ms after the last step, exactly like
            # pan/zoom.
            self._note_viewport_interaction("dimension-scrub")
        self._set_view_state(state)
        if immediate_axis_only:
            self._sync_slice_controls_immediately(axis)
        else:
            self._sync_controls_from_view_state()
            self.update_dimension_controls()
            self.update_complex_indicators()
            self.update_shift_indicators()
            self._interactive_slice_controls_synced_state = None
        self.request_render(reason=reason, interactive=interactive)
        # Adjacent-slice prefetch (opt-in setting) rides every slice change:
        # the scheduler is latest-wins and runs on the speculative kernel
        # lane, so scrub bursts collapse and visible work always goes first.
        # (This call was severed when the legacy normal-image update path was
        # deleted; the setting had been silently dead since.)
        if state.montage_axis is None:
            schedule_prefetch = getattr(renderer, "_schedule_prefetch_nearby_slices", None)
            if callable(schedule_prefetch):
                schedule_prefetch(state, self.renderer._evaluation_colormap_lut(state))
        self._notify_sync("dims")

    def _apply_synced_dimension_state(self, state) -> None:
        """Apply dimension sync through the window's normal state boundary."""

        previous = self.view_state
        if state == previous:
            return
        line_axis_changed = previous.line_axis != state.line_axis
        self._set_view_state(state)
        if line_axis_changed and self.view_state.line_axis is not None:
            self.profile_axes = (self.view_state.line_axis,)
            if hasattr(self, "profile_dock"):
                self.profile_dock.set_axes(self.data.shape, self.view_state.line_axis)
        self._sync_controls_from_view_state()
        self.update_dimension_controls()
        self.update_complex_indicators()
        self.update_shift_indicators()
        self.apply_axis_flips()
        self.request_render(reason="sync-dims", interactive=False)

    def _on_slice_index_changed(self, axis, value):
        axis = int(axis)
        if axis >= self.view_state.ndim:
            return
        state = self.view_state.with_slice(axis, value).with_axis_range(axis, None)
        if state.montage_axis == axis:
            state = state.with_montage_axis(None)
        self._apply_slice_state(
            axis, state, reason="slice", interactive=True, immediate_axis_only=True
        )

    def _on_slice_text_changed(self, axis, text):
        axis = int(axis)
        text = str(text).strip()
        if axis >= self.view_state.ndim:
            return
        if text == "":
            midpoint = center_index(self.data.shape[axis])
            state = self.view_state.with_slice(axis, midpoint).with_axis_range(axis, None)
            if state.montage_axis == axis:
                state = state.with_montage_axis(None)
            self._apply_slice_state(
                axis,
                state,
                reason="slice-empty-midpoint",
                interactive=True,
                immediate_axis_only=False,
            )
            return
        try:
            selection = parse_slice_selection(text, self.data.shape[axis])
        except ValueError:
            show_status_message(self, f"Could not understand slice selection: {text}", timeout=2000)
            self._reset_controls_to_view_state()
            return
        indices = selection.indices
        if not indices:
            show_status_message(self, f"Could not understand slice selection: {text}", timeout=2000)
            self._reset_controls_to_view_state()
            return
        text = selection.text
        if selection.kind == "scalar":
            state = self.view_state.with_slice(axis, indices[0]).with_axis_range(axis, None)
            if state.montage_axis == axis:
                state = state.with_montage_axis(None)
            self._apply_slice_state(
                axis, state, reason="slice", interactive=True, immediate_axis_only=True
            )
            return
        if self.view_state.image_axes is not None and axis in self.view_state.image_axes:
            state = self.view_state.with_axis_range(axis, indices=indices, text=text)
        else:
            state = self.view_state.with_montage_axis(axis, indices=indices, text=text)
        self._apply_slice_state(
            axis, state, reason="slice-range", interactive=True, immediate_axis_only=False
        )

    def _on_channel_clicked(self, name):
        self._set_channel(name, user_selected=True)
        self.render(reason="channel", force_autolevel=True)

    def _set_channel(self, channel, *, user_selected: bool, force_autolevel: bool = True):
        self._channel_user_selected = bool(user_selected)
        self._set_view_state(self.view_state.with_channel(channel))
        if (
            self.view_state.channel == ChannelMode.ANGLE
            and self.view_state.scale != ScaleMode.LINEAR
        ):
            # Log/symlog of a signed cyclic phase is meaningless and renders
            # near-black; phase always displays on a linear scale.
            self._set_view_state(self.view_state.with_scale(ScaleMode.LINEAR))
            show_status_message(self, "Phase channel uses a linear scale.", timeout=2500)
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
        elif (
            is_complex
            and not getattr(self, "_channel_user_selected", False)
            and channel == ChannelMode.REAL
        ):
            target = ChannelMode.COMPLEX
        if target is None or target == channel:
            return False
        self._set_view_state(self.view_state.with_channel(target))
        self._apply_channel_colormap()
        return True

    def _on_scale_clicked(self, scale):
        if scale == "symlog":
            mode = ScaleMode.SYMLOG
        elif scale == "log":
            mode = ScaleMode.LOG
        else:
            mode = ScaleMode.LINEAR
        self._set_view_state(self.view_state.with_scale(mode))
        self._force_autolevel = True
        self.render(reason="scale", force_autolevel=True)

    def _set_document(self, document):
        self.operation_coordinator.set_document(document)
        self.base_data = self.operation_coordinator.base_data
        self.document = self.operation_coordinator.document
        self.operation_evaluator = self.operation_coordinator.evaluator
        if hasattr(self, "_refresh_memory_policy"):
            self._refresh_memory_policy(active_render=False)
        self.data = self._derived_info()
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self._coerce_channel_for_current_dtype()
        self._sync_controls_to_current_data()
        self._force_autolevel = True
        self._update_channel_controls()
        self._update_operation_dock()
        self._notify_sync("operations")

    def _sync_controls_to_current_data(self):
        ndim = self.data.ndim
        self.singleton = [size == 1 for size in self.data.shape]
        self._set_view_state(self.view_state.for_shape(self.data.shape, preserve_flags=True))
        self.domain = [Domain.NATIVE for _ in range(ndim)]

        if self._current_is_complex():
            self.can_combine_as_complex = [False] * ndim
        else:
            self.can_combine_as_complex = [self.data.shape[i] == 2 for i in range(ndim)]
        self.combined_as_complex = [
            self._current_is_complex() and self.data.shape[i] == 1 for i in range(ndim)
        ]

        valid_dims = [i for i in range(ndim) if not self.singleton[i]]
        if ndim >= 1 and (
            self.line_plot_dimension >= ndim or self.singleton[self.line_plot_dimension]
        ):
            self.line_plot_dimension = valid_dims[0] if valid_dims else 0
        self.profile_axes = tuple(axis for axis in getattr(self, "profile_axes", ()) if axis < ndim)
        if not self.profile_axes and ndim >= 1:
            self.profile_axes = (self.line_plot_dimension,)
        if self.profile_axes:
            self.line_plot_dimension = self.profile_axes[0]

        for i, container in enumerate(getattr(self, "dim_containers", [])):
            visible = i < ndim
            container.setVisible(visible)
            self.widgets["buttons"]["primary"][i].setVisible(visible)
            self.widgets["buttons"]["secondary"][i].setVisible(visible)
            self.widgets["buttons"]["profile"][i].setVisible(visible)
            self.widgets["spins"]["slice_indices"][i].setVisible(visible)
            if visible:
                self.widgets["labels"]["dims"][i].setText(f"[{self.data.shape[i]}]")
                self.widgets["spins"]["slice_indices"][i].setMaximum(self.data.shape[i] - 1)
                self.widgets["spins"]["slice_indices"][i].setValue(
                    min(self.widgets["spins"]["slice_indices"][i].value(), self.data.shape[i] - 1)
                )

        self.tab_widget.setTabEnabled(0, ndim >= 2)
        self.tab_widget.setVisible(ndim >= 2)
        central = self.centralWidget()
        if central is not None:
            if ndim < 2:
                # 1D shows only toolbar + dimension strip up top; cap the
                # central area so the profile dock gets the freed height
                # instead of a large empty canvas region.
                central.setMaximumHeight(max(120, central.sizeHint().height()))
            else:
                central.setMaximumHeight(16_777_215)
        if hasattr(self, "profile_dock"):
            self.profile_dock.set_axes(self.data.shape, self.line_plot_dimension)
            if ndim == 1:
                self.layout_manager.set_managed_dock_visible(
                    self.profile_dock, True, reason="one-dimensional"
                )
        if hasattr(self, "dimension_strip"):
            panel_manager = getattr(self, "panel_manager", None)
            if panel_manager is not None and hasattr(self.dimension_strip, "set_profile_available"):
                with contextlib.suppress(KeyError):
                    self.dimension_strip.set_profile_available(panel_manager.is_visible("profile"))
            self.dimension_strip.update_state(
                self.data.shape, self.view_state, self.profile_axes, axes=self.document.current_axes
            )

        self._update_array_info_label()

        self.update_complex_indicators()
        self.update_shift_indicators()
        self.update_dimension_controls()
        # The structural rebuild above wrote bound widgets directly, so the
        # binder's change detection must not skip the re-apply.
        self.state_binder.forget()
        self._sync_controls_from_view_state()

    def _update_array_info_label(self):
        info_label = (
            self.widgets.get("labels", {}).get("arrayInfo")
            if isinstance(getattr(self, "widgets", None), dict)
            else None
        )
        if info_label is None:
            return
        nbytes = getattr(self.data, "nbytes", None)
        size_text = (
            ""
            if nbytes is None
            else f" · {nbytes / 1e6:.1f} MB"
            if nbytes >= 1e6
            else f" · {nbytes / 1e3:.0f} kB"
        )
        info_label.setText(f"{tuple(self.data.shape)} {self.data.dtype}")
        info_label.setToolTip(
            f"shape {tuple(self.data.shape)} · dtype {self.data.dtype}{size_text}"
        )
        toolbar = getattr(self, "display_toolbar", None)
        if toolbar is not None and hasattr(toolbar, "sync_center_separator"):
            toolbar.sync_center_separator()

    def _update_operation_dock(self):
        from time import perf_counter

        start = perf_counter()
        if hasattr(self, "operation_dock"):
            self.operation_dock.set_operations(
                self.document.operations,
                output_shape=self.document.current_shape,
                derived_estimate=self.operation_evaluator.derived_estimate(),
                operation_shapes=self._operation_shapes(),
                steps=self.document.steps,
                operation_dtypes=self._operation_dtypes(),
            )
            self._sync_progressive_docks()
        self._last_operation_dock_ms = (perf_counter() - start) * 1000.0

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
        self.render(reason="data-changed", force_autolevel=False)
        return self.document.revision

    def _derived_info(self):
        dtypes = self.operation_coordinator.operation_dtype_estimates()
        dtype = dtypes[-1] if dtypes else getattr(self.base_data, "dtype", np.dtype(float))
        return derived_info_for(self.document, dtype=dtype)

    def _current_is_complex(self):
        return np.issubdtype(np.dtype(self.data.dtype), np.complexfloating)


def _indices_from_slice_text(text, axis_size):
    return parse_slice_selection(text, axis_size).indices
