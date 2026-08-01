"""Qt-free helpers for live image profile state."""

from __future__ import annotations

from arrayscope.display.geometry import DisplayGeometry


class ProfileCoordinator:
    def state_from_marker(self, view_state, image_x, image_y, line_axis=None):
        if line_axis is None:
            line_axis = view_state.line_axis
        if line_axis is None:
            return None
        states = DisplayGeometry(
            view_state, _display_shape_for_state(view_state)
        ).view_point_to_profile_states(image_x, image_y, (line_axis,))
        return states[0] if states else None


def _display_shape_for_state(view_state):
    if view_state.image_axes is None:
        return (1, 1)
    y_axis, x_axis = view_state.image_axes
    y_indices = view_state.axis_range_indices[y_axis]
    x_indices = view_state.axis_range_indices[x_axis]
    return (
        len(y_indices) if y_indices is not None else view_state.shape[y_axis],
        len(x_indices) if x_indices is not None else view_state.shape[x_axis],
    )
