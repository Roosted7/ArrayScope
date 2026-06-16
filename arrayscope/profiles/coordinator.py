"""Qt-free helpers for live image profile state."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.display.geometry import DisplayGeometry
from arrayscope.profiles.model import profile_y_range


@dataclass(frozen=True)
class ProfileRender:
    view_state: object
    line_result: object
    marker_position: tuple[int, int] | None = None
    y_range: tuple[float, float] | None = None


class ProfileCoordinator:
    def clamp_marker(self, view_state, image_x, image_y):
        if view_state.image_axes is None:
            return None
        return DisplayGeometry(view_state, _display_shape_for_state(view_state)).clamp_view_point(image_x, image_y)

    def state_from_marker(self, view_state, image_x, image_y, line_axis=None):
        if line_axis is None:
            line_axis = view_state.line_axis
        if line_axis is None:
            return None
        states = DisplayGeometry(view_state, _display_shape_for_state(view_state)).view_point_to_profile_states(image_x, image_y, (line_axis,))
        return states[0] if states else None

    def render_from_marker(self, evaluator, view_state, image_x, image_y, *, line_axis=None, y_range_mode="match_image", image_levels=None):
        clamped = self.clamp_marker(view_state, image_x, image_y)
        if clamped is None:
            return None
        profile_state = self.state_from_marker(view_state, clamped[0], clamped[1], line_axis=line_axis)
        if profile_state is None:
            return None
        return ProfileRender(
            view_state=profile_state,
            line_result=evaluator.line(profile_state),
            marker_position=clamped,
            y_range=profile_y_range(y_range_mode, image_levels),
        )


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
