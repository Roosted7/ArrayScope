"""Qt-free helpers for live image profile state."""

from __future__ import annotations

from dataclasses import dataclass

from .profile import clamp_marker_position, profile_state_from_image_hover, profile_y_range


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
        return clamp_marker_position(view_state.shape, view_state.image_axes, image_x, image_y)

    def state_from_marker(self, view_state, image_x, image_y, line_axis=None):
        return profile_state_from_image_hover(view_state, image_x, image_y, line_axis=line_axis)

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
