"""Pure helpers for image-driven line profiles."""

from __future__ import annotations

from typing import Optional, Tuple

from arrayscope.core.view_state import ViewState


def clamp_marker_position(shape, image_axes, image_x, image_y):
    primary_axis, secondary_axis = tuple(int(axis) for axis in image_axes)
    x = int(round(float(image_x)))
    y = int(round(float(image_y)))
    x = max(0, min(x, int(shape[secondary_axis]) - 1))
    y = max(0, min(y, int(shape[primary_axis]) - 1))
    return x, y


def profile_states_from_marker(state: ViewState, image_x, image_y, profile_axes) -> Tuple[ViewState, ...]:
    states = []
    for axis in profile_axes:
        profile_state = profile_state_from_image_hover(state, image_x, image_y, line_axis=axis)
        if profile_state is not None:
            states.append(profile_state)
    return tuple(states)


def image_hover_indices(state: ViewState, image_x, image_y) -> Optional[Tuple[int, int]]:
    """Map image item coordinates to array indices for the current image axes."""
    if state.image_axes is None:
        return None

    primary_axis, secondary_axis = state.image_axes
    x_index = int(round(float(image_x)))
    y_index = int(round(float(image_y)))

    if x_index < 0 or x_index >= state.shape[secondary_axis]:
        return None
    if y_index < 0 or y_index >= state.shape[primary_axis]:
        return None

    return x_index, y_index


def profile_state_from_image_hover(state: ViewState, image_x, image_y, line_axis=None) -> Optional[ViewState]:
    """Build a line-profile view state from a hover position in image coordinates."""
    hover_indices = image_hover_indices(state, image_x, image_y)
    if hover_indices is None:
        return None

    x_index, y_index = hover_indices
    primary_axis, secondary_axis = state.image_axes
    if line_axis is None:
        line_axis = state.line_axis
    if line_axis is None:
        return None

    line_axis = int(line_axis)
    if line_axis < 0 or line_axis >= state.ndim:
        raise ValueError(f"line axis {line_axis} is out of bounds for {state.ndim}D shape")

    slice_indices = list(state.slice_indices)
    if line_axis != primary_axis:
        slice_indices[primary_axis] = y_index
    if line_axis != secondary_axis:
        slice_indices[secondary_axis] = x_index

    return ViewState(
        ndim=state.ndim,
        shape=state.shape,
        image_axes=state.image_axes,
        line_axis=line_axis,
        slice_indices=tuple(slice_indices),
        channel=state.channel,
        scale=state.scale,
        axis_flipped=state.axis_flipped,
        axis_fftshifted=state.axis_fftshifted,
    )


def profile_y_range(mode: str, image_levels):
    if mode == "match_image" and image_levels is not None:
        low, high = image_levels
        return (float(low), float(high))
    return None
