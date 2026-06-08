"""Pure NumPy display preparation for ArrayScope views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from arrayscope.core.view_state import ChannelMode, ScaleMode, ViewState


@dataclass(frozen=True)
class DisplayImage:
    data: np.ndarray
    histogram_data: Optional[np.ndarray] = None
    default_levels: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class DisplayLine:
    data: np.ndarray
    axis: int


def symlog(data, C=0):
    return np.sign(data) * np.log10(1 + np.abs(data) / 10**C)


def apply_channel(data, channel):
    if hasattr(channel, "value"):
        channel = channel.value
    channel = ChannelMode(channel)
    if channel == ChannelMode.COMPLEX:
        return np.abs(data)
    if channel == ChannelMode.ABS:
        return np.abs(data)
    if channel == ChannelMode.ANGLE:
        return np.angle(data)
    if channel == ChannelMode.REAL:
        return np.real(data)
    if channel == ChannelMode.IMAG:
        return np.imag(data)
    raise ValueError(f"unsupported channel mode: {channel}")


def complex_to_rgb(data, colormap_lut=None):
    phase = np.angle(data)
    magnitude = np.abs(data).astype(float)
    phase_index = (phase + np.pi) / (2.0 * np.pi)
    phase_index = np.nan_to_num(phase_index, nan=0.0, posinf=0.0, neginf=0.0)
    lut_index = np.clip((phase_index * 255).astype(int), 0, 255)

    if colormap_lut is None:
        colors = _default_phase_lut()[lut_index]
    else:
        lut = np.asarray(colormap_lut)
        if lut.ndim != 2 or lut.shape[0] == 0 or lut.shape[1] < 3:
            raise ValueError("colormap_lut must have shape (N, 3) or (N, 4)")
        scaled_index = np.clip((phase_index * (lut.shape[0] - 1)).astype(int), 0, lut.shape[0] - 1)
        colors = lut[scaled_index, :3]

    return colors.astype(np.uint8), magnitude


def make_image(data, state, colormap_lut=None):
    state = _validated_state_for_data(data, state)
    if state.image_axes is None:
        raise ValueError("image_axes must be set to make an image")

    image_data, present_axes = _extract_display_axes(data, state, state.image_axes)
    image_data = _apply_display_axis_ranges(image_data, state, present_axes)
    image_data = _reorder_present_axes(image_data, present_axes, state.image_axes)
    image_data = _ensure_image_rank(image_data)

    if state.channel == ChannelMode.COMPLEX:
        rgb_data, magnitude_data = complex_to_rgb(image_data, colormap_lut=colormap_lut)
        return DisplayImage(data=rgb_data, histogram_data=magnitude_data)

    default_levels = None
    if state.channel == ChannelMode.ANGLE:
        default_levels = (-np.pi, np.pi)

    image_data = apply_channel(image_data, state.channel)
    image_data = _apply_scale(image_data, state.scale)
    image_data = np.nan_to_num(image_data)
    return DisplayImage(data=image_data, default_levels=default_levels)


def make_image_from_slab(slab, request, colormap_lut=None):
    """Create a display image from an already evaluated image slab."""
    state = request.view_state
    if state.image_axes is None:
        raise ValueError("image_axes must be set to make an image")

    image_data = np.asarray(slab)
    present_axes = _present_axes_for_slab(state, state.image_axes)
    image_data = _apply_display_axis_ranges(image_data, state, present_axes, applied_axes=getattr(request, "ranged_axes", ()))
    image_data = _reorder_present_axes(image_data, present_axes, state.image_axes)
    image_data = _ensure_image_rank(image_data)

    if state.channel == ChannelMode.COMPLEX:
        rgb_data, magnitude_data = complex_to_rgb(image_data, colormap_lut=colormap_lut)
        return DisplayImage(data=rgb_data, histogram_data=magnitude_data)

    default_levels = None
    if state.channel == ChannelMode.ANGLE:
        default_levels = (-np.pi, np.pi)

    image_data = apply_channel(image_data, state.channel)
    image_data = _apply_scale(image_data, state.scale)
    image_data = np.nan_to_num(image_data)
    return DisplayImage(data=image_data, default_levels=default_levels)


def with_slice_index(state, axis, index):
    return state.with_slice(axis, index)


def make_export_frame(data, state, frame_axis, frame_index, colormap_lut=None):
    """Create one export frame using the same image path as the on-screen view."""
    export_state = with_slice_index(state, frame_axis, frame_index)
    return make_image(data, export_state, colormap_lut=colormap_lut)


def make_line(data, state):
    state = _validated_state_for_data(data, state)
    if state.line_axis is None:
        raise ValueError("line_axis must be set to make a line")

    line_data, present_axes = _extract_display_axes(data, state, (state.line_axis,))
    line_data = _apply_display_axis_ranges(line_data, state, present_axes)
    if state.channel != ChannelMode.COMPLEX:
        line_data = apply_channel(line_data, state.channel)
        line_data = _apply_scale(line_data, state.scale)
    line_data = _ensure_line_rank(line_data)
    return DisplayLine(data=line_data, axis=state.line_axis)


def make_line_from_slab(slab, request):
    state = request.view_state
    if state.line_axis is None:
        raise ValueError("line_axis must be set to make a line")

    line_data = np.asarray(slab)
    present_axes = _present_axes_for_slab(state, (state.line_axis,))
    line_data = _apply_display_axis_ranges(line_data, state, present_axes, applied_axes=getattr(request, "ranged_axes", ()))
    if state.channel != ChannelMode.COMPLEX:
        line_data = apply_channel(line_data, state.channel)
        line_data = _apply_scale(line_data, state.scale)
    line_data = _ensure_line_rank(line_data)
    return DisplayLine(data=line_data, axis=state.line_axis)


def make_scalar_from_slab(slab, request):
    state = request.view_state
    value = apply_channel(np.asarray(slab), state.channel)
    value = _apply_scale(value, state.scale)
    return np.asarray(value).item()


def _extract_display_axes(data, state, display_axes):
    display_axes = tuple(int(axis) for axis in display_axes)
    display_axis_set = set(display_axes)
    index = []
    present_axes = []
    for axis in range(state.ndim):
        if axis in display_axis_set:
            index.append(slice(None))
            present_axes.append(axis)
        else:
            index.append(int(state.slice_indices[axis]))
    return np.asarray(data[tuple(index)]), tuple(present_axes)


def _present_axes_for_slab(state, display_axes):
    display_axis_set = {int(axis) for axis in display_axes}
    return tuple(axis for axis in range(state.ndim) if axis in display_axis_set)


def _apply_scale(data, scale):
    if hasattr(scale, "value"):
        scale = scale.value
    scale = ScaleMode(scale)
    if scale == ScaleMode.SYMLOG:
        return symlog(data)
    return data


def _apply_display_axis_ranges(data, state, present_axes, *, applied_axes=()):
    result = np.asarray(data)
    applied_axes = {int(axis) for axis in applied_axes}
    for result_axis, original_axis in enumerate(tuple(int(axis) for axis in present_axes)):
        indices = state.axis_range_indices[original_axis]
        if indices is not None and original_axis not in applied_axes:
            result = np.take(result, tuple(indices), axis=result_axis)
    return result


def _reorder_present_axes(data, present_axes, display_axes):
    present_axes = tuple(int(axis) for axis in present_axes)
    display_axes = tuple(int(axis) for axis in display_axes)
    if len(display_axes) <= 1 or present_axes == display_axes:
        return data
    permutation = tuple(present_axes.index(axis) for axis in display_axes)
    return np.transpose(data, permutation)


def _ensure_image_rank(data):
    result = np.asarray(data)
    if result.ndim == 2:
        return result
    if result.ndim == 3 and result.shape[-1] in (3, 4):
        return result
    raise ValueError(f"image data must be 2D scalar or RGB/RGBA, got shape {result.shape}")


def _ensure_line_rank(data):
    result = np.asarray(data)
    if result.ndim == 0:
        return result.reshape((1,))
    if result.ndim == 1:
        return result
    return np.ravel(result)


def _validated_state_for_data(data, state):
    state = ViewState(
        ndim=state.ndim,
        shape=state.shape,
        image_axes=state.image_axes,
        line_axis=state.line_axis,
        slice_indices=state.slice_indices,
        channel=state.channel,
        scale=state.scale,
        axis_flipped=state.axis_flipped,
        axis_fftshifted=state.axis_fftshifted,
        montage_axis=state.montage_axis,
        montage_columns=state.montage_columns,
        montage_indices=state.montage_indices,
        montage_text=state.montage_text,
        axis_range_indices=state.axis_range_indices,
        axis_range_text=state.axis_range_text,
    )
    if tuple(data.shape) != state.shape:
        raise ValueError(f"data shape {tuple(data.shape)} does not match view state shape {state.shape}")
    return state


def _default_phase_lut():
    positions = np.linspace(0.0, 1.0, 256, endpoint=True)
    return np.array([_hsv_to_rgb(position, 1.0, 1.0) for position in positions], dtype=np.uint8)


def _hsv_to_rgb(hue, saturation, value):
    hue = float(hue % 1.0)
    chroma = value * saturation
    h_prime = hue * 6.0
    x = chroma * (1.0 - abs((h_prime % 2.0) - 1.0))

    if h_prime < 1.0:
        red, green, blue = chroma, x, 0.0
    elif h_prime < 2.0:
        red, green, blue = x, chroma, 0.0
    elif h_prime < 3.0:
        red, green, blue = 0.0, chroma, x
    elif h_prime < 4.0:
        red, green, blue = 0.0, x, chroma
    elif h_prime < 5.0:
        red, green, blue = x, 0.0, chroma
    else:
        red, green, blue = chroma, 0.0, x

    m = value - chroma
    return [round((red + m) * 255), round((green + m) * 255), round((blue + m) * 255)]
