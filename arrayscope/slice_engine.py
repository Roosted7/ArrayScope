"""Pure NumPy display preparation for ArrayScope views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .view_state import ChannelMode, ScaleMode, ViewState


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

    image_data = data[_slice_for_display_axes(state, state.image_axes)]

    # Preserve the existing viewer orientation: when primary precedes secondary
    # in ndarray order, transpose after slicing.
    if state.image_axes[0] < state.image_axes[1]:
        image_data = np.transpose(image_data)

    image_data = np.squeeze(image_data)

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


def make_line(data, state):
    state = _validated_state_for_data(data, state)
    if state.line_axis is None:
        raise ValueError("line_axis must be set to make a line")

    line_data = data[_slice_for_display_axes(state, (state.line_axis,))]
    line_data = apply_channel(line_data, state.channel)
    line_data = _apply_scale(line_data, state.scale)
    line_data = np.squeeze(line_data)
    return DisplayLine(data=line_data, axis=state.line_axis)


def _slice_for_display_axes(state, display_axes):
    display_axes = set(display_axes)
    index = []
    for axis in range(state.ndim):
        if axis in display_axes:
            index.append(slice(None))
        else:
            selected_index = state.slice_indices[axis]
            index.append(slice(selected_index, selected_index + 1))
    return tuple(index)


def _apply_scale(data, scale):
    scale = ScaleMode(scale)
    if scale == ScaleMode.SYMLOG:
        return symlog(data)
    return data


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
