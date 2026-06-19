"""Pure NumPy display preparation for ArrayScope views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from arrayscope.core.view_state import ChannelMode, ScaleMode, ViewState
from arrayscope.display.lod import LodInfo
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    ShaderScale,
    TexturePlaneKind,
    apply_scale as apply_shader_scale,
    apply_phase_lut,
    extract_component,
)


@dataclass(frozen=True)
class DisplayImage:
    data: np.ndarray
    histogram_data: Optional[np.ndarray] = None
    default_levels: Optional[Tuple[float, float]] = None
    rgb_already_windowed: bool = False
    shader_mapping: ShaderMapping | None = None
    texture_kind: TexturePlaneKind | None = None
    semantic_data: np.ndarray | None = None
    lod: LodInfo | None = None


@dataclass(frozen=True)
class DisplayLine:
    data: np.ndarray
    axis: int


def symlog(data, C=0):
    return apply_shader_scale(data, ShaderScale.SYMLOG, symlog_constant=C)


def apply_channel(data, channel):
    if hasattr(channel, "value"):
        channel = channel.value
    channel = ChannelMode(channel)
    if channel == ChannelMode.COMPLEX:
        return extract_component(data, ShaderComponent.ABS)
    if channel == ChannelMode.ABS:
        return extract_component(data, ShaderComponent.ABS)
    if channel == ChannelMode.ANGLE:
        return extract_component(data, ShaderComponent.ANGLE)
    if channel == ChannelMode.REAL:
        return extract_component(data, ShaderComponent.REAL)
    if channel == ChannelMode.IMAG:
        return extract_component(data, ShaderComponent.IMAG)
    raise ValueError(f"unsupported channel mode: {channel}")


def complex_to_rgb(data, colormap_lut=None):
    try:
        return apply_phase_lut(data, colormap_lut)
    except ValueError as exc:
        if colormap_lut is not None:
            raise ValueError("colormap_lut must have shape (N, 3) or (N, 4)") from exc
        raise


def make_image(data, state, colormap_lut=None):
    state = _validated_state_for_data(data, state)
    if state.image_axes is None:
        raise ValueError("image_axes must be set to make an image")

    image_data, present_axes = _extract_display_axes(data, state, state.image_axes)
    image_data = _apply_display_axis_ranges(image_data, state, present_axes)
    image_data = _reorder_present_axes(image_data, present_axes, state.image_axes)
    image_data = _ensure_image_rank(image_data)

    channel = _channel_mode(state.channel)
    if channel == ChannelMode.COMPLEX:
        rgb_data, magnitude_data = complex_to_rgb(image_data, colormap_lut=colormap_lut)
        magnitude_data = _apply_scale(magnitude_data, state.scale)
        return DisplayImage(
            data=rgb_data,
            histogram_data=magnitude_data,
            rgb_already_windowed=False,
            shader_mapping=ShaderMapping(
                component=ShaderComponent.ABS,
                scale=_shader_scale(state.scale),
                display_mode=ShaderDisplayMode.PHASE_COLOR,
                lut_identity=_lut_identity(colormap_lut),
                lut_data=colormap_lut,
            ),
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            semantic_data=image_data,
        )

    default_levels = None
    if channel == ChannelMode.ANGLE:
        default_levels = (-np.pi, np.pi)

    image_data = apply_channel(image_data, channel)
    image_data = _apply_scale(image_data, state.scale)
    image_data = np.nan_to_num(image_data)
    return DisplayImage(
        data=image_data,
        default_levels=default_levels,
        shader_mapping=ShaderMapping(
            component=_shader_component_for_channel(channel),
            scale=_shader_scale(state.scale),
            display_mode=ShaderDisplayMode.SCALAR,
        ),
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image_data,
    )


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

    channel = _channel_mode(state.channel)
    if channel == ChannelMode.COMPLEX:
        rgb_data, magnitude_data = complex_to_rgb(image_data, colormap_lut=colormap_lut)
        magnitude_data = _apply_scale(magnitude_data, state.scale)
        return DisplayImage(
            data=rgb_data,
            histogram_data=magnitude_data,
            rgb_already_windowed=False,
            shader_mapping=ShaderMapping(
                component=ShaderComponent.ABS,
                scale=_shader_scale(state.scale),
                display_mode=ShaderDisplayMode.PHASE_COLOR,
                lut_identity=_lut_identity(colormap_lut),
                lut_data=colormap_lut,
            ),
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            semantic_data=image_data,
        )

    default_levels = None
    if channel == ChannelMode.ANGLE:
        default_levels = (-np.pi, np.pi)

    image_data = apply_channel(image_data, channel)
    image_data = _apply_scale(image_data, state.scale)
    image_data = np.nan_to_num(image_data)
    return DisplayImage(
        data=image_data,
        default_levels=default_levels,
        shader_mapping=ShaderMapping(
            component=_shader_component_for_channel(channel),
            scale=_shader_scale(state.scale),
            display_mode=ShaderDisplayMode.SCALAR,
        ),
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image_data,
    )


def make_shader_image_from_slab(slab, request, colormap_lut=None):
    """Create a shader-capable display image from an evaluated image slab."""
    state = request.view_state
    if state.image_axes is None:
        raise ValueError("image_axes must be set to make an image")

    image_data = np.asarray(slab)
    present_axes = _present_axes_for_slab(state, state.image_axes)
    image_data = _apply_display_axis_ranges(image_data, state, present_axes, applied_axes=getattr(request, "ranged_axes", ()))
    image_data = _reorder_present_axes(image_data, present_axes, state.image_axes)
    image_data = _ensure_image_rank(image_data)

    channel = _channel_mode(state.channel)
    if np.iscomplexobj(image_data):
        component = _shader_component_for_channel(channel)
        phase_color = channel in {ChannelMode.COMPLEX, ChannelMode.ANGLE}
        mapping = ShaderMapping(
            component=component,
            scale=_shader_scale(state.scale),
            display_mode=ShaderDisplayMode.PHASE_COLOR if phase_color else ShaderDisplayMode.SCALAR,
            lut_identity=_lut_identity(colormap_lut) if phase_color else None,
            lut_data=colormap_lut if phase_color else None,
        )
        histogram_data = apply_shader_scale(
            extract_component(image_data, component),
            mapping.scale,
            symlog_constant=mapping.symlog_constant,
        )
        complex_data = np.ascontiguousarray(image_data.astype(np.complex64, copy=False))
        return DisplayImage(
            data=complex_data,
            histogram_data=histogram_data,
            default_levels=(-np.pi, np.pi) if channel == ChannelMode.ANGLE else None,
            rgb_already_windowed=False,
            shader_mapping=mapping,
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            semantic_data=complex_data,
        )

    default_levels = None
    if channel == ChannelMode.ANGLE:
        default_levels = (-np.pi, np.pi)
    component = apply_channel(image_data, channel)
    mapping = ShaderMapping(
        component=ShaderComponent.REAL,
        scale=_shader_scale(state.scale),
        display_mode=ShaderDisplayMode.SCALAR,
    )
    histogram_data = apply_shader_scale(component, mapping.scale, symlog_constant=mapping.symlog_constant)
    return DisplayImage(
        data=np.ascontiguousarray(np.asarray(component, dtype=np.float32)),
        histogram_data=histogram_data,
        default_levels=default_levels,
        shader_mapping=mapping,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=np.ascontiguousarray(np.asarray(component, dtype=np.float32)),
    )


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
    channel = _channel_mode(state.channel)
    if channel != ChannelMode.COMPLEX:
        line_data = apply_channel(line_data, channel)
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
    channel = _channel_mode(state.channel)
    if channel != ChannelMode.COMPLEX:
        line_data = apply_channel(line_data, channel)
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
    if scale == ScaleMode.LOG:
        return apply_shader_scale(data, ShaderScale.LOG)
    if scale == ScaleMode.SYMLOG:
        return apply_shader_scale(data, ShaderScale.SYMLOG)
    return apply_shader_scale(data, ShaderScale.LINEAR)


def _shader_scale(scale) -> ShaderScale:
    if hasattr(scale, "value"):
        scale = scale.value
    scale = ScaleMode(scale)
    if scale == ScaleMode.LOG:
        return ShaderScale.LOG
    if scale == ScaleMode.SYMLOG:
        return ShaderScale.SYMLOG
    return ShaderScale.LINEAR


def _shader_component_for_channel(channel) -> ShaderComponent:
    channel = _channel_mode(channel)
    if channel == ChannelMode.IMAG:
        return ShaderComponent.IMAG
    if channel in {ChannelMode.COMPLEX, ChannelMode.ABS}:
        return ShaderComponent.ABS
    if channel == ChannelMode.ANGLE:
        return ShaderComponent.ANGLE
    return ShaderComponent.REAL


def _lut_identity(colormap_lut):
    if colormap_lut is None:
        return None
    lut = np.asarray(colormap_lut)
    return (tuple(lut.shape), str(lut.dtype), lut.tobytes())


def _channel_mode(channel):
    if hasattr(channel, "value"):
        channel = channel.value
    return ChannelMode(channel)


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
