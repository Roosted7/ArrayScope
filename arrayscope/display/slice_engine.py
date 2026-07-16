"""Pure NumPy display preparation for ArrayScope views."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
    level_data: np.ndarray | None = None
    level_stats: object | None = None


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
    image_data = _sanitize_nonfinite_for_display(image_data)
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
    image_data = _sanitize_nonfinite_for_display(image_data)
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


def make_shader_image_from_slab(slab, request, colormap_lut=None, *, provisional_histogram: bool = False):
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
        if provisional_histogram:
            histogram_data = None
            level_data = _sample_shader_level_data(
                image_data,
                component,
                mapping.scale,
                symlog_constant=mapping.symlog_constant,
            )
        else:
            histogram_data = apply_shader_scale(
                extract_component(image_data, component),
                mapping.scale,
                symlog_constant=mapping.symlog_constant,
            )
            level_data = None
        complex_data = np.ascontiguousarray(image_data.astype(np.complex64, copy=False))
        return DisplayImage(
            data=complex_data,
            histogram_data=histogram_data,
            default_levels=(-np.pi, np.pi) if channel == ChannelMode.ANGLE else None,
            rgb_already_windowed=False,
            shader_mapping=mapping,
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            semantic_data=complex_data,
            level_data=level_data,
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
    if provisional_histogram:
        histogram_data = None
        level_data = _sample_shader_level_data(component, ShaderComponent.REAL, mapping.scale, symlog_constant=mapping.symlog_constant)
    else:
        histogram_data = apply_shader_scale(component, mapping.scale, symlog_constant=mapping.symlog_constant)
        level_data = None
    component = np.ascontiguousarray(np.asarray(component, dtype=np.float32))
    return DisplayImage(
        data=component,
        histogram_data=histogram_data,
        default_levels=default_levels,
        shader_mapping=mapping,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=component,
        level_data=level_data,
    )


def complex_texture_shader_mapping(state, colormap_lut=None):
    """Canonical shader mapping for a COMPLEX_RG32F texture under ``state``.

    A complex texture is only displayable together with the mapping that says
    HOW to read it; the mapping is a pure function of the current channel,
    scale, and LUT — not of the plane's history.  Callers presenting a
    resident complex plane whose recorded mapping is gone (metadata is
    session-scoped; the pyramid cache is not) must mint the current mapping
    here instead of presenting unmapped complex texels: drawn without
    phase-color the magnitude runs through the cyclic LUT and zero magnitude
    renders LUT[0] orange (field defect 2026-07-16 09:14).
    """

    channel = _channel_mode(state.channel)
    phase_color = channel in {ChannelMode.COMPLEX, ChannelMode.ANGLE}
    return ShaderMapping(
        component=_shader_component_for_channel(channel),
        scale=_shader_scale(state.scale),
        display_mode=ShaderDisplayMode.PHASE_COLOR if phase_color else ShaderDisplayMode.SCALAR,
        lut_identity=_lut_identity(colormap_lut) if phase_color else None,
        lut_data=colormap_lut if phase_color else None,
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


def _sanitize_nonfinite_for_display(image_data):
    """Replace NaN/±Inf with values inside the finite data range.

    A bare ``nan_to_num`` maps ±Inf to the float max, which blows the display
    window to ±1e308/±3.4e38 and hides all finite structure. Clamping to the
    finite min/max keeps autolevels meaningful; NaN renders at the low end.
    """
    if not np.issubdtype(np.asarray(image_data).dtype, np.floating):
        return np.nan_to_num(image_data)
    from arrayscope.display.levels import finite_bounds

    bounds = finite_bounds(image_data)
    if bounds is None:
        return np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = bounds
    return np.nan_to_num(image_data, nan=low, posinf=high, neginf=low)


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


def _sample_shader_level_data(
    data,
    component: ShaderComponent,
    scale: ShaderScale,
    *,
    symlog_constant: float = 0.0,
    limit: int = 512,
) -> np.ndarray:
    arr = np.asarray(data)
    if arr.size == 0:
        return np.asarray((), dtype=np.float32)
    limit = max(1, int(limit))
    sampled = _spatial_level_sample(arr, limit=limit)
    values = extract_component(sampled.reshape(-1), component)
    values = apply_shader_scale(values, scale, symlog_constant=symlog_constant)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _spatial_level_sample(arr: np.ndarray, *, limit: int) -> np.ndarray:
    """Return a small deterministic sample spread over the image plane."""

    values = np.asarray(arr)
    if values.size <= int(limit):
        return values.reshape(-1)
    if values.ndim < 2:
        indices = np.linspace(0, values.size - 1, int(limit), dtype=np.int64)
        return values.reshape(-1)[indices]

    flat_indices = _spatial_level_indices(
        int(values.shape[0]),
        int(values.shape[1]),
        int(limit),
    )
    trailing_shape = tuple(values.shape[2:])
    return values.reshape(int(values.shape[0]) * int(values.shape[1]), *trailing_shape)[flat_indices]


@lru_cache(maxsize=128)
def _spatial_level_indices(height: int, width: int, limit: int) -> np.ndarray:
    height = max(1, int(height))
    width = max(1, int(width))
    limit = max(1, int(limit))
    # Choose a grid with roughly square spacing in pixel units.  This keeps the
    # sample spread across the full tile instead of walking a flattened stride.
    rows = max(1, int(np.sqrt(float(limit) * float(height) / float(width))))
    cols = max(1, int(limit) // rows)
    while rows * cols > limit and cols > 1:
        cols -= 1
    while rows * cols > limit and rows > 1:
        rows -= 1
    row_indices = _even_spatial_indices(height, rows)
    col_indices = _even_spatial_indices(width, cols)
    return (row_indices[:, None] * int(width) + col_indices[None, :]).reshape(-1)


def _even_spatial_indices(size: int, count: int) -> np.ndarray:
    size = max(1, int(size))
    count = max(1, min(int(count), size))
    if count == 1:
        return np.asarray([size // 2], dtype=np.int64)
    indices = np.linspace(0, size - 1, count, dtype=np.int64)
    center = size // 2
    indices[int(np.argmin(np.abs(indices - center)))] = center
    indices = np.unique(indices)
    if indices.size < count:
        filler = np.linspace(0, size - 1, count * 2, dtype=np.int64)
        indices = np.unique(np.concatenate((indices, filler)))
    return np.asarray(np.sort(indices)[:count], dtype=np.int64)


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
