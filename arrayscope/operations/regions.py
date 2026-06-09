"""Hashable region and stage-cache planning primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class AxisRegionKind(Enum):
    POINT = "point"
    SLICE = "slice"
    INDICES = "indices"
    ALL = "all"


def axis_region_kind(value) -> AxisRegionKind:
    if isinstance(value, AxisRegionKind):
        return value
    enum_value = getattr(value, "value", value)
    return AxisRegionKind(enum_value)


@dataclass(frozen=True)
class AxisRegion:
    kind: AxisRegionKind
    value: Any = None


@dataclass(frozen=True)
class RegionSpec:
    axes: tuple[AxisRegion, ...]


@dataclass(frozen=True)
class StageKey:
    document_key: object
    operation_prefix: tuple
    region: RegionSpec
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class StageCacheCandidate:
    stage_index: int
    operation_prefix: tuple
    region: RegionSpec
    shape: tuple[int, ...]
    dtype: str
    estimated_nbytes: int | None
    priority: str
    reason: str
    retain: bool = True
    retain_reason: str = ""


def region_from_index_spec(shape, spec) -> RegionSpec:
    shape = tuple(int(size) for size in shape)
    axes = []
    for axis, (size, item) in enumerate(zip(shape, tuple(spec))):
        if isinstance(item, int):
            axes.append(AxisRegion(AxisRegionKind.POINT, _normalize_index(item, size)))
        elif _is_index_array(item):
            indices = tuple(int(index) for index in np.asarray(item, dtype=np.int64).ravel())
            axes.append(AxisRegion(AxisRegionKind.INDICES, tuple(_normalize_index(index, size) for index in indices)))
        elif _is_full_slice(item):
            axes.append(AxisRegion(AxisRegionKind.ALL))
        elif isinstance(item, slice):
            start, stop, step = item.indices(size)
            axes.append(AxisRegion(AxisRegionKind.SLICE, (int(start), int(stop), int(step))))
        else:
            raise TypeError(f"unsupported index item for axis {axis}: {item!r}")
    if len(axes) != len(shape):
        raise ValueError("index spec length must match shape length")
    return RegionSpec(tuple(axes))


def index_spec_from_region(region: RegionSpec) -> tuple:
    spec = []
    for axis in region.axes:
        kind = axis_region_kind(axis.kind)
        if kind == AxisRegionKind.ALL:
            spec.append(slice(None))
        elif kind == AxisRegionKind.POINT:
            spec.append(int(axis.value))
        elif kind == AxisRegionKind.SLICE:
            start, stop, step = axis.value
            spec.append(slice(int(start), None if stop is None else int(stop), int(step)))
        elif kind == AxisRegionKind.INDICES:
            spec.append(tuple(int(index) for index in axis.value))
        else:
            raise TypeError(f"unsupported region kind: {axis.kind!r}")
    return tuple(spec)


def region_ndim(region: RegionSpec) -> int:
    return len(region.axes)


def region_axis_kinds(region: RegionSpec) -> tuple[str, ...]:
    return tuple(axis_region_kind(axis.kind).value for axis in region.axes)


def region_to_numpy_index(region: RegionSpec) -> tuple:
    return index_spec_from_region(region)


def normalize_region(shape, region: RegionSpec) -> RegionSpec:
    return region_from_index_spec(shape, index_spec_from_region(region))


def region_text(region: RegionSpec) -> str:
    parts = []
    for axis in region.axes:
        kind = axis_region_kind(axis.kind)
        if kind == AxisRegionKind.ALL:
            parts.append(":")
        elif kind == AxisRegionKind.POINT:
            parts.append(str(int(axis.value)))
        elif kind == AxisRegionKind.SLICE:
            start, stop, step = axis.value
            stop_text = "" if stop is None else str(int(stop))
            if int(step) == 1:
                parts.append(f"{int(start)}:{stop_text}")
            else:
                parts.append(f"{int(start)}:{stop_text}:{int(step)}")
        elif kind == AxisRegionKind.INDICES:
            parts.append("[" + ",".join(str(int(index)) for index in axis.value) + "]")
    return "[" + ", ".join(parts) + "]"


def region_axes_equal(left: RegionSpec, right: RegionSpec) -> bool:
    return tuple(left.axes) == tuple(right.axes)


def replace_region_axis(region: RegionSpec, axis: int, value: AxisRegion) -> RegionSpec:
    axes = list(region.axes)
    axes[int(axis)] = value
    return RegionSpec(tuple(axes))


def insert_region_axis(region: RegionSpec, axis: int, value: AxisRegion) -> RegionSpec:
    axes = list(region.axes)
    axes.insert(int(axis), value)
    return RegionSpec(tuple(axes))


def remove_region_axis(region: RegionSpec, axis: int) -> RegionSpec:
    axes = list(region.axes)
    axes.pop(int(axis))
    return RegionSpec(tuple(axes))


def region_shape(shape, region: RegionSpec) -> tuple[int, ...]:
    shape = tuple(int(size) for size in shape)
    result = []
    for size, axis in zip(shape, region.axes):
        kind = axis_region_kind(axis.kind)
        if kind == AxisRegionKind.POINT:
            continue
        if kind == AxisRegionKind.ALL:
            result.append(size)
        elif kind == AxisRegionKind.SLICE:
            start, stop, step = axis.value
            result.append(max(0, len(range(*slice(int(start), None if stop is None else int(stop), int(step)).indices(size)))))
        elif kind == AxisRegionKind.INDICES:
            result.append(len(tuple(axis.value)))
    return tuple(result)


def region_is_full(region: RegionSpec) -> bool:
    return all(axis_region_kind(axis.kind) == AxisRegionKind.ALL for axis in region.axes)


def region_nbytes(shape, dtype, region: RegionSpec) -> int | None:
    if dtype is None:
        return None
    return int(np.prod(region_shape(shape, region), dtype=np.int64)) * np.dtype(dtype).itemsize


def region_contains(container: RegionSpec, contained: RegionSpec, shape) -> bool:
    if len(container.axes) != len(contained.axes):
        return False
    shape = tuple(int(size) for size in shape)
    return all(
        _axis_contains(container_axis, contained_axis, size)
        for container_axis, contained_axis, size in zip(container.axes, contained.axes, shape)
    )


def apply_subregion(data, *, source_region: RegionSpec, target_region: RegionSpec, shape):
    if not region_contains(source_region, target_region, shape):
        raise ValueError(f"source region {region_text(source_region)} does not contain target {region_text(target_region)}")
    local_axes = []
    for source_axis, target_axis, size in zip(source_region.axes, target_region.axes, tuple(int(size) for size in shape)):
        if axis_region_kind(source_axis.kind) == AxisRegionKind.POINT:
            continue
        local_axes.append(_local_axis_region(source_axis, target_axis, size))
    return apply_region(data, RegionSpec(tuple(local_axes)))


def expand_region_axes(region: RegionSpec, axes: tuple[int, ...]) -> RegionSpec:
    expanded = list(region.axes)
    for axis in tuple(int(axis) for axis in axes):
        if axis < 0 or axis >= len(expanded):
            raise ValueError(f"axis {axis} is out of range for region with {len(expanded)} axes")
        expanded[axis] = AxisRegion(AxisRegionKind.ALL)
    return RegionSpec(tuple(expanded))


def apply_region(data, region: RegionSpec):
    if not any(axis_region_kind(axis.kind) == AxisRegionKind.INDICES for axis in region.axes):
        return data[index_spec_from_region(region)]
    result = data
    result_axis = 0
    for axis_region in region.axes:
        kind = axis_region_kind(axis_region.kind)
        if kind == AxisRegionKind.POINT:
            result = np.take(result, int(axis_region.value), axis=result_axis)
            continue
        if kind == AxisRegionKind.INDICES:
            result = np.take(result, np.asarray(axis_region.value, dtype=np.int64), axis=result_axis)
        else:
            slicer = [slice(None)] * np.ndim(result)
            slicer[result_axis] = index_spec_from_region(RegionSpec((axis_region,)))[0]
            result = result[tuple(slicer)]
        result_axis += 1
    return result


def take_axis_region(data, axis_region: AxisRegion, axis_size: int, *, axis: int):
    kind = axis_region_kind(axis_region.kind)
    if kind == AxisRegionKind.ALL:
        return data
    if kind == AxisRegionKind.POINT:
        return np.take(data, int(axis_region.value), axis=int(axis))
    if kind == AxisRegionKind.INDICES:
        return np.take(data, np.asarray(axis_region.value, dtype=np.int64), axis=int(axis))
    start, stop, step = axis_region.value
    indices = np.arange(int(axis_size), dtype=np.int64)[slice(int(start), None if stop is None else int(stop), int(step))]
    return np.take(data, indices, axis=int(axis))


def axis_in_region_result(region: RegionSpec, original_axis: int) -> int:
    return sum(1 for axis in range(int(original_axis)) if axis_region_kind(region.axes[axis].kind) != AxisRegionKind.POINT)


def _axis_contains(container: AxisRegion, contained: AxisRegion, size: int) -> bool:
    container_indices = _axis_indices(container, size)
    contained_indices = _axis_indices(contained, size)
    if isinstance(container_indices, int):
        if isinstance(contained_indices, int):
            return int(container_indices) == int(contained_indices)
        contained_values = tuple(int(index) for index in np.asarray(contained_indices, dtype=np.int64).ravel())
        return len(contained_values) == 1 and int(container_indices) == contained_values[0]
    container_set = set(int(index) for index in np.asarray(container_indices, dtype=np.int64).ravel())
    if isinstance(contained_indices, int):
        return int(contained_indices) in container_set
    return all(int(index) in container_set for index in np.asarray(contained_indices, dtype=np.int64).ravel())


def _local_axis_region(source_axis: AxisRegion, target_axis: AxisRegion, size: int) -> AxisRegion:
    source_indices = _axis_indices(source_axis, size)
    target_indices = _axis_indices(target_axis, size)
    if isinstance(source_indices, int):
        return AxisRegion(AxisRegionKind.POINT, 0)
    source_tuple = tuple(int(index) for index in np.asarray(source_indices, dtype=np.int64).ravel())
    positions = {index: position for position, index in enumerate(source_tuple)}
    if isinstance(target_indices, int):
        return AxisRegion(AxisRegionKind.POINT, positions[int(target_indices)])
    local = tuple(positions[int(index)] for index in np.asarray(target_indices, dtype=np.int64).ravel())
    return _axis_region_from_indices(local)


def _axis_indices(axis_region: AxisRegion, size: int):
    kind = axis_region_kind(axis_region.kind)
    size = int(size)
    if kind == AxisRegionKind.ALL:
        return np.arange(size, dtype=np.int64)
    if kind == AxisRegionKind.POINT:
        return _normalize_index(int(axis_region.value), size)
    if kind == AxisRegionKind.INDICES:
        return np.asarray(tuple(_normalize_index(index, size) for index in axis_region.value), dtype=np.int64)
    start, stop, step = axis_region.value
    return np.arange(size, dtype=np.int64)[slice(int(start), None if stop is None else int(stop), int(step))]


def _axis_region_from_indices(indices) -> AxisRegion:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return AxisRegion(AxisRegionKind.SLICE, (0, 0, 1))
    if indices.size == 1:
        return AxisRegion(AxisRegionKind.POINT, int(indices[0]))
    step = int(indices[1] - indices[0])
    if step != 0 and np.array_equal(indices, indices[0] + step * np.arange(indices.size)):
        start = int(indices[0])
        stop = int(indices[-1] + step)
        return AxisRegion(AxisRegionKind.SLICE, (start, None if stop < 0 else stop, step))
    return AxisRegion(AxisRegionKind.INDICES, tuple(int(index) for index in indices))


def _normalize_index(index: int, size: int) -> int:
    index = int(index)
    size = int(size)
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError(f"index {index} is out of range for axis size {size}")
    return index


def _is_full_slice(item) -> bool:
    return isinstance(item, slice) and item.start is None and item.stop is None and item.step is None


def _is_index_array(item) -> bool:
    return isinstance(item, np.ndarray) or isinstance(item, (tuple, list))
