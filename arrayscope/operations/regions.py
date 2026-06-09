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
        kind = AxisRegionKind(axis.kind)
        if kind == AxisRegionKind.ALL:
            spec.append(slice(None))
        elif kind == AxisRegionKind.POINT:
            spec.append(int(axis.value))
        elif kind == AxisRegionKind.SLICE:
            start, stop, step = axis.value
            spec.append(slice(int(start), int(stop), int(step)))
        elif kind == AxisRegionKind.INDICES:
            spec.append(tuple(int(index) for index in axis.value))
        else:
            raise TypeError(f"unsupported region kind: {axis.kind!r}")
    return tuple(spec)


def region_shape(shape, region: RegionSpec) -> tuple[int, ...]:
    shape = tuple(int(size) for size in shape)
    result = []
    for size, axis in zip(shape, region.axes):
        kind = AxisRegionKind(axis.kind)
        if kind == AxisRegionKind.POINT:
            continue
        if kind == AxisRegionKind.ALL:
            result.append(size)
        elif kind == AxisRegionKind.SLICE:
            start, stop, step = axis.value
            result.append(max(0, len(range(int(start), int(stop), int(step)))))
        elif kind == AxisRegionKind.INDICES:
            result.append(len(tuple(axis.value)))
    return tuple(result)


def region_is_full(region: RegionSpec) -> bool:
    return all(AxisRegionKind(axis.kind) == AxisRegionKind.ALL for axis in region.axes)


def region_nbytes(shape, dtype, region: RegionSpec) -> int | None:
    if dtype is None:
        return None
    return int(np.prod(region_shape(shape, region), dtype=np.int64)) * np.dtype(dtype).itemsize


def expand_region_axes(region: RegionSpec, axes: tuple[int, ...]) -> RegionSpec:
    expanded = list(region.axes)
    for axis in tuple(int(axis) for axis in axes):
        if axis < 0 or axis >= len(expanded):
            raise ValueError(f"axis {axis} is out of range for region with {len(expanded)} axes")
        expanded[axis] = AxisRegion(AxisRegionKind.ALL)
    return RegionSpec(tuple(expanded))


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
