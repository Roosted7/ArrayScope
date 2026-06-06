"""Shared axis and index validation helpers."""

from __future__ import annotations


def _ndim(shape_or_ndim) -> int:
    if isinstance(shape_or_ndim, int):
        return int(shape_or_ndim)
    return len(tuple(shape_or_ndim))


def validate_axis(shape_or_ndim, axis, *, label: str = "axis") -> int:
    axis = int(axis)
    ndim = _ndim(shape_or_ndim)
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{label} {axis} is out of bounds for {ndim}D data")
    return axis


def validate_distinct_axes(shape_or_ndim, axes, *, count: int | None = None) -> tuple[int, ...]:
    axes = tuple(validate_axis(shape_or_ndim, axis) for axis in axes)
    if count is not None and len(axes) != int(count):
        raise ValueError(f"axes must contain exactly {count} axes")
    if len(set(axes)) != len(axes):
        raise ValueError("axes must be distinct")
    return axes


def clamp_index(shape, axis, index) -> int:
    shape = tuple(int(size) for size in shape)
    axis = validate_axis(shape, axis)
    size = shape[axis]
    if size < 1:
        raise ValueError(f"axis {axis} has invalid size {size}")
    return max(0, min(int(index), size - 1))


def non_singleton_axes(shape) -> tuple[int, ...]:
    return tuple(axis for axis, size in enumerate(tuple(shape)) if int(size) != 1)
