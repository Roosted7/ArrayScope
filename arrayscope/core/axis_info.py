"""Qt-free axis identity metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class AxisInfo:
    id: str
    label: str
    size: int
    unit: str | None = None
    coordinate: str | None = None
    source_index: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "size", int(self.size))
        if self.size < 1:
            raise ValueError("axis size must be at least 1")
        if self.unit is not None:
            object.__setattr__(self, "unit", str(self.unit))
        if self.coordinate is not None:
            object.__setattr__(self, "coordinate", str(self.coordinate))
        if self.source_index is not None:
            object.__setattr__(self, "source_index", int(self.source_index))


AxisInfoTuple = Tuple[AxisInfo, ...]


def default_axes(shape) -> AxisInfoTuple:
    return tuple(
        AxisInfo(id=f"axis-{axis}", label=f"Dim {axis}", size=int(size), source_index=axis)
        for axis, size in enumerate(shape)
    )


def axes_for_shape(axes, shape) -> AxisInfoTuple:
    shape = tuple(int(size) for size in shape)
    if axes is None:
        return default_axes(shape)
    axes = tuple(_coerce_axis_info(axis) for axis in axes)
    if len(axes) != len(shape):
        raise ValueError("axis metadata length must match shape length")
    return tuple(replace(axis, size=size) for axis, size in zip(axes, shape))


def output_axes_for_operations(axes, operations) -> AxisInfoTuple:
    result = tuple(axes)
    for operation in operations:
        result = output_axes_for_operation(result, operation)
    return result


def output_axes_for_operation(axes, operation) -> AxisInfoTuple:
    name = type(operation).__name__
    if name in {"Crop"}:
        axis = _axis(operation, axes)
        return _replace_axis_size(axes, axis, int(operation.stop) - int(operation.start))
    if name in {"ReverseAxis", "FFTShift", "CenteredFFT", "CenteredIFFT", "Conjugate"}:
        return tuple(axes)
    if name in {"Mean", "Sum", "Maximum", "Minimum", "RootSumSquares"}:
        axis = _axis(operation, axes)
        return tuple(axis_info for index, axis_info in enumerate(axes) if index != axis)
    if name in {"CombineRealImagAxis"}:
        axis = _axis(operation, axes)
        return _replace_axis(axes, axis, replace(axes[axis], size=1, coordinate="complex"))
    if name in {"SplitComplexAxis"}:
        axis = _axis(operation, axes)
        return _replace_axis(axes, axis, replace(axes[axis], size=2, coordinate="real-imag"))
    return tuple(axes)


def _coerce_axis_info(axis) -> AxisInfo:
    if isinstance(axis, AxisInfo):
        return axis
    if isinstance(axis, dict):
        return AxisInfo(**axis)
    raise TypeError(f"unsupported axis metadata: {axis!r}")


def _axis(operation, axes) -> int:
    axis = int(operation.axis)
    if axis < 0 or axis >= len(axes):
        raise ValueError(f"axis {axis} is out of bounds for {len(axes)}D metadata")
    return axis


def _replace_axis_size(axes, axis, size):
    return _replace_axis(axes, axis, replace(axes[axis], size=int(size)))


def _replace_axis(axes, axis, axis_info):
    result = list(axes)
    result[int(axis)] = axis_info
    return tuple(result)

