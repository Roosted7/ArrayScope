"""Qt-free operation pipeline for array-derived views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, Tuple

import numpy as np

from arrayscope.operations import dim_ops
from arrayscope.core.axis_utils import validate_axis


Shape = Tuple[int, ...]


class ArrayOperation(Protocol):
    """Operation that can derive an array and predict its output shape."""

    def apply(self, data):
        """Return data after applying this operation."""

    def output_shape(self, shape: Shape) -> Shape:
        """Return the shape produced from an input shape."""


@dataclass(frozen=True)
class Crop:
    axis: int
    start: int
    stop: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        start, stop = _validate_crop_bounds(data.shape[axis], self.start, self.stop)
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(start, stop)
        return data[tuple(slices)]

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        start, stop = _validate_crop_bounds(shape[axis], self.start, self.stop)
        return _replace_axis(shape, axis, stop - start)


@dataclass(frozen=True)
class ReverseAxis:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.flip(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)


@dataclass(frozen=True)
class Conjugate:
    def apply(self, data):
        return np.conjugate(data)

    def output_shape(self, shape: Shape) -> Shape:
        return tuple(shape)


@dataclass(frozen=True)
class Mean:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.mean(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)


@dataclass(frozen=True)
class RootSumSquares:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.sqrt(np.sum(np.abs(data) ** 2, axis=axis))

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)


@dataclass(frozen=True)
class Sum:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.sum(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)


@dataclass(frozen=True)
class Maximum:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.max(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)


@dataclass(frozen=True)
class Minimum:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.min(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)


@dataclass(frozen=True)
class CenteredFFT:
    axis: int

    def apply(self, data):
        return dim_ops.centered_fft(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)


@dataclass(frozen=True)
class CenteredIFFT:
    axis: int

    def apply(self, data):
        return dim_ops.centered_ifft(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)


@dataclass(frozen=True)
class FFTShift:
    axis: int

    def apply(self, data):
        return dim_ops.apply_fftshift(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)


@dataclass(frozen=True)
class CombineRealImagAxis:
    axis: int

    def apply(self, data):
        return dim_ops.combine_real_imag_axis(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        if shape[axis] != 2:
            raise ValueError(f"axis {axis} must have size 2 to combine as complex")
        return _replace_axis(shape, axis, 1)


@dataclass(frozen=True)
class SplitComplexAxis:
    axis: int

    def apply(self, data):
        return dim_ops.split_complex_axis(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        if shape[axis] != 1:
            raise ValueError(f"axis {axis} must have size 1 to split to real/imag")
        return _replace_axis(shape, axis, 2)


@dataclass(frozen=True)
class ArrayDocument:
    """Base array plus an ordered list of operations for its derived view."""

    base_data: object
    operations: Tuple[ArrayOperation, ...] = field(default_factory=tuple)
    current_shape: Shape = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "current_shape", evaluate_shape(np.shape(self.base_data), self.operations))

    @property
    def shape(self) -> Shape:
        return self.current_shape

    def with_operation(self, operation: ArrayOperation) -> "ArrayDocument":
        return replace(self, operations=self.operations + (operation,))

    def without_last_operation(self) -> "ArrayDocument":
        if not self.operations:
            return self
        return replace(self, operations=self.operations[:-1])

    def materialize(self):
        return evaluate(self.base_data, self.operations)


def evaluate(base_data, operations):
    data = base_data
    for operation in operations:
        data = operation.apply(data)
    return data


def evaluate_shape(base_shape, operations) -> Shape:
    shape = tuple(int(size) for size in base_shape)
    for operation in operations:
        shape = operation.output_shape(shape)
    return shape


def _validate_axis(shape, axis) -> int:
    return validate_axis(shape, axis)


def _validate_crop_bounds(axis_size, start, stop):
    start = int(start)
    stop = int(stop)
    if start < 0 or stop < start or stop > axis_size:
        raise ValueError(f"crop bounds [{start}, {stop}) are invalid for axis size {axis_size}")
    return start, stop


def _replace_axis(shape: Shape, axis: int, size: int) -> Shape:
    updated = list(shape)
    updated[axis] = size
    return tuple(updated)


def _remove_axis(shape: Shape, axis: int) -> Shape:
    return tuple(size for index, size in enumerate(shape) if index != axis)
