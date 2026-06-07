"""Qt-free operation pipeline for array-derived views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4
from typing import Protocol, Tuple

import numpy as np

from arrayscope.core.axis_info import axes_for_shape, output_axes_for_operations
from arrayscope.core.axis_utils import validate_axis
from arrayscope.operations import dim_ops


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
class OperationStep:
    operation: ArrayOperation
    enabled: bool = True
    step_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True, init=False)
class ArrayDocument:
    """Base array plus an ordered list of operations for its derived view."""

    base_data: object
    steps: Tuple[OperationStep, ...] = field(default_factory=tuple)
    revision: int = 0
    base_axes: tuple = field(default_factory=tuple)
    current_axes: tuple = field(init=False)
    current_shape: Shape = field(init=False)

    def __init__(self, base_data, operations=(), *, steps=None, revision=0, axes=None):
        object.__setattr__(self, "base_data", base_data)
        if steps is None:
            steps = tuple(OperationStep(operation) for operation in operations)
        else:
            steps = tuple(_coerce_step(step) for step in steps)
        base_shape = np.shape(self.base_data)
        base_axes = axes_for_shape(axes, base_shape)
        current_shape = evaluate_shape(base_shape, tuple(step.operation for step in steps if step.enabled))
        current_axes = output_axes_for_operations(base_axes, tuple(step.operation for step in steps if step.enabled))
        if tuple(axis.size for axis in current_axes) != tuple(current_shape):
            current_axes = axes_for_shape(current_axes, current_shape)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "revision", int(revision))
        object.__setattr__(self, "base_axes", base_axes)
        object.__setattr__(self, "current_shape", current_shape)
        object.__setattr__(self, "current_axes", current_axes)

    @property
    def shape(self) -> Shape:
        return self.current_shape

    @property
    def operations(self) -> Tuple[ArrayOperation, ...]:
        return self.enabled_operations

    @property
    def enabled_operations(self) -> Tuple[ArrayOperation, ...]:
        return tuple(step.operation for step in self.steps if step.enabled)

    def with_operation(self, operation: ArrayOperation) -> "ArrayDocument":
        return ArrayDocument(self.base_data, steps=self.steps + (OperationStep(operation),), revision=self.revision, axes=self.base_axes)

    def without_last_operation(self) -> "ArrayDocument":
        if not self.steps:
            return self
        return ArrayDocument(self.base_data, steps=self.steps[:-1], revision=self.revision, axes=self.base_axes)

    def with_step_enabled(self, index, enabled) -> "ArrayDocument":
        index = _validate_step_index(index, self.steps)
        steps = list(self.steps)
        steps[index] = replace(steps[index], enabled=bool(enabled))
        return ArrayDocument(self.base_data, steps=tuple(steps), revision=self.revision, axes=self.base_axes)

    def with_replaced_operation(self, index, operation: ArrayOperation) -> "ArrayDocument":
        index = _validate_step_index(index, self.steps)
        steps = list(self.steps)
        steps[index] = replace(steps[index], operation=operation)
        return ArrayDocument(self.base_data, steps=tuple(steps), revision=self.revision, axes=self.base_axes)

    def with_data_changed(self, base_data=None) -> "ArrayDocument":
        data = self.base_data if base_data is None else base_data
        axes = self.base_axes if data is self.base_data else None
        steps = self.steps if data is self.base_data else ()
        return ArrayDocument(data, steps=steps, revision=self.revision + 1, axes=axes)

    def materialize(self):
        return evaluate(self.base_data, self.enabled_operations)


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


def _coerce_step(step) -> OperationStep:
    if isinstance(step, OperationStep):
        return step
    return OperationStep(step)


def _validate_step_index(index, steps):
    index = int(index)
    if index < 0 or index >= len(steps):
        raise IndexError("operation step index is out of range")
    return index


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
