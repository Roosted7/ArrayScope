"""Qt-free operation pipeline for array-derived views."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4
from typing import Protocol, Tuple

import numpy as np

from arrayscope.core.axis_info import axes_for_shape, output_axes_for_operations
from arrayscope.core.axis_utils import validate_axis
from arrayscope.operations import dim_ops
from arrayscope.operations.capabilities import OperationCapabilities, OperationKind, default_chunkable_axes
from arrayscope.operations.regions import (
    AxisRegion,
    AxisRegionKind,
    RegionSpec,
    axis_region_kind,
    axis_in_region_result,
    replace_region_axis,
    insert_region_axis,
    take_axis_region,
)


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

    def output_dtype(self, input_dtype):
        return _same_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        output_shape = self.output_shape(input_shape)
        return _capabilities(OperationKind.VIEW, ndim=len(output_shape), can_fuse=True)

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        start, stop = _validate_crop_bounds(input_shape[axis], self.start, self.stop)
        return replace_region_axis(output_region, axis, _crop_axis_region(output_region.axes[axis], start, stop))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del input_region, output_region
        return data


@dataclass(frozen=True)
class ReverseAxis:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.flip(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)

    def output_dtype(self, input_dtype):
        return _same_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        _validate_axis(input_shape, self.axis)
        return _capabilities(OperationKind.VIEW, ndim=len(input_shape), can_fuse=True)

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return replace_region_axis(output_region, axis, _reverse_axis_region(output_region.axes[axis], input_shape[axis]))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del input_region, output_region
        return data


@dataclass(frozen=True)
class Conjugate:
    def apply(self, data):
        return np.conjugate(data)

    def output_shape(self, shape: Shape) -> Shape:
        return tuple(shape)

    def output_dtype(self, input_dtype):
        return _same_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        return _capabilities(OperationKind.ELEMENTWISE, ndim=len(input_shape), can_fuse=True)

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        del input_shape
        return output_region

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del input_region, output_region
        return np.conjugate(data)


@dataclass(frozen=True)
class Mean:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.mean(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.mean(np.empty((1,), dtype=np.dtype(input_dtype))).dtype

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.REDUCTION,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return insert_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del output_region
        return np.mean(data, axis=axis_in_region_result(input_region, self.axis))


@dataclass(frozen=True)
class RootSumSquares:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.sqrt(np.sum(np.abs(data) ** 2, axis=axis))

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.asarray(np.abs(np.empty((1,), dtype=np.dtype(input_dtype)))).dtype

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.REDUCTION,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
            temp_multiplier=3.0,
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return insert_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del output_region
        return np.sqrt(np.sum(np.abs(data) ** 2, axis=axis_in_region_result(input_region, self.axis)))


@dataclass(frozen=True)
class Sum:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.sum(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.sum(np.empty((1,), dtype=np.dtype(input_dtype))).dtype

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.REDUCTION,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return insert_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del output_region
        return np.sum(data, axis=axis_in_region_result(input_region, self.axis))


@dataclass(frozen=True)
class Maximum:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.max(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)

    def output_dtype(self, input_dtype):
        return _same_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.REDUCTION,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return insert_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del output_region
        return np.max(data, axis=axis_in_region_result(input_region, self.axis))


@dataclass(frozen=True)
class Minimum:
    axis: int

    def apply(self, data):
        axis = _validate_axis(data.shape, self.axis)
        return np.min(data, axis=axis)

    def output_shape(self, shape: Shape) -> Shape:
        axis = _validate_axis(shape, self.axis)
        return _remove_axis(shape, axis)

    def output_dtype(self, input_dtype):
        return _same_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.REDUCTION,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return insert_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del output_region
        return np.min(data, axis=axis_in_region_result(input_region, self.axis))


@dataclass(frozen=True)
class CenteredFFT:
    axis: int

    def apply(self, data):
        return dim_ops.centered_fft(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.result_type(np.dtype(input_dtype), np.complex64)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.TRANSFORM,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
            temp_multiplier=6.0,
            cache_stage=True,
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return replace_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        axis = _validate_axis(input_region.axes, self.axis)
        slab_axis = axis_in_region_result(input_region, axis)
        if evaluation_context is None:
            transformed = dim_ops.centered_fft(data, slab_axis)
        else:
            transformed = dim_ops.centered_fft(data, slab_axis, workers=int(evaluation_context.fft_workers))
        return take_axis_region(transformed, output_region.axes[axis], transformed.shape[slab_axis], axis=slab_axis)


@dataclass(frozen=True)
class CenteredIFFT:
    axis: int

    def apply(self, data):
        return dim_ops.centered_ifft(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.result_type(np.dtype(input_dtype), np.complex64)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        axis = _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.TRANSFORM,
            ndim=len(input_shape),
            blocking_axes=(axis,),
            expands_request_axes=(axis,),
            temp_multiplier=6.0,
            cache_stage=True,
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return replace_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        axis = _validate_axis(input_region.axes, self.axis)
        slab_axis = axis_in_region_result(input_region, axis)
        if evaluation_context is None:
            transformed = dim_ops.centered_ifft(data, slab_axis)
        else:
            transformed = dim_ops.centered_ifft(data, slab_axis, workers=int(evaluation_context.fft_workers))
        return take_axis_region(transformed, output_region.axes[axis], transformed.shape[slab_axis], axis=slab_axis)


@dataclass(frozen=True)
class FFTShift:
    axis: int

    def apply(self, data):
        return dim_ops.apply_fftshift(data, self.axis)

    def output_shape(self, shape: Shape) -> Shape:
        _validate_axis(shape, self.axis)
        return tuple(shape)

    def output_dtype(self, input_dtype):
        return _same_dtype(input_dtype)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        _validate_axis(input_shape, self.axis)
        return _capabilities(
            OperationKind.VIEW,
            ndim=len(input_shape),
            can_fuse=True,
            notes=("Current NumPy fftshift implementation may allocate.",),
        )

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return replace_region_axis(output_region, axis, _fftshift_axis_region(output_region.axes[axis], input_shape[axis]))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        del input_region, output_region
        return data


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

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.result_type(np.dtype(input_dtype), np.complex64)

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        output_shape = self.output_shape(input_shape)
        return _capabilities(OperationKind.RESHAPE, ndim=len(output_shape))

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return replace_region_axis(output_region, axis, AxisRegion(AxisRegionKind.ALL))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        axis = _validate_axis(input_region.axes, self.axis)
        slab_axis = axis_in_region_result(input_region, axis)
        combined = dim_ops.combine_real_imag_axis(data, slab_axis)
        if axis_region_kind(output_region.axes[axis].kind) == AxisRegionKind.POINT:
            return np.take(combined, 0, axis=slab_axis)
        return take_axis_region(combined, output_region.axes[axis], 1, axis=slab_axis)


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

    def output_dtype(self, input_dtype):
        if input_dtype is None:
            return None
        return np.asarray(np.real(np.empty((1,), dtype=np.dtype(input_dtype)))).dtype

    def capabilities(self, input_shape: Shape, input_dtype=None) -> OperationCapabilities:
        output_shape = self.output_shape(input_shape)
        return _capabilities(OperationKind.RESHAPE, ndim=len(output_shape))

    def required_input_region(self, input_shape: Shape, output_region: RegionSpec) -> RegionSpec:
        axis = _validate_axis(input_shape, self.axis)
        return replace_region_axis(output_region, axis, AxisRegion(AxisRegionKind.POINT, 0))

    def apply_to_region(self, data, *, input_region: RegionSpec, output_region: RegionSpec, evaluation_context=None):
        axis = _validate_axis(output_region.axes, self.axis)
        requested = output_region.axes[axis]
        if axis_region_kind(requested.kind) == AxisRegionKind.POINT:
            return np.real(data) if int(requested.value) == 0 else np.imag(data)
        slab_axis = axis_in_region_result(output_region, axis)
        expanded = np.expand_dims(data, axis=slab_axis)
        split = dim_ops.split_complex_axis(expanded, slab_axis)
        return take_axis_region(split, requested, 2, axis=slab_axis)


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

    def mark_base_data_changed(self) -> "ArrayDocument":
        return ArrayDocument(self.base_data, steps=self.steps, revision=self.revision + 1, axes=self.base_axes)

    def reload_base_data(self, data, *, preserve_steps=True) -> "ArrayDocument":
        steps = self.steps if preserve_steps else ()
        axes = self.base_axes if np.shape(data) == np.shape(self.base_data) else None
        return ArrayDocument(data, steps=steps, revision=self.revision + 1, axes=axes)

    def replace_base_and_clear_steps(self, data) -> "ArrayDocument":
        return ArrayDocument(data, steps=(), revision=self.revision + 1)

    def with_data_changed(self, base_data=None) -> "ArrayDocument":
        if base_data is None or base_data is self.base_data:
            return self.mark_base_data_changed()
        return self.replace_base_and_clear_steps(base_data)

    def materialize(self):
        return evaluate(self.base_data, self.enabled_operations)


def evaluate(base_data, operations):
    from arrayscope.operations.optimizer import optimize_operations

    optimized = optimize_operations(np.shape(base_data), getattr(base_data, "dtype", None), operations)
    data = base_data
    for operation in optimized.operations:
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
    if hasattr(step, "operation") and hasattr(step, "enabled"):
        return OperationStep(step.operation, enabled=bool(step.enabled), step_id=str(getattr(step, "step_id", uuid4().hex)))
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


def _axis_region_indices(axis_region: AxisRegion, size: int):
    kind = axis_region_kind(axis_region.kind)
    size = int(size)
    if kind == AxisRegionKind.ALL:
        return np.arange(size, dtype=np.int64)
    if kind == AxisRegionKind.POINT:
        index = int(axis_region.value)
        if index < 0:
            index += size
        return index
    if kind == AxisRegionKind.INDICES:
        indices = np.asarray(axis_region.value, dtype=np.int64)
        return np.where(indices < 0, indices + size, indices)
    start, stop, step = axis_region.value
    return np.arange(size, dtype=np.int64)[slice(int(start), None if stop is None else int(stop), int(step))]


def _axis_region_from_indices(indices) -> AxisRegion:
    if isinstance(indices, (int, np.integer)):
        return AxisRegion(AxisRegionKind.POINT, int(indices))
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return AxisRegion(AxisRegionKind.SLICE, (0, 0, 1))
    if indices.size == 1:
        start = int(indices[0])
        return AxisRegion(AxisRegionKind.SLICE, (start, start + 1, 1))
    step = int(indices[1] - indices[0])
    if step > 0 and np.array_equal(indices, indices[0] + step * np.arange(indices.size)):
        start = int(indices[0])
        stop = int(indices[-1] + step)
        return AxisRegion(AxisRegionKind.SLICE, (start, stop, step))
    if step < 0 and np.array_equal(indices, indices[0] + step * np.arange(indices.size)):
        start = int(indices[0])
        stop = int(indices[-1] + step)
        return AxisRegion(AxisRegionKind.SLICE, (start, None if stop < 0 else stop, step))
    return AxisRegion(AxisRegionKind.INDICES, tuple(int(index) for index in indices))


def _crop_axis_region(axis_region: AxisRegion, start: int, stop: int) -> AxisRegion:
    kind = axis_region_kind(axis_region.kind)
    start = int(start)
    stop = int(stop)
    output_size = stop - start
    if kind == AxisRegionKind.ALL:
        return AxisRegion(AxisRegionKind.SLICE, (start, stop, 1))
    indices = _axis_region_indices(axis_region, output_size)
    if isinstance(indices, (int, np.integer)):
        return AxisRegion(AxisRegionKind.POINT, start + int(indices))
    return _axis_region_from_indices(np.asarray(indices, dtype=np.int64) + start)


def _reverse_axis_region(axis_region: AxisRegion, size: int) -> AxisRegion:
    indices = _axis_region_indices(axis_region, int(size))
    if isinstance(indices, (int, np.integer)):
        return AxisRegion(AxisRegionKind.POINT, int(size) - 1 - int(indices))
    return _axis_region_from_indices(int(size) - 1 - np.asarray(indices, dtype=np.int64))


def _fftshift_axis_region(axis_region: AxisRegion, size: int) -> AxisRegion:
    indices = _axis_region_indices(axis_region, int(size))
    offset = int(np.ceil(int(size) / 2.0))
    if isinstance(indices, (int, np.integer)):
        return AxisRegion(AxisRegionKind.POINT, (int(indices) + offset) % int(size))
    return _axis_region_from_indices((np.asarray(indices, dtype=np.int64) + offset) % int(size))


def _same_dtype(input_dtype):
    return None if input_dtype is None else np.dtype(input_dtype)


def _capabilities(
    kind: OperationKind,
    *,
    ndim: int,
    blocking_axes=(),
    expands_request_axes=(),
    temp_multiplier: float = 1.0,
    cache_stage: bool = False,
    can_fuse: bool = False,
    notes=(),
) -> OperationCapabilities:
    chunkable_axes = default_chunkable_axes(kind, ndim=ndim, blocking_axes=blocking_axes)
    return OperationCapabilities(
        kind=kind,
        blocking_axes=tuple(int(axis) for axis in blocking_axes),
        chunkable_axes=tuple(int(axis) for axis in chunkable_axes),
        expands_request_axes=tuple(int(axis) for axis in expands_request_axes),
        cache_stage=bool(cache_stage),
        temp_multiplier=float(temp_multiplier),
        can_fuse=bool(can_fuse),
        notes=tuple(notes),
    )
