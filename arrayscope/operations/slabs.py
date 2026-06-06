"""Lazy slab planning and evaluation for operation-backed views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from arrayscope.operations import dim_ops
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CenteredIFFT,
    CombineRealImagAxis,
    Conjugate,
    Crop,
    FFTShift,
    Maximum,
    Mean,
    Minimum,
    ReverseAxis,
    RootSumSquares,
    SplitComplexAxis,
    Sum,
)


@dataclass(frozen=True)
class SlabRequest:
    kind: Literal["image", "line", "scalar", "export_frame"]
    view_state: object
    keep_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]
    frame_axis: int | None = None
    frame_index: int | None = None


@dataclass(frozen=True)
class SlabPlan:
    base_index: tuple
    base_shape: tuple[int, ...]
    slab_ops: tuple
    output_shape: tuple[int, ...]
    estimated_nbytes: int | None = None
    requires_full_materialization: bool = False


def request_for_image(view_state) -> SlabRequest:
    return SlabRequest("image", view_state, tuple(view_state.image_axes or ()), tuple(view_state.slice_indices))


def request_for_line(view_state) -> SlabRequest:
    keep_axes = () if view_state.line_axis is None else (int(view_state.line_axis),)
    return SlabRequest("line", view_state, keep_axes, tuple(view_state.slice_indices))


def request_for_scalar(view_state, index) -> SlabRequest:
    slices = list(view_state.slice_indices)
    for axis, value in enumerate(index):
        slices[axis] = int(value)
    return SlabRequest("scalar", view_state, (), tuple(slices))


def request_for_export_frame(view_state, frame_axis, frame_index) -> SlabRequest:
    slices = list(view_state.slice_indices)
    slices[int(frame_axis)] = int(frame_index)
    return SlabRequest(
        "export_frame",
        view_state,
        tuple(view_state.image_axes or ()),
        tuple(slices),
        frame_axis=int(frame_axis),
        frame_index=int(frame_index),
    )


def plan_slab(document: ArrayDocument, request: SlabRequest) -> SlabPlan:
    spec = _spec_for_request(document.current_shape, request)
    base_spec = _base_spec_for_ops(document.enabled_operations, np.shape(document.base_data), spec)
    base_shape = _shape_after_index(np.shape(document.base_data), base_spec)
    dtype = getattr(document.base_data, "dtype", None)
    estimated = None if dtype is None else int(np.prod(base_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    output_shape = _shape_after_index(document.current_shape, spec)
    return SlabPlan(
        base_index=tuple(base_spec),
        base_shape=tuple(int(v) for v in base_shape),
        slab_ops=tuple(document.enabled_operations),
        output_shape=tuple(int(v) for v in output_shape),
        estimated_nbytes=estimated,
    )


def evaluate_slab(document: ArrayDocument, request: SlabRequest):
    spec = _spec_for_request(document.current_shape, request)
    return _evaluate_ops(document.base_data, tuple(document.enabled_operations), spec)


def _spec_for_request(shape, request):
    keep_axes = set(int(axis) for axis in request.keep_axes)
    spec = []
    for axis, size in enumerate(shape):
        if axis in keep_axes:
            spec.append(slice(None))
        else:
            index = int(request.slice_indices[axis])
            if index < 0 or index >= int(size):
                raise IndexError(f"slice index {index} is out of range for axis {axis} with size {size}")
            spec.append(index)
    return tuple(spec)


def _evaluate_ops(data, operations, desired_spec):
    if not operations:
        return data[tuple(desired_spec)]

    previous_ops = operations[:-1]
    operation = operations[-1]
    input_shape = _shape_after_ops(np.shape(data), previous_ops)
    output_spec = tuple(desired_spec)

    if _is(operation, Crop):
        input_spec = list(output_spec)
        axis = int(operation.axis)
        input_spec[axis] = _crop_spec(input_spec[axis], int(operation.start), int(operation.stop))
        return _evaluate_ops(data, previous_ops, tuple(input_spec))

    if _is(operation, ReverseAxis):
        input_spec = list(output_spec)
        axis = int(operation.axis)
        input_spec[axis] = _reverse_spec(input_spec[axis], input_shape[axis])
        return _evaluate_ops(data, previous_ops, tuple(input_spec))

    if _is(operation, FFTShift):
        axis = int(operation.axis)
        if isinstance(output_spec[axis], int):
            input_spec = list(output_spec)
            input_spec[axis] = _fftshift_source_index(output_spec[axis], input_shape[axis])
            return _evaluate_ops(data, previous_ops, tuple(input_spec))
        slab = _evaluate_ops(data, previous_ops, output_spec)
        return dim_ops.apply_fftshift(slab, _axis_in_result(output_spec, axis))

    if _is(operation, CenteredFFT, CenteredIFFT):
        axis = int(operation.axis)
        if isinstance(output_spec[axis], int):
            input_spec = list(output_spec)
            requested_index = int(output_spec[axis])
            input_spec[axis] = slice(None)
            slab = _evaluate_ops(data, previous_ops, tuple(input_spec))
            slab_axis = _axis_in_result(tuple(input_spec), axis)
            if _is(operation, CenteredFFT):
                transformed = dim_ops.centered_fft(slab, slab_axis)
            else:
                transformed = dim_ops.centered_ifft(slab, slab_axis)
            return np.take(transformed, requested_index, axis=slab_axis)
        slab = _evaluate_ops(data, previous_ops, output_spec)
        slab_axis = _axis_in_result(output_spec, axis)
        if _is(operation, CenteredFFT):
            return dim_ops.centered_fft(slab, slab_axis)
        return dim_ops.centered_ifft(slab, slab_axis)

    if _is(operation, Conjugate):
        return np.conjugate(_evaluate_ops(data, previous_ops, output_spec))

    if _is(operation, Mean, Sum, Maximum, Minimum, RootSumSquares):
        axis = int(operation.axis)
        input_spec = list(output_spec)
        input_spec.insert(axis, slice(None))
        slab = _evaluate_ops(data, previous_ops, tuple(input_spec))
        slab_axis = _axis_in_result(tuple(input_spec), axis)
        if _is(operation, Mean):
            return np.mean(slab, axis=slab_axis)
        if _is(operation, Sum):
            return np.sum(slab, axis=slab_axis)
        if _is(operation, Maximum):
            return np.max(slab, axis=slab_axis)
        if _is(operation, Minimum):
            return np.min(slab, axis=slab_axis)
        return np.sqrt(np.sum(np.abs(slab) ** 2, axis=slab_axis))

    if _is(operation, CombineRealImagAxis):
        axis = int(operation.axis)
        input_spec = list(output_spec)
        output_axis_was_scalar = isinstance(input_spec[axis], int)
        input_spec[axis] = slice(None)
        slab = _evaluate_ops(data, previous_ops, tuple(input_spec))
        combined = dim_ops.combine_real_imag_axis(slab, _axis_in_result(tuple(input_spec), axis))
        if output_axis_was_scalar:
            return np.take(combined, 0, axis=_axis_in_result(tuple(input_spec), axis))
        return combined

    if _is(operation, SplitComplexAxis):
        axis = int(operation.axis)
        input_spec = list(output_spec)
        requested = input_spec[axis]
        input_spec[axis] = 0
        slab = _evaluate_ops(data, previous_ops, tuple(input_spec))
        if isinstance(requested, int):
            return np.real(slab) if int(requested) == 0 else np.imag(slab)
        expanded = np.expand_dims(slab, axis=_axis_in_result(tuple(output_spec), axis))
        return dim_ops.split_complex_axis(expanded, _axis_in_result(tuple(output_spec), axis))

    materialized = data
    for op in operations:
        materialized = op.apply(materialized)
    return materialized[output_spec]


def _base_spec_for_ops(operations, base_shape, desired_spec):
    if not operations:
        return tuple(desired_spec)
    previous_ops = operations[:-1]
    operation = operations[-1]
    input_shape = _shape_after_ops(base_shape, previous_ops)
    output_spec = tuple(desired_spec)

    if _is(operation, Crop):
        input_spec = list(output_spec)
        input_spec[int(operation.axis)] = _crop_spec(input_spec[int(operation.axis)], int(operation.start), int(operation.stop))
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, ReverseAxis):
        axis = int(operation.axis)
        input_spec = list(output_spec)
        input_spec[axis] = _reverse_spec(input_spec[axis], input_shape[axis])
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, FFTShift) and isinstance(output_spec[int(operation.axis)], int):
        axis = int(operation.axis)
        input_spec = list(output_spec)
        input_spec[axis] = _fftshift_source_index(input_spec[axis], input_shape[axis])
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, CenteredFFT, CenteredIFFT) and isinstance(output_spec[int(operation.axis)], int):
        input_spec = list(output_spec)
        input_spec[int(operation.axis)] = slice(None)
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, Mean, Sum, Maximum, Minimum, RootSumSquares):
        input_spec = list(output_spec)
        input_spec.insert(int(operation.axis), slice(None))
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, CombineRealImagAxis):
        input_spec = list(output_spec)
        input_spec[int(operation.axis)] = slice(None)
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, SplitComplexAxis):
        input_spec = list(output_spec)
        input_spec[int(operation.axis)] = 0
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    return _base_spec_for_ops(previous_ops, base_shape, output_spec)


def _shape_after_ops(shape, operations):
    result = tuple(int(size) for size in shape)
    for operation in operations:
        result = operation.output_shape(result)
    return result


def _shape_after_index(shape, index):
    result = []
    for size, item in zip(shape, index):
        if isinstance(item, int):
            continue
        start, stop, step = item.indices(int(size))
        result.append(max(0, len(range(start, stop, step))))
    return tuple(result)


def _offset_spec(item, offset):
    if isinstance(item, int):
        return int(item) + offset
    start = offset if item.start is None else int(item.start) + offset
    stop = None if item.stop is None else int(item.stop) + offset
    return slice(start, stop, item.step)


def _crop_spec(item, start_offset, stop_offset):
    if isinstance(item, int):
        return int(item) + start_offset
    if item.start is None and item.stop is None and item.step is None:
        return slice(start_offset, stop_offset)
    return _offset_spec(item, start_offset)


def _reverse_spec(item, size):
    if isinstance(item, int):
        return int(size) - 1 - int(item)
    return slice(None, None, -1)


def _fftshift_source_index(index, size):
    return (int(index) + int(np.ceil(int(size) / 2.0))) % int(size)


def _axis_in_result(spec, original_axis):
    return sum(1 for axis in range(int(original_axis)) if not isinstance(spec[axis], int))


def _is(operation, *classes):
    names = {cls.__name__ for cls in classes}
    return isinstance(operation, classes) or type(operation).__name__ in names
