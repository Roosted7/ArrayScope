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
from arrayscope.operations.planner import build_region_plan
from arrayscope.operations.regions import region_from_index_spec


@dataclass(frozen=True)
class SlabRequest:
    kind: Literal["image", "line", "scalar", "export_frame"]
    view_state: object
    keep_axes: tuple[int, ...]
    slice_indices: tuple[int, ...]
    ranged_axes: tuple[int, ...] = ()
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
    region_plan: object | None = None


def request_for_image(view_state) -> SlabRequest:
    keep_axes = tuple(view_state.image_axes or ())
    return SlabRequest("image", view_state, keep_axes, tuple(view_state.slice_indices), _ranged_keep_axes(view_state, keep_axes))


def request_for_line(view_state) -> SlabRequest:
    keep_axes = () if view_state.line_axis is None else (int(view_state.line_axis),)
    return SlabRequest("line", view_state, keep_axes, tuple(view_state.slice_indices), _ranged_keep_axes(view_state, keep_axes))


def request_for_scalar(view_state, index) -> SlabRequest:
    slices = list(view_state.slice_indices)
    for axis, value in enumerate(index):
        slices[axis] = int(value)
    return SlabRequest("scalar", view_state, (), tuple(slices))


def request_for_export_frame(view_state, frame_axis, frame_index) -> SlabRequest:
    slices = list(view_state.slice_indices)
    slices[int(frame_axis)] = int(frame_index)
    keep_axes = tuple(view_state.image_axes or ())
    return SlabRequest(
        "export_frame",
        view_state,
        keep_axes,
        tuple(slices),
        _ranged_keep_axes(view_state, keep_axes),
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
    final_region = region_from_index_spec(document.current_shape, spec)
    required_input_region = region_from_index_spec(np.shape(document.base_data), base_spec)
    return SlabPlan(
        base_index=tuple(base_spec),
        base_shape=tuple(int(v) for v in base_shape),
        slab_ops=tuple(document.enabled_operations),
        output_shape=tuple(int(v) for v in output_shape),
        estimated_nbytes=estimated,
        region_plan=build_region_plan(
            request_kind=request.kind,
            base_shape=np.shape(document.base_data),
            base_dtype=dtype,
            operations=document.enabled_operations,
            final_region=final_region,
            required_input_region=required_input_region,
        ),
    )


def evaluate_slab(document: ArrayDocument, request: SlabRequest):
    spec = _spec_for_request(document.current_shape, request)
    return _evaluate_ops(document.base_data, tuple(document.enabled_operations), spec)


def _spec_for_request(shape, request):
    keep_axes = set(int(axis) for axis in request.keep_axes)
    spec = []
    for axis, size in enumerate(shape):
        if axis in keep_axes:
            spec.append(_axis_item_for_keep_axis(request.view_state, axis))
        else:
            index = int(request.slice_indices[axis])
            if index < 0 or index >= int(size):
                raise IndexError(f"slice index {index} is out of range for axis {axis} with size {size}")
            spec.append(index)
    return tuple(spec)


def _ranged_keep_axes(view_state, keep_axes):
    axis_ranges = getattr(view_state, "axis_range_indices", ())
    ranged = []
    for axis in keep_axes:
        if axis < len(axis_ranges) and axis_ranges[int(axis)] is not None:
            ranged.append(int(axis))
    return tuple(ranged)


def _axis_item_for_keep_axis(view_state, axis):
    axis_ranges = getattr(view_state, "axis_range_indices", ())
    if int(axis) >= len(axis_ranges):
        return slice(None)
    indices = axis_ranges[int(axis)]
    if indices is None:
        return slice(None)
    return _sequence_to_basic_index(np.asarray(indices, dtype=np.int64))


def _evaluate_ops(data, operations, desired_spec):
    # Keep this proven execution path while RegionPlan metadata is introduced.
    # A later planner integration will move operation-specific region transitions
    # behind planner-owned contracts.
    if not operations:
        return _apply_index(data, tuple(desired_spec))

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
        input_spec = list(output_spec)
        input_spec[axis] = _fftshift_spec(output_spec[axis], input_shape[axis])
        if _can_remap_without_post_op(output_spec[axis]):
            return _evaluate_ops(data, previous_ops, tuple(input_spec))
        slab = _evaluate_ops(data, previous_ops, output_spec)
        return dim_ops.apply_fftshift(slab, _axis_in_result(output_spec, axis))

    if _is(operation, CenteredFFT, CenteredIFFT):
        axis = int(operation.axis)
        if not _is_full_slice(output_spec[axis]):
            input_spec = list(output_spec)
            requested_item = output_spec[axis]
            input_spec[axis] = slice(None)
            slab = _evaluate_ops(data, previous_ops, tuple(input_spec))
            slab_axis = _axis_in_result(tuple(input_spec), axis)
            if _is(operation, CenteredFFT):
                transformed = dim_ops.centered_fft(slab, slab_axis)
            else:
                transformed = dim_ops.centered_ifft(slab, slab_axis)
            return _take_item(transformed, requested_item, input_shape[axis], axis=slab_axis)
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
    if _is(operation, FFTShift):
        axis = int(operation.axis)
        input_spec = list(output_spec)
        input_spec[axis] = _fftshift_spec(input_spec[axis], input_shape[axis])
        return _base_spec_for_ops(previous_ops, base_shape, tuple(input_spec))
    if _is(operation, CenteredFFT, CenteredIFFT) and not _is_full_slice(output_spec[int(operation.axis)]):
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
        if _is_index_array(item):
            result.append(int(np.asarray(item).size))
            continue
        start, stop, step = item.indices(int(size))
        result.append(max(0, len(range(start, stop, step))))
    return tuple(result)


def _crop_spec(item, start_offset, stop_offset):
    output_size = int(stop_offset) - int(start_offset)
    indices = _indices_for_item(item, output_size)
    if isinstance(indices, int):
        return int(start_offset) + indices
    return _sequence_to_basic_index(np.asarray(indices, dtype=np.int64) + int(start_offset))


def _reverse_spec(item, size):
    indices = _indices_for_item(item, int(size))
    if isinstance(indices, int):
        return int(size) - 1 - indices
    return _sequence_to_basic_index(int(size) - 1 - np.asarray(indices, dtype=np.int64))


def _fftshift_source_index(index, size):
    return (int(index) + int(np.ceil(int(size) / 2.0))) % int(size)


def _fftshift_spec(item, size):
    indices = _indices_for_item(item, int(size))
    offset = int(np.ceil(int(size) / 2.0))
    if isinstance(indices, int):
        return (indices + offset) % int(size)
    return _sequence_to_basic_index((np.asarray(indices, dtype=np.int64) + offset) % int(size))


def _indices_for_item(item, size):
    size = int(size)
    if isinstance(item, int):
        index = int(item)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(f"index {item} is out of range for axis size {size}")
        return index
    if _is_index_array(item):
        indices = np.asarray(item, dtype=np.int64)
        indices = np.where(indices < 0, indices + size, indices)
        if np.any((indices < 0) | (indices >= size)):
            raise IndexError(f"index array is out of range for axis size {size}")
        return indices
    return np.arange(size, dtype=np.int64)[item]


def _sequence_to_basic_index(indices):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        return indices
    if indices.size == 0:
        return slice(0, 0, 1)
    if indices.size == 1:
        start = int(indices[0])
        return slice(start, start + 1, 1)
    step = int(indices[1] - indices[0])
    if step != 0 and np.array_equal(indices, indices[0] + step * np.arange(indices.size)):
        start = int(indices[0])
        stop = int(indices[-1] + step)
        if stop < 0:
            stop = None
        return slice(start, stop, step)
    return indices


def _apply_index(data, spec):
    if not any(_is_index_array(item) for item in spec):
        return data[tuple(spec)]
    result = data
    result_axis = 0
    for item in spec:
        if isinstance(item, int):
            result = np.take(result, int(item), axis=result_axis)
            continue
        if _is_index_array(item):
            result = np.take(result, np.asarray(item, dtype=np.int64), axis=result_axis)
        else:
            slicer = [slice(None)] * np.ndim(result)
            slicer[result_axis] = item
            result = result[tuple(slicer)]
        result_axis += 1
    return result


def _take_item(data, item, axis_size, *, axis):
    indices = _indices_for_item(item, int(axis_size))
    if isinstance(indices, int):
        return np.take(data, indices, axis=axis)
    return np.take(data, indices, axis=axis)


def _is_index_array(item):
    return isinstance(item, np.ndarray)


def _is_full_slice(item):
    return isinstance(item, slice) and item.start is None and item.stop is None and item.step is None


def _can_remap_without_post_op(_item):
    return True


def _axis_in_result(spec, original_axis):
    return sum(1 for axis in range(int(original_axis)) if not isinstance(spec[axis], int))


def _is(operation, *classes):
    names = {cls.__name__ for cls in classes}
    return isinstance(operation, classes) or type(operation).__name__ in names
