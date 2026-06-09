"""Conservative operation shape, dtype, and memory-cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from arrayscope.operations.capabilities import OperationCapabilities, OperationKind as DeclaredOperationKind, normalize_capabilities

OperationKind = Literal["view", "elementwise", "reduction", "transform", "reshape"]


@dataclass(frozen=True)
class OperationCost:
    kind: OperationKind
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: np.dtype | None
    output_dtype: np.dtype | None
    requires_full_axis: tuple[int, ...] = ()
    estimated_input_bytes: int | None = None
    estimated_output_bytes: int | None = None
    estimated_peak_bytes: int | None = None
    temp_multiplier: float = 1.0
    can_chunk: bool = False
    chunkable_axes: tuple[int, ...] = ()
    blocking_axes: tuple[int, ...] = ()
    expands_request_axes: tuple[int, ...] = ()
    cache_stage: bool = False
    can_fuse: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineCost:
    operation_costs: tuple[OperationCost, ...]
    output_shape: tuple[int, ...]
    output_dtype: np.dtype | None
    estimated_peak_bytes: int | None
    estimated_output_bytes: int | None
    chunkable_axes: tuple[int, ...] = ()
    blocking_axes: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()
    optimization_steps: tuple[str, ...] = ()
    original_operation_count: int = 0
    optimized_operation_count: int = 0


def operation_output_dtype(input_dtype, operation) -> np.dtype | None:
    if input_dtype is None:
        return None
    if hasattr(operation, "output_dtype"):
        output = operation.output_dtype(np.dtype(input_dtype))
        return None if output is None else np.dtype(output)
    return np.dtype(input_dtype)


def estimate_operation_cost(input_shape, input_dtype, operation) -> OperationCost:
    input_shape = tuple(int(size) for size in input_shape)
    input_dtype = None if input_dtype is None else np.dtype(input_dtype)
    output_shape = tuple(int(size) for size in operation.output_shape(input_shape))
    output_dtype = operation_output_dtype(input_dtype, operation)
    input_bytes = _array_bytes(input_shape, input_dtype)
    output_bytes = _array_bytes(output_shape, output_dtype)
    capabilities = _operation_capabilities(operation, input_shape, input_dtype, output_shape=output_shape)
    peak = _estimate_peak_bytes(capabilities, input_bytes=input_bytes, output_bytes=output_bytes)
    notes = tuple(capabilities.notes)
    for axis in capabilities.expands_request_axes:
        label = "transform axis" if capabilities.kind == DeclaredOperationKind.TRANSFORM else "axis"
        notes += (f"Requires full {label} {axis}.",)
    return _cost(
        capabilities.kind.value,
        input_shape,
        output_shape,
        input_dtype,
        output_dtype,
        input_bytes,
        output_bytes,
        peak,
        requires_full_axis=capabilities.expands_request_axes,
        temp_multiplier=capabilities.temp_multiplier,
        can_chunk=bool(capabilities.chunkable_axes),
        chunkable_axes=capabilities.chunkable_axes,
        blocking_axes=capabilities.blocking_axes,
        expands_request_axes=capabilities.expands_request_axes,
        cache_stage=capabilities.cache_stage,
        can_fuse=capabilities.can_fuse,
        notes=notes,
    )


def estimate_pipeline_cost(base_shape, base_dtype, operations, *, optimize: bool = True) -> PipelineCost:
    original_operations = tuple(operations)
    optimization_steps = ()
    if optimize:
        from arrayscope.operations.optimizer import format_optimization_steps, optimize_operations

        optimized = optimize_operations(base_shape, base_dtype, original_operations)
        operations = optimized.operations
        optimization_steps = format_optimization_steps(optimized.steps)
    else:
        operations = original_operations
    shape = tuple(int(size) for size in base_shape)
    dtype = None if base_dtype is None else np.dtype(base_dtype)
    costs = []
    warnings = []
    peak = None
    axis_labels = tuple(range(len(shape)))
    blocking_labels = set()
    for operation in tuple(operations):
        cost = estimate_operation_cost(shape, dtype, operation)
        costs.append(cost)
        for axis in cost.blocking_axes:
            if 0 <= int(axis) < len(axis_labels):
                blocking_labels.add(axis_labels[int(axis)])
        axis_labels = _output_axis_labels(axis_labels, operation, cost)
        shape = cost.output_shape
        dtype = cost.output_dtype
        if cost.estimated_peak_bytes is not None:
            peak = cost.estimated_peak_bytes if peak is None else max(peak, cost.estimated_peak_bytes)
        if cost.requires_full_axis:
            warnings.append(f"{type(operation).__name__} requires full axis {cost.requires_full_axis[0]}.")
    output_bytes = _array_bytes(shape, dtype)
    final_blocking_axes = tuple(index for index, label in enumerate(axis_labels) if label in blocking_labels)
    chunkable_axes = tuple(index for index in range(len(shape)) if index not in set(final_blocking_axes))
    return PipelineCost(
        operation_costs=tuple(costs),
        output_shape=shape,
        output_dtype=dtype,
        estimated_peak_bytes=peak,
        estimated_output_bytes=output_bytes,
        chunkable_axes=chunkable_axes,
        blocking_axes=final_blocking_axes,
        warnings=tuple(warnings),
        optimization_steps=tuple(optimization_steps),
        original_operation_count=len(original_operations),
        optimized_operation_count=len(tuple(operations)),
    )


def format_operation_cost(cost: OperationCost) -> str:
    dtype = "unknown" if cost.output_dtype is None else str(cost.output_dtype)
    peak = "unknown" if cost.estimated_peak_bytes is None else _format_bytes(cost.estimated_peak_bytes)
    return f"{cost.kind} -> shape {cost.output_shape}, dtype {dtype}, peak {peak}"


def _cost(
    kind,
    input_shape,
    output_shape,
    input_dtype,
    output_dtype,
    input_bytes,
    output_bytes,
    peak_bytes,
    *,
    requires_full_axis=(),
    temp_multiplier=1.0,
    can_chunk=False,
    chunkable_axes=None,
    blocking_axes=(),
    expands_request_axes=(),
    cache_stage=False,
    can_fuse=False,
    notes=(),
):
    if chunkable_axes is None:
        chunkable_axes = tuple(range(len(output_shape))) if can_chunk or kind in {"view", "elementwise", "reshape", "reduction"} else ()
    return OperationCost(
        kind=kind,
        input_shape=input_shape,
        output_shape=output_shape,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        requires_full_axis=tuple(requires_full_axis),
        estimated_input_bytes=input_bytes,
        estimated_output_bytes=output_bytes,
        estimated_peak_bytes=peak_bytes,
        temp_multiplier=float(temp_multiplier),
        can_chunk=bool(can_chunk or chunkable_axes),
        chunkable_axes=tuple(chunkable_axes),
        blocking_axes=tuple(blocking_axes),
        expands_request_axes=tuple(expands_request_axes),
        cache_stage=bool(cache_stage),
        can_fuse=bool(can_fuse),
        notes=tuple(notes),
    )


def _operation_capabilities(operation, input_shape, input_dtype, *, output_shape) -> OperationCapabilities:
    if hasattr(operation, "capabilities"):
        return normalize_capabilities(operation.capabilities(input_shape, input_dtype), ndim=len(input_shape))
    return OperationCapabilities(
        kind=DeclaredOperationKind.ELEMENTWISE,
        chunkable_axes=tuple(range(len(output_shape))),
        can_fuse=True,
    )


def _estimate_peak_bytes(capabilities: OperationCapabilities, *, input_bytes, output_bytes) -> int | None:
    kind = capabilities.kind
    if kind == DeclaredOperationKind.VIEW:
        return output_bytes
    if kind == DeclaredOperationKind.TRANSFORM:
        return None if output_bytes is None else int(output_bytes * capabilities.temp_multiplier)
    if kind == DeclaredOperationKind.REDUCTION and capabilities.temp_multiplier > 1.0:
        return None if input_bytes is None or output_bytes is None else int(input_bytes * capabilities.temp_multiplier + output_bytes)
    return _sum_known(input_bytes, output_bytes)


def _output_axis_labels(axis_labels, operation, cost: OperationCost):
    labels = list(axis_labels)
    removed = len(cost.input_shape) - len(cost.output_shape)
    if removed == 1 and cost.blocking_axes:
        axis = int(cost.blocking_axes[0])
        if 0 <= axis < len(labels):
            labels.pop(axis)
    return tuple(labels)


def _array_bytes(shape, dtype) -> int | None:
    if dtype is None:
        return None
    return int(np.prod(tuple(int(size) for size in shape), dtype=np.int64)) * np.dtype(dtype).itemsize


def _sum_known(*values) -> int | None:
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values))


def _format_bytes(nbytes: int) -> str:
    value = float(int(nbytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
