"""Conservative operation shape, dtype, and memory-cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

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
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineCost:
    operation_costs: tuple[OperationCost, ...]
    output_shape: tuple[int, ...]
    output_dtype: np.dtype | None
    estimated_peak_bytes: int | None
    estimated_output_bytes: int | None
    warnings: tuple[str, ...] = ()


class OperationCapabilities(Protocol):
    def output_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]: ...


def operation_output_dtype(input_dtype, operation) -> np.dtype | None:
    if input_dtype is None:
        return None
    dtype = np.dtype(input_dtype)
    name = type(operation).__name__
    if name in {"Crop", "ReverseAxis", "FFTShift", "Conjugate", "Maximum", "Minimum"}:
        return dtype
    if name == "Mean":
        return np.mean(np.empty((1,), dtype=dtype)).dtype
    if name == "Sum":
        return np.sum(np.empty((1,), dtype=dtype)).dtype
    if name == "RootSumSquares":
        return np.asarray(np.abs(np.empty((1,), dtype=dtype))).dtype
    if name in {"CenteredFFT", "CenteredIFFT", "CombineRealImagAxis"}:
        return np.result_type(dtype, np.complex64)
    if name == "SplitComplexAxis":
        return np.asarray(np.real(np.empty((1,), dtype=dtype))).dtype
    return dtype


def estimate_operation_cost(input_shape, input_dtype, operation) -> OperationCost:
    input_shape = tuple(int(size) for size in input_shape)
    input_dtype = None if input_dtype is None else np.dtype(input_dtype)
    output_shape = tuple(int(size) for size in operation.output_shape(input_shape))
    output_dtype = operation_output_dtype(input_dtype, operation)
    input_bytes = _array_bytes(input_shape, input_dtype)
    output_bytes = _array_bytes(output_shape, output_dtype)
    name = type(operation).__name__
    axis = int(getattr(operation, "axis", 0)) if hasattr(operation, "axis") else None

    if name == "Crop":
        return _cost("view", input_shape, output_shape, input_dtype, output_dtype, input_bytes, output_bytes, output_bytes, can_chunk=True)
    if name == "ReverseAxis":
        return _cost("view", input_shape, output_shape, input_dtype, output_dtype, input_bytes, output_bytes, output_bytes, can_chunk=True)
    if name == "FFTShift":
        return _cost(
            "view",
            input_shape,
            output_shape,
            input_dtype,
            output_dtype,
            input_bytes,
            output_bytes,
            output_bytes,
            can_chunk=True,
            notes=("Current NumPy fftshift implementation may allocate.",),
        )
    if name == "Conjugate":
        return _cost("elementwise", input_shape, output_shape, input_dtype, output_dtype, input_bytes, output_bytes, _sum_known(input_bytes, output_bytes))
    if name in {"CombineRealImagAxis", "SplitComplexAxis"}:
        return _cost("reshape", input_shape, output_shape, input_dtype, output_dtype, input_bytes, output_bytes, _sum_known(input_bytes, output_bytes))
    if name in {"Mean", "Sum", "Maximum", "Minimum"}:
        return _cost(
            "reduction",
            input_shape,
            output_shape,
            input_dtype,
            output_dtype,
            input_bytes,
            output_bytes,
            _sum_known(input_bytes, output_bytes),
            requires_full_axis=(axis,),
            notes=(f"Requires full axis {axis}.",),
        )
    if name == "RootSumSquares":
        peak = None if input_bytes is None or output_bytes is None else int(input_bytes * 3.0 + output_bytes)
        return _cost(
            "reduction",
            input_shape,
            output_shape,
            input_dtype,
            output_dtype,
            input_bytes,
            output_bytes,
            peak,
            requires_full_axis=(axis,),
            temp_multiplier=3.0,
            notes=(f"Requires full axis {axis}; includes abs/square temporaries.",),
        )
    if name in {"CenteredFFT", "CenteredIFFT"}:
        peak = None if output_bytes is None else int(output_bytes * 6.0)
        return _cost(
            "transform",
            input_shape,
            output_shape,
            input_dtype,
            output_dtype,
            input_bytes,
            output_bytes,
            peak,
            requires_full_axis=(axis,),
            temp_multiplier=6.0,
            can_chunk=False,
            notes=(f"Requires full transform axis {axis}; centered FFT uses backend temporaries.",),
        )
    return _cost("elementwise", input_shape, output_shape, input_dtype, output_dtype, input_bytes, output_bytes, _sum_known(input_bytes, output_bytes))


def estimate_pipeline_cost(base_shape, base_dtype, operations) -> PipelineCost:
    shape = tuple(int(size) for size in base_shape)
    dtype = None if base_dtype is None else np.dtype(base_dtype)
    costs = []
    warnings = []
    peak = None
    for operation in tuple(operations):
        cost = estimate_operation_cost(shape, dtype, operation)
        costs.append(cost)
        shape = cost.output_shape
        dtype = cost.output_dtype
        if cost.estimated_peak_bytes is not None:
            peak = cost.estimated_peak_bytes if peak is None else max(peak, cost.estimated_peak_bytes)
        if cost.requires_full_axis:
            warnings.append(f"{type(operation).__name__} requires full axis {cost.requires_full_axis[0]}.")
    output_bytes = _array_bytes(shape, dtype)
    return PipelineCost(
        operation_costs=tuple(costs),
        output_shape=shape,
        output_dtype=dtype,
        estimated_peak_bytes=peak,
        estimated_output_bytes=output_bytes,
        warnings=tuple(warnings),
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
    notes=(),
):
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
        can_chunk=bool(can_chunk),
        notes=tuple(notes),
    )


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
