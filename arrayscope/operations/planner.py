"""Pure operation-region planning contracts for future stage caching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.operations.capabilities import OperationCapabilities, OperationKind
from arrayscope.operations.cost import OperationCost, estimate_operation_cost, estimate_pipeline_cost, operation_output_dtype
from arrayscope.operations.optimizer import format_optimization_steps, optimize_operations
from arrayscope.operations.regions import (
    AxisRegionKind,
    RegionSpec,
    StageCacheCandidate,
    axis_region_kind,
    expand_region_axes,
    region_from_index_spec,
    region_nbytes,
)


@dataclass(frozen=True)
class OperationStageInfo:
    stage_index: int
    operation: object | None
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    input_dtype: np.dtype | None
    output_dtype: np.dtype | None
    capabilities: OperationCapabilities | None
    cost: OperationCost | None


@dataclass(frozen=True)
class StageRegionTransition:
    stage_index: int
    operation: object
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_region: RegionSpec
    required_input_region: RegionSpec
    expanded_axes: tuple[int, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class RegionPlan:
    request_kind: str
    final_region: RegionSpec
    required_input_region: RegionSpec
    stages: tuple[OperationStageInfo, ...]
    transitions: tuple[StageRegionTransition, ...]
    cache_candidates: tuple[StageCacheCandidate, ...]
    estimated_peak_bytes: int | None
    warnings: tuple[str, ...] = ()
    optimization_steps: tuple[str, ...] = ()
    original_operation_count: int = 0
    optimized_operation_count: int = 0


def build_operation_stages(base_shape, base_dtype, operations) -> tuple[OperationStageInfo, ...]:
    shape = tuple(int(size) for size in base_shape)
    dtype = None if base_dtype is None else np.dtype(base_dtype)
    stages = [
        OperationStageInfo(
            stage_index=0,
            operation=None,
            input_shape=shape,
            output_shape=shape,
            input_dtype=dtype,
            output_dtype=dtype,
            capabilities=None,
            cost=None,
        )
    ]
    for index, operation in enumerate(tuple(operations), start=1):
        cost = estimate_operation_cost(shape, dtype, operation)
        stages.append(
            OperationStageInfo(
                stage_index=index,
                operation=operation,
                input_shape=shape,
                output_shape=cost.output_shape,
                input_dtype=dtype,
                output_dtype=cost.output_dtype,
                capabilities=_operation_capabilities(operation, shape, dtype),
                cost=cost,
            )
        )
        shape = cost.output_shape
        dtype = cost.output_dtype
    return tuple(stages)


def candidate_stage_cache_points(base_shape, base_dtype, operations, final_region) -> tuple[StageCacheCandidate, ...]:
    stages = build_operation_stages(base_shape, base_dtype, operations)
    current_region = final_region
    candidates = []
    for stage in reversed(stages[1:]):
        capabilities = stage.capabilities
        if capabilities is None:
            continue
        candidate_region = expand_region_axes(current_region, tuple(axis for axis in capabilities.expands_request_axes if axis < len(current_region.axes)))
        if capabilities.cache_stage:
            dtype_text = "unknown" if stage.output_dtype is None else str(np.dtype(stage.output_dtype))
            candidates.append(
                StageCacheCandidate(
                    stage_index=stage.stage_index,
                    operation_prefix=tuple(operations[: stage.stage_index]),
                    region=candidate_region,
                    shape=stage.output_shape,
                    dtype=dtype_text,
                    estimated_nbytes=region_nbytes(stage.output_shape, stage.output_dtype, candidate_region),
                    priority=_candidate_priority(capabilities),
                    reason=f"{type(stage.operation).__name__} declares cache_stage",
                    retain=True,
                    retain_reason="cacheable stage",
                )
            )
        current_region = candidate_region
    ordered = list(reversed(candidates))
    if ordered:
        for index, candidate in enumerate(tuple(ordered)):
            is_last = index == len(ordered) - 1
            ordered[index] = StageCacheCandidate(
                stage_index=candidate.stage_index,
                operation_prefix=candidate.operation_prefix,
                region=candidate.region,
                shape=candidate.shape,
                dtype=candidate.dtype,
                estimated_nbytes=candidate.estimated_nbytes,
                priority="highest" if is_last else candidate.priority,
                reason=candidate.reason,
                retain=is_last,
                retain_reason="latest reusable cacheable stage" if is_last else "superseded by later retained cacheable stage",
            )
    return tuple(ordered)


def candidate_stage_cache_points_from_transitions(stages, operations, transitions) -> tuple[StageCacheCandidate, ...]:
    stage_by_index = {stage.stage_index: stage for stage in stages}
    raw = []
    for transition in transitions:
        stage = stage_by_index.get(transition.stage_index)
        capabilities = None if stage is None else stage.capabilities
        if capabilities is None or not capabilities.cache_stage:
            continue
        dtype_text = "unknown" if stage.output_dtype is None else str(np.dtype(stage.output_dtype))
        candidate_region = expand_region_axes(transition.output_region, transition.expanded_axes)
        raw.append(
            (
                transition,
                capabilities,
                dtype_text,
                candidate_region,
                region_nbytes(transition.output_shape, stage.output_dtype, candidate_region),
            )
        )
    candidates = []
    for index, (transition, capabilities, dtype_text, candidate_region, nbytes) in enumerate(raw):
        is_last = index == len(raw) - 1
        priority = "highest" if is_last else _candidate_priority(capabilities)
        candidates.append(
            StageCacheCandidate(
                stage_index=transition.stage_index,
                operation_prefix=tuple(operations[: transition.stage_index]),
                region=candidate_region,
                shape=transition.output_shape,
                dtype=dtype_text,
                estimated_nbytes=nbytes,
                priority=priority,
                reason=f"{type(transition.operation).__name__} declares cache_stage",
                retain=is_last,
                retain_reason="latest reusable cacheable stage" if is_last else "superseded by later retained cacheable stage",
            )
        )
    return tuple(candidates)


def plan_region_request(document, request) -> RegionPlan:
    original_operations = tuple(document.enabled_operations)
    base_shape = tuple(int(size) for size in np.shape(document.base_data))
    base_dtype = getattr(document.base_data, "dtype", None)
    optimized = optimize_operations(base_shape, base_dtype, original_operations)
    if tuple(optimized.output_shape) != tuple(document.current_shape):
        raise ValueError(
            f"optimized output shape {optimized.output_shape} does not match document shape {document.current_shape}"
        )
    operations = optimized.operations
    stages = build_operation_stages(base_shape, base_dtype, operations)
    pipeline = estimate_pipeline_cost(base_shape, base_dtype, operations, optimize=False)
    current_region = final_region_for_request(document.current_shape, request)
    transitions = []
    warnings = list(pipeline.warnings)
    for stage in reversed(stages[1:]):
        operation = stage.operation
        required = required_input_region_for_operation(operation, stage.input_shape, current_region)
        transition = StageRegionTransition(
            stage_index=stage.stage_index,
            operation=operation,
            input_shape=stage.input_shape,
            output_shape=stage.output_shape,
            output_region=current_region,
            required_input_region=required,
            expanded_axes=(
                tuple(stage.capabilities.expands_request_axes)
                if stage.capabilities is not None
                else expanded_axes_for_transition(current_region, required, operation)
            ),
            reason=_transition_reason(operation, current_region, required),
        )
        transitions.append(transition)
        current_region = required
    transitions = tuple(reversed(transitions))
    region_peak = estimate_region_plan_peak(base_shape, base_dtype, transitions, current_region)
    return RegionPlan(
        request_kind=str(request.kind),
        final_region=final_region_for_request(document.current_shape, request),
        required_input_region=current_region,
        stages=stages,
        transitions=transitions,
        cache_candidates=candidate_stage_cache_points_from_transitions(stages, operations, transitions),
        estimated_peak_bytes=region_peak,
        warnings=tuple(warnings),
        optimization_steps=format_optimization_steps(optimized.steps),
        original_operation_count=len(original_operations),
        optimized_operation_count=len(operations),
    )


def estimate_region_plan_peak(base_shape, base_dtype, transitions, required_input_region) -> int | None:
    peak = region_nbytes(tuple(int(size) for size in base_shape), base_dtype, required_input_region)
    dtype = None if base_dtype is None else np.dtype(base_dtype)
    for transition in tuple(transitions):
        input_dtype = dtype
        output_dtype = _operation_output_dtype(transition.operation, input_dtype)
        input_bytes = region_nbytes(transition.input_shape, input_dtype, transition.required_input_region)
        output_bytes = region_nbytes(transition.output_shape, output_dtype, transition.output_region)
        capabilities = _operation_capabilities(transition.operation, transition.input_shape, input_dtype)
        if capabilities is None:
            stage_peak = _sum_known(input_bytes, output_bytes)
        elif capabilities.kind == OperationKind.VIEW:
            stage_peak = output_bytes
        elif capabilities.kind == OperationKind.TRANSFORM:
            stage_peak = None if output_bytes is None else int(output_bytes * capabilities.temp_multiplier)
        elif capabilities.kind == OperationKind.REDUCTION and capabilities.temp_multiplier > 1.0:
            stage_peak = None if input_bytes is None or output_bytes is None else int(input_bytes * capabilities.temp_multiplier + output_bytes)
        else:
            stage_peak = _sum_known(input_bytes, output_bytes)
        peak = _max_known(peak, stage_peak)
        dtype = output_dtype
    return peak


def final_region_for_request(shape, request) -> RegionSpec:
    spec = []
    keep_axes = set(int(axis) for axis in request.keep_axes)
    for axis, size in enumerate(tuple(int(size) for size in shape)):
        if axis in keep_axes:
            spec.append(_axis_item_for_keep_axis(request.view_state, axis))
        else:
            index = int(request.slice_indices[axis])
            if index < 0 or index >= int(size):
                raise IndexError(f"slice index {index} is out of range for axis {axis} with size {size}")
            spec.append(index)
    return region_from_index_spec(shape, tuple(spec))


def required_input_region_for_operation(operation, input_shape, output_region: RegionSpec) -> RegionSpec:
    return operation.required_input_region(tuple(int(size) for size in input_shape), output_region)


def apply_operation_to_region(operation, data, *, input_region: RegionSpec, output_region: RegionSpec):
    return operation.apply_to_region(data, input_region=input_region, output_region=output_region)


def expanded_axes_for_transition(output_region: RegionSpec, required_input_region: RegionSpec, operation) -> tuple[int, ...]:
    del operation
    expanded = []
    for axis, required_axis in enumerate(required_input_region.axes[: len(output_region.axes)]):
        if axis_region_kind(required_axis.kind) == AxisRegionKind.ALL and axis_region_kind(output_region.axes[axis].kind) != AxisRegionKind.ALL:
            expanded.append(axis)
    return tuple(expanded)


def _operation_capabilities(operation, shape, dtype) -> OperationCapabilities | None:
    if not hasattr(operation, "capabilities"):
        return None
    return operation.capabilities(shape, dtype)


def _operation_output_dtype(operation, input_dtype):
    return operation_output_dtype(input_dtype, operation)


def _sum_known(*values) -> int | None:
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values))


def _max_known(left, right) -> int | None:
    if left is None or right is None:
        return None
    return max(int(left), int(right))


def _candidate_priority(capabilities: OperationCapabilities) -> str:
    if capabilities.kind == OperationKind.TRANSFORM:
        return "high"
    if capabilities.kind == OperationKind.REDUCTION:
        return "medium"
    return "low"


def _axis_item_for_keep_axis(view_state, axis):
    axis_ranges = getattr(view_state, "axis_range_indices", ())
    if int(axis) >= len(axis_ranges):
        return slice(None)
    indices = axis_ranges[int(axis)]
    if indices is None:
        return slice(None)
    return _sequence_to_basic_index(np.asarray(indices, dtype=np.int64))


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


def _transition_reason(operation, output_region, required_region) -> str:
    if output_region == required_region:
        return "same region"
    return f"{type(operation).__name__} maps output region to required input region"
