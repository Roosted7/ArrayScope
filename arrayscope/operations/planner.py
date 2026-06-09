"""Pure operation-region planning contracts for future stage caching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.operations.capabilities import OperationCapabilities, OperationKind
from arrayscope.operations.cost import OperationCost, estimate_operation_cost, estimate_pipeline_cost
from arrayscope.operations.regions import (
    RegionSpec,
    StageCacheCandidate,
    expand_region_axes,
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
class RegionPlan:
    request_kind: str
    final_region: RegionSpec
    required_input_region: RegionSpec
    stages: tuple[OperationStageInfo, ...]
    cache_candidates: tuple[StageCacheCandidate, ...]
    estimated_peak_bytes: int | None
    warnings: tuple[str, ...] = ()


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
                )
            )
        current_region = candidate_region
    return tuple(reversed(candidates))


def build_region_plan(
    *,
    request_kind: str,
    base_shape,
    base_dtype,
    operations,
    final_region: RegionSpec,
    required_input_region: RegionSpec | None = None,
) -> RegionPlan:
    operations = tuple(operations)
    stages = build_operation_stages(base_shape, base_dtype, operations)
    pipeline = estimate_pipeline_cost(base_shape, base_dtype, operations)
    required = final_region if required_input_region is None else required_input_region
    return RegionPlan(
        request_kind=str(request_kind),
        final_region=final_region,
        required_input_region=required,
        stages=stages,
        cache_candidates=candidate_stage_cache_points(base_shape, base_dtype, operations, final_region),
        estimated_peak_bytes=pipeline.estimated_peak_bytes,
        warnings=tuple(pipeline.warnings),
    )


def _operation_capabilities(operation, shape, dtype) -> OperationCapabilities | None:
    if not hasattr(operation, "capabilities"):
        return None
    return operation.capabilities(shape, dtype)


def _candidate_priority(capabilities: OperationCapabilities) -> str:
    if capabilities.kind == OperationKind.TRANSFORM:
        return "high"
    if capabilities.kind == OperationKind.REDUCTION:
        return "medium"
    return "low"
