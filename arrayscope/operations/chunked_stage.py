"""Chunked materialization for reusable operation stages."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.regions import (
    AxisRegion,
    AxisRegionKind,
    RegionSpec,
    axis_region_kind,
    region_nbytes,
    region_shape,
)
from arrayscope.operations.regions import StageCacheCandidate
from arrayscope.operations.slabs import materialize_stage_candidate, stage_key_for_candidate
from arrayscope.operations.stage_cache import StageValue


@dataclass(frozen=True)
class StageChunkPlan:
    candidate: object
    chunk_axes: tuple[int, ...]
    blocking_axes: tuple[int, ...]
    chunks: tuple[RegionSpec, ...]
    estimated_chunk_bytes: int


@dataclass(frozen=True)
class ChunkedStageProgress:
    completed_chunks: int
    total_chunks: int
    bytes_written: int


def plan_chunked_stage_materialization(
    region_plan,
    candidate,
    memory_policy,
    *,
    target_chunk_bytes: int | None = None,
    allowed_chunk_axes: tuple[int, ...] | None = None,
) -> StageChunkPlan | None:
    if not _region_is_simple_full(candidate.region):
        return None
    if allowed_chunk_axes is None:
        allowed_chunk_axes = stage_materialization_allowed_chunk_axes(candidate.shape)
    allowed = {int(axis) for axis in tuple(allowed_chunk_axes)}
    if not allowed:
        return None
    stage_budget = int(getattr(memory_policy, "stage_cache_budget_bytes", 0) or 0)
    target = int(target_chunk_bytes or (stage_budget if stage_budget > 0 else 256 * 1024 * 1024))
    total = candidate.estimated_nbytes
    if total is None or int(total) <= target:
        return None
    blocking_axes = _blocking_axes_through_candidate(region_plan, int(candidate.stage_index))
    chunk_axes = tuple(
        axis
        for axis, region_axis in enumerate(candidate.region.axes)
        if axis in allowed
        and axis not in blocking_axes
        and axis_region_kind(region_axis.kind) == AxisRegionKind.ALL
        and int(candidate.shape[axis]) > 1
    )
    if not chunk_axes:
        return None
    axis = max(chunk_axes, key=lambda value: int(candidate.shape[value]))
    dtype_size = max(1, np.dtype(candidate.dtype).itemsize)
    other = 1
    for index, size in enumerate(tuple(int(size) for size in candidate.shape)):
        if index != axis:
            other *= max(1, size)
    axis_size = int(candidate.shape[axis])
    chunk_len = max(1, int(target // max(dtype_size * other, 1)))
    if axis_size > 1:
        chunk_len = max(2, chunk_len)
    chunk_len = max(1, min(axis_size, chunk_len))
    chunks = []
    for start in range(0, int(candidate.shape[axis]), chunk_len):
        stop = min(int(candidate.shape[axis]), start + chunk_len)
        axes = list(candidate.region.axes)
        axes[axis] = AxisRegion(AxisRegionKind.SLICE, (int(start), int(stop), 1))
        chunks.append(RegionSpec(tuple(axes)))
    if len(chunks) <= 1:
        return None
    estimated = int(region_nbytes(candidate.shape, candidate.dtype, chunks[0]) or min(int(total), target))
    return StageChunkPlan(candidate=candidate, chunk_axes=(int(axis),), blocking_axes=blocking_axes, chunks=tuple(chunks), estimated_chunk_bytes=estimated)


def stage_materialization_allowed_chunk_axes(shape, *, preferred_axes=None) -> tuple[int, ...]:
    ndim = len(tuple(shape))
    if preferred_axes is None:
        return tuple(range(ndim))
    result = []
    for axis in tuple(preferred_axes):
        axis = int(axis)
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {axis} is out of range for ndim {ndim}")
        if axis not in result:
            result.append(axis)
    for axis in range(ndim):
        if axis not in result:
            result.append(axis)
    return tuple(result)


def materialize_stage_candidate_chunked(
    document,
    region_plan,
    candidate,
    *,
    stage_cache=None,
    document_key=None,
    cancellation_token=None,
    evaluation_context=None,
    memory_policy=None,
    target_chunk_bytes: int | None = None,
    allowed_chunk_axes: tuple[int, ...] | None = None,
) -> StageValue:
    plan = plan_chunked_stage_materialization(
        region_plan,
        candidate,
        memory_policy or object(),
        target_chunk_bytes=target_chunk_bytes,
        allowed_chunk_axes=allowed_chunk_axes,
    )
    if plan is None:
        return materialize_stage_candidate(
            document,
            region_plan,
            candidate,
            stage_cache=stage_cache,
            document_key=document_key,
            cancellation_token=cancellation_token,
            evaluation_context=evaluation_context,
        )
    _check_cancelled(cancellation_token)
    output = np.empty(region_shape(candidate.shape, candidate.region), dtype=np.dtype(candidate.dtype))
    completed = 0
    for chunk in plan.chunks:
        _check_cancelled(cancellation_token)
        chunk_candidate = _candidate_for_region(candidate, chunk)
        value = materialize_stage_candidate(
            document,
            region_plan,
            chunk_candidate,
            stage_cache=None,
            document_key=document_key,
            cancellation_token=cancellation_token,
            evaluation_context=evaluation_context,
        )
        output[_local_index_for_chunk(candidate.region, chunk)] = value.data
        completed += 1
        _check_cancelled(cancellation_token)
    nbytes = int(output.nbytes)
    result = StageValue(
        data=output,
        region=candidate.region,
        stage_index=int(candidate.stage_index),
        nbytes=nbytes,
        priority=str(candidate.priority),
        recompute_cost=float(getattr(candidate, "estimated_recompute_cost", 0.0) or 0.0),
        visible_reuse=bool(getattr(candidate, "visible_reuse", True)),
        prefetch_only=bool(getattr(candidate, "prefetch_only", False)),
    )
    _check_cancelled(cancellation_token)
    if stage_cache is not None:
        stage_cache.put(stage_key_for_candidate(document_key, candidate), result)
    del completed
    return result


def _blocking_axes_through_candidate(region_plan, stage_index: int) -> tuple[int, ...]:
    axes = set()
    for stage in tuple(getattr(region_plan, "stages", ())):
        if int(getattr(stage, "stage_index", 0)) > int(stage_index):
            continue
        capabilities = getattr(stage, "capabilities", None)
        axes.update(int(axis) for axis in tuple(getattr(capabilities, "blocking_axes", ()) or ()))
    return tuple(sorted(axes))


def _region_is_simple_full(region: RegionSpec) -> bool:
    return all(axis_region_kind(axis.kind) == AxisRegionKind.ALL for axis in region.axes)


def _candidate_for_region(candidate, region: RegionSpec) -> StageCacheCandidate:
    return StageCacheCandidate(
        stage_index=candidate.stage_index,
        operation_prefix=candidate.operation_prefix,
        region=region,
        shape=candidate.shape,
        dtype=candidate.dtype,
        estimated_nbytes=None,
        priority=candidate.priority,
        reason=candidate.reason,
        retain=candidate.retain,
        retain_reason=candidate.retain_reason,
        estimated_recompute_cost=getattr(candidate, "estimated_recompute_cost", 0.0),
        visible_reuse=getattr(candidate, "visible_reuse", True),
        prefetch_only=getattr(candidate, "prefetch_only", False),
    )


def _local_index_for_chunk(container: RegionSpec, chunk: RegionSpec) -> tuple:
    del container
    index = []
    for axis in chunk.axes:
        kind = axis_region_kind(axis.kind)
        if kind == AxisRegionKind.ALL:
            index.append(slice(None))
        elif kind == AxisRegionKind.SLICE:
            start, stop, step = axis.value
            if int(step) != 1:
                raise ValueError("chunk stage assignment requires unit-step slices")
            index.append(slice(int(start), int(stop)))
        else:
            raise ValueError("chunk stage assignment only supports all/slice axes")
    return tuple(index)


def _check_cancelled(token) -> None:
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()
