"""Lazy slab planning and evaluation for operation-backed views."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Literal

import numpy as np

from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.planner import apply_operation_to_region, plan_region_request
from arrayscope.operations.regions import (
    StageCacheCandidate,
    StageKey,
    apply_region,
    apply_subregion,
    index_spec_from_region,
    region_contains,
    region_shape,
)
from arrayscope.operations.stage_cache import StageValue


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
    region_plan = plan_region_request(document, request)
    base_shape = region_shape(np.shape(document.base_data), region_plan.required_input_region)
    dtype = getattr(document.base_data, "dtype", None)
    estimated = None if dtype is None else int(np.prod(base_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    output_shape = region_shape(document.current_shape, region_plan.final_region)
    return SlabPlan(
        base_index=index_spec_from_region(region_plan.required_input_region),
        base_shape=tuple(int(v) for v in base_shape),
        slab_ops=tuple(transition.operation for transition in region_plan.transitions),
        output_shape=tuple(int(v) for v in output_shape),
        estimated_nbytes=estimated,
        requires_full_materialization=False,
        region_plan=region_plan,
    )


def evaluate_slab(document: ArrayDocument, request: SlabRequest, *, stage_cache=None, document_key=None):
    plan = plan_slab(document, request)
    return evaluate_slab_from_plan(document, request, plan, stage_cache=stage_cache, document_key=document_key)


def evaluate_slab_from_plan(document: ArrayDocument, request: SlabRequest, plan: SlabPlan, *, stage_cache=None, document_key=None):
    del request
    region_plan = plan.region_plan
    if stage_cache is not None and region_plan.cache_candidates:
        return _evaluate_slab_with_stage_cache(document, region_plan, stage_cache, document_key)
    data = apply_region(document.base_data, region_plan.required_input_region)
    for transition in region_plan.transitions:
        data = apply_operation_to_region(
            transition.operation,
            data,
            input_region=transition.required_input_region,
            output_region=transition.output_region,
        )
    return data


def stage_key_for_candidate(document_key, candidate: StageCacheCandidate) -> StageKey:
    return StageKey(
        document_key=document_key,
        operation_prefix=tuple(candidate.operation_prefix),
        region=candidate.region,
        dtype=str(candidate.dtype),
        shape=tuple(int(size) for size in candidate.shape),
    )


def _evaluate_slab_with_stage_cache(document: ArrayDocument, region_plan, stage_cache, document_key):
    document_key = _default_stage_document_key(document) if document_key is None else document_key
    candidates = tuple(region_plan.cache_candidates)
    candidate_by_stage = {int(candidate.stage_index): candidate for candidate in candidates}
    stage_shape_by_index = {stage.stage_index: stage.output_shape for stage in region_plan.stages}

    for candidate in candidates:
        stage_cache.note_candidate(_candidate_summary(candidate))

    store_stage_indices = _store_stage_indices(candidates, stage_cache)
    hit_stage = None
    data = None
    current_region = None
    for candidate in _lookup_candidates(candidates):
        key = stage_key_for_candidate(document_key, candidate)
        value = stage_cache.get_containing(key) if hasattr(stage_cache, "get_containing") else stage_cache.get(key)
        if value is None:
            continue
        shape = stage_shape_by_index.get(int(candidate.stage_index), candidate.shape)
        if not region_contains(value.region, candidate.region, shape):
            continue
        data = apply_subregion(value.data, source_region=value.region, target_region=candidate.region, shape=shape)
        current_region = candidate.region
        hit_stage = int(candidate.stage_index)
        break

    if data is None:
        data = apply_region(document.base_data, region_plan.required_input_region)
        current_region = region_plan.required_input_region
        hit_stage = 0

    for transition in region_plan.transitions:
        if int(transition.stage_index) <= int(hit_stage):
            continue
        candidate = candidate_by_stage.get(int(transition.stage_index))
        output_region = (
            candidate.region
            if candidate is not None and int(candidate.stage_index) in store_stage_indices
            else transition.output_region
        )
        input_region = transition.operation.required_input_region(transition.input_shape, output_region)
        operation_input = apply_subregion(
            data,
            source_region=current_region,
            target_region=input_region,
            shape=transition.input_shape,
        )
        data = apply_operation_to_region(
            transition.operation,
            operation_input,
            input_region=input_region,
            output_region=output_region,
        )
        current_region = output_region
        if candidate is not None and int(candidate.stage_index) in store_stage_indices:
            nbytes = _stage_nbytes(data, candidate)
            stage_cache.put(
                stage_key_for_candidate(document_key, candidate),
                StageValue(
                    data=data,
                    region=candidate.region,
                    stage_index=int(candidate.stage_index),
                    nbytes=nbytes,
                    priority=str(candidate.priority),
                    recompute_cost=_candidate_recompute_cost(candidate),
                    visible_reuse=bool(getattr(candidate, "visible_reuse", True)),
                    prefetch_only=bool(getattr(candidate, "prefetch_only", False)),
                ),
            )

    if current_region != region_plan.final_region:
        return apply_subregion(
            data,
            source_region=current_region,
            target_region=region_plan.final_region,
            shape=document.current_shape,
        )
    return data


def _default_stage_document_key(document: ArrayDocument):
    dtype = getattr(document.base_data, "dtype", None)
    return (
        id(document.base_data),
        tuple(np.shape(document.base_data)),
        None if dtype is None else str(np.dtype(dtype)),
        int(getattr(document, "revision", 0)),
    )


def _stage_nbytes(data, candidate: StageCacheCandidate) -> int:
    nbytes = getattr(data, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    estimated = candidate.estimated_nbytes
    return 1 if estimated is None else int(estimated)


def _candidate_summary(candidate: StageCacheCandidate) -> str:
    retain = "yes" if getattr(candidate, "retain", True) else "no"
    reason = getattr(candidate, "retain_reason", "")
    return f"stage={candidate.stage_index}, priority={candidate.priority}, bytes={candidate.estimated_nbytes}, retain={retain}, {reason}"


def _lookup_candidates(candidates):
    retained = [candidate for candidate in tuple(candidates) if getattr(candidate, "retain", True)]
    non_retained = [candidate for candidate in tuple(candidates) if not getattr(candidate, "retain", True)]
    return tuple(reversed(retained)) + tuple(reversed(non_retained))


def _store_stage_indices(candidates, stage_cache) -> set[int]:
    candidates = tuple(candidates)
    max_bytes = int(getattr(stage_cache, "max_bytes", 0))
    fitting = []
    for candidate in candidates:
        if _candidate_fits(candidate, max_bytes):
            fitting.append(candidate)
        elif hasattr(stage_cache, "note_refused"):
            stage_cache.note_refused(_candidate_summary(candidate))
    if not fitting:
        return set()
    scored = sorted(fitting, key=_candidate_retention_score, reverse=True)
    selected = scored[0]
    return {int(selected.stage_index)}


def _candidate_fits(candidate: StageCacheCandidate, max_bytes: int) -> bool:
    estimated = candidate.estimated_nbytes
    return estimated is None or int(estimated) <= int(max_bytes)


def _candidate_retention_score(candidate: StageCacheCandidate) -> float:
    priority = {
        "lowest": 0.0,
        "low": 1000.0,
        "medium": 2000.0,
        "high": 3000.0,
        "highest": 4000.0,
    }.get(str(candidate.priority), 1000.0)
    estimated = candidate.estimated_nbytes
    byte_penalty = 0.0 if estimated is None else log2(max(int(estimated), 1)) * 8.0
    retain_bonus = 750.0 if getattr(candidate, "retain", True) else 0.0
    visible_bonus = 500.0 if getattr(candidate, "visible_reuse", True) else 0.0
    prefetch_penalty = 600.0 if getattr(candidate, "prefetch_only", False) else 0.0
    return priority + retain_bonus + visible_bonus + _candidate_recompute_cost(candidate) - prefetch_penalty - byte_penalty


def _candidate_recompute_cost(candidate: StageCacheCandidate) -> float:
    value = float(getattr(candidate, "estimated_recompute_cost", 0.0) or 0.0)
    if value > 0.0:
        return value
    estimated = getattr(candidate, "estimated_nbytes", None)
    if estimated is None:
        return 0.0
    return min(float(max(0, int(estimated))) / (1024.0 * 1024.0), 10_000.0)


def _ranged_keep_axes(view_state, keep_axes):
    axis_ranges = getattr(view_state, "axis_range_indices", ())
    ranged = []
    for axis in keep_axes:
        if axis < len(axis_ranges) and axis_ranges[int(axis)] is not None:
            ranged.append(int(axis))
    return tuple(ranged)
