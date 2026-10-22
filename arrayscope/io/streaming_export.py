"""Strict, opt-in incremental NPY export using the existing exact region owner.

Chunk working estimates are admission targets, not process RSS guarantees.
Every multi-chunk export must advance through disjoint bounds on one fixed
source axis; a capability label alone never permits repeated whole-source work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from math import prod
from operator import index

import numpy as np

from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.core.memory_budget import DEFAULT_VISIBLE_RENDER_BUDGET_BYTES, format_bytes
from arrayscope.io.numpy_save import save_derived_array_chunks
from arrayscope.io.region_export import (
    RegionExportProgress,
    RegionExportRequest,
    RegionExportResult,
    validate_region_export_snapshot,
)
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import FFTShift, SplitAxis
from arrayscope.operations.planner import build_operation_stages, plan_output_region
from arrayscope.operations.regions import (
    AxisRegionKind,
    axis_region_kind,
    region_from_index_spec,
    region_nbytes,
)
from arrayscope.operations.registry import operation_entries
from arrayscope.operations.slabs import evaluate_region_plan

_LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_TARGET_BYTES = 64 * 1024**2


class StreamingExportRefused(ValueError):
    """The exact pipeline cannot satisfy this bounded streaming contract."""


@dataclass(frozen=True)
class StreamingExportPlan:
    request: RegionExportRequest = field(compare=False, repr=False)
    output_shape: tuple[int, ...]
    output_dtype: np.dtype
    output_bytes: int
    chunk_shape: tuple[int, ...]
    chunk_count: int
    estimated_peak_bytes: int
    total_source_read_bytes: int
    warnings: tuple[str, ...]
    needs_confirmation: bool
    stream_axis: int | None
    chunk_length: int
    chunk_target_bytes: int
    source_axis: int | None
    source_direction: int


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


def _token_and_context(token, context):
    if token is None:
        token = getattr(context, "cancellation_token", None)
    if context is None:
        context = EvaluationContext(ComputeLane.STAGE, token, 1, None)
    else:
        context = replace(context, cancellation_token=token)
    return token, context


def _validate_request(request):
    if request.path.suffix.lower() != ".npy" or request.sidecar:
        raise StreamingExportRefused("Streaming export supports values-only .npy output.")
    builtin_types = {entry.operation_type for entry in operation_entries()}
    for operation in request.document.enabled_operations:
        if type(operation) in {FFTShift, SplitAxis}:
            raise StreamingExportRefused(
                f"Streaming export does not yet support {type(operation).__name__}: "
                "its coordinate mapping can require large index arrays. "
                "Use non-streaming export for this operation."
            )
        if type(operation) not in builtin_types:
            raise StreamingExportRefused(
                f"Streaming export requires built-in metadata-only operations; "
                f"{type(operation).__name__} is unsupported."
            )
    validate_region_export_snapshot(request)


def _bounds_for_chunk(request, stream_axis, offset, length):
    bounds = list(request.ranges)
    if stream_axis is not None:
        start, stop = bounds[stream_axis]
        bounds[stream_axis] = (start + offset, min(start + offset + length, stop))
    return tuple(bounds)


def _plan_chunk(request, stream_axis, offset, length):
    bounds = _bounds_for_chunk(request, stream_axis, offset, length)
    shape = tuple(stop - start for start, stop in bounds)
    region = region_from_index_spec(
        request.document.current_shape,
        tuple(
            slice(None) if start == 0 and stop == size else slice(start, stop)
            for (start, stop), size in zip(bounds, request.document.current_shape, strict=True)
        ),
    )
    return plan_output_region(request.document, region, request_kind="streaming-export"), shape


def _chunk_cost(request, plan, shape, dtype):
    output_bytes = prod(shape) * dtype.itemsize
    if plan.stages[-1].output_dtype != dtype or plan.estimated_peak_bytes is None:
        raise StreamingExportRefused("Streaming export requires a known, stable dtype and cost.")
    source_bytes = region_nbytes(
        request.document.base_data.shape,
        request.document.base_data.dtype,
        plan.required_input_region,
    )
    if source_bytes is None:
        raise StreamingExportRefused("Streaming export requires known source read costs.")
    # Reserve another output block conservatively. The writer actually buffers
    # at most 1 MiB and releases each block before requesting the next one.
    peak = plan.estimated_peak_bytes + output_bytes
    return peak, source_bytes


def _axis_bounds(axis, size):
    kind = axis_region_kind(axis.kind)
    if kind == AxisRegionKind.ALL:
        return (0, size) if size else None
    if kind == AxisRegionKind.POINT:
        value = int(axis.value)
        return value, value + 1
    if kind == AxisRegionKind.SLICE:
        values = range(*slice(*axis.value).indices(size))
        if not values:
            return None
        return min(values[0], values[-1]), max(values[0], values[-1]) + 1
    values = axis.value
    return (min(values), max(values) + 1) if values else None


def _source_bounds(request, plan):
    return tuple(
        _axis_bounds(axis, size)
        for axis, size in zip(
            plan.required_input_region.axes, request.document.base_data.shape, strict=True
        )
    )


def _advances(previous, current, direction):
    if previous is None or current is None:
        return False
    return current[0] >= previous[1] if direction > 0 else current[1] <= previous[0]


def _source_progression(previous, current):
    for axis, (before, after) in enumerate(zip(previous, current, strict=True)):
        for direction in (1, -1):
            if _advances(before, after, direction):
                return axis, direction
    raise StreamingExportRefused(
        "Streaming chunks revisit overlapping source bounds. This pipeline requires "
        "non-streaming export or a different leading output axis."
    )


def _verify_progression(previous, current, axis, direction):
    if not _advances(previous[axis], current[axis], direction):
        raise StreamingExportRefused(
            "Streaming source bounds stopped advancing without overlap; output was not published."
        )


def plan_streaming_export(
    request,
    *,
    chunk_target_bytes=DEFAULT_CHUNK_TARGET_BYTES,
    cancellation_token=None,
    evaluation_context=None,
):
    """Inspect all chunk demands without payload reads or retaining their plans."""
    token, context = _token_and_context(cancellation_token, evaluation_context)
    _check_cancelled(token)
    target = index(chunk_target_bytes)
    if target < 1:
        raise ValueError("streaming chunk target must be a positive integer byte count")
    _validate_request(request)
    stages = build_operation_stages(
        request.document.base_data.shape,
        request.document.base_data.dtype,
        request.document.enabled_operations,
    )
    dtype = stages[-1].output_dtype
    if dtype is None or np.dtype(dtype).kind not in "biufc":
        raise StreamingExportRefused("Streaming export requires a known numeric output dtype.")
    dtype = np.dtype(dtype)
    output_shape = request.output_shape
    output_bytes = prod(output_shape) * dtype.itemsize
    retained = tuple(
        axis for axis, size in enumerate(request.selected_shape) if not request.squeeze or size != 1
    )
    stream_axis = retained[0] if retained else None
    extent = 1 if stream_axis is None else request.selected_shape[stream_axis]
    if output_bytes == 0:
        return StreamingExportPlan(
            request,
            output_shape,
            dtype,
            0,
            output_shape,
            0,
            0,
            0,
            (),
            False,
            stream_axis,
            0,
            target,
            None,
            0,
        )
    row_bytes = output_bytes // extent
    if row_bytes > target:
        raise StreamingExportRefused(
            f"A single leading output slab needs {format_bytes(row_bytes)}, exceeding "
            f"the {format_bytes(target)} streaming chunk target; "
            "select a smaller trailing region or use non-streaming export."
        )
    unit_plan, unit_shape = _plan_chunk(request, stream_axis, 0, 1)
    unit_peak, _source_bytes = _chunk_cost(request, unit_plan, unit_shape, dtype)
    if unit_peak > target:
        raise StreamingExportRefused(
            f"A single exact output slab requires estimated working memory "
            f"{format_bytes(unit_peak)}, exceeding the {format_bytes(target)} streaming "
            "chunk target; full-axis transforms and decoded source chunks must fit."
        )
    length = min(extent, max(1, target // max(1, 2 * row_bytes)))
    # Working estimates can have fixed codec or transform overhead. Check the
    # actual candidate rather than assuming cost scales linearly with rows.
    while True:
        _check_cancelled(token)
        candidate, shape = _plan_chunk(request, stream_axis, 0, length)
        peak, _source_bytes = _chunk_cost(request, candidate, shape, dtype)
        if peak <= target:
            break
        if length == 1:
            raise StreamingExportRefused(
                "Streaming source cost changed; the minimum chunk no longer fits "
                "the working target. Reload the source and try again."
            )
        length = max(1, length // 2)
    count = (extent + length - 1) // length
    previous = None
    source_axis = None
    direction = 0
    total_source_bytes = 0
    peak = 0
    warnings = []
    for offset in range(0, extent, length):
        _check_cancelled(token)
        plan, shape = _plan_chunk(request, stream_axis, offset, length)
        chunk_peak, source_bytes = _chunk_cost(request, plan, shape, dtype)
        if chunk_peak > target:
            raise StreamingExportRefused(
                f"A later exact chunk needs {format_bytes(chunk_peak)}, exceeding the "
                f"{format_bytes(target)} streaming working target; select a smaller region."
            )
        current = _source_bounds(request, plan)
        if previous is not None:
            if source_axis is None:
                source_axis, direction = _source_progression(previous, current)
            else:
                _verify_progression(previous, current, source_axis, direction)
        previous = current
        total_source_bytes += source_bytes
        peak = max(peak, chunk_peak)
        warnings.extend(warning for warning in plan.warnings if warning not in warnings)
    _check_cancelled(token)
    validate_region_export_snapshot(request)
    policy = getattr(context, "memory_policy", None)
    budget = int(
        getattr(policy, "visible_render_budget_bytes", DEFAULT_VISIBLE_RENDER_BUDGET_BYTES)
    )
    chunk_shape = list(output_shape)
    if chunk_shape:
        chunk_shape[0] = length
    return StreamingExportPlan(
        request,
        output_shape,
        dtype,
        output_bytes,
        tuple(chunk_shape),
        count,
        peak,
        total_source_bytes,
        tuple(warnings),
        bool(warnings) or peak > budget or output_bytes > budget,
        stream_axis,
        length,
        target,
        source_axis,
        direction,
    )


def export_streaming_region(
    request,
    *,
    prepared=None,
    chunk_target_bytes=DEFAULT_CHUNK_TARGET_BYTES,
    cancellation_token=None,
    evaluation_context=None,
    progress_callback=None,
):
    """Evaluate bounded exact blocks and publish through the shared atomic writer."""
    token, context = _token_and_context(cancellation_token, evaluation_context)
    _check_cancelled(token)
    _validate_request(request)
    if prepared is None:
        prepared = plan_streaming_export(
            request,
            chunk_target_bytes=chunk_target_bytes,
            cancellation_token=token,
            evaluation_context=context,
        )
    elif prepared.request is not request:
        raise ValueError("prepared streaming plan belongs to a different request")

    def report(phase, message, completed=0, total=None):
        if progress_callback is not None:
            progress_callback(RegionExportProgress(phase, message, completed, total))

    report("evaluating", "Streaming exact scientific output in bounded chunks…")
    singleton_axes = tuple(
        axis for axis, size in enumerate(request.selected_shape) if request.squeeze and size == 1
    )

    def chunks():
        if not prepared.chunk_count:
            return
        previous = None
        total_source_bytes = 0
        extent = 1 if prepared.stream_axis is None else request.selected_shape[prepared.stream_axis]
        for offset in range(0, extent, prepared.chunk_length):
            _check_cancelled(token)
            validate_region_export_snapshot(request)
            plan, shape = _plan_chunk(request, prepared.stream_axis, offset, prepared.chunk_length)
            peak, source_bytes = _chunk_cost(request, plan, shape, prepared.output_dtype)
            if peak > prepared.estimated_peak_bytes or peak > prepared.chunk_target_bytes:
                raise StreamingExportRefused("Streaming chunk cost changed after preflight.")
            current = _source_bounds(request, plan)
            if previous is not None:
                _verify_progression(
                    previous, current, prepared.source_axis, prepared.source_direction
                )
            previous = current
            total_source_bytes += source_bytes
            if total_source_bytes > prepared.total_source_read_bytes:
                raise StreamingExportRefused("Streaming source demand changed after preflight.")
            data = np.asarray(
                evaluate_region_plan(
                    request.document,
                    plan,
                    cancellation_token=token,
                    evaluation_context=context,
                )
            )
            _check_cancelled(token)
            if data.shape != shape or data.dtype != prepared.output_dtype:
                raise ValueError(
                    f"streaming chunk produced {data.shape}, {data.dtype}; "
                    f"expected {shape}, {prepared.output_dtype}"
                )
            data = np.squeeze(data, axis=singleton_axes)
            yield data
            del data
        if total_source_bytes != prepared.total_source_read_bytes:
            raise StreamingExportRefused("Streaming source demand changed after preflight.")

    def before_publish():
        _check_cancelled(token)
        validate_region_export_snapshot(request)

    paths = save_derived_array_chunks(
        request.path,
        shape=prepared.output_shape,
        dtype=prepared.output_dtype,
        chunks=chunks(),
        cancellation_token=token,
        progress_callback=lambda count: report(
            "writing", "Writing streamed array…", count, prepared.output_bytes
        ),
        before_publish=before_publish,
    )
    # Publication is final even if a late observer or cancellation then fails.
    try:
        report("complete", "Streamed array exported.", prepared.output_bytes, prepared.output_bytes)
    except Exception:
        _LOGGER.exception("Streaming export committed, but its completion observer failed")
    return RegionExportResult(tuple(paths), prepared.output_shape, prepared.output_dtype)
