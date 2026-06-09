"""Pure visible-render scheduling decisions for ArrayScope."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from arrayscope.core.memory_budget import format_bytes
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.slabs import plan_slab, request_for_image


CHUNK_TARGET_MAX_BYTES = 128 * 1024 * 1024
MIN_CHUNK_SIZE = 16
DEGRADED_TARGET_FRACTION = 0.25
PREFETCH_IDLE_DELAY_MS = 150
MAX_IDLE_PREFETCH_SLICES = 2
FFT_PREFETCH_PEAK_BYTES = 32 * 1024 * 1024
CACHE_PREFETCH_SKIP_FRACTION = 0.80


class RenderDecisionKind(Enum):
    CACHED = "cached"
    SYNC = "sync"
    ASYNC_EXACT = "async_exact"
    ASYNC_CHUNKED = "async_chunked"
    DEGRADED_PREVIEW = "degraded_preview"
    REFUSE = "refuse"


@dataclass(frozen=True)
class RenderCostContext:
    view_shape: tuple[int, ...]
    display_shape: tuple[int, int] | None
    image_axes: tuple[int, int] | None
    estimated_display_bytes: int
    estimated_slab_bytes: int | None
    estimated_pipeline_peak_bytes: int | None
    render_budget_bytes: int
    has_operations: bool
    has_transform: bool
    has_reduction: bool
    chunkable: bool
    cache_hit: bool = False
    pipeline_chunkable_axes: tuple[int, ...] = ()
    pipeline_blocking_axes: tuple[int, ...] = ()
    dtype_itemsize: int = 8


@dataclass(frozen=True)
class RenderDecision:
    kind: RenderDecisionKind
    reason: str
    should_keep_previous_image: bool = True
    should_show_overlay: bool = False
    overlay_text: str = ""
    status_text: str = ""
    degraded_factor: int = 1
    chunk_axis: int | None = None
    chunk_size: int | None = None


def estimate_visible_render_context(document, view_state, *, display_bytes, render_budget_bytes, cache_hit=False) -> RenderCostContext:
    pipeline = estimate_pipeline_cost(
        np.shape(document.base_data),
        getattr(document.base_data, "dtype", None),
        document.enabled_operations,
    )
    slab_bytes = None
    try:
        slab_bytes = plan_slab(document, request_for_image(view_state)).estimated_nbytes
    except Exception:
        slab_bytes = None
    kinds = {cost.kind for cost in pipeline.operation_costs}
    dtype = pipeline.output_dtype or getattr(document.base_data, "dtype", np.dtype(float))
    return RenderCostContext(
        view_shape=tuple(int(size) for size in view_state.shape),
        display_shape=_display_shape(view_state),
        image_axes=None if view_state.image_axes is None else tuple(int(axis) for axis in view_state.image_axes),
        estimated_display_bytes=int(display_bytes),
        estimated_slab_bytes=slab_bytes,
        estimated_pipeline_peak_bytes=pipeline.estimated_peak_bytes,
        render_budget_bytes=int(render_budget_bytes),
        has_operations=bool(document.enabled_operations),
        has_transform="transform" in kinds,
        has_reduction="reduction" in kinds,
        chunkable=bool(pipeline.chunkable_axes),
        cache_hit=bool(cache_hit),
        pipeline_chunkable_axes=tuple(pipeline.chunkable_axes),
        pipeline_blocking_axes=tuple(pipeline.blocking_axes),
        dtype_itemsize=int(np.dtype(dtype).itemsize),
    )


def choose_visible_render_decision(context: RenderCostContext) -> RenderDecision:
    if context.cache_hit:
        return RenderDecision(RenderDecisionKind.CACHED, "cache hit", should_keep_previous_image=False)
    peak = context.estimated_pipeline_peak_bytes
    slab = context.estimated_slab_bytes
    budget = int(context.render_budget_bytes)
    if (
        context.estimated_display_bytes <= budget
        and (peak is None or peak <= budget)
        and (slab is None or slab <= budget)
    ):
        return RenderDecision(RenderDecisionKind.ASYNC_EXACT, "within render budget")
    chunk_axis, chunk_size = choose_chunk_axis_and_size(None, context)
    if (context.has_transform or context.has_reduction or (slab is not None and slab > budget)) and chunk_axis is not None:
        return RenderDecision(
            RenderDecisionKind.ASYNC_CHUNKED,
            "chunked exact render fits per-chunk budget",
            should_show_overlay=True,
            overlay_text="Updating view in chunks...",
            chunk_axis=chunk_axis,
            chunk_size=chunk_size,
        )
    factor = _degraded_factor(context)
    if factor is not None:
        return RenderDecision(
            RenderDecisionKind.DEGRADED_PREVIEW,
            "degraded preview fits render budget",
            should_show_overlay=True,
            overlay_text="Preview only: exact view exceeds budget",
            status_text=_over_budget_status(context, factor=factor),
            degraded_factor=factor,
        )
    return RenderDecision(
        RenderDecisionKind.REFUSE,
        "exact render exceeds budget and no degraded preview fits",
        should_show_overlay=True,
        overlay_text="Exact view over budget",
        status_text=_over_budget_status(context),
    )


def degraded_view_state(view_state, *, factor: int):
    factor = max(1, int(factor))
    if factor <= 1 or view_state.image_axes is None:
        return view_state
    result = view_state
    for axis in view_state.image_axes:
        existing = view_state.axis_range_indices[axis]
        indices = tuple(existing[::factor]) if existing is not None else tuple(range(0, int(view_state.shape[axis]), factor))
        if not indices:
            indices = (0,)
        result = result.with_axis_range(axis, indices, text=f"preview/{factor}")
    return result


def choose_chunk_axis_and_size(view_state, context: RenderCostContext) -> tuple[int | None, int | None]:
    del view_state
    if context.display_shape is None or not context.chunkable:
        return None, None
    target = min(CHUNK_TARGET_MAX_BYTES, max(1, int(context.render_budget_bytes) // 2))
    dominant_bytes = max(
        int(context.estimated_display_bytes or 0),
        int(context.estimated_slab_bytes or 0),
        int(context.estimated_pipeline_peak_bytes or 0),
    )
    y_size, x_size = (int(context.display_shape[0]), int(context.display_shape[1]))
    if context.image_axes is None:
        return None, None
    candidates = ((context.image_axes[0], y_size, x_size), (context.image_axes[1], x_size, y_size))
    for original_axis, axis_size, other_size in candidates:
        if original_axis not in context.pipeline_chunkable_axes:
            continue
        bytes_per_index = max(1, int(other_size) * max(1, context.dtype_itemsize))
        display_limited = int(target // bytes_per_index)
        if dominant_bytes > target:
            peak_limited = max(1, int(axis_size * (float(target) / float(dominant_bytes))))
            chunk_size = min(display_limited, peak_limited)
        else:
            chunk_size = display_limited
        chunk_size = max(MIN_CHUNK_SIZE, int(chunk_size))
        chunk_size = min(int(axis_size), int(chunk_size))
        if chunk_size >= MIN_CHUNK_SIZE and chunk_size < int(axis_size):
            return int(original_axis), chunk_size
    return None, None


def _display_shape(view_state):
    if view_state.image_axes is None:
        return None
    shape = []
    for axis in view_state.image_axes:
        indices = view_state.axis_range_indices[axis]
        shape.append(len(indices) if indices is not None else int(view_state.shape[axis]))
    return tuple(shape)


def _degraded_factor(context):
    if context.display_shape is None or max(context.display_shape) < 512:
        return None
    target = int(context.render_budget_bytes * DEGRADED_TARGET_FRACTION)
    for factor in (2, 4, 8, 16):
        degraded_bytes = int(np.ceil(context.display_shape[0] / factor)) * int(np.ceil(context.display_shape[1] / factor)) * max(1, context.dtype_itemsize)
        if degraded_bytes <= target:
            return factor
    return None


def _over_budget_status(context, *, factor=None):
    peak = context.estimated_pipeline_peak_bytes or context.estimated_display_bytes
    text = f"Image view needs estimated peak {format_bytes(peak)} over budget {format_bytes(context.render_budget_bytes)}."
    if factor is not None:
        text += f" Showing {factor}x preview."
    else:
        text += " Use crop/range, lower FFT workers, or increase Performance > Render Memory Budget."
    return text
