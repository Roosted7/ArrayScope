"""Cooperative chunked visible-image evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.display.slice_engine import DisplayImage
from arrayscope.operations.cancellation import EvaluationCancelled


@dataclass(frozen=True)
class ChunkedImageResult:
    display_image: DisplayImage
    eval_ms: float
    chunk_count: int
    slab_nbytes: int | None
    mode: str = "chunked"


def evaluate_image_snapshot_chunked(
    document,
    view_state,
    *,
    chunk_axis: int,
    chunk_size: int,
    colormap_lut=None,
    cancellation_token=None,
    stage_cache=None,
    stage_document_key=None,
    evaluation_context=None,
):
    from arrayscope.operations.evaluator import EvaluationResult, evaluate_image_snapshot

    _check_cancelled(cancellation_token)
    if view_state.image_axes is None:
        result = _evaluate_image_snapshot(
            evaluate_image_snapshot,
            document,
            view_state,
            colormap_lut=colormap_lut,
            cancellation_token=cancellation_token,
            stage_cache=stage_cache,
            stage_document_key=stage_document_key,
            evaluation_context=evaluation_context,
        )
        _check_cancelled(cancellation_token)
        return result
    original_axis = int(chunk_axis)
    display_axis = tuple(int(axis) for axis in view_state.image_axes).index(original_axis)
    axis_indices = view_state.axis_range_indices[original_axis]
    if axis_indices is None:
        axis_indices = tuple(range(int(view_state.shape[original_axis])))
    else:
        axis_indices = tuple(axis_indices)
    chunk_size = max(1, int(chunk_size))
    chunks = [axis_indices[start : start + chunk_size] for start in range(0, len(axis_indices), chunk_size)]
    if len(chunks) <= 1:
        result = _evaluate_image_snapshot(
            evaluate_image_snapshot,
            document,
            view_state,
            colormap_lut=colormap_lut,
            cancellation_token=cancellation_token,
            stage_cache=stage_cache,
            stage_document_key=stage_document_key,
            evaluation_context=evaluation_context,
        )
        _check_cancelled(cancellation_token)
        return result

    start_time = perf_counter()
    out_data = None
    out_hist = None
    total_slab_nbytes = 0
    for chunk_number, indices in enumerate(chunks):
        _check_cancelled(cancellation_token)
        chunk_state = view_state.with_axis_range(original_axis, indices, text=f"chunk {chunk_number + 1}/{len(chunks)}")
        result = _evaluate_image_snapshot(
            evaluate_image_snapshot,
            document,
            chunk_state,
            colormap_lut=colormap_lut,
            cancellation_token=cancellation_token,
            stage_cache=stage_cache,
            stage_document_key=stage_document_key,
            evaluation_context=evaluation_context,
        )
        image = result.value
        if out_data is None:
            full_shape = list(image.data.shape)
            full_shape[display_axis] = len(axis_indices)
            out_data = np.empty(tuple(full_shape), dtype=image.data.dtype)
            if image.histogram_data is not None:
                hist_shape = list(image.histogram_data.shape)
                hist_shape[display_axis] = len(axis_indices)
                out_hist = np.empty(tuple(hist_shape), dtype=image.histogram_data.dtype)
        start = chunk_number * chunk_size
        stop = start + image.data.shape[display_axis]
        _assign_axis(out_data, image.data, display_axis, start, stop)
        if out_hist is not None and image.histogram_data is not None:
            _assign_axis(out_hist, image.histogram_data, display_axis, start, stop)
        if result.slab_nbytes is not None:
            total_slab_nbytes += int(result.slab_nbytes)
        _check_cancelled(cancellation_token)

    _check_cancelled(cancellation_token)
    display = DisplayImage(data=out_data, histogram_data=out_hist, default_levels=image.default_levels)
    return EvaluationResult(
        value=display,
        eval_ms=(perf_counter() - start_time) * 1000.0,
        slab_shape=tuple(out_data.shape),
        slab_nbytes=total_slab_nbytes,
        mode="chunked",
        chunk_count=len(chunks),
        region_plan=getattr(result, "region_plan", None),
    )


def _assign_axis(output, chunk, axis, start, stop):
    index = [slice(None)] * output.ndim
    index[int(axis)] = slice(int(start), int(stop))
    output[tuple(index)] = chunk


def _evaluate_image_snapshot(evaluate, document, view_state, *, colormap_lut, cancellation_token, stage_cache, stage_document_key, evaluation_context):
    kwargs = {
        "colormap_lut": colormap_lut,
        "cancellation_token": cancellation_token,
    }
    if stage_cache is not None or stage_document_key is not None:
        kwargs["stage_cache"] = stage_cache
        kwargs["stage_document_key"] = stage_document_key
    if evaluation_context is not None:
        kwargs["evaluation_context"] = evaluation_context
    return evaluate(document, view_state, **kwargs)


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()
