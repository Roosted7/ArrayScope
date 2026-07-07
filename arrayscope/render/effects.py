"""Qt-free montage evaluation effects for the modular render pipeline.

R2 moves worker-side tile evaluation out of ``window.frame_renderer`` so the
pipeline can submit kernel tasks without reaching back into the GUI
orchestrator.  These functions deliberately return the same payload shapes as
the legacy methods while keeping all Qt/backend work out of the module.
"""

from __future__ import annotations

from dataclasses import replace
import os
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from arrayscope.display.lod import LodInfo, factor_xy_for_level
from arrayscope.display.montage import RenderedTile
from arrayscope.display.model.montage_levels import (
    provisional_tile_level_stats,
    sample_tile_level_stats,
)
from arrayscope.display.pyramid import PyramidLevelKey
from arrayscope.display.shader_mapping import (
    TexturePlaneKind,
    apply_scale as apply_shader_scale,
    extract_component,
)
from arrayscope.display.slice_engine import (
    make_image,
    make_image_from_slab,
    make_shader_image_from_slab,
)
from arrayscope.operations.capabilities import (
    pipeline_commutes_for_display_lod,
    pipeline_supports_reduced_display_lod,
)
from arrayscope.operations.evaluator import evaluate_image_snapshot, stage_document_key
from arrayscope.operations.pipeline import ArrayDocument, evaluate as evaluate_pipeline
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec
from arrayscope.operations.slabs import (
    evaluate_slab_from_stage,
    plan_slab,
    request_for_image,
)
from arrayscope.operations.source_read import read_base_region
from arrayscope.presentation import LevelPhase
from arrayscope.render.ladder import TileLodState
from arrayscope.window import montage_lod


def evaluate_exact_tile(
    session,
    tile,
    *,
    stage_cache,
    stage_materializer=None,
    cancellation_token=None,
    evaluation_context=None,
):
    """Evaluate one exact montage tile without touching Qt state.

    ``session`` is the immutable-ish montage snapshot object already used by
    the legacy worker path; external services that used to come from
    ``self.win`` are explicit keyword arguments.
    """

    start = perf_counter()
    stage_key = session.stage_fan_in.tile_stage_keys.get(int(tile.montage_index))
    stage_value = None if stage_key is None else session.stage_fan_in.values.get(stage_key)
    if stage_value is None and stage_key is not None and stage_cache is not None:
        getter = stage_cache.get_containing if hasattr(stage_cache, "get_containing") else stage_cache.get
        stage_value = getter(stage_key)
    if stage_value is not None:
        request = request_for_image(tile.view_state)
        plan = session.stage_fan_in.tile_stage_plans.get(int(tile.montage_index))
        candidate = session.stage_fan_in.tile_stage_candidates.get(int(tile.montage_index))
        if plan is None or candidate is None:
            plan = plan_slab(session.document, request)
            candidates = tuple(getattr(plan.region_plan, "cache_candidates", ()))
            if stage_materializer is not None:
                document_key = stage_document_key(session.document)
                candidate = next(
                    (
                        item
                        for item in candidates
                        if stage_materializer.key_for_candidate(document_key, item) == stage_key
                    ),
                    None,
                )
        if candidate is not None:
            slab = evaluate_slab_from_stage(
                session.document,
                request,
                plan,
                stage_value,
                candidate,
                cancellation_token=cancellation_token,
                evaluation_context=evaluation_context,
            )
            if bool(getattr(session, "shader_display", False)):
                display_image = make_shader_image_from_slab(
                    slab,
                    request,
                    colormap_lut=session.colormap_lut,
                    provisional_histogram=True,
                )
            else:
                display_image = make_image_from_slab(
                    slab,
                    request,
                    colormap_lut=session.colormap_lut,
                )
            display_image = attach_montage_tile_level_stats(
                display_image,
                tile,
                refined=not bool(getattr(session, "shader_display", False)),
            )
            from arrayscope.operations.evaluator import EvaluationResult

            return EvaluationResult(
                value=display_image,
                eval_ms=(perf_counter() - start) * 1000.0,
                slab_shape=tuple(np.shape(slab)),
                slab_nbytes=int(getattr(slab, "nbytes", 0)),
                region_plan=plan.region_plan,
                compute_path="stage_backed",
            )

    result = evaluate_image_snapshot(
        session.document,
        tile.view_state,
        colormap_lut=session.colormap_lut,
        cancellation_token=cancellation_token,
        shader_display=bool(getattr(session, "shader_display", False)),
        provisional_histogram=bool(getattr(session, "shader_display", False)),
        stage_cache=stage_cache,
        stage_document_key=stage_document_key(session.document),
        evaluation_context=evaluation_context,
    )
    return replace(
        result,
        value=attach_montage_tile_level_stats(
            result.value,
            tile,
            refined=not bool(getattr(session, "shader_display", False)),
        ),
    )


def evaluate_preview_tile(
    session,
    tile,
    *,
    demand,
    semantic_source_id,
    level: int | None = None,
    cancellation_token=None,
    shader_display: bool,
    evaluation_context=None,
):
    """Evaluate a display-only preview payload for a cold exact tile."""

    if not can_evaluate_preview(session, tile):
        return None
    if not can_evaluate_reduced_preview(session, tile):
        return _evaluate_tile_native_output_preview(
            session,
            tile,
            demand=demand,
            semantic_source_id=semantic_source_id,
            cancellation_token=cancellation_token,
            shader_display=shader_display,
            evaluation_context=evaluation_context,
        )
    level = preview_evaluation_level(session, demand) if level is None else int(level)
    factor_xy = factor_xy_for_level(demand, level)
    reduced_base, preview_state = read_reduced_preview_base_and_state(
        session.document,
        tile.view_state,
        factor_xy=factor_xy,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    preview_document = ArrayDocument(
        reduced_base,
        steps=session.document.steps,
        revision=session.document.revision,
    )
    result = evaluate_image_snapshot(
        preview_document,
        preview_state,
        colormap_lut=session.colormap_lut,
        cancellation_token=cancellation_token,
        degraded=True,
        shader_display=bool(shader_display),
        provisional_histogram=True,
        evaluation_context=evaluation_context,
    )
    value = replace(
        result.value,
        semantic_data=None,
        level_data=getattr(result.value, "level_data", None),
        level_stats=getattr(result.value, "level_stats", None),
        lod=LodInfo(
            level=level,
            factor=max(int(factor_xy[0]), int(factor_xy[1])),
            source_shape=tuple(int(value) for value in session.plan.tile_shape[:2]),
            texture_shape=tuple(int(value) for value in np.shape(result.value.data)[:2]),
            gutter=0,
        ),
    )
    rendered = rendered_tile_from_evaluation_result(
        tile,
        replace(result, value=value, compute_path="preview_reduced_input"),
    )
    key = montage_lod.pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=semantic_source_id,
        shader_display=bool(shader_display),
    )
    source, _histogram, _kind = montage_lod.texture_source_for_rendered(
        rendered,
        shader_display=bool(shader_display),
    )
    histogram = getattr(value, "histogram_data", None)
    return (
        key,
        np.asarray(source),
        None if histogram is None else np.asarray(histogram),
        getattr(value, "shader_mapping", None),
        getattr(value, "texture_kind", None),
        getattr(value, "level_data", None),
        getattr(value, "level_stats", None),
    )


def evaluate_shared_preview(
    session,
    seed_tile,
    tiles,
    *,
    demand,
    level: int | None = None,
    cancellation_token=None,
    shader_display: bool,
    evaluation_context=None,
    upload_preview_useful: bool = False,
):
    """Evaluate one reduced display volume and fan it out as preview planes."""

    if not shared_preview_is_useful(
        session,
        seed_tile,
        demand,
        upload_preview_useful=upload_preview_useful,
    ):
        return ()
    level = preview_evaluation_level(session, demand) if level is None else int(level)
    factor_xy = factor_xy_for_level(demand, level)
    independent_tiles = preview_pipeline_commutes_for_display_lod(session, seed_tile)
    axis_overrides = {}
    slice_remaps = {}
    if independent_tiles:
        override = _shared_preview_axis_override(session, tiles)
        if override is not None:
            axis, values, local_indices = override
            axis_overrides[int(axis)] = values
            slice_remaps[int(axis)] = local_indices
    reduced_base, _preview_state = read_reduced_preview_base_and_state(
        session.document,
        seed_tile.view_state,
        factor_xy=factor_xy,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
        axis_region_overrides=axis_overrides,
    )
    transformed = _evaluate_reduced_preview_volume(
        session.document,
        reduced_base,
        cancellation_token=cancellation_token,
    )
    previews = []
    shader_preview = bool(shader_display) or (
        not bool(getattr(session, "rgb", False)) and not np.iscomplexobj(transformed)
    )
    preview_document = (
        ArrayDocument(transformed, revision=session.document.revision)
        if shader_preview
        else None
    )
    for tile in tuple(tiles or ()):
        _check_preview_cancelled(cancellation_token)
        preview_state = reduced_preview_view_state(
            tile.view_state,
            np.shape(transformed),
            factor_xy=factor_xy,
            slice_remap=_shared_preview_slice_remap(session, tile, slice_remaps=slice_remaps),
        )
        if shader_preview:
            result = evaluate_image_snapshot(
                preview_document,
                preview_state,
                colormap_lut=session.colormap_lut,
                cancellation_token=cancellation_token,
                degraded=True,
                shader_display=True,
                provisional_histogram=True,
                evaluation_context=evaluation_context,
            )
            value = result.value
        else:
            value = make_image(transformed, preview_state, colormap_lut=session.colormap_lut)
        value = replace(
            value,
            semantic_data=None,
            level_data=getattr(value, "level_data", None),
            level_stats=getattr(value, "level_stats", None),
            lod=LodInfo(
                level=level,
                factor=max(int(factor_xy[0]), int(factor_xy[1])),
                source_shape=tuple(int(value) for value in session.plan.tile_shape[:2]),
                texture_shape=tuple(int(value) for value in np.shape(value.data)[:2]),
                gutter=0,
            ),
        )
        rendered = RenderedTile(
            tile=tile,
            image=value.data,
            histogram_data=value.histogram_data,
            eval_ms=0.0,
            slab_shape=tuple(np.shape(value.data)),
            slab_nbytes=int(getattr(np.asarray(value.data), "nbytes", 0) or 0),
            shader_mapping=getattr(value, "shader_mapping", None),
            texture_kind=getattr(value, "texture_kind", None),
            semantic_data=None,
            semantic_histogram_data=None,
            lod=getattr(value, "lod", None),
            level_data=getattr(value, "level_data", None),
            level_stats=getattr(value, "level_stats", None),
        )
        key = montage_lod.pyramid_key_for_rendered(
            rendered,
            demand=demand,
            level=level,
            semantic_source_id=session.tile_semantic_source_id(tile.source_index),
            shader_display=bool(shader_display),
        )
        source, _histogram, _kind = montage_lod.texture_source_for_rendered(
            rendered,
            shader_display=bool(shader_display),
        )
        histogram = getattr(value, "histogram_data", None)
        previews.append(
            (
                int(tile.montage_index),
                key,
                np.asarray(source),
                None if histogram is None else np.asarray(histogram),
                getattr(value, "shader_mapping", None),
                getattr(value, "texture_kind", None),
                getattr(value, "level_data", None),
                getattr(value, "level_stats", None),
            )
        )
    return tuple(previews)


def tile_lod_states(session, demand=None, *, tile_numbers=None) -> tuple[TileLodState, ...]:
    """Snapshot ladder inputs from lifecycle records and pyramid residency.

    The lifecycle machine owns acknowledged presentation/residency events; the
    pyramid cache is the physical store that may contain resident levels for a
    rendered tile.  This helper reads both without mutating either.
    """

    allowed = None if tile_numbers is None else {int(value) for value in tuple(tile_numbers)}
    ranked: list[tuple[tuple, TileLodState]] = []
    payloads = dict(getattr(getattr(session, "tile_presentation_state", None), "payloads", {}) or {})
    rendered_tiles = dict(getattr(session, "rendered_tiles", {}) or {})
    visible_numbers = set(getattr(session, "visible_tile_numbers", ()) or ())
    focus = _viewport_focus(getattr(session, "view_range", None))
    preview_cache = None
    preview_cache_fn = getattr(session, "preview_floor_cache", None)
    if callable(preview_cache_fn):
        preview_cache = preview_cache_fn()
    for tile in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ()):
        tile_number = int(tile.montage_index)
        if allowed is not None and tile_number not in allowed:
            continue
        if tile_number in set(getattr(session, "skipped_tiles", ()) or ()):
            continue
        if tile_number in set(getattr(session, "active_tile_requests", ()) or ()):
            continue
        record = session.lifecycle.peek(tile_number)
        resident_levels = set(_resident_levels_from_lifecycle(record))
        rendered = rendered_tiles.get(tile_number)
        if rendered is not None:
            # Native resident. For *planning* this dominates every pyramid
            # level (finest_available()==0 → no steps), so probing the
            # pyramid per acceptable level here was pure overhead — at
            # ~10 keys × N tiles × replan it was a measurable slice of the
            # O(N²) replan storm. Presented-level *selection* happens in the
            # commit path, which reads the pyramid directly.
            resident_levels.add(0)
        payload = payloads.get(tile_number)
        lod = None if payload is None else getattr(payload, "lod", None)
        presented_level = None if lod is None else int(getattr(lod, "level", 0) or 0)
        ranked.append(
            (
                _tile_priority_rank(tile, focus=focus, visible=tile_number in visible_numbers),
                TileLodState(
                    tile_number=tile_number,
                    resident_levels=tuple(sorted(resident_levels)),
                    presented_level=presented_level,
                    floor_available=_floor_available(session, tile, demand, preview_cache=preview_cache),
                ),
            )
        )
    # Priority order is part of the contract: the ladder preserves this
    # order inside each rung and the kernel executes FIFO within equal
    # priority, so visible tiles fill center-out before off-screen tiles.
    ranked.sort(key=lambda item: item[0])
    return tuple(state for _rank, state in ranked)


def _viewport_focus(view_range) -> tuple[float, float] | None:
    try:
        (x0, x1), (y0, y1) = view_range
        return ((float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0)
    except Exception:
        return None


def _tile_priority_rank(tile, *, focus, visible: bool) -> tuple:
    if focus is None:
        return (0 if visible else 1, 0.0, int(tile.montage_index))
    center_x = float(getattr(tile, "x0", 0)) + float(getattr(tile, "width", 0)) / 2.0
    center_y = float(getattr(tile, "y0", 0)) + float(getattr(tile, "height", 0)) / 2.0
    distance = (center_x - focus[0]) ** 2 + (center_y - focus[1]) ** 2
    return (0 if visible else 1, float(distance), int(tile.montage_index))


def rendered_tile_from_evaluation_result(tile, result) -> RenderedTile:
    value = result.value
    return RenderedTile(
        tile=tile,
        image=value.data,
        histogram_data=value.histogram_data,
        eval_ms=float(getattr(result, "eval_ms", 0.0) or 0.0),
        slab_shape=tuple(getattr(result, "slab_shape", np.shape(value.data))),
        slab_nbytes=getattr(result, "slab_nbytes", None),
        shader_mapping=getattr(value, "shader_mapping", None),
        texture_kind=getattr(value, "texture_kind", None),
        semantic_data=getattr(value, "semantic_data", None),
        semantic_histogram_data=getattr(value, "semantic_histogram_data", None),
        lod=getattr(value, "lod", None),
        level_data=getattr(value, "level_data", None),
        level_stats=getattr(value, "level_stats", None),
    )


def _resident_levels_from_lifecycle(record) -> tuple[int, ...]:
    if record is None:
        return ()
    levels = []
    for key, entry in dict(getattr(record, "levels", {}) or {}).items():
        if getattr(entry, "phase", None) is not LevelPhase.RESIDENT:
            continue
        level_xy = tuple(getattr(key, "level_xy", ()) or ())
        if level_xy:
            levels.append(max(int(value) for value in level_xy))
    return tuple(sorted(set(levels)))


def _floor_available(session, tile, demand, *, preview_cache) -> bool:
    if preview_cache is None or demand is None:
        return False
    semantic_id = session.tile_semantic_source_id(tile.source_index)
    key = preview_claim_key(
        session,
        tile,
        demand=demand,
        semantic_source_id=semantic_id,
        shader_display=bool(getattr(session, "shader_display", False)),
    )
    return preview_cache.peek(key) is not None


def can_evaluate_preview(session, tile) -> bool:
    document = getattr(session, "document", None)
    view_state = getattr(tile, "view_state", None)
    if document is None or view_state is None or getattr(view_state, "image_axes", None) is None:
        return False
    base_shape = tuple(int(size) for size in np.shape(getattr(document, "base_data", ())))
    if len(base_shape) != int(getattr(view_state, "ndim", len(base_shape))):
        return False
    return True


def can_evaluate_reduced_preview(session, tile) -> bool:
    if not can_evaluate_preview(session, tile):
        return False
    document = getattr(session, "document", None)
    view_state = getattr(tile, "view_state", None)
    base_shape = tuple(int(size) for size in np.shape(getattr(document, "base_data", ())))
    dtype = getattr(getattr(document, "base_data", None), "dtype", None)
    return pipeline_supports_reduced_display_lod(
        getattr(document, "enabled_operations", ()),
        base_shape,
        dtype,
        display_axes=tuple(int(axis) for axis in view_state.image_axes),
    )


def preview_is_useful(session, tile, demand, *, upload_preview_useful: bool = False) -> bool:
    if not can_evaluate_preview(session, tile):
        return False
    if demand is None:
        return False
    if preview_evaluation_level(session, demand) <= int(getattr(demand, "desired_level", 0) or 0):
        return False
    return bool(preview_pipeline_commutes_for_display_lod(session, tile) or upload_preview_useful)


def shared_preview_is_useful(session, tile, demand, *, upload_preview_useful: bool = False) -> bool:
    if not can_evaluate_reduced_preview(session, tile):
        return False
    if demand is None:
        return False
    if preview_evaluation_level(session, demand) <= int(getattr(demand, "desired_level", 0) or 0):
        return False
    if preview_pipeline_commutes_for_display_lod(session, tile):
        return True
    return _shared_transform_preview_enabled()


def preview_pipeline_commutes_for_display_lod(session, tile) -> bool:
    document = getattr(session, "document", None)
    operations = tuple(getattr(document, "enabled_operations", ()) or ())
    base_shape = tuple(int(size) for size in np.shape(getattr(document, "base_data", ())))
    dtype = getattr(getattr(document, "base_data", None), "dtype", None)
    return pipeline_commutes_for_display_lod(operations, base_shape, dtype)


def preview_evaluation_level(session, demand) -> int:
    desired = int(getattr(demand, "desired_level", 0) or 0)
    preview = int(getattr(session, "lod_preview_level", 0) or 0)
    return max(desired, preview)


def read_reduced_preview_base_and_state(
    document,
    view_state,
    *,
    factor_xy: tuple[int, int],
    cancellation_token=None,
    evaluation_context=None,
    axis_region_overrides=None,
) -> tuple[np.ndarray, object]:
    axis_region_overrides = {
        int(axis): tuple(int(value) for value in tuple(values or ()))
        for axis, values in dict(axis_region_overrides or {}).items()
    }
    base_shape = tuple(int(size) for size in np.shape(document.base_data))
    image_axes = tuple(int(axis) for axis in view_state.image_axes)
    reduced_shape = list(base_shape)
    reduced_shape[image_axes[0]] = _reduced_axis_length(
        _display_axis_region_for_preview(view_state, image_axes[0], base_shape[image_axes[0]]),
        base_shape[image_axes[0]],
        max(1, int(factor_xy[1])),
    )
    reduced_shape[image_axes[1]] = _reduced_axis_length(
        _display_axis_region_for_preview(view_state, image_axes[1], base_shape[image_axes[1]]),
        base_shape[image_axes[1]],
        max(1, int(factor_xy[0])),
    )
    reduced_shape = tuple(int(size) for size in reduced_shape)
    planning_document = ArrayDocument(
        np.empty(reduced_shape, dtype=getattr(document.base_data, "dtype", np.float32)),
        steps=document.steps,
        revision=document.revision,
    )
    planning_state = reduced_preview_view_state(view_state, reduced_shape, factor_xy=factor_xy)
    region_plan = plan_slab(planning_document, request_for_image(planning_state)).region_plan
    required = region_plan.required_input_region
    read_axes = []
    slice_remap = {}
    for axis, size in enumerate(base_shape):
        if axis in image_axes:
            read_axes.append(_display_axis_region_for_preview(view_state, axis, size))
            continue
        if axis in axis_region_overrides:
            values = axis_region_overrides[axis]
            read_axes.append(axis_region_for_preview_indices(values, int(size)))
            local_indices = {int(value): int(offset) for offset, value in enumerate(values)}
            slice_remap[axis] = int(local_indices.get(int(view_state.slice_indices[axis]), 0))
            continue
        read_axis, local_index = _native_preview_axis_region(
            required.axes[axis],
            int(view_state.slice_indices[axis]),
        )
        read_axes.append(read_axis)
        slice_remap[axis] = local_index
    tile_base = read_base_region(
        document.base_data,
        RegionSpec(tuple(read_axes)),
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    reduced = reduce_array_display_axes(tile_base, image_axes, factor_xy)
    preview_state = reduced_preview_view_state(
        view_state,
        np.shape(reduced),
        factor_xy=factor_xy,
        slice_remap=slice_remap,
    )
    return reduced, preview_state


def axis_region_for_preview_indices(values, size: int) -> AxisRegion:
    indices = tuple(int(value) for value in tuple(values or ()))
    if not indices:
        return AxisRegion(AxisRegionKind.SLICE, (0, 0, 1))
    if indices == tuple(range(int(size))):
        return AxisRegion(AxisRegionKind.ALL)
    start = indices[0]
    stop = indices[-1] + 1
    if start >= 0 and stop <= int(size) and indices == tuple(range(start, stop)):
        return AxisRegion(AxisRegionKind.SLICE, (start, stop, 1))
    return AxisRegion(AxisRegionKind.INDICES, indices)


def reduce_array_display_axes(
    array,
    image_axes: tuple[int, int],
    factor_xy: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(array)
    y_axis, x_axis = (int(image_axes[0]), int(image_axes[1]))
    factor_x, factor_y = (max(1, int(factor_xy[0])), max(1, int(factor_xy[1])))
    reduced = reduce_nd_axis_mean(values, y_axis, factor_y)
    reduced = reduce_nd_axis_mean(reduced, x_axis, factor_x)
    return reduced


def reduce_nd_axis_mean(values: np.ndarray, axis: int, factor: int) -> np.ndarray:
    if int(factor) <= 1 or int(values.shape[int(axis)]) <= 1:
        return values
    if int(factor) & (int(factor) - 1):
        raise ValueError("preview reduction factor must be a power of two")
    axis = int(axis)
    length = int(values.shape[axis])
    starts = np.arange(0, length, int(factor))
    if np.iscomplexobj(values):
        source = values.astype(np.complex64, copy=False)
    else:
        source = values.astype(np.float32, copy=False)
    sums = np.add.reduceat(source, starts, axis=axis, dtype=source.dtype)
    counts = np.diff(np.append(starts, length)).astype(np.float32)
    shape = [1] * sums.ndim
    shape[axis] = len(starts)
    reduced = sums / counts.reshape(shape)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        return np.clip(np.rint(reduced), info.min, info.max).astype(values.dtype)
    if np.iscomplexobj(values):
        return reduced.astype(np.complex64, copy=False)
    return reduced.astype(np.float32, copy=False)


def reduce_display_payload_axes(array, factor_xy: tuple[int, int]) -> np.ndarray:
    values = np.asarray(array)
    reduced = reduce_nd_axis_mean(values, 0, max(1, int(factor_xy[1])))
    reduced = reduce_nd_axis_mean(reduced, 1, max(1, int(factor_xy[0])))
    return reduced


def reduced_preview_view_state(view_state, shape, *, factor_xy: tuple[int, int], slice_remap=None):
    slice_remap = {} if slice_remap is None else dict(slice_remap)
    shape = tuple(int(size) for size in shape)
    image_axes = tuple(int(axis) for axis in view_state.image_axes)
    axis_factors = {
        image_axes[0]: max(1, int(factor_xy[1])),
        image_axes[1]: max(1, int(factor_xy[0])),
    }
    slice_indices = []
    axis_ranges = []
    axis_text = []
    for axis, size in enumerate(shape):
        if axis in image_axes:
            factor = axis_factors[axis]
            axis_ranges.append(None)
            axis_text.append(None)
            slice_indices.append(
                min(max(0, int(view_state.slice_indices[axis]) // factor), max(0, size - 1))
            )
        else:
            axis_ranges.append(None)
            axis_text.append(None)
            slice_indices.append(
                min(
                    max(0, int(slice_remap.get(axis, view_state.slice_indices[axis]))),
                    max(0, size - 1),
                )
            )
    return replace(
        view_state,
        shape=shape,
        slice_indices=tuple(slice_indices),
        axis_range_indices=tuple(axis_ranges),
        axis_range_text=tuple(axis_text),
        montage_axis=None,
        montage_columns=None,
        montage_indices=None,
        montage_text=None,
    )


def attach_montage_tile_level_stats(display_image, tile, *, refined: bool = False):
    if getattr(display_image, "level_stats", None) is not None:
        return display_image
    level_data = getattr(display_image, "level_data", None)
    if level_data is not None:
        stats = (
            sample_tile_level_stats(level_data, int(tile.source_index), refined=True)
            if refined
            else provisional_tile_level_stats(level_data, int(tile.source_index))
        )
        if stats is not None:
            return replace(display_image, level_stats=stats)
    values = montage_refined_level_values(
        SimpleNamespace(
            image=getattr(display_image, "data", None),
            histogram_data=getattr(display_image, "histogram_data", None),
            shader_mapping=getattr(display_image, "shader_mapping", None),
            semantic_data=getattr(display_image, "semantic_data", None),
        )
    )
    stats = (
        sample_tile_level_stats(values, int(tile.source_index), refined=True)
        if refined
        else provisional_tile_level_stats(values, int(tile.source_index))
    )
    if stats is not None:
        return replace(display_image, level_stats=stats)
    return display_image


def montage_refined_level_values(rendered) -> np.ndarray:
    semantic_histogram = getattr(rendered, "semantic_histogram_data", None)
    if semantic_histogram is not None:
        return np.asarray(semantic_histogram)
    histogram = getattr(rendered, "histogram_data", None)
    if histogram is not None:
        return np.asarray(histogram)
    mapping = getattr(rendered, "shader_mapping", None)
    semantic = getattr(rendered, "semantic_data", None)
    if mapping is not None and semantic is not None:
        values = extract_component(np.asarray(semantic), getattr(mapping, "component", "real"))
        return apply_shader_scale(
            values,
            getattr(mapping, "scale", "linear"),
            symlog_constant=float(getattr(mapping, "symlog_constant", 0.0) or 0.0),
        )
    image = getattr(rendered, "image", None)
    if image is None:
        return np.asarray((), dtype=np.float32)
    image = np.asarray(image)
    if np.iscomplexobj(image):
        return np.abs(image).astype(np.float32, copy=False)
    return image


def preview_claim_key(session, tile, *, demand, semantic_source_id, shader_display: bool):
    level = preview_evaluation_level(session, demand)
    return PyramidLevelKey(
        source_id=semantic_source_id,
        tile_id=int(tile.source_index),
        component=_preview_claim_component(session, shader_display=shader_display),
        level_xy=(int(level), int(level)),
    )


def _evaluate_tile_native_output_preview(
    session,
    tile,
    *,
    demand,
    semantic_source_id,
    cancellation_token,
    shader_display: bool,
    evaluation_context,
):
    level = preview_evaluation_level(session, demand)
    factor_xy = factor_xy_for_level(demand, level)
    result = evaluate_image_snapshot(
        session.document,
        tile.view_state,
        colormap_lut=session.colormap_lut,
        cancellation_token=cancellation_token,
        degraded=True,
        shader_display=bool(shader_display),
        provisional_histogram=True,
        evaluation_context=evaluation_context,
    )
    reduced_data = reduce_display_payload_axes(result.value.data, factor_xy)
    histogram = getattr(result.value, "histogram_data", None)
    reduced_histogram = None if histogram is None else reduce_display_payload_axes(histogram, factor_xy)
    value = replace(
        result.value,
        data=reduced_data,
        histogram_data=reduced_histogram,
        semantic_data=None,
        level_data=getattr(result.value, "level_data", None),
        level_stats=getattr(result.value, "level_stats", None),
        lod=LodInfo(
            level=level,
            factor=max(int(factor_xy[0]), int(factor_xy[1])),
            source_shape=tuple(int(value) for value in session.plan.tile_shape[:2]),
            texture_shape=tuple(int(value) for value in np.shape(reduced_data)[:2]),
            gutter=0,
        ),
    )
    rendered = rendered_tile_from_evaluation_result(
        tile,
        replace(result, value=value, compute_path="preview_native_output_reduction"),
    )
    key = montage_lod.pyramid_key_for_rendered(
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=semantic_source_id,
        shader_display=bool(shader_display),
    )
    source, _histogram, _kind = montage_lod.texture_source_for_rendered(
        rendered,
        shader_display=bool(shader_display),
    )
    return (
        key,
        np.asarray(source),
        None if reduced_histogram is None else np.asarray(reduced_histogram),
        getattr(value, "shader_mapping", None),
        getattr(value, "texture_kind", None),
        getattr(value, "level_data", None),
        getattr(value, "level_stats", None),
    )


def _native_preview_axis_region(axis_region: AxisRegion, output_index: int) -> tuple[AxisRegion, int]:
    kind = AxisRegionKind(axis_region.kind)
    if kind == AxisRegionKind.POINT:
        index = int(axis_region.value)
        return AxisRegion(AxisRegionKind.SLICE, (index, index + 1, 1)), 0
    if kind == AxisRegionKind.ALL:
        return axis_region, int(output_index)
    if kind == AxisRegionKind.SLICE:
        start, stop, step = axis_region.value
        values = tuple(range(int(start), int(stop), int(step)))
        return axis_region, values.index(int(output_index)) if int(output_index) in values else 0
    if kind == AxisRegionKind.INDICES:
        values = tuple(int(value) for value in axis_region.value)
        return axis_region, values.index(int(output_index)) if int(output_index) in values else 0
    return axis_region, 0


def _reduced_axis_length(axis_region: AxisRegion, size: int, factor: int) -> int:
    if AxisRegionKind(axis_region.kind) == AxisRegionKind.ALL:
        length = int(size)
    elif AxisRegionKind(axis_region.kind) == AxisRegionKind.SLICE:
        start, stop, step = axis_region.value
        length = len(range(*slice(int(start), int(stop), int(step)).indices(int(size))))
    elif AxisRegionKind(axis_region.kind) == AxisRegionKind.INDICES:
        length = len(tuple(axis_region.value))
    else:
        length = 1
    return max(1, int(np.ceil(length / max(1, int(factor)))))


def _display_axis_region_for_preview(view_state, axis: int, size: int) -> AxisRegion:
    indices = view_state.axis_range_indices[int(axis)]
    if indices is None:
        return AxisRegion(AxisRegionKind.ALL)
    values = tuple(int(index) for index in indices)
    if not values:
        return AxisRegion(AxisRegionKind.SLICE, (0, 0, 1))
    start = values[0]
    stop = values[-1] + 1
    if start >= 0 and stop <= int(size) and values == tuple(range(start, stop)):
        return AxisRegion(AxisRegionKind.SLICE, (start, stop, 1))
    return AxisRegion(AxisRegionKind.INDICES, values)


def _shared_transform_preview_enabled() -> bool:
    return str(os.environ.get("ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _shared_preview_axis_override(session, tiles):
    axis = getattr(session, "montage_axis", None)
    if axis is None:
        return None
    values = tuple(
        dict.fromkeys(
            int(getattr(tile, "source_index", 0))
            for tile in tuple(tiles or ())
        )
    )
    if not values:
        return None
    local_indices = {int(source_index): int(offset) for offset, source_index in enumerate(values)}
    return int(axis), values, local_indices


def _evaluate_reduced_preview_volume(document, reduced_base, *, cancellation_token):
    _check_preview_cancelled(cancellation_token)
    data = evaluate_pipeline(reduced_base, getattr(document, "enabled_operations", ()))
    _check_preview_cancelled(cancellation_token)
    return data


def _shared_preview_slice_remap(session, tile, *, slice_remaps=None) -> dict[int, int]:
    axis = getattr(session, "montage_axis", None)
    if axis is None:
        return {}
    axis = int(axis)
    source_index = int(tile.source_index)
    local_indices = dict((slice_remaps or {}).get(axis, {}) or {})
    return {axis: int(local_indices.get(source_index, source_index))}


def _check_preview_cancelled(cancellation_token) -> None:
    if cancellation_token is not None and bool(getattr(cancellation_token, "cancelled", False)):
        from arrayscope.operations.cancellation import EvaluationCancelled

        raise EvaluationCancelled()


def _preview_claim_component(session, *, shader_display: bool) -> str:
    if bool(shader_display):
        return str(TexturePlaneKind.COMPLEX_RG32F.value)
    if not bool(shader_display) and bool(getattr(session, "rgb", False)):
        return str(TexturePlaneKind.RGB8.value)
    return "scalar"


__all__ = [
    "attach_montage_tile_level_stats",
    "axis_region_for_preview_indices",
    "can_evaluate_preview",
    "can_evaluate_reduced_preview",
    "evaluate_exact_tile",
    "evaluate_preview_tile",
    "evaluate_shared_preview",
    "montage_refined_level_values",
    "preview_claim_key",
    "preview_evaluation_level",
    "preview_is_useful",
    "preview_pipeline_commutes_for_display_lod",
    "read_reduced_preview_base_and_state",
    "reduce_array_display_axes",
    "reduce_display_payload_axes",
    "reduce_nd_axis_mean",
    "reduced_preview_view_state",
    "rendered_tile_from_evaluation_result",
    "shared_preview_is_useful",
    "tile_lod_states",
]
