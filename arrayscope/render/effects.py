"""Qt-free montage evaluation effects for the modular render pipeline.

R2 moves worker-side tile evaluation out of the window frame controller so the
pipeline can submit kernel tasks without reaching back into the GUI
orchestrator.  These functions deliberately return the same payload shapes as
the legacy methods while keeping all Qt/backend work out of the module.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from arrayscope.display.lod import LodInfo, factor_xy_for_level
from arrayscope.display.model.montage_levels import (
    LevelEvidenceQuality,
    TileLevelStats,
    provisional_tile_level_stats,
    sample_tile_level_stats,
    tile_level_stats_with_quality,
)
from arrayscope.display.model.tile_identity import (
    acknowledged_identity_satisfies_target,
    tile_ack_identity,
)
from arrayscope.display.model.tile_priority import prioritize_tile_numbers
from arrayscope.display.montage import RenderedTile
from arrayscope.display.pyramid import MaterializedLodPage, plan_source_grid_pages
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderScale,
    TexturePlaneKind,
    extract_component,
    mapped_scalar,
)
from arrayscope.display.shader_mapping import (
    apply_scale as apply_shader_scale,
)
from arrayscope.display.slice_engine import (
    make_image,
    make_image_from_slab,
    make_shader_image_from_slab,
)
from arrayscope.gpu.chunk_summary import aggregate_chunk_summaries, summarize_chunk
from arrayscope.operations.capabilities import (
    pipeline_commutes_for_display_lod,
    pipeline_supports_reduced_display_lod,
)
from arrayscope.operations.evaluator import evaluate_image_snapshot, stage_document_key
from arrayscope.operations.pipeline import ArrayDocument
from arrayscope.operations.pipeline import evaluate as evaluate_pipeline
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec, region_contains
from arrayscope.operations.slabs import (
    evaluate_slab_from_stage,
    plan_slab,
    request_for_image,
)
from arrayscope.operations.source_read import read_base_region
from arrayscope.presentation import LevelPhase
from arrayscope.render import lod as render_lod
from arrayscope.render.ladder import TileLodState

SHARED_PREVIEW_ROUTE = ("shared-transform-preview", 1, "sample-display-axes-before-operations")


def _check_render_cancelled(token) -> None:
    if token is not None and bool(getattr(token, "cancelled", False)):
        from arrayscope.operations.cancellation import EvaluationCancelled

        raise EvaluationCancelled()


def _evaluate_native_tile_result(
    session,
    tile,
    *,
    stage_cache,
    stage_materializer=None,
    cancellation_token=None,
    evaluation_context=None,
):
    """Evaluate one native montage tile without touching Qt state.

    ``session`` is the immutable-ish montage snapshot object already used by
    the legacy worker path; external services that used to come from
    ``self.win`` are explicit keyword arguments.
    """

    _check_render_cancelled(cancellation_token)
    start = perf_counter()
    tile_number = int(tile.montage_index)
    stage_key = session.stage_fan_in.tile_stage_keys.get(tile_number)
    plan = session.stage_fan_in.tile_stage_plans.get(tile_number)
    candidate = session.stage_fan_in.tile_stage_candidates.get(tile_number)
    if stage_key is None and candidate is not None and stage_materializer is not None:
        document_key = stage_document_key(session.document)
        stage_key = stage_materializer.key_for_candidate(document_key, candidate)
    stage_value = None if stage_key is None else session.stage_fan_in.values.get(stage_key)
    if stage_value is None and stage_key is not None and stage_cache is not None:
        getter = (
            stage_cache.get_containing
            if hasattr(stage_cache, "get_containing")
            else stage_cache.get
        )
        stage_value = getter(stage_key)
    if stage_value is not None:
        request = request_for_image(tile.view_state)
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
            if candidate is None and stage_key is not None:
                for item in candidates:
                    if tuple(getattr(item, "operation_prefix", ()) or ()) != tuple(
                        getattr(stage_key, "operation_prefix", ()) or ()
                    ):
                        continue
                    if str(getattr(item, "dtype", "")) != str(getattr(stage_key, "dtype", "")):
                        continue
                    if tuple(getattr(item, "shape", ()) or ()) != tuple(
                        getattr(stage_key, "shape", ()) or ()
                    ):
                        continue
                    try:
                        contains = region_contains(stage_value.region, item.region, stage_key.shape)
                    except Exception:
                        contains = False
                    if contains:
                        candidate = item
                        break
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
            _check_render_cancelled(cancellation_token)
            canonical_orientation = bool(getattr(session, "canonical_orientation", False))
            if bool(getattr(session, "shader_display", False)):
                display_image = make_shader_image_from_slab(
                    slab,
                    request,
                    colormap_lut=session.colormap_lut,
                    provisional_histogram=True,
                    canonical_orientation=canonical_orientation,
                )
            else:
                display_image = make_image_from_slab(
                    slab,
                    request,
                    colormap_lut=session.colormap_lut,
                    canonical_orientation=canonical_orientation,
                )
            display_image = attach_montage_tile_level_stats(
                display_image,
                tile,
                refined=not bool(getattr(session, "shader_display", False)),
            )
            _check_render_cancelled(cancellation_token)
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
        canonical_orientation=bool(getattr(session, "canonical_orientation", False)),
    )
    _check_render_cancelled(cancellation_token)
    result = replace(
        result,
        value=attach_montage_tile_level_stats(
            result.value,
            tile,
            refined=not bool(getattr(session, "shader_display", False)),
        ),
    )
    _check_render_cancelled(cancellation_token)
    return result


def evaluate_target_tile(
    session,
    tile,
    *,
    level: int,
    demand,
    semantic_source_id,
    stage_cache,
    stage_materializer=None,
    cancellation_token=None,
    shader_display: bool,
    evaluation_context=None,
):
    """Evaluate the current display target for one montage tile.

    Level 0 is the native semantic target and returns an ``EvaluationResult``
    for ``rendered_tiles``.  Coarser targets are display-only payload rows for
    the pyramid/presentation path; they must never masquerade as native
    semantic tiles.
    """

    if int(level) <= 0:
        return _evaluate_native_tile_result(
            session,
            tile,
            stage_cache=stage_cache,
            stage_materializer=stage_materializer,
            cancellation_token=cancellation_token,
            evaluation_context=evaluation_context,
        )
    return evaluate_preview_tile(
        session,
        tile,
        demand=demand,
        semantic_source_id=semantic_source_id,
        level=int(level),
        cancellation_token=cancellation_token,
        shader_display=shader_display,
        evaluation_context=evaluation_context,
        stage_cache=stage_cache,
        stage_materializer=stage_materializer,
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
    stage_cache=None,
    stage_materializer=None,
):
    """Evaluate a display-only payload for a cold tile."""

    return _evaluate_tile_native_output_preview(
        session,
        tile,
        demand=demand,
        semantic_source_id=semantic_source_id,
        level=level,
        cancellation_token=cancellation_token,
        shader_display=shader_display,
        evaluation_context=evaluation_context,
        stage_cache=stage_cache,
        stage_materializer=stage_materializer,
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
        sample_display_axes=True,
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
        ArrayDocument(transformed, revision=session.document.revision) if shader_preview else None
    )
    for tile in tuple(tiles or ()):
        _check_preview_cancelled(cancellation_token)
        preview_state = reduced_preview_view_state(
            tile.view_state,
            np.shape(transformed),
            factor_xy=factor_xy,
            slice_remap=_shared_preview_slice_remap(session, tile, slice_remaps=slice_remaps),
        )
        canonical_orientation = bool(getattr(session, "canonical_orientation", False))
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
                canonical_orientation=canonical_orientation,
            )
            value = result.value
        else:
            value = make_image(
                transformed,
                preview_state,
                colormap_lut=session.colormap_lut,
                canonical_orientation=canonical_orientation,
            )
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
            quality="preview",
        )
        semantic_source_id = session.tile_semantic_source_id(tile.source_index)
        format_key = render_lod.page_set_key_for_rendered(
            rendered,
            demand=demand,
            level=level,
            semantic_source_id=semantic_source_id,
            shader_display=bool(shader_display),
        )
        source, _histogram, texture_kind = render_lod.texture_source_for_rendered(
            rendered,
            shader_display=bool(shader_display),
        )
        template_plan = format_key.plans[0]
        source_height, source_width = (int(value) for value in session.plan.tile_shape[:2])
        plans = plan_source_grid_pages(
            # Reduced-before-operation shared values are deliberately
            # non-semantic.  Keep their value identity separate from direct
            # canonical display-plane pages so residency can provide a floor
            # without suppressing or aliasing later exact materialization.
            content_key=(
                "src-anchored",
                semantic_source_id,
                ("display-plane", SHARED_PREVIEW_ROUTE),
            ),
            valid_source_rect_yx=(0, source_height, 0, source_width),
            reduction_yx=template_plan.reduction_yx,
            stored_page_shape=template_plan.stored_page_shape,
            dtype=template_plan.key.dtype,
            representation=template_plan.key.representation,
            reducer=template_plan.reducer,
        )
        key = render_lod.LodPageSetKey(
            source_id=semantic_source_id,
            tile_id=int(tile.source_index),
            level_xy=format_key.level_xy,
            reducer=format_key.reducer,
            plans=plans,
        )
        pages = _materialize_shared_preview_pages(source, plans=plans)
        histogram = _preview_display_histogram(
            rendered, source, texture_kind, getattr(value, "histogram_data", None)
        )
        previews.append(
            (
                int(tile.montage_index),
                key,
                pages,
                None if histogram is None else np.asarray(histogram),
                getattr(value, "shader_mapping", None),
                texture_kind,
                getattr(value, "level_data", None),
                getattr(value, "level_stats", None),
            )
        )
    return tuple(previews)


def _materialize_shared_preview_pages(
    values,
    *,
    plans,
) -> tuple[MaterializedLodPage, ...]:
    """Partition one worker-computed stored plane under its exact page plans.

    Shared-transform preview performs its bounded numeric work before fan-out;
    reducing each row again during GUI admission would both block Qt and
    shrink its native geometry twice.  This check maps that already-stored
    plane to the canonical global stored rectangles, proving complete,
    non-overlapping coverage before any page reaches the cache.
    """

    source = np.asarray(values)
    requested = tuple(plans)
    if not requested:
        raise ValueError("shared preview requires at least one canonical page plan")
    stored_y0 = min(plan.stored_rect_yx[0] for plan in requested)
    stored_y1 = max(plan.stored_rect_yx[1] for plan in requested)
    stored_x0 = min(plan.stored_rect_yx[2] for plan in requested)
    stored_x1 = max(plan.stored_rect_yx[3] for plan in requested)
    expected_shape = (stored_y1 - stored_y0, stored_x1 - stored_x0)
    if tuple(source.shape[:2]) != expected_shape:
        raise ValueError(
            "shared preview stored shape disagrees with canonical page coverage: "
            f"{source.shape[:2]} != {expected_shape}"
        )
    covered = np.zeros(expected_shape, dtype=np.bool_)
    pages = []
    for plan in requested:
        y0, y1, x0, x1 = plan.stored_rect_yx
        rows = slice(y0 - stored_y0, y1 - stored_y0)
        columns = slice(x0 - stored_x0, x1 - stored_x0)
        if np.any(covered[rows, columns]):
            raise ValueError("shared preview canonical page plans overlap")
        covered[rows, columns] = True
        page_values = np.ascontiguousarray(
            source[rows, columns],
            dtype=np.dtype(plan.key.dtype),
        )
        pages.append(MaterializedLodPage(plan, page_values))
    if not np.all(covered):
        raise ValueError("shared preview canonical page plans leave incomplete stored coverage")
    return tuple(pages)


def tile_lod_states(
    session, demand=None, *, tile_numbers=None, scope=None
) -> tuple[TileLodState, ...]:
    """Snapshot ladder inputs from lifecycle records and pyramid residency.

    The lifecycle machine owns acknowledged presentation/residency events; the
    pyramid cache is the physical store that may contain resident levels for a
    rendered tile.  This helper reads both without mutating either.
    """

    if tile_numbers is None and scope is not None:
        tile_numbers = tuple(getattr(scope, "visible_tile_numbers", ()) or ())
    allowed = None if tile_numbers is None else {int(value) for value in tuple(tile_numbers)}
    states: list[TileLodState] = []
    payloads = dict(
        getattr(getattr(session, "tile_presentation_state", None), "payloads", {}) or {}
    )
    skipped_numbers = set(getattr(session, "skipped_tiles", ()) or ())
    active_request_numbers = set(getattr(session, "active_tile_requests", ()) or ())
    backend_identities = dict(getattr(session.lifecycle, "backend_presented_identities", {}) or {})
    presented_numbers = set(getattr(session.lifecycle, "presented_tiles", ()) or ())
    preview_cache = getattr(session, "lod_page_cache", None)
    plan_tiles = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    # Walk the ALLOWED tiles, not the whole montage. Filtering inside the loop
    # made a 12-tile viewport on a 4000-tile montage pay for all 4000. Tile
    # ``montage_index`` is its position in ``plan_tiles``, and sorting keeps
    # the original plan-order output.
    if allowed is None:
        scoped_tiles: tuple = plan_tiles
    else:
        scoped_tiles = tuple(
            plan_tiles[number] for number in sorted(allowed) if 0 <= number < len(plan_tiles)
        )
    for tile in scoped_tiles:
        tile_number = int(tile.montage_index)
        if allowed is not None and tile_number not in allowed:
            continue
        if tile_number in skipped_numbers:
            continue
        if tile_number in active_request_numbers:
            continue
        record = session.lifecycle.peek(tile_number)
        resident_levels = set(
            _resident_levels_from_lifecycle(
                record,
                source_id=session.tile_semantic_source_id(int(tile.source_index)),
                tile_id=int(tile.source_index),
                page_cache=preview_cache,
            )
        )
        payload = payloads.get(tile_number)
        payload_current = False
        if payload is not None and int(getattr(payload, "source_index", -1)) == int(
            tile.source_index
        ):
            if backend_identities:
                payload_current = backend_identities.get(tile_number) == tile_ack_identity(payload)
            else:
                payload_current = tile_number in presented_numbers
        if not payload_current:
            payload = None
        ready_ref = None if record is None else (record.target_payload or record.fallback_payload)
        ready_level = None if ready_ref is None else int(getattr(ready_ref, "lod_level", 0) or 0)
        ready_quality = "" if ready_ref is None else str(getattr(ready_ref, "quality", "") or "")
        committable_exact_payload = bool(
            payload is not None
            and str(getattr(payload, "quality", "exact") or "exact") != "preview"
        )
        committable_exact_ready = bool(
            ready_ref is not None and str(getattr(ready_ref, "quality", "") or "") != "preview"
        )
        if (
            not committable_exact_payload
            and not committable_exact_ready
            and getattr(session, "rendered_tiles", None) is not None
            and tile_number not in session.rendered_tiles
        ):
            # Lifecycle residency is physical pyramid evidence, not by itself
            # a committable payload. After an index scroll a cold edge can own
            # a resident reduced level while having neither a native RenderedTile
            # nor a presentable wrapper. Letting that level satisfy the ladder
            # produces zero steps and strands either an atomic CPU successor
            # or a shader frame at fallback quality forever. A preview wrapper
            # does not make a physically resident target rung committable.
            resident_levels.clear()
        lod = None if payload is None else getattr(payload, "lod", None)
        presented_level = None if lod is None else int(getattr(lod, "level", 0) or 0)
        presented_quality = (
            "exact" if payload is None else str(getattr(payload, "quality", "exact") or "exact")
        )
        desired = 0 if demand is None else max(0, int(getattr(demand, "desired_level", 0) or 0))
        target_quality_available = bool(
            tile_number in getattr(session, "rendered_tiles", {})
            or any(int(level) <= desired for level in resident_levels)
            or (
                presented_level is not None
                and int(presented_level) <= desired
                and presented_quality != "preview"
            )
        )
        blank = payload is None and not resident_levels
        visible_missing_count = int(getattr(scope, "visible_missing_count", 0) or 0)
        allow_preview = bool((blank and not target_quality_available) or visible_missing_count >= 2)
        states.append(
            TileLodState(
                tile_number=tile_number,
                resident_levels=tuple(sorted(resident_levels)),
                presented_level=presented_level,
                ready_level=ready_level,
                ready_quality=ready_quality,
                presented_quality=presented_quality,
                current_presentation_quality=presented_quality,
                allow_preview=allow_preview,
                target_quality_available=target_quality_available,
                floor_available=_floor_available(
                    session, tile, demand, preview_cache=preview_cache
                ),
            )
        )
    context_owner = getattr(session, "tile_priority_context", None)
    if not callable(context_owner):
        raise RuntimeError("live frame session has no tile-priority owner")
    ordered_numbers = prioritize_tile_numbers(
        (state.tile_number for state in states),
        plan_tiles=plan_tiles,
        context=context_owner(),
    )
    by_number = {int(state.tile_number): state for state in states}
    return tuple(
        replace(
            by_number[int(tile_number)],
            scheduling_rank=rank,
        )
        for rank, tile_number in enumerate(ordered_numbers)
    )


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
        lod_source_data=getattr(value, "lod_source_data", None),
        lod=getattr(value, "lod", None),
        level_data=getattr(value, "level_data", None),
        level_stats=getattr(value, "level_stats", None),
    )


def _resident_levels_from_lifecycle(
    record,
    *,
    source_id=None,
    tile_id=None,
    page_cache=None,
) -> tuple[int, ...]:
    if record is None:
        return ()
    levels = []
    # Read the mapping in place: this runs once per tile per ladder snapshot,
    # so copying it made the copy itself the dominant cost. The loop body only
    # reads, so there is no mutate-during-iteration hazard.
    for key, entry in (getattr(record, "levels", None) or {}).items():
        if getattr(entry, "phase", None) is not LevelPhase.RESIDENT:
            continue
        # Scope residency to the tile's CURRENT source.  After a scroll each
        # grid position points at a new source, but the lifecycle record keeps
        # the previous source's resident level entries.  Counting those made
        # the ladder believe the demanded level was already resident for the
        # new source and skip its refinement rung — stranding the tile at its
        # coarse floor (the onscreen-only mixed-LOD scroll stall; offscreen has
        # no prior-source residency to go stale).  best_floor_key already
        # filters by source, so the presentation and the ladder now agree.
        if source_id is not None and getattr(key, "source_id", None) != source_id:
            continue
        if tile_id is not None and int(getattr(key, "tile_id", -1)) != int(tile_id):
            continue
        if page_cache is not None and not render_lod._page_set_exact(page_cache, key):
            continue
        level_xy = tuple(getattr(key, "level_xy", ()) or ())
        if level_xy:
            levels.append(max(int(value) for value in level_xy))
    return tuple(sorted(set(levels)))


def _floor_available(session, tile, demand, *, preview_cache) -> bool:
    if preview_cache is None or demand is None:
        return False
    return (
        session._best_floor_key(int(tile.source_index), tile_number=int(tile.montage_index))
        is not None
    )


def can_evaluate_preview(session, tile) -> bool:
    document = getattr(session, "document", None)
    view_state = getattr(tile, "view_state", None)
    if document is None or view_state is None or getattr(view_state, "image_axes", None) is None:
        return False
    base_shape = tuple(int(size) for size in np.shape(getattr(document, "base_data", ())))
    return len(base_shape) == int(getattr(view_state, "ndim", len(base_shape)))


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
    if str(getattr(session, "lod_policy_mode", "")) == "resident":
        return False
    if not can_evaluate_reduced_preview(session, tile):
        return False
    if demand is None:
        return False
    if preview_evaluation_level(session, demand) < int(getattr(demand, "desired_level", 0) or 0):
        return False
    if preview_pipeline_commutes_for_display_lod(session, tile):
        return True
    return _shared_transform_preview_enabled()


def shared_transform_target_level(session, demand) -> int:
    """Next shared reduced-volume target level.

    Higher-quality shared work is allowed only after the first shared preview
    floor is actually presented. Desired payloads waiting for backend ack are
    not enough; otherwise a target pass can supersede the first fill before it
    reaches the screen.
    """

    desired = int(getattr(demand, "desired_level", 0) or 0)
    preview = int(preview_evaluation_level(session, demand))
    if desired <= 0:
        return preview
    planned = tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ())
    if not planned:
        return preview
    for tile in planned:
        tile_number = int(tile.montage_index)
        if tile_number in getattr(session, "rendered_tiles", {}):
            continue
        payload = presented_preview_payload(session, tile_number)
        if payload is None:
            return preview
        level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
        if level <= desired:
            return preview
    return desired


def shared_transform_candidate_tiles(
    session,
    *,
    level: int,
    tile_numbers=None,
    include_missing: bool = True,
    require_presented_preview: bool = False,
    exact_pass: bool = False,
):
    """Tiles whose shared reduced-volume target would improve presentation.

    ``include_missing`` is the first-pixel path: blank planned tiles should get
    the cheap shared preview even while finer work is also possible elsewhere.
    ``require_presented_preview`` is the quality path: only upgrade tiles whose
    current preview is acknowledged on screen, so finer shared work cannot race
    ahead of the first visible fill and leave black/pending slots behind.  The
    caller owns the plan-wide barrier; once it opens every acknowledged preview
    remains a target candidate, including a retained preview finer than a newly
    coarsened demand.  Quality is a semantic target fact, not just an LOD number.
    """

    target_level = int(level)
    allowed = None if tile_numbers is None else {int(tile) for tile in tuple(tile_numbers or ())}
    for candidate in tuple(getattr(getattr(session, "plan", None), "tiles", ()) or ()):
        tile_number = int(candidate.montage_index)
        if allowed is not None and tile_number not in allowed:
            continue
        if require_presented_preview:
            payload = presented_first_pixel_payload(session, tile_number)
            lifecycle = getattr(session, "lifecycle", None)
            if (
                payload is None
                or lifecycle is None
                or not lifecycle.target_unsettled_tiles((tile_number,))
            ):
                continue
            yield candidate
            continue
        if tile_number in getattr(session, "rendered_tiles", {}):
            continue
        payload = getattr(session, "display_tile_payloads", {}).get(tile_number)
        if payload is None:
            if include_missing:
                yield candidate
            continue
        if include_missing and payload_identity_dead(session, tile_number, payload):
            # A payload the backend can never acknowledge (its typed identity
            # cannot satisfy the tile's current target) is not coverage — it
            # is a stale wrapper that every commit silently rejects. Counting
            # it as resident starved the tile of any producer while the
            # first-pixel barrier waited on exactly this acknowledgement
            # (field stall 2026-07-16, session 148). Regenerate it fresh.
            yield candidate
            continue
        quality = str(getattr(payload, "quality", "exact"))
        payload_level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
        if quality == "preview" and payload_level > target_level:
            yield candidate
            continue
        if exact_pass and quality != "exact":
            # This pass produces quality="exact" payloads at ``level``. A
            # non-exact payload — a retained FALLBACK floor, or a preview at
            # (or finer than) the target level from a finer-demand
            # generation — can never settle the tile's exact target, however
            # correct its pixels look. Candidacy on this pass is therefore
            # the target fact, not a level comparison: churn 2026-07-16
            # parked 38 fallback-at-desired-level tiles with open targets
            # and no producer, immune to retargets (dossier
            # stale-empty-tiles-2026-07-16.md, member 5).
            lifecycle = getattr(session, "lifecycle", None)
            if lifecycle is not None and lifecycle.target_unsettled_tiles((tile_number,)):
                yield candidate


def payload_identity_dead(session, tile_number: int, payload) -> bool:
    """Whether the backend can never acknowledge this payload for its target.

    The commit path only presents payloads whose typed identity satisfies the
    tile's lifecycle target; anything else is rejected without state change.
    A payload that fails that check while nothing is presented can therefore
    never open the tile's first-pixel obligation — schedulers must treat it
    as missing, not as coverage.
    """

    tile_number = int(tile_number)
    lifecycle = getattr(session, "lifecycle", None)
    if lifecycle is None:
        return False
    if tile_number in (getattr(lifecycle, "presented_tiles", None) or frozenset()):
        return False
    record = lifecycle.peek(tile_number)
    target = None if record is None or record.target is None else record.target.identity
    if target is None:
        return False
    identity = getattr(payload, "tile_identity", None)
    if identity is None:
        identity = getattr(payload, "source_id", None)
    return not acknowledged_identity_satisfies_target(identity, target)


def presented_preview_payload(session, tile_number: int):
    """Acknowledged preview payload for scheduling higher shared quality."""

    payload = presented_first_pixel_payload(session, tile_number)
    if payload is None or str(getattr(payload, "quality", "exact")) != "preview":
        return None
    return payload


def presented_first_pixel_payload(session, tile_number: int):
    """Current physically acknowledged payload, regardless of old quality.

    A payload labelled ``exact`` under an earlier coarse demand remains valid
    first-pixel coverage after zooming in, but it is not the new target.  The
    lifecycle owns that target-settlement decision; historical payload quality
    must not suppress the shared target's only producer.
    """

    tile_number = int(tile_number)
    lifecycle = getattr(session, "lifecycle", None)
    if tile_number not in set(getattr(lifecycle, "presented_tiles", ()) or ()):
        return None
    payload = dict(
        getattr(getattr(session, "tile_presentation_state", None), "payloads", {}) or {}
    ).get(tile_number)
    if payload is None:
        return None
    backend_identities = dict(getattr(lifecycle, "backend_presented_identities", {}) or {})
    if backend_identities and backend_identities.get(tile_number) != tile_ack_identity(payload):
        return None
    return payload


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
    sample_display_axes: bool = False,
) -> tuple[np.ndarray, object]:
    axis_region_overrides = {
        int(axis): tuple(int(value) for value in tuple(values or ()))
        for axis, values in dict(axis_region_overrides or {}).items()
    }
    base_shape = tuple(int(size) for size in np.shape(document.base_data))
    image_axes = tuple(int(axis) for axis in view_state.image_axes)
    reduced_shape = list(base_shape)
    display_y_region = _display_axis_region_for_preview(
        view_state, image_axes[0], base_shape[image_axes[0]]
    )
    display_x_region = _display_axis_region_for_preview(
        view_state, image_axes[1], base_shape[image_axes[1]]
    )
    if sample_display_axes:
        preview_y_region = _sample_preview_axis_region(
            display_y_region,
            base_shape[image_axes[0]],
            max(1, int(factor_xy[1])),
        )
        preview_x_region = _sample_preview_axis_region(
            display_x_region,
            base_shape[image_axes[1]],
            max(1, int(factor_xy[0])),
        )
        reduced_shape[image_axes[0]] = _axis_region_length(
            preview_y_region,
            base_shape[image_axes[0]],
        )
        reduced_shape[image_axes[1]] = _axis_region_length(
            preview_x_region,
            base_shape[image_axes[1]],
        )
    else:
        preview_y_region = display_y_region
        preview_x_region = display_x_region
        reduced_shape[image_axes[0]] = _reduced_axis_length(
            display_y_region,
            base_shape[image_axes[0]],
            max(1, int(factor_xy[1])),
        )
        reduced_shape[image_axes[1]] = _reduced_axis_length(
            display_x_region,
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
        if axis == image_axes[0]:
            read_axes.append(preview_y_region)
            continue
        if axis == image_axes[1]:
            read_axes.append(preview_x_region)
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
    base = read_base_region(
        document.base_data,
        RegionSpec(tuple(read_axes)),
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    reduced = (
        np.asarray(base)
        if sample_display_axes
        else reduce_array_display_axes(base, image_axes, factor_xy)
    )
    preview_state = reduced_preview_view_state(
        view_state,
        np.shape(reduced),
        factor_xy=factor_xy,
        slice_remap=slice_remap,
    )
    return reduced, preview_state


def _sample_preview_axis_region(axis_region: AxisRegion, size: int, factor: int) -> AxisRegion:
    factor = max(1, int(factor))
    kind = AxisRegionKind(axis_region.kind)
    if factor <= 1:
        return axis_region
    if kind == AxisRegionKind.ALL:
        return AxisRegion(AxisRegionKind.SLICE, (0, int(size), factor))
    if kind == AxisRegionKind.SLICE:
        start, stop, step = axis_region.value
        return AxisRegion(
            AxisRegionKind.SLICE,
            (int(start), int(stop), max(1, int(step)) * factor),
        )
    if kind == AxisRegionKind.INDICES:
        return AxisRegion(
            AxisRegionKind.INDICES,
            tuple(
                int(value)
                for offset, value in enumerate(tuple(axis_region.value))
                if offset % factor == 0
            ),
        )
    return axis_region


def _reduced_axis_length(axis_region: AxisRegion, size: int, factor: int) -> int:
    return max(1, int(np.ceil(_axis_region_length(axis_region, size) / max(1, int(factor)))))


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
    if length % int(factor) == 0:
        # Tiles are normally power-of-two aligned. Reshape lets NumPy reduce
        # contiguous blocks directly; non-divisible edges retain the exact
        # reduceat/count path below.
        block_shape = list(source.shape)
        block_shape[axis : axis + 1] = [length // int(factor), int(factor)]
        reduced = source.reshape(block_shape).mean(axis=axis + 1, dtype=source.dtype)
    else:
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
    existing = getattr(display_image, "level_stats", None)
    if existing is not None:
        # A normal single-image evaluation labels its local evidence source
        # as zero. Once that value becomes a montage tile, the plan's source
        # index is authoritative; carrying the local label made one source
        # appear repeatedly sampled while another could never complete.
        return replace(
            display_image,
            level_stats=tile_level_stats_with_quality(
                existing,
                getattr(existing, "evidence_quality", 0),
                source_index=int(tile.source_index),
            ),
        )
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


def chunk_level_stats_for_pages(pages, *, source_index: int, mapping=None) -> TileLevelStats | None:
    """Aggregate worker-prepared chunk summaries into rough tile evidence.

    The materialized arrays are read only when a display-specific mapping is
    needed. Complex mapping crosses through the shared mapped-scalar function
    used by the shader mirror and CPU backend. Numeric aggregation remains
    worker-side; GUI admission sees only immutable TileLevelStats.
    """

    materialized = tuple(pages or ())
    if not materialized:
        return None
    summaries = []
    for page in materialized:
        if not isinstance(page, MaterializedLodPage):
            raise TypeError("chunk level evidence requires MaterializedLodPage values")
        if mapping is None or _page_summary_matches_mapping(page, mapping):
            summaries.append(page.summary)
            continue
        mapped = mapped_scalar(np.asarray(page.values), mapping)
        summaries.append(
            summarize_chunk(
                page.key,
                mapped,
                weights=_page_source_weights(page),
            )
        )
    aggregate = aggregate_chunk_summaries(summaries)
    if aggregate.bounds is None:
        return None
    return TileLevelStats(
        source_index=int(source_index),
        bounds=aggregate.bounds,
        sample=aggregate.representative_sample,
        refined=False,
        evidence_quality=LevelEvidenceQuality.ROUGH_PREVIEW,
    )


def _page_summary_matches_mapping(page: MaterializedLodPage, mapping) -> bool:
    if getattr(mapping, "scale", None) != ShaderScale.LINEAR:
        return False
    component = getattr(mapping, "component", None)
    values = np.asarray(page.values)
    if np.iscomplexobj(values):
        return component == ShaderComponent.ABS
    return component == ShaderComponent.REAL


def _page_source_weights(page: MaterializedLodPage) -> np.ndarray:
    y_weights = np.asarray(
        [stop - start for start, stop in page.plan.source_y_bins],
        dtype=np.float64,
    )
    x_weights = np.asarray(
        [stop - start for start, stop in page.plan.source_x_bins],
        dtype=np.float64,
    )
    return y_weights[:, np.newaxis] * x_weights[np.newaxis, :]


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


def _preview_display_histogram(rendered, source, texture_kind, histogram):
    if render_lod.display_histogram_matches_texture(histogram, source):
        return None if histogram is None else np.asarray(histogram)
    if not render_lod.texture_requires_display_histogram(source, texture_kind):
        return None if histogram is None else np.asarray(histogram)
    values = montage_refined_level_values(rendered)
    if render_lod.display_histogram_matches_texture(values, source):
        return np.asarray(values, dtype=np.float32)
    source = np.asarray(source)
    if np.iscomplexobj(source):
        return np.abs(source).astype(np.float32, copy=False)
    return np.asarray(source, dtype=np.float32)


def preview_claim_key(session, tile, *, demand, semantic_source_id, shader_display: bool):
    record = session.lifecycle.peek(int(tile.montage_index))
    if record is None:
        return None
    level = preview_evaluation_level(session, demand)
    for key in record.levels:
        if (
            getattr(key, "source_id", None) == semantic_source_id
            and int(getattr(key, "tile_id", -1)) == int(tile.source_index)
            and max(tuple(getattr(key, "level_xy", (0, 0)))) == int(level)
        ):
            return key
    return None


def _evaluate_tile_native_output_preview(
    session,
    tile,
    *,
    demand,
    semantic_source_id,
    level: int | None,
    cancellation_token,
    shader_display: bool,
    evaluation_context,
    stage_cache=None,
    stage_materializer=None,
):
    level = preview_evaluation_level(session, demand) if level is None else int(level)
    result = _evaluate_native_tile_result(
        session,
        tile,
        stage_cache=stage_cache,
        stage_materializer=stage_materializer,
        cancellation_token=cancellation_token,
        evaluation_context=evaluation_context,
    )
    _check_render_cancelled(cancellation_token)
    value = result.value
    rendered = rendered_tile_from_evaluation_result(tile, result)
    key = render_lod.page_set_key_for(
        session,
        rendered,
        demand=demand,
        level=level,
        semantic_source_id=semantic_source_id,
    )
    source = render_lod.canonical_value_source_for_rendered(
        rendered, shader_display=bool(shader_display)
    )
    source_origin_yx = render_lod.source_origin_yx_for_session(session, source)
    pages = tuple(
        render_lod.materialize_lod_page(
            source,
            source_origin_yx=source_origin_yx,
            plan=plan,
        )
        for plan in key.plans
    )
    texture_kind = (
        TexturePlaneKind.COMPLEX_RG32F
        if pages[0].key.representation == "complex_rg32f"
        else TexturePlaneKind.SCALAR_R32F
    )
    mapping = getattr(value, "shader_mapping", None)
    rough_level_stats = chunk_level_stats_for_pages(
        pages,
        source_index=int(tile.source_index),
        mapping=mapping,
    )
    _check_render_cancelled(cancellation_token)
    return (
        key,
        pages,
        None,
        mapping,
        texture_kind,
        None,
        rough_level_stats,
    )


def _native_preview_axis_region(
    axis_region: AxisRegion, output_index: int
) -> tuple[AxisRegion, int]:
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


def _axis_region_length(axis_region: AxisRegion, size: int) -> int:
    if AxisRegionKind(axis_region.kind) == AxisRegionKind.ALL:
        length = int(size)
    elif AxisRegionKind(axis_region.kind) == AxisRegionKind.SLICE:
        start, stop, step = axis_region.value
        length = len(range(*slice(int(start), int(stop), int(step)).indices(int(size))))
    elif AxisRegionKind(axis_region.kind) == AxisRegionKind.INDICES:
        length = len(tuple(axis_region.value))
    else:
        length = 1
    return max(1, int(length))


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
    """Shared transform preview is an R3 pipeline path, not a runtime fork."""

    return True


def _shared_preview_axis_override(session, tiles):
    axis = getattr(session, "montage_axis", None)
    if axis is None:
        return None
    values = tuple(
        dict.fromkeys(int(getattr(tile, "source_index", 0)) for tile in tuple(tiles or ()))
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
    "evaluate_preview_tile",
    "evaluate_shared_preview",
    "evaluate_target_tile",
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
    "shared_transform_candidate_tiles",
    "shared_transform_target_level",
    "tile_lod_states",
]
