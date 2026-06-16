"""Stage-aware rendered montage tile prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.compute_policy import ComputeLane
from arrayscope.display.slice_engine import make_image_from_slab
from arrayscope.operations.evaluator import EvaluationResult, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.slabs import evaluate_slab_from_stage, plan_slab, request_for_image


@dataclass(frozen=True)
class MontagePrefetchDecision:
    tile_number: int | None
    source_index: int | None
    decision: str
    reason: str = ""
    stage_key: object | None = None
    tile_key: object | None = None


def schedule_near_viewport_montage_prefetch(window, session, *, max_tiles: int | None = None) -> tuple[MontagePrefetchDecision, ...]:
    if _busy(window):
        return _record(window, (MontagePrefetchDecision(None, None, "blocked_visible_busy", "visible work is busy"),))
    if not window._is_current_montage_session(session.session_id, session.key):
        return _record(window, (MontagePrefetchDecision(None, None, "stale", "session is stale"),))
    if not session.document.enabled_operations:
        return _record(window, (MontagePrefetchDecision(None, None, "blocked_no_stage", "raw montage tiles rely on visible-level commit ordering"),))
    if window.operation_evaluator._tile_cache.bytes_used > int(window.operation_evaluator._tile_cache.max_bytes * 0.8):
        return _record(window, (MontagePrefetchDecision(None, None, "blocked_budget", "tile cache is near capacity"),))
    governor = getattr(window, "resource_governor", None)
    if governor is not None:
        decision = governor.decide_montage_prefetch(stage_ready_or_in_flight=True, visible_busy=False)
        if not decision.allowed:
            return _record(window, (MontagePrefetchDecision(None, None, "blocked_governor", decision.reason),))
        max_tiles = decision.max_items
    if max_tiles is None:
        max_tiles = 2

    decisions = []
    scheduled = 0
    for tile in _candidate_tiles(session):
        if scheduled >= int(max_tiles):
            break
        tile_key = window.operation_evaluator.montage_tile_key(
            tile.view_state,
            montage_axis=session.montage_axis,
            source_index=tile.source_index,
            colormap_lut=session.colormap_lut,
            document=session.document,
        )
        if window.operation_evaluator.cached_montage_tile(
            tile.view_state,
            montage_axis=session.montage_axis,
            source_index=tile.source_index,
            colormap_lut=session.colormap_lut,
        ) is not None:
            decisions.append(MontagePrefetchDecision(int(tile.montage_index), int(tile.source_index), "hit", tile_key=tile_key))
            continue
        stage = _stage_for_tile(window, session, tile)
        if stage == "in_flight":
            decisions.append(MontagePrefetchDecision(int(tile.montage_index), int(tile.source_index), "waiting_stage_in_flight", "nearby tile waits for shared stage", tile_key=tile_key))
            continue
        if stage is None and session.document.enabled_operations:
            decisions.append(MontagePrefetchDecision(int(tile.montage_index), int(tile.source_index), "skipped_stage_missing", "would recompute expensive stage per tile", tile_key=tile_key))
            continue

        def evaluate(tile=tile, stage=stage):
            context = window._evaluation_context(ComputeLane.PREFETCH, None)
            start = perf_counter()
            if stage is not None:
                stage_value, candidate, plan = stage
                request = request_for_image(tile.view_state)
                slab = evaluate_slab_from_stage(
                    session.document,
                    request,
                    plan,
                    stage_value,
                    candidate,
                    evaluation_context=context,
                )
                display_image = make_image_from_slab(slab, request, colormap_lut=session.colormap_lut)
                return EvaluationResult(
                    value=display_image,
                    eval_ms=(perf_counter() - start) * 1000.0,
                    slab_shape=tuple(np.shape(slab)),
                    slab_nbytes=int(getattr(slab, "nbytes", 0)),
                    region_plan=plan.region_plan,
                )
            return evaluate_image_snapshot(
                session.document,
                tile.view_state,
                colormap_lut=session.colormap_lut,
                stage_cache=window.operation_evaluator.stage_cache,
                stage_document_key=stage_document_key(session.document),
                evaluation_context=context,
            )

        def done(result, tile=tile, session_id=session.session_id, session_key=session.key):
            if not window._is_current_montage_session(session_id, session_key):
                window.operation_evaluator.note_prefetch_stale()
                return
            window.operation_evaluator.store_montage_tile_result(
                tile,
                montage_axis=session.montage_axis,
                colormap_lut=session.colormap_lut,
                result=result,
            )
            window.operation_evaluator.prefetch_stored += 1

        started = window.prefetch_evaluation_controller.start_prefetch(evaluate, on_done=done, key=("montage_tile_prefetch", tile_key), memory_budget_bytes=window._memory_policy().tile_cache_budget_bytes)
        if started.scheduled:
            scheduled += 1
            window.operation_evaluator.note_prefetch_scheduled()
            decisions.append(MontagePrefetchDecision(int(tile.montage_index), int(tile.source_index), "scheduled", tile_key=tile_key))
        elif started.reason == "deduped":
            window.operation_evaluator.note_prefetch_deduped()
            decisions.append(MontagePrefetchDecision(int(tile.montage_index), int(tile.source_index), "deduped", tile_key=tile_key))
        else:
            decisions.append(MontagePrefetchDecision(int(tile.montage_index), int(tile.source_index), started.reason, tile_key=tile_key))

    if not decisions:
        decisions.append(MontagePrefetchDecision(None, None, "blocked_no_tile", "no nearby uncached tile"))
    return _record(window, tuple(decisions))


def _candidate_tiles(session):
    excluded = set(int(tile.montage_index) for tile in getattr(session, "visible_tiles", ()))
    excluded.update(int(index) for index in getattr(session, "rendered_tiles", ()))
    excluded.update(int(index) for index in getattr(session, "loading_tiles", ()))
    excluded.update(int(index) for index in getattr(session, "skipped_tiles", ()))
    for tile in tuple(session.plan.tiles):
        if int(tile.montage_index) not in excluded:
            yield tile


def _stage_for_tile(window, session, tile):
    request = request_for_image(tile.view_state)
    plan = plan_slab(session.document, request)
    retained = tuple(candidate for candidate in getattr(plan.region_plan, "cache_candidates", ()) if getattr(candidate, "retain", True))
    if not retained:
        return None
    candidate = retained[-1]
    key = window.operation_evaluator.stage_materializer.key_for_candidate(stage_document_key(session.document), candidate)
    cache = window.operation_evaluator.stage_cache
    value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
    if value is None:
        in_flight = getattr(window.operation_evaluator.stage_materializer, "_in_flight", {})
        if key in in_flight:
            return "in_flight"
        return None
    return value, candidate, plan


def _busy(window) -> bool:
    return bool(
        window.visible_evaluation_controller.is_busy()
        or window.montage_tile_evaluation_controller.is_busy()
        or window.stage_evaluation_controller.is_busy()
    )


def _record(window, decisions: tuple[MontagePrefetchDecision, ...]) -> tuple[MontagePrefetchDecision, ...]:
    window._last_montage_prefetch_decisions = decisions
    return decisions
