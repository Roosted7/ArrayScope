"""Stage-aware rendered montage tile prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.model.tile_priority import MontageTilePriorityQueue
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.scheduler import FrameTarget
from arrayscope.kernel import Lane as WorkLane, WorkItem
from arrayscope.display.slice_engine import make_image_from_slab, make_shader_image_from_slab
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
    if _interaction_active(window):
        # User interaction owns the GUI thread and the worker lanes.  The
        # walk resumes from the next flush/completion invitation; speculation
        # must never add a millisecond to a scrub or drag.
        return _record(window, (MontagePrefetchDecision(None, None, "blocked_interaction", "viewport interaction active"),))
    if _busy(window, session):
        return _record(window, (MontagePrefetchDecision(None, None, "blocked_visible_busy", "visible work is busy"),))
    if not window._montage_session_is_current(session):
        return _record(window, (MontagePrefetchDecision(None, None, "stale", "session is stale"),))
    if not session.document.enabled_operations:
        return _record(window, (MontagePrefetchDecision(None, None, "blocked_no_stage", "raw montage tiles rely on visible-level commit ordering"),))
    preview_walk_only = False
    if window.win.operation_evaluator._display_cache.bytes_used > int(window.win.operation_evaluator._display_cache.max_bytes * 0.8):
        # Background preview walk (ADR 0050): a full display cache used to
        # stop speculation for the rest of the stack.  When the retained
        # preview level is active, keep walking never-visited indices in
        # preview-only mode — evaluate, pin the tiny preview plane, discard
        # the native result — so every index floors instantly forever while
        # the display cache stays untouched.
        if _preview_cache_active(session):
            preview_walk_only = True
        else:
            return _record(window, (MontagePrefetchDecision(None, None, "blocked_budget", "display cache is near capacity"),))
    governor = getattr(window.win, "resource_governor", None)
    if governor is not None:
        decision = governor.decide_montage_prefetch(stage_ready_or_in_flight=True, visible_busy=False)
        if not decision.allowed:
            return _record(window, (MontagePrefetchDecision(None, None, "blocked_governor", decision.reason),))
        max_tiles = decision.max_items
    if max_tiles is None:
        max_tiles = 2

    decisions = []
    scheduled = 0
    shader_display = bool(getattr(session, "shader_display", False))
    for tile in _candidate_tiles(session):
        if scheduled >= int(max_tiles):
            break
        if preview_walk_only and _preview_resident(session, tile):
            # The walk's only purpose here is the pinned preview; skip
            # indices that already have one instead of re-evaluating them.
            continue
        tile_key = window.win.operation_evaluator.montage_tile_key(
            tile.view_state,
            montage_axis=session.montage_axis,
            source_index=tile.source_index,
            colormap_lut=session.colormap_lut,
            document=session.document,
            shader_display=shader_display,
        )
        if window.win.operation_evaluator.cached_montage_tile(
            tile.view_state,
            montage_axis=session.montage_axis,
            source_index=tile.source_index,
            colormap_lut=session.colormap_lut,
            shader_display=shader_display,
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
            context = window.win._evaluation_context(ComputeLane.PREFETCH, None)
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
                if shader_display:
                    display_image = make_shader_image_from_slab(
                        slab,
                        request,
                        colormap_lut=session.colormap_lut,
                        provisional_histogram=True,
                    )
                else:
                    display_image = make_image_from_slab(slab, request, colormap_lut=session.colormap_lut)
                result = EvaluationResult(
                    value=display_image,
                    eval_ms=(perf_counter() - start) * 1000.0,
                    slab_shape=tuple(np.shape(slab)),
                    slab_nbytes=int(getattr(slab, "nbytes", 0)),
                    region_plan=plan.region_plan,
                )
            else:
                result = evaluate_image_snapshot(
                    session.document,
                    tile.view_state,
                    colormap_lut=session.colormap_lut,
                    stage_cache=window.win.operation_evaluator.stage_cache,
                    stage_document_key=stage_document_key(session.document),
                    evaluation_context=context,
                    shader_display=shader_display,
                    provisional_histogram=bool(shader_display),
                )
            # Worker-side, like the visible-path ingest admissions (ADR 0041
            # gate 1 forbids commit-callback reduction): every walked index
            # leaves its pinned preview plane behind, so it re-presents
            # instantly through the floor forever — even if the native
            # result below is discarded or evicted.
            _admit_walk_preview(session, tile, result)
            return result

        def done(result, tile=tile, session_id=session.session_id, session_key=session.key, preview_walk_only=preview_walk_only):
            if not window._is_current_montage_session(session_id, session_key):
                window.win.operation_evaluator.note_prefetch_stale()
                return
            if not preview_walk_only:
                rendered = window.win.operation_evaluator.store_montage_tile_result(
                    tile,
                    montage_axis=session.montage_axis,
                    colormap_lut=session.colormap_lut,
                    result=result,
                    shader_display=shader_display,
                )
                # In preview-only mode the admission already happened
                # worker-side; storing the native result would churn the
                # display cache this mode exists to protect.
                window.win.operation_evaluator.prefetch_stored += 1
                _warm_prefetched_pyqtgraph_tile(window, session, tile, rendered, tile_key)
            # Walk continuation (ADR 0050 background preview walk): flush
            # paths only invite prefetch while something is happening, so at
            # true idle the walk stalled after one batch.  Each completion
            # invites the next batch — deferred and coalesced, because the
            # scheduling pass (candidate scan + stage probes) is synchronous
            # GUI work that must never ride on the completion callback while
            # the user interacts.  Ends itself on no-candidates / governor /
            # busy; any natural flush invitation re-breaks those states.
            _invite_walk_continuation(window)

        budget_bytes = int(window._memory_policy().display_cache_budget_bytes)
        # Admission control needs the tile's actual footprint.  The display
        # budget (gigabytes) here meant a single walk item filled the whole
        # SPECULATIVE_RESIDENCY lane, so the visible tiles' demanded-level
        # materializations were admission-blocked for the entire session —
        # the stale-LOD symptom (0 materializations completed, 2.6k blocked).
        tile_h, tile_w = (int(value) for value in tuple(session.plan.tile_shape))
        estimated_tile_bytes = max(1, tile_h * tile_w * 16)  # complex128 worst case
        started = window.win.prefetch_evaluation_controller.start_prefetch(
            evaluate,
            on_done=done,
            key=("montage_tile_prefetch", tile_key),
            memory_budget_bytes=budget_bytes,
            work_item=WorkItem(
                key=("montage_tile_prefetch", tile_key),
                lane=WorkLane.SPECULATIVE_RESIDENCY,
                frame_target=FrameTarget(
                    semantic_key=tile_key,
                    viewport_key=("montage-near", int(tile.montage_index)),
                    presentation_key=("prefetch",),
                    quality="retained",
                ),
                supersession_key=("montage-tile-prefetch", session.key, int(tile.montage_index)),
                supersession_value=tile_key,
                estimated_bytes=estimated_tile_bytes,
                expected_value=1.0,
                reusable_output=True,
            ),
        )
        if started.scheduled:
            scheduled += 1
            window.win.operation_evaluator.note_prefetch_scheduled()
            decisions.append(
                MontagePrefetchDecision(
                    int(tile.montage_index),
                    int(tile.source_index),
                    "scheduled_preview_walk" if preview_walk_only else "scheduled",
                    tile_key=tile_key,
                )
            )
        elif started.reason == "deduped":
            window.win.operation_evaluator.note_prefetch_deduped()
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
    candidates = tuple(tile for tile in tuple(session.plan.tiles) if int(tile.montage_index) not in excluded)
    if not candidates:
        return ()
    # Plan order is row-major, which would prefetch from the plan's corner;
    # speculate on the tiles nearest the viewport/focus instead.
    context_builder = getattr(session, "_tile_priority_context", None)
    if context_builder is None:
        return candidates
    return MontageTilePriorityQueue(candidates, context=context_builder()).ordered_tiles()


def _stage_for_tile(window, session, tile):
    request = request_for_image(tile.view_state)
    plan = plan_slab(session.document, request)
    retained = tuple(candidate for candidate in getattr(plan.region_plan, "cache_candidates", ()) if getattr(candidate, "retain", True))
    if not retained:
        return None
    candidate = retained[-1]
    key = window.win.operation_evaluator.stage_materializer.key_for_candidate(stage_document_key(session.document), candidate)
    cache = window.win.operation_evaluator.stage_cache
    value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
    if value is None:
        in_flight = getattr(window.win.operation_evaluator.stage_materializer, "_in_flight", {})
        if key in in_flight:
            return "in_flight"
        return None
    return value, candidate, plan


def _interaction_active(window) -> bool:
    try:
        from arrayscope.window.frame_renderer import _viewport_interaction_active

        return bool(_viewport_interaction_active(window))
    except Exception:
        return False


def _busy(window, session=None) -> bool:
    if session is not None and (
        getattr(session, "pending_tiles", None)
        or getattr(session, "loading_tiles", None)
        or getattr(session, "active_tile_requests", None)
        or getattr(session, "dirty_payloads", None)
        or getattr(session, "pending_removals", None)
        # Demanded-but-missing LOD levels of *visible* tiles outrank the
        # walk for lane capacity: speculation waits until they are drained.
        or getattr(session, "pending_rung_materializations", None)
        or session.stage_fan_in.active_requests
    ):
        return True
    return bool(
        window.win.visible_evaluation_controller.is_busy()
        or window.win.montage_tile_evaluation_controller.is_busy()
        or window.win.stage_evaluation_controller.is_busy()
    )


def _warm_prefetched_pyqtgraph_tile(window, session, tile, rendered, tile_key) -> None:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    if not (
        bool(getattr(capabilities, "persistent_tile_residency", False))
        and str(getattr(capabilities, "tile_residency_kind", "none") or "none") == "cpu_item"
    ):
        return
    warm = getattr(getattr(window.win, "img_view", None), "warmTiledResidency", None)
    if not callable(warm):
        return
    if not window._montage_session_is_current(session):
        return
    tile_number = int(getattr(tile, "montage_index", -1))
    if tile_number < 0:
        return
    try:
        source_ids = (
            window._montage_tile_source_ids(session)
            if hasattr(window, "_montage_tile_source_ids")
            else {tile_number: tile_key}
        )
        payload = session._ensure_display_tile_payload(
            tile_number,
            rendered,
            source_ids,
            lod_factor=int(session._selected_lod_factor()),
        )
        geometry = DisplayGeometry(
            view_state=session.view_state,
            display_shape=tuple(session.plan.display_shape),
            montage=session.plan.geometry,
            montage_tile_states=session.ensure_tile_states(),
        )
        try:
            levels = tuple(float(value) for value in window.win.img_view.getLevels())
        except Exception:
            levels = tuple(float(value) for value in getattr(session, "user_levels_override", None) or (0.0, 1.0))
        warm(
            payloads={tile_number: payload},
            geometry=geometry,
            levels=(float(levels[0]), float(levels[1])),
            rgb_already_windowed=not bool(getattr(session, "shader_display", False)),
            tile_residency_budget_bytes=0,
            frame_plan=getattr(session, "frame_plan", None),
        )
    except Exception:
        return


def _invite_walk_continuation(window, *, delay_ms: int = 80) -> None:
    """Defer-and-coalesce the next walk batch off the completion callback.

    One pending invitation at a time; it re-resolves the current session at
    fire time so a scrub that replaced the session never revives a stale
    walk.  The delay yields the GUI thread to any queued input events first.
    """

    if getattr(window, "_montage_walk_invite_pending", False):
        return
    window._montage_walk_invite_pending = True

    def fire():
        window._montage_walk_invite_pending = False
        if _interaction_active(window):
            return
        current = getattr(window, "_montage_session", None)
        if current is None or not window._montage_session_is_current(current):
            return
        schedule_near_viewport_montage_prefetch(window, current)

    try:
        from pyqtgraph.Qt import QtCore

        # Receiver-scoped: the invitation dies with the window instead of
        # firing into a torn-down renderer (architecture guard).
        QtCore.QTimer.singleShot(int(delay_ms), window.win, fire)
    except Exception:
        window._montage_walk_invite_pending = False


def _preview_cache_active(session) -> bool:
    return (
        getattr(session, "pyramid_cache", None) is not None
        and int(getattr(session, "lod_preview_level", 0) or 0) > 0
    )


def _preview_resident(session, tile) -> bool:
    """True when the pinned preview cache already holds this tile's plane."""

    preview = getattr(session, "pyramid_cache", None)
    level = int(getattr(session, "lod_preview_level", 0) or 0)
    if preview is None or level <= 0:
        return False
    from arrayscope.display.pyramid import PyramidLevelKey
    from arrayscope.render.lod import floor_component_tags

    semantic_id = session.tile_semantic_source_id(int(tile.source_index))
    for component in floor_component_tags(session):
        key = PyramidLevelKey(
            source_id=semantic_id,
            tile_id=int(tile.source_index),
            component=component,
            level_xy=(level, level),
        )
        if preview.peek(key) is not None:
            return True
    return False


def _admit_walk_preview(session, tile, result) -> bool:
    """Pin the retained preview level for a prefetched tile, worker-side.

    ADR 0050 background preview walk: prefetch used to warm only the display
    cache, which evicts; the pinned preview cache does not.  Admission is
    singleflight-guarded, so a concurrent visible evaluation of the same
    index never duplicates the reduction.
    """

    preview = getattr(session, "pyramid_cache", None)
    level = int(getattr(session, "lod_preview_level", 0) or 0)
    if preview is None or level <= 0:
        return False
    try:
        from arrayscope.window.frame_renderer import _rendered_tile_from_evaluation_result
        from arrayscope.render.lod import admit_retained_preview_level

        rendered = _rendered_tile_from_evaluation_result(tile, result)
        return bool(
            admit_retained_preview_level(
                preview,
                rendered,
                semantic_source_id=session.tile_semantic_source_id(int(tile.source_index)),
                preview_level=level,
            )
        )
    except Exception:
        # Speculative polish must never fail the prefetch result.
        return False


def _record(window, decisions: tuple[MontagePrefetchDecision, ...]) -> tuple[MontagePrefetchDecision, ...]:
    window._last_montage_prefetch_decisions = decisions
    return decisions
