"""Stage-aware rendered montage tile prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.app.errors import handle_ui_exception
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.frame_targets import FrameTarget
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.model.tile_priority import MontageTilePriorityQueue
from arrayscope.display.pyramid import (
    LodPageCache,
    LodPagePlan,
    MaterializedLodPage,
    materialize_source_grid_pages,
)
from arrayscope.display.slice_engine import make_image_from_slab, make_shader_image_from_slab
from arrayscope.kernel import Lane as WorkLane
from arrayscope.kernel import Priority, WorkItem
from arrayscope.operations.evaluator import (
    EvaluationResult,
    evaluate_image_snapshot,
    stage_document_key,
)
from arrayscope.operations.slabs import evaluate_slab_from_stage, plan_slab, request_for_image
from arrayscope.render.effects import rendered_tile_from_evaluation_result
from arrayscope.render.lod import (
    LodPageSetKey,
    _page_set_exact,
    canonical_value_source_for_rendered,
    page_set_key_for_rendered,
    page_set_key_for_tile,
)
from arrayscope.window.frame_effects import interactive_active


@dataclass(frozen=True)
class MontagePrefetchDecision:
    tile_number: int | None
    source_index: int | None
    decision: str
    reason: str = ""
    stage_key: object | None = None
    tile_key: object | None = None


@dataclass(frozen=True)
class _RetainedPreviewClaim:
    """GUI-owned singleflight claim carried across one prefetch worker."""

    key: LodPageSetKey
    claimed_plans: tuple[LodPagePlan, ...]
    owner: object
    cache: LodPageCache
    demand: object
    level: int


@dataclass(frozen=True)
class _MontagePrefetchWorkerResult:
    """Immutable worker output; cache/lifecycle admission stays GUI-owned."""

    evaluation: EvaluationResult
    preview_pages: tuple[MaterializedLodPage, ...] = ()


def schedule_near_viewport_montage_prefetch(
    window, session, *, max_tiles: int | None = None
) -> tuple[MontagePrefetchDecision, ...]:
    if _interaction_active(window):
        # User interaction owns the GUI thread and the worker lanes.  The
        # walk resumes from the next flush/completion invitation; speculation
        # must never add a millisecond to a scrub or drag.
        return _record(
            window,
            (
                MontagePrefetchDecision(
                    None, None, "blocked_interaction", "viewport interaction active"
                ),
            ),
        )
    if _busy(window, session):
        return _record(
            window,
            (MontagePrefetchDecision(None, None, "blocked_visible_busy", "visible work is busy"),),
        )
    if not window._frame_session_is_current(session):
        return _record(window, (MontagePrefetchDecision(None, None, "stale", "session is stale"),))
    if not session.document.enabled_operations:
        return _record(
            window,
            (
                MontagePrefetchDecision(
                    None,
                    None,
                    "blocked_no_stage",
                    "raw montage tiles rely on visible-level commit ordering",
                ),
            ),
        )
    preview_walk_only = False
    if window.win.operation_evaluator._display_cache.bytes_used > int(
        window.win.operation_evaluator._display_cache.max_bytes * 0.8
    ):
        # Background preview walk (ADR 0050): a full display cache used to
        # stop speculation for the rest of the stack.  When the retained
        # preview level is active, keep walking never-visited indices in
        # preview-only mode — evaluate, pin the tiny preview plane, discard
        # the native result — so every index floors instantly forever while
        # the display cache stays untouched.
        if _preview_cache_active(session):
            preview_walk_only = True
        else:
            return _record(
                window,
                (
                    MontagePrefetchDecision(
                        None, None, "blocked_budget", "display cache is near capacity"
                    ),
                ),
            )
    if max_tiles is None:
        max_tiles = _owner_prefetch_batch_limit(window)
    if _owner_memory_pressure_blocks_prefetch(window):
        return _record(
            window,
            (MontagePrefetchDecision(None, None, "blocked_memory_pressure", "memory pressure"),),
        )

    decisions = []
    scheduled = 0
    shader_display = bool(getattr(session, "shader_display", False))
    canonical_orientation = bool(getattr(session, "canonical_orientation", False))
    direction = _montage_prefetch_direction(window)
    candidates = (
        _candidate_tiles(session, direction=direction) if direction else _candidate_tiles(session)
    )
    for tile in candidates:
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
        if (
            window.win.operation_evaluator.cached_montage_tile(
                tile.view_state,
                montage_axis=session.montage_axis,
                source_index=tile.source_index,
                colormap_lut=session.colormap_lut,
                shader_display=shader_display,
            )
            is not None
        ):
            decisions.append(
                MontagePrefetchDecision(
                    int(tile.montage_index), int(tile.source_index), "hit", tile_key=tile_key
                )
            )
            continue
        stage = _stage_for_tile(window, session, tile)
        if stage == "in_flight":
            decisions.append(
                MontagePrefetchDecision(
                    int(tile.montage_index),
                    int(tile.source_index),
                    "waiting_stage_in_flight",
                    "nearby tile waits for shared stage",
                    tile_key=tile_key,
                )
            )
            continue
        if stage is None and session.document.enabled_operations:
            decisions.append(
                MontagePrefetchDecision(
                    int(tile.montage_index),
                    int(tile.source_index),
                    "skipped_stage_missing",
                    "would recompute expensive stage per tile",
                    tile_key=tile_key,
                )
            )
            continue
        preview_claim = _claim_walk_preview(session, tile)

        def evaluate(tile=tile, stage=stage, preview_claim=preview_claim):
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
                        canonical_orientation=canonical_orientation,
                    )
                else:
                    display_image = make_image_from_slab(
                        slab,
                        request,
                        colormap_lut=session.colormap_lut,
                        canonical_orientation=canonical_orientation,
                    )
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
                    canonical_orientation=canonical_orientation,
                )
            pages = _materialize_walk_preview(
                session,
                tile,
                result,
                preview_claim,
                shader_display=shader_display,
            )
            return _MontagePrefetchWorkerResult(result, pages)

        def done(
            worker_result,
            tile=tile,
            session_id=session.session_id,
            session_key=session.key,
            preview_walk_only=preview_walk_only,
            preview_claim=preview_claim,
        ):
            if not window._is_current_frame_session(session_id, session_key):
                _release_walk_preview_claim(session, preview_claim)
                window.win.operation_evaluator.note_prefetch_stale()
                if preview_claim is not None:
                    _wake_current_after_walk_preview_terminal(window)
                return
            try:
                _admit_walk_preview_result(
                    session,
                    tile,
                    preview_claim,
                    worker_result.preview_pages,
                )
                if not preview_walk_only:
                    rendered = window.win.operation_evaluator.store_montage_tile_result(
                        tile,
                        montage_axis=session.montage_axis,
                        colormap_lut=session.colormap_lut,
                        result=worker_result.evaluation,
                        shader_display=shader_display,
                    )
                    # Preview-only mode deliberately avoids display-cache churn;
                    # its checked pages were admitted above on the GUI thread.
                    window.win.operation_evaluator.prefetch_stored += 1
                    _warm_prefetched_tiled_residency(window, session, tile, rendered)
            finally:
                _release_walk_preview_claim(session, preview_claim)
                if preview_claim is not None:
                    _wake_walk_preview_admission(window, session)
                # Walk continuation (ADR 0050 background preview walk): flush
                # paths only invite prefetch while something is happening, so at
                # true idle the walk stalled after one batch.  Each completion
                # invites the next batch — deferred and coalesced.
                _invite_walk_continuation(window)

        def dropped(preview_claim=preview_claim):
            _release_walk_preview_claim(session, preview_claim)
            if preview_claim is not None:
                _wake_current_after_walk_preview_terminal(window)

        def failed(exc, preview_claim=preview_claim):
            dropped(preview_claim)
            handle_ui_exception("montage prefetch", exc)

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
            on_error=failed,
            on_stale=dropped,
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
            decisions.append(
                MontagePrefetchDecision(
                    int(tile.montage_index), int(tile.source_index), "deduped", tile_key=tile_key
                )
            )
        else:
            _release_walk_preview_claim(session, preview_claim)
            decisions.append(
                MontagePrefetchDecision(
                    int(tile.montage_index),
                    int(tile.source_index),
                    started.reason,
                    tile_key=tile_key,
                )
            )

    if not decisions:
        decisions.append(
            MontagePrefetchDecision(None, None, "blocked_no_tile", "no nearby uncached tile")
        )
    return _record(window, tuple(decisions))


def _candidate_tiles(session, *, direction: int = 0):
    excluded = {int(tile.montage_index) for tile in getattr(session, "visible_tiles", ())}
    excluded.update(int(index) for index in getattr(session, "rendered_tiles", ()))
    excluded.update(int(index) for index in getattr(session, "loading_tiles", ()))
    excluded.update(int(index) for index in getattr(session, "skipped_tiles", ()))
    candidates = tuple(
        tile for tile in tuple(session.plan.tiles) if int(tile.montage_index) not in excluded
    )
    if not candidates:
        return ()
    # Plan order is row-major, which would prefetch from the plan's corner;
    # speculate on the tiles nearest the viewport/focus instead.
    context_builder = getattr(session, "tile_priority_context", None)
    if not callable(context_builder):
        raise RuntimeError("live frame session has no tile-priority owner")
    context = context_builder()
    ordered = MontageTilePriorityQueue(candidates, context=context).ordered_tiles()
    direction = 1 if int(direction) > 0 else (-1 if int(direction) < 0 else 0)
    if direction == 0:
        return ordered
    visible_numbers = {
        int(visible.montage_index) for visible in getattr(session, "visible_tiles", ())
    }
    visible_sources = tuple(
        int(tile.source_index)
        for tile in tuple(session.plan.tiles)
        if int(tile.montage_index) in visible_numbers
    )
    if not visible_sources:
        return ordered
    boundary = max(visible_sources) if direction > 0 else min(visible_sources)

    def direction_rank(tile) -> tuple[int, int]:
        tile_number = int(tile.montage_index)
        band = (
            0
            if tile_number in context.priority_tiles or tile_number in context.visible_tiles
            else (1 if tile_number in context.near_tiles else 2)
        )
        source_index = int(tile.source_index)
        ahead = source_index > boundary if direction > 0 else source_index < boundary
        return band, (0 if ahead else 1)

    # Python's stable sort preserves the canonical viewport-distance order
    # inside each band's ahead and reversal-guard partitions. A WAITING tile
    # can therefore never overtake a NEAR tile merely because it is ahead.
    return tuple(sorted(ordered, key=direction_rank))


def _montage_prefetch_direction(window) -> int:
    momentum = getattr(window, "_montage_prefetch_momentum", None)
    if momentum is None:
        return 0
    return int(momentum.plan().direction)


def _stage_for_tile(window, session, tile):
    request = request_for_image(tile.view_state)
    plan = plan_slab(session.document, request)
    retained = tuple(
        candidate
        for candidate in getattr(plan.region_plan, "cache_candidates", ())
        if getattr(candidate, "retain", True)
    )
    if not retained:
        return None
    candidate = retained[-1]
    key = window.win.operation_evaluator.stage_materializer.key_for_candidate(
        stage_document_key(session.document), candidate
    )
    cache = window.win.operation_evaluator.stage_cache
    value = cache.get_containing(key) if hasattr(cache, "get_containing") else cache.get(key)
    if value is None:
        in_flight = getattr(window.win.operation_evaluator.stage_materializer, "_in_flight", {})
        if key in in_flight:
            return "in_flight"
        return None
    return value, candidate, plan


def _interaction_active(window) -> bool:
    return bool(interactive_active(window))


def _busy(window, session=None) -> bool:
    if session is not None and (
        not session.required_target_settled()
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


def _owner_prefetch_batch_limit(window) -> int:
    governor = getattr(window.win, "resource_governor", None)
    profile = getattr(governor, "profile", "balanced")
    profile_name = str(getattr(profile, "value", profile)).lower()
    if profile_name == "aggressive":
        return 4
    if profile_name == "conservative":
        return 1
    return 2


def _owner_memory_pressure_blocks_prefetch(window) -> bool:
    governor = getattr(window.win, "resource_governor", None)
    diagnostics = None if governor is None else governor.diagnostics()
    pressure = getattr(getattr(diagnostics, "pressure", None), "memory_pressure", None)
    pressure_name = str(getattr(pressure, "value", pressure)).lower()
    return pressure_name in {"elevated", "high"}


def _warm_prefetched_tiled_residency(window, session, tile, rendered) -> None:
    capabilities = image_view_backend_capabilities(getattr(window.win, "img_view", None))
    if not (
        bool(getattr(capabilities, "persistent_tile_residency", False))
        and str(getattr(capabilities, "tile_residency_kind", "none") or "none")
        in {"cpu_item", "gpu_atlas"}
    ):
        return
    warm = getattr(getattr(window.win, "img_view", None), "warmTiledResidency", None)
    if not callable(warm):
        return
    if not window._frame_session_is_current(session):
        return
    tile_number = int(getattr(tile, "montage_index", -1))
    if tile_number < 0:
        return
    source_ids = window._montage_tile_source_ids(session)
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
    except (AttributeError, TypeError, ValueError):
        levels = tuple(
            float(value) for value in getattr(session, "user_levels_override", None) or (0.0, 1.0)
        )
    warm(
        payloads={tile_number: payload},
        geometry=geometry,
        levels=(float(levels[0]), float(levels[1])),
        rgb_already_windowed=not bool(getattr(session, "shader_display", False)),
        tile_residency_budget_bytes=int(getattr(session, "tile_residency_budget_bytes", 0) or 0),
        frame_plan=getattr(session, "frame_plan", None),
    )


def _invite_walk_continuation(window) -> None:
    """Ask the kernel to admit the next speculative walk batch.

    One pending invitation at a time; it re-resolves the current session when
    the speculative lane grants the callback, so a scrub that replaced the
    session never revives stale side work.
    """

    if getattr(window, "_montage_walk_invite_pending", False):
        return
    window._montage_walk_invite_pending = True
    session = getattr(window, "_frame_session", None)
    if session is None:
        window._montage_walk_invite_pending = False
        return
    generation = (
        getattr(session, "key", None),
        int(getattr(session, "session_id", 0) or 0),
        int(getattr(session, "viewport_revision", 0) or 0),
    )

    def fire(_value=None, generation=generation):
        window._montage_walk_invite_pending = False
        if _interaction_active(window):
            return
        current = getattr(window, "_frame_session", None)
        if current is None or not window._frame_session_is_current(current):
            return
        current_generation = (
            getattr(current, "key", None),
            int(getattr(current, "session_id", 0) or 0),
            int(getattr(current, "viewport_revision", 0) or 0),
        )
        if current_generation != generation:
            return
        schedule_near_viewport_montage_prefetch(window, current)

    kernel = getattr(window.win, "kernel", None)
    if kernel is None:
        window._montage_walk_invite_pending = False
        return
    handle = kernel.submit_speculative_batch(
        kind="montage-prefetch-walk",
        scope=f"montage:{getattr(session, 'key', None)!r}:prefetch",
        generation=generation,
        key=("montage-prefetch-walk", generation),
        fn=lambda: True,
        on_done=fire,
        on_stale=lambda: setattr(window, "_montage_walk_invite_pending", False),
        lane=WorkLane.SPECULATIVE_RESIDENCY,
        priority=Priority.PREFETCH,
        max_items=1,
    )
    if handle is None:
        window._montage_walk_invite_pending = False


def _preview_cache_active(session) -> bool:
    return (
        getattr(session, "lod_page_cache", None) is not None
        and int(getattr(session, "lod_preview_level", 0) or 0) > 0
    )


def _preview_resident(session, tile) -> bool:
    """True when the pinned preview cache already holds this tile's plane."""

    preview = getattr(session, "lod_page_cache", None)
    level = int(getattr(session, "lod_preview_level", 0) or 0)
    if preview is None or level <= 0:
        return False
    semantic_id = session.tile_semantic_source_id(int(tile.source_index))
    rec = session.lifecycle.peek(int(tile.montage_index))
    if rec is None:
        return False
    for key, entry in rec.levels.items():
        if (
            getattr(key, "source_id", None) == semantic_id
            and int(getattr(key, "tile_id", -1)) == int(tile.source_index)
            and max(tuple(getattr(key, "level_xy", (0, 0)))) == level
            and entry.phase.value == "resident"
            and _page_set_exact(preview, key)
        ):
            return True
    return False


def _claim_walk_preview(session, tile) -> _RetainedPreviewClaim | None:
    """Claim missing retained-preview pages on the GUI scheduling boundary."""

    preview = getattr(session, "lod_page_cache", None)
    level = int(getattr(session, "lod_preview_level", 0) or 0)
    demand = getattr(getattr(session, "lod_policy_decision", None), "demand", None)
    if preview is None or level <= 0 or demand is None:
        return None
    key = page_set_key_for_tile(session, tile, demand=demand, level=level)
    if _page_set_exact(preview, key):
        return None
    owner = (
        "montage-prefetch-preview",
        id(session),
        int(getattr(session, "session_id", 0) or 0),
        int(tile.montage_index),
        key,
    )
    claimed = tuple(preview.claim_plans(key.plans, owner))
    if not claimed:
        return None
    return _RetainedPreviewClaim(
        key=key,
        claimed_plans=claimed,
        owner=owner,
        cache=preview,
        demand=demand,
        level=level,
    )


def _materialize_walk_preview(
    session,
    tile,
    result,
    claim: _RetainedPreviewClaim | None,
    *,
    shader_display: bool,
) -> tuple[MaterializedLodPage, ...]:
    """Pure worker step: return checked pages without mutating live owners."""

    if claim is None:
        return ()
    rendered = rendered_tile_from_evaluation_result(tile, result)
    actual_key = page_set_key_for_rendered(
        rendered,
        demand=claim.demand,
        level=claim.level,
        semantic_source_id=session.tile_semantic_source_id(int(tile.source_index)),
        shader_display=bool(shader_display),
    )
    if actual_key != claim.key:
        raise ValueError("prefetch result disagrees with its precomputed canonical page route")
    source = canonical_value_source_for_rendered(
        rendered,
        shader_display=bool(shader_display),
    )
    return materialize_source_grid_pages(
        source,
        source_origin_yx=(0, 0),
        plans=claim.claimed_plans,
    )


def _admit_walk_preview_result(
    session,
    tile,
    claim: _RetainedPreviewClaim | None,
    pages: tuple[MaterializedLodPage, ...],
) -> bool:
    """Admit checked worker pages and lifecycle state on the GUI thread."""

    if claim is None:
        if pages:
            raise ValueError("prefetch returned preview pages without a page claim")
        return False
    supplied = tuple(pages)
    if tuple(page.key for page in supplied) != tuple(plan.key for plan in claim.claimed_plans):
        raise ValueError("prefetch returned pages outside its claimed canonical plans")
    preview = claim.cache
    if getattr(session, "lod_page_cache", None) is not preview:
        return False
    for page in supplied:
        preview.admit_as(page.key, page, owner=claim.owner)
    exact_pages = preview.exact_pages(claim.key.plans)
    if exact_pages is None:
        return False
    return bool(
        session.admit_preview_plane(
            int(tile.montage_index),
            claim.key,
            exact_pages,
            quality="preview",
        )
    )


def _release_walk_preview_claim(
    session,
    claim: _RetainedPreviewClaim | None,
) -> tuple:
    if claim is None:
        return ()
    return tuple(claim.cache.release_owner_claims(claim.owner))


def _wake_walk_preview_admission(window, session) -> None:
    """Wake the existing semantic and physical presentation owners once."""

    window.request_montage_replan(session)
    effects = getattr(getattr(session, "pipeline", None), "effects", None)
    request_presentation = getattr(effects, "request_presentation", None)
    if callable(request_presentation):
        request_presentation()


def _wake_current_after_walk_preview_terminal(window) -> None:
    """Replan the live session after stale work releases shared page claims."""

    current = getattr(window, "_frame_session", None)
    if current is None or not window._frame_session_is_current(current):
        return
    _wake_walk_preview_admission(window, current)


def _record(
    window, decisions: tuple[MontagePrefetchDecision, ...]
) -> tuple[MontagePrefetchDecision, ...]:
    window._last_montage_prefetch_decisions = decisions
    return decisions
