"""Idle reusable-stage warmup for ArrayScope windows."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.scheduler import EvalPriority
from arrayscope.operations.evaluator import stage_document_key
from arrayscope.operations.chunked_stage import materialize_stage_candidate_chunked, stage_materialization_allowed_chunk_axes
from arrayscope.operations.slabs import plan_slab, request_for_image


@dataclass(frozen=True)
class StageWarmupDecision:
    decision: str
    key: object | None = None
    candidate_bytes: int | None = None
    budget_bytes: int = 0
    reason: str = ""


def schedule_stage_warmup(window, view_state) -> StageWarmupDecision:
    if _visible_work_busy(window):
        return _record(window, StageWarmupDecision("blocked_idle", reason="visible work is busy"))
    document = window.document
    request = request_for_image(view_state)
    plan = plan_slab(document, request)
    candidates = tuple(candidate for candidate in getattr(plan.region_plan, "cache_candidates", ()) if getattr(candidate, "retain", True))
    if not candidates:
        return _record(window, StageWarmupDecision("blocked_no_candidate", reason="no retained cacheable stage"))
    candidate = candidates[-1]
    budget = int(window._memory_policy().stage_cache_budget_bytes)
    estimated = candidate.estimated_nbytes
    if estimated is not None and int(estimated) > budget:
        return _record(
            window,
            StageWarmupDecision(
                "blocked_budget",
                candidate_bytes=int(estimated),
                budget_bytes=budget,
                reason="candidate exceeds stage cache budget",
            ),
        )

    materializer = window.operation_evaluator.stage_materializer
    document_key = stage_document_key(document)
    result = materializer.request_stage(document_key, candidate)
    decision = StageWarmupDecision(
        result.decision if result.decision != "attached" else "in_flight",
        key=result.key,
        candidate_bytes=None if estimated is None else int(estimated),
        budget_bytes=budget,
        reason=result.reason or result.decision,
    )
    _record(window, decision)
    if result.decision != "scheduled" or result.request is None:
        return decision

    render_generation = window._capture_render_generation()

    def evaluate(token, request=result.request, plan=plan):
        context = window._evaluation_context(ComputeLane.STAGE, token)
        return materialize_stage_candidate_chunked(
            document,
            plan.region_plan,
            request.candidate,
            stage_cache=window.operation_evaluator.stage_cache,
            document_key=request.document_key,
            cancellation_token=token,
            evaluation_context=context,
            memory_policy=context.memory_policy,
            allowed_chunk_axes=stage_materialization_allowed_chunk_axes(request.candidate.shape),
        )

    def done(value, key=result.key, generation=render_generation):
        window.operation_evaluator.stage_materializer.complete(key, value)
        if not window._is_current_render_generation(generation):
            return
        _record(window, StageWarmupDecision("completed", key=key, candidate_bytes=decision.candidate_bytes, budget_bytes=budget, reason="stage warmup complete"))

    def error(exc, key=result.key):
        window.operation_evaluator.stage_materializer.fail(key, exc)
        _record(window, StageWarmupDecision("failed", key=key, candidate_bytes=decision.candidate_bytes, budget_bytes=budget, reason=str(exc)))

    controller = getattr(window, "stage_evaluation_controller", window.visible_evaluation_controller)
    controller.start_latest(
        evaluate,
        key=("stage_warmup", result.key),
        priority=EvalPriority.PREFETCH,
        replace_group=f"stage-warmup:{hash(result.key)}",
        on_done=done,
        on_error=error,
        on_stale=lambda key=result.key: window.operation_evaluator.stage_materializer.cancel(key),
        pass_token=True,
    )
    return decision


def _visible_work_busy(window) -> bool:
    coordinator = getattr(window, "render_coordinator", None)
    return bool(
        getattr(window.visible_evaluation_controller, "is_busy", lambda: False)()
        or getattr(window.montage_tile_evaluation_controller, "is_busy", lambda: False)()
        or (coordinator is not None and getattr(coordinator, "has_pending_render", False))
    )


def _record(window, decision: StageWarmupDecision) -> StageWarmupDecision:
    window._last_stage_warmup_decision = decision
    return decision
