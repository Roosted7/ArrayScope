"""Runtime diagnostics snapshot construction for ArrayScope windows."""

from __future__ import annotations

from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.runtime_diagnostics import (
    CanvasPreserveRuntimeDiagnostics,
    MontageRuntimeDiagnostics,
    MontageTimingDiagnostics,
    RenderCoalescerDiagnostics,
    RenderRuntimeDiagnostics,
    RenderTimingDiagnostics,
    WindowRuntimeDiagnostics,
)
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.operations import fft_backend
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.regions import region_text


def collect_runtime_diagnostics_snapshot(window) -> WindowRuntimeDiagnostics:
    policy = window._refresh_memory_policy(
        active_render=(
            getattr(window, "visible_evaluation_controller", None).is_busy()
            if hasattr(window, "visible_evaluation_controller")
            else False
        )
    )
    schedulers = []
    for name in (
        "visible_evaluation_controller",
        "montage_tile_evaluation_controller",
        "stage_evaluation_controller",
        "pixel_evaluation_controller",
        "profile_evaluation_controller",
        "roi_evaluation_controller",
        "prefetch_evaluation_controller",
    ):
        controller = getattr(window, name, None)
        if controller is not None:
            schedulers.append(controller.diagnostics())

    session = getattr(window, "_montage_session", None)
    canvas = getattr(window, "_current_montage_canvas", None)
    canvas_bytes = None
    if canvas is not None:
        canvas_bytes = int(getattr(canvas.data, "nbytes", 0))
        histogram = getattr(canvas, "histogram_data", None)
        if histogram is not None:
            canvas_bytes += int(getattr(histogram, "nbytes", 0))
    montage = MontageRuntimeDiagnostics(
        active=session is not None,
        session_id=None if session is None else int(session.session_id),
        canvas_rect=None
        if canvas is None
        else (
            int(canvas.origin_x),
            int(canvas.origin_y),
            int(canvas.origin_x + canvas.display_shape[1]),
            int(canvas.origin_y + canvas.display_shape[0]),
        ),
        canvas_shape=None if canvas is None else tuple(int(size) for size in canvas.display_shape),
        canvas_bytes=canvas_bytes,
        loaded_tiles=0 if session is None else len(session.rendered_tiles),
        loading_tiles=0 if session is None else len(session.loading_tiles),
        pending_tiles=0 if session is None else len(session.pending_tiles),
        pending_level_tiles=0 if session is None else len(getattr(session, "pending_level_tiles", ())),
        skipped_tiles=0 if session is None else len(session.skipped_tiles),
        visible_tiles=0 if session is None else len(session.visible_tiles),
        attached_stage_requests=0 if session is None else len(getattr(session, "attached_stage_requests", ())),
        display_mode=str(getattr(window.img_view, "montageDisplayMode", lambda: "canvas")()),
        show_loading_overlays=False if session is None else bool(session.show_loading_overlays),
    )

    decision = getattr(window, "_last_render_decision", None)
    context = getattr(window, "_last_render_context", None)
    render = RenderRuntimeDiagnostics(
        last_decision_kind="" if decision is None else str(getattr(decision.kind, "value", decision.kind)),
        last_decision_reason="" if decision is None else str(getattr(decision, "reason", "")),
        last_context_summary="" if context is None else str(context),
        last_request_key=str(getattr(window, "_last_render_request_key", "") or ""),
        last_error=str(getattr(window, "_last_render_error", "") or ""),
        estimated_display_bytes=None if context is None else int(getattr(context, "estimated_display_bytes", 0)),
        render_budget_bytes=None if context is None else int(getattr(context, "render_budget_bytes", 0)),
    )

    coalescer = getattr(window, "render_coordinator", None)
    backend_choice, workers_choice = fft_backend.get_fft_runtime_options()
    resolved = fft_backend.resolve_fft_backend(backend_choice.value)
    cost = estimate_pipeline_cost(
        window.base_data.shape,
        getattr(window.base_data, "dtype", None),
        tuple(window.document.enabled_operations),
    )
    region_plan = window.operation_evaluator.planner_diagnostics()
    capability_stage_count = None if region_plan is None else len(tuple(getattr(region_plan, "stages", ())))
    candidates = () if region_plan is None else tuple(getattr(region_plan, "cache_candidates", ()))
    transitions = () if region_plan is None else tuple(getattr(region_plan, "transitions", ()))
    expanded_axes = tuple(sorted({int(axis) for transition in transitions for axis in getattr(transition, "expanded_axes", ())}))
    stage_cache_diagnostics = window.operation_evaluator.stage_cache_diagnostics()
    stage_materialization_diagnostics = window.operation_evaluator.stage_materialization_diagnostics()
    upload_timing = (
        window.img_view.lastImageUploadTiming()
        if hasattr(getattr(window, "img_view", None), "lastImageUploadTiming")
        else None
    )
    compute_policy = getattr(window, "compute_policy", None)
    lanes = (
        ComputeLane.VISIBLE,
        ComputeLane.MONTAGE_TILE,
        ComputeLane.STAGE,
        ComputeLane.PREFETCH,
        ComputeLane.PROFILE,
        ComputeLane.ROI,
        ComputeLane.PIXEL,
    )
    compute_worker_summaries = (
        tuple(f"{lane.value}={compute_policy.workers_for_lane(lane)}" for lane in lanes)
        if compute_policy is not None
        else ()
    )
    compute_fft_worker_summaries = (
        tuple(f"{lane.value}={compute_policy.fft_workers_for_lane(lane)}" for lane in lanes)
        if compute_policy is not None
        else ()
    )

    return WindowRuntimeDiagnostics(
        memory_policy=policy,
        image_cache=window.operation_evaluator.image_cache_diagnostics(),
        tile_cache=window.operation_evaluator.tile_cache_diagnostics(),
        profile_cache=window.operation_evaluator.profile_cache_diagnostics(),
        stage_cache=stage_cache_diagnostics,
        stage_materialization=stage_materialization_diagnostics,
        stage_warmup=getattr(window, "_last_stage_warmup_decision", None),
        montage_prefetch=tuple(getattr(window, "_last_montage_prefetch_decisions", ()) or ()),
        schedulers=tuple(schedulers),
        render=render,
        montage=montage,
        canvas_preserve=(
            window.layout_manager.canvas_preserver.diagnostics()
            if hasattr(getattr(window, "layout_manager", None), "canvas_preserver")
            else CanvasPreserveRuntimeDiagnostics()
        ),
        render_timing=RenderTimingDiagnostics(
            last_render_sync_ms=getattr(window, "_last_render_sync_ms", None),
            last_control_sync_ms=getattr(window, "_last_control_sync_ms", None),
            last_planning_ms=getattr(window, "_last_planning_ms", None),
            last_worker_queue_wait_ms=getattr(window, "_last_worker_queue_wait_ms", None),
            last_evaluation_ms=getattr(window, "_last_render_completed_ms", None),
            last_display_commit_ms=getattr(window, "_last_display_commit_ms", None),
            last_set_image_ms=getattr(window, "_last_set_image_ms", None),
            last_levels_histogram_ms=getattr(window, "_last_levels_histogram_ms", None),
            last_operation_dock_ms=getattr(window, "_last_operation_dock_ms", None),
            last_inspection_refresh_ms=getattr(window, "_last_inspection_refresh_ms", None),
        ),
        montage_timing=MontageTimingDiagnostics(
            last_tile_eval_ms=getattr(window, "_last_montage_tile_eval_ms", None),
            last_tile_cache_lookup_ms=getattr(window, "_last_montage_tile_cache_lookup_ms", None),
            last_tile_cache_hit=getattr(window, "_last_montage_tile_cache_hit", None),
            last_stage_cache_lookup_ms=getattr(stage_cache_diagnostics, "last_lookup_ms", None),
            last_stage_cache_hit=getattr(stage_cache_diagnostics, "last_lookup_hit", None),
            last_stage_attach_wait_ms=getattr(window, "_last_montage_stage_attach_wait_ms", None),
            last_level_stats_ms=getattr(window, "_last_montage_level_stats_ms", None),
            last_visible_upload_ms=None if upload_timing is None else upload_timing.visible_upload_ms,
            last_histogram_upload_ms=None if upload_timing is None else upload_timing.histogram_upload_ms,
            last_histogram_recompute_ms=None if upload_timing is None else upload_timing.histogram_recompute_ms,
            last_rgb_window_ms=None if upload_timing is None else upload_timing.rgb_window_ms,
            last_tile_layer_upload_ms=None if upload_timing is None else upload_timing.tile_layer_upload_ms,
            last_tile_layer_rgb_window_ms=None if upload_timing is None else upload_timing.tile_layer_rgb_window_ms,
            last_level_sync_ms=None if upload_timing is None else upload_timing.level_sync_ms,
            last_canvas_compose_ms=getattr(window, "_last_montage_canvas_compose_ms", None),
            last_canvas_patch_ms=getattr(window, "_last_montage_canvas_patch_ms", None),
            last_canvas_commit_ms=getattr(window, "_last_montage_canvas_commit_ms", None),
            last_set_image_ms=getattr(window, "_last_set_image_ms", None),
            last_overlay_update_ms=getattr(window, "_last_montage_overlay_update_ms", None),
            cached_tiles_last_session=int(getattr(window, "_montage_cached_tiles_last_session", 0) or 0),
            missing_tiles_last_session=int(getattr(window, "_montage_missing_tiles_last_session", 0) or 0),
            patched_tiles_last_flush=int(getattr(window, "_montage_patched_tiles_last_flush", 0) or 0),
            upload_visible_bytes=0 if upload_timing is None else int(upload_timing.visible_bytes),
            upload_histogram_bytes=0 if upload_timing is None else int(upload_timing.histogram_bytes),
            upload_fast_same_object=False if upload_timing is None else bool(upload_timing.fast_same_object),
            tile_layer_visible_items=0 if upload_timing is None else int(upload_timing.tile_layer_visible_items),
            tile_layer_items_updated=0 if upload_timing is None else int(upload_timing.tile_layer_items_updated),
            tile_layer_items_skipped=0 if upload_timing is None else int(upload_timing.tile_layer_items_skipped),
            tile_layer_rgb_window_tiles=0 if upload_timing is None else int(upload_timing.tile_layer_rgb_window_tiles),
            coalesced_commits=int(getattr(window, "_montage_coalesced_commits", 0) or 0),
        ),
        render_coalescer=RenderCoalescerDiagnostics(
            pending=False if coalescer is None else bool(coalescer.has_pending_render),
            interactive_active=False if coalescer is None else bool(coalescer.interactive_active),
            requested=0 if coalescer is None else int(coalescer.requested),
            flushed=0 if coalescer is None else int(coalescer.flushed),
            coalesced=0 if coalescer is None else int(coalescer.coalesced),
            deferred_side_panel_refreshes=0 if coalescer is None else int(coalescer.deferred_side_panel_refreshes),
        ),
        fft_backend_choice=backend_choice.value,
        fft_backend_resolved=resolved.name,
        fft_workers_choice=workers_choice.value,
        fft_workers_resolved=int(fft_backend.runtime_fft_workers()),
        compute_worker_summaries=compute_worker_summaries,
        compute_fft_worker_summaries=compute_fft_worker_summaries,
        operation_count=len(tuple(window.document.enabled_operations)),
        derived_shape=tuple(int(size) for size in window.document.current_shape),
        derived_dtype=str(window.data.dtype),
        pipeline_peak_bytes=cost.estimated_peak_bytes,
        pipeline_warnings=tuple(cost.warnings),
        optimized_operation_count=(
            getattr(region_plan, "optimized_operation_count", None)
            if region_plan is not None
            else cost.optimized_operation_count
        ),
        operation_optimization_summaries=(
            tuple(getattr(region_plan, "optimization_steps", ()))
            if region_plan is not None
            else tuple(cost.optimization_steps)
        ),
        capability_stage_count=capability_stage_count,
        stage_cache_candidate_count=None if region_plan is None else len(candidates),
        stage_cache_candidate_summaries=tuple(_stage_cache_candidate_summary(candidate) for candidate in candidates),
        operation_final_region="" if region_plan is None else region_text(region_plan.final_region),
        operation_required_input_region="" if region_plan is None else region_text(region_plan.required_input_region),
        operation_expanded_axes=expanded_axes,
        operation_transition_summaries=tuple(_region_transition_summary(transition) for transition in transitions),
    )


def _stage_cache_candidate_summary(candidate):
    region = getattr(candidate, "region", None)
    axes = "n/a" if region is None else ",".join(str(getattr(axis.kind, "value", axis.kind)) for axis in region.axes)
    nbytes = getattr(candidate, "estimated_nbytes", None)
    size = "unknown" if nbytes is None else format_bytes(int(nbytes))
    return (
        f"stage {getattr(candidate, 'stage_index', '?')} "
        f"{getattr(candidate, 'priority', 'n/a')} {size}, axes={axes}, "
        f"retain={'yes' if getattr(candidate, 'retain', True) else 'no'} "
        f"{getattr(candidate, 'retain_reason', '')}, "
        f"{getattr(candidate, 'reason', '')}"
    )


def _region_transition_summary(transition):
    expanded = tuple(getattr(transition, "expanded_axes", ()))
    expanded_text = "n/a" if not expanded else ",".join(str(int(axis)) for axis in expanded)
    return (
        f"stage {getattr(transition, 'stage_index', '?')} {type(getattr(transition, 'operation', object())).__name__} "
        f"output={region_text(transition.output_region)} "
        f"input={region_text(transition.required_input_region)} "
        f"expanded={expanded_text}"
    )
