"""Runtime diagnostics snapshot construction for ArrayScope windows."""

from __future__ import annotations

from time import perf_counter

from arrayscope.core.compute_policy import ComputeLane
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
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.model.tile_identity import tile_ack_identity
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
        "histogram_evaluation_controller",
        "pixel_evaluation_controller",
        "profile_evaluation_controller",
        "roi_evaluation_controller",
        "prefetch_evaluation_controller",
    ):
        controller = getattr(window, name, None)
        if controller is not None:
            schedulers.append(controller.diagnostics())

    session = getattr(window.renderer, "_frame_session", None)
    overlay_count = _montage_overlay_count(window)
    presentation = _presentation_diagnostics(window)
    lod_decision = None if session is None else getattr(session, "lod_policy_decision", None)
    lod_demand = None if lod_decision is None else getattr(lod_decision, "demand", None)
    lod_page_cache = None if session is None else getattr(session, "lod_page_cache", None)
    lod_page_families = (
        ()
        if lod_page_cache is None
        else tuple(
            sorted(
                (
                    tuple(int(step) for step in reduction),
                    str(reducer),
                    int(count),
                )
                for (
                    reduction,
                    reducer,
                ), count in lod_page_cache.resident_lod_reducer_counts().items()
            )
        )
    )
    lod_tile_levels = _montage_payload_level_counts(session)
    presented_lod = _montage_presented_lod(session, lod_decision)
    lifecycle_snapshot = None if session is None else session.lifecycle_snapshot()
    lifecycle_phase_counts = {} if lifecycle_snapshot is None else dict(lifecycle_snapshot.counts)
    tile_identity_probe = _tile_identity_probe(window, session)
    retention_started_at = getattr(window.renderer, "_slice_retention_started_at", None)
    stage_values = (
        {} if session is None else dict(getattr(session.stage_fan_in, "values", {}) or {})
    )
    stage_bindings = (
        {} if session is None else dict(getattr(session.stage_fan_in, "tile_stage_keys", {}) or {})
    )
    unresolved_stage_bindings = {
        int(tile): key for tile, key in stage_bindings.items() if key not in stage_values
    }
    semantic_evidence = (
        {
            "target_population": 0,
            "covered_sources": (),
            "covered_source_count": 0,
            "pending_batches": 0,
            "inflight_generation": None,
            "blocking_reason": "inactive",
            "source_batch_limit": 0,
            "pixel_limit": 0,
        }
        if session is None
        else session.semantic_level_evidence_diagnostics()
    )
    montage = MontageRuntimeDiagnostics(
        active=session is not None,
        session_id=None if session is None else int(session.session_id),
        loaded_tiles=0 if session is None else len(session.rendered_tiles),
        loading_tiles=0 if session is None else len(session.loading_tiles),
        active_tile_requests=0
        if session is None
        else len(getattr(session, "active_tile_requests", ())),
        target_unsettled_tiles=(
            0 if session is None else len(session.required_target_unsettled_tiles())
        ),
        pending_payload_upserts=0
        if session is None
        else len(getattr(session, "pending_payload_upserts", ())),
        pending_removals=0 if session is None else len(getattr(session, "pending_removals", ())),
        pending_level_tiles=0
        if session is None
        else len(getattr(session, "pending_level_tiles", ())),
        level_scan_remaining_tiles=0
        if session is None
        else int(getattr(session, "level_scan_remaining_tiles", 0) or 0),
        histogram_aggregate_inflight=(
            False
            if session is None
            else bool(getattr(session, "histogram_aggregate_inflight", False))
        ),
        semantic_evidence_target_population=int(semantic_evidence["target_population"]),
        semantic_evidence_covered_sources=tuple(semantic_evidence["covered_sources"]),
        semantic_evidence_covered_source_count=int(semantic_evidence["covered_source_count"]),
        semantic_evidence_pending_batches=int(semantic_evidence["pending_batches"]),
        semantic_evidence_inflight_generation=semantic_evidence["inflight_generation"],
        semantic_evidence_blocking_reason=str(semantic_evidence["blocking_reason"]),
        semantic_evidence_source_batch_limit=int(semantic_evidence["source_batch_limit"]),
        semantic_evidence_pixel_limit=int(semantic_evidence["pixel_limit"]),
        skipped_tiles=0 if session is None else len(session.skipped_tiles),
        visible_tiles=0 if session is None else len(session.visible_tiles),
        presented_tiles=0 if session is None else len(session.lifecycle.presented_tiles),
        overlay_count=overlay_count,
        attached_stage_requests=0
        if session is None
        else len(session.stage_fan_in.attached_requests),
        waiting_stage_requests=len(unresolved_stage_bindings),
        final_commit_pending=False
        if session is None
        else bool(getattr(session, "final_commit_pending", False)),
        flush_pending=False if session is None else bool(getattr(session, "flush_pending", False)),
        presentation_draw_count=int(presentation.get("draw_count", 0) or 0),
        tile_presentation_request_count=int(
            presentation.get("tile_presentation_request_count", 0) or 0
        ),
        tile_presentation_draw_count=int(presentation.get("tile_presentation_draw_count", 0) or 0),
        tile_presentation_draw_pending=bool(
            presentation.get("tile_presentation_draw_pending", False)
        ),
        tile_visual_visible_pages=int(presentation.get("tile_visual_visible_pages", 0) or 0),
        overlays_above_tiles=bool(presentation.get("overlays_above_tiles", False)),
        display_mode=str(getattr(window.img_view, "montageDisplayMode", lambda: "none")()),
        backend_chosen=str(
            getattr(
                window.renderer,
                "_last_montage_backend_actual",
                getattr(
                    getattr(window.renderer, "_last_montage_backend_choice", None),
                    "backend",
                    "none",
                ),
            )
        ),
        backend_reason=str(
            getattr(getattr(window.renderer, "_last_montage_backend_choice", None), "reason", "")
        ),
        backend_warning=str(getattr(window.renderer, "_last_montage_backend_warning", "") or ""),
        wgpu_page_pools=tuple(
            dict(row) for row in tuple(presentation.get("wgpu_page_pools", ()) or ())
        ),
        wgpu_page_table_resident_count=int(presentation.get("page_table_resident_count", 0) or 0),
        wgpu_atomic_warm_pinned_pages=int(
            presentation.get("wgpu_atomic_warm_pinned_pages", 0) or 0
        ),
        wgpu_uploads_total=int(presentation.get("wgpu_uploads_total", 0) or 0),
        wgpu_upload_bytes_total=int(presentation.get("wgpu_upload_bytes_total", 0) or 0),
        wgpu_uploads_by_level=tuple(
            dict(row) for row in tuple(presentation.get("wgpu_uploads_by_level", ()) or ())
        ),
        wgpu_active_resident_bytes=int(presentation.get("wgpu_active_resident_bytes", 0) or 0),
        wgpu_allocated_pool_bytes=int(presentation.get("wgpu_allocated_pool_bytes", 0) or 0),
        wgpu_pool_grows_total=int(presentation.get("wgpu_pool_grows_total", 0) or 0),
        wgpu_pool_growth_copy_bytes_total=int(
            presentation.get("wgpu_pool_growth_copy_bytes_total", 0) or 0
        ),
        wgpu_last_pool_exhaustion=str(presentation.get("wgpu_last_pool_exhaustion", "") or ""),
        show_loading_overlays=False if session is None else bool(session.show_loading_overlays),
        tile_lod_desired_factor=1
        if lod_demand is None
        else int(getattr(lod_demand, "desired_factor", 1) or 1),
        tile_lod_applied_factor=int(presented_lod[1]),
        tile_lod_desired_factor_xy=(1, 1)
        if lod_demand is None
        else tuple(int(value) for value in getattr(lod_demand, "desired_factor_xy", (1, 1))),
        tile_lod_applied_factor_xy=tuple(int(value) for value in presented_lod[2]),
        tile_lod_source_texels_per_pixel_xy=(0.0, 0.0)
        if lod_demand is None
        else tuple(
            float(value) for value in getattr(lod_demand, "source_texels_per_pixel_xy", (0.0, 0.0))
        ),
        tile_lod_policy="native-only"
        if lod_decision is None
        else str(getattr(lod_decision, "policy", "native-only") or "native-only"),
        tile_lod_reason=_presented_lod_reason(lod_decision, presented_lod),
        tile_lod_applied_level=int(presented_lod[0]),
        tile_lod_resident_tile_levels=lod_tile_levels,
        tile_lod_pyramid_bytes=0
        if lod_page_cache is None
        else int(getattr(lod_page_cache, "bytes_used", 0) or 0),
        tile_lod_pyramid_entries=0 if lod_page_cache is None else len(lod_page_cache),
        tile_lod_pyramid_hits=0
        if lod_page_cache is None
        else int(getattr(lod_page_cache, "hits", 0) or 0),
        tile_lod_pyramid_misses=0
        if lod_page_cache is None
        else int(getattr(lod_page_cache, "misses", 0) or 0),
        tile_lod_pyramid_evictions=0
        if lod_page_cache is None
        else int(getattr(lod_page_cache, "evictions", 0) or 0),
        tile_lod_page_families=lod_page_families,
        tile_lod_pending_materializations=(
            0
            if session is None
            else len(getattr(session, "pending_rung_materializations", ()) or ())
            + (
                0
                if lod_page_cache is None
                else int(getattr(lod_page_cache, "pending_count", 0) or 0)
            )
        ),
        tile_lod_materializations_completed=0
        if session is None
        else int(getattr(session, "lod_materializations_completed", 0) or 0),
        tile_lod_preview_reduced_scheduled=int(
            getattr(window.renderer, "_montage_preview_reduced_scheduled", 0) or 0
        ),
        tile_lod_preview_reduced_blocked=int(
            getattr(window.renderer, "_montage_preview_reduced_blocked", 0) or 0
        ),
        tile_lod_preview_reduced_failures=int(
            getattr(window.renderer, "_montage_preview_reduced_failures", 0) or 0
        ),
        tile_lod_preview_reduced_last_gate=str(
            getattr(window.renderer, "_montage_preview_reduced_last_gate", "") or ""
        ),
        tile_lod_rung_evaluations=_rung_evaluation_rows(session),
        tile_lod_coarse_rung_gates=_coarse_rung_gate_rows(session),
        tile_lod_pipeline_counters=_pipeline_counter_row(session),
        tile_lod_ladder_floor_level=int(getattr(_ladder_policy(session), "floor_level", -1)),
        tile_lod_ladder_preview_level=int(getattr(_ladder_policy(session), "preview_level", -1)),
        tile_lod_ladder_reduced_input=bool(
            getattr(_ladder_policy(session), "reduced_input_available", False)
        ),
        tile_lod_preview_presentations=0
        if session is None
        else int(getattr(session, "lod_preview_presentations", 0) or 0),
        tile_lod_stats_cross_level_reuses=0
        if session is None
        else int(getattr(session, "lod_stats_cross_level_reuses", 0) or 0),
        tile_lod_stats_recomputes=0
        if session is None
        else int(getattr(session, "lod_stats_recomputes", 0) or 0),
        tile_lod_cross_level_reductions=0
        if session is None
        else int(getattr(session, "lod_cross_level_reductions", 0) or 0),
        tile_lod_pipeline_reruns_avoided=int(
            getattr(window.renderer, "_montage_quality_pipeline_reruns_avoided", 0) or 0
        ),
        tile_histogram_lod_swap_recomputes=int(
            getattr(window.img_view, "tile_histogram_lod_swap_recomputes", 0) or 0
        ),
        tile_histogram_cross_level_reuses=int(
            getattr(window.img_view, "tile_histogram_cross_level_reuses", 0) or 0
        ),
        tile_compute_cache_hits=0
        if session is None
        else int(getattr(session, "tile_compute_cache_hits", 0) or 0),
        tile_compute_stage_backed=0
        if session is None
        else int(getattr(session, "tile_compute_stage_backed", 0) or 0),
        tile_compute_direct=0
        if session is None
        else int(getattr(session, "tile_compute_direct", 0) or 0),
        tile_compute_waiting_for_stage=0
        if session is None
        else int(getattr(session, "tile_compute_waiting_for_stage", 0) or 0),
        tile_compute_stage_backed_ms=0.0
        if session is None
        else float(getattr(session, "tile_compute_stage_backed_ms", 0.0) or 0.0),
        tile_compute_direct_ms=0.0
        if session is None
        else float(getattr(session, "tile_compute_direct_ms", 0.0) or 0.0),
        tile_compute_stage_backed_max_ms=0.0
        if session is None
        else float(getattr(session, "tile_compute_stage_backed_max_ms", 0.0) or 0.0),
        tile_compute_direct_max_ms=0.0
        if session is None
        else float(getattr(session, "tile_compute_direct_max_ms", 0.0) or 0.0),
        lead_direct_tiles=0
        if session is None
        else int(getattr(session, "lead_direct_tiles", 0) or 0),
        stage_backed_tiles_pending=0
        if session is None
        else int(getattr(session, "stage_backed_tiles_pending", 0) or 0),
        retained_stage_index=None
        if session is None
        else getattr(session, "retained_stage_index", None),
        retained_stage_decision=""
        if session is None
        else str(getattr(session, "retained_stage_decision", "") or ""),
        repeated_expensive_stage_per_tile=False
        if session is None
        else bool(getattr(session, "repeated_expensive_stage_per_tile", False)),
        priority_retargeted_tiles=0
        if session is None
        else int(getattr(session, "priority_retargeted_tiles", 0) or 0),
        resident_crop_rebind_last_gate=str(
            getattr(window.renderer, "resident_crop_rebind_last_gate", "") or ""
        ),
        resident_crop_rebind_totals=dict(
            getattr(window.renderer, "resident_crop_rebind_totals", None) or {}
        ),
        lifecycle_parked=0 if session is None else len(session.lifecycle.parked_tiles),
        lifecycle_evaluating=0 if session is None else len(session.lifecycle.evaluating_tiles),
        lifecycle_dangling_claims=0
        if session is None
        else len(session.lifecycle.dangling_claims()),
        lifecycle_semantic_mismatches=_lifecycle_semantic_mismatches(session),
        lifecycle_identity_rejections=0
        if session is None
        else int(session.lifecycle.identity_rejections),
        dirty_payload_tiles=0
        if session is None
        else len(getattr(session, "dirty_payloads", ()) or ()),
        ledger_needs_first_pixel=int(lifecycle_phase_counts.get("needs_first_pixel", 0) or 0),
        ledger_fallback_shown=int(lifecycle_phase_counts.get("fallback_shown", 0) or 0),
        ledger_target_schedulable=int(lifecycle_phase_counts.get("target_schedulable", 0) or 0),
        ledger_target_waiting_stage=int(lifecycle_phase_counts.get("target_waiting_stage", 0) or 0),
        ledger_target_running=int(lifecycle_phase_counts.get("target_running", 0) or 0),
        ledger_target_ready=int(lifecycle_phase_counts.get("target_ready", 0) or 0),
        ledger_target_emitted=int(lifecycle_phase_counts.get("target_emitted", 0) or 0),
        ledger_target_presented=int(lifecycle_phase_counts.get("target_presented", 0) or 0),
        ledger_orphan_running=0
        if lifecycle_snapshot is None
        else int(lifecycle_snapshot.orphan_running),
        ledger_parked_without_producer=(
            0 if lifecycle_snapshot is None else int(lifecycle_snapshot.parked_without_producer)
        ),
        backend_stale_identities=_backend_stale_identities(session),
        slice_retention_transitions=int(
            getattr(window.renderer, "_frame_session_transitions_retained", 0) or 0
        ),
        slice_retention_replacements=int(
            getattr(window.renderer, "_slice_retention_replacements", 0) or 0
        ),
        slice_retention_active=retention_started_at is not None,
        slice_retention_inflight_age_ms=(
            0.0
            if retention_started_at is None
            else max(0.0, (perf_counter() - float(retention_started_at)) * 1000.0)
        ),
        slice_retention_last_replacement_ms=float(
            getattr(window.renderer, "_slice_retention_last_replacement_ms", 0.0) or 0.0
        ),
        slice_retention_max_replacement_ms=float(
            getattr(window.renderer, "_slice_retention_max_replacement_ms", 0.0) or 0.0
        ),
        stall_assertions=int(getattr(window.renderer, "_montage_stall_assertions", 0) or 0),
        last_stall_signature=tuple(
            int(value)
            for value in (getattr(window.renderer, "_montage_watchdog_last_stall", ()) or ())
        ),
        tile_identity_probe=tile_identity_probe,
        presented_order_sample=()
        if session is None
        else tuple(
            int(index) for index in tuple(getattr(session, "presented_order", ()) or ())[:64]
        ),
    )

    decision = getattr(window, "_last_render_decision", None)
    context = getattr(window, "_last_render_context", None)
    render = RenderRuntimeDiagnostics(
        last_decision_kind=""
        if decision is None
        else str(getattr(decision.kind, "value", decision.kind)),
        last_decision_reason="" if decision is None else str(getattr(decision, "reason", "")),
        last_context_summary="" if context is None else str(context),
        last_request_key=str(getattr(window, "_last_render_request_key", "") or ""),
        last_error=str(getattr(window, "_last_render_error", "") or ""),
        estimated_display_bytes=None
        if context is None
        else int(getattr(context, "estimated_display_bytes", 0)),
        render_budget_bytes=None
        if context is None
        else int(getattr(context, "render_budget_bytes", 0)),
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
    capability_stage_count = (
        None if region_plan is None else len(tuple(getattr(region_plan, "stages", ())))
    )
    candidates = () if region_plan is None else tuple(getattr(region_plan, "cache_candidates", ()))
    transitions = () if region_plan is None else tuple(getattr(region_plan, "transitions", ()))
    expanded_axes = tuple(
        sorted(
            {
                int(axis)
                for transition in transitions
                for axis in getattr(transition, "expanded_axes", ())
            }
        )
    )
    stage_cache_diagnostics = window.operation_evaluator.stage_cache_diagnostics()
    stage_materialization_diagnostics = (
        window.operation_evaluator.stage_materialization_diagnostics()
    )
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
        ComputeLane.HISTOGRAM,
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

    image_backend_selected = getattr(
        getattr(getattr(window, "app_settings", None), "image_rendering_backend", "pyqtgraph"),
        "value",
        "pyqtgraph",
    )
    image_backend_actual = image_view_backend_capabilities(getattr(window, "img_view", None)).name

    return WindowRuntimeDiagnostics(
        memory_policy=policy,
        display_cache=window.operation_evaluator.display_cache_diagnostics(),
        profile_cache=window.operation_evaluator.profile_cache_diagnostics(),
        stage_cache=stage_cache_diagnostics,
        stage_materialization=stage_materialization_diagnostics,
        montage_prefetch=tuple(
            getattr(window.renderer, "_last_montage_prefetch_decisions", ()) or ()
        ),
        resource_governor=(
            None
            if getattr(window, "resource_governor", None) is None
            else window.resource_governor.diagnostics(
                channels=(
                    "montage_tile_result",
                    "montage_commit",
                    "montage_cold_commit",
                    "tile_layer_commit",
                    "histogram_refresh",
                    "histogram_preview",
                    "roi_refresh",
                    "profile_update",
                    "pixel_hover",
                    "kernel_bridge_drain",
                )
            )
        ),
        schedulers=tuple(schedulers),
        render=render,
        montage=montage,
        kernel=None if getattr(window, "kernel", None) is None else window.kernel.diagnostics(),
        canvas_preserve=(
            window.layout_manager.canvas_preserver.diagnostics()
            if hasattr(getattr(window, "layout_manager", None), "canvas_preserver")
            else CanvasPreserveRuntimeDiagnostics()
        ),
        render_timing=RenderTimingDiagnostics(
            last_render_sync_ms=getattr(window.renderer, "_last_render_sync_ms", None),
            last_control_sync_ms=getattr(window.renderer, "_last_control_sync_ms", None),
            last_planning_ms=getattr(window, "_last_planning_ms", None),
            last_worker_queue_wait_ms=getattr(window, "_last_worker_queue_wait_ms", None),
            last_evaluation_ms=getattr(window, "_last_render_completed_ms", None),
            last_display_commit_ms=getattr(window.renderer, "_last_display_commit_ms", None),
            last_set_image_ms=getattr(window.renderer, "_last_set_image_ms", None),
            last_levels_histogram_ms=getattr(window.renderer, "_last_levels_histogram_ms", None),
            last_operation_dock_ms=getattr(window, "_last_operation_dock_ms", None),
            last_inspection_refresh_ms=getattr(window, "_last_inspection_refresh_ms", None),
        ),
        montage_timing=MontageTimingDiagnostics(
            last_viewport_plan_ms=getattr(window.renderer, "_last_montage_viewport_plan_ms", None),
            last_cache_resolve_ms=getattr(window.renderer, "_last_montage_cache_resolve_ms", None),
            last_stage_plan_ms=getattr(window.renderer, "_last_montage_stage_plan_ms", None),
            last_session_setup_ms=getattr(window.renderer, "_last_frame_session_setup_ms", None),
            last_initial_commit_ms=getattr(
                window.renderer, "_last_montage_initial_commit_ms", None
            ),
            last_tile_eval_ms=getattr(window.renderer, "_last_montage_tile_eval_ms", None),
            last_display_cache_lookup_ms=getattr(
                window.renderer, "_last_montage_display_cache_lookup_ms", None
            ),
            last_display_cache_hit=getattr(
                window.renderer, "_last_montage_display_cache_hit", None
            ),
            last_stage_cache_lookup_ms=getattr(stage_cache_diagnostics, "last_lookup_ms", None),
            last_stage_cache_hit=getattr(stage_cache_diagnostics, "last_lookup_hit", None),
            last_stage_attach_wait_ms=getattr(
                window.renderer, "_last_montage_stage_attach_wait_ms", None
            ),
            last_level_stats_ms=getattr(window.renderer, "_last_montage_level_stats_ms", None),
            last_tile_payload_build_ms=getattr(
                window.renderer, "_last_montage_tile_payload_build_ms", None
            ),
            last_visible_upload_ms=None
            if upload_timing is None
            else upload_timing.visible_upload_ms,
            last_histogram_upload_ms=None
            if upload_timing is None
            else upload_timing.histogram_upload_ms,
            last_histogram_recompute_ms=None
            if upload_timing is None
            else upload_timing.histogram_recompute_ms,
            last_rgb_window_ms=None if upload_timing is None else upload_timing.rgb_window_ms,
            last_tile_layer_upload_ms=None
            if upload_timing is None
            else upload_timing.tile_layer_upload_ms,
            last_tile_layer_rgb_window_ms=None
            if upload_timing is None
            else upload_timing.tile_layer_rgb_window_ms,
            last_level_sync_ms=None if upload_timing is None else upload_timing.level_sync_ms,
            last_tile_commit_ms=getattr(window.renderer, "_last_montage_tile_commit_ms", None),
            last_tile_prepare_apply_ms=getattr(
                window.renderer, "_last_montage_tile_prepare_apply_ms", None
            ),
            last_tile_layer_apply_ms=getattr(
                window.renderer, "_last_montage_tile_layer_apply_ms", None
            ),
            last_tile_acknowledge_ms=getattr(
                window.renderer, "_last_montage_tile_acknowledge_ms", None
            ),
            last_tile_retained_store_ms=getattr(
                window.renderer, "_last_montage_tile_retained_store_ms", None
            ),
            last_tile_state_publish_ms=getattr(
                window.renderer, "_last_montage_tile_state_publish_ms", None
            ),
            last_tile_geometry_sync_ms=getattr(
                window.renderer, "_last_montage_tile_geometry_sync_ms", None
            ),
            last_tile_identity_check_ms=getattr(
                window.renderer, "_last_montage_tile_identity_check_ms", None
            ),
            last_tile_followup_ms=getattr(window.renderer, "_last_montage_tile_followup_ms", None),
            last_set_image_ms=getattr(window.renderer, "_last_set_image_ms", None),
            last_overlay_update_ms=getattr(
                window.renderer, "_last_montage_overlay_update_ms", None
            ),
            cached_tiles_last_session=int(
                getattr(window.renderer, "_montage_cached_tiles_last_session", 0) or 0
            ),
            missing_tiles_last_session=int(
                getattr(window.renderer, "_montage_missing_tiles_last_session", 0) or 0
            ),
            committed_tile_upserts_last_flush=int(
                getattr(window.renderer, "_montage_committed_tile_upserts_last_flush", 0) or 0
            ),
            upload_visible_bytes=0 if upload_timing is None else int(upload_timing.visible_bytes),
            upload_histogram_bytes=0
            if upload_timing is None
            else int(upload_timing.histogram_bytes),
            upload_fast_same_object=False
            if upload_timing is None
            else bool(upload_timing.fast_same_object),
            tile_layer_visible_items=0
            if upload_timing is None
            else int(upload_timing.tile_layer_visible_items),
            tile_layer_items_created=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_items_created", 0)),
            tile_layer_items_updated=0
            if upload_timing is None
            else int(upload_timing.tile_layer_items_updated),
            tile_layer_items_skipped=0
            if upload_timing is None
            else int(upload_timing.tile_layer_items_skipped),
            tile_layer_rgb_window_tiles=0
            if upload_timing is None
            else int(upload_timing.tile_layer_rgb_window_tiles),
            tile_layer_image_replacements=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_image_replacements", 0)),
            tile_layer_existing_items_shown=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_existing_items_shown", 0)),
            tile_layer_relocated_tiles=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_relocated_tiles", 0)),
            tile_layer_resident_items=0
            if upload_timing is None
            else int(upload_timing.tile_layer_resident_items),
            tile_layer_storage_capacity=0
            if upload_timing is None
            else int(upload_timing.tile_layer_storage_capacity),
            tile_layer_storage_rebuilds=0
            if upload_timing is None
            else int(upload_timing.tile_layer_storage_rebuilds),
            tile_layer_storage_evictions=0
            if upload_timing is None
            else int(upload_timing.tile_layer_storage_evictions),
            tile_layer_texture_uploads=0
            if upload_timing is None
            else int(upload_timing.tile_layer_texture_uploads),
            tile_layer_texture_upload_bytes=0
            if upload_timing is None
            else int(upload_timing.tile_layer_texture_upload_bytes),
            tile_layer_texture_prepare_ms=None
            if upload_timing is None
            else upload_timing.tile_layer_texture_prepare_ms,
            tile_layer_texture_submit_ms=None
            if upload_timing is None
            else upload_timing.tile_layer_texture_submit_ms,
            tile_layer_vertex_uploads=0
            if upload_timing is None
            else int(upload_timing.tile_layer_vertex_uploads),
            tile_layer_level_updates=0
            if upload_timing is None
            else int(upload_timing.tile_layer_level_updates),
            tile_layer_estimated_gpu_bytes=0
            if upload_timing is None
            else int(upload_timing.tile_layer_estimated_gpu_bytes),
            tile_layer_cpu_shadow_bytes=0
            if upload_timing is None
            else int(upload_timing.tile_layer_cpu_shadow_bytes),
            tile_layer_page_count=0
            if upload_timing is None
            else int(upload_timing.tile_layer_page_count),
            tile_layer_active_pages=0
            if upload_timing is None
            else int(upload_timing.tile_layer_active_pages),
            tile_layer_device_max_texture_size=0
            if upload_timing is None
            else int(upload_timing.tile_layer_device_max_texture_size),
            tile_layer_budget_bytes=0
            if upload_timing is None
            else int(upload_timing.tile_layer_budget_bytes),
            tile_layer_near_resident_items=0
            if upload_timing is None
            else int(upload_timing.tile_layer_near_resident_items),
            tile_layer_warm_resident_items=0
            if upload_timing is None
            else int(upload_timing.tile_layer_warm_resident_items),
            tile_layer_evicted_near_items=0
            if upload_timing is None
            else int(upload_timing.tile_layer_evicted_near_items),
            tile_layer_capacity_warning=""
            if upload_timing is None
            else str(upload_timing.tile_layer_capacity_warning),
            tile_layer_lod_level=0
            if upload_timing is None
            else int(upload_timing.tile_layer_lod_level),
            tile_layer_lod_factor=1
            if upload_timing is None
            else int(upload_timing.tile_layer_lod_factor),
            tile_layer_source_texels_per_pixel=0.0
            if upload_timing is None
            else float(upload_timing.tile_layer_source_texels_per_pixel),
            tile_layer_gutter_pixels=0
            if upload_timing is None
            else int(upload_timing.tile_layer_gutter_pixels),
            tile_layer_mipmap_updates=0
            if upload_timing is None
            else int(upload_timing.tile_layer_mipmap_updates),
            tile_layer_mipmap_available=False
            if upload_timing is None
            else bool(upload_timing.tile_layer_mipmap_available),
            tile_layer_complex_texture_uploads=0
            if upload_timing is None
            else int(upload_timing.tile_layer_complex_texture_uploads),
            tile_layer_lod_level_swaps_zero_upload=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_lod_level_swaps_zero_upload", 0)),
            tile_layer_lod_level_swaps_with_upload=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_lod_level_swaps_with_upload", 0)),
            tile_layer_superseded_reclaimed_under_pressure=0
            if upload_timing is None
            else int(getattr(upload_timing, "tile_layer_superseded_reclaimed_under_pressure", 0)),
            tile_layer_shader_uniform_updates=0
            if upload_timing is None
            else int(upload_timing.tile_layer_shader_uniform_updates),
            cpu_complex_prep_ms=None
            if upload_timing is None
            else upload_timing.cpu_complex_prep_ms,
            coalesced_commits=int(getattr(window.renderer, "_montage_coalesced_commits", 0) or 0),
        ),
        render_coalescer=RenderCoalescerDiagnostics(
            pending=False if coalescer is None else bool(coalescer.has_pending_render),
            interactive_active=False if coalescer is None else bool(coalescer.interactive_active),
            requested=0 if coalescer is None else int(coalescer.requested),
            flushed=0 if coalescer is None else int(coalescer.flushed),
            coalesced=0 if coalescer is None else int(coalescer.coalesced),
            deferred_side_panel_refreshes=0
            if coalescer is None
            else int(coalescer.deferred_side_panel_refreshes),
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
        stage_cache_candidate_summaries=tuple(
            _stage_cache_candidate_summary(candidate) for candidate in candidates
        ),
        operation_final_region="" if region_plan is None else region_text(region_plan.final_region),
        operation_required_input_region=""
        if region_plan is None
        else region_text(region_plan.required_input_region),
        operation_expanded_axes=expanded_axes,
        operation_transition_summaries=tuple(
            _region_transition_summary(transition) for transition in transitions
        ),
        image_rendering_backend=image_backend_actual,
        image_rendering_backend_selected=str(image_backend_selected),
        image_rendering_backend_actual=image_backend_actual,
    )


def _presentation_diagnostics(window) -> dict[str, object]:
    getter = getattr(getattr(window, "img_view", None), "presentation_diagnostics", None)
    if callable(getter):
        try:
            return dict(getter())
        except Exception:
            return {}
    return {}


def _tile_identity_probe(window, session) -> tuple[dict[str, object], ...]:
    """Merge semantic/lifecycle rows with what the backend physically draws."""

    if session is None:
        return ()
    semantic_rows = tuple(
        dict(row)
        for row in getattr(
            session,
            "diagnostic_tile_identity_rows",
            lambda **_kwargs: (),
        )()
    )
    getter = getattr(getattr(window, "img_view", None), "tileTruthPhysicalRows", None)
    physical_rows = (
        {}
        if not callable(getter)
        else {int(tile): dict(row) for tile, row in dict(getter() or {}).items()}
    )
    return tuple(
        {
            **row,
            **physical_rows.get(int(row.get("tile", -1)), {}),
        }
        for row in semantic_rows
    )


def _lifecycle_semantic_mismatches(session) -> int:
    """Count disagreement between lifecycle state and execution adapters."""

    if session is None:
        return 0
    from arrayscope.presentation import Semantic

    evaluated = {rec.tile_number for rec in session.lifecycle if rec.semantic is Semantic.EVALUATED}
    rendered = {int(tile) for tile in session.rendered_tiles}
    loading = {int(tile) for tile in session.loading_tiles}
    # Rendered-but-not-yet-presented tiles keep loading intent for atomic
    # replacement, so evaluated+loading is agreement until acknowledgement.
    return len((evaluated ^ rendered) - loading) + len(
        session.lifecycle.evaluating_tiles - loading - rendered
    )


def _backend_stale_identities(session) -> int:
    """ADR 0051 rule 1: drawn tiles whose backend slot identity differs from
    the session's current payload.  Nonzero at idle = visibly stale tiles."""

    if session is None:
        return 0
    lifecycle = getattr(session, "lifecycle", None)
    identities = dict(getattr(lifecycle, "backend_presented_identities", {}) or {})
    if not identities:
        return 0
    payloads = getattr(session, "display_tile_payloads", {}) or {}
    stale = 0
    for tile_number, shown_identity in identities.items():
        current = payloads.get(int(tile_number))
        if current is not None and tile_ack_identity(current) != shown_identity:
            stale += 1
    return stale


def _montage_presented_lod(session, lod_decision) -> tuple[int, int, tuple[int, int]]:
    """Applied LOD as presented on screen, not as session-wide consensus.

    ``montage_quality_applied_factor`` must describe the payloads the committed
    presentation actually shows; the session-wide decision reads as native
    while any tile is still streaming its level.
    """

    summary = getattr(session, "presented_lod_summary", None)
    if callable(summary):
        level, factor, factor_xy = summary()
        return (int(level), int(factor), tuple(int(value) for value in factor_xy))
    if lod_decision is None:
        return (0, 1, (1, 1))
    return (
        int(getattr(lod_decision, "applied_level", 0) or 0),
        int(getattr(lod_decision, "applied_factor", 1) or 1),
        tuple(int(value) for value in getattr(lod_decision, "applied_factor_xy", (1, 1))),
    )


def _presented_lod_reason(lod_decision, presented_lod) -> str:
    """Reason text consistent with the *presented* LOD fields next to it.

    The decision's reason is a snapshot from the last policy evaluation; the
    screen can converge afterwards (ingest-presented levels) with no event
    re-running the policy, leaving "…while the demanded level materializes"
    hanging in the diagnostics indefinitely.  Diagnostics describe current
    state, so the residency wording is re-derived from the presented level
    against the demanded one; non-residency reasons pass through untouched.
    """

    if lod_decision is None:
        return ""
    reason = str(getattr(lod_decision, "reason", "") or "")
    from arrayscope.display.lod import (
        LOD_POLICY_RESIDENT,
        LOD_REASON_NATIVE_SCALE,
        LOD_REASON_RESIDENT_COARSER,
        LOD_REASON_RESIDENT_FINER,
        LOD_REASON_RESIDENT_MATCH,
        LOD_REASON_RESIDENT_NATIVE_FALLBACK,
    )

    if str(getattr(lod_decision, "policy", "")) != LOD_POLICY_RESIDENT:
        return reason
    if reason not in (
        LOD_REASON_RESIDENT_MATCH,
        LOD_REASON_RESIDENT_NATIVE_FALLBACK,
        LOD_REASON_RESIDENT_FINER,
        LOD_REASON_RESIDENT_COARSER,
        LOD_REASON_NATIVE_SCALE,
    ):
        return reason
    demand = getattr(lod_decision, "demand", None)
    desired = int(getattr(demand, "desired_level", 0) or 0)
    presented_level = int(presented_lod[0])
    if presented_level == desired:
        return LOD_REASON_RESIDENT_MATCH if presented_level > 0 else LOD_REASON_NATIVE_SCALE
    if presented_level == 0:
        return LOD_REASON_RESIDENT_NATIVE_FALLBACK
    if presented_level < desired:
        return LOD_REASON_RESIDENT_FINER
    return LOD_REASON_RESIDENT_COARSER


def _montage_payload_level_counts(session) -> tuple[tuple[int, int], ...]:
    if session is None:
        return ()
    counts: dict[int, int] = {}
    for payload in dict(getattr(session, "display_tile_payloads", {}) or {}).values():
        level = int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
        counts[level] = counts.get(level, 0) + 1
    return tuple(sorted(counts.items()))


def _pipeline_counter_row(session) -> dict[str, int]:
    """The frame pipeline's own counters, which nothing surfaced before."""

    as_dict = getattr(
        getattr(getattr(session, "pipeline", None), "counters", None), "as_dict", None
    )
    if not callable(as_dict):
        return {}
    return {str(name): int(value) for name, value in as_dict().items()}


def _coarse_rung_gate_rows(session) -> tuple[tuple[str, int], ...]:
    """Cumulative "why no coarse rung" over the session, in tile-plans.

    Cumulative, not last-plan: the final plan of a settled fill is converged,
    and reporting its refusal would answer "why was there no preview" with the
    reason the tiles are *now* covered.
    """

    getter = getattr(getattr(session, "pipeline", None), "coarse_rung_refusals", None)
    if not callable(getter):
        return ()
    return tuple((str(reason), int(count)) for reason, count in getter())


def _ladder_policy(session):
    """The ladder policy the session's pipeline plans against, or None.

    Read rather than re-derived: `reduced_input_available` is captured once at
    pipeline construction, so what the plan used and what the predicate would
    say now can differ — and the plan is what the counters describe.
    """

    return getattr(getattr(getattr(session, "pipeline", None), "ladder", None), "policy", None)


def _rung_evaluation_rows(session) -> tuple[dict[str, object], ...]:
    """Per-(rung, level) evaluation cost owned by the session's frame pipeline.

    Absent before the first ladder plan, so a missing pipeline is an empty
    reading rather than a row of zeros.
    """

    timings = getattr(getattr(session, "pipeline", None), "rung_timings", None)
    if timings is None:
        return ()
    return tuple(dict(row) for row in timings.rows())


def _montage_overlay_count(window) -> int:
    getter = getattr(getattr(window, "img_view", None), "montageTileOverlayCount", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:
            return 0
    return 0


def _stage_cache_candidate_summary(candidate):
    region = getattr(candidate, "region", None)
    axes = (
        "n/a"
        if region is None
        else ",".join(str(getattr(axis.kind, "value", axis.kind)) for axis in region.axes)
    )
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
