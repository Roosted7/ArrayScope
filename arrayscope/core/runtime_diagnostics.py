"""Pure runtime diagnostics snapshots and formatting."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from arrayscope.core.cache_status import CacheDiagnosticsSnapshot
from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.memory_policy import MemoryPolicy, format_memory_policy
from arrayscope.core.resource_governor import ResourceGovernorDiagnostics, ResourcePressure


_TELEMETRY_ONLY_FEEDBACK_CHANNELS = frozenset(
    {
        "montage_cold_commit",
        "montage_layout_commit",
        "montage_present_total",
        "montage_priority_retarget",
        "tile_layer_commit",
    }
)


@dataclass(frozen=True)
class ImageUploadTiming:
    total_ms: float | None = None
    visible_upload_ms: float | None = None
    histogram_upload_ms: float | None = None
    histogram_bind_ms: float | None = None
    histogram_recompute_ms: float | None = None
    level_sync_ms: float | None = None
    rgb_window_ms: float | None = None
    tile_layer_upload_ms: float | None = None
    tile_layer_rgb_window_ms: float | None = None
    profile_bounds_ms: float | None = None
    visible_bytes: int = 0
    visible_pixels: int = 0
    histogram_bytes: int = 0
    histogram_pixels: int = 0
    fast_same_object: bool = False
    mode: str = ""
    tile_layer_visible_items: int = 0
    tile_layer_items_created: int = 0
    tile_layer_items_updated: int = 0
    tile_layer_items_skipped: int = 0
    tile_layer_rgb_window_tiles: int = 0
    tile_layer_image_replacements: int = 0
    tile_layer_existing_items_shown: int = 0
    tile_layer_relocated_tiles: int = 0
    tile_layer_resident_items: int = 0
    tile_layer_storage_capacity: int = 0
    tile_layer_storage_rebuilds: int = 0
    tile_layer_storage_evictions: int = 0
    tile_layer_texture_uploads: int = 0
    tile_layer_texture_upload_bytes: int = 0
    tile_layer_texture_prepare_ms: float | None = None
    tile_layer_texture_submit_ms: float | None = None
    tile_layer_vertex_uploads: int = 0
    tile_layer_level_updates: int = 0
    tile_layer_level_update_pending_items: int = 0
    tile_layer_estimated_gpu_bytes: int = 0
    tile_layer_cpu_shadow_bytes: int = 0
    tile_layer_page_count: int = 0
    tile_layer_active_pages: int = 0
    tile_layer_device_max_texture_size: int = 0
    tile_layer_budget_bytes: int = 0
    tile_layer_near_resident_items: int = 0
    tile_layer_warm_resident_items: int = 0
    tile_layer_evicted_near_items: int = 0
    tile_layer_capacity_warning: str = ""
    tile_layer_lod_level: int = 0
    tile_layer_lod_factor: int = 1
    tile_layer_source_texels_per_pixel: float = 0.0
    tile_layer_gutter_pixels: int = 0
    tile_layer_mipmap_updates: int = 0
    tile_layer_mipmap_available: bool = False
    tile_layer_complex_texture_uploads: int = 0
    tile_layer_shader_uniform_updates: int = 0
    tile_layer_lod_level_swaps_zero_upload: int = 0
    tile_layer_lod_level_swaps_with_upload: int = 0
    tile_layer_superseded_reclaimed_under_pressure: int = 0
    cpu_complex_prep_ms: float | None = None


@dataclass(frozen=True)
class MontageRuntimeDiagnostics:
    active: bool
    session_id: int | None = None
    loaded_tiles: int = 0
    loading_tiles: int = 0
    pending_tiles: int = 0
    pending_completed_tiles: int = 0
    pending_payload_upserts: int = 0
    pending_removals: int = 0
    pending_level_tiles: int = 0
    level_scan_remaining_tiles: int = 0
    skipped_tiles: int = 0
    visible_tiles: int = 0
    presented_tiles: int = 0
    overlay_count: int = 0
    attached_stage_requests: int = 0
    waiting_stage_requests: int = 0
    final_commit_pending: bool = False
    flush_pending: bool = False
    presentation_draw_count: int = 0
    tile_presentation_request_count: int = 0
    tile_presentation_draw_count: int = 0
    tile_presentation_draw_pending: bool = False
    tile_visual_visible_pages: int = 0
    overlays_above_tiles: bool = False
    display_mode: str = "none"
    backend_chosen: str = "none"
    backend_reason: str = ""
    backend_warning: str = ""
    show_loading_overlays: bool = False
    tile_lod_desired_factor: int = 1
    tile_lod_applied_factor: int = 1
    tile_lod_desired_factor_xy: tuple[int, int] = (1, 1)
    tile_lod_applied_factor_xy: tuple[int, int] = (1, 1)
    tile_lod_source_texels_per_pixel_xy: tuple[float, float] = (0.0, 0.0)
    tile_lod_policy: str = "native-only"
    tile_lod_reason: str = ""
    tile_lod_applied_level: int = 0
    # ((level, tile count), ...) over currently built display payloads.
    tile_lod_resident_tile_levels: tuple[tuple[int, int], ...] = ()
    tile_lod_pyramid_bytes: int = 0
    tile_lod_pyramid_entries: int = 0
    tile_lod_pyramid_hits: int = 0
    tile_lod_pyramid_misses: int = 0
    tile_lod_pyramid_evictions: int = 0
    tile_lod_pending_materializations: int = 0
    tile_lod_materializations_completed: int = 0
    tile_lod_ingest_reductions: int = 0
    # ADR 0050 zero-redundant-work counters: histogram/level recomputes caused
    # by display-LOD level swaps must stay 0; the reuse counters make the
    # avoided work observable in JSONL A/B traces.
    tile_lod_stats_cross_level_reuses: int = 0
    tile_lod_stats_recomputes: int = 0
    tile_lod_cross_level_reductions: int = 0
    tile_lod_pipeline_reruns_avoided: int = 0
    tile_lod_stage_hits_serving_derivations: int = 0
    tile_histogram_lod_swap_recomputes: int = 0
    tile_histogram_cross_level_reuses: int = 0
    tile_compute_cache_hits: int = 0
    tile_compute_stage_backed: int = 0
    tile_compute_direct: int = 0
    tile_compute_waiting_for_stage: int = 0
    tile_compute_stage_backed_ms: float = 0.0
    tile_compute_direct_ms: float = 0.0
    tile_compute_stage_backed_max_ms: float = 0.0
    tile_compute_direct_max_ms: float = 0.0
    lead_direct_tiles: int = 0
    stage_backed_tiles_pending: int = 0
    retained_stage_index: int | None = None
    retained_stage_decision: str = ""
    repeated_expensive_stage_per_tile: bool = False
    priority_retargeted_tiles: int = 0
    presented_order_sample: tuple[int, ...] = ()
    # ADR 0051: single-owner tile lifecycle machine.
    lifecycle_parked: int = 0
    lifecycle_evaluating: int = 0
    lifecycle_presented: int = 0
    lifecycle_dangling_claims: int = 0
    # Migration parity: legacy collections that disagree with the machine's
    # mirrored semantic axis (must trend to 0 before P2 deletes them).
    lifecycle_semantic_mismatches: int = 0
    # P2 identity-aware acknowledgement: acks the machine refused because the
    # backend slot held a different payload identity than emitted.  Nonzero =
    # a false-ack door tried to open and was closed structurally.
    lifecycle_identity_rejections: int = 0
    # Tiles whose payload is dirty (queued for re-presentation).
    dirty_payload_tiles: int = 0
    # Stall-watchdog rescues (ADR 0051): each one means every montage pump
    # was dead while work remained — a lost wakeup that must be root-caused;
    # the watchdog is a safety net, not the fix.  The signature captures what
    # was frozen: (session, pending, evaluating, active, dirty, upserts, lod).
    stall_repairs: int = 0
    last_stall_signature: tuple[int, ...] = ()
    # ADR 0051 rule 1 ground truth: drawn tiles whose backend-reported slot
    # identity differs from the session's current payload — nonzero at idle
    # means visibly stale tiles (the entire 2026-07-05 defect family).
    backend_stale_identities: int = 0


@dataclass(frozen=True)
class RenderRuntimeDiagnostics:
    last_decision_kind: str = ""
    last_decision_reason: str = ""
    last_context_summary: str = ""
    last_request_key: str = ""
    last_error: str = ""
    estimated_display_bytes: int | None = None
    render_budget_bytes: int | None = None


@dataclass(frozen=True)
class RenderCoalescerDiagnostics:
    pending: bool = False
    interactive_active: bool = False
    requested: int = 0
    flushed: int = 0
    coalesced: int = 0
    deferred_side_panel_refreshes: int = 0


@dataclass(frozen=True)
class RenderTimingDiagnostics:
    last_render_sync_ms: float | None = None
    last_control_sync_ms: float | None = None
    last_planning_ms: float | None = None
    last_worker_queue_wait_ms: float | None = None
    last_evaluation_ms: float | None = None
    last_display_commit_ms: float | None = None
    last_set_image_ms: float | None = None
    last_levels_histogram_ms: float | None = None
    last_operation_dock_ms: float | None = None
    last_inspection_refresh_ms: float | None = None


@dataclass(frozen=True)
class MontageTimingDiagnostics:
    last_viewport_plan_ms: float | None = None
    last_cache_resolve_ms: float | None = None
    last_stage_plan_ms: float | None = None
    last_session_setup_ms: float | None = None
    last_initial_commit_ms: float | None = None
    last_tile_eval_ms: float | None = None
    last_display_cache_lookup_ms: float | None = None
    last_display_cache_hit: bool | None = None
    last_stage_cache_lookup_ms: float | None = None
    last_stage_cache_hit: bool | None = None
    last_stage_attach_wait_ms: float | None = None
    last_level_stats_ms: float | None = None
    last_tile_payload_build_ms: float | None = None
    last_visible_upload_ms: float | None = None
    last_histogram_upload_ms: float | None = None
    last_histogram_recompute_ms: float | None = None
    last_rgb_window_ms: float | None = None
    last_tile_layer_upload_ms: float | None = None
    last_tile_layer_rgb_window_ms: float | None = None
    last_level_sync_ms: float | None = None
    last_tile_commit_ms: float | None = None
    last_set_image_ms: float | None = None
    last_overlay_update_ms: float | None = None
    cached_tiles_last_session: int = 0
    missing_tiles_last_session: int = 0
    committed_tile_upserts_last_flush: int = 0
    upload_visible_bytes: int = 0
    upload_histogram_bytes: int = 0
    upload_fast_same_object: bool = False
    tile_layer_visible_items: int = 0
    tile_layer_items_created: int = 0
    tile_layer_items_updated: int = 0
    tile_layer_items_skipped: int = 0
    tile_layer_rgb_window_tiles: int = 0
    tile_layer_image_replacements: int = 0
    tile_layer_existing_items_shown: int = 0
    tile_layer_relocated_tiles: int = 0
    tile_layer_resident_items: int = 0
    tile_layer_storage_capacity: int = 0
    tile_layer_storage_rebuilds: int = 0
    tile_layer_storage_evictions: int = 0
    tile_layer_texture_uploads: int = 0
    tile_layer_texture_upload_bytes: int = 0
    tile_layer_texture_prepare_ms: float | None = None
    tile_layer_texture_submit_ms: float | None = None
    tile_layer_vertex_uploads: int = 0
    tile_layer_level_updates: int = 0
    tile_layer_estimated_gpu_bytes: int = 0
    tile_layer_cpu_shadow_bytes: int = 0
    tile_layer_page_count: int = 0
    tile_layer_active_pages: int = 0
    tile_layer_device_max_texture_size: int = 0
    tile_layer_budget_bytes: int = 0
    tile_layer_near_resident_items: int = 0
    tile_layer_warm_resident_items: int = 0
    tile_layer_evicted_near_items: int = 0
    tile_layer_capacity_warning: str = ""
    tile_layer_lod_level: int = 0
    tile_layer_lod_factor: int = 1
    tile_layer_source_texels_per_pixel: float = 0.0
    tile_layer_gutter_pixels: int = 0
    tile_layer_mipmap_updates: int = 0
    tile_layer_mipmap_available: bool = False
    tile_layer_complex_texture_uploads: int = 0
    tile_layer_shader_uniform_updates: int = 0
    tile_layer_lod_level_swaps_zero_upload: int = 0
    tile_layer_lod_level_swaps_with_upload: int = 0
    tile_layer_superseded_reclaimed_under_pressure: int = 0
    cpu_complex_prep_ms: float | None = None
    coalesced_commits: int = 0

    @property
    def upload_total_bytes(self) -> int:
        return (
            int(self.upload_visible_bytes)
            + int(self.upload_histogram_bytes)
            + int(self.tile_layer_texture_upload_bytes)
        )


@dataclass(frozen=True)
class CanvasPreserveRuntimeDiagnostics:
    active: bool = False
    generation: int = 0
    mode: str = "best_effort"
    platform: str = ""
    last_transition: str = ""
    last_result: str = "none"
    target_canvas_size: tuple[int, int] | None = None
    final_canvas_size: tuple[int, int] | None = None
    final_window_size: tuple[int, int] | None = None
    last_delta: tuple[int, int] | None = None
    attempts_used: int = 0
    strong_used: bool = False
    strong_available: bool = False
    constraints_active: bool = False
    events: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowRuntimeDiagnostics:
    memory_policy: MemoryPolicy
    display_cache: CacheDiagnosticsSnapshot
    profile_cache: CacheDiagnosticsSnapshot
    stage_cache: object
    schedulers: tuple[object, ...]
    render: RenderRuntimeDiagnostics
    montage: MontageRuntimeDiagnostics
    canvas_preserve: CanvasPreserveRuntimeDiagnostics
    fft_backend_choice: str
    fft_backend_resolved: str
    fft_workers_choice: str
    fft_workers_resolved: int
    operation_count: int
    derived_shape: tuple[int, ...]
    derived_dtype: str
    pipeline_peak_bytes: int | None
    compute_worker_summaries: tuple[str, ...] = ()
    compute_fft_worker_summaries: tuple[str, ...] = ()
    pipeline_warnings: tuple[str, ...] = ()
    optimized_operation_count: int | None = None
    operation_optimization_summaries: tuple[str, ...] = ()
    capability_stage_count: int | None = None
    stage_cache_candidate_count: int | None = None
    stage_cache_candidate_summaries: tuple[str, ...] = ()
    operation_final_region: str = ""
    operation_required_input_region: str = ""
    operation_expanded_axes: tuple[int, ...] = ()
    operation_transition_summaries: tuple[str, ...] = ()
    render_timing: RenderTimingDiagnostics = field(default_factory=RenderTimingDiagnostics)
    montage_timing: MontageTimingDiagnostics = field(default_factory=MontageTimingDiagnostics)
    render_coalescer: RenderCoalescerDiagnostics = field(default_factory=RenderCoalescerDiagnostics)
    work_graph: object | None = None
    stage_materialization: object | None = None
    montage_prefetch: tuple[object, ...] = ()
    resource_governor: ResourceGovernorDiagnostics | None = None
    image_rendering_backend: str = "stable"
    image_rendering_backend_selected: str = "stable"
    image_rendering_backend_actual: str = "stable"


def format_runtime_diagnostics(snapshot: WindowRuntimeDiagnostics) -> str:
    return "\n\n".join(f"{title}\n{text}" for title, text in format_runtime_diagnostics_sections(snapshot).items())


def format_runtime_diagnostics_sections(snapshot: WindowRuntimeDiagnostics) -> dict[str, str]:
    sections = {
        "Realtime": "\n".join(_realtime_lines(snapshot)),
        "Montage": "\n".join(_montage_lines(snapshot)),
        "Render": "\n".join(_render_lines(snapshot)),
        "Feedback": "\n".join(_feedback_lines(snapshot.resource_governor)),
        "Work Graph": "\n".join(_work_graph_lines(snapshot.work_graph)),
        "Schedulers": "\n".join(_scheduler_lines(snapshot.schedulers)),
        "Caches": "\n".join(
            (
                _cache_line("Display tiles", snapshot.display_cache),
                _cache_line("Profiles/scalars", snapshot.profile_cache),
                _stage_cache_line("Stage cache", snapshot.stage_cache),
                _stage_materialization_line("Stage materialization", snapshot.stage_materialization),
            )
        ),
        "Memory": format_memory_policy(snapshot.memory_policy),
        "Compute": "\n".join(
            (
                "Workers: " + (", ".join(snapshot.compute_worker_summaries) or "n/a"),
                "FFT workers: " + (", ".join(snapshot.compute_fft_worker_summaries) or "n/a"),
            )
        ),
        "FFT": "\n".join(
            (
                f"Backend: {snapshot.fft_backend_choice} -> {snapshot.fft_backend_resolved}",
                f"Workers: {snapshot.fft_workers_choice} -> {snapshot.fft_workers_resolved}",
            )
        ),
        "Canvas Preserve": "\n".join(_canvas_preserve_lines(snapshot.canvas_preserve)),
        "Operations": "\n".join(_operation_lines(snapshot)),
    }
    return sections


def _realtime_lines(snapshot: WindowRuntimeDiagnostics) -> tuple[str, ...]:
    return (
        (
            "Bottleneck: "
            f"{runtime_bottleneck_text(snapshot)}"
        ),
        (
            "Feedback: "
            f"{_pressure_summary(snapshot.resource_governor)}"
        ),
        (
            "Renderer:\n"
            f"  image={snapshot.image_rendering_backend_actual} "
            f"setting={snapshot.image_rendering_backend_selected} "
            f"montage={snapshot.montage.backend_chosen}"
        ),
        (
            "Render:\n"
            f"  decision={snapshot.render.last_decision_kind or 'n/a'}\n"
            f"  control={_ms_text(snapshot.render_timing.last_control_sync_ms)} "
            f"eval={_ms_text(snapshot.render_timing.last_evaluation_ms)} "
            f"commit={_ms_text(snapshot.render_timing.last_display_commit_ms)} "
            f"sync={_ms_text(snapshot.render_timing.last_render_sync_ms)}"
        ),
        (
            "Montage:\n"
            f"  active={snapshot.montage.active} mode={snapshot.montage.display_mode}\n"
            f"  tiles visible={snapshot.montage.visible_tiles} loaded={snapshot.montage.loaded_tiles} "
            f"presented={snapshot.montage.presented_tiles} "
            f"pending={snapshot.montage.pending_tiles} "
            f"overlays={snapshot.montage.overlay_count}"
        ),
        (
            "Tile layer:\n"
            f"  visible={snapshot.montage_timing.tile_layer_visible_items} "
            f"resident={snapshot.montage_timing.tile_layer_resident_items}/"
            f"{snapshot.montage_timing.tile_layer_storage_capacity} "
            f"({_ratio_percent_text(snapshot.montage_timing.tile_layer_resident_items, snapshot.montage_timing.tile_layer_storage_capacity)}) "
            f"created={snapshot.montage_timing.tile_layer_items_created} "
            f"updated={snapshot.montage_timing.tile_layer_items_updated} "
            f"shown={snapshot.montage_timing.tile_layer_existing_items_shown} "
            f"moved={snapshot.montage_timing.tile_layer_relocated_tiles} "
            f"skipped={snapshot.montage_timing.tile_layer_items_skipped} "
            f"rgb_tiles={snapshot.montage_timing.tile_layer_rgb_window_tiles}\n"
            f"  rgb={_ms_text(snapshot.montage_timing.last_tile_layer_rgb_window_ms)} "
            f"upload={_ms_text(snapshot.montage_timing.last_tile_layer_upload_ms)} "
            f"gpu={format_bytes(snapshot.montage_timing.tile_layer_estimated_gpu_bytes)} "
            f"budget={format_bytes(snapshot.montage_timing.tile_layer_budget_bytes)} "
            f"({_ratio_percent_text(snapshot.montage_timing.tile_layer_estimated_gpu_bytes, snapshot.montage_timing.tile_layer_budget_bytes)}) "
            f"pages={snapshot.montage_timing.tile_layer_active_pages}/"
            f"{snapshot.montage_timing.tile_layer_page_count}"
        ),
        (
            "Upload: "
            f"total={format_bytes(snapshot.montage_timing.upload_total_bytes)} "
            f"visible={format_bytes(snapshot.montage_timing.upload_visible_bytes)} "
            f"histogram={format_bytes(snapshot.montage_timing.upload_histogram_bytes)} "
            f"tile_texture={format_bytes(snapshot.montage_timing.tile_layer_texture_upload_bytes)} "
            f"same object={snapshot.montage_timing.upload_fast_same_object}"
        ),
        (
            "Coalescer: "
            f"pending={snapshot.render_coalescer.pending}, "
            f"requested={snapshot.render_coalescer.requested}, "
            f"flushed={snapshot.render_coalescer.flushed}, "
            f"coalesced={snapshot.render_coalescer.coalesced}"
        ),
        _montage_prefetch_line("Montage prefetch", snapshot.montage_prefetch),
    )


def _feedback_lines(diagnostics: ResourceGovernorDiagnostics | None) -> tuple[str, ...]:
    if diagnostics is None:
        return ("n/a",)
    lines = [
        f"Pressure: {_pressure_summary(diagnostics)}",
        (
            "CPU: "
            f"system={_percent_text(diagnostics.system_cpu_percent)} "
            f"process={_percent_text(diagnostics.process_cpu_percent)} "
            f"load1={_float_or_na(diagnostics.load_average_1m)} "
            f"source={diagnostics.telemetry_source}"
        ),
    ]
    if diagnostics.lane_decisions:
        lines.append("Lane workers:")
        for decision in diagnostics.lane_decisions:
            reason = _compact_reason(decision.reason)
            suffix = "" if not reason else f" ({reason})"
            lines.append(f"  {decision.lane.value}: {decision.target_workers}/{decision.max_workers}{suffix}")
    if diagnostics.feedback_channels:
        lines.append("Channels:")
        inactive = []
        active_channel_lines = []
        telemetry_channel_lines = []
        for channel in diagnostics.feedback_channels:
            if (
                channel.elapsed_ewma_ms is None
                and channel.per_item_ewma_ms is None
                and float(channel.last_elapsed_ms) <= 0.0
            ):
                inactive.append(channel.channel)
                continue
            target = (
                telemetry_channel_lines
                if str(channel.channel) in _TELEMETRY_ONLY_FEEDBACK_CHANNELS
                else active_channel_lines
            )
            target.append(_feedback_channel_line(channel))
        lines.extend(active_channel_lines or ("  n/a",))
        if telemetry_channel_lines:
            lines.append("Telemetry-only:")
            lines.extend(telemetry_channel_lines)
        if inactive:
            lines.append("  Inactive:")
            lines.extend(f"    - {name}" for name in inactive)
    if diagnostics.ui_decisions:
        lines.append("UI decisions:")
        for decision in diagnostics.ui_decisions:
            reason = _compact_reason(decision.reason)
            suffix = "" if not reason else f" ({reason})"
            lines.append(
                f"  {decision.channel}: "
                f"batch={decision.batch_limit} "
                f"budget={decision.budget_ms:.1f} ms "
                f"interval={decision.interval_ms} ms "
                f"byte-cap={format_bytes(decision.byte_cap)}"
                f"{suffix}"
            )
    if diagnostics.recent_over_warning_callbacks:
        lines.append("Callbacks over warning:")
        for callback in diagnostics.recent_over_warning_callbacks[-8:]:
            label = callback.channel
            details = []
            if callback.work_class:
                details.append(f"class={callback.work_class}")
            if callback.backend:
                details.append(f"backend={callback.backend}")
            details.append(f"elapsed={_ms_text(callback.elapsed_ms)}")
            details.append(f"items={callback.processed_items}")
            details.append(f"bytes={format_bytes(callback.processed_bytes)}")
            lines.append(f"  {label}: " + " ".join(details))
    return tuple(lines)


def _compact_reason(reason: str) -> str:
    reason = str(reason or "").strip()
    if not reason or reason == "profile baseline":
        return ""
    lower = reason.lower()
    if "high ui" in lower:
        return "high UI"
    if "elevated ui" in lower:
        return "elevated UI"
    if "prefetch kept narrow" in lower:
        return "narrow"
    if "backlog" in lower:
        return "backlog"
    if "memory" in lower:
        return "memory"
    if "headroom" in lower:
        return "CPU"
    return _short_debug_text(reason, limit=28)


def _pressure_summary(diagnostics: ResourceGovernorDiagnostics | None) -> str:
    if diagnostics is None:
        return "n/a"
    pressure = diagnostics.pressure
    ui_source = _ui_pressure_source(diagnostics)
    ui_text = pressure.ui_pressure.value if ui_source is None else f"{pressure.ui_pressure.value}({ui_source})"
    return (
        f"ui={ui_text} "
        f"cpu_headroom={pressure.cpu_headroom:.0%} "
        f"memory={pressure.memory_pressure.value} "
        f"cache={pressure.cache_pressure.value}"
    )


def _ui_pressure_source(diagnostics: ResourceGovernorDiagnostics) -> str | None:
    if diagnostics.pressure.ui_pressure not in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
        return None
    channels = tuple(
        channel
        for channel in diagnostics.feedback_channels
        if channel.elapsed_ewma_ms is not None and float(channel.elapsed_ewma_ms) > 0.0
        and str(channel.channel) not in _TELEMETRY_ONLY_FEEDBACK_CHANNELS
    )
    if not channels:
        return None
    channel = max(channels, key=lambda item: float(item.elapsed_ewma_ms or 0.0))
    return str(channel.channel)


def _feedback_channel_line(channel) -> str:
    return (
        f"  {channel.channel}:\n"
        f"    last={_ms_text(channel.last_elapsed_ms)} "
        f"ewma={_ms_text(channel.elapsed_ewma_ms)} "
        f"per-item={_ms_text(channel.per_item_ewma_ms)} "
        f"bytes={format_bytes(channel.last_byte_count)}\n"
        f"    batch={channel.batch_limit} "
        f"budget={channel.budget_ms:.1f} ms "
        f"interval={channel.interval_ms} ms"
    )


def runtime_bottleneck_text(snapshot: WindowRuntimeDiagnostics) -> str:
    """Return the current bottleneck cause from live work, not old timings."""

    active_work = runtime_has_live_work(snapshot)
    governor = snapshot.resource_governor
    if active_work and governor is not None:
        if governor.pressure.ui_pressure in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
            return "UI fan-in"
        if governor.pressure.memory_pressure in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
            return "memory"
    if active_work and snapshot.montage.active and snapshot.montage.tile_compute_waiting_for_stage:
        return "stage compute"
    if active_work and snapshot.montage_timing.tile_layer_rgb_window_tiles:
        return "RGB window/upload"
    if active_work and snapshot.montage.pending_tiles:
        return "tile compute"
    return "idle"


def runtime_has_live_work(snapshot: WindowRuntimeDiagnostics) -> bool:
    """Return whether the snapshot contains work that can still change the view."""

    coalescer = snapshot.render_coalescer
    if bool(getattr(coalescer, "pending", False)):
        return True
    work_graph = getattr(snapshot, "work_graph", None)
    if work_graph is not None and (
        int(getattr(work_graph, "active", 0) or 0)
        or int(getattr(work_graph, "queued", 0) or 0)
        or int(getattr(work_graph, "visible_backlog", 0) or 0)
    ):
        return True
    if any(
        int(getattr(scheduler, name, 0) or 0)
        for scheduler in tuple(snapshot.schedulers or ())
        for name in ("pending", "running", "queued")
    ):
        return True
    montage = snapshot.montage
    return bool(
        int(getattr(montage, "pending_tiles", 0) or 0)
        or int(getattr(montage, "pending_completed_tiles", 0) or 0)
        or int(getattr(montage, "pending_payload_upserts", 0) or 0)
        or int(getattr(montage, "pending_removals", 0) or 0)
        or int(getattr(montage, "loading_tiles", 0) or 0)
        or int(getattr(montage, "pending_level_tiles", 0) or 0)
        or int(getattr(montage, "level_scan_remaining_tiles", 0) or 0)
        or int(getattr(montage, "attached_stage_requests", 0) or 0)
        or int(getattr(montage, "waiting_stage_requests", 0) or 0)
        or bool(getattr(montage, "final_commit_pending", False))
        or bool(getattr(montage, "flush_pending", False))
        or bool(getattr(montage, "tile_presentation_draw_pending", False))
    )


def _field_default(field_obj):
    if field_obj.default is not dataclasses.MISSING:
        return field_obj.default
    if field_obj.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field_obj.default_factory()  # type: ignore[misc]
    return dataclasses.MISSING


def _compact_field_value(name: str, value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int) and "bytes" in name:
        return format_bytes(value)
    if isinstance(value, (tuple, list)):
        inner = ",".join(_short_debug_text(item, limit=24) for item in tuple(value)[:6])
        suffix = ",..." if len(tuple(value)) > 6 else ""
        return f"({inner}{suffix})"
    return _short_debug_text(value, limit=48)


def _auto_extra_lines(obj, covered: frozenset[str], *, heading: str = "More", width: int = 104) -> tuple[str, ...]:
    """Compact ``name=value`` dump of every dataclass field not curated above.

    Guarantee behind the diagnostics dialog contract: a field added to a
    diagnostics dataclass is visible in its tab (and the All tab) without
    touching the formatter — curated lines show the important values at the
    top, this block catches everything else.  Fields still at their default
    are hidden to keep the dump compact.
    """

    pairs: list[str] = []
    for field_obj in dataclasses.fields(obj):
        if field_obj.name in covered:
            continue
        value = getattr(obj, field_obj.name)
        default = _field_default(field_obj)
        if default is not dataclasses.MISSING and value == default:
            continue
        pairs.append(f"{field_obj.name}={_compact_field_value(field_obj.name, value)}")
    if not pairs:
        return ()
    lines: list[str] = []
    current = "  "
    for pair in pairs:
        if len(current) + len(pair) > width and current.strip():
            lines.append(current.rstrip())
            current = "  "
        current += pair + " "
    if current.strip():
        lines.append(current.rstrip())
    return (f"{heading} (non-default):", *lines)


def _ms_group(label: str, items) -> str:
    """One compact timing line; ``None`` entries are hidden (no n/a noise)."""

    parts = [f"{name}={float(value):.1f}" for name, value in items if value is not None]
    return f"{label} (ms): " + (" ".join(parts) if parts else "n/a")


# Fields rendered by the curated lines below.  Everything else in the
# dataclass is emitted by _auto_extra_lines, so nothing can be invisible;
# tests/core/test_runtime_diagnostics.py pins both properties.
_RENDER_COVERED = frozenset(
    {
        "last_decision_kind",
        "last_decision_reason",
        "last_context_summary",
        "last_request_key",
        "last_error",
    }
)
_RENDER_TIMING_COVERED = frozenset(
    {
        "last_render_sync_ms",
        "last_control_sync_ms",
        "last_planning_ms",
        "last_worker_queue_wait_ms",
        "last_evaluation_ms",
        "last_display_commit_ms",
        "last_set_image_ms",
        "last_levels_histogram_ms",
        "last_operation_dock_ms",
        "last_inspection_refresh_ms",
    }
)
_COALESCER_COVERED = frozenset(
    {
        "pending",
        "interactive_active",
        "requested",
        "flushed",
        "coalesced",
        "deferred_side_panel_refreshes",
    }
)


def _render_lines(snapshot: WindowRuntimeDiagnostics) -> tuple[str, ...]:
    timing = snapshot.render_timing
    error_line = (f"Error: {snapshot.render.last_error}",) if snapshot.render.last_error else ()
    return (
        f"Decision: {snapshot.render.last_decision_kind or 'n/a'} | {snapshot.render.last_decision_reason or 'n/a'}",
        _ms_group(
            "Timing",
            (
                ("sync", timing.last_render_sync_ms),
                ("control", timing.last_control_sync_ms),
                ("plan", timing.last_planning_ms),
                ("queue_wait", timing.last_worker_queue_wait_ms),
                ("eval", timing.last_evaluation_ms),
                ("commit", timing.last_display_commit_ms),
                ("set_image", timing.last_set_image_ms),
                ("levels_hist", timing.last_levels_histogram_ms),
                ("dock", timing.last_operation_dock_ms),
                ("inspect", timing.last_inspection_refresh_ms),
            ),
        ),
        (
            "Coalescer: "
            f"pending={_bool_text(snapshot.render_coalescer.pending)} "
            f"interactive={_bool_text(snapshot.render_coalescer.interactive_active)} "
            f"requested={snapshot.render_coalescer.requested} "
            f"flushed={snapshot.render_coalescer.flushed} "
            f"coalesced={snapshot.render_coalescer.coalesced} "
            f"deferred_refreshes={snapshot.render_coalescer.deferred_side_panel_refreshes}"
        ),
        *error_line,
        f"Request: {_request_text(snapshot.render.last_request_key)}",
        f"Context:\n{_wrapped_debug_text(snapshot.render.last_context_summary or 'n/a', indent='  ')}",
        *_auto_extra_lines(snapshot.render, _RENDER_COVERED, heading="More render"),
        *_auto_extra_lines(snapshot.render_timing, _RENDER_TIMING_COVERED, heading="More timing"),
        *_auto_extra_lines(snapshot.render_coalescer, _COALESCER_COVERED, heading="More coalescer"),
    )


_CANVAS_PRESERVE_COVERED = frozenset(
    {
        "mode",
        "platform",
        "active",
        "generation",
        "last_transition",
        "last_result",
        "target_canvas_size",
        "final_canvas_size",
        "final_window_size",
        "last_delta",
        "attempts_used",
        "strong_used",
        "strong_available",
        "constraints_active",
        "events",
    }
)


def _canvas_preserve_lines(canvas_preserve: CanvasPreserveRuntimeDiagnostics) -> tuple[str, ...]:
    events = tuple(canvas_preserve.events)
    recent_events = ("Recent events:", *(f"  {event}" for event in events)) if events else ("Recent events: n/a",)
    return (
        (
            f"Mode: {canvas_preserve.mode} platform={canvas_preserve.platform or 'n/a'} "
            f"active={_bool_text(canvas_preserve.active)} generation={canvas_preserve.generation}"
        ),
        f"Last: transition={canvas_preserve.last_transition or 'n/a'} result={canvas_preserve.last_result or 'n/a'}",
        (
            f"Canvas: target={_size_text(canvas_preserve.target_canvas_size)} "
            f"final={_size_text(canvas_preserve.final_canvas_size)} "
            f"window={_size_text(canvas_preserve.final_window_size)} "
            f"delta={_size_text(canvas_preserve.last_delta)}"
        ),
        (
            f"Attempts: {canvas_preserve.attempts_used} "
            f"strong={_bool_text(canvas_preserve.strong_used)}"
            f"/{_bool_text(canvas_preserve.strong_available)} "
            f"constraints={_bool_text(canvas_preserve.constraints_active)}"
        ),
        *recent_events,
        *_auto_extra_lines(canvas_preserve, _CANVAS_PRESERVE_COVERED, heading="More canvas"),
    )


_MONTAGE_COVERED = frozenset(
    {
        "active",
        "session_id",
        "display_mode",
        "backend_chosen",
        "backend_reason",
        "backend_warning",
        "show_loading_overlays",
        "visible_tiles",
        "loaded_tiles",
        "presented_tiles",
        "loading_tiles",
        "pending_tiles",
        "pending_level_tiles",
        "skipped_tiles",
        "lifecycle_presented",
        "lifecycle_parked",
        "lifecycle_evaluating",
        "lifecycle_dangling_claims",
        "lifecycle_semantic_mismatches",
        "lifecycle_identity_rejections",
        "dirty_payload_tiles",
        "backend_stale_identities",
        "pending_completed_tiles",
        "pending_payload_upserts",
        "pending_removals",
        "level_scan_remaining_tiles",
        "flush_pending",
        "final_commit_pending",
        "tile_lod_policy",
        "tile_lod_desired_factor",
        "tile_lod_desired_factor_xy",
        "tile_lod_applied_factor",
        "tile_lod_applied_factor_xy",
        "tile_lod_applied_level",
        "tile_lod_source_texels_per_pixel_xy",
        "tile_lod_reason",
        "tile_lod_resident_tile_levels",
        "tile_lod_pyramid_bytes",
        "tile_lod_pyramid_entries",
        "tile_lod_pyramid_hits",
        "tile_lod_pyramid_misses",
        "tile_lod_pyramid_evictions",
        "tile_lod_pending_materializations",
        "tile_lod_materializations_completed",
        "tile_lod_ingest_reductions",
        "tile_lod_stats_cross_level_reuses",
        "tile_lod_stats_recomputes",
        "tile_lod_cross_level_reductions",
        "tile_lod_pipeline_reruns_avoided",
        "tile_lod_stage_hits_serving_derivations",
        "tile_histogram_lod_swap_recomputes",
        "tile_histogram_cross_level_reuses",
        "overlay_count",
        "presentation_draw_count",
        "tile_presentation_draw_count",
        "tile_presentation_request_count",
        "tile_presentation_draw_pending",
        "tile_visual_visible_pages",
        "overlays_above_tiles",
        "attached_stage_requests",
        "waiting_stage_requests",
        "retained_stage_index",
        "retained_stage_decision",
        "repeated_expensive_stage_per_tile",
        "tile_compute_cache_hits",
        "tile_compute_stage_backed",
        "tile_compute_direct",
        "tile_compute_waiting_for_stage",
        "tile_compute_stage_backed_ms",
        "tile_compute_direct_ms",
        "tile_compute_stage_backed_max_ms",
        "tile_compute_direct_max_ms",
        "lead_direct_tiles",
        "stage_backed_tiles_pending",
    }
)
_MONTAGE_TIMING_COVERED = frozenset(
    {
        "last_viewport_plan_ms",
        "last_cache_resolve_ms",
        "last_stage_plan_ms",
        "last_session_setup_ms",
        "last_initial_commit_ms",
        "last_tile_eval_ms",
        "last_display_cache_lookup_ms",
        "last_display_cache_hit",
        "last_stage_cache_lookup_ms",
        "last_stage_cache_hit",
        "last_stage_attach_wait_ms",
        "last_level_stats_ms",
        "last_tile_payload_build_ms",
        "last_visible_upload_ms",
        "last_histogram_upload_ms",
        "last_histogram_recompute_ms",
        "last_rgb_window_ms",
        "last_tile_layer_upload_ms",
        "last_tile_layer_rgb_window_ms",
        "last_level_sync_ms",
        "last_tile_commit_ms",
        "last_set_image_ms",
        "last_overlay_update_ms",
        "cached_tiles_last_session",
        "missing_tiles_last_session",
        "committed_tile_upserts_last_flush",
        "coalesced_commits",
        "upload_visible_bytes",
        "upload_histogram_bytes",
        "upload_fast_same_object",
        "tile_layer_visible_items",
        "tile_layer_items_created",
        "tile_layer_items_updated",
        "tile_layer_items_skipped",
        "tile_layer_rgb_window_tiles",
        "tile_layer_existing_items_shown",
        "tile_layer_relocated_tiles",
        "tile_layer_resident_items",
        "tile_layer_storage_capacity",
        "tile_layer_storage_rebuilds",
        "tile_layer_storage_evictions",
        "tile_layer_texture_uploads",
        "tile_layer_texture_upload_bytes",
        "tile_layer_vertex_uploads",
        "tile_layer_level_updates",
        "tile_layer_shader_uniform_updates",
        "tile_layer_complex_texture_uploads",
        "tile_layer_estimated_gpu_bytes",
        "tile_layer_budget_bytes",
        "tile_layer_cpu_shadow_bytes",
        "tile_layer_page_count",
        "tile_layer_active_pages",
        "tile_layer_device_max_texture_size",
        "tile_layer_near_resident_items",
        "tile_layer_warm_resident_items",
        "tile_layer_capacity_warning",
        "tile_layer_lod_level",
        "tile_layer_lod_factor",
        "tile_layer_source_texels_per_pixel",
        "tile_layer_gutter_pixels",
        "tile_layer_mipmap_available",
        "tile_layer_mipmap_updates",
        "tile_layer_lod_level_swaps_zero_upload",
        "tile_layer_lod_level_swaps_with_upload",
        "tile_layer_superseded_reclaimed_under_pressure",
    }
)


def _montage_lines(snapshot: WindowRuntimeDiagnostics) -> tuple[str, ...]:
    montage = snapshot.montage
    timing = snapshot.montage_timing
    session_text = montage.session_id if montage.session_id is not None else "n/a"
    warning_lines = tuple(
        f"WARNING: {text}"
        for text in (montage.backend_warning, timing.tile_layer_capacity_warning)
        if text
    )
    return (
        # -- state that explains what the user sees, most important first --
        (
            "Tiles: "
            f"visible={montage.visible_tiles} loaded={montage.loaded_tiles} "
            f"presented={montage.presented_tiles} loading={montage.loading_tiles} "
            f"pending={montage.pending_tiles} pending_lvls={montage.pending_level_tiles} "
            f"skipped={montage.skipped_tiles}"
        ),
        (
            "Lifecycle: "
            f"presented={montage.lifecycle_presented} parked={montage.lifecycle_parked} "
            f"evaluating={montage.lifecycle_evaluating} "
            f"dangling_claims={montage.lifecycle_dangling_claims} "
            f"mismatches={montage.lifecycle_semantic_mismatches} "
            f"identity_rejections={montage.lifecycle_identity_rejections} "
            f"dirty={montage.dirty_payload_tiles} "
            f"BACKEND_STALE={montage.backend_stale_identities}"
        ),
        (
            "LOD: "
            f"{montage.tile_lod_policy} level={montage.tile_lod_applied_level} "
            f"desired={montage.tile_lod_desired_factor}{montage.tile_lod_desired_factor_xy} "
            f"applied={montage.tile_lod_applied_factor}{montage.tile_lod_applied_factor_xy} "
            f"texpp=({montage.tile_lod_source_texels_per_pixel_xy[0]:.2f},"
            f"{montage.tile_lod_source_texels_per_pixel_xy[1]:.2f})"
        ),
        f"LOD reason: {montage.tile_lod_reason or 'n/a'}",
        (
            "LOD residency: "
            f"tile_levels={montage.tile_lod_resident_tile_levels} "
            f"pyramid={format_bytes(montage.tile_lod_pyramid_bytes)}/{montage.tile_lod_pyramid_entries}e "
            f"hit/miss/evict={montage.tile_lod_pyramid_hits}/{montage.tile_lod_pyramid_misses}/"
            f"{montage.tile_lod_pyramid_evictions} "
            f"pending={montage.tile_lod_pending_materializations} "
            f"completed={montage.tile_lod_materializations_completed} "
            f"ingest={montage.tile_lod_ingest_reductions}"
        ),
        *warning_lines,
        (
            "Queues: "
            f"completed={montage.pending_completed_tiles} upserts={montage.pending_payload_upserts} "
            f"removals={montage.pending_removals} level_scan={montage.level_scan_remaining_tiles} "
            f"flush={_bool_text(montage.flush_pending)} final={_bool_text(montage.final_commit_pending)}"
        ),
        (
            "Session: "
            f"{session_text} active={_bool_text(montage.active)} mode={montage.display_mode} "
            f"backend={montage.backend_chosen} loading_overlays={_bool_text(montage.show_loading_overlays)}"
        ),
        f"Backend reason: {montage.backend_reason or 'n/a'}",
        (
            "Presentation: "
            f"draws={montage.presentation_draw_count} "
            f"tile_draw={montage.tile_presentation_draw_count}/{montage.tile_presentation_request_count} "
            f"pending={_bool_text(montage.tile_presentation_draw_pending)} "
            f"pages={montage.tile_visual_visible_pages} overlays={montage.overlay_count} "
            f"above_tiles={_bool_text(montage.overlays_above_tiles)}"
        ),
        (
            "Compute: "
            f"cache_hit={montage.tile_compute_cache_hits} stage_backed={montage.tile_compute_stage_backed} "
            f"direct={montage.tile_compute_direct} waiting_stage={montage.tile_compute_waiting_for_stage} "
            f"lead_direct={montage.lead_direct_tiles} stage_pending={montage.stage_backed_tiles_pending} "
            f"stage_waits={montage.attached_stage_requests}/{montage.waiting_stage_requests}"
        ),
        (
            "Compute time (ms): "
            f"stage_backed={montage.tile_compute_stage_backed_ms:.1f}"
            f"/max {montage.tile_compute_stage_backed_max_ms:.1f} "
            f"direct={montage.tile_compute_direct_ms:.1f}/max {montage.tile_compute_direct_max_ms:.1f}"
        ),
        (
            "Reusable stage: "
            f"stage={montage.retained_stage_index if montage.retained_stage_index is not None else 'n/a'} "
            f"{montage.retained_stage_decision or 'n/a'} "
            f"repeated_per_tile={_bool_text(montage.repeated_expensive_stage_per_tile)}"
        ),
        (
            "LOD reuse: "
            f"stats_reused={montage.tile_lod_stats_cross_level_reuses} "
            f"stats_recomputed={montage.tile_lod_stats_recomputes} "
            f"level_from_level={montage.tile_lod_cross_level_reductions} "
            f"reruns_avoided={montage.tile_lod_pipeline_reruns_avoided} "
            f"stage_hits={montage.tile_lod_stage_hits_serving_derivations} "
            f"hist_recomputes={montage.tile_histogram_lod_swap_recomputes} "
            f"hist_reuses={montage.tile_histogram_cross_level_reuses}"
        ),
        # -- timings, grouped; n/a entries hidden --
        _ms_group(
            "Plan",
            (
                ("viewport", timing.last_viewport_plan_ms),
                ("cache_resolve", timing.last_cache_resolve_ms),
                ("stage_plan", timing.last_stage_plan_ms),
                ("setup", timing.last_session_setup_ms),
                ("first_commit", timing.last_initial_commit_ms),
            ),
        ),
        _ms_group(
            "Evaluate",
            (
                ("tile", timing.last_tile_eval_ms),
                ("cache_lookup", timing.last_display_cache_lookup_ms),
                ("stage_lookup", timing.last_stage_cache_lookup_ms),
                ("attach_wait", timing.last_stage_attach_wait_ms),
                ("levels", timing.last_level_stats_ms),
                ("payload", timing.last_tile_payload_build_ms),
            ),
        )
        + (
            f" | cache_hit={_bool_text(timing.last_display_cache_hit)}"
            f" stage_hit={_bool_text(timing.last_stage_cache_hit)}"
        ),
        _ms_group(
            "Present",
            (
                ("visible", timing.last_visible_upload_ms),
                ("hist", timing.last_histogram_upload_ms),
                ("hist_recompute", timing.last_histogram_recompute_ms),
                ("rgb", timing.last_rgb_window_ms),
                ("tile_upload", timing.last_tile_layer_upload_ms),
                ("tile_rgb", timing.last_tile_layer_rgb_window_ms),
                ("level_sync", timing.last_level_sync_ms),
                ("commit", timing.last_tile_commit_ms),
                ("set_image", timing.last_set_image_ms),
                ("overlay", timing.last_overlay_update_ms),
            ),
        ),
        (
            "Flush: "
            f"upserts_last={timing.committed_tile_upserts_last_flush} "
            f"coalesced={timing.coalesced_commits} "
            f"cache_session={timing.cached_tiles_last_session}/{timing.missing_tiles_last_session} (hit/miss)"
        ),
        # -- tile layer (GPU backend) --
        (
            "Layer items: "
            f"visible={timing.tile_layer_visible_items} "
            f"resident={timing.tile_layer_resident_items}/{timing.tile_layer_storage_capacity} "
            f"({_ratio_percent_text(timing.tile_layer_resident_items, timing.tile_layer_storage_capacity)}) "
            f"created={timing.tile_layer_items_created} updated={timing.tile_layer_items_updated} "
            f"shown={timing.tile_layer_existing_items_shown} moved={timing.tile_layer_relocated_tiles} "
            f"skipped={timing.tile_layer_items_skipped} rgb={timing.tile_layer_rgb_window_tiles}"
        ),
        (
            "Layer storage: "
            f"gpu={format_bytes(timing.tile_layer_estimated_gpu_bytes)}/"
            f"{format_bytes(timing.tile_layer_budget_bytes)} "
            f"({_ratio_percent_text(timing.tile_layer_estimated_gpu_bytes, timing.tile_layer_budget_bytes)}) "
            f"pages={timing.tile_layer_active_pages}/{timing.tile_layer_page_count} "
            f"near={timing.tile_layer_near_resident_items} warm={timing.tile_layer_warm_resident_items} "
            f"rebuilds={timing.tile_layer_storage_rebuilds} evictions={timing.tile_layer_storage_evictions} "
            f"max_tex={timing.tile_layer_device_max_texture_size or 'n/a'} "
            f"shadow={format_bytes(timing.tile_layer_cpu_shadow_bytes)}"
        ),
        (
            "Layer submissions: "
            f"textures={timing.tile_layer_texture_uploads} "
            f"bytes={format_bytes(timing.tile_layer_texture_upload_bytes)} "
            f"vertices={timing.tile_layer_vertex_uploads} levels={timing.tile_layer_level_updates} "
            f"uniforms={timing.tile_layer_shader_uniform_updates} "
            f"complex={timing.tile_layer_complex_texture_uploads}"
        ),
        (
            "Layer LOD: "
            f"level={timing.tile_layer_lod_level} factor={timing.tile_layer_lod_factor} "
            f"texpp={timing.tile_layer_source_texels_per_pixel:.2f} "
            f"gutter={timing.tile_layer_gutter_pixels} "
            f"mipmap={_bool_text(timing.tile_layer_mipmap_available)}"
            f"/{timing.tile_layer_mipmap_updates} "
            f"swaps={timing.tile_layer_lod_level_swaps_zero_upload}z/"
            f"{timing.tile_layer_lod_level_swaps_with_upload}u "
            f"reclaimed={timing.tile_layer_superseded_reclaimed_under_pressure}"
        ),
        (
            "Upload: "
            f"total={format_bytes(timing.upload_total_bytes)} "
            f"visible={format_bytes(timing.upload_visible_bytes)} "
            f"hist={format_bytes(timing.upload_histogram_bytes)} "
            f"tile_tex={format_bytes(timing.tile_layer_texture_upload_bytes)} "
            f"same_object={_bool_text(timing.upload_fast_same_object)}"
        ),
        _montage_prefetch_line("Prefetch", snapshot.montage_prefetch),
        *_auto_extra_lines(montage, _MONTAGE_COVERED, heading="More montage"),
        *_auto_extra_lines(timing, _MONTAGE_TIMING_COVERED, heading="More layer/timing"),
    )


def _operation_lines(snapshot: WindowRuntimeDiagnostics) -> tuple[str, ...]:
    optimization_lines = tuple(f"  {_short_debug_text(summary, limit=140)}" for summary in snapshot.operation_optimization_summaries)
    transition_lines = tuple(f"  {_short_debug_text(transition, limit=160)}" for transition in snapshot.operation_transition_summaries)
    candidate_lines = tuple(f"  {_short_debug_text(candidate, limit=160)}" for candidate in snapshot.stage_cache_candidate_summaries)
    warning_lines = tuple(f"Warning: {warning}" for warning in snapshot.pipeline_warnings)
    stage_cache_lines = _stage_cache_operation_lines(snapshot.stage_cache)
    return (
        f"Count: {snapshot.operation_count}",
        f"Optimized count: {'n/a' if snapshot.optimized_operation_count is None else snapshot.optimized_operation_count}",
        f"Derived: {snapshot.derived_shape} {snapshot.derived_dtype}",
        f"Pipeline peak: {'n/a' if snapshot.pipeline_peak_bytes is None else format_bytes(snapshot.pipeline_peak_bytes)}",
        f"Final region: {_short_debug_text(snapshot.operation_final_region or 'n/a')}",
        f"Required input: {_short_debug_text(snapshot.operation_required_input_region or 'n/a')}",
        f"Expanded axes: {_axes_text(snapshot.operation_expanded_axes)}",
        *(warning_lines or ()),
        "Optimizations:",
        *(optimization_lines or ("  n/a",)),
        "Transitions:",
        *(transition_lines or ("  n/a",)),
        f"Capability stages: {'n/a' if snapshot.capability_stage_count is None else snapshot.capability_stage_count}",
        f"Stage cache candidates: {'n/a' if snapshot.stage_cache_candidate_count is None else snapshot.stage_cache_candidate_count}",
        *(("Candidates:",) if candidate_lines else ()),
        *candidate_lines,
        *(("Stage cache recent:",) if stage_cache_lines else ()),
        *stage_cache_lines,
    )


def _cache_line(name: str, cache: CacheDiagnosticsSnapshot) -> str:
    hit_rate = "n/a" if cache.hit_rate is None else f"{cache.hit_rate:.0%}"
    return (
        f"{name}: {cache.status.value}, entries={cache.entries}, "
        f"bytes={format_bytes(cache.bytes_used)} / {format_bytes(cache.max_bytes)}, "
        f"hits={cache.hits}, misses={cache.misses}, evictions={cache.evictions}, hit-rate={hit_rate}"
    )


def _stage_cache_line(name: str, cache) -> str:
    hit_rate = "n/a" if cache.hit_rate is None else f"{cache.hit_rate:.0%}"
    return (
        f"{name}: entries={cache.entries}, "
        f"bytes={format_bytes(cache.bytes_used)} / {format_bytes(cache.max_bytes)}, "
        f"hits={cache.hits}, misses={cache.misses}, evictions={cache.evictions}, hit-rate={hit_rate}, "
        f"candidates={cache.candidates_seen}, stores={cache.stores}, refused={cache.refused_over_budget}"
    )


def _stage_materialization_line(name: str, diagnostics) -> str:
    if diagnostics is None:
        return f"{name}: n/a"
    candidate = getattr(diagnostics, "candidate_bytes", None)
    candidate_text = "unknown" if candidate is None else format_bytes(int(candidate))
    return (
        f"{name}: decision={getattr(diagnostics, 'decision', '') or 'n/a'}, "
        f"candidate={candidate_text}, budget={format_bytes(int(getattr(diagnostics, 'budget_bytes', 0)))}, "
        f"in-flight={getattr(diagnostics, 'in_flight', 0)}, scheduled={getattr(diagnostics, 'scheduled', 0)}, "
        f"attached={getattr(diagnostics, 'attached', 0)}, completed={getattr(diagnostics, 'completed', 0)}, "
        f"refused={getattr(diagnostics, 'refused', 0)}, consequence={getattr(diagnostics, 'consequence', '') or 'n/a'}"
    )



def _montage_prefetch_line(name: str, decisions: tuple[object, ...]) -> str:
    if not decisions:
        return f"{name}: n/a"
    parts = []
    for decision in decisions[:4]:
        tile = getattr(decision, "tile_number", None)
        source = getattr(decision, "source_index", None)
        label = getattr(decision, "decision", "") or "n/a"
        reason = getattr(decision, "reason", "") or ""
        tile_text = "n/a" if tile is None else str(int(tile))
        source_text = "n/a" if source is None else str(int(source))
        parts.append(f"tile={tile_text} source={source_text} decision={label}" + (f" reason={reason}" if reason else ""))
    return f"{name}: " + "; ".join(parts)


def _stage_cache_operation_lines(cache) -> tuple[str, ...]:
    lines = []
    for label, value in (
        ("Stage cache last hit", getattr(cache, "last_hit", "")),
        ("Stage cache last miss", getattr(cache, "last_miss", "")),
        ("Stage cache last store", getattr(cache, "last_store", "")),
        ("Stage cache last refused", getattr(cache, "last_refused", "")),
    ):
        if value:
            lines.append(f"{label}: {value}")
    return tuple(lines)


def _size_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "n/a"
    return f"{int(size[0])}x{int(size[1])}"


def _bytes_or_na(value: int | None) -> str:
    return "n/a" if value is None else format_bytes(int(value))


def _short_debug_text(value: object, *, limit: int = 220) -> str:
    text = str(value)
    if "b'" in text or 'b"' in text:
        return f"captured ({len(text)} chars; payload hidden)"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 24)] + f"... ({len(text)} chars)"


def _wrapped_debug_text(value: object, *, indent: str = "", limit: int = 120) -> str:
    text = _short_debug_text(value, limit=max(limit * 3, limit))
    if len(text) <= limit:
        return indent + text
    words = text.split(" ")
    lines = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= limit:
            current += " " + word
        else:
            lines.append(indent + current)
            current = word
    if current:
        lines.append(indent + current)
    return "\n".join(lines)


def _request_text(value: str) -> str:
    if not value:
        return "n/a"
    text = str(value)
    if len(text) <= 220 and "b'" not in text and 'b"' not in text:
        return text
    return f"captured ({len(text)} chars; payload hidden)"


def _axes_text(axes: tuple[int, ...]) -> str:
    return "n/a" if not axes else ",".join(str(int(axis)) for axis in axes)


def _ms_text(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f} ms"


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _percent_text(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.0f}%"


def _ratio_percent_text(used: int | float | None, total: int | float | None) -> str:
    if used is None or total is None:
        return "n/a"
    total_float = float(total)
    if total_float <= 0.0:
        return "n/a"
    return f"{(float(used) / total_float * 100.0):.1f}%"


def _float_or_na(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


def _scheduler_lines(schedulers: tuple[object, ...]) -> tuple[str, ...]:
    lines: list[str] = []
    idle: list[str] = []
    event_only: list[str] = []
    for scheduler in schedulers:
        active_parts = _nonzero_parts(
            (
                ("pending", scheduler.pending),
                ("running", scheduler.running),
                ("queued", scheduler.queued),
            )
        )
        event_parts = _nonzero_parts(
            (
                ("completed", scheduler.completed),
                ("cancelled", scheduler.cancelled),
                ("stale", scheduler.stale),
                ("failed", scheduler.failed),
                ("prefetch", scheduler.prefetch_scheduled),
                ("deduped", scheduler.prefetch_deduped),
                ("limited", scheduler.prefetch_limited),
                ("blocked_idle", scheduler.prefetch_idle_blocked),
                ("blocked_visible", scheduler.prefetch_visible_busy_blocked),
                ("blocked_cost", scheduler.prefetch_cost_blocked),
                ("active_preserved", getattr(scheduler, "active_preserved", 0)),
                ("queued_collapsed", getattr(scheduler, "queued_collapsed", 0)),
                ("stale_reused", getattr(scheduler, "stale_reused", 0)),
            )
        )
        lanes = tuple(getattr(scheduler, "work_lanes", ()) or ())
        if lanes:
            event_parts.append("lanes=" + ",".join(str(lane) for lane in lanes))
        if not active_parts and not event_parts:
            idle.append(str(scheduler.name))
            continue
        if not active_parts and event_parts == [f"completed={int(scheduler.completed)}"]:
            event_only.append(f"{scheduler.name}: completed={int(scheduler.completed)}")
            continue
        prefix = [*active_parts] if active_parts else ["idle"]
        lines.append(f"{scheduler.name}: " + ", ".join((*prefix, *event_parts)))
    if event_only:
        lines.append("Completed:")
        lines.extend(f"  - {name}" for name in event_only)
    if idle:
        lines.append("Inactive:")
        lines.extend(f"  - {name}" for name in idle)
    return tuple(lines) or ("n/a",)


def _work_graph_lines(work_graph) -> tuple[str, ...]:
    if work_graph is None:
        return ("n/a",)
    lanes = dict(getattr(work_graph, "lanes", {}) or {})
    lines = [
        f"active={int(getattr(work_graph, 'active', 0) or 0)} "
        f"queued={int(getattr(work_graph, 'queued', 0) or 0)} "
        f"completed_keys={int(getattr(work_graph, 'completed_keys', 0) or 0)}"
    ]
    if not lanes:
        lines.append("lanes: n/a")
        return tuple(lines)
    for lane, counters in sorted(lanes.items()):
        counters = dict(counters or {})
        parts = _nonzero_parts(
            (
                ("queued", counters.get("queued", 0)),
                ("admitted", counters.get("admitted", 0)),
                ("dropped", counters.get("dropped", 0)),
                ("superseded", counters.get("superseded", 0)),
                ("completed", counters.get("completed", 0)),
                ("failed", counters.get("failed", 0)),
                ("rescheduled", counters.get("rescheduled", 0)),
                ("reusable", counters.get("reusable_finished", 0)),
                ("deadline", counters.get("deadline_missed", 0)),
                ("blocked", counters.get("blocked_by_budget", 0)),
            )
        )
        lines.append(f"{lane}: " + (", ".join(parts) if parts else "idle"))
    return tuple(lines)


def _nonzero_parts(fields: tuple[tuple[str, object], ...]) -> list[str]:
    parts = []
    for name, value in fields:
        number = int(value or 0)
        if number:
            parts.append(f"{name}={number}")
    return parts
