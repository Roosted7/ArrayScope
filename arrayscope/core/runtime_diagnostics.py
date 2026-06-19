"""Pure runtime diagnostics snapshots and formatting."""

from __future__ import annotations

from dataclasses import dataclass, field

from arrayscope.core.cache_status import CacheDiagnosticsSnapshot
from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.memory_policy import MemoryPolicy, format_memory_policy
from arrayscope.core.resource_governor import ResourceGovernorDiagnostics, ResourcePressure


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
    tile_layer_items_updated: int = 0
    tile_layer_items_skipped: int = 0
    tile_layer_rgb_window_tiles: int = 0
    tile_layer_resident_items: int = 0
    tile_layer_storage_capacity: int = 0
    tile_layer_storage_rebuilds: int = 0
    tile_layer_storage_evictions: int = 0
    tile_layer_texture_uploads: int = 0
    tile_layer_texture_upload_bytes: int = 0
    tile_layer_vertex_uploads: int = 0
    tile_layer_level_updates: int = 0
    tile_layer_estimated_gpu_bytes: int = 0
    tile_layer_cpu_shadow_bytes: int = 0


@dataclass(frozen=True)
class MontageRuntimeDiagnostics:
    active: bool
    session_id: int | None = None
    canvas_rect: tuple[int, int, int, int] | None = None
    canvas_shape: tuple[int, int] | None = None
    canvas_bytes: int | None = None
    loaded_tiles: int = 0
    loading_tiles: int = 0
    pending_tiles: int = 0
    pending_level_tiles: int = 0
    skipped_tiles: int = 0
    visible_tiles: int = 0
    attached_stage_requests: int = 0
    display_mode: str = "canvas"
    backend_setting: str = "auto"
    backend_chosen: str = "canvas"
    backend_reason: str = ""
    backend_fallback_available: str = "canvas"
    backend_warning: str = ""
    show_loading_overlays: bool = False
    tile_compute_cache_hits: int = 0
    tile_compute_stage_backed: int = 0
    tile_compute_direct: int = 0
    tile_compute_waiting_for_stage: int = 0
    lead_direct_tiles: int = 0
    stage_backed_tiles_pending: int = 0
    retained_stage_index: int | None = None
    retained_stage_decision: str = ""
    repeated_expensive_stage_per_tile: bool = False


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
    last_tile_eval_ms: float | None = None
    last_tile_cache_lookup_ms: float | None = None
    last_tile_cache_hit: bool | None = None
    last_stage_cache_lookup_ms: float | None = None
    last_stage_cache_hit: bool | None = None
    last_stage_attach_wait_ms: float | None = None
    last_level_stats_ms: float | None = None
    last_visible_upload_ms: float | None = None
    last_histogram_upload_ms: float | None = None
    last_histogram_recompute_ms: float | None = None
    last_rgb_window_ms: float | None = None
    last_tile_layer_upload_ms: float | None = None
    last_tile_layer_rgb_window_ms: float | None = None
    last_level_sync_ms: float | None = None
    last_canvas_compose_ms: float | None = None
    last_canvas_patch_ms: float | None = None
    last_canvas_commit_ms: float | None = None
    last_set_image_ms: float | None = None
    last_overlay_update_ms: float | None = None
    cached_tiles_last_session: int = 0
    missing_tiles_last_session: int = 0
    patched_tiles_last_flush: int = 0
    upload_visible_bytes: int = 0
    upload_histogram_bytes: int = 0
    upload_fast_same_object: bool = False
    tile_layer_visible_items: int = 0
    tile_layer_items_updated: int = 0
    tile_layer_items_skipped: int = 0
    tile_layer_rgb_window_tiles: int = 0
    tile_layer_resident_items: int = 0
    tile_layer_storage_capacity: int = 0
    tile_layer_storage_rebuilds: int = 0
    tile_layer_storage_evictions: int = 0
    tile_layer_texture_uploads: int = 0
    tile_layer_texture_upload_bytes: int = 0
    tile_layer_vertex_uploads: int = 0
    tile_layer_level_updates: int = 0
    tile_layer_estimated_gpu_bytes: int = 0
    tile_layer_cpu_shadow_bytes: int = 0
    coalesced_commits: int = 0


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
    image_cache: CacheDiagnosticsSnapshot
    tile_cache: CacheDiagnosticsSnapshot
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
    stage_materialization: object | None = None
    stage_warmup: object | None = None
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
        "Feedback": "\n".join(_feedback_lines(snapshot.resource_governor)),
        "Montage": "\n".join(_montage_lines(snapshot)),
        "Render": "\n".join(_render_lines(snapshot)),
        "Schedulers": "\n".join(_scheduler_lines(snapshot.schedulers)),
        "Caches": "\n".join(
            (
                _cache_line("Image", snapshot.image_cache),
                _cache_line("Montage tiles", snapshot.tile_cache),
                _cache_line("Profiles/scalars", snapshot.profile_cache),
                _stage_cache_line("Stage cache", snapshot.stage_cache),
                _stage_materialization_line("Stage materialization", snapshot.stage_materialization),
                _stage_warmup_line("Stage warmup", snapshot.stage_warmup),
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
            f"{_bottleneck_text(snapshot)}"
        ),
        (
            "Feedback: "
            f"{_pressure_summary(snapshot.resource_governor)}"
        ),
        (
            "Renderer:\n"
            f"  image={snapshot.image_rendering_backend_actual} "
            f"setting={snapshot.image_rendering_backend_selected} "
            f"montage={snapshot.montage.backend_chosen} montage_setting={snapshot.montage.backend_setting}"
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
            f"pending={snapshot.montage.pending_tiles}\n"
            f"  canvas={_bytes_or_na(snapshot.montage.canvas_bytes)}"
        ),
        (
            "Tile layer:\n"
            f"  visible={snapshot.montage_timing.tile_layer_visible_items} "
            f"resident={snapshot.montage_timing.tile_layer_resident_items}/"
            f"{snapshot.montage_timing.tile_layer_storage_capacity} "
            f"updated={snapshot.montage_timing.tile_layer_items_updated} "
            f"skipped={snapshot.montage_timing.tile_layer_items_skipped} "
            f"rgb_tiles={snapshot.montage_timing.tile_layer_rgb_window_tiles}\n"
            f"  rgb={_ms_text(snapshot.montage_timing.last_tile_layer_rgb_window_ms)} "
            f"upload={_ms_text(snapshot.montage_timing.last_tile_layer_upload_ms)} "
            f"gpu={format_bytes(snapshot.montage_timing.tile_layer_estimated_gpu_bytes)}"
        ),
        (
            "Upload: "
            f"visible={format_bytes(snapshot.montage_timing.upload_visible_bytes)} "
            f"histogram={format_bytes(snapshot.montage_timing.upload_histogram_bytes)} "
            f"same object={snapshot.montage_timing.upload_fast_same_object}"
        ),
        (
            "Coalescer: "
            f"pending={snapshot.render_coalescer.pending}, "
            f"requested={snapshot.render_coalescer.requested}, "
            f"flushed={snapshot.render_coalescer.flushed}, "
            f"coalesced={snapshot.render_coalescer.coalesced}"
        ),
        _stage_warmup_line("Stage warmup", snapshot.stage_warmup),
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
        for channel in diagnostics.feedback_channels:
            if (
                channel.elapsed_ewma_ms is None
                and channel.per_item_ewma_ms is None
                and float(channel.last_elapsed_ms) <= 0.0
            ):
                inactive.append(channel.channel)
                continue
            active_channel_lines.append(
                f"  {channel.channel}:\n"
                f"    last={_ms_text(channel.last_elapsed_ms)} "
                f"ewma={_ms_text(channel.elapsed_ewma_ms)} "
                f"per-item={_ms_text(channel.per_item_ewma_ms)}\n"
                f"    batch={channel.batch_limit} "
                f"budget={channel.budget_ms:.1f} ms "
                f"interval={channel.interval_ms} ms"
            )
        lines.extend(active_channel_lines or ("  n/a",))
        if inactive:
            lines.append("  Inactive:")
            lines.extend(f"    - {name}" for name in inactive)
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
    )
    if not channels:
        return None
    channel = max(channels, key=lambda item: float(item.elapsed_ewma_ms or 0.0))
    return str(channel.channel)


def _bottleneck_text(snapshot: WindowRuntimeDiagnostics) -> str:
    governor = snapshot.resource_governor
    if governor is not None:
        if governor.pressure.ui_pressure in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
            return "UI fan-in"
        if governor.pressure.memory_pressure in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
            return "memory"
    if snapshot.montage.active and snapshot.montage.tile_compute_waiting_for_stage:
        return "stage compute"
    if snapshot.montage_timing.tile_layer_rgb_window_tiles:
        return "RGB window/upload"
    if snapshot.montage.pending_tiles:
        return "tile compute"
    return "idle"


def _render_lines(snapshot: WindowRuntimeDiagnostics) -> tuple[str, ...]:
    return (
        f"Decision: {snapshot.render.last_decision_kind or 'n/a'}",
        f"Reason: {snapshot.render.last_decision_reason or 'n/a'}",
        f"Context:\n{_wrapped_debug_text(snapshot.render.last_context_summary or 'n/a', indent='  ')}",
        f"Request: {_request_text(snapshot.render.last_request_key)}",
        f"Error: {snapshot.render.last_error or 'n/a'}",
        (
            "Coalescer: "
            f"pending={snapshot.render_coalescer.pending}, "
            f"interactive={snapshot.render_coalescer.interactive_active}, "
            f"requested={snapshot.render_coalescer.requested}, "
            f"flushed={snapshot.render_coalescer.flushed}, "
            f"coalesced={snapshot.render_coalescer.coalesced}, "
            f"deferred refreshes={snapshot.render_coalescer.deferred_side_panel_refreshes}"
        ),
        f"Timing render sync: {_ms_text(snapshot.render_timing.last_render_sync_ms)}",
        f"Timing control sync: {_ms_text(snapshot.render_timing.last_control_sync_ms)}",
        f"Timing planning: {_ms_text(snapshot.render_timing.last_planning_ms)}",
        f"Timing worker queue wait: {_ms_text(snapshot.render_timing.last_worker_queue_wait_ms)}",
        f"Timing evaluation: {_ms_text(snapshot.render_timing.last_evaluation_ms)}",
        f"Timing display commit: {_ms_text(snapshot.render_timing.last_display_commit_ms)}",
        f"Timing set image: {_ms_text(snapshot.render_timing.last_set_image_ms)}",
        f"Timing levels/histogram: {_ms_text(snapshot.render_timing.last_levels_histogram_ms)}",
        f"Timing operation dock: {_ms_text(snapshot.render_timing.last_operation_dock_ms)}",
        f"Timing inspection refresh: {_ms_text(snapshot.render_timing.last_inspection_refresh_ms)}",
    )


def _canvas_preserve_lines(canvas_preserve: CanvasPreserveRuntimeDiagnostics) -> tuple[str, ...]:
    events = tuple(canvas_preserve.events)
    recent_events = ("Recent events:", *(f"  {event}" for event in events)) if events else ("Recent events: n/a",)
    return (
        f"Mode: {canvas_preserve.mode}",
        f"Platform: {canvas_preserve.platform or 'n/a'}",
        f"Active: {canvas_preserve.active}",
        f"Last transition: {canvas_preserve.last_transition or 'n/a'}",
        f"Last result: {canvas_preserve.last_result or 'n/a'}",
        f"Target canvas: {_size_text(canvas_preserve.target_canvas_size)}",
        f"Final canvas: {_size_text(canvas_preserve.final_canvas_size)}",
        f"Final window: {_size_text(canvas_preserve.final_window_size)}",
        f"Last delta: {_size_text(canvas_preserve.last_delta)}",
        f"Attempts used: {canvas_preserve.attempts_used}",
        f"Strong used: {canvas_preserve.strong_used}",
        f"Strong available: {canvas_preserve.strong_available}",
        f"Constraints active: {canvas_preserve.constraints_active}",
        *recent_events,
    )


def _montage_lines(snapshot: WindowRuntimeDiagnostics) -> tuple[str, ...]:
    return (
        f"Active: {snapshot.montage.active}",
        f"Session: {snapshot.montage.session_id if snapshot.montage.session_id is not None else 'n/a'}",
        f"Canvas rect: {snapshot.montage.canvas_rect if snapshot.montage.canvas_rect is not None else 'n/a'}",
        f"Canvas shape: {snapshot.montage.canvas_shape if snapshot.montage.canvas_shape is not None else 'n/a'}",
        f"Canvas bytes: {_bytes_or_na(snapshot.montage.canvas_bytes)}",
        (
            "Tiles: "
            f"visible={snapshot.montage.visible_tiles} loaded={snapshot.montage.loaded_tiles} "
            f"loading={snapshot.montage.loading_tiles} pending={snapshot.montage.pending_tiles} "
            f"pending levels={snapshot.montage.pending_level_tiles} "
            f"skipped={snapshot.montage.skipped_tiles}"
        ),
        f"Attached stage waits: {snapshot.montage.attached_stage_requests}",
        f"Display mode: {snapshot.montage.display_mode}",
        f"Display backend: {snapshot.montage.backend_chosen} (setting={snapshot.montage.backend_setting}, fallback={snapshot.montage.backend_fallback_available})",
        f"Display backend reason: {snapshot.montage.backend_reason or 'n/a'}",
        f"Warning: {snapshot.montage.backend_warning}" if snapshot.montage.backend_warning else "Warning: n/a",
        f"Loading overlays: {snapshot.montage.show_loading_overlays}",
        (
            "Reusable stage: "
            f"stage={snapshot.montage.retained_stage_index if snapshot.montage.retained_stage_index is not None else 'n/a'} "
            f"{snapshot.montage.retained_stage_decision or 'n/a'}, "
            f"repeated per tile={'yes' if snapshot.montage.repeated_expensive_stage_per_tile else 'no'}"
        ),
        (
            "Tile compute: "
            f"cache_hit={snapshot.montage.tile_compute_cache_hits} "
            f"stage_backed={snapshot.montage.tile_compute_stage_backed} "
            f"direct={snapshot.montage.tile_compute_direct} "
            f"waiting_stage={snapshot.montage.tile_compute_waiting_for_stage}"
        ),
        f"Lead direct tiles: {snapshot.montage.lead_direct_tiles}",
        f"Timing tile eval: {_ms_text(snapshot.montage_timing.last_tile_eval_ms)}",
        f"Timing tile cache lookup: {_ms_text(snapshot.montage_timing.last_tile_cache_lookup_ms)}",
        f"Tile cache hit: {_bool_text(snapshot.montage_timing.last_tile_cache_hit)}",
        f"Timing stage cache lookup: {_ms_text(snapshot.montage_timing.last_stage_cache_lookup_ms)}",
        f"Stage cache hit: {_bool_text(snapshot.montage_timing.last_stage_cache_hit)}",
        f"Timing attached stage wait: {_ms_text(snapshot.montage_timing.last_stage_attach_wait_ms)}",
        f"Timing level stats: {_ms_text(snapshot.montage_timing.last_level_stats_ms)}",
        f"Timing visible upload: {_ms_text(snapshot.montage_timing.last_visible_upload_ms)}",
        f"Timing histogram upload: {_ms_text(snapshot.montage_timing.last_histogram_upload_ms)}",
        f"Timing histogram recompute: {_ms_text(snapshot.montage_timing.last_histogram_recompute_ms)}",
        f"Timing RGB window: {_ms_text(snapshot.montage_timing.last_rgb_window_ms)}",
        f"Timing tile layer upload: {_ms_text(snapshot.montage_timing.last_tile_layer_upload_ms)}",
        f"Timing tile layer RGB window: {_ms_text(snapshot.montage_timing.last_tile_layer_rgb_window_ms)}",
        f"Timing level sync: {_ms_text(snapshot.montage_timing.last_level_sync_ms)}",
        f"Timing canvas compose: {_ms_text(snapshot.montage_timing.last_canvas_compose_ms)}",
        f"Timing canvas patch: {_ms_text(snapshot.montage_timing.last_canvas_patch_ms)}",
        f"Timing canvas commit: {_ms_text(snapshot.montage_timing.last_canvas_commit_ms)}",
        f"Timing montage set image: {_ms_text(snapshot.montage_timing.last_set_image_ms)}",
        f"Timing overlay update: {_ms_text(snapshot.montage_timing.last_overlay_update_ms)}",
        f"Tile cache last session: cached={snapshot.montage_timing.cached_tiles_last_session} missing={snapshot.montage_timing.missing_tiles_last_session}",
        f"Patched tiles last flush: {snapshot.montage_timing.patched_tiles_last_flush}",
        (
            "Tile layer items: "
            f"visible={snapshot.montage_timing.tile_layer_visible_items} "
            f"resident={snapshot.montage_timing.tile_layer_resident_items}/"
            f"{snapshot.montage_timing.tile_layer_storage_capacity} "
            f"updated={snapshot.montage_timing.tile_layer_items_updated} "
            f"skipped={snapshot.montage_timing.tile_layer_items_skipped}"
        ),
        f"Tile layer RGB window tiles: {snapshot.montage_timing.tile_layer_rgb_window_tiles}",
        (
            "Tile layer storage: "
            f"rebuilds={snapshot.montage_timing.tile_layer_storage_rebuilds} "
            f"evictions={snapshot.montage_timing.tile_layer_storage_evictions} "
            f"gpu={format_bytes(snapshot.montage_timing.tile_layer_estimated_gpu_bytes)} "
            f"cpu_shadow={format_bytes(snapshot.montage_timing.tile_layer_cpu_shadow_bytes)}"
        ),
        (
            "Tile layer submissions: "
            f"textures={snapshot.montage_timing.tile_layer_texture_uploads} "
            f"bytes={format_bytes(snapshot.montage_timing.tile_layer_texture_upload_bytes)} "
            f"vertices={snapshot.montage_timing.tile_layer_vertex_uploads} "
            f"levels={snapshot.montage_timing.tile_layer_level_updates}"
        ),
        _montage_prefetch_line("Montage prefetch", snapshot.montage_prefetch),
        f"Coalesced montage commits: {snapshot.montage_timing.coalesced_commits}",
        (
            "Upload: "
            f"visible={format_bytes(snapshot.montage_timing.upload_visible_bytes)} "
            f"histogram={format_bytes(snapshot.montage_timing.upload_histogram_bytes)} "
            f"same object={snapshot.montage_timing.upload_fast_same_object}"
        ),
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


def _stage_warmup_line(name: str, diagnostics) -> str:
    if diagnostics is None:
        return f"{name}: n/a"
    candidate = getattr(diagnostics, "candidate_bytes", None)
    candidate_text = "unknown" if candidate is None else format_bytes(int(candidate))
    return (
        f"{name}: decision={getattr(diagnostics, 'decision', '') or 'n/a'}, "
        f"candidate={candidate_text}, budget={format_bytes(int(getattr(diagnostics, 'budget_bytes', 0)))}, "
        f"reason={getattr(diagnostics, 'reason', '') or 'n/a'}"
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
            )
        )
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


def _nonzero_parts(fields: tuple[tuple[str, object], ...]) -> list[str]:
    parts = []
    for name, value in fields:
        number = int(value or 0)
        if number:
            parts.append(f"{name}={number}")
    return parts
