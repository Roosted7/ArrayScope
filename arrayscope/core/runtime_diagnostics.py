"""Pure runtime diagnostics snapshots and formatting."""

from __future__ import annotations

from dataclasses import dataclass, field

from arrayscope.core.cache_status import CacheDiagnosticsSnapshot
from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.memory_policy import MemoryPolicy, format_memory_policy


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
    show_loading_overlays: bool = False


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


def format_runtime_diagnostics(snapshot: WindowRuntimeDiagnostics) -> str:
    return "\n\n".join(f"{title}\n{text}" for title, text in format_runtime_diagnostics_sections(snapshot).items())


def format_runtime_diagnostics_sections(snapshot: WindowRuntimeDiagnostics) -> dict[str, str]:
    sections = {
        "Memory": format_memory_policy(snapshot.memory_policy),
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
        "Schedulers": "\n".join(_scheduler_line(scheduler) for scheduler in snapshot.schedulers),
        "Render": "\n".join(
            (
                f"Decision: {snapshot.render.last_decision_kind or 'n/a'}",
                f"Reason: {snapshot.render.last_decision_reason or 'n/a'}",
                f"Context: {snapshot.render.last_context_summary or 'n/a'}",
                f"Request: {snapshot.render.last_request_key or 'n/a'}",
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
        ),
        "Canvas Preserve": "\n".join(
            (
                f"Mode: {snapshot.canvas_preserve.mode}",
                f"Platform: {snapshot.canvas_preserve.platform or 'n/a'}",
                f"Active: {snapshot.canvas_preserve.active}",
                f"Last transition: {snapshot.canvas_preserve.last_transition or 'n/a'}",
                f"Last result: {snapshot.canvas_preserve.last_result or 'n/a'}",
                f"Target canvas: {_size_text(snapshot.canvas_preserve.target_canvas_size)}",
                f"Final canvas: {_size_text(snapshot.canvas_preserve.final_canvas_size)}",
                f"Final window: {_size_text(snapshot.canvas_preserve.final_window_size)}",
                f"Last delta: {_size_text(snapshot.canvas_preserve.last_delta)}",
                f"Attempts used: {snapshot.canvas_preserve.attempts_used}",
                f"Strong used: {snapshot.canvas_preserve.strong_used}",
                f"Strong available: {snapshot.canvas_preserve.strong_available}",
                f"Constraints active: {snapshot.canvas_preserve.constraints_active}",
                "Recent events:",
                *(f"  {event}" for event in snapshot.canvas_preserve.events),
            )
        ),
        "Montage": "\n".join(
            (
                f"Active: {snapshot.montage.active}",
                f"Session: {snapshot.montage.session_id if snapshot.montage.session_id is not None else 'n/a'}",
                f"Canvas rect: {snapshot.montage.canvas_rect if snapshot.montage.canvas_rect is not None else 'n/a'}",
                f"Canvas shape: {snapshot.montage.canvas_shape if snapshot.montage.canvas_shape is not None else 'n/a'}",
                f"Canvas bytes: {'n/a' if snapshot.montage.canvas_bytes is None else format_bytes(snapshot.montage.canvas_bytes)}",
                (
                    "Tiles: "
                    f"visible={snapshot.montage.visible_tiles} loaded={snapshot.montage.loaded_tiles} "
                    f"loading={snapshot.montage.loading_tiles} pending={snapshot.montage.pending_tiles} "
                    f"pending levels={snapshot.montage.pending_level_tiles} "
                    f"skipped={snapshot.montage.skipped_tiles}"
                ),
                f"Attached stage waits: {snapshot.montage.attached_stage_requests}",
                f"Display mode: {snapshot.montage.display_mode}",
                f"Loading overlays: {snapshot.montage.show_loading_overlays}",
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
                    f"updated={snapshot.montage_timing.tile_layer_items_updated} "
                    f"skipped={snapshot.montage_timing.tile_layer_items_skipped}"
                ),
                f"Tile layer RGB window tiles: {snapshot.montage_timing.tile_layer_rgb_window_tiles}",
                _montage_prefetch_line("Montage prefetch", snapshot.montage_prefetch),
                f"Coalesced montage commits: {snapshot.montage_timing.coalesced_commits}",
                (
                    "Upload: "
                    f"visible={format_bytes(snapshot.montage_timing.upload_visible_bytes)} "
                    f"histogram={format_bytes(snapshot.montage_timing.upload_histogram_bytes)} "
                    f"same object={snapshot.montage_timing.upload_fast_same_object}"
                ),
            )
        ),
        "FFT": "\n".join(
            (
                f"Backend: {snapshot.fft_backend_choice} -> {snapshot.fft_backend_resolved}",
                f"Workers: {snapshot.fft_workers_choice} -> {snapshot.fft_workers_resolved}",
            )
        ),
        "Compute": "\n".join(
            (
                "Workers: " + (", ".join(snapshot.compute_worker_summaries) or "n/a"),
                "FFT workers: " + (", ".join(snapshot.compute_fft_worker_summaries) or "n/a"),
            )
        ),
        "Operations": "\n".join(
            (
                f"Count: {snapshot.operation_count}",
                f"Optimized count: {'n/a' if snapshot.optimized_operation_count is None else snapshot.optimized_operation_count}",
                f"Derived: {snapshot.derived_shape} {snapshot.derived_dtype}",
                f"Pipeline peak: {'n/a' if snapshot.pipeline_peak_bytes is None else format_bytes(snapshot.pipeline_peak_bytes)}",
                "Optimizations:",
                *(f"  {summary}" for summary in snapshot.operation_optimization_summaries),
                f"Final region: {snapshot.operation_final_region or 'n/a'}",
                f"Required input: {snapshot.operation_required_input_region or 'n/a'}",
                f"Expanded axes: {_axes_text(snapshot.operation_expanded_axes)}",
                "Transitions:",
                *(f"  {transition}" for transition in snapshot.operation_transition_summaries),
                f"Capability stages: {'n/a' if snapshot.capability_stage_count is None else snapshot.capability_stage_count}",
                f"Stage cache candidates: {'n/a' if snapshot.stage_cache_candidate_count is None else snapshot.stage_cache_candidate_count}",
                *(f"Candidate: {candidate}" for candidate in snapshot.stage_cache_candidate_summaries),
                *_stage_cache_operation_lines(snapshot.stage_cache),
                *(f"Warning: {warning}" for warning in snapshot.pipeline_warnings),
            )
        ),
    }
    return sections


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


def _axes_text(axes: tuple[int, ...]) -> str:
    return "n/a" if not axes else ",".join(str(int(axis)) for axis in axes)


def _ms_text(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f} ms"


def _bool_text(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def _scheduler_line(scheduler) -> str:
    return (
        f"{scheduler.name}: pending={scheduler.pending}, running={scheduler.running}, queued={scheduler.queued}, "
        f"completed={scheduler.completed}, cancelled={scheduler.cancelled}, stale={scheduler.stale}, failed={scheduler.failed}, "
        f"prefetch scheduled={scheduler.prefetch_scheduled}, deduped={scheduler.prefetch_deduped}, limited={scheduler.prefetch_limited}, "
        f"blocked idle={scheduler.prefetch_idle_blocked}, visible={scheduler.prefetch_visible_busy_blocked}, cost={scheduler.prefetch_cost_blocked}"
    )
