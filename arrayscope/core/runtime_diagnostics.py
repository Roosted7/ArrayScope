"""Pure runtime diagnostics snapshots and formatting."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.core.cache_status import CacheDiagnosticsSnapshot
from arrayscope.core.memory_budget import format_bytes
from arrayscope.core.memory_policy import MemoryPolicy, format_memory_policy


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
    skipped_tiles: int = 0
    visible_tiles: int = 0
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
class WindowRuntimeDiagnostics:
    memory_policy: MemoryPolicy
    image_cache: CacheDiagnosticsSnapshot
    tile_cache: CacheDiagnosticsSnapshot
    profile_cache: CacheDiagnosticsSnapshot
    schedulers: tuple[object, ...]
    render: RenderRuntimeDiagnostics
    montage: MontageRuntimeDiagnostics
    fft_backend_choice: str
    fft_backend_resolved: str
    fft_workers_choice: str
    fft_workers_resolved: int
    operation_count: int
    derived_shape: tuple[int, ...]
    derived_dtype: str
    pipeline_peak_bytes: int | None
    pipeline_warnings: tuple[str, ...] = ()


def format_runtime_diagnostics(snapshot: WindowRuntimeDiagnostics) -> str:
    sections = [
        "Memory",
        format_memory_policy(snapshot.memory_policy),
        "",
        "Caches",
        _cache_line("Image", snapshot.image_cache),
        _cache_line("Montage tiles", snapshot.tile_cache),
        _cache_line("Profiles/scalars", snapshot.profile_cache),
        "",
        "Schedulers",
        *(_scheduler_line(scheduler) for scheduler in snapshot.schedulers),
        "",
        "Render",
        f"Decision: {snapshot.render.last_decision_kind or 'n/a'}",
        f"Reason: {snapshot.render.last_decision_reason or 'n/a'}",
        f"Context: {snapshot.render.last_context_summary or 'n/a'}",
        f"Request: {snapshot.render.last_request_key or 'n/a'}",
        f"Error: {snapshot.render.last_error or 'n/a'}",
        "",
        "Montage",
        f"Active: {snapshot.montage.active}",
        f"Session: {snapshot.montage.session_id if snapshot.montage.session_id is not None else 'n/a'}",
        f"Canvas rect: {snapshot.montage.canvas_rect if snapshot.montage.canvas_rect is not None else 'n/a'}",
        f"Canvas shape: {snapshot.montage.canvas_shape if snapshot.montage.canvas_shape is not None else 'n/a'}",
        f"Canvas bytes: {'n/a' if snapshot.montage.canvas_bytes is None else format_bytes(snapshot.montage.canvas_bytes)}",
        (
            "Tiles: "
            f"visible={snapshot.montage.visible_tiles} loaded={snapshot.montage.loaded_tiles} "
            f"loading={snapshot.montage.loading_tiles} pending={snapshot.montage.pending_tiles} "
            f"skipped={snapshot.montage.skipped_tiles}"
        ),
        f"Loading overlays: {snapshot.montage.show_loading_overlays}",
        "",
        "FFT",
        f"Backend: {snapshot.fft_backend_choice} -> {snapshot.fft_backend_resolved}",
        f"Workers: {snapshot.fft_workers_choice} -> {snapshot.fft_workers_resolved}",
        "",
        "Operations",
        f"Count: {snapshot.operation_count}",
        f"Derived: {snapshot.derived_shape} {snapshot.derived_dtype}",
        f"Pipeline peak: {'n/a' if snapshot.pipeline_peak_bytes is None else format_bytes(snapshot.pipeline_peak_bytes)}",
    ]
    sections.extend(f"Warning: {warning}" for warning in snapshot.pipeline_warnings)
    return "\n".join(str(line) for line in sections)


def _cache_line(name: str, cache: CacheDiagnosticsSnapshot) -> str:
    hit_rate = "n/a" if cache.hit_rate is None else f"{cache.hit_rate:.0%}"
    return (
        f"{name}: {cache.status.value}, entries={cache.entries}, "
        f"bytes={format_bytes(cache.bytes_used)} / {format_bytes(cache.max_bytes)}, "
        f"hits={cache.hits}, misses={cache.misses}, evictions={cache.evictions}, hit-rate={hit_rate}"
    )


def _scheduler_line(scheduler: SchedulerDiagnostics) -> str:
    return (
        f"{scheduler.name}: pending={scheduler.pending}, running={scheduler.running}, queued={scheduler.queued}, "
        f"completed={scheduler.completed}, cancelled={scheduler.cancelled}, stale={scheduler.stale}, failed={scheduler.failed}, "
        f"prefetch scheduled={scheduler.prefetch_scheduled}, deduped={scheduler.prefetch_deduped}, limited={scheduler.prefetch_limited}, "
        f"blocked idle={scheduler.prefetch_idle_blocked}, visible={scheduler.prefetch_visible_busy_blocked}, cost={scheduler.prefetch_cost_blocked}"
    )
