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
    pipeline_warnings: tuple[str, ...] = ()
    capability_stage_count: int | None = None
    stage_cache_candidate_count: int | None = None
    stage_cache_candidate_summaries: tuple[str, ...] = ()
    operation_final_region: str = ""
    operation_required_input_region: str = ""
    operation_expanded_axes: tuple[int, ...] = ()
    operation_transition_summaries: tuple[str, ...] = ()


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
                    f"skipped={snapshot.montage.skipped_tiles}"
                ),
                f"Loading overlays: {snapshot.montage.show_loading_overlays}",
            )
        ),
        "FFT": "\n".join(
            (
                f"Backend: {snapshot.fft_backend_choice} -> {snapshot.fft_backend_resolved}",
                f"Workers: {snapshot.fft_workers_choice} -> {snapshot.fft_workers_resolved}",
            )
        ),
        "Operations": "\n".join(
            (
                f"Count: {snapshot.operation_count}",
                f"Derived: {snapshot.derived_shape} {snapshot.derived_dtype}",
                f"Pipeline peak: {'n/a' if snapshot.pipeline_peak_bytes is None else format_bytes(snapshot.pipeline_peak_bytes)}",
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


def _scheduler_line(scheduler) -> str:
    return (
        f"{scheduler.name}: pending={scheduler.pending}, running={scheduler.running}, queued={scheduler.queued}, "
        f"completed={scheduler.completed}, cancelled={scheduler.cancelled}, stale={scheduler.stale}, failed={scheduler.failed}, "
        f"prefetch scheduled={scheduler.prefetch_scheduled}, deduped={scheduler.prefetch_deduped}, limited={scheduler.prefetch_limited}, "
        f"blocked idle={scheduler.prefetch_idle_blocked}, visible={scheduler.prefetch_visible_busy_blocked}, cost={scheduler.prefetch_cost_blocked}"
    )
