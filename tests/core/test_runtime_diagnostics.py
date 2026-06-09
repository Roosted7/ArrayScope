from arrayscope.core.cache_status import CacheDiagnosticsSnapshot, CacheStatus
from arrayscope.core.memory_policy import MemoryProfileChoice, compute_memory_policy
from arrayscope.core.runtime_diagnostics import (
    CanvasPreserveRuntimeDiagnostics,
    MontageTimingDiagnostics,
    MontageRuntimeDiagnostics,
    RenderRuntimeDiagnostics,
    RenderTimingDiagnostics,
    WindowRuntimeDiagnostics,
    format_runtime_diagnostics,
)
from arrayscope.operations.stage_cache import StageCacheDiagnostics
from arrayscope.window.evaluation_controller import SchedulerDiagnostics


def _cache():
    return CacheDiagnosticsSnapshot(status=CacheStatus.READY, entries=0, bytes_used=0, max_bytes=1024, hit_rate=None)


def test_format_runtime_diagnostics_includes_all_major_sections():
    policy = compute_memory_policy(profile=MemoryProfileChoice.BALANCED, render_cap_mb=512, input_nbytes=1, system=None)
    scheduler = SchedulerDiagnostics("visible", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    snapshot = WindowRuntimeDiagnostics(
        memory_policy=policy,
        image_cache=_cache(),
        tile_cache=_cache(),
        profile_cache=_cache(),
        stage_cache=StageCacheDiagnostics(
            entries=0,
            bytes_used=0,
            max_bytes=1024,
            hits=0,
            misses=0,
            evictions=0,
            hit_rate=None,
            candidates_seen=0,
            stores=0,
            refused_over_budget=0,
            last_hit="",
            last_miss="stage=1",
            last_store="stage=1",
        ),
        schedulers=(scheduler,),
        render=RenderRuntimeDiagnostics(),
        montage=MontageRuntimeDiagnostics(active=False),
        canvas_preserve=CanvasPreserveRuntimeDiagnostics(events=("start gen=1",)),
        render_timing=RenderTimingDiagnostics(last_render_sync_ms=1.25),
        montage_timing=MontageTimingDiagnostics(last_canvas_compose_ms=2.5, cached_tiles_last_session=3, missing_tiles_last_session=4),
        fft_backend_choice="auto",
        fft_backend_resolved="numpy",
        fft_workers_choice="auto",
        fft_workers_resolved=1,
        operation_count=0,
        derived_shape=(4, 5),
        derived_dtype="float32",
        pipeline_peak_bytes=None,
        optimized_operation_count=0,
        operation_optimization_summaries=("removed Conjugate pair",),
        operation_final_region="[:, :, 3]",
        operation_required_input_region="[:, :, :]",
        operation_expanded_axes=(2,),
        operation_transition_summaries=("stage 1 CenteredFFT output=[:, :, 3] input=[:, :, :] expanded=2",),
    )

    text = format_runtime_diagnostics(snapshot)

    for heading in ("Memory", "Caches", "Schedulers", "Render", "Canvas Preserve", "Montage", "FFT", "Operations"):
        assert heading in text
    assert "hit-rate=n/a" in text
    assert "start gen=1" in text
    assert "Final region: [:, :, 3]" in text
    assert "Required input: [:, :, :]" in text
    assert "Expanded axes: 2" in text
    assert "Optimized count: 0" in text
    assert "removed Conjugate pair" in text
    assert "stage 1 CenteredFFT" in text
    assert "Stage cache:" in text
    assert "Stage cache last miss: stage=1" in text
    assert "Stage cache last store: stage=1" in text
    assert "Coalescer: pending=False, interactive=False" in text
    assert "Timing render sync: 1.25 ms" in text
    assert "Timing worker queue wait: n/a" in text
    assert "Timing canvas compose: 2.50 ms" in text
    assert "Tile cache last session: cached=3 missing=4" in text
