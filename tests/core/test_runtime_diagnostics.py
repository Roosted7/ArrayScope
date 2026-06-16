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
from arrayscope.window.stage_warmup import StageWarmupDecision
from arrayscope.window.montage_prefetch import MontagePrefetchDecision
from arrayscope.core.scheduler import SchedulerDiagnostics


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
        montage=MontageRuntimeDiagnostics(active=False, display_mode="tile_layer"),
        canvas_preserve=CanvasPreserveRuntimeDiagnostics(events=("start gen=1",)),
        render_timing=RenderTimingDiagnostics(last_render_sync_ms=1.25),
        stage_warmup=StageWarmupDecision("scheduled", candidate_bytes=128, budget_bytes=1024, reason="tiles wait for shared stage"),
        montage_prefetch=(MontagePrefetchDecision(12, 12, "skipped_stage_missing", "would recompute expensive stage per tile"),),
        montage_timing=MontageTimingDiagnostics(
            last_canvas_compose_ms=2.5,
            last_visible_upload_ms=10.0,
            last_histogram_upload_ms=5.0,
            last_histogram_recompute_ms=3.0,
            last_rgb_window_ms=2.0,
            last_tile_layer_upload_ms=0.25,
            last_tile_layer_rgb_window_ms=1.5,
            last_level_sync_ms=1.0,
            cached_tiles_last_session=3,
            missing_tiles_last_session=4,
            upload_visible_bytes=1024,
            upload_histogram_bytes=512,
            upload_fast_same_object=True,
            tile_layer_visible_items=50,
            tile_layer_items_updated=1,
            tile_layer_items_skipped=49,
            tile_layer_rgb_window_tiles=1,
            coalesced_commits=7,
        ),
        fft_backend_choice="auto",
        fft_backend_resolved="numpy",
        fft_workers_choice="auto",
        fft_workers_resolved=1,
        compute_worker_summaries=("visible=1", "montage_tile=2"),
        compute_fft_worker_summaries=("visible=4", "montage_tile=1"),
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

    for heading in ("Memory", "Caches", "Schedulers", "Render", "Canvas Preserve", "Montage", "FFT", "Compute", "Operations"):
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
    assert "Stage warmup: decision=scheduled" in text
    assert "Stage cache last miss: stage=1" in text
    assert "Stage cache last store: stage=1" in text
    assert "Montage prefetch: tile=12 source=12 decision=skipped_stage_missing reason=would recompute expensive stage per tile" in text
    assert "Coalescer: pending=False, interactive=False" in text
    assert "Timing render sync: 1.25 ms" in text
    assert "Timing worker queue wait: n/a" in text
    assert "Timing canvas compose: 2.50 ms" in text
    assert "Display mode: tile_layer" in text
    assert "Timing visible upload: 10.00 ms" in text
    assert "Timing histogram upload: 5.00 ms" in text
    assert "Timing histogram recompute: 3.00 ms" in text
    assert "Timing RGB window: 2.00 ms" in text
    assert "Timing tile layer upload: 0.25 ms" in text
    assert "Timing tile layer RGB window: 1.50 ms" in text
    assert "Timing level sync: 1.00 ms" in text
    assert "Coalesced montage commits: 7" in text
    assert "Tile layer items: visible=50 updated=1 skipped=49" in text
    assert "Tile layer RGB window tiles: 1" in text
    assert "Upload: visible=1.0 KiB histogram=512 B same object=True" in text
    assert "Tile cache last session: cached=3 missing=4" in text
    assert "Workers: visible=1, montage_tile=2" in text
    assert "FFT workers: visible=4, montage_tile=1" in text
