import json

from arrayscope.core.cache_status import CacheDiagnosticsSnapshot, CacheStatus
from arrayscope.core.diagnostics_jsonl import (
    diagnostics_jsonl_line,
    diagnostics_snapshot_record,
    diagnostics_start_record,
    diagnostics_to_jsonable,
)
from arrayscope.core.memory_policy import MemoryProfileChoice, compute_memory_policy
from arrayscope.core.runtime_diagnostics import (
    CanvasPreserveRuntimeDiagnostics,
    MontageTimingDiagnostics,
    MontageRuntimeDiagnostics,
    RenderRuntimeDiagnostics,
    RenderTimingDiagnostics,
    WindowRuntimeDiagnostics,
    format_runtime_diagnostics,
    format_runtime_diagnostics_sections,
)
from arrayscope.operations.stage_cache import StageCacheDiagnostics
from arrayscope.window.stage_warmup import StageWarmupDecision
from arrayscope.window.montage_prefetch import MontagePrefetchDecision
from arrayscope.core.scheduler import SchedulerDiagnostics


def _cache():
    return CacheDiagnosticsSnapshot(status=CacheStatus.READY, entries=0, bytes_used=0, max_bytes=1024, hit_rate=None)


def _snapshot():
    policy = compute_memory_policy(profile=MemoryProfileChoice.BALANCED, render_cap_mb=512, input_nbytes=1, system=None)
    scheduler = SchedulerDiagnostics("visible", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return WindowRuntimeDiagnostics(
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
            last_miss="stage=1",
            last_store="stage=1",
        ),
        schedulers=(scheduler,),
        render=RenderRuntimeDiagnostics(last_request_key="('image', b'\\xf9\\x7f\\x10')"),
        montage=MontageRuntimeDiagnostics(
            active=False,
            display_mode="tile_layer",
            deferred_display_tiles=6,
            backend_setting="auto",
            backend_chosen="tile_layer",
            backend_reason="RGB/complex montage canvas pixels 3000000 > 2000000",
            tile_compute_cache_hits=3,
            tile_compute_stage_backed=4,
            tile_compute_direct=1,
            tile_compute_waiting_for_stage=2,
            lead_direct_tiles=1,
            retained_stage_index=3,
            retained_stage_decision="hit",
        ),
        canvas_preserve=CanvasPreserveRuntimeDiagnostics(events=("start gen=1",)),
        render_timing=RenderTimingDiagnostics(last_render_sync_ms=1.25),
        stage_warmup=StageWarmupDecision("scheduled", candidate_bytes=128, budget_bytes=1024, reason="tiles wait for shared stage"),
        montage_prefetch=(MontagePrefetchDecision(12, 12, "skipped_stage_missing", "would recompute expensive stage per tile"),),
        montage_timing=MontageTimingDiagnostics(
            last_viewport_plan_ms=0.5,
            last_cache_resolve_ms=1.5,
            last_stage_plan_ms=0.75,
            last_session_setup_ms=2.25,
            last_initial_commit_ms=3.5,
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
            tile_layer_resident_items=80,
            tile_layer_storage_capacity=128,
            tile_layer_storage_rebuilds=1,
            tile_layer_storage_evictions=2,
            tile_layer_texture_uploads=3,
            tile_layer_texture_upload_bytes=4096,
            tile_layer_vertex_uploads=1,
            tile_layer_level_updates=1,
            tile_layer_estimated_gpu_bytes=8192,
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


def test_diagnostics_jsonl_serializes_nested_snapshot_and_records():
    snapshot = _snapshot()

    start = diagnostics_start_record(
        snapshot,
        recorded_at="2026-06-20T10:00:00+00:00",
        app_version="0.0.test",
        cwd="/tmp/project",
        pid=123,
        python_version="3.test",
        platform="test-platform",
        interval_ms=500,
    )
    record = diagnostics_snapshot_record(snapshot, sequence=7, recorded_at="2026-06-20T10:00:01+00:00")
    decoded = json.loads(diagnostics_jsonl_line(record))

    assert start["schema_version"] == 1
    assert start["event"] == "start"
    assert start["config"]["derived_shape"] == [4, 5]
    assert start["config"]["memory_profile"] == "balanced"
    assert decoded["event"] == "snapshot"
    assert decoded["sequence"] == 7
    assert decoded["diagnostics"]["memory_policy"]["profile"] == "balanced"
    assert decoded["diagnostics"]["derived_shape"] == [4, 5]
    assert decoded["diagnostics"]["schedulers"][0]["name"] == "visible"
    assert decoded["diagnostics"]["canvas_preserve"]["events"] == ["start gen=1"]


def test_diagnostics_jsonable_unknown_objects_are_stable_strings():
    class LocalThing:
        pass

    value = diagnostics_to_jsonable({"thing": LocalThing(), "values": {3, 1, 2}, "status": CacheStatus.READY})

    assert value["thing"].endswith(".LocalThing>")
    assert value["values"] == [1, 2, 3]
    assert value["status"] == "Ready"


def test_format_runtime_diagnostics_includes_all_major_sections():
    snapshot = _snapshot()

    text = format_runtime_diagnostics(snapshot)

    for heading in ("Realtime", "Feedback", "Montage", "Render", "Schedulers", "Caches", "Memory", "Compute", "FFT", "Canvas Preserve", "Operations"):
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
    assert "Request: captured" in text
    assert "\\xf9" not in text
    assert "Coalescer: pending=False, interactive=False" in text
    assert "Timing render sync: 1.25 ms" in text
    assert "Timing worker queue wait: n/a" in text
    assert "Timing canvas compose: 2.50 ms" in text
    assert "Display mode: tile_layer" in text
    assert "Display backend: tile_layer (setting=auto, fallback=canvas)" in text
    assert "Display backend reason: RGB/complex montage" in text
    assert "Reusable stage: stage=3 hit, repeated per tile=no" in text
    assert "Tile compute: cache_hit=3 stage_backed=4 direct=1 waiting_stage=2" in text
    assert "Lead direct tiles: 1" in text
    assert "Timing viewport plan: 0.50 ms" in text
    assert "Timing cache resolve: 1.50 ms" in text
    assert "Timing stage plan: 0.75 ms" in text
    assert "Timing session setup: 2.25 ms" in text
    assert "Timing initial commit: 3.50 ms" in text
    assert "Tile layer:\n  visible=50 resident=80/128 updated=1 skipped=49 rgb_tiles=1" in text
    assert "Timing visible upload: 10.00 ms" in text
    assert "Timing histogram upload: 5.00 ms" in text
    assert "Timing histogram recompute: 3.00 ms" in text
    assert "Timing RGB window: 2.00 ms" in text
    assert "Timing tile layer upload: 0.25 ms" in text
    assert "Timing tile layer RGB window: 1.50 ms" in text
    assert "Timing level sync: 1.00 ms" in text
    assert "Coalesced montage commits: 7" in text
    assert "Tile layer items: visible=50 resident=80/128 updated=1 skipped=49" in text
    assert "Tile layer RGB window tiles: 1" in text
    assert "Tile layer storage: rebuilds=1 evictions=2 pages=0/0 near=0 warm=0 gpu=8.0 KiB budget=0 B max_texture=n/a cpu_shadow=0 B" in text
    assert "Tile layer submissions: textures=3 bytes=4.0 KiB vertices=1 levels=1" in text
    assert "Upload: visible=1.0 KiB histogram=512 B same object=True" in text
    assert "Tile cache last session: cached=3 missing=4" in text
    assert "Workers: visible=1, montage_tile=2" in text
    assert "FFT workers: visible=4, montage_tile=1" in text
    assert "Inactive:" in text
    assert "Context:\n" in text


def test_runtime_diagnostics_avoids_long_feedback_worker_lines():
    from arrayscope.core.compute_policy import ComputeLane, ComputePolicy
    from arrayscope.core.resource_governor import (
        FeedbackChannelDiagnostics,
        LaneWorkerDecision,
        ResourceGovernorDiagnostics,
        ResourcePressure,
        ResourcePressureState,
    )

    policy = compute_memory_policy(profile=MemoryProfileChoice.BALANCED, render_cap_mb=512, input_nbytes=1, system=None)
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
        ),
        schedulers=(SchedulerDiagnostics("visible", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),),
        render=RenderRuntimeDiagnostics(last_context_summary=" ".join(["verylongcontext"] * 60)),
        montage=MontageRuntimeDiagnostics(active=False),
        canvas_preserve=CanvasPreserveRuntimeDiagnostics(),
        fft_backend_choice="auto",
        fft_backend_resolved="numpy",
        fft_workers_choice="auto",
        fft_workers_resolved=1,
        operation_count=0,
        derived_shape=(1,),
        derived_dtype="float32",
        pipeline_peak_bytes=None,
        resource_governor=ResourceGovernorDiagnostics(
            pressure=ResourcePressureState(ResourcePressure.NORMAL, 0.5, ResourcePressure.LOW, ResourcePressure.NORMAL, ""),
            lane_decisions=(
                LaneWorkerDecision(ComputeLane.MONTAGE_TILE, 8, 1, 8, "profile baseline"),
                LaneWorkerDecision(ComputeLane.PREFETCH, 1, 1, 1, "prefetch kept narrow while user-visible work is active"),
            ),
            feedback_channels=(
                FeedbackChannelDiagnostics("montage_commit", 15.0, 1, 15.0, 15.0, 4.0, 1, 30),
                FeedbackChannelDiagnostics("roi_refresh", 0.0, 0, None, None, 8.0, 8, 16),
            ),
        ),
    )

    sections = format_runtime_diagnostics_sections(snapshot)

    assert "Lane workers:\n  montage_tile: 8/8" in sections["Feedback"]
    assert "  Inactive:\n    - roi_refresh" in sections["Feedback"]
    assert all(len(line) <= 145 for line in sections["Render"].splitlines())
