import json
from dataclasses import replace

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
from arrayscope.window.montage_prefetch import MontagePrefetchDecision
from arrayscope.core.frame_targets import FrameTarget, SchedulerDiagnostics


def _cache():
    return CacheDiagnosticsSnapshot(status=CacheStatus.READY, entries=0, bytes_used=0, max_bytes=1024, hit_rate=None)


def _snapshot():
    policy = compute_memory_policy(profile=MemoryProfileChoice.BALANCED, render_cap_mb=512, input_nbytes=1, system=None)
    scheduler = SchedulerDiagnostics(
        "visible",
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        active_preserved=2,
        queued_collapsed=3,
        stale_reused=1,
        presented_target=FrameTarget("semantic", None, "presentation", "exact-visible"),
    )
    return WindowRuntimeDiagnostics(
        memory_policy=policy,
        display_cache=_cache(),
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
        schedulers=(scheduler, SchedulerDiagnostics("idle", 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)),
        render=RenderRuntimeDiagnostics(last_request_key="('image', b'\\xf9\\x7f\\x10')"),
        montage=MontageRuntimeDiagnostics(
            active=False,
            display_mode="tile_layer",
            backend_chosen="tile_layer",
            backend_reason="pyqtgraph supports direct tiled montage payloads",
            tile_lod_desired_factor=4,
            tile_lod_applied_factor=1,
            tile_lod_desired_factor_xy=(4, 2),
            tile_lod_applied_factor_xy=(1, 1),
            tile_lod_source_texels_per_pixel_xy=(8.0, 3.0),
            tile_lod_policy="native-only",
            tile_lod_reason="desired LOD is deferred until asynchronous multi-resolution residency exists",
            tile_lod_page_families=(((1, 2), "mean", 3), ((3, 3), "phase_vector", 1)),
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
        montage_prefetch=(MontagePrefetchDecision(12, 12, "skipped_stage_missing", "would recompute expensive stage per tile"),),
        montage_timing=MontageTimingDiagnostics(
            last_viewport_plan_ms=0.5,
            last_cache_resolve_ms=1.5,
            last_stage_plan_ms=0.75,
            last_session_setup_ms=2.25,
            last_initial_commit_ms=3.5,
            last_tile_payload_build_ms=2.5,
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
            tile_layer_budget_bytes=16384,
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
    assert decoded["diagnostics"]["schedulers"][0]["active_preserved"] == 2
    assert decoded["diagnostics"]["schedulers"][0]["queued_collapsed"] == 3
    assert decoded["diagnostics"]["schedulers"][0]["stale_reused"] == 1
    assert decoded["diagnostics"]["schedulers"][0]["presented_target"]["quality"] == "exact-visible"
    assert decoded["diagnostics"]["canvas_preserve"]["events"] == ["start gen=1"]
    assert decoded["diagnostics"]["montage"]["tile_lod_page_families"] == [
        [[1, 2], "mean", 3],
        [[3, 3], "phase_vector", 1],
    ]


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
    assert "Stage cache last miss: stage=1" in text
    assert "Stage cache last store: stage=1" in text
    assert "Prefetch: tile=12 source=12 decision=skipped_stage_missing reason=would recompute expensive stage per tile" in text
    assert "Request: captured" in text
    assert "\\xf9" not in text
    assert "Coalescer: pending=no interactive=no" in text
    assert "sync=1.2" in text  # render timing group, n/a entries hidden
    assert "queue_wait" not in text  # None timing is hidden, not rendered as n/a
    assert "payload=2.5" in text
    assert "mode=tile_layer" in text
    assert "backend=tile_layer" in text
    assert "Backend reason: pyqtgraph supports direct tiled montage payloads" in text
    assert "LOD: native-only level=0 desired=4(4, 2) applied=1(1, 1) texpp=(8.00,3.00)" in text
    assert "LOD reason: desired LOD is deferred until asynchronous multi-resolution residency exists" in text
    assert "Lifecycle: parked=0 evaluating=0 dangling_claims=0 mismatches=0 identity_rejections=0 dirty=0 BACKEND_STALE=0" in text
    assert "Kernel" in text
    assert "Queues: upserts=0 removals=0 level_scan=0 flush=no final=no" in text
    assert "Reusable stage: stage=3 hit repeated_per_tile=no" in text
    assert "Compute: cache_hit=3 stage_backed=4 direct=1 waiting_stage=2 lead_direct=1" in text
    assert "Plan (ms): viewport=0.5 cache_resolve=1.5 stage_plan=0.8 setup=2.2 first_commit=3.5" in text
    assert "Present (ms): visible=10.0 hist=5.0 hist_recompute=3.0 rgb=2.0 tile_upload=0.2 tile_rgb=1.5 level_sync=1.0" in text
    assert "Flush: upserts_last=0 coalesced=7 cache_session=3/4" in text
    assert "Layer items: visible=50 resident=80/128 (62.5%) created=0 updated=1 shown=0 moved=0 skipped=49 rgb=1" in text
    assert "Layer storage: gpu=8.0 KiB/16.0 KiB (50.0%) pages=0/0 near=0 warm=0 rebuilds=1 evictions=2" in text
    assert "Layer submissions: textures=3 bytes=4.0 KiB vertices=1 levels=1" in text
    assert "Upload: total=5.5 KiB visible=1.0 KiB hist=512 B tile_tex=4.0 KiB same_object=yes" in text
    assert "Workers: visible=1, montage_tile=2" in text
    assert "FFT workers: visible=4, montage_tile=1" in text
    assert "active_preserved=2" in text
    assert "queued_collapsed=3" in text
    assert "stale_reused=1" in text
    assert "Inactive:" in text
    assert "Context:\n" in text


def _nondefault_probe_value(current):
    if isinstance(current, bool):
        return not current
    if isinstance(current, int):
        return int(current) + 7
    if isinstance(current, float):
        return float(current) + 7.5
    if current is None:
        return 7.5
    if isinstance(current, str):
        return current + "-probe"
    if isinstance(current, tuple):
        return (*current, "probe")
    return "probe"


def _assert_every_field_visible(dataclass_type, covered, format_section):
    """ADR 0051 diagnostics contract: a field is either curated (covered set)
    or appears in the auto 'More (non-default)' block — nothing invisible."""
    import dataclasses as _dc

    names = {field.name for field in _dc.fields(dataclass_type)}
    unknown = set(covered) - names
    assert not unknown, f"covered set names unknown fields: {sorted(unknown)}"
    instance = dataclass_type() if dataclass_type is not MontageRuntimeDiagnostics else dataclass_type(active=True)
    for name in sorted(names - set(covered)):
        probe = replace(instance, **{name: _nondefault_probe_value(getattr(instance, name))})
        text = "\n".join(format_section(probe))
        assert f"{name}=" in text, f"field {name} invisible in diagnostics"


def test_every_montage_field_is_visible_in_its_tab():
    from arrayscope.core import runtime_diagnostics as rd

    _assert_every_field_visible(
        MontageRuntimeDiagnostics,
        rd._MONTAGE_COVERED,
        lambda obj: rd._auto_extra_lines(obj, rd._MONTAGE_COVERED),
    )
    _assert_every_field_visible(
        MontageTimingDiagnostics,
        rd._MONTAGE_TIMING_COVERED,
        lambda obj: rd._auto_extra_lines(obj, rd._MONTAGE_TIMING_COVERED),
    )


def test_every_render_field_is_visible_in_its_tab():
    from arrayscope.core import runtime_diagnostics as rd
    from arrayscope.core.runtime_diagnostics import RenderCoalescerDiagnostics

    _assert_every_field_visible(
        RenderRuntimeDiagnostics,
        rd._RENDER_COVERED,
        lambda obj: rd._auto_extra_lines(obj, rd._RENDER_COVERED),
    )
    _assert_every_field_visible(
        RenderTimingDiagnostics,
        rd._RENDER_TIMING_COVERED,
        lambda obj: rd._auto_extra_lines(obj, rd._RENDER_TIMING_COVERED),
    )
    _assert_every_field_visible(
        RenderCoalescerDiagnostics,
        rd._COALESCER_COVERED,
        lambda obj: rd._auto_extra_lines(obj, rd._COALESCER_COVERED),
    )


def test_every_canvas_preserve_field_is_visible_in_its_tab():
    from arrayscope.core import runtime_diagnostics as rd

    _assert_every_field_visible(
        CanvasPreserveRuntimeDiagnostics,
        rd._CANVAS_PRESERVE_COVERED,
        lambda obj: rd._auto_extra_lines(obj, rd._CANVAS_PRESERVE_COVERED),
    )


def test_all_tab_concatenates_every_section():
    snapshot = _snapshot()
    sections = format_runtime_diagnostics_sections(snapshot)
    all_text = format_runtime_diagnostics(snapshot)
    for title, body in sections.items():
        assert f"{title}\n" in all_text
        assert body in all_text


def test_runtime_diagnostics_avoids_long_feedback_worker_lines():
    from arrayscope.core.compute_policy import ComputeLane
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
        display_cache=_cache(),
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
                FeedbackChannelDiagnostics("montage_present_total", 20.0, 1, 20.0, 20.0, 4.0, 1, 30, 1024),
                FeedbackChannelDiagnostics("roi_refresh", 0.0, 0, None, None, 8.0, 8, 16),
            ),
        ),
    )

    sections = format_runtime_diagnostics_sections(snapshot)

    assert "Lane workers:\n  montage_tile: 8/8" in sections["Feedback"]
    assert "Channels:\n  montage_commit:" in sections["Feedback"]
    assert "Telemetry-only:\n  montage_present_total:" in sections["Feedback"]
    assert "UI decisions:" not in sections["Feedback"]
    assert "  Inactive:\n    - roi_refresh" in sections["Feedback"]
    assert all(len(line) <= 145 for line in sections["Render"].splitlines())


def test_runtime_bottleneck_ignores_stale_ui_pressure_when_idle():
    from arrayscope.core.resource_governor import (
        ResourceGovernorDiagnostics,
        ResourcePressure,
        ResourcePressureState,
    )

    snapshot = replace(
        _snapshot(),
        montage_timing=MontageTimingDiagnostics(),
        resource_governor=ResourceGovernorDiagnostics(
            pressure=ResourcePressureState(ResourcePressure.HIGH, 0.5, ResourcePressure.LOW, ResourcePressure.NORMAL, "")
        ),
    )

    sections = format_runtime_diagnostics_sections(snapshot)

    assert "Bottleneck: idle" in sections["Realtime"]


def test_runtime_bottleneck_ignores_stale_rgb_timing_when_idle():
    snapshot = replace(
        _snapshot(),
        montage_timing=MontageTimingDiagnostics(tile_layer_rgb_window_tiles=7),
    )

    sections = format_runtime_diagnostics_sections(snapshot)

    assert "Bottleneck: idle" in sections["Realtime"]


def test_runtime_bottleneck_reports_rgb_only_for_live_presentation_work():
    snapshot = replace(
        _snapshot(),
        montage=replace(_snapshot().montage, pending_payload_upserts=3),
        montage_timing=MontageTimingDiagnostics(tile_layer_rgb_window_tiles=7),
    )

    sections = format_runtime_diagnostics_sections(snapshot)

    assert "Bottleneck: RGB window/upload" in sections["Realtime"]
