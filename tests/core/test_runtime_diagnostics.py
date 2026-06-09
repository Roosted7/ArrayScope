from arrayscope.core.cache_status import CacheDiagnosticsSnapshot, CacheStatus
from arrayscope.core.memory_policy import MemoryProfileChoice, compute_memory_policy
from arrayscope.core.runtime_diagnostics import (
    MontageRuntimeDiagnostics,
    RenderRuntimeDiagnostics,
    WindowRuntimeDiagnostics,
    format_runtime_diagnostics,
)
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
        schedulers=(scheduler,),
        render=RenderRuntimeDiagnostics(),
        montage=MontageRuntimeDiagnostics(active=False),
        fft_backend_choice="auto",
        fft_backend_resolved="numpy",
        fft_workers_choice="auto",
        fft_workers_resolved=1,
        operation_count=0,
        derived_shape=(4, 5),
        derived_dtype="float32",
        pipeline_peak_bytes=None,
    )

    text = format_runtime_diagnostics(snapshot)

    for heading in ("Memory", "Caches", "Schedulers", "Render", "Montage", "FFT", "Operations"):
        assert heading in text
    assert "hit-rate=n/a" in text
