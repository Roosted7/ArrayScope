"""Host resource telemetry used by adaptive scheduling policy."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

from arrayscope.core.memory_policy import SystemMemorySnapshot, sample_system_memory


@dataclass(frozen=True)
class CpuSnapshot:
    logical_count: int
    process_cpu_percent: float | None
    system_cpu_percent: float | None
    load_average_1m: float | None
    source: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceSnapshot:
    memory: SystemMemorySnapshot
    cpu: CpuSnapshot
    timestamp_monotonic: float


def sample_resource_snapshot(*, psutil_module=None, cpu_count: int | None = None) -> ResourceSnapshot:
    memory = sample_system_memory(psutil_module=psutil_module)
    logical_count = max(1, int(cpu_count if cpu_count is not None else (os.cpu_count() or 1)))
    warnings: list[str] = []
    process_cpu = None
    system_cpu = None
    source = "fallback"
    if psutil_module is None:
        try:
            import psutil as psutil_module
        except Exception:
            psutil_module = None
    if psutil_module is not None:
        try:
            system_cpu = float(psutil_module.cpu_percent(interval=None))
            process_cpu = float(psutil_module.Process().cpu_percent(interval=None))
            source = "psutil"
        except Exception:
            warnings.append("psutil CPU telemetry unavailable")
    else:
        warnings.append("psutil unavailable; CPU telemetry disabled")
    try:
        load_average = float(os.getloadavg()[0])
    except Exception:
        load_average = None
    return ResourceSnapshot(
        memory=memory,
        cpu=CpuSnapshot(
            logical_count=logical_count,
            process_cpu_percent=process_cpu,
            system_cpu_percent=system_cpu,
            load_average_1m=load_average,
            source=source,
            warnings=tuple(warnings),
        ),
        timestamp_monotonic=time.monotonic(),
    )
