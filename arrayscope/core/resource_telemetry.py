"""Host resource telemetry used by adaptive scheduling policy."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from arrayscope.core.memory_policy import SystemMemorySnapshot, sample_system_memory

_CPU_SAMPLERS: dict[int, _CpuSampler] = {}


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


def sample_resource_snapshot(
    *, psutil_module=None, cpu_count: int | None = None
) -> ResourceSnapshot:
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
            system_cpu, process_cpu = _cpu_sampler(psutil_module).read()
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


class _CpuSampler:
    """A warmed, stable pair of psutil CPU-percent samplers.

    `cpu_percent(interval=None)` reports the busy fraction *since the previous
    call on the same object*.  The first call has no previous reading, so it
    returns 0.0 by construction — documented psutil behaviour, not an error.

    That zero is indistinguishable from a genuinely idle host, and the governor
    maps `system_cpu <= 0.0` onto a neutral 0.5 headroom.  An un-warmed sampler
    therefore pins `cpu_headroom` at the neutral value and makes the
    `cpu_headroom < 0.15` branch in `core/resource_governor.py` unreachable.
    Burn the first reading here so the first sample a caller sees is live.
    """

    __slots__ = ("_process", "_psutil")

    def __init__(self, psutil_module) -> None:
        # The module reference is held for the lifetime of the cache entry:
        # the cache is keyed by `id()`, so a collected module could otherwise
        # let a later object reuse the address and inherit this sampler.
        self._psutil = psutil_module
        self._process = psutil_module.Process()
        # The discarded warm-up. Both samplers keep independent deltas.
        psutil_module.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)

    def read(self) -> tuple[float, float]:
        return (
            float(self._psutil.cpu_percent(interval=None)),
            float(self._process.cpu_percent(interval=None)),
        )


def _cpu_sampler(psutil_module) -> _CpuSampler:
    """Return the warmed sampler for this psutil module, creating it once."""

    key = id(psutil_module)
    sampler = _CPU_SAMPLERS.get(key)
    if sampler is None:
        sampler = _CpuSampler(psutil_module)
        _CPU_SAMPLERS[key] = sampler
    return sampler
