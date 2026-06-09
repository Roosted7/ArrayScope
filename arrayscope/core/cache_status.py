"""Pure cache-status helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheStatus(str, Enum):
    COLD = "Cold"
    CACHED = "Cached"
    COMPUTING = "Computing"
    PREFETCHING = "Prefetching"
    READY = "Ready"
    STALE = "Stale"
    EVICTED = "Evicted"
    STALE_IGNORED = "Stale ignored"
    ERROR = "Error"


@dataclass(frozen=True)
class CacheStatusSnapshot:
    status: CacheStatus
    message: str = ""


@dataclass(frozen=True)
class CacheDiagnosticsSnapshot:
    status: CacheStatus
    message: str = ""
    entries: int = 0
    bytes_used: int = 0
    max_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    last_eval_ms: float | None = None
    hit_rate: float | None = None
    prefetch_scheduled: int = 0
    prefetch_deduped: int = 0
    prefetch_limited: int = 0
    prefetch_skipped: int = 0
    prefetch_stored: int = 0
    prefetch_stale: int = 0
    degraded_evaluations: int = 0
    refused_evaluations: int = 0
    chunked_evaluations: int = 0
    cancelled_evaluations: int = 0
    scheduler_pending: int = 0
    scheduler_running: int = 0
    scheduler_cancelled: int = 0
    scheduler_stale: int = 0


def cache_status_for_hit(hit: bool, has_error: bool = False) -> CacheStatusSnapshot:
    if has_error:
        return CacheStatusSnapshot(CacheStatus.ERROR, "Evaluation error")
    if hit:
        return CacheStatusSnapshot(CacheStatus.CACHED, "Using cached result")
    return CacheStatusSnapshot(CacheStatus.COMPUTING, "Evaluating")


def cache_status_ready(message: str = "Evaluated and cached") -> CacheStatusSnapshot:
    return CacheStatusSnapshot(CacheStatus.READY, message)


def cache_status_computing(message: str = "Evaluating") -> CacheStatusSnapshot:
    return CacheStatusSnapshot(CacheStatus.COMPUTING, message)


def cache_status_error(error) -> CacheStatusSnapshot:
    return CacheStatusSnapshot(CacheStatus.ERROR, f"Evaluation error: {error}")


def cache_status_prefetching(message: str = "Prefetching") -> CacheStatusSnapshot:
    return CacheStatusSnapshot(CacheStatus.PREFETCHING, message)


def cache_status_stale_ignored(message: str = "Ignored stale result") -> CacheStatusSnapshot:
    return CacheStatusSnapshot(CacheStatus.STALE_IGNORED, message)
