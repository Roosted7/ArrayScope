"""Pure cache-status helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheStatus(str, Enum):
    COLD = "Cold"
    CACHED = "Cached"
    COMPUTING = "Computing"
    READY = "Ready"
    STALE = "Stale"
    ERROR = "Error"


@dataclass(frozen=True)
class CacheStatusSnapshot:
    status: CacheStatus
    message: str = ""


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
