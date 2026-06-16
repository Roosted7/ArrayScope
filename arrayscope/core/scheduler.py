"""Qt-free scheduler request and diagnostics models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class EvalPriority(IntEnum):
    VISIBLE_IMAGE = 0
    LIVE_PROFILE = 10
    SELECTED_ROI = 20
    HOVER_EXACT = 30
    PREFETCH = 40


@dataclass(frozen=True)
class EvalRequest:
    key: object
    priority: EvalPriority
    generation: int
    replace_group: str
    group_generation: int
    memory_budget_bytes: int | None = None


@dataclass(frozen=True)
class PrefetchStart:
    scheduled: bool
    reason: str = "scheduled"


@dataclass(frozen=True)
class SchedulerDiagnostics:
    name: str
    max_workers: int
    pending: int
    running: int
    queued: int
    started: int
    cancelled: int
    stale: int
    completed: int
    failed: int
    prefetch_scheduled: int
    prefetch_deduped: int
    prefetch_limited: int
    prefetch_idle_blocked: int
    prefetch_visible_busy_blocked: int
    prefetch_cost_blocked: int
