"""Qt-free scheduler request and diagnostics models.

Canonical priority vocabulary lives in the kernel (redesign R0). This module
keeps a compatibility alias only until the legacy controllers are deleted at
the end of R1; new code imports `Priority` from `arrayscope.kernel`.
"""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.kernel.task import Priority as EvalPriority  # noqa: F401


@dataclass(frozen=True)
class FrameTarget:
    semantic_key: object
    viewport_key: object
    presentation_key: object
    quality: str
    deadline_ns: int = 0


@dataclass(frozen=True)
class FrameProgress:
    presented: FrameTarget | None = None
    active: FrameTarget | None = None
    queued_latest: FrameTarget | None = None


@dataclass(frozen=True)
class EvalRequest:
    key: object
    priority: EvalPriority
    generation: int
    replace_group: str
    group_generation: int
    memory_budget_bytes: int | None = None
    frame_target: FrameTarget | None = None
    supersession_key: object | None = None
    supersession_value: object | None = None
    work_item: object | None = None


@dataclass(frozen=True)
class WorkStart:
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
    active_preserved: int = 0
    queued_collapsed: int = 0
    stale_reused: int = 0
    fallback_event_polls: int = 0
    fallback_idle_polls: int = 0
    presented_target: FrameTarget | None = None
    active_target: FrameTarget | None = None
    queued_latest_target: FrameTarget | None = None
    work_lanes: tuple[str, ...] = ()
    work_graph: object | None = None
