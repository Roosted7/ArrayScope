"""Qt-free frame-target request and diagnostics models.

Priority vocabulary lives in the kernel; import ``Priority`` from
``arrayscope.kernel``. This module holds only the shared frame-target and
diagnostics dataclasses exchanged between the window layer and the kernel
evaluation adapter.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    kernel: object | None = None
