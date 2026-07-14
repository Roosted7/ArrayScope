"""Task model: the one vocabulary for schedulable work.

This module is the canonical owner of lanes, priorities, and work metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable


class Lane(str, Enum):
    """Work lane: what kind of pipeline stage a task belongs to.

    Lane membership decides starvation policy (visible lanes are never blocked
    by speculative work) and concurrency quotas, not FIFO ordering.
    """

    __str__ = str.__str__
    __format__ = str.__format__

    VISIBLE_PLANNING = "visible_planning"
    VISIBLE_MATERIALIZATION = "visible_materialization"
    DISPLAY_PREVIEW = "display_preview"
    DISPLAY_PREPARATION = "display_preparation"
    BACKEND_COMMIT = "backend_commit"
    GUI_FAN_IN = "gui_fan_in"
    HISTOGRAM_REFINEMENT = "histogram_refinement"
    PROFILE_ROI_HOVER = "profile_roi_hover"
    STAGE_MATERIALIZATION = "stage_materialization"
    SPECULATIVE_RESIDENCY = "speculative_residency"


VISIBLE_LANES = frozenset(
    {
        Lane.VISIBLE_PLANNING,
        Lane.VISIBLE_MATERIALIZATION,
        Lane.DISPLAY_PREVIEW,
        Lane.DISPLAY_PREPARATION,
        Lane.BACKEND_COMMIT,
        Lane.GUI_FAN_IN,
        Lane.STAGE_MATERIALIZATION,
    }
)

# Spatial ranks are ordinal tile positions and therefore start at zero. Work
# explicitly known not to produce a tile can use this floor; control tasks
# keep the default rank because they may be the producer of the ranked work.
UNRANKED_SCHEDULING_RANK = 1_000_000


class Priority(IntEnum):
    """Execution priority *within* the ready set. Lower runs first.

    Unlike the pre-redesign ``EvalPriority`` (which never reached the thread
    pool), this value genuinely orders worker pulls.
    """

    INTERACTIVE = 0
    VISIBLE_IMAGE = 10
    HOVER = 20
    HISTOGRAM = 30
    LIVE_PROFILE = 40
    SELECTED_ROI = 50
    VISIBLE_ROI = 60
    HIDDEN_ROI = 70
    PREFETCH = 80


@dataclass(frozen=True)
class Supersession:
    """Latest-only family. Submitting a new value makes older values stale.

    ``family`` identifies what is being targeted (e.g. one tile's texture,
    one histogram); ``value`` identifies the concrete target (e.g. a semantic
    key + level). Tasks whose family value is no longer current are dropped
    from the queue, cooperatively cancelled while running (unless
    ``TaskSpec.reusable``), and classified stale at delivery.
    """

    family: object
    value: object


class TaskOutcome(str, Enum):
    __str__ = str.__str__

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"           # finished, but superseded/scope-cleared
    STALE_REUSED = "stale_reused"  # stale, but an on_reuse handler consumed it
    DROPPED = "dropped"       # never ran (superseded, cleared, dep failure, deadline)


@dataclass(frozen=True)
class TaskSpec:
    """Immutable description of one unit of background work.

    ``fn`` runs on a worker thread. With ``pass_token=True`` it receives the
    cooperative ``CancellationToken`` as its only argument. ``fn`` must be
    self-contained: it must not touch Qt, widgets, or any GUI-owned state.

    ``deps`` are task keys that must have COMPLETED before this task becomes
    ready. A failed/dropped dependency drops the dependent (reason
    ``dependency_failed``); silence is never an option.
    """

    key: object
    fn: Callable[..., Any]
    lane: Lane = Lane.VISIBLE_MATERIALIZATION
    priority: Priority = Priority.VISIBLE_IMAGE
    scheduling_rank: int = 0
    scope: str = "default"
    deps: tuple = ()
    supersession: Supersession | None = None
    deadline_ns: int = 0
    estimated_cpu_ms: float = 0.0
    estimated_bytes: int = 0
    expected_value: float = 0.0
    reusable: bool = False
    pass_token: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(str(self.lane)))
        object.__setattr__(self, "priority", Priority(int(self.priority)))
        scheduling_rank = int(self.scheduling_rank)
        if scheduling_rank < 0:
            raise ValueError("scheduling_rank must be non-negative")
        object.__setattr__(self, "scheduling_rank", scheduling_rank)
        object.__setattr__(self, "scope", str(self.scope))
        object.__setattr__(self, "deps", tuple(self.deps or ()))
        object.__setattr__(self, "deadline_ns", max(0, int(self.deadline_ns or 0)))
        object.__setattr__(self, "estimated_cpu_ms", max(0.0, float(self.estimated_cpu_ms or 0.0)))
        object.__setattr__(self, "estimated_bytes", max(0, int(self.estimated_bytes or 0)))
        object.__setattr__(self, "expected_value", max(0.0, float(self.expected_value or 0.0)))

    @property
    def visible(self) -> bool:
        return self.lane in VISIBLE_LANES

    def scope_prefixes(self) -> tuple[str, ...]:
        """This scope plus every ancestor (``"a:b:c"`` → ``a``, ``a:b``, ``a:b:c``).

        Clearing any ancestor scope makes this task stale; this replaces the
        pre-redesign group/child-group bookkeeping maps.
        """

        parts = self.scope.split(":")
        return tuple(":".join(parts[: index + 1]) for index in range(len(parts)))


@dataclass(frozen=True)
class WorkItem:
    """Diagnostic/admission metadata for one unit of work.

    The kernel owns execution and counters; callers use ``WorkItem`` only to
    carry lane, dependency, supersession, deadline, and estimate metadata into
    a ``TaskSpec``.
    """

    key: object
    lane: Lane
    frame_target: object | None = None
    quality: str = ""
    supersession_key: object | None = None
    supersession_value: object | None = None
    deadline_ns: int = 0
    estimated_cpu_ms: float = 0.0
    estimated_bytes: int = 0
    dependency_keys: tuple[object, ...] = ()
    expected_value: float = 0.0
    reusable_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane", Lane(str(self.lane)))
        quality = str(self.quality or getattr(self.frame_target, "quality", "") or "")
        object.__setattr__(self, "quality", quality)
        deadline = int(self.deadline_ns or getattr(self.frame_target, "deadline_ns", 0) or 0)
        object.__setattr__(self, "deadline_ns", max(0, deadline))
        object.__setattr__(self, "estimated_cpu_ms", max(0.0, float(self.estimated_cpu_ms or 0.0)))
        object.__setattr__(self, "estimated_bytes", max(0, int(self.estimated_bytes or 0)))
        object.__setattr__(self, "dependency_keys", tuple(self.dependency_keys or ()))
        object.__setattr__(self, "expected_value", max(0.0, float(self.expected_value or 0.0)))
        object.__setattr__(self, "reusable_output", bool(self.reusable_output))


class CancellationToken:
    """Cooperative cancellation flag, checked by task functions."""

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return bool(self._cancelled)


@dataclass
class LaneCounters:
    """Deterministic per-lane counters for kernel-owned work."""

    queued: int = 0
    admitted: int = 0
    started: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    dropped: int = 0
    superseded: int = 0
    stale: int = 0
    stale_reused: int = 0
    deadline_missed: int = 0
    blocked_by_quota: int = 0
    dependency_failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


def complete_inline_work(owner, item: WorkItem) -> None:
    """Record bounded inline work against the owning kernel, when present."""

    kernel = getattr(owner, "kernel", None)
    if kernel is None:
        kernel = getattr(getattr(owner, "win", None), "kernel", None)
    if kernel is not None and hasattr(kernel, "note_inline_work"):
        kernel.note_inline_work(item)


__all__ = [
    "CancellationToken",
    "Lane",
    "LaneCounters",
    "Priority",
    "Supersession",
    "TaskOutcome",
    "TaskSpec",
    "UNRANKED_SCHEDULING_RANK",
    "VISIBLE_LANES",
    "WorkItem",
    "complete_inline_work",
]
