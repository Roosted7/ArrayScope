"""Qt-free work admission and lifecycle counters.

The work graph is the control-plane model above concrete Qt controllers.  It
does not run work; it records which work is eligible to run, which queued work
was superseded, and how admitted work completed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter_ns

from arrayscope.core.scheduler import FrameTarget


class WorkLane(StrEnum):
    VISIBLE_PLANNING = "visible_planning"
    VISIBLE_MATERIALIZATION = "visible_materialization"
    DISPLAY_PREPARATION = "display_preparation"
    BACKEND_COMMIT = "backend_commit"
    GUI_FAN_IN = "gui_fan_in"
    HISTOGRAM_REFINEMENT = "histogram_refinement"
    PROFILE_ROI_HOVER = "profile_roi_hover"
    STAGE_MATERIALIZATION = "stage_materialization"
    SPECULATIVE_RESIDENCY = "speculative_residency"


VISIBLE_LANES = frozenset(
    {
        WorkLane.VISIBLE_PLANNING,
        WorkLane.VISIBLE_MATERIALIZATION,
        WorkLane.DISPLAY_PREPARATION,
        WorkLane.BACKEND_COMMIT,
        WorkLane.GUI_FAN_IN,
        WorkLane.STAGE_MATERIALIZATION,
    }
)


@dataclass(frozen=True)
class WorkItem:
    key: object
    lane: WorkLane
    frame_target: FrameTarget | None = None
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
        lane = self.lane if isinstance(self.lane, WorkLane) else WorkLane(str(self.lane))
        object.__setattr__(self, "lane", lane)
        quality = str(self.quality or getattr(self.frame_target, "quality", "") or "")
        object.__setattr__(self, "quality", quality)
        deadline = int(self.deadline_ns or getattr(self.frame_target, "deadline_ns", 0) or 0)
        object.__setattr__(self, "deadline_ns", max(0, deadline))
        object.__setattr__(self, "estimated_cpu_ms", max(0.0, float(self.estimated_cpu_ms or 0.0)))
        object.__setattr__(self, "estimated_bytes", max(0, int(self.estimated_bytes or 0)))
        object.__setattr__(self, "dependency_keys", tuple(self.dependency_keys or ()))
        object.__setattr__(self, "expected_value", max(0.0, float(self.expected_value or 0.0)))
        object.__setattr__(self, "reusable_output", bool(self.reusable_output))


@dataclass(frozen=True)
class WorkAdmissionDecision:
    item: WorkItem
    admitted: bool
    reason: str = "admitted"


@dataclass
class WorkLaneCounters:
    queued: int = 0
    admitted: int = 0
    dropped: int = 0
    superseded: int = 0
    completed: int = 0
    failed: int = 0
    rescheduled: int = 0
    reusable_finished: int = 0
    deadline_missed: int = 0
    blocked_by_budget: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "queued": int(self.queued),
            "admitted": int(self.admitted),
            "dropped": int(self.dropped),
            "superseded": int(self.superseded),
            "completed": int(self.completed),
            "failed": int(self.failed),
            "rescheduled": int(self.rescheduled),
            "reusable_finished": int(self.reusable_finished),
            "deadline_missed": int(self.deadline_missed),
            "blocked_by_budget": int(self.blocked_by_budget),
        }


@dataclass(frozen=True)
class WorkGraphDiagnostics:
    lanes: dict[str, dict[str, int]]
    active: int = 0
    queued: int = 0
    completed_keys: int = 0


@dataclass
class WorkGraph:
    optional_value_threshold: float = 0.0
    _active: dict[object, WorkItem] = field(default_factory=dict, init=False, repr=False)
    _queued: dict[object, WorkItem] = field(default_factory=dict, init=False, repr=False)
    _queued_by_supersession: dict[object, set[object]] = field(default_factory=dict, init=False, repr=False)
    _budget_blocked: set[object] = field(default_factory=set, init=False, repr=False)
    _completed: set[object] = field(default_factory=set, init=False, repr=False)
    _supersession_values: dict[object, object] = field(default_factory=dict, init=False, repr=False)
    _counters: dict[WorkLane, WorkLaneCounters] = field(default_factory=dict, init=False, repr=False)
    _visible_backlog: int = 0

    def submit(
        self,
        item: WorkItem,
        *,
        available_budget: bool = True,
        visible_backlog: bool | None = None,
        now_ns: int | None = None,
        supersedes: bool = True,
    ) -> WorkAdmissionDecision:
        item = _coerce_item(item)
        lane = item.lane
        counters = self._lane_counters(lane)
        if not supersedes and self._is_item_stale(item):
            self._pop_queued(item.key)
            counters.dropped += 1
            return WorkAdmissionDecision(item=item, admitted=False, reason="stale")
        if supersedes:
            self._drop_superseded_queued(item)
            if item.supersession_key is not None:
                self._supersession_values[item.supersession_key] = item.supersession_value
        if self._is_item_stale(item):
            counters.dropped += 1
            return WorkAdmissionDecision(item=item, admitted=False, reason="stale")
        missing_dependencies = tuple(key for key in item.dependency_keys if key not in self._completed)
        if missing_dependencies:
            self._queue_item(item)
            counters.queued += 1
            counters.rescheduled += 1
            return WorkAdmissionDecision(item=item, admitted=False, reason="dependencies")
        if self._optional_blocked(item, available_budget=available_budget, visible_backlog=visible_backlog):
            if item.key not in self._queued or item.key not in self._budget_blocked:
                counters.blocked_by_budget += 1
            if item.key in self._queued:
                self._budget_blocked.add(item.key)
            return WorkAdmissionDecision(item=item, admitted=False, reason="budget")
        if item.deadline_ns and int(now_ns if now_ns is not None else perf_counter_ns()) > item.deadline_ns:
            counters.deadline_missed += 1
            if not _is_visible_item(item):
                counters.dropped += 1
                return WorkAdmissionDecision(item=item, admitted=False, reason="deadline")
        self._pop_queued(item.key)
        previous_active = self._active.get(item.key)
        self._active[item.key] = item
        counters.admitted += 1
        if previous_active is None and _is_visible_item(item):
            self._visible_backlog += 1
        return WorkAdmissionDecision(item=item, admitted=True)

    def complete_inline(self, item: WorkItem, *, available_budget: bool = True) -> WorkAdmissionDecision:
        decision = self.submit(item, available_budget=available_budget)
        if decision.admitted:
            self.complete(item.key)
        return decision

    def admit_ready(self, *, available_budget: bool = True) -> tuple[WorkAdmissionDecision, ...]:
        decisions: list[WorkAdmissionDecision] = []
        for item in sorted(tuple(self._queued.values()), key=_admission_sort_key):
            if item.key not in self._queued:
                continue
            if any(key not in self._completed for key in item.dependency_keys):
                continue
            decisions.append(
                self.submit(
                    item,
                    available_budget=available_budget,
                    visible_backlog=bool(self.visible_backlog),
                    supersedes=False,
                )
            )
        return tuple(decisions)

    def complete(self, key: object, *, reusable_output: bool = False, stale: bool = False) -> None:
        item = self._active.pop(key, None)
        if item is None:
            return
        stale = bool(stale or self._is_item_stale(item))
        counters = self._lane_counters(item.lane)
        if _is_visible_item(item):
            self._visible_backlog = max(0, int(self._visible_backlog) - 1)
        if stale and (reusable_output or item.reusable_output):
            counters.reusable_finished += 1
            self._completed.add(item.key)
            return
        if stale:
            counters.dropped += 1
            return
        counters.completed += 1
        self._completed.add(item.key)

    def fail(self, key: object, *, stale: bool = False) -> None:
        item = self._active.pop(key, None)
        if item is None:
            return
        counters = self._lane_counters(item.lane)
        if _is_visible_item(item):
            self._visible_backlog = max(0, int(self._visible_backlog) - 1)
        if stale or self._is_item_stale(item):
            counters.dropped += 1
        else:
            counters.failed += 1

    def drop(self, key: object, *, stale: bool = True) -> None:
        item = self._pop_queued(key)
        if item is None:
            item = self._active.pop(key, None)
        if item is None:
            return
        if _is_visible_item(item):
            self._visible_backlog = max(0, int(self._visible_backlog) - 1)
        self._lane_counters(item.lane).dropped += 1

    def reschedule(self, key: object, *, reason: str = "budget") -> None:
        del reason
        item = self._active.pop(key, None)
        was_active = item is not None
        if item is None:
            item = self._queued.get(key)
        if item is None:
            return
        if was_active and _is_visible_item(item):
            self._visible_backlog = max(0, int(self._visible_backlog) - 1)
        self._queue_item(item)
        counters = self._lane_counters(item.lane)
        counters.rescheduled += 1
        counters.queued += 1

    def mark_completed(self, key: object) -> None:
        self._completed.add(key)

    def is_stale(self, item: WorkItem | None) -> bool:
        return False if item is None else self._is_item_stale(_coerce_item(item))

    def diagnostics(self) -> WorkGraphDiagnostics:
        lanes = {}
        for lane, counters in self._counters.items():
            values = counters.as_dict()
            if any(values.values()):
                lanes[lane.value] = values
        return WorkGraphDiagnostics(
            lanes=lanes,
            active=len(self._active),
            queued=len(self._queued),
            completed_keys=len(self._completed),
        )

    @property
    def visible_backlog(self) -> int:
        return int(self._visible_backlog)

    def _optional_blocked(
        self,
        item: WorkItem,
        *,
        available_budget: bool,
        visible_backlog: bool | None,
    ) -> bool:
        if _is_visible_item(item):
            return False
        if not bool(available_budget):
            return True
        backlog = self.visible_backlog > 0 if visible_backlog is None else bool(visible_backlog)
        if backlog and item.expected_value <= 0.0:
            return True
        return item.expected_value < float(self.optional_value_threshold)

    def _drop_superseded_queued(self, item: WorkItem) -> None:
        if item.supersession_key is None:
            return
        for key in tuple(self._queued_by_supersession.get(item.supersession_key, ())):
            queued = self._queued.get(key)
            if queued is None:
                self._discard_queued_index(item.supersession_key, key)
                continue
            if queued.supersession_value == item.supersession_value:
                continue
            self._pop_queued(key)
            counters = self._lane_counters(queued.lane)
            counters.superseded += 1
            counters.dropped += 1

    def _is_item_stale(self, item: WorkItem) -> bool:
        if item.supersession_key is None:
            return False
        current = self._supersession_values.get(item.supersession_key, item.supersession_value)
        return current != item.supersession_value

    def _lane_counters(self, lane: WorkLane) -> WorkLaneCounters:
        lane = lane if isinstance(lane, WorkLane) else WorkLane(str(lane))
        return self._counters.setdefault(lane, WorkLaneCounters())

    def _queue_item(self, item: WorkItem) -> None:
        previous = self._queued.get(item.key)
        if previous is not None and previous.supersession_key != item.supersession_key:
            self._discard_queued_index(previous.supersession_key, item.key)
        self._queued[item.key] = item
        self._budget_blocked.discard(item.key)
        if item.supersession_key is not None:
            self._queued_by_supersession.setdefault(item.supersession_key, set()).add(item.key)

    def _pop_queued(self, key: object) -> WorkItem | None:
        item = self._queued.pop(key, None)
        if item is not None:
            self._budget_blocked.discard(key)
            self._discard_queued_index(item.supersession_key, key)
        return item

    def _discard_queued_index(self, supersession_key: object | None, key: object) -> None:
        if supersession_key is None:
            return
        keys = self._queued_by_supersession.get(supersession_key)
        if keys is None:
            return
        keys.discard(key)
        if not keys:
            self._queued_by_supersession.pop(supersession_key, None)


def _coerce_item(item: WorkItem) -> WorkItem:
    if not isinstance(item, WorkItem):
        raise TypeError("work graph requires a WorkItem")
    return item


def _is_visible_item(item: WorkItem) -> bool:
    return item.lane in VISIBLE_LANES and item.quality != "retained"


def complete_inline_work(owner, item: WorkItem) -> None:
    graph = getattr(owner, "work_graph", None)
    if graph is not None:
        graph.complete_inline(item)


def _admission_sort_key(item: WorkItem) -> tuple[int, float, float, str]:
    lane_rank = 0 if _is_visible_item(item) else 1
    deadline = float("inf") if int(item.deadline_ns or 0) <= 0 else float(item.deadline_ns)
    return (lane_rank, deadline, -float(item.expected_value), repr(item.key))


__all__ = [
    "WorkAdmissionDecision",
    "WorkGraph",
    "WorkGraphDiagnostics",
    "WorkItem",
    "WorkLane",
    "WorkLaneCounters",
    "VISIBLE_LANES",
    "complete_inline_work",
]
