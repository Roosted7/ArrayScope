"""The kernel scheduler: one owner for background execution.

Replaces the pre-redesign split between bookkeeping-only admission, per-purpose
FIFO worker pools where priorities were decorative, and per-call-site staleness
checks. Here, priorities order real worker pulls, dependencies gate real
execution, and staleness has exactly one arbiter.

Staleness has three sources, all decided here:

1. **Supersession** — the task's family points at an older value.
2. **Scope clear** — the task's scope (or an ancestor, ``"a:b"`` is a child
   of ``"a"``) was cleared after the task was submitted.
3. **Key resubmission** — a newer instance of the same key was submitted.

Locking: one condition variable (`self._cond`, on `self._lock`) guards all
mutable state. Worker threads block on it; every state change that can make
a task ready notifies it. Handlers are never invoked under the lock and
never from worker threads.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import traceback
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import monotonic, perf_counter_ns
from typing import Any

from arrayscope.core.trace import emit_trace
from arrayscope.kernel.completions import CompletionEvent, CompletionQueue
from arrayscope.kernel.task import (
    CancellationToken,
    Lane,
    LaneCounters,
    Priority,
    Supersession,
    TaskOutcome,
    TaskSpec,
    WorkItem,
)
from arrayscope.operations.cancellation import EvaluationCancelled

_QUEUED = "queued"
_PARKED_DEPS = "parked_deps"
_PARKED_QUOTA = "parked_quota"
_RUNNING = "running"
_DONE = "done"

_SENTINEL = object()


@dataclass
class _Record:
    seq: int
    spec: TaskSpec
    scope_snapshot: tuple[tuple[str, int], ...]
    token: CancellationToken = field(default_factory=CancellationToken)
    state: str = _QUEUED
    superseded: bool = False
    unmet_deps: set = field(default_factory=set)
    on_done: Callable[[Any], None] | None = None
    on_error: Callable[[BaseException], None] | None = None
    on_stale: Callable[[], None] | None = None
    on_reuse: Callable[[Any], None] | None = None
    submitted_ns: int = 0
    quota_blocked_noted: bool = False


class TaskHandle:
    """Caller-side view of one submitted task instance."""

    __slots__ = ("_kernel", "_seq", "key")

    def __init__(self, kernel: Kernel, seq: int, key: object) -> None:
        self._kernel = kernel
        self._seq = seq
        self.key = key

    def cancel(self) -> None:
        self._kernel._cancel_seq(self._seq)


@dataclass(frozen=True)
class KernelDiagnostics:
    lanes: dict[str, dict[str, int]]
    queued: int
    running: int
    active: int
    parked_deps: int
    parked_quota: int
    workers: int
    completed_keys: int
    scopes: int
    oldest_queued_ms: float
    visible_backlog: int = 0


class Kernel:
    """Submit → schedule → execute → deliver, under one lock and one queue."""

    def __init__(
        self,
        backend=None,
        *,
        completion_queue: CompletionQueue | None = None,
        speculative_fraction: float = 0.25,
        optional_value_threshold: float = 0.0,
        handler_error_hook: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        from arrayscope.kernel.workers import ThreadWorkerBackend

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self.completions = completion_queue if completion_queue is not None else CompletionQueue()
        self._backend = backend if backend is not None else ThreadWorkerBackend()
        self._seq = itertools.count(1)
        self._epoch = itertools.count(1)
        self._records: dict[int, _Record] = {}
        self._by_key: dict[object, int] = {}
        self._ready: list[tuple[int, int, int, float, int]] = []
        self._parked_quota: list[int] = []
        self._dep_waiters: dict[object, set[int]] = {}
        # Keys whose newest instance completed non-stale, mapped to the scope
        # they completed under. Satisfies deps. Bounded: ``clear_scope`` purges
        # the cleared scope's entries (a cleared result must not satisfy a
        # later dependency), and ``forget_results`` releases memory for long
        # sessions without touching staleness.
        self._completed_keys: dict[object, str] = {}
        self._scope_epochs: dict[str, int] = {}
        self._supersession: dict[object, object] = {}
        self._counters: dict[Lane, LaneCounters] = {}
        self._running_visible = 0
        self._running_optional = 0
        self._running_by_lane: dict[Lane, int] = {}
        self._lane_quotas: dict[Lane, int] = {}
        self._queued_visible = 0
        self._speculative_fraction = min(1.0, max(0.05, float(speculative_fraction)))
        self.optional_value_threshold = float(optional_value_threshold)
        self._handler_error_hook = handler_error_hook or _default_handler_error_hook
        self._shutting_down = False
        self._last_shutdown_diagnostics: tuple[dict[str, object], ...] = ()
        # Set under the lock when a cancel/drop released parked work; public
        # entry points flush it into a backend wake after releasing the lock
        # (the inline backend executes task functions from ``wake``).
        self._release_wake_pending = False
        self._backend.attach(self)

    # ------------------------------------------------------------------ API

    def submit(
        self,
        spec: TaskSpec,
        *,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_stale: Callable[[], None] | None = None,
        on_reuse: Callable[[Any], None] | None = None,
    ) -> TaskHandle | None:
        """Admit one task. Returns None when dropped at the door (stale/closed)."""

        with self._lock:
            if self._shutting_down:
                return None
            seq = next(self._seq)
            record = _Record(
                seq=seq,
                spec=spec,
                scope_snapshot=self._scope_snapshot_locked(spec),
                submitted_ns=perf_counter_ns(),
            )
            record.on_done, record.on_error = on_done, on_error
            record.on_stale, record.on_reuse = on_stale, on_reuse
            counters = self._lane(spec.lane)
            counters.queued += 1
            if spec.supersession is not None:
                self._advance_supersession_locked(spec.supersession, exclude_seq=seq)
            previous_seq = self._by_key.get(spec.key)
            if previous_seq is not None:
                self._supersede_instance_locked(previous_seq)
            self._records[seq] = record
            self._by_key[spec.key] = seq
            record.unmet_deps = {dep for dep in spec.deps if dep not in self._completed_keys}
            if record.unmet_deps:
                record.state = _PARKED_DEPS
                for dep in record.unmet_deps:
                    self._dep_waiters.setdefault(dep, set()).add(seq)
                wake = False
            else:
                self._enqueue_ready_locked(record)
                wake = True
            counters.admitted += 1
            if spec.visible:
                self._queued_visible += 1
            handle = TaskHandle(self, seq, spec.key)
            self._cond.notify_all()
        if wake:
            self._backend.wake()
        # Superseding the previous instance may have dropped the last
        # blocking item while the replacement parked on dependencies.
        self._flush_release_wake()
        emit_trace(
            "kernel_submit",
            task_seq=int(seq),
            key=spec.key,
            lane=str(spec.lane),
            priority=int(spec.priority),
            scheduling_rank=int(spec.scheduling_rank),
            presentation_phase=int(spec.presentation_phase),
            coverage_pass_open=bool(spec.coverage_pass_open),
            session_id=int(spec.session_id),
            tile_number=int(spec.tile_number),
            scopes=spec.scope_prefixes(),
            deps=tuple(spec.deps),
            visible=bool(spec.visible),
            supersession=(
                None
                if spec.supersession is None
                else (spec.supersession.family, spec.supersession.value)
            ),
        )
        return handle

    def submit_speculative_batch(
        self,
        *,
        kind: str,
        scope: str,
        generation: object,
        fn: Callable[..., Any],
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_stale: Callable[[], None] | None = None,
        priority: Priority = Priority.PREFETCH,
        lane: Lane = Lane.SPECULATIVE_RESIDENCY,
        key: object | None = None,
        max_items: int = 0,
        pass_token: bool = False,
        presentation_phase: int = 0,
        coverage_pass_open: bool = False,
        session_id: int = 0,
        tile_number: int = -1,
    ) -> TaskHandle | None:
        """Submit one supersedable speculative batch.

        Callers own the side-effect mechanics; the kernel owns batch identity,
        latest-only staleness, and lane admission.
        """

        batch_key = key if key is not None else (str(kind), str(scope), generation)
        value = (generation, max(0, int(max_items or 0)))
        return self.submit(
            TaskSpec(
                key=batch_key,
                fn=fn,
                lane=lane,
                priority=priority,
                scope=scope,
                supersession=Supersession((str(kind), str(scope)), value),
                reusable=False,
                pass_token=pass_token,
                presentation_phase=presentation_phase,
                coverage_pass_open=coverage_pass_open,
                session_id=session_id,
                tile_number=tile_number,
            ),
            on_done=on_done,
            on_error=on_error,
            on_stale=on_stale,
        )

    def supersede(self, family: object, value: object) -> None:
        """Advance a supersession family without submitting a replacement."""

        with self._lock:
            self._advance_supersession_locked(Supersession(family, value))
            self._cond.notify_all()
        self._flush_release_wake()

    def clear_scope(self, scope: str) -> None:
        """Make every task in ``scope`` (and child scopes) stale.

        Queued/parked tasks are dropped immediately; running tasks get their
        token cancelled (reusable tasks may still finish and be reused).
        """

        scope = str(scope)
        with self._lock:
            self._scope_epochs[scope] = next(self._epoch)
            self._purge_completed_locked(scope)
            for record in tuple(self._records.values()):
                if not self._record_is_stale_locked(record):
                    continue
                if record.state in (_QUEUED, _PARKED_DEPS, _PARKED_QUOTA):
                    self._drop_record_locked(record, reason="scope_cleared")
                elif record.state == _RUNNING and not record.spec.reusable:
                    record.token.cancel()
            self._cond.notify_all()
        self._flush_release_wake()

    def forget_results(self, prefix: str) -> int:
        """Forget completion memory for ``prefix`` and its child scopes.

        Releases dependency-satisfaction records for long sessions without
        making anything stale: live tasks keep running, and epochs are
        untouched. A later submission depending on a forgotten key parks
        until that key is produced again. Returns the number of keys
        forgotten. Inline work (``note_inline_work``) completes under the
        root scope ``""`` and is forgotten by ``forget_results("")``.
        """

        with self._lock:
            forgotten = self._purge_completed_locked(str(prefix))
        emit_trace("kernel_forget_results", prefix=str(prefix), forgotten=int(forgotten))
        return forgotten

    def rerank_unstarted_tile_tasks(
        self,
        *,
        session_id: int,
        scheduling_ranks: dict[int, int],
    ) -> int:
        """Apply current-camera tile ranks to every unstarted task in a session.

        Camera changes do not invalidate materialization, so retargeting must
        not cancel and resubmit unchanged work. The kernel instead updates the
        immutable task descriptions for queued/dependency/quota-parked records
        and atomically rebuilds its ready heap. Running work is deliberately
        left alone; only work which has not started can still be reordered.
        """

        session_id = int(session_id)
        if session_id <= 0:
            return 0
        normalized_ranks = {
            int(tile_number): int(scheduling_rank)
            for tile_number, scheduling_rank in scheduling_ranks.items()
        }
        if any(scheduling_rank < 0 for scheduling_rank in normalized_ranks.values()):
            raise ValueError("scheduling ranks must be non-negative")
        if not normalized_ranks:
            return 0

        updated = 0
        ready_updated = 0
        parked_updated = 0
        updated_tiles: set[int] = set()
        with self._lock:
            for record in self._records.values():
                if record.state not in (_QUEUED, _PARKED_DEPS, _PARKED_QUOTA):
                    continue
                spec = record.spec
                if int(spec.session_id) != session_id:
                    continue
                scheduling_rank = normalized_ranks.get(int(spec.tile_number))
                if scheduling_rank is None or scheduling_rank == int(spec.scheduling_rank):
                    continue
                record.spec = replace(spec, scheduling_rank=scheduling_rank)
                updated += 1
                updated_tiles.add(int(spec.tile_number))
                if record.state == _QUEUED:
                    ready_updated += 1
                else:
                    parked_updated += 1
            if updated:
                self._ready = [
                    self._ready_entry_locked(record)
                    for record in self._records.values()
                    if record.state == _QUEUED
                ]
                heapq.heapify(self._ready)
                self._cond.notify_all()

        if updated:
            emit_trace(
                "kernel_rerank",
                session_id=session_id,
                updated_tasks=int(updated),
                updated_tiles=len(updated_tiles),
                ready_updated=int(ready_updated),
                parked_updated=int(parked_updated),
            )
            self._backend.wake()
        return int(updated)

    def set_lane_quota(self, lane: Lane, quota: int | None) -> None:
        """Set or clear the maximum concurrent tasks for one lane.

        The thread backend supplies physical workers; lane quotas shape which
        ready records those workers may pull. ``0`` parks the lane; ``None``
        clears the hint.
        """

        lane = Lane(str(lane))
        normalized = None if quota is None else max(0, int(quota))
        with self._lock:
            current = self._lane_quotas.get(lane)
            if current == normalized and (normalized is not None or lane not in self._lane_quotas):
                return
            if normalized is None:
                self._lane_quotas.pop(lane, None)
            else:
                self._lane_quotas[lane] = normalized
            self._release_parked_quota_locked()
            self._cond.notify_all()
        self._backend.wake()

    def note_inline_work(self, item: WorkItem) -> None:
        """Record already-executed GUI-thread work in kernel diagnostics.

        Some R1-era frame renderer code still performs bounded commit/fan-in
        work synchronously on the GUI thread. It is not submitted for worker
        execution, but it should be visible in the same lane counters until R2
        ports those effects into pipeline tasks.
        """

        item = item if isinstance(item, WorkItem) else WorkItem(**item)
        with self._lock:
            counters = self._lane(item.lane)
            counters.queued += 1
            counters.admitted += 1
            counters.started += 1
            counters.completed += 1
            self._completed_keys[item.key] = ""

    def shutdown(self, timeout: float = 5.0) -> None:
        shutdown_started = monotonic()
        queued_cancelled = 0
        running_cancelled = 0
        with self._lock:
            self._shutting_down = True
            for record in tuple(self._records.values()):
                record.token.cancel()
                if record.state in (_QUEUED, _PARKED_DEPS, _PARKED_QUOTA):
                    self._lane(record.spec.lane).cancelled += 1
                    self._decrement_queued_locked(record.spec)
                    self._forget_record_locked(record)
                    # Every admitted task owes exactly one terminal
                    # completion: queued ``on_stale`` cleanup owners
                    # (in-flight dedup, residency pins) silently skipped when
                    # shutdown deleted their records without delivery (codex
                    # review 2026-07-19, finding 4). The Qt bridge closes
                    # before kernel shutdown by design; delivery is the
                    # kernel's contract, draining is the consumer's.
                    self._deliver_locked(record, TaskOutcome.CANCELLED, reason="kernel_shutdown")
                    queued_cancelled += 1
                elif record.state == _RUNNING:
                    running_cancelled += 1
            # Heap/parked entries are indexes over records. Every unstarted
            # record is gone, so retaining their sequence numbers only makes
            # post-shutdown diagnostics lie about queued work.
            self._ready.clear()
            self._parked_quota.clear()
            self._cond.notify_all()
        emit_trace(
            "kernel_shutdown",
            action="cancel",
            queued_cancelled=int(queued_cancelled),
            running_cancelled=int(running_cancelled),
            timeout_s=float(timeout),
        )
        alive_threads = tuple(self._backend.shutdown(timeout=timeout) or ())
        with self._lock:
            diagnostics = tuple(
                {
                    "task_seq": int(record.seq),
                    "key": record.spec.key,
                    "lane": str(record.spec.lane),
                    "scope": str(record.spec.scope or ""),
                    "scopes": tuple(record.spec.scope_prefixes()),
                }
                for record in self._records.values()
                if record.state == _RUNNING
            )
            self._last_shutdown_diagnostics = diagnostics
        if alive_threads:
            elapsed_ms = (monotonic() - shutdown_started) * 1000.0
            scopes = tuple(row["scope"] or repr(row["key"]) for row in diagnostics)
            emit_trace(
                "kernel_shutdown",
                action="timeout",
                timeout_s=float(timeout),
                elapsed_ms=float(elapsed_ms),
                alive_threads=alive_threads,
                running_tasks=diagnostics,
            )
            warnings.warn(
                "ArrayScope kernel shutdown exceeded "
                f"{float(timeout):.3f}s; alive_threads={alive_threads!r}; "
                f"running_scopes={scopes!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            elapsed_ms = (monotonic() - shutdown_started) * 1000.0
            emit_trace(
                "kernel_shutdown",
                action="complete",
                elapsed_ms=float(elapsed_ms),
                queued_cancelled=int(queued_cancelled),
                running_cancelled=int(running_cancelled),
            )

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Block until no task is queued, dep-parked, quota-parked, or running.

        Dep-parked tasks whose dependencies can never complete cause a
        timeout — that is a test failure signal, not a state to hide.
        """

        deadline = monotonic() + float(timeout)
        with self._cond:
            while True:
                busy = any(
                    record.state in (_RUNNING, _QUEUED, _PARKED_DEPS, _PARKED_QUOTA)
                    for record in self._records.values()
                )
                if not busy:
                    return True
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(timeout=min(0.05, remaining))

    def dispatch_event(self, event: CompletionEvent) -> TaskOutcome:
        """Run the handlers for one drained event on the *calling* thread.

        Re-classifies staleness at dispatch time: work that completed just
        before its target changed must not commit stale pixels.
        """

        record: _Record | None = getattr(event, "_record", None)
        outcome = event.outcome
        if record is not None and outcome == TaskOutcome.COMPLETED:
            with self._lock:
                if self._record_is_stale_locked(record):
                    outcome = TaskOutcome.STALE
                    counters = self._lane(event.spec.lane)
                    counters.completed -= 1
                    counters.stale += 1
                    self._completed_keys.pop(event.spec.key, None)
        on_done = on_error = on_stale = on_reuse = None
        if record is not None:
            on_done, on_error = record.on_done, record.on_error
            on_stale, on_reuse = record.on_stale, record.on_reuse
        if outcome == TaskOutcome.COMPLETED:
            self._call(on_done, event.value, context="task completion")
        elif outcome == TaskOutcome.FAILED:
            if on_error is not None:
                self._call(on_error, event.error, context="task failure")
            else:
                self._handler_error_hook("unhandled task failure", event.error)
        elif outcome == TaskOutcome.STALE:
            if on_reuse is not None and event.value is not None:
                self._call(on_reuse, event.value, context="stale reuse")
                with self._lock:
                    self._lane(event.spec.lane).stale_reused += 1
                outcome = TaskOutcome.STALE_REUSED
            self._call(on_stale, context="stale notification")
        else:  # CANCELLED / DROPPED
            self._call(on_stale, context="stale notification")
        return outcome

    def diagnostics(self) -> KernelDiagnostics:
        with self._lock:
            lanes = {
                lane.value: counters.as_dict()
                for lane, counters in self._counters.items()
                if any(counters.as_dict().values())
            }
            oldest_ms = 0.0
            now = perf_counter_ns()
            for record in self._records.values():
                if record.state == _QUEUED:
                    oldest_ms = max(oldest_ms, (now - record.submitted_ns) / 1e6)
            return KernelDiagnostics(
                lanes=lanes,
                queued=sum(1 for r in self._records.values() if r.state == _QUEUED),
                running=sum(1 for r in self._records.values() if r.state == _RUNNING),
                active=sum(1 for r in self._records.values() if r.state == _RUNNING),
                parked_deps=sum(1 for r in self._records.values() if r.state == _PARKED_DEPS),
                parked_quota=len(self._parked_quota),
                workers=int(getattr(self._backend, "workers", 0)),
                completed_keys=len(self._completed_keys),
                scopes=len(self._scope_epochs),
                oldest_queued_ms=oldest_ms,
                visible_backlog=int(self._queued_visible + self._running_visible),
            )

    @property
    def visible_backlog(self) -> int:
        with self._lock:
            return int(self._queued_visible + self._running_visible)

    @property
    def last_shutdown_diagnostics(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._last_shutdown_diagnostics)

    def has_live_task(self, key: object) -> bool:
        with self._lock:
            seq = self._by_key.get(key)
            if seq is None:
                return False
            record = self._records.get(seq)
            return bool(record is not None and record.state != _DONE and not record.superseded)

    def has_completed_task(self, key: object) -> bool:
        with self._lock:
            return key in self._completed_keys

    # ------------------------------------------------- backend entry points

    def _notify_workers(self, all_workers: bool = False) -> None:
        with self._lock:
            if all_workers:
                self._cond.notify_all()
            else:
                self._cond.notify()

    def _take_next(self, *, block: bool = True, stop: threading.Event | None = None):
        """Pop the highest-priority runnable record. Worker-thread entry."""

        with self._lock:
            while True:
                record = self._pop_ready_locked()
                if record is not None:
                    return record
                if not block or self._shutting_down or (stop is not None and stop.is_set()):
                    return None
                self._cond.wait(timeout=0.1)

    def _execute(self, record: _Record) -> None:
        """Run one task function. Worker-thread entry; no lock held."""

        spec = record.spec
        emit_trace(
            "kernel_start",
            task_seq=int(record.seq),
            key=spec.key,
            lane=str(spec.lane),
            priority=int(spec.priority),
            rung=int(spec.rung),
            level=int(spec.level),
            session_id=int(spec.session_id),
            tile_number=int(spec.tile_number),
            scheduling_generation=int(spec.scheduling_generation),
        )
        with self._lock:
            self._lane(spec.lane).started += 1
        if record.token.cancelled:
            self._finish(record, TaskOutcome.CANCELLED)
            return
        # `fn_ns` is the body alone: kernel_start fires before the started
        # bookkeeping and kernel_finish after delivery, so their timestamp
        # difference prices queue and lock work into the evaluation.  Two
        # clock reads per task make per-(rung, level) evaluation cost
        # readable straight off the trace.
        started_ns = perf_counter_ns()
        try:
            value = spec.fn(record.token) if spec.pass_token else spec.fn()
        except EvaluationCancelled:
            self._finish(record, TaskOutcome.CANCELLED, fn_ns=perf_counter_ns() - started_ns)
            return
        except BaseException as exc:
            exc.arrayscope_traceback = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            self._finish(
                record, TaskOutcome.FAILED, error=exc, fn_ns=perf_counter_ns() - started_ns
            )
            return
        self._finish(
            record,
            TaskOutcome.COMPLETED,
            value=value,
            fn_ns=perf_counter_ns() - started_ns,
        )

    # ---------------------------------------------------------- internals

    def _finish(
        self,
        record: _Record,
        outcome: TaskOutcome,
        *,
        value=None,
        error=None,
        fn_ns: int = -1,
    ) -> None:
        spec = record.spec
        wake = False
        with self._lock:
            if spec.visible:
                self._running_visible = max(0, self._running_visible - 1)
            else:
                self._running_optional = max(0, self._running_optional - 1)
            lane_running = int(self._running_by_lane.get(spec.lane, 0) or 0)
            if lane_running <= 1:
                self._running_by_lane.pop(spec.lane, None)
            else:
                self._running_by_lane[spec.lane] = lane_running - 1
            counters = self._lane(spec.lane)
            if outcome == TaskOutcome.COMPLETED and self._record_is_stale_locked(record):
                outcome = TaskOutcome.STALE
            if outcome == TaskOutcome.COMPLETED:
                counters.completed += 1
                self._completed_keys[spec.key] = spec.scope
                self._promote_dependents_locked(spec.key)
                wake = True
            elif outcome == TaskOutcome.STALE:
                counters.stale += 1
                self._fail_dependents_locked(spec.key)
            elif outcome == TaskOutcome.FAILED:
                counters.failed += 1
                self._fail_dependents_locked(spec.key)
            else:
                counters.cancelled += 1
                self._fail_dependents_locked(spec.key)
            self._release_parked_quota_locked()
            self._forget_record_locked(record)
            self._deliver_locked(record, outcome, value=value, error=error)
            self._cond.notify_all()
        emit_trace(
            "kernel_finish",
            task_seq=int(record.seq),
            key=spec.key,
            lane=str(spec.lane),
            outcome=str(outcome.value),
            error_type=None if error is None else type(error).__name__,
            rung=int(spec.rung),
            level=int(spec.level),
            session_id=int(spec.session_id),
            tile_number=int(spec.tile_number),
            scheduling_generation=int(spec.scheduling_generation),
            fn_ns=int(fn_ns),
        )
        if wake:
            self._backend.wake()
        # Failing dependents may have dropped the last blocking item.
        self._flush_release_wake()

    def _deliver_locked(
        self, record: _Record, outcome: TaskOutcome, *, value=None, error=None, reason: str = ""
    ) -> None:
        event = CompletionEvent(
            spec=record.spec, outcome=outcome, value=value, error=error, reason=reason
        )
        object.__setattr__(event, "_record", record)
        self.completions.put(event)

    def _enqueue_ready_locked(self, record: _Record) -> None:
        record.state = _QUEUED
        heapq.heappush(self._ready, self._ready_entry_locked(record))

    def _ready_entry_locked(self, record: _Record) -> tuple[int, int, int, float, int]:
        spec = record.spec
        deadline = float("inf") if spec.deadline_ns <= 0 else float(spec.deadline_ns)
        rank = 0 if spec.visible else 1
        return (
            rank,
            int(spec.priority),
            int(spec.scheduling_rank),
            deadline,
            record.seq,
        )

    def _pop_ready_locked(self) -> _Record | None:
        while self._ready:
            _rank, _priority, _scheduling_rank, _deadline, seq = heapq.heappop(self._ready)
            record = self._records.get(seq)
            if record is None or record.state != _QUEUED:
                continue
            spec = record.spec
            counters = self._lane(spec.lane)
            if self._record_is_stale_locked(record):
                self._drop_record_locked(record, reason="superseded")
                continue
            if record.token.cancelled:
                counters.cancelled += 1
                self._decrement_queued_locked(spec)
                self._forget_record_locked(record)
                self._deliver_locked(record, TaskOutcome.CANCELLED, reason="cancelled_before_start")
                self._wake_after_blocking_removal_locked(spec)
                continue
            if spec.deadline_ns and perf_counter_ns() > spec.deadline_ns:
                counters.deadline_missed += 1
                if not spec.visible:
                    self._drop_record_locked(record, reason="deadline", superseded=False)
                    continue
            if self._lane_quota_blocked_locked(spec) or (
                not spec.visible and self._optional_blocked_locked(spec)
            ):
                if not record.quota_blocked_noted:
                    counters.blocked_by_quota += 1
                    record.quota_blocked_noted = True
                record.state = _PARKED_QUOTA
                self._parked_quota.append(record.seq)
                continue
            record.state = _RUNNING
            self._decrement_queued_locked(spec)
            if spec.visible:
                self._running_visible += 1
            else:
                self._running_optional += 1
            self._running_by_lane[spec.lane] = int(self._running_by_lane.get(spec.lane, 0) or 0) + 1
            return record
        return None

    def _lane_quota_blocked_locked(self, spec: TaskSpec) -> bool:
        quota = self._lane_quotas.get(spec.lane)
        if quota is None:
            return False
        return int(self._running_by_lane.get(spec.lane, 0) or 0) >= int(quota)

    def _optional_blocked_locked(self, spec: TaskSpec) -> bool:
        quota = max(1, int(getattr(self._backend, "workers", 1) * self._speculative_fraction))
        if self._running_optional >= quota:
            return True
        backlog = (self._queued_visible + self._running_visible) > 0
        if backlog and spec.expected_value <= 0.0:
            return True
        return spec.expected_value < self.optional_value_threshold

    def _release_parked_quota_locked(self) -> None:
        if not self._parked_quota:
            return
        parked, self._parked_quota = self._parked_quota, []
        for seq in parked:
            record = self._records.get(seq)
            if record is not None and record.state == _PARKED_QUOTA:
                self._enqueue_ready_locked(record)

    def _promote_dependents_locked(self, key: object) -> None:
        for seq in tuple(self._dep_waiters.pop(key, ())):
            record = self._records.get(seq)
            if record is None or record.state != _PARKED_DEPS:
                continue
            record.unmet_deps.discard(key)
            if not record.unmet_deps:
                self._enqueue_ready_locked(record)

    def _fail_dependents_locked(self, key: object) -> None:
        for seq in tuple(self._dep_waiters.pop(key, ())):
            record = self._records.get(seq)
            if record is None or record.state != _PARKED_DEPS:
                continue
            counters = self._lane(record.spec.lane)
            counters.dependency_failed += 1
            self._drop_record_locked(record, reason="dependency_failed", superseded=False)
            self._fail_dependents_locked(record.spec.key)

    def _drop_record_locked(self, record: _Record, *, reason: str, superseded: bool = True) -> None:
        counters = self._lane(record.spec.lane)
        counters.dropped += 1
        if superseded:
            counters.superseded += 1
        self._decrement_queued_locked(record.spec)
        self._forget_record_locked(record)
        self._deliver_locked(record, TaskOutcome.DROPPED, reason=reason)
        self._wake_after_blocking_removal_locked(record.spec)

    def _wake_after_blocking_removal_locked(self, spec: TaskSpec) -> None:
        """Removing the last blocking item is the same wake edge a completion is.

        Optional records park behind visible backlog and only
        ``_release_parked_quota_locked`` returns them to the ready heap.
        Every completion runs that release in ``_finish``, but a queued or
        parked visible record removed by cancellation/drop never reaches
        ``_finish`` — without this edge the parked optional work strands
        until an unrelated quota transition (lost-wakeup family #7,
        codex review 2026-07-19; fix conventions in
        docs/redesign/stale-empty-tiles-2026-07-16.md).
        """

        if not spec.visible:
            return
        if (self._queued_visible + self._running_visible) > 0:
            return
        if not self._parked_quota:
            return
        released = len(self._parked_quota)
        self._release_parked_quota_locked()
        self._release_wake_pending = True
        emit_trace(
            "kernel_wake_edge",
            reason="last_blocking_item_removed",
            released=int(released),
            key=spec.key,
            lane=str(spec.lane),
        )
        self._cond.notify_all()

    def _flush_release_wake(self) -> None:
        """Turn a pending parked-work release into a backend wake, lock-free."""

        with self._lock:
            pending = self._release_wake_pending
            self._release_wake_pending = False
        if pending:
            self._backend.wake()

    def _supersede_instance_locked(self, seq: int) -> None:
        record = self._records.get(seq)
        if record is None:
            return
        record.superseded = True
        if record.state in (_QUEUED, _PARKED_DEPS, _PARKED_QUOTA):
            self._drop_record_locked(record, reason="resubmitted")
        elif record.state == _RUNNING and not record.spec.reusable:
            record.token.cancel()

    def _cancel_seq(self, seq: int) -> None:
        with self._lock:
            record = self._records.get(seq)
            if record is None:
                return
            record.token.cancel()
            if record.state in (_QUEUED, _PARKED_DEPS, _PARKED_QUOTA):
                self._lane(record.spec.lane).cancelled += 1
                self._decrement_queued_locked(record.spec)
                self._forget_record_locked(record)
                self._deliver_locked(record, TaskOutcome.CANCELLED, reason="cancelled")
                self._wake_after_blocking_removal_locked(record.spec)
            self._cond.notify_all()
        self._flush_release_wake()

    def _decrement_queued_locked(self, spec: TaskSpec) -> None:
        if spec.visible:
            self._queued_visible = max(0, self._queued_visible - 1)

    def _forget_record_locked(self, record: _Record) -> None:
        record.state = _DONE
        for dep in record.unmet_deps:
            waiters = self._dep_waiters.get(dep)
            if waiters is not None:
                waiters.discard(record.seq)
                if not waiters:
                    self._dep_waiters.pop(dep, None)
        record.unmet_deps = set()
        self._records.pop(record.seq, None)
        if self._by_key.get(record.spec.key) == record.seq:
            self._by_key.pop(record.spec.key, None)

    def _advance_supersession_locked(
        self, supersession: Supersession, *, exclude_seq: int | None = None
    ) -> None:
        current = self._supersession.get(supersession.family, _SENTINEL)
        if current is not _SENTINEL and current == supersession.value:
            return
        self._supersession[supersession.family] = supersession.value
        for record in tuple(self._records.values()):
            if exclude_seq is not None and record.seq == exclude_seq:
                continue
            spec = record.spec
            if spec.supersession is None or spec.supersession.family != supersession.family:
                continue
            if spec.supersession.value == supersession.value:
                continue
            if record.state in (_QUEUED, _PARKED_DEPS, _PARKED_QUOTA):
                self._drop_record_locked(record, reason="superseded")
            elif record.state == _RUNNING and not spec.reusable:
                record.token.cancel()

    def _purge_completed_locked(self, prefix: str) -> int:
        child_prefix = f"{prefix}:"
        stale_keys = tuple(
            key
            for key, scope in self._completed_keys.items()
            if scope == prefix or scope.startswith(child_prefix)
        )
        for key in stale_keys:
            del self._completed_keys[key]
        return len(stale_keys)

    def _scope_snapshot_locked(self, spec: TaskSpec) -> tuple[tuple[str, int], ...]:
        return tuple(
            (prefix, self._scope_epochs.get(prefix, 0)) for prefix in spec.scope_prefixes()
        )

    def _record_is_stale_locked(self, record: _Record) -> bool:
        if self._shutting_down:
            return True
        if record.superseded:
            return True
        spec = record.spec
        if spec.supersession is not None:
            current = self._supersession.get(spec.supersession.family, _SENTINEL)
            if current is not _SENTINEL and current != spec.supersession.value:
                return True
        for prefix, epoch_at_submit in record.scope_snapshot:
            if self._scope_epochs.get(prefix, 0) > epoch_at_submit:
                return True
        return False

    def _lane(self, lane: Lane) -> LaneCounters:
        return self._counters.setdefault(Lane(str(lane)), LaneCounters())

    def _call(self, handler, *args, context: str) -> None:
        if handler is None:
            return
        try:
            handler(*args)
        except Exception as exc:
            self._handler_error_hook(context, exc)


def _default_handler_error_hook(context: str, error: BaseException | None) -> None:
    if error is None:
        return
    raise error


__all__ = ["Kernel", "KernelDiagnostics", "TaskHandle"]
