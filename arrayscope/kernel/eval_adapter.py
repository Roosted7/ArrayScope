"""Temporary EvaluationController surface over the execution kernel.

R1 keeps the old controller *call sites* alive while moving execution to one
Kernel plus one QtKernelBridge. This adapter maps public submissions onto
TaskSpec; it intentionally does not recreate the old per-controller queues,
thread pools, runnables, or drain timers.
"""

from __future__ import annotations

from collections import defaultdict

from arrayscope.app.errors import handle_ui_exception
from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.core.scheduler import EvalPriority, FrameProgress, FrameTarget, SchedulerDiagnostics, WorkStart
from arrayscope.kernel.qt_bridge import QtKernelBridge
from arrayscope.kernel.scheduler import Kernel
from arrayscope.kernel.task import Lane, Priority, Supersession, TaskSpec, WorkItem
from arrayscope.kernel.workers import ThreadWorkerBackend

prefer_pyside6()

import pyqtgraph.Qt as Qt


_UNSET = object()


class KernelEvaluationController(Qt.QtCore.QObject):
    """Public EvaluationController API backed by a shared kernel."""

    queueEventReady = Qt.QtCore.Signal()

    def __init__(
        self,
        kernel: Kernel | None = None,
        bridge: QtKernelBridge | None = None,
        parent=None,
        *,
        max_workers=None,
        name: str = "evaluation",
        lane_default: Lane = Lane.VISIBLE_MATERIALIZATION,
        priority_default: Priority = Priority.HOVER,
        max_callback_dispatch_per_drain: int = 4,
        max_queue_events_per_drain: int = 64,
        apply_lane_quota: bool = True,
    ) -> None:
        if isinstance(kernel, Qt.QtCore.QObject) and bridge is None and parent is None:
            parent = kernel
            kernel = None
        super().__init__(parent)
        self.name = str(name)
        self.kernel = kernel
        self._owns_kernel = self.kernel is None
        if self.kernel is None:
            self.kernel = Kernel(
                ThreadWorkerBackend(workers=max_workers, name=f"arrayscope-{self.name}"),
                handler_error_hook=handle_ui_exception,
            )
        self.bridge = bridge
        self._owns_bridge = self.bridge is None
        if self.bridge is None:
            self.bridge = QtKernelBridge(
                self.kernel,
                self,
                max_items_per_drain=max_callback_dispatch_per_drain,
                max_events_per_drain=max_queue_events_per_drain,
            )
        self.lane_default = Lane(str(lane_default))
        self.priority_default = Priority(int(priority_default))
        self.generation = 0
        self._closed = False
        self._max_workers = max(1, int(max_workers)) if max_workers is not None else None
        self._max_prefetch = 32
        self._prefetch_keys: set[object] = set()
        self._pending_generations: set[int] = set()
        self._pending_prefetch: set[object] = set()
        self._generation_group: dict[int, str] = {}
        self._generation_target: dict[int, FrameTarget | None] = {}
        self._group_generations: dict[str, int] = {}
        self._group_epoch = 0
        self._known_groups: set[str] = set()
        self._frame_progress: dict[str, FrameProgress] = defaultdict(FrameProgress)
        self._prefetch_scheduled_count = 0
        self._prefetch_deduped_count = 0
        self._prefetch_limited_count = 0
        self._prefetch_idle_blocked_count = 0
        self._prefetch_visible_busy_blocked_count = 0
        self._prefetch_cost_blocked_count = 0
        self._active_preserved_count = 0
        self._queued_collapsed_count = 0
        self._manual_stale_reused_count = 0
        self._callback_budget_ms: float | None = None
        self._max_callback_dispatch_per_drain = max(1, int(max_callback_dispatch_per_drain))
        self._max_queue_events_per_drain = max(1, int(max_queue_events_per_drain))
        self._apply_lane_quota = bool(apply_lane_quota)
        if self._max_workers is not None:
            self.set_max_workers(self._max_workers)

    # ------------------------------------------------------------------ API

    def notify_when_capacity(self, key, fn) -> None:
        self.bridge.notify_when_capacity(key, fn)

    def cancel_pending(self):
        self.generation += 1
        self.kernel.clear_scope(self._scope(""))
        self._pending_generations.clear()
        self._generation_group.clear()
        self._generation_target.clear()
        self._reset_progress()
        return self.generation

    def clear_queued(self):
        self.cancel_prefetch()
        return self.cancel_pending()

    def clear_group(self, replace_group: str):
        replace_group = str(replace_group)
        groups = {replace_group}
        groups.update(group for group in self._known_groups if group.startswith(f"{replace_group}:"))
        for group in groups:
            self.advance_group(group)
            self._frame_progress[group] = FrameProgress(presented=self._frame_progress[group].presented)
        self.kernel.clear_scope(self._scope(replace_group))

    def advance_group(self, replace_group: str) -> int:
        replace_group = str(replace_group)
        self._known_groups.add(replace_group)
        self._group_epoch += 1
        self._group_generations[replace_group] = self._group_epoch
        return self._group_epoch

    def group_generation(self, replace_group: str) -> int:
        return int(self._group_generations.get(str(replace_group), 0))

    def shutdown_for_close(self):
        self._closed = True
        self.cancel_prefetch()
        self.kernel.clear_scope(self._scope(""))
        if self._owns_bridge:
            self.bridge.close()
        if self._owns_kernel:
            self.kernel.shutdown()
        return self.generation

    def start(
        self,
        fn,
        *,
        on_done,
        on_error=None,
        on_stale=None,
        on_slow=None,
        slow_ms=100,
        priority=EvalPriority.HOVER,
        key=None,
        replace_group="default",
        frame_target: FrameTarget | None = None,
        supersession_key=None,
        supersession_value=_UNSET,
        work_item: WorkItem | None = None,
    ):
        return self.start_latest(
            fn,
            key=("compat", self.generation + 1) if key is None else key,
            priority=priority,
            replace_group=replace_group,
            on_done=on_done,
            on_error=on_error,
            on_stale=on_stale,
            on_slow=on_slow,
            slow_ms=slow_ms,
            frame_target=frame_target,
            supersession_key=supersession_key,
            supersession_value=supersession_value,
            work_item=work_item,
        )

    def start_latest(
        self,
        fn,
        *,
        key,
        priority,
        replace_group,
        on_done,
        on_error=None,
        on_stale=None,
        on_slow=None,
        slow_ms=100,
        memory_budget_bytes=None,
        pass_token=False,
        frame_target: FrameTarget | None = None,
        supersession_key=None,
        supersession_value=_UNSET,
        work_item: WorkItem | None = None,
    ):
        replace_group = str(replace_group)
        if supersession_key is None and (work_item is None or work_item.supersession_key is None):
            self.clear_group(replace_group)
            group_generation = self.advance_group(replace_group)
        else:
            group_generation = self.group_generation(replace_group)
        return self._submit(
            fn,
            key=key,
            priority=priority,
            replace_group=replace_group,
            group_generation=group_generation,
            on_done=on_done,
            on_error=on_error,
            on_stale=on_stale,
            on_reuse_stale=None,
            on_slow=on_slow,
            slow_ms=slow_ms,
            memory_budget_bytes=memory_budget_bytes,
            pass_token=pass_token,
            frame_target=frame_target,
            supersession_key=supersession_key,
            supersession_value=supersession_value,
            work_item=work_item,
            reusable=False,
        )

    def start_active_plus_latest(
        self,
        fn,
        *,
        key,
        priority,
        replace_group,
        on_done,
        on_error=None,
        on_stale=None,
        on_reuse_stale=None,
        on_slow=None,
        slow_ms=100,
        memory_budget_bytes=None,
        pass_token=False,
        frame_target: FrameTarget | None = None,
        supersession_key=None,
        supersession_value=_UNSET,
        work_item: WorkItem | None = None,
    ):
        replace_group = str(replace_group)
        had_pending = any(group == replace_group for group in self._generation_group.values())
        if supersession_key is None and (work_item is None or work_item.supersession_key is None):
            self.kernel.clear_scope(self._scope(replace_group))
            group_generation = self.advance_group(replace_group)
            self._queued_collapsed_count += int(had_pending)
        else:
            group_generation = self.group_generation(replace_group)
            self._queued_collapsed_count += int(had_pending)
        if had_pending:
            self._active_preserved_count += 1
        return self._submit(
            fn,
            key=key,
            priority=priority,
            replace_group=replace_group,
            group_generation=group_generation,
            on_done=on_done,
            on_error=on_error,
            on_stale=on_stale,
            on_reuse_stale=on_reuse_stale,
            on_slow=on_slow,
            slow_ms=slow_ms,
            memory_budget_bytes=memory_budget_bytes,
            pass_token=pass_token,
            frame_target=frame_target,
            supersession_key=supersession_key,
            supersession_value=supersession_value,
            work_item=work_item,
            reusable=True,
        )

    def start_prefetch(
        self,
        fn,
        on_done=None,
        *,
        key=None,
        memory_budget_bytes=None,
        idle_elapsed=None,
        blocked_reason=None,
        work_item: WorkItem | None = None,
    ):
        if self._closed:
            return WorkStart(False, "closed")
        if blocked_reason:
            self._note_prefetch_blocked(blocked_reason)
            return WorkStart(False, blocked_reason)
        if idle_elapsed is False:
            self._note_prefetch_blocked("idle")
            return WorkStart(False, "idle")
        if memory_budget_bytes is not None and int(memory_budget_bytes) <= 0:
            self._note_prefetch_blocked("cost")
            return WorkStart(False, "cost")
        key = ("prefetch", id(fn), self.generation + 1) if key is None else key
        if key in self._prefetch_keys:
            self._prefetch_deduped_count += 1
            return WorkStart(False, "deduped")
        if len(self._prefetch_keys) >= self._max_prefetch:
            self._prefetch_limited_count += 1
            return WorkStart(False, "limited")
        self._prefetch_keys.add(key)
        self._pending_prefetch.add(key)
        self._prefetch_scheduled_count += 1
        item_key = getattr(work_item, "key", ("prefetch", self.name, key))
        spec = self._task_spec(
            fn,
            key=item_key,
            priority=Priority.PREFETCH,
            replace_group="prefetch",
            memory_budget_bytes=memory_budget_bytes,
            pass_token=False,
            frame_target=getattr(work_item, "frame_target", None),
            supersession_key=getattr(work_item, "supersession_key", None),
            supersession_value=getattr(work_item, "supersession_value", _UNSET),
            work_item=work_item,
            reusable=False,
            lane=Lane.SPECULATIVE_RESIDENCY,
        )

        def done(value, *, key=key):
            self._prefetch_keys.discard(key)
            self._pending_prefetch.discard(key)
            if on_done is not None:
                on_done(value)

        def stale(*, key=key):
            self._prefetch_keys.discard(key)
            self._pending_prefetch.discard(key)

        self.kernel.submit(spec, on_done=done, on_error=lambda _exc: stale(), on_stale=stale)
        return WorkStart(True)

    def cancel_prefetch(self) -> None:
        self.kernel.clear_scope(self._scope("prefetch"))
        self._prefetch_keys.clear()
        self._pending_prefetch.clear()

    def set_max_workers(self, count: int) -> None:
        self._max_workers = max(1, int(count))
        if self._apply_lane_quota:
            self.kernel.set_lane_quota(self.lane_default, self._max_workers)

    def set_reported_max_workers(self, count: int) -> None:
        """Update compatibility diagnostics without owning kernel quotas."""

        self._max_workers = max(1, int(count))

    def set_max_callback_dispatch_per_drain(self, count: int) -> None:
        self._max_callback_dispatch_per_drain = max(1, int(count))
        self.bridge.set_max_items_per_drain(self._max_callback_dispatch_per_drain)

    def set_callback_budget_ms(self, budget_ms: float | None) -> None:
        self._callback_budget_ms = None if budget_ms is None else max(0.25, float(budget_ms))
        self.bridge.set_budget_ms(self._callback_budget_ms)

    def set_max_prefetch(self, count: int) -> None:
        self._max_prefetch = max(0, int(count))

    def is_busy(self) -> bool:
        return self.has_running_or_pending()

    def has_running_or_pending(self) -> bool:
        return bool(self._pending_generations or self._pending_prefetch)

    def frame_progress(self, replace_group: str) -> FrameProgress:
        return self._frame_progress.get(str(replace_group), FrameProgress())

    def note_stale_reused(self) -> None:
        self._manual_stale_reused_count += 1

    def diagnostics(self) -> SchedulerDiagnostics:
        diag = self.kernel.diagnostics()
        lane_values = dict(diag.lanes.get(self.lane_default.value, {}) or {})
        if self.name == "prefetch":
            lane_values = dict(diag.lanes.get(Lane.SPECULATIVE_RESIDENCY.value, {}) or lane_values)
        progress = self._diagnostic_frame_progress()
        return SchedulerDiagnostics(
            name=self.name,
            max_workers=int(self._max_workers or getattr(diag, "workers", 0) or 0),
            pending=len(self._pending_generations) + len(self._pending_prefetch),
            running=int(lane_values.get("started", 0)) - int(lane_values.get("completed", 0)) - int(lane_values.get("failed", 0)) - int(lane_values.get("cancelled", 0)) - int(lane_values.get("stale", 0)),
            queued=int(getattr(diag, "queued", 0) + getattr(diag, "parked_deps", 0) + getattr(diag, "parked_quota", 0)),
            started=int(lane_values.get("started", 0)),
            cancelled=int(lane_values.get("cancelled", 0)),
            stale=int(lane_values.get("stale", 0)) + int(lane_values.get("dropped", 0)),
            completed=int(lane_values.get("completed", 0)),
            failed=int(lane_values.get("failed", 0)),
            prefetch_scheduled=int(self._prefetch_scheduled_count),
            prefetch_deduped=int(self._prefetch_deduped_count),
            prefetch_limited=int(self._prefetch_limited_count),
            prefetch_idle_blocked=int(self._prefetch_idle_blocked_count),
            prefetch_visible_busy_blocked=int(self._prefetch_visible_busy_blocked_count),
            prefetch_cost_blocked=int(self._prefetch_cost_blocked_count),
            active_preserved=int(self._active_preserved_count),
            queued_collapsed=int(self._queued_collapsed_count),
            stale_reused=int(lane_values.get("stale_reused", 0)) + int(self._manual_stale_reused_count),
            fallback_event_polls=int(self.bridge.fallback_event_polls),
            fallback_idle_polls=int(self.bridge.fallback_idle_polls),
            presented_target=progress.presented,
            active_target=progress.active,
            queued_latest_target=progress.queued_latest,
            work_lanes=tuple(sorted(diag.lanes)),
            kernel=diag,
        )

    # --------------------------------------------------------------- internals

    def _submit(
        self,
        fn,
        *,
        key,
        priority,
        replace_group,
        group_generation,
        on_done,
        on_error,
        on_stale,
        on_reuse_stale,
        on_slow,
        slow_ms,
        memory_budget_bytes,
        pass_token,
        frame_target,
        supersession_key,
        supersession_value,
        work_item,
        reusable,
    ):
        if self._closed:
            if on_stale is not None:
                on_stale()
            return None
        self.generation += 1
        generation = self.generation
        self._known_groups.add(str(replace_group))
        self._pending_generations.add(generation)
        self._generation_group[generation] = str(replace_group)
        self._generation_target[generation] = frame_target
        self._note_progress_submitted(str(replace_group), frame_target)
        spec = self._task_spec(
            fn,
            key=getattr(work_item, "key", (self.name, key)),
            priority=priority,
            replace_group=replace_group,
            memory_budget_bytes=memory_budget_bytes,
            pass_token=pass_token,
            frame_target=frame_target,
            supersession_key=supersession_key,
            supersession_value=supersession_value,
            work_item=work_item,
            reusable=reusable or bool(getattr(work_item, "reusable_output", False)),
        )
        if on_slow is not None:
            # Timer category: UI cosmetic. Slow-work notification delay only;
            # task completion and cancellation are kernel-owned.
            Qt.QtCore.QTimer.singleShot(
                int(slow_ms),
                self.bridge,
                lambda generation=generation: self._emit_slow(generation, on_slow),
            )

        self.kernel.submit(
            spec,
            on_done=lambda value, generation=generation, on_done=on_done: self._done(generation, value, on_done),
            on_error=lambda exc, generation=generation, on_error=on_error: self._error(generation, exc, on_error),
            on_stale=lambda generation=generation, on_stale=on_stale: self._stale(generation, on_stale),
            on_reuse=(
                None
                if on_reuse_stale is None
                else lambda value, on_reuse_stale=on_reuse_stale: on_reuse_stale(value)
            ),
        )
        return generation

    def _task_spec(
        self,
        fn,
        *,
        key,
        priority,
        replace_group,
        memory_budget_bytes,
        pass_token,
        frame_target,
        supersession_key,
        supersession_value,
        work_item,
        reusable,
        lane: Lane | None = None,
    ) -> TaskSpec:
        lane_value = Lane(str(lane or getattr(work_item, "lane", self.lane_default)))
        priority_value = Priority(int(priority if priority is not None else self.priority_default))
        sup_key = supersession_key
        if sup_key is None and work_item is not None:
            sup_key = work_item.supersession_key
        sup_value = supersession_value
        if sup_value is _UNSET and work_item is not None:
            sup_value = work_item.supersession_value
        if sup_key is not None and sup_value is _UNSET:
            sup_value = sup_key
        supersession = (
            None
            if sup_key is None
            else Supersession((self.name, str(replace_group), sup_key), sup_value)
        )
        estimated_bytes = (
            memory_budget_bytes
            if memory_budget_bytes is not None
            else getattr(work_item, "estimated_bytes", 0)
        )
        return TaskSpec(
            key=key,
            fn=fn,
            lane=lane_value,
            priority=priority_value,
            scope=self._scope(replace_group),
            deps=tuple(getattr(work_item, "dependency_keys", ()) or ()),
            supersession=supersession,
            deadline_ns=int(getattr(work_item, "deadline_ns", getattr(frame_target, "deadline_ns", 0)) or 0),
            estimated_cpu_ms=float(getattr(work_item, "estimated_cpu_ms", 0.0) or 0.0),
            estimated_bytes=int(estimated_bytes or 0),
            expected_value=float(getattr(work_item, "expected_value", 0.0) or 0.0),
            reusable=bool(reusable),
            pass_token=bool(pass_token),
        )

    def _scope(self, replace_group: str) -> str:
        replace_group = str(replace_group)
        return self.name if replace_group == "" else f"{self.name}:{replace_group}"

    def _done(self, generation: int, value, on_done) -> None:
        group = self._generation_group.get(generation)
        target = self._generation_target.get(generation)
        self._forget_generation(generation)
        if group is not None:
            self._note_progress_finished(group, target, presented=True)
        if on_done is not None:
            on_done(value)

    def _error(self, generation: int, exc: BaseException, on_error) -> None:
        group = self._generation_group.get(generation)
        target = self._generation_target.get(generation)
        self._forget_generation(generation)
        if group is not None:
            self._note_progress_finished(group, target, presented=False)
        if on_error is not None:
            on_error(exc)
        else:
            handle_ui_exception("background evaluation", exc)

    def _stale(self, generation: int, on_stale) -> None:
        group = self._generation_group.get(generation)
        target = self._generation_target.get(generation)
        self._forget_generation(generation)
        if group is not None:
            self._note_progress_finished(group, target, presented=False)
        if on_stale is not None:
            on_stale()

    def _forget_generation(self, generation: int) -> None:
        self._pending_generations.discard(generation)
        self._generation_group.pop(generation, None)
        self._generation_target.pop(generation, None)

    def _emit_slow(self, generation: int, callback) -> None:
        if generation in self._pending_generations:
            callback()

    def _note_progress_submitted(self, group: str, target: FrameTarget | None) -> None:
        if target is None:
            return
        current = self._frame_progress.get(group, FrameProgress())
        if current.active is None:
            self._frame_progress[group] = FrameProgress(
                presented=current.presented,
                active=target,
                queued_latest=current.queued_latest,
            )
        else:
            self._frame_progress[group] = FrameProgress(
                presented=current.presented,
                active=current.active,
                queued_latest=target,
            )

    def _note_progress_finished(self, group: str, target: FrameTarget | None, *, presented: bool) -> None:
        current = self._frame_progress.get(group, FrameProgress())
        active = None if current.active == target else current.active
        queued = None if current.queued_latest == target else current.queued_latest
        self._frame_progress[group] = FrameProgress(
            presented=target if presented and target is not None else current.presented,
            active=active,
            queued_latest=queued,
        )

    def _reset_progress(self) -> None:
        self._frame_progress = defaultdict(FrameProgress)

    def _diagnostic_frame_progress(self) -> FrameProgress:
        presented = active = queued = None
        for progress in self._frame_progress.values():
            presented = progress.presented or presented
            active = progress.active or active
            queued = progress.queued_latest or queued
        return FrameProgress(presented=presented, active=active, queued_latest=queued)

    def _note_prefetch_blocked(self, reason: str) -> None:
        reason = str(reason)
        if reason == "idle":
            self._prefetch_idle_blocked_count += 1
        elif reason == "visible_busy":
            self._prefetch_visible_busy_blocked_count += 1
        elif reason in {"cost", "budget", "memory"}:
            self._prefetch_cost_blocked_count += 1


__all__ = ["KernelEvaluationController"]
