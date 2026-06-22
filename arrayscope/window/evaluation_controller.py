"""Qt-side background evaluation orchestration."""

from __future__ import annotations

from collections import deque
from queue import SimpleQueue

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.app.errors import handle_ui_exception, traceback_text
from arrayscope.core.gui_callback_budget import GuiCallbackBudget
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.core.scheduler import EvalPriority, EvalRequest, FrameProgress, FrameTarget, PrefetchStart, SchedulerDiagnostics

prefer_pyside6()

import pyqtgraph.Qt as Qt


class CancellationToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return bool(self._cancelled)


class _EvaluationRunnable(Qt.QtCore.QRunnable):
    def __init__(self, request, fn, queue, token, *, pass_token=False):
        super().__init__()
        self.request = request
        self.fn = fn
        self.queue = queue
        self.token = token
        self.pass_token = bool(pass_token)
        self.started = False

    def run(self):
        self.started = True
        self.queue.put(("started", self.request.generation, None))
        if self.token.cancelled:
            self.queue.put(("cancelled", self.request.generation, None))
            return
        try:
            value = self.fn(self.token) if self.pass_token else self.fn()
            self.queue.put(("finished", self.request.generation, value))
        except EvaluationCancelled:
            self.queue.put(("cancelled", self.request.generation, None))
        except Exception as exc:
            exc.arrayscope_traceback = traceback_text(exc)
            self.queue.put(("failed", self.request.generation, exc))


class _PrefetchRunnable(Qt.QtCore.QRunnable):
    def __init__(self, fn, queue, key):
        super().__init__()
        self.fn = fn
        self.queue = queue
        self.key = key

    def run(self):
        try:
            self.queue.put(("prefetch_done", self.key, self.fn()))
        except Exception as exc:
            exc.arrayscope_traceback = traceback_text(exc)
            self.queue.put(("prefetch_failed", self.key, exc))


class EvaluationController(Qt.QtCore.QObject):
    def __init__(
        self,
        parent=None,
        *,
        max_workers=None,
        name="evaluation",
        max_callback_dispatch_per_drain: int = 4,
        max_queue_events_per_drain: int = 64,
    ):
        super().__init__(parent)
        self.name = str(name)
        self.pool = Qt.QtCore.QThreadPool(self)
        if max_workers is not None:
            self.pool.setMaxThreadCount(max(1, int(max_workers)))
        self.generation = 0
        self._pending = set()
        self._started = set()
        self._runnables = {}
        self._requests = {}
        self._handlers = {}
        self._tokens = {}
        self._group_generations = {}
        self._group_request_generations = {}
        self._group_child_groups = {}
        self._group_epoch = 0
        self._frame_progress = {}
        self._prefetch_keys = set()
        self._max_prefetch = 32
        self._shutting_down = False
        self._completed_count = 0
        self._cancelled_count = 0
        self._stale_count = 0
        self._failed_count = 0
        self._prefetch_scheduled_count = 0
        self._prefetch_deduped_count = 0
        self._prefetch_limited_count = 0
        self._prefetch_idle_blocked_count = 0
        self._prefetch_visible_busy_blocked_count = 0
        self._prefetch_cost_blocked_count = 0
        self._active_preserved_count = 0
        self._queued_collapsed_count = 0
        self._stale_reused_count = 0
        self._max_callback_dispatch_per_drain = max(1, int(max_callback_dispatch_per_drain))
        self._max_queue_events_per_drain = max(1, int(max_queue_events_per_drain))
        self._callback_budget_ms: float | None = None
        self._pending_queue_events = deque()
        self._drain_continuation_pending = False
        self._queue = SimpleQueue()
        self._poll_timer = Qt.QtCore.QTimer(self)
        self._poll_timer.setInterval(10)
        self._poll_timer.timeout.connect(self._drain_queue)

    def cancel_pending(self):
        self.generation += 1
        for token in self._tokens.values():
            token.cancel()
        self._pending.clear()
        return self.generation

    def clear_queued(self):
        self.generation += 1
        self.pool.clear()
        for token in self._tokens.values():
            token.cancel()
        self._pending.clear()
        self._handlers.clear()
        self._remove_not_started_runnables()
        self._prefetch_keys.clear()
        if not self._runnables:
            self._poll_timer.stop()
        return self.generation

    def clear_group(self, replace_group: str):
        replace_group = str(replace_group)
        self.advance_group(replace_group)
        groups = {replace_group}
        groups.update(self._group_child_groups.get(replace_group, ()))
        for group in tuple(groups):
            self.advance_group(group)
            for generation in tuple(self._group_request_generations.get(group, ())):
                token = self._tokens.get(generation)
                if token is not None:
                    token.cancel()
                runnable = self._runnables.get(generation)
                if runnable is not None and not getattr(runnable, "started", False):
                    self._discard_generation(generation, stale=True)
            self._refresh_frame_progress(group)
        if not self._runnables:
            self._poll_timer.stop()

    def advance_group(self, replace_group: str) -> int:
        replace_group = str(replace_group)
        self._group_epoch += 1
        self._group_generations[replace_group] = self._group_epoch
        return self._group_generations[replace_group]

    def group_generation(self, replace_group: str) -> int:
        return int(self._group_generations.get(str(replace_group), 0))

    def shutdown_for_close(self):
        self._shutting_down = True
        return self.clear_queued()

    def start(self, fn, *, on_done, on_error=None, on_stale=None, on_slow=None, slow_ms=100):
        return self.start_latest(
            fn,
            key=("compat", self.generation + 1),
            priority=EvalPriority.VISIBLE_IMAGE,
            replace_group="default",
            on_done=on_done,
            on_error=on_error,
            on_stale=on_stale,
            on_slow=on_slow,
            slow_ms=slow_ms,
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
    ):
        replace_group = str(replace_group)
        self.clear_group(replace_group)
        self.generation += 1
        generation = self.generation
        group_generation = self.advance_group(replace_group)
        request = EvalRequest(
            key=key,
            priority=EvalPriority(priority),
            generation=generation,
            replace_group=replace_group,
            group_generation=group_generation,
            memory_budget_bytes=memory_budget_bytes,
            frame_target=frame_target,
            supersession_key=supersession_key,
        )
        token = CancellationToken()
        self._pending.add(generation)
        self._requests[generation] = request
        self._tokens[generation] = token
        self._handlers[generation] = (on_done, on_error, on_stale, None)
        self._index_request(request)
        if on_slow is not None:
            Qt.QtCore.QTimer.singleShot(int(slow_ms), lambda generation=generation: self._emit_slow(generation, on_slow))

        runnable = _EvaluationRunnable(request, fn, self._queue, token, pass_token=pass_token)
        self._runnables[generation] = runnable
        self._refresh_frame_progress(replace_group)
        self.pool.start(runnable)
        self._ensure_polling()
        return generation

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
    ):
        replace_group = str(replace_group)
        self._collapse_group_queued(replace_group)
        self.generation += 1
        generation = self.generation
        group_generation = self.advance_group(replace_group)
        request = EvalRequest(
            key=key,
            priority=EvalPriority(priority),
            generation=generation,
            replace_group=replace_group,
            group_generation=group_generation,
            memory_budget_bytes=memory_budget_bytes,
            frame_target=frame_target,
            supersession_key=supersession_key,
        )
        token = CancellationToken()
        self._pending.add(generation)
        self._requests[generation] = request
        self._tokens[generation] = token
        self._handlers[generation] = (on_done, on_error, on_stale, on_reuse_stale)
        self._index_request(request)
        if on_slow is not None:
            Qt.QtCore.QTimer.singleShot(int(slow_ms), lambda generation=generation: self._emit_slow(generation, on_slow))

        runnable = _EvaluationRunnable(request, fn, self._queue, token, pass_token=pass_token)
        self._runnables[generation] = runnable
        self._refresh_frame_progress(replace_group)
        self.pool.start(runnable)
        self._ensure_polling()
        return generation

    def start_prefetch(self, fn, on_done=None, *, key=None, memory_budget_bytes=None, idle_elapsed=None, blocked_reason=None):
        if self._shutting_down:
            return PrefetchStart(False, "closed")
        if blocked_reason:
            self._note_prefetch_blocked(blocked_reason)
            return PrefetchStart(False, blocked_reason)
        if idle_elapsed is False:
            self._note_prefetch_blocked("idle")
            return PrefetchStart(False, "idle")
        if memory_budget_bytes is not None and int(memory_budget_bytes) <= 0:
            self._note_prefetch_blocked("cost")
            return PrefetchStart(False, "cost")
        key = ("prefetch", id(fn), len(self._runnables)) if key is None else key
        if key in self._prefetch_keys:
            self._prefetch_deduped_count += 1
            return PrefetchStart(False, "deduped")
        if len(self._prefetch_keys) >= self._max_prefetch:
            self._prefetch_limited_count += 1
            return PrefetchStart(False, "limited")
        self._prefetch_keys.add(key)
        self._prefetch_scheduled_count += 1
        runnable = _PrefetchRunnable(fn, self._queue, key)
        self._runnables[key] = runnable
        if on_done is not None:
            self._handlers[key] = (on_done, None, None, None)
        self.pool.start(runnable)
        self._ensure_polling()
        return PrefetchStart(True)

    def cancel_prefetch(self) -> None:
        for key in tuple(self._prefetch_keys):
            self._prefetch_keys.discard(key)
            self._runnables.pop(key, None)
            self._handlers.pop(key, None)
        self.pool.clear()

    def set_max_workers(self, count: int) -> None:
        self.pool.setMaxThreadCount(max(1, int(count)))

    def set_max_callback_dispatch_per_drain(self, count: int) -> None:
        self._max_callback_dispatch_per_drain = max(1, int(count))

    def set_callback_budget_ms(self, budget_ms: float | None) -> None:
        self._callback_budget_ms = None if budget_ms is None else max(0.25, float(budget_ms))

    def set_max_prefetch(self, count: int) -> None:
        self._max_prefetch = max(0, int(count))

    def is_busy(self) -> bool:
        return self.has_running_or_pending()

    def has_running_or_pending(self) -> bool:
        return bool(self._pending or self._started or self._runnables)

    def frame_progress(self, replace_group: str) -> FrameProgress:
        return self._frame_progress.get(str(replace_group), FrameProgress())

    def note_stale_reused(self) -> None:
        self._stale_reused_count += 1

    def diagnostics(self) -> SchedulerDiagnostics:
        running = len(self._started)
        pending = len(self._pending)
        queued = max(0, len(self._runnables) - running)
        progress = self._diagnostic_frame_progress()
        return SchedulerDiagnostics(
            name=self.name,
            max_workers=int(self.pool.maxThreadCount()),
            pending=pending,
            running=running,
            queued=queued,
            started=running,
            cancelled=int(self._cancelled_count),
            stale=int(self._stale_count),
            completed=int(self._completed_count),
            failed=int(self._failed_count),
            prefetch_scheduled=int(self._prefetch_scheduled_count),
            prefetch_deduped=int(self._prefetch_deduped_count),
            prefetch_limited=int(self._prefetch_limited_count),
            prefetch_idle_blocked=int(self._prefetch_idle_blocked_count),
            prefetch_visible_busy_blocked=int(self._prefetch_visible_busy_blocked_count),
            prefetch_cost_blocked=int(self._prefetch_cost_blocked_count),
            active_preserved=int(self._active_preserved_count),
            queued_collapsed=int(self._queued_collapsed_count),
            stale_reused=int(self._stale_reused_count),
            presented_target=progress.presented,
            active_target=progress.active,
            queued_latest_target=progress.queued_latest,
        )

    def _emit_slow(self, generation, callback):
        request = self._requests.get(generation)
        if request is not None and generation in self._pending and request.group_generation == self.group_generation(request.replace_group):
            callback()

    def _ensure_polling(self):
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _drain_queue(self):
        self._drain_continuation_pending = False
        budget = GuiCallbackBudget(
            channel=f"{self.name}_queue_drain",
            work_class="evaluation_callback",
            backend="qt",
            target_ms=(
                float(self._callback_budget_ms)
                if self._callback_budget_ms is not None
                else 8.0
            ),
            item_cap=int(self._max_callback_dispatch_per_drain),
            byte_cap=0,
        )
        processed_events = 0
        while (self._pending_queue_events or not self._queue.empty()) and processed_events < self._max_queue_events_per_drain:
            if self._pending_queue_events:
                kind, key, value = self._pending_queue_events.popleft()
            else:
                kind, key, value = self._queue.get()
            processed_events += 1
            if kind == "started":
                self._started.add(key)
                request = self._requests.get(key)
                if request is not None:
                    self._refresh_frame_progress(request.replace_group)
                continue
            if kind == "cancelled":
                self._cancelled_count += 1
                self._discard_generation(key, stale=True)
                continue
            if kind == "prefetch_done":
                self._prefetch_keys.discard(key)
                self._runnables.pop(key, None)
                on_done, _on_error, _on_stale, _on_reuse = self._handlers.pop(key, (None, None, None, None))
                if on_done is not None:
                    try:
                        on_done(value)
                    except Exception as exc:
                        handle_ui_exception("prefetch callback", exc)
                    budget.record_item()
                    if budget.should_yield():
                        break
                continue
            if kind == "prefetch_failed":
                self._prefetch_keys.discard(key)
                self._runnables.pop(key, None)
                self._handlers.pop(key, None)
                continue
            if kind == "finished":
                self._finish(key, value)
                budget.record_item()
            elif kind == "failed":
                self._fail(key, value)
                budget.record_item()
            if budget.should_yield():
                break
        if (self._pending_queue_events or not self._queue.empty()) and (self._runnables or self._handlers):
            while not self._queue.empty():
                self._pending_queue_events.append(self._queue.get())
            self._schedule_drain_continuation()
        if not self._runnables and self._queue.empty() and not self._pending_queue_events:
            self._poll_timer.stop()

    def _schedule_drain_continuation(self) -> None:
        if self._drain_continuation_pending:
            return
        self._drain_continuation_pending = True
        Qt.QtCore.QTimer.singleShot(0, self._drain_queue)

    def _finish(self, generation, value):
        request = self._requests.get(generation)
        stale = request is None or request.group_generation != self.group_generation(request.replace_group) or self._shutting_down
        replace_group = None if request is None else request.replace_group
        target = None if request is None else request.frame_target
        self._cleanup_generation(generation)
        on_done, _on_error, on_stale, on_reuse = self._handlers.pop(generation, (None, None, None, None))
        if on_done is None:
            if replace_group is not None:
                self._refresh_frame_progress(replace_group)
            return
        if stale:
            self._stale_count += 1
            if on_reuse is not None:
                try:
                    on_reuse(value)
                    self._stale_reused_count += 1
                except Exception as exc:
                    handle_ui_exception("stale evaluation reuse callback", exc)
            if on_stale is not None:
                on_stale()
            if replace_group is not None:
                self._refresh_frame_progress(replace_group)
            return
        self._completed_count += 1
        on_done(value)
        if replace_group is not None:
            self._refresh_frame_progress(replace_group, presented=target)

    def _fail(self, generation, exc):
        request = self._requests.get(generation)
        stale = request is None or request.group_generation != self.group_generation(request.replace_group) or self._shutting_down
        replace_group = None if request is None else request.replace_group
        self._cleanup_generation(generation)
        _on_done, on_error, on_stale, _on_reuse = self._handlers.pop(generation, (None, None, None, None))
        if stale:
            self._stale_count += 1
            if on_stale is not None:
                on_stale()
            if replace_group is not None:
                self._refresh_frame_progress(replace_group)
            return
        self._failed_count += 1
        if on_error is not None:
            on_error(exc)
        else:
            handle_ui_exception("background evaluation", exc)
        if replace_group is not None:
            self._refresh_frame_progress(replace_group)

    def _discard_generation(self, generation, *, stale=False):
        request = self._requests.get(generation)
        replace_group = None if request is None else request.replace_group
        _on_done, _on_error, on_stale, _on_reuse = self._handlers.pop(generation, (None, None, None, None))
        self._cleanup_generation(generation)
        if stale and on_stale is not None:
            self._stale_count += 1
            on_stale()
        if replace_group is not None:
            self._refresh_frame_progress(replace_group)

    def _cleanup_generation(self, generation):
        self._pending.discard(generation)
        self._started.discard(generation)
        self._runnables.pop(generation, None)
        request = self._requests.pop(generation, None)
        if request is not None:
            self._unindex_request(request)
        self._tokens.pop(generation, None)

    def _index_request(self, request):
        group = str(request.replace_group)
        self._group_request_generations.setdefault(group, set()).add(request.generation)
        for parent in self._group_parents(group):
            self._group_child_groups.setdefault(parent, set()).add(group)

    def _unindex_request(self, request):
        group = str(request.replace_group)
        generations = self._group_request_generations.get(group)
        if generations is not None:
            generations.discard(request.generation)
            if not generations:
                self._group_request_generations.pop(group, None)
                self._group_generations.pop(group, None)
                for parent in self._group_parents(group):
                    children = self._group_child_groups.get(parent)
                    if children is None:
                        continue
                    children.discard(group)
                    if not children:
                        self._group_child_groups.pop(parent, None)

    @staticmethod
    def _group_parents(group: str) -> tuple[str, ...]:
        parts = str(group).split(":")
        if len(parts) <= 1:
            return ()
        return tuple(":".join(parts[:index]) for index in range(1, len(parts)))

    def _remove_not_started_runnables(self):
        for key, runnable in tuple(self._runnables.items()):
            if isinstance(key, int) and not getattr(runnable, "started", False):
                self._discard_generation(key, stale=True)

    def _collapse_group_queued(self, replace_group: str) -> None:
        groups = {str(replace_group)}
        groups.update(self._group_child_groups.get(str(replace_group), ()))
        preserved = 0
        collapsed = 0
        for group in tuple(groups):
            for generation in tuple(self._group_request_generations.get(group, ())):
                runnable = self._runnables.get(generation)
                if runnable is None:
                    continue
                if getattr(runnable, "started", False) or generation in self._started:
                    preserved += 1
                    continue
                token = self._tokens.get(generation)
                if token is not None:
                    token.cancel()
                self._discard_generation(generation, stale=True)
                collapsed += 1
        self._active_preserved_count += int(preserved)
        self._queued_collapsed_count += int(collapsed)
        self._refresh_frame_progress(replace_group)

    def _refresh_frame_progress(self, replace_group: str, *, presented: FrameTarget | None = None) -> None:
        replace_group = str(replace_group)
        previous = self._frame_progress.get(replace_group, FrameProgress())
        active = self._latest_frame_target(replace_group, started=True)
        queued = self._latest_frame_target(replace_group, started=False)
        current = FrameProgress(
            presented=presented if presented is not None else previous.presented,
            active=active,
            queued_latest=queued,
        )
        if current == FrameProgress():
            self._frame_progress.pop(replace_group, None)
        else:
            self._frame_progress[replace_group] = current

    def _latest_frame_target(self, replace_group: str, *, started: bool) -> FrameTarget | None:
        generations = self._group_request_generations.get(str(replace_group), ())
        candidates = []
        for generation in generations:
            request = self._requests.get(generation)
            runnable = self._runnables.get(generation)
            if request is None or runnable is None or request.frame_target is None:
                continue
            is_started = bool(getattr(runnable, "started", False) or generation in self._started)
            if is_started == bool(started):
                candidates.append((int(generation), request.frame_target))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def _diagnostic_frame_progress(self) -> FrameProgress:
        for progress in self._frame_progress.values():
            if progress.active is not None or progress.queued_latest is not None:
                return progress
        for progress in self._frame_progress.values():
            if progress.presented is not None:
                return progress
        return FrameProgress()

    def _note_prefetch_blocked(self, reason):
        if reason == "idle":
            self._prefetch_idle_blocked_count += 1
        elif reason == "visible_busy":
            self._prefetch_visible_busy_blocked_count += 1
        elif reason == "cost":
            self._prefetch_cost_blocked_count += 1
