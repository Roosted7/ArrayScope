"""Qt-side background evaluation orchestration."""

from __future__ import annotations

from collections import deque
from queue import SimpleQueue

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.app.errors import handle_ui_exception, traceback_text
from arrayscope.core.gui_callback_budget import GuiCallbackBudget
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.core.scheduler import EvalPriority, EvalRequest, FrameProgress, FrameTarget, SchedulerDiagnostics, WorkStart
from arrayscope.core.work_graph import WorkItem

prefer_pyside6()

import pyqtgraph.Qt as Qt


_UNSET = object()


class CancellationToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return bool(self._cancelled)


class _EvaluationRunnable(Qt.QtCore.QRunnable):
    def __init__(self, request, fn, queue, token, *, pass_token=False, notify_queue=None):
        super().__init__()
        self.request = request
        self.fn = fn
        self.queue = queue
        self.token = token
        self.pass_token = bool(pass_token)
        self.notify_queue = notify_queue
        self.started = False

    def _put(self, item) -> None:
        self.queue.put(item)
        if self.notify_queue is not None:
            try:
                self.notify_queue()
            except RuntimeError:
                # The owning QObject can be deleted while cooperative
                # cancellation is unwinding on a worker thread during window
                # close.  The result is already queued, and closed controllers
                # intentionally ignore late callbacks.
                pass

    def run(self):
        self.started = True
        self._put(("started", self.request.generation, None))
        if self.token.cancelled:
            self._put(("cancelled", self.request.generation, None))
            return
        try:
            value = self.fn(self.token) if self.pass_token else self.fn()
            self._put(("finished", self.request.generation, value))
        except EvaluationCancelled:
            self._put(("cancelled", self.request.generation, None))
        except Exception as exc:
            exc.arrayscope_traceback = traceback_text(exc)
            self._put(("failed", self.request.generation, exc))


class _PrefetchRunnable(Qt.QtCore.QRunnable):
    def __init__(self, fn, queue, key, *, notify_queue=None):
        super().__init__()
        self.fn = fn
        self.queue = queue
        self.key = key
        self.notify_queue = notify_queue

    def _put(self, item) -> None:
        self.queue.put(item)
        if self.notify_queue is not None:
            try:
                self.notify_queue()
            except RuntimeError:
                pass

    def run(self):
        try:
            self._put(("prefetch_done", self.key, self.fn()))
        except Exception as exc:
            exc.arrayscope_traceback = traceback_text(exc)
            self._put(("prefetch_failed", self.key, exc))


class EvaluationController(Qt.QtCore.QObject):
    queueEventReady = Qt.QtCore.Signal()

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
        self._supersession_values = {}
        self._frame_progress = {}
        self._prefetch_keys = set()
        self._prefetch_work_items = {}
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
        # Fallback timer, not the primary drain path. Worker notifications emit
        # `queueEventReady`; this single-shot safety net handles Qt bindings
        # that occasionally miss a cross-thread signal while work is active.
        # The interval adapts: while the signal path is proven healthy the
        # net backs off (fewer wakeups and less GIL churn under long compute);
        # the moment the net catches an event the signal should have
        # delivered, it snaps back to the fast interval.
        self._drain_fallback_min_ms = 10
        self._drain_fallback_max_ms = 100
        self._drain_fallback_interval_ms = self._drain_fallback_min_ms
        self._fallback_event_polls_count = 0
        self._fallback_idle_polls_count = 0
        self._drain_fallback_timer = Qt.QtCore.QTimer(self)
        self._drain_fallback_timer.setSingleShot(True)
        self._drain_fallback_timer.setInterval(self._drain_fallback_interval_ms)
        self._drain_fallback_timer.timeout.connect(self._on_drain_fallback)
        try:
            self.queueEventReady.connect(self._drain_queue, Qt.QtCore.Qt.ConnectionType.QueuedConnection)
        except Exception:
            self.queueEventReady.connect(self._drain_queue)

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
        for key in tuple(self._prefetch_keys):
            self._note_prefetch_work_dropped(key)
        self._prefetch_keys.clear()
        if not self._runnables:
            self._drain_fallback_timer.stop()
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
            self._drain_fallback_timer.stop()

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
        priority = EvalPriority(priority)
        supersession_value = self._normalize_supersession_value(supersession_key, supersession_value)
        if not self._admit_work_item(priority, work_item):
            if on_stale is not None:
                on_stale()
            return None
        if supersession_key is None:
            self.clear_group(replace_group)
            group_generation = self.advance_group(replace_group)
        else:
            self._collapse_group_queued(
                replace_group,
                supersession_key=supersession_key,
                preserve_started=False,
            )
            group_generation = self.group_generation(replace_group)
            self._advance_supersession(replace_group, supersession_key, supersession_value)
        self.generation += 1
        generation = self.generation
        request = EvalRequest(
            key=key,
            priority=priority,
            generation=generation,
            replace_group=replace_group,
            group_generation=group_generation,
            memory_budget_bytes=memory_budget_bytes,
            frame_target=frame_target,
            supersession_key=supersession_key,
            supersession_value=supersession_value,
            work_item=work_item,
        )
        token = CancellationToken()
        self._pending.add(generation)
        self._requests[generation] = request
        self._tokens[generation] = token
        self._handlers[generation] = (on_done, on_error, on_stale, None)
        self._index_request(request)
        if on_slow is not None:
            # User-visible timeout with this controller as receiver context.
            # `_emit_slow` rechecks generation and supersession before
            # surfacing delayed-work feedback.
            Qt.QtCore.QTimer.singleShot(int(slow_ms), self, lambda generation=generation: self._emit_slow(generation, on_slow))

        runnable = _EvaluationRunnable(
            request,
            fn,
            self._queue,
            token,
            pass_token=pass_token,
            notify_queue=self._notify_queue_event,
        )
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
        supersession_value=_UNSET,
        work_item: WorkItem | None = None,
    ):
        replace_group = str(replace_group)
        priority = EvalPriority(priority)
        supersession_value = self._normalize_supersession_value(supersession_key, supersession_value)
        if not self._admit_work_item(priority, work_item):
            if on_stale is not None:
                on_stale()
            return None
        self._collapse_group_queued(replace_group, supersession_key=supersession_key)
        self.generation += 1
        generation = self.generation
        if supersession_key is None:
            group_generation = self.advance_group(replace_group)
        else:
            group_generation = self.group_generation(replace_group)
            self._advance_supersession(replace_group, supersession_key, supersession_value)
        request = EvalRequest(
            key=key,
            priority=priority,
            generation=generation,
            replace_group=replace_group,
            group_generation=group_generation,
            memory_budget_bytes=memory_budget_bytes,
            frame_target=frame_target,
            supersession_key=supersession_key,
            supersession_value=supersession_value,
            work_item=work_item,
        )
        token = CancellationToken()
        self._pending.add(generation)
        self._requests[generation] = request
        self._tokens[generation] = token
        self._handlers[generation] = (on_done, on_error, on_stale, on_reuse_stale)
        self._index_request(request)
        if on_slow is not None:
            # User-visible timeout with this controller as receiver context.
            # `_emit_slow` rechecks generation and supersession before
            # surfacing delayed-work feedback.
            Qt.QtCore.QTimer.singleShot(int(slow_ms), self, lambda generation=generation: self._emit_slow(generation, on_slow))

        runnable = _EvaluationRunnable(
            request,
            fn,
            self._queue,
            token,
            pass_token=pass_token,
            notify_queue=self._notify_queue_event,
        )
        self._runnables[generation] = runnable
        self._refresh_frame_progress(replace_group)
        self.pool.start(runnable)
        self._ensure_polling()
        return generation

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
        if self._shutting_down:
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
        key = ("prefetch", id(fn), len(self._runnables)) if key is None else key
        if key in self._prefetch_keys:
            self._prefetch_deduped_count += 1
            return WorkStart(False, "deduped")
        if len(self._prefetch_keys) >= self._max_prefetch:
            self._prefetch_limited_count += 1
            return WorkStart(False, "limited")
        graph = self._work_graph()
        if graph is not None and work_item is not None:
            visible_backlog = bool(getattr(graph, "visible_backlog", 0))
            decision = graph.submit(
                work_item,
                available_budget=not visible_backlog,
                visible_backlog=visible_backlog,
            )
            if not decision.admitted:
                self._note_prefetch_blocked(decision.reason)
                return WorkStart(False, decision.reason)
        self._prefetch_keys.add(key)
        if work_item is not None:
            self._prefetch_work_items[key] = work_item
        self._prefetch_scheduled_count += 1
        runnable = _PrefetchRunnable(fn, self._queue, key, notify_queue=self._notify_queue_event)
        self._runnables[key] = runnable
        if on_done is not None:
            self._handlers[key] = (on_done, None, None, None)
        self.pool.start(runnable)
        self._ensure_polling()
        return WorkStart(True)

    def cancel_prefetch(self) -> None:
        for key in tuple(self._prefetch_keys):
            self._prefetch_keys.discard(key)
            self._note_prefetch_work_dropped(key)
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
        work_lanes = tuple(
            sorted(
                {
                    str(getattr(getattr(request, "work_item", None), "lane", ""))
                    for request in self._requests.values()
                    if getattr(request, "work_item", None) is not None
                }
            )
        )
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
            fallback_event_polls=int(self._fallback_event_polls_count),
            fallback_idle_polls=int(self._fallback_idle_polls_count),
            presented_target=progress.presented,
            active_target=progress.active,
            queued_latest_target=progress.queued_latest,
            work_lanes=work_lanes,
            work_graph=None if self._work_graph() is None else self._work_graph().diagnostics(),
        )

    def _emit_slow(self, generation, callback):
        request = self._requests.get(generation)
        if request is not None and generation in self._pending and not self._is_request_stale(request):
            callback()

    def _ensure_polling(self):
        # Queued signal drain: worker completions and bounded continuations
        # emit `queueEventReady`. The fallback timer is a low-frequency
        # single-shot safety net, not semantic ordering.
        self._notify_queue_event()
        self._schedule_drain_fallback()

    def _notify_queue_event(self) -> None:
        if self._shutting_down:
            return
        try:
            self.queueEventReady.emit()
        except RuntimeError:
            pass

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
                self._note_prefetch_work_finished(key, failed=False)
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
                self._note_prefetch_work_finished(key, failed=True)
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
        if budget.processed_items > 0 or budget.elapsed_ms >= budget.warning_ms:
            recorder = getattr(getattr(self.parent(), "resource_governor", None), "record_gui_callback_observation", None)
            if callable(recorder):
                recorder(budget.observation())
        if (self._pending_queue_events or not self._queue.empty()) and (self._runnables or self._handlers):
            while not self._queue.empty():
                self._pending_queue_events.append(self._queue.get())
            self._schedule_drain_continuation()
            self._schedule_drain_fallback()
        if self._runnables or self._handlers:
            self._schedule_drain_fallback()
        elif self._queue.empty() and not self._pending_queue_events:
            self._drain_fallback_timer.stop()

    def _schedule_drain_continuation(self) -> None:
        if self._drain_continuation_pending:
            return
        self._drain_continuation_pending = True
        self._notify_queue_event()

    def _schedule_drain_fallback(self) -> None:
        if self._shutting_down:
            return
        if not self._drain_fallback_timer.isActive():
            self._drain_fallback_timer.start()

    def _on_drain_fallback(self) -> None:
        # Adapt the safety-net cadence from what this poll actually found.
        # Events sitting in the queue mean the fallback found pending work:
        # poll fast again, but report this as an event-bearing poll rather
        # than proof that the queued signal path failed.
        # An empty poll means the signal path is doing its job: back off.
        if self._pending_queue_events or not self._queue.empty():
            self._fallback_event_polls_count += 1
            self._drain_fallback_interval_ms = self._drain_fallback_min_ms
        else:
            self._fallback_idle_polls_count += 1
            self._drain_fallback_interval_ms = min(
                self._drain_fallback_max_ms, self._drain_fallback_interval_ms * 2
            )
        self._drain_fallback_timer.setInterval(self._drain_fallback_interval_ms)
        self._drain_queue()

    def _finish(self, generation, value):
        request = self._requests.get(generation)
        stale = request is None or self._is_request_stale(request)
        replace_group = None if request is None else request.replace_group
        target = None if request is None else request.frame_target
        self._cleanup_generation(generation)
        on_done, _on_error, on_stale, on_reuse = self._handlers.pop(generation, (None, None, None, None))
        if on_done is None:
            self._note_work_finished(request, stale=stale, failed=False)
            if replace_group is not None:
                self._refresh_frame_progress(replace_group)
            return
        if stale:
            self._stale_count += 1
            reused = False
            if on_reuse is not None:
                try:
                    on_reuse(value)
                    self._stale_reused_count += 1
                    reused = True
                except Exception as exc:
                    handle_ui_exception("stale evaluation reuse callback", exc)
            self._note_work_finished(request, stale=True, failed=False, reusable=reused)
            if on_stale is not None:
                on_stale()
            if replace_group is not None:
                self._refresh_frame_progress(replace_group)
            return
        self._completed_count += 1
        self._note_work_finished(request, stale=False, failed=False)
        on_done(value)
        if replace_group is not None:
            self._refresh_frame_progress(replace_group, presented=target)

    def _fail(self, generation, exc):
        request = self._requests.get(generation)
        stale = request is None or self._is_request_stale(request)
        replace_group = None if request is None else request.replace_group
        self._cleanup_generation(generation)
        _on_done, on_error, on_stale, _on_reuse = self._handlers.pop(generation, (None, None, None, None))
        self._note_work_finished(request, stale=stale, failed=True)
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
        if request is not None:
            self._note_work_dropped(request)
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
                for key in tuple(self._supersession_values):
                    if key[0] == group:
                        self._supersession_values.pop(key, None)
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

    def _collapse_group_queued(self, replace_group: str, *, supersession_key=None, preserve_started: bool = True) -> None:
        groups = {str(replace_group)}
        groups.update(self._group_child_groups.get(str(replace_group), ()))
        preserved = 0
        collapsed = 0
        for group in tuple(groups):
            for generation in tuple(self._group_request_generations.get(group, ())):
                request = self._requests.get(generation)
                if supersession_key is not None and (
                    request is None or request.supersession_key != supersession_key
                ):
                    continue
                runnable = self._runnables.get(generation)
                if runnable is None:
                    continue
                if preserve_started and (getattr(runnable, "started", False) or generation in self._started):
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

    @staticmethod
    def _normalize_supersession_value(supersession_key, supersession_value):
        if supersession_key is None:
            return None
        if supersession_value is _UNSET:
            return supersession_key
        return supersession_value

    def _advance_supersession(self, replace_group: str, supersession_key, supersession_value) -> None:
        self._supersession_values[(str(replace_group), supersession_key)] = supersession_value

    def _is_request_stale(self, request) -> bool:
        if self._shutting_down:
            return True
        if request.group_generation != self.group_generation(request.replace_group):
            return True
        if request.supersession_key is None:
            return False
        current = self._supersession_values.get((str(request.replace_group), request.supersession_key), _UNSET)
        return current is _UNSET or current != request.supersession_value

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

    def _work_graph(self):
        return getattr(self.parent(), "work_graph", None)

    def _admit_work_item(self, priority: EvalPriority, work_item: WorkItem | None) -> bool:
        graph = self._work_graph()
        if graph is None:
            return True
        if priority == EvalPriority.VISIBLE_IMAGE and work_item is None:
            raise ValueError("visible evaluation submissions require an admitted WorkItem")
        if work_item is None:
            return True
        visible_backlog = bool(getattr(graph, "visible_backlog", 0))
        decision = graph.submit(
            work_item,
            available_budget=priority != EvalPriority.PREFETCH or not visible_backlog,
            visible_backlog=visible_backlog,
        )
        return bool(decision.admitted)

    def _note_work_finished(self, request, *, stale: bool, failed: bool, reusable: bool = False) -> None:
        graph = self._work_graph()
        item = None if request is None else getattr(request, "work_item", None)
        if graph is None or item is None:
            return
        if failed:
            graph.fail(item.key, stale=stale)
        else:
            graph.complete(item.key, stale=stale, reusable_output=reusable)

    def _note_work_dropped(self, request) -> None:
        graph = self._work_graph()
        item = None if request is None else getattr(request, "work_item", None)
        if graph is None or item is None:
            return
        graph.drop(item.key)

    def _note_prefetch_work_finished(self, key, *, failed: bool) -> None:
        graph = self._work_graph()
        item = self._prefetch_work_items.pop(key, None)
        if graph is None or item is None:
            return
        if failed:
            graph.fail(item.key)
        else:
            graph.complete(item.key)

    def _note_prefetch_work_dropped(self, key) -> None:
        graph = self._work_graph()
        item = self._prefetch_work_items.pop(key, None)
        if graph is None or item is None:
            return
        graph.drop(item.key)
