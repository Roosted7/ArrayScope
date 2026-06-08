"""Qt-side background evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from queue import SimpleQueue

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.app.errors import handle_ui_exception, traceback_text

prefer_pyside6()

import pyqtgraph.Qt as Qt


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
    memory_budget_bytes: int | None = None


@dataclass(frozen=True)
class PrefetchStart:
    scheduled: bool
    reason: str = "scheduled"


class CancellationToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return bool(self._cancelled)


class _EvaluationRunnable(Qt.QtCore.QRunnable):
    def __init__(self, request, fn, queue, token):
        super().__init__()
        self.request = request
        self.fn = fn
        self.queue = queue
        self.token = token
        self.started = False

    def run(self):
        self.started = True
        self.queue.put(("started", self.request.generation, None))
        if self.token.cancelled:
            self.queue.put(("cancelled", self.request.generation, None))
            return
        try:
            self.queue.put(("finished", self.request.generation, self.fn()))
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
    def __init__(self, parent=None, *, max_workers=None, name="evaluation"):
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
        self._prefetch_keys = set()
        self._max_prefetch = 32
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
        self.pool.clear()
        for generation, request in tuple(self._requests.items()):
            if request.replace_group != replace_group:
                continue
            token = self._tokens.get(generation)
            if token is not None:
                token.cancel()
            runnable = self._runnables.get(generation)
            if runnable is not None and not getattr(runnable, "started", False):
                self._discard_generation(generation, stale=True)
        if not self._runnables:
            self._poll_timer.stop()

    def shutdown_for_close(self):
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
    ):
        replace_group = str(replace_group)
        self.clear_group(replace_group)
        self.generation += 1
        generation = self.generation
        request = EvalRequest(
            key=key,
            priority=EvalPriority(priority),
            generation=generation,
            replace_group=replace_group,
            memory_budget_bytes=memory_budget_bytes,
        )
        token = CancellationToken()
        self._pending.add(generation)
        self._requests[generation] = request
        self._tokens[generation] = token
        self._group_generations[replace_group] = generation
        self._handlers[generation] = (on_done, on_error, on_stale)
        if on_slow is not None:
            Qt.QtCore.QTimer.singleShot(int(slow_ms), lambda generation=generation: self._emit_slow(generation, on_slow))

        runnable = _EvaluationRunnable(request, fn, self._queue, token)
        self._runnables[generation] = runnable
        self.pool.start(runnable)
        self._ensure_polling()
        return generation

    def start_prefetch(self, fn, on_done=None, *, key=None, memory_budget_bytes=None):
        del memory_budget_bytes
        key = ("prefetch", id(fn), len(self._runnables)) if key is None else key
        if key in self._prefetch_keys:
            return PrefetchStart(False, "deduped")
        if len(self._prefetch_keys) >= self._max_prefetch:
            return PrefetchStart(False, "limited")
        self._prefetch_keys.add(key)
        runnable = _PrefetchRunnable(fn, self._queue, key)
        self._runnables[key] = runnable
        if on_done is not None:
            self._handlers[key] = (on_done, None, None)
        self.pool.start(runnable)
        self._ensure_polling()
        return PrefetchStart(True)

    def _emit_slow(self, generation, callback):
        if generation in self._pending and generation == self.generation:
            callback()

    def _ensure_polling(self):
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _drain_queue(self):
        while not self._queue.empty():
            kind, key, value = self._queue.get()
            if kind == "started":
                self._started.add(key)
                continue
            if kind == "cancelled":
                self._discard_generation(key, stale=True)
                continue
            if kind == "prefetch_done":
                self._prefetch_keys.discard(key)
                self._runnables.pop(key, None)
                on_done, _on_error, _on_stale = self._handlers.pop(key, (None, None, None))
                if on_done is not None:
                    try:
                        on_done(value)
                    except Exception as exc:
                        handle_ui_exception("prefetch callback", exc)
                continue
            if kind == "prefetch_failed":
                self._prefetch_keys.discard(key)
                self._runnables.pop(key, None)
                self._handlers.pop(key, None)
                continue
            if kind == "finished":
                self._finish(key, value)
            elif kind == "failed":
                self._fail(key, value)
        if not self._runnables:
            self._poll_timer.stop()

    def _finish(self, generation, value):
        request = self._requests.get(generation)
        self._cleanup_generation(generation)
        on_done, _on_error, on_stale = self._handlers.pop(generation, (None, None, None))
        if on_done is None:
            return
        if request is None or generation != self._group_generations.get(request.replace_group):
            if on_stale is not None:
                on_stale()
            return
        on_done(value)

    def _fail(self, generation, exc):
        request = self._requests.get(generation)
        self._cleanup_generation(generation)
        _on_done, on_error, on_stale = self._handlers.pop(generation, (None, None, None))
        if request is None or generation != self._group_generations.get(request.replace_group):
            if on_stale is not None:
                on_stale()
            return
        if on_error is not None:
            on_error(exc)
        else:
            handle_ui_exception("background evaluation", exc)

    def _discard_generation(self, generation, *, stale=False):
        _on_done, _on_error, on_stale = self._handlers.pop(generation, (None, None, None))
        self._cleanup_generation(generation)
        if stale and on_stale is not None:
            on_stale()

    def _cleanup_generation(self, generation):
        self._pending.discard(generation)
        self._started.discard(generation)
        self._runnables.pop(generation, None)
        self._requests.pop(generation, None)
        self._tokens.pop(generation, None)

    def _remove_not_started_runnables(self):
        for key, runnable in tuple(self._runnables.items()):
            if isinstance(key, int) and not getattr(runnable, "started", False):
                self._discard_generation(key, stale=True)
