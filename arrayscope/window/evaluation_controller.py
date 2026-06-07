"""Qt-side background evaluation orchestration."""

from __future__ import annotations

from queue import SimpleQueue

from arrayscope.app.qt_binding import prefer_pyside6
from arrayscope.app.errors import handle_ui_exception, traceback_text

prefer_pyside6()

import pyqtgraph.Qt as Qt


class _EvaluationRunnable(Qt.QtCore.QRunnable):
    def __init__(self, generation, fn, queue):
        super().__init__()
        self.generation = int(generation)
        self.fn = fn
        self.queue = queue

    def run(self):
        try:
            self.queue.put(("finished", self.generation, self.fn()))
        except Exception as exc:
            exc.arrayscope_traceback = traceback_text(exc)
            self.queue.put(("failed", self.generation, exc))


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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = Qt.QtCore.QThreadPool.globalInstance()
        self.generation = 0
        self._pending = set()
        self._runnables = {}
        self._handlers = {}
        self._queue = SimpleQueue()
        self._poll_timer = Qt.QtCore.QTimer(self)
        self._poll_timer.setInterval(10)
        self._poll_timer.timeout.connect(self._drain_queue)

    def cancel_pending(self):
        self.generation += 1
        self._pending.clear()
        return self.generation

    def start(self, fn, *, on_done, on_error=None, on_stale=None, on_slow=None, slow_ms=100):
        self.generation += 1
        generation = self.generation
        self._pending.add(generation)
        self._handlers[generation] = (on_done, on_error, on_stale)
        if on_slow is not None:
            Qt.QtCore.QTimer.singleShot(int(slow_ms), lambda generation=generation: self._emit_slow(generation, on_slow))

        runnable = _EvaluationRunnable(generation, fn, self._queue)
        self._runnables[generation] = runnable
        self.pool.start(runnable)
        self._ensure_polling()
        return generation

    def start_prefetch(self, fn, on_done=None):
        key = ("prefetch", id(fn), len(self._runnables))
        runnable = _PrefetchRunnable(fn, self._queue, key)
        self._runnables[key] = runnable
        if on_done is not None:
            self._handlers[key] = (on_done, None, None)
        self.pool.start(runnable)
        self._ensure_polling()

    def _emit_slow(self, generation, callback):
        if generation in self._pending and generation == self.generation:
            callback()

    def _ensure_polling(self):
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _drain_queue(self):
        while not self._queue.empty():
            kind, key, value = self._queue.get()
            if kind == "prefetch_done":
                self._runnables.pop(key, None)
                on_done, _on_error, _on_stale = self._handlers.pop(key, (None, None, None))
                if on_done is not None:
                    try:
                        on_done(value)
                    except Exception as exc:
                        handle_ui_exception("prefetch callback", exc)
                continue
            if kind == "prefetch_failed":
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
        self._runnables.pop(generation, None)
        on_done, _on_error, on_stale = self._handlers.pop(generation, (None, None, None))
        self._pending.discard(generation)
        if on_done is None:
            return
        if generation != self.generation:
            if on_stale is not None:
                on_stale()
            return
        on_done(value)

    def _fail(self, generation, exc):
        self._runnables.pop(generation, None)
        _on_done, on_error, on_stale = self._handlers.pop(generation, (None, None, None))
        self._pending.discard(generation)
        if generation != self.generation:
            if on_stale is not None:
                on_stale()
            return
        if on_error is not None:
            on_error(exc)
        else:
            handle_ui_exception("background evaluation", exc)
