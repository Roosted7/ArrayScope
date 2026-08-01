"""Completion delivery: the single fan-in between workers and the GUI.

Workers never call handlers. They append `CompletionEvent`s to a
`CompletionQueue` and fire its notify hook. The consumer (QtKernelBridge in
the app, a plain loop in tests) drains events in bounded batches and invokes
handlers on its own thread.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from arrayscope.kernel.task import TaskOutcome, TaskSpec


@dataclass(frozen=True)
class CompletionEvent:
    spec: TaskSpec
    outcome: TaskOutcome
    value: Any = None
    error: BaseException | None = None
    reason: str = ""


class CompletionQueue:
    """Thread-safe FIFO of completion events with a wake hook.

    ``notify`` is called (from worker threads) after an event is appended
    while the queue was previously empty-or-not — it must be cheap and
    re-entrant-safe; the Qt bridge emits a queued signal from it. Exceptions
    from ``notify`` are swallowed: a dying consumer must not corrupt workers
    (the events remain queued for the fallback drain).
    """

    def __init__(self, notify: Callable[[], None] | None = None) -> None:
        self._events: deque[CompletionEvent] = deque()
        self._lock = threading.Lock()
        self._notify = notify

    def set_notify(self, notify: Callable[[], None] | None) -> None:
        self._notify = notify

    def put(self, event: CompletionEvent) -> None:
        with self._lock:
            self._events.append(event)
        notify = self._notify
        if notify is not None:
            with contextlib.suppress(Exception):
                notify()

    def pop(self) -> CompletionEvent | None:
        with self._lock:
            return self._events.popleft() if self._events else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def empty(self) -> bool:
        return len(self) == 0


__all__ = ["CompletionEvent", "CompletionQueue"]
