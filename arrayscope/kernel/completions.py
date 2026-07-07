"""Completion delivery: the single fan-in between workers and the GUI.

Workers never call handlers. They append `CompletionEvent`s to a
`CompletionQueue` and fire its notify hook. The consumer (QtKernelBridge in
the app, a plain loop in tests) drains events in bounded batches and invokes
handlers on its own thread.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from arrayscope.kernel.task import TaskOutcome, TaskSpec


@dataclass(frozen=True)
class CompletionEvent:
    spec: TaskSpec
    outcome: TaskOutcome
    value: Any = None
    error: BaseException | None = None
    reason: str = ""


@dataclass
class DrainBudget:
    """Bounded drain: item, byte, and elapsed-time limits.

    Mirrors the GUI-thread contract (interactive ≤ 4 ms, idle ≤ 8 ms). The Qt
    bridge converts this into a `GuiCallbackBudget` observation for the
    resource governor; the kernel itself stays Qt- and governor-free.
    """

    max_items: int = 8
    max_ms: float = 8.0
    max_bytes: int = 0  # 0 = unlimited

    def start(self) -> "_DrainState":
        return _DrainState(self)


class _DrainState:
    __slots__ = ("budget", "items", "bytes", "_t0")

    def __init__(self, budget: DrainBudget) -> None:
        self.budget = budget
        self.items = 0
        self.bytes = 0
        self._t0 = perf_counter()

    def record(self, *, item_bytes: int = 0) -> None:
        self.items += 1
        self.bytes += max(0, int(item_bytes))

    @property
    def elapsed_ms(self) -> float:
        return (perf_counter() - self._t0) * 1000.0

    def exhausted(self) -> bool:
        if self.items >= max(1, int(self.budget.max_items)):
            return True
        if self.budget.max_bytes and self.bytes >= self.budget.max_bytes:
            return True
        return self.elapsed_ms >= float(self.budget.max_ms)


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
            try:
                notify()
            except Exception:
                pass

    def pop(self) -> CompletionEvent | None:
        with self._lock:
            return self._events.popleft() if self._events else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def empty(self) -> bool:
        return len(self) == 0


__all__ = ["CompletionEvent", "CompletionQueue", "DrainBudget"]
