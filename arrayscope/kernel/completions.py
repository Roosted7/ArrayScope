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


class RoutedCompletionQueue(CompletionQueue):
    """A completion queue with a second consumer, chosen per event.

    ADR 0065 stage 1: montage-scoped completion *bodies* stop running on the
    GUI thread. The split is made here, at the one fan-in, because this is the
    only place that sees every event before a consumer does.

    ``put`` runs inside ``Kernel._deliver_locked`` — under the kernel lock, on
    a worker thread. That is the whole constraint on this class:

    - the predicate must be a pure, cheap test of already-computed fields (the
      shipped one is two ``str.startswith`` calls) — never a session lookup,
      never anything that allocates per event;
    - routing takes no lock beyond the sink's own ``deque.append``, which is
      atomic. Acquiring a session or presentation lock here would invert ADR
      0065's ordering (order lock → kernel lock) and deadlock the pool.

    The sink is duck-typed — ``append(event)`` then ``wake()`` — so the owner
    thread is not an import dependency of ``arrayscope.kernel``, which stays
    montage-ignorant and numpy-free. The scope predicate lives with the owner.

    ``wake`` inherits the base class's exception discipline: a dying consumer
    must not corrupt workers, and a swallowed wake still leaves the event in
    the sink for the consumer's next pass. ``pop`` deliberately drains only the
    base deque: routed events are the sink's, and a fallback drain must not
    reach past their owner and run them on the GUI thread after all.

    Wiring note (the cycle is real): the owner needs the kernel, the kernel
    needs its queue, and the queue needs the owner. Break it after
    construction — ``kernel.completions = RoutedCompletionQueue(...)`` — while
    no worker is running, rather than by making any of the three late-bound.
    """

    def __init__(
        self,
        route_predicate: Callable[[CompletionEvent], bool],
        route_sink: Any,
        notify: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(notify)
        self._route_predicate = route_predicate
        self._route_sink = route_sink

    def put(self, event: CompletionEvent) -> None:
        if self._route_predicate(event):
            self._route_sink.append(event)
            with contextlib.suppress(Exception):
                self._route_sink.wake()
            return
        super().put(event)


__all__ = ["CompletionEvent", "CompletionQueue", "RoutedCompletionQueue"]
