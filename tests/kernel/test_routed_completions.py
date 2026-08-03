"""RoutedCompletionQueue: the fan-in's second consumer.

Pins the split ADR 0065 stage 1 makes at the one place that sees every
completion: an event the predicate claims goes to the sink and *only* the
sink, everything else keeps the base queue's path unchanged, and a sink whose
wake is dying cannot corrupt a worker — the same discipline the base class
already applies to ``notify``.

The predicate used here is a local stub, not the montage one: this queue is in
``arrayscope.kernel``, which stays montage-ignorant, and the real predicate is
adjudicated beside its owner in ``tests/presentation``.
"""

from __future__ import annotations

from types import SimpleNamespace

from arrayscope.kernel.completions import (
    CompletionEvent,
    RoutedCompletionQueue,
)
from arrayscope.kernel.task import Lane, TaskOutcome, TaskSpec


def _event(scope: str, key: str = "k") -> CompletionEvent:
    spec = TaskSpec(key=key, fn=lambda: None, lane=Lane.VISIBLE_MATERIALIZATION, scope=scope)
    return CompletionEvent(spec=spec, outcome=TaskOutcome.COMPLETED, value=key)


class _Sink:
    """The route-sink duck type: append then wake, nothing else."""

    def __init__(self, wake_error: Exception | None = None):
        self.events: list[CompletionEvent] = []
        self.wakes = 0
        self._wake_error = wake_error

    def append(self, event: CompletionEvent) -> None:
        self.events.append(event)

    def wake(self) -> None:
        self.wakes += 1
        if self._wake_error is not None:
            raise self._wake_error


def _routes_owned(event: CompletionEvent) -> bool:
    return event.spec.scope.startswith("owned:")


def test_a_routed_event_reaches_the_sink_and_never_the_base_deque():
    sink = _Sink()
    notified: list[int] = []
    queue = RoutedCompletionQueue(_routes_owned, sink, notify=lambda: notified.append(1))

    queue.put(_event("owned:1"))

    assert [event.value for event in sink.events] == ["k"]
    assert sink.wakes == 1
    assert len(queue) == 0
    assert queue.pop() is None
    assert notified == []


def test_an_unrouted_event_keeps_the_base_path_and_its_notify():
    sink = _Sink()
    notified: list[int] = []
    queue = RoutedCompletionQueue(_routes_owned, sink, notify=lambda: notified.append(1))

    queue.put(_event("other:1"))

    assert sink.events == []
    assert sink.wakes == 0
    assert len(queue) == 1
    assert queue.pop().value == "k"
    assert notified == [1]


def test_a_dying_sink_wake_does_not_reach_the_worker():
    # `put` runs under the kernel lock on a worker thread: an exception here
    # would abandon the delivery mid-lock. The event is already in the sink,
    # so the next wake finds it.
    sink = _Sink(wake_error=RuntimeError("consumer is gone"))
    queue = RoutedCompletionQueue(_routes_owned, sink)

    queue.put(_event("owned:1"))

    assert len(sink.events) == 1
    assert sink.wakes == 1


def test_each_path_keeps_its_own_order_while_the_two_interleave():
    sink = _Sink()
    queue = RoutedCompletionQueue(_routes_owned, sink)

    for index in range(6):
        scope = "owned:x" if index % 2 == 0 else "other:x"
        queue.put(_event(scope, key=f"k{index}"))

    assert [event.value for event in sink.events] == ["k0", "k2", "k4"]
    drained = []
    while (event := queue.pop()) is not None:
        drained.append(event.value)
    assert drained == ["k1", "k3", "k5"]


def test_the_predicate_reads_the_event_the_kernel_delivers():
    # `Kernel._deliver_locked` builds the event and stamps `_record` on it; the
    # predicate sees that object, not a copy, so it may read only `spec`.
    seen: list[object] = []

    def record(event) -> bool:
        seen.append(event)
        return False

    queue = RoutedCompletionQueue(record, _Sink())

    event = _event("other:1")
    object.__setattr__(event, "_record", SimpleNamespace(seq=7))
    queue.put(event)

    assert seen == [event]
    assert queue.pop() is event
