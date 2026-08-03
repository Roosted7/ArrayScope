"""The presentation owner thread and its routing predicate (ADR 0065 stage 1).

Two claims are worth a test here, and they are not the same claim.

The first is that routing actually **moves execution**: a routed completion's
body runs on the owner thread, not the caller's. Every test that only asserted
"the handler ran" would pass with the routing deleted, so the pin is the
thread's identity taken from inside the handler, mutation-checked against the
same event dispatched without routing.

The second is that the migration bridge holds. The session-order lock is the
only thing keeping the GUI thread and the owner from writing session state at
once, so a dispatch must wait while a GUI-side section holds it, and the wait
must be counted — the ADR's exit condition for the lock is a GUI-side
acquisition count that trends to zero, which nobody can read if it is not
recorded.

The third is the anti-convoy rule, which is a claim about *order* rather than
about correctness: holding the lock per batch is not enough, because a Python
lock hands off by race and the owner re-acquires it in the same breath it
released it. Measured on the 272-tile wgpu probe, that barging cost GUI-side
sections 50–550 ms at a stretch; with the announcement and the stand-aside in
place the same probe charges them under a millisecond apiece. The pins below
are counting ones — how many events of a queued stream the drain gets through
while a GUI section is waiting — mutation-checked against the same stream with
the stand-aside removed.

The fourth is the batch itself (ADR 0065 stage 2 slice D3b, mechanism M1). The
drain took the lock per *event* until 2026-08-03, and slice D3 measured that as
the routed fill's remaining cost — ~700 hand-offs per fill on one lock, each
cheap, 1.0-1.4 s of an 8 s fill together. A batch is bounded twice over, by
count and by elapsed dispatch, and both bounds are pinned below because either
alone can be satisfied by an implementation that starves the other side: an
unbounded count would hold the bridge for a whole completion stream, and an
unbounded clock would do the same for eight expensive bodies.

What they deliberately do not pin is the *balance* the stand-aside strikes:
one turn per announcement rather than priority until the waiter queue empties.
Both variants pass every assertion here, because two threads and one lock
cannot reproduce the many-thread contention that separates them; the arm that
gave the GUI absolute priority was refuted on the probe (owner starved 5.4 s
of an 8.4 s fill, commit cohorts 55–91 → ~10), which is a wall-clock verdict
and belongs there rather than in a test that would have to adjudicate a clock.

No clock is adjudicated: the waits below are bounded event waits and every
assertion is on a counter.
"""

from __future__ import annotations

import threading
from time import sleep

import pytest

from arrayscope.kernel import InlineWorkerBackend, Kernel, TaskSpec
from arrayscope.kernel.completions import RoutedCompletionQueue
from arrayscope.presentation.owner import (
    _EVENT_BATCH_LIMIT,
    _EVENT_BATCH_MS,
    PresentationOwner,
    montage_completion_route,
)

#: Bounds a join or a handler wait that a working owner satisfies immediately.
#: Only a wedged owner ever spends it.
OWNER_WAIT_S = 5.0

#: How long a blocked dispatch is watched before "it did not run" is believed.
#: One-sided: the positive half of that test is an unbounded-in-practice wait.
BLOCKED_OBSERVATION_S = 0.2

#: A stream long enough that "the owner ran all of it" and "the owner was
#: interrupted" are different counts, short enough that the interrupted arm
#: does not sit on the stand-aside's ceiling for long.
#:
#: Denominated in *batches* since M1, and that is not cosmetic: the stand-aside
#: happens once per batch, so a stream of one or two batches leaves too few
#: stand-asides for `BLOCKED_OBSERVATION_S` to distinguish "held off" from
#: "slow", and the ceiling pin below would flake rather than fail.
YIELD_STREAM_EVENTS = 10 * _EVENT_BATCH_LIMIT

#: Per-event cost for the arm that needs a real interval during which a GUI
#: section can be waiting. Nothing is timed against it.
YIELD_EVENT_SECONDS = 0.005


@pytest.fixture
def owned_kernel():
    """Build a kernel whose montage-scoped completions reach a live owner.

    The wiring order is the production one: the owner needs the kernel, the
    kernel needs a queue, the queue needs the owner, and the cycle is broken by
    replacing ``kernel.completions`` before any work is submitted.
    """

    built = []

    def build(*, error_hook=None):
        kernel = Kernel(InlineWorkerBackend())
        owner = PresentationOwner(kernel, error_hook=error_hook)
        kernel.completions = RoutedCompletionQueue(montage_completion_route, owner)
        owner.start()
        built.append((kernel, owner))
        return kernel, owner

    try:
        yield build
    finally:
        for kernel, owner in built:
            owner.shutdown(timeout=OWNER_WAIT_S)
            kernel.shutdown(timeout=OWNER_WAIT_S)


class _ThreadRecorder:
    """Records which thread ran a completion body, and says when it has."""

    def __init__(self) -> None:
        self.ran = threading.Event()
        self.thread_name = ""
        self.thread_ident = 0
        self.value = None

    def __call__(self, value=None) -> None:
        self.thread_name = threading.current_thread().name
        self.thread_ident = threading.get_ident()
        self.value = value
        self.ran.set()


# --------------------------------------------------------------- the predicate


@pytest.mark.parametrize(
    ("scope", "routed"),
    [
        ("montage-stage-plan:('vol', 0)", True),
        ("montage-stage-plan:7", True),
        # Per-tile admissions and the frame-admission continuation are the
        # larger prize (7.1 s of the 124 s trace) and stay on the bridge: five
        # attempts have routed them and five have censored the 272-tile probe.
        # Slice D3b is the last of them and the one that retires the standing
        # diagnosis — batching the drain's lock sections cut owner acquisitions
        # 700 -> ~290 and left the GUI's wait unchanged, so the cost is the
        # owner's occupancy of the one bridge lock, not the hand-offs.
        ("frame:7", False),
        ("frame:7:tile:3", False),
        # Prefetch bodies warm wgpu residency directly: stage 3, not stage 1.
        ("montage:('vol', 0):prefetch", False),
        ("montage:('vol', 0):histogram", False),
        ("montage:('vol', 0):wgpu-histogram", False),
        ("montage:semantic-level-evidence", False),
        ("montage:7", False),
        ("frame", False),
        ("preframe:7", False),
        ("default", False),
        ("slice-prefetch", False),
    ],
)
def test_the_route_claims_only_stage_plan_scopes(scope, routed):
    event = _stub_event(scope)

    assert montage_completion_route(event) is routed


@pytest.mark.parametrize("scope", [None, 7, ("montage-stage-plan:7",)])
def test_a_scope_that_is_not_a_string_never_routes(scope):
    assert montage_completion_route(_stub_event(scope)) is False


def _stub_event(scope):
    """An event carrying just the field the predicate is allowed to read."""

    class _Spec:
        pass

    spec = _Spec()
    if scope is not None:
        spec.scope = scope

    class _Event:
        pass

    event = _Event()
    event.spec = spec
    return event


# ------------------------------------------------------------------ the thread


def test_a_routed_completion_body_runs_on_the_owner_thread(owned_kernel):
    kernel, owner = owned_kernel()
    recorder = _ThreadRecorder()

    kernel.submit(
        TaskSpec(key="tile-1", fn=lambda: 7, scope="montage-stage-plan:1"), on_done=recorder
    )

    assert recorder.ran.wait(OWNER_WAIT_S)
    assert recorder.thread_name == "presentation-owner"
    assert recorder.thread_ident != threading.get_ident()
    assert recorder.value == 7
    assert owner.events_dispatched == 1


def test_an_unrouted_completion_body_still_runs_on_its_drainer(owned_kernel):
    # The mutation arm of the test above: same kernel, same handler, a scope
    # the predicate declines. If routing were a no-op the two would agree.
    kernel, owner = owned_kernel()
    recorder = _ThreadRecorder()

    kernel.submit(
        TaskSpec(key="hist-1", fn=lambda: 7, scope="montage:('vol', 0):histogram"),
        on_done=recorder,
    )
    event = kernel.completions.pop()
    assert event is not None
    kernel.dispatch_event(event)

    assert recorder.ran.is_set()
    assert recorder.thread_ident == threading.get_ident()
    assert owner.events_dispatched == 0


def test_a_dispatch_waits_for_a_gui_side_section_and_counts_the_wait(owned_kernel):
    kernel, owner = owned_kernel()
    recorder = _ThreadRecorder()

    with owner.timed_order_lock("gui"):
        kernel.submit(
            TaskSpec(key="tile-2", fn=lambda: 1, scope="montage-stage-plan:2"), on_done=recorder
        )
        assert not recorder.ran.wait(BLOCKED_OBSERVATION_S)
        assert owner.events_dispatched == 0

    assert recorder.ran.wait(OWNER_WAIT_S)
    assert owner.events_dispatched == 1
    assert owner.gui_lock_acquisitions == 1
    assert owner.owner_lock_acquisitions == 1
    assert owner.owner_lock_wait_ms_max > 0.0


def test_the_order_lock_admits_the_gui_side_reentrantly(owned_kernel):
    _kernel, owner = owned_kernel()

    with owner.timed_order_lock("gui"), owner.timed_order_lock("gui"):
        assert owner.gui_lock_acquisitions == 2


def test_an_unknown_lock_side_is_refused_rather_than_miscounted(owned_kernel):
    _kernel, owner = owned_kernel()

    with pytest.raises(ValueError, match="order-lock side"), owner.timed_order_lock("worker"):
        pass

    assert owner.gui_lock_acquisitions == 0
    assert owner.owner_lock_acquisitions == 0


# ----------------------------------------------------------- the stand-aside


def test_a_gui_section_takes_its_turn_mid_stream_and_the_drain_carries_on():
    kernel, owner = _queued_stream(YIELD_EVENT_SECONDS)
    try:
        owner.start()
        owner.wake()
        # Queue up only once the owner is already mid-stream: a section that
        # arrives before the first dispatch is admitted by an idle lock and
        # says nothing about hand-off order.
        assert kernel.dispatching.wait(OWNER_WAIT_S)

        with owner.timed_order_lock("gui"):
            dispatched_at_entry = owner.events_dispatched

        # A turn was given up, and it was taken before the stream ran out.
        assert dispatched_at_entry < YIELD_STREAM_EVENTS
        assert owner.owner_yields_to_gui > 0
        # And giving it up is a pause, not a handover: the drain finishes.
        assert kernel.drained.wait(OWNER_WAIT_S)
        assert owner.events_dispatched == YIELD_STREAM_EVENTS
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)


def test_a_turn_nobody_takes_holds_the_drain_to_its_ceiling_then_releases_it():
    # The announcement is raised directly, and nothing ever comes to take the
    # turn. A real waiter takes it at the very next hand-off, which is a
    # difference of one event; an unclaimed announcement is what separates the
    # two arms of this pin by the whole stream, and it is also the case the
    # ceiling exists for — the drain must not be parked by a section that
    # never arrives.
    kernel, owner = _queued_stream(0.0)
    try:
        owner._gui_waiting = 1
        owner.start()
        owner.wake()
        assert kernel.dispatching.wait(OWNER_WAIT_S)

        assert not kernel.drained.wait(BLOCKED_OBSERVATION_S)
        held_off = owner.events_dispatched
        assert held_off < YIELD_STREAM_EVENTS
        assert owner.owner_yields_to_gui > 0

        owner._gui_waiting = 0
        assert kernel.drained.wait(OWNER_WAIT_S)
        assert owner.events_dispatched == YIELD_STREAM_EVENTS
    finally:
        owner._gui_waiting = 0
        owner.shutdown(timeout=OWNER_WAIT_S)


def test_without_the_stand_aside_the_drain_runs_the_whole_stream_past_the_flag(monkeypatch):
    # The mutation arm of the pin above: the announcement still happens, the
    # drain just stops reading it, and the entire stream goes through while a
    # GUI section is waiting.
    monkeypatch.setattr(PresentationOwner, "_yield_to_gui", lambda self: None)
    kernel, owner = _queued_stream(0.0)
    try:
        owner._gui_waiting = 1
        owner.start()
        owner.wake()

        assert kernel.drained.wait(OWNER_WAIT_S)
        assert owner.events_dispatched == YIELD_STREAM_EVENTS
        assert owner.owner_yields_to_gui == 0
    finally:
        owner._gui_waiting = 0
        owner.shutdown(timeout=OWNER_WAIT_S)


class _CountingKernel:
    """A kernel stand-in that says when it started and when it ran out.

    The optional per-event sleep is what makes "a GUI section was waiting
    while the owner worked" an askable question at all: a body that returns
    instantly leaves no interval to wait during. Nothing here is timed — every
    assertion above reads an event counter.
    """

    def __init__(self, seconds: float, expected: int) -> None:
        self._seconds = seconds
        self._expected = int(expected)
        self._seen = 0
        self.dispatching = threading.Event()
        self.drained = threading.Event()

    def dispatch_event(self, event) -> None:
        self.dispatching.set()
        if self._seconds > 0.0:
            sleep(self._seconds)
        self._seen += 1
        if self._seen >= self._expected:
            self.drained.set()


def _queued_stream(seconds: float):
    """An unstarted owner with a full queue of same-cost events."""

    kernel = _CountingKernel(seconds, YIELD_STREAM_EVENTS)
    owner = PresentationOwner(kernel)
    for _ in range(YIELD_STREAM_EVENTS):
        owner.append(_dispatchable(lambda: None))
    return kernel, owner


# ---------------------------------------------------------- the bounded batch


def test_one_lock_section_covers_a_whole_batch_of_events():
    """M1's claim, counted: acquisitions are per batch, not per event.

    The stream is several batches long so the count distinguishes the two
    designs by a factor rather than by one: per-event locking makes it
    ``YIELD_STREAM_EVENTS`` acquisitions, which is the mutation this fails
    against, and the batched drain makes it one per ``_EVENT_BATCH_LIMIT``.

    ``owner_lock_acquisitions`` is asserted alongside because it is the number
    the migration reads, and a batch counter that did not move it would be
    bookkeeping rather than a mechanism.
    """

    kernel, owner = _queued_stream(0.0)
    try:
        owner.start()
        owner.wake()
        assert kernel.drained.wait(OWNER_WAIT_S)
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)

    snapshot = owner.snapshot()
    assert snapshot["events_dispatched"] == YIELD_STREAM_EVENTS
    assert snapshot["owner_events_per_batch_max"] == _EVENT_BATCH_LIMIT
    expected_batches = YIELD_STREAM_EVENTS // _EVENT_BATCH_LIMIT
    assert snapshot["owner_batch_acquisitions"] == expected_batches
    assert snapshot["owner_lock_acquisitions"] == expected_batches


def test_a_batch_closes_on_its_time_budget_before_its_event_count():
    """The second bound, which the count bound cannot stand in for.

    Eight bodies that each cost more than the whole budget would hold the
    bridge for eight times a frame if only the count closed a batch. Each event
    here costs more than ``_EVENT_BATCH_MS`` on its own, so every batch must be
    one event long — and the events are not free, which is the only way to ask
    this question at all.

    No clock is adjudicated: the assertion is on the batch counter, and the
    sleep is what the *product* would have to be slower than for the bound to
    be wrong.
    """

    seconds = (_EVENT_BATCH_MS / 1000.0) * 2.0
    kernel = _CountingKernel(seconds, 3)
    owner = PresentationOwner(kernel)
    for _ in range(3):
        owner.append(_dispatchable(lambda: None))
    try:
        owner.start()
        owner.wake()
        assert kernel.drained.wait(OWNER_WAIT_S)
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)

    snapshot = owner.snapshot()
    assert snapshot["events_dispatched"] == 3
    assert snapshot["owner_events_per_batch_max"] == 1
    assert snapshot["owner_batch_acquisitions"] == 3


# ------------------------------------------------------------ the batch tail


def test_a_deferred_callback_runs_once_at_the_tail_of_the_batch():
    """M2's barrier, expressed the only way it is observable: as an order.

    The gate post is the caller. Deferred from inside the first event of a
    stream, it must run after the *last* one — that is what turns a commit
    cohort into "everything this batch admitted" instead of "whatever arrived
    during the previous commit". A test that only asserted the callback ran
    would pass with the deferral deleted, so the whole claim is the position.
    """

    owner = PresentationOwner(_NullKernel())
    order: list[str] = []
    done = threading.Event()

    def admit(index: int) -> None:
        order.append(f"event-{index}")
        if index == 0:
            assert owner.defer_to_batch_tail(lambda: order.append("tail")) is True

    for index in range(3):
        owner.append(_dispatchable(lambda index=index: admit(index)))
    owner.post(lambda: (order.append("message"), done.set()))

    owner.start()
    try:
        assert done.wait(OWNER_WAIT_S)
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)

    assert order == ["event-0", "event-1", "event-2", "tail", "message"]
    assert owner.snapshot()["batch_tail_deferrals"] == 1
    assert owner.snapshot()["pending_batch_tail"] == 0


def test_a_deferral_from_another_thread_is_refused_rather_than_queued():
    """A refusal is the caller keeping its own wakeup, so it must be honest.

    Off the owner thread there is no batch to be the tail of, and a queued
    callback nobody drains is the lost wakeup the presentation gate may not
    have. Both refusal terms are checked here — the wrong thread, and the owner
    thread with nothing queued behind it — because the caller posts inline on
    exactly this answer.
    """

    owner = PresentationOwner(_NullKernel())
    owner.append(_dispatchable(lambda: None))

    assert owner.defer_to_batch_tail(lambda: None) is False

    ran = threading.Event()
    answers: list[bool] = []

    def only_event() -> None:
        # Inside a batch, but the deque behind it is empty: the tail is now.
        answers.append(owner.defer_to_batch_tail(lambda: None))
        ran.set()

    owner._events.clear()
    owner.append(_dispatchable(only_event))
    owner.start()
    owner.wake()
    try:
        assert ran.wait(OWNER_WAIT_S)
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)

    assert answers == [False]
    assert owner.snapshot()["batch_tail_deferrals"] == 0


# --------------------------------------------------------------- containment


def test_a_raising_body_is_counted_and_the_thread_keeps_draining(owned_kernel):
    hooked: list[BaseException] = []
    kernel, owner = owned_kernel(error_hook=hooked.append)
    survivor = _ThreadRecorder()

    def explode(_value) -> None:
        raise RuntimeError("body went bad")

    kernel.submit(
        TaskSpec(key="tile-3", fn=lambda: 1, scope="montage-stage-plan:3"), on_done=explode
    )
    kernel.submit(
        TaskSpec(key="tile-4", fn=lambda: 2, scope="montage-stage-plan:3"), on_done=survivor
    )

    assert survivor.ran.wait(OWNER_WAIT_S)
    assert survivor.thread_name == "presentation-owner"
    assert owner.errors == 1
    assert [type(exc).__name__ for exc in hooked] == ["RuntimeError"]
    assert "body went bad" in owner.last_error
    assert owner.events_dispatched == 2


def test_a_raising_error_hook_does_not_kill_the_owner(owned_kernel):
    def hook(_exc) -> None:
        raise ValueError("the hook is broken too")

    kernel, owner = owned_kernel(error_hook=hook)
    survivor = _ThreadRecorder()

    def explode(_value) -> None:
        raise RuntimeError("body went bad")

    kernel.submit(
        TaskSpec(key="tile-5", fn=lambda: 1, scope="montage-stage-plan:5"), on_done=explode
    )
    kernel.submit(
        TaskSpec(key="tile-6", fn=lambda: 2, scope="montage-stage-plan:5"), on_done=survivor
    )

    assert survivor.ran.wait(OWNER_WAIT_S)
    assert owner.errors == 1


# ---------------------------------------------------------------- lifecycle


def test_start_and_shutdown_are_both_idempotent():
    owner = PresentationOwner(_NullKernel())
    try:
        owner.start()
        owner.start()
        assert _owner_threads() == 1
        assert owner.snapshot()["running"] is True

        assert owner.shutdown(timeout=OWNER_WAIT_S) is True
        assert owner.shutdown(timeout=OWNER_WAIT_S) is True
        assert owner.snapshot()["running"] is False
        assert _owner_threads() == 0

        owner.start()
        recorder = _ThreadRecorder()
        owner.append(_dispatchable(recorder))
        owner.wake()
        assert recorder.ran.wait(OWNER_WAIT_S)
        assert owner.events_dispatched == 1
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)


def test_shutdown_before_start_joins_nothing():
    owner = PresentationOwner(_NullKernel())

    assert owner.shutdown(timeout=OWNER_WAIT_S) is True
    assert owner.snapshot()["running"] is False


def test_events_still_queued_at_shutdown_are_dropped_not_run():
    # A teardown must not run montage bodies against a window that is already
    # unwinding, so the pending queue is counted out rather than drained.
    owner = PresentationOwner(_NullKernel())
    recorder = _ThreadRecorder()
    owner.append(_dispatchable(recorder))

    assert owner.shutdown(timeout=OWNER_WAIT_S) is True

    assert not recorder.ran.is_set()
    assert owner.snapshot()["events_dropped_at_stop"] == 1
    assert owner.snapshot()["pending_events"] == 0


# -------------------------------------------------------------- posted work


def test_a_posted_callback_runs_on_the_owner_under_the_order_lock(owned_kernel):
    """ADR 0065 stage 2 slice B: ``post`` is the ladder retarget's entry.

    Two claims in one, because either alone would pass with the other broken:
    the callback runs on the *owner* thread (not the poster's), and it runs
    inside an owner-side lock section — which is what makes a posted retarget
    serialize against GUI session sections the same way a dispatched
    completion does.
    """

    _kernel, owner = owned_kernel()
    recorder = _ThreadRecorder()
    held: list[bool] = []

    def body() -> None:
        # `acquire(blocking=False)` from the owner's own thread would succeed
        # on a re-entrant lock regardless, so the claim is checked from a
        # foreign thread: it must NOT be able to take the lock right now.
        probe = threading.Thread(target=lambda: held.append(owner.order_lock.acquire(False)))
        probe.start()
        probe.join(OWNER_WAIT_S)
        recorder()

    owner.post(body)

    assert recorder.ran.wait(OWNER_WAIT_S)
    assert recorder.thread_name == "presentation-owner"
    assert recorder.thread_ident != threading.get_ident()
    assert held == [False]
    snapshot = owner.snapshot()
    assert snapshot["messages_posted"] == 1
    assert snapshot["messages_dispatched"] == 1
    assert snapshot["pending_messages"] == 0
    assert snapshot["owner_lock_acquisitions"] >= 1


def test_a_raising_posted_callback_is_counted_and_the_thread_keeps_draining(owned_kernel):
    failures: list[BaseException] = []
    _kernel, owner = owned_kernel(error_hook=failures.append)
    recorder = _ThreadRecorder()

    def boom() -> None:
        raise RuntimeError("posted body failed")

    owner.post(boom)
    owner.post(recorder)

    assert recorder.ran.wait(OWNER_WAIT_S)
    snapshot = owner.snapshot()
    assert snapshot["errors"] == 1
    assert snapshot["messages_dispatched"] == 2
    assert [type(exc).__name__ for exc in failures] == ["RuntimeError"]


def test_posted_messages_drain_after_the_events_of_the_batch():
    """The barrier the replan gate needs, expressed as an order.

    Not a preference: a message interleaved between events fires one full
    ladder retarget per completion, which is the O(N^2) storm
    ``request_montage_replan`` exists to prevent. Draining messages once the
    event deque is empty is stage 2 decision 1's "tail of the drain batch",
    and it is only observable as an order — hence the sequence, not a count.
    """

    owner = PresentationOwner(_NullKernel())
    order: list[str] = []
    done = threading.Event()

    for index in range(3):
        owner.append(_dispatchable(lambda index=index: order.append(f"event-{index}")))
    owner.post(lambda: order.append("message"))
    owner.post(lambda: (order.append("message"), done.set()))

    owner.start()
    try:
        assert done.wait(OWNER_WAIT_S)
    finally:
        owner.shutdown(timeout=OWNER_WAIT_S)

    assert order == ["event-0", "event-1", "event-2", "message", "message"]


def test_posted_messages_still_queued_at_shutdown_are_dropped_not_run():
    owner = PresentationOwner(_NullKernel())
    recorder = _ThreadRecorder()
    owner.post(recorder)

    assert owner.shutdown(timeout=OWNER_WAIT_S) is True

    assert not recorder.ran.is_set()
    snapshot = owner.snapshot()
    assert snapshot["messages_dropped_at_stop"] == 1
    assert snapshot["pending_messages"] == 0
    # And a stopped owner says so, so a caller runs its work inline instead of
    # handing it to a queue nothing drains.
    assert owner.accepts_posts() is False


def _owner_threads() -> int:
    return sum(1 for thread in threading.enumerate() if thread.name == "presentation-owner")


def test_the_snapshot_carries_every_counter(owned_kernel):
    kernel, owner = owned_kernel()
    recorder = _ThreadRecorder()

    kernel.submit(
        TaskSpec(key=("tile", 9), fn=lambda: 1, scope="montage-stage-plan:9"), on_done=recorder
    )
    assert recorder.ran.wait(OWNER_WAIT_S)

    snapshot = owner.snapshot()
    assert set(snapshot) == {
        "running",
        "pending_events",
        "pending_messages",
        "pending_batch_tail",
        "owner_batch_acquisitions",
        "owner_events_per_batch_max",
        "batch_tail_deferrals",
        "batch_tail_dropped_at_stop",
        "events_dispatched",
        "events_dropped_at_stop",
        "messages_posted",
        "messages_dispatched",
        "messages_dropped_at_stop",
        "worst_event_ms",
        "worst_event_key_head",
        "owner_lock_acquisitions",
        "owner_lock_wait_ms_total",
        "owner_lock_wait_ms_max",
        "gui_lock_acquisitions",
        "gui_lock_wait_ms_total",
        "gui_lock_wait_ms_max",
        "owner_yields_to_gui",
        "gui_waiting",
        "errors",
        "last_error",
    }
    assert snapshot["events_dispatched"] == 1
    assert snapshot["worst_event_key_head"] == "'tile'"
    assert snapshot["errors"] == 0


class _NullKernel:
    """Enough kernel for the lifecycle: one dispatch entry point."""

    def dispatch_event(self, event) -> None:
        event.body()


def _dispatchable(body):
    """A routed event stand-in that `_NullKernel` can dispatch."""

    class _Event:
        pass

    class _Spec:
        key = "lifecycle"
        scope = "montage-stage-plan:0"

    event = _Event()
    event.spec = _Spec()
    event.body = body
    return event
