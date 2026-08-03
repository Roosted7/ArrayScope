"""The presentation owner thread: montage presentation state's off-GUI writer.

ADR 0065 stage 1, the infrastructure half. The 2026-08-02 trace put 14.8 s of
the session's 124 s into completion *bodies* run on the GUI thread inside the
kernel bridge drain — per-tile admissions and `montage-stage-plan` bodies at
p50 13.7 ms, over half a frame in 199 of 269 observations. Those bodies are
pure Python bookkeeping over session state; nothing in them is Qt. They run
here instead, on one dedicated thread, so the GUI thread's montage work can
shrink to validate → submit → present.

Stage 1 ships that machinery with the stage-plan family routed and the
per-tile admissions left on the bridge: while the migration bridge is a single
lock, routing a 500-event admission stream serializes it against GUI sections
that hold the lock for tens of milliseconds each, and the measured result was
a censored fill. :func:`montage_completion_route` carries the numbers and the
stage-2 condition for widening it.

What this module owns:

- the thread, and the only loop that calls ``kernel.dispatch_event`` off the
  GUI thread — and, from stage 2 slice B, the loop that runs **posted
  operations** (:meth:`PresentationOwner.post`) as well: the ladder retarget's
  fire-and-forget callers hand it work the same way the kernel hands it
  completions;
- the **session-order lock** (:attr:`PresentationOwner.order_lock`), the ADR's
  migration bridge. Until the last GUI-side session writer becomes a posted
  message, this one ``RLock`` serializes owner pipeline sections against GUI
  session sections. It is scaffolding: the end state has no GUI-side session
  sections and ``gui_lock_acquisitions`` trends to zero, which is why it is
  counted rather than assumed;
- the routing predicate, because deciding what is montage work is presentation
  knowledge — ``arrayscope.kernel`` stays montage-ignorant.

**Lock ordering: order_lock → kernel lock, never the reverse.** The owner takes
``order_lock`` and then calls into the kernel, which takes its own lock;
``Kernel._deliver_locked`` calls ``CompletionQueue.put`` while holding the
kernel lock, so anything reachable from a ``put`` — the routing predicate, the
sink's ``append``/``wake`` — must touch no session or presentation lock at all.
Both of those are therefore lock-free here: a bare ``deque.append`` and an
``Event.set``.

Dispatch takes the lock **per event, not per batch**, and stands aside between
events for an announced GUI section (see :meth:`PresentationOwner._yield_to_gui`).
The GUI thread competes for the same lock, and a batch held end to end is
exactly the multi-frame stall the owner exists to remove. Both rules are
written for the volume stage 2 will route, not for the six stage-plan bodies a
fill produces today.
"""

from __future__ import annotations

import threading
import traceback
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter, sleep
from typing import Any

#: Same spirit as the bridge's `_diagnostic_key_identity`: enough of a key to
#: route a report to a call site, never enough to format a semantic payload.
_KEY_HEAD_LIMIT = 120

#: Stand-aside granularity between events while a GUI section waits. Short
#: enough that the GUI's wait is bounded by the lock hand-off rather than by
#: this poll, long enough that a streak of them is not a spin.
_YIELD_SLEEP_S = 0.0002

#: Ceiling on one stand-aside episode (~50 ms). A turn is given up for one
#: admission; this bounds the case where the announced section never arrives
#: to take it. The owner is a producer with its own liveness obligations —
#: capacity waiters, backlog re-arms — so an announcement that is never
#: honoured (it can only be a bug, but bugs happen while a migration bridge is
#: in the tree) must not park the drain forever.
_YIELD_MAX_CHECKS = 250

_GUI = "gui"
_OWNER = "owner"


def montage_completion_route(event: Any) -> bool:
    """Whether this completion's body belongs to the owner thread.

    Stage 1 routes **one** family: ``montage-stage-plan:`` bodies. They are the
    single worst offender in the bridge drain — p50 13.7 ms, over half a frame
    in 199 of 269 live observations, 4.4 s of the 124 s trace — and they are
    rare, about six per fill, so moving them cannot contend the session-order
    lock against anything. That combination is what makes them safe to route
    while the migration bridge is still a lock.

    **Per-tile ``frame:`` completions are deliberately not routed**, though
    they are the larger number (7.1 s of admissions in the same trace). Stage 1
    routed them at first and it *censored the 272-tile wgpu fill*: measured,
    the admission stream is ~500 events per fill and the bridge serializes
    every one of them against the GUI's own session sections, whose hold time
    is dominated by ``retarget_frame_pipeline`` at ~89 ms × 52 calls — 4.6 s,
    65% of all GUI-side hold, against 1.3 s of actual owner work. The owner
    spent 5.4-5.6 s of an 8.6 s fill waiting for the lock; commit cohorts fell
    from 55–91 tiles to about 10, the presentation gate's arm→fire went p50
    52 → 115 ms, and the fill never settled. It is also self-feeding: late
    admissions provoke replans, and retargets rose 31 → 52 against the same
    workload.

    Nothing about that is fixable from this side — no stand-aside policy makes
    an 89 ms GUI hold cheap — and it is not meant to be. ``retarget_frame_pipeline``
    *is* the R5 model rebuild ADR 0065 defers to **stage 2** ("becomes an owner
    message rather than a GUI callback at that point"). The ``frame:`` clause
    belongs with it: when stage 2 has shrunk the GUI sections, routing the
    admissions costs what the ADR predicted instead of the fill.

    Stage 2 slice B measured a second, tighter coupling in the same direction,
    and it reorders the plan. The replan gate's ``fire()`` cannot become a
    posted operation *before* this clause opens. Its coalescing has never come
    from the armed flag — that only counts requests — but from the GUI event
    loop's own occupancy: a ``singleShot(1)`` fire cannot run until the current
    20-45 ms commit ends, so every completion that landed inside it folds into
    one gate. The owner has no equivalent while its event deque holds only the
    ~6 stage-plan bodies a fill produces: ``_drain``'s tail-of-batch rule fires
    immediately, one retarget per completion, and the 272-tile probe measured
    exactly that (5, 140, 21 of 272 presented, 3/3 censored, bisected to that
    one line). Only when the per-tile admissions arrive *here* is there a batch
    whose tail is a real barrier. So the two land together, or neither does.

    **Stage 2 slice D2a tried exactly that pair and the probe refused it a
    third time**, and this time the counters say precisely which precondition
    is still missing. Widening this clause (with the ``fire()`` conversion, and
    also with it reverted — both arms measured) censored the 272-tile wgpu
    probe: 90/78/49, 186/272-late-106, 122/271/264 of 272 across three runs
    against 3/3 ``272/272 within-5s`` on the same tree with only this line
    narrowed. What did *not* happen is the stage-1 signature: commit cohorts
    stayed healthy (104, 91, 55 tiles) rather than collapsing to ~10, and the
    GUI was never starved — ``gui_lock_wait_ms_total`` was 10-37 ms for a whole
    fill and ``owner_yields_to_gui`` fired six times. The owner starved instead:
    ``owner_lock_wait_ms_total`` 1.8-1.9 s of an ~8 s fill, single waits to
    289 ms, across 631 acquisitions.

    Instrumented GUI-side lock **holds** for one such fill name the two
    remaining owners, and neither is preparation:

    ==============================  =====  ========  =======
    GUI section                     calls  total ms  max ms
    ==============================  =====  ========  =======
    ``commit_pending_session``         43    2350.2    526.2
    ``retarget_frame_pipeline``        38    2009.1    193.7
    ``_on_presentation_gate``          41    1784.7    278.1
    ==============================  =====  ========  =======

    (the gate encloses the commit, so the two nest). The control arm holds the
    same shape — 1913 ms / 1335 ms — so this is the GUI's steady state, not
    something routing caused. ``retarget_frame_pipeline`` is still a GUI-side
    section 28-38 times per fill at ~50 ms mean: slice B moved the
    *fire-and-forget* callers (``post_frame_pipeline_retarget``), and decision
    10's items 2-5 — the stage-plan tail, the index-window rebuild, the
    resize/viewport callers, the viewport-only latch — are still owed.
    ``commit_pending_session`` holds the lock across the whole commit including
    the backend submit, which needs no session ordering at all.

    So the condition to re-test this clause under is not "the prepare is
    cheaper" (slice D2a made it a computation with no lock inside, and the
    probe still refused): it is **GUI-side hold under roughly a frame**. While
    two sections hold it for 3-4 s of an 8 s fill, 630 owner acquisitions
    cannot fit between them, and no policy on this side changes that.

    Two further exclusions are deliberate rather than incidental:

    - Histogram and level-evidence completions (``montage:…:histogram``,
      ``montage:semantic-level-evidence``) write display state the GUI owns.
    - **Prefetch** completions (``montage:…:prefetch``) look like admissions
      but their body calls ``img_view.warmTiledResidency`` — a wgpu upload —
      from ``montage_prefetch._warm_prefetched_tiled_residency``. Running that
      off the GUI thread is unsafe, and its 2.0 s across the 124 s trace does
      not buy the posted-warm indirection it would need. Stage 3 work.

    Cheap by contract: this runs under the kernel lock, on a worker thread,
    once per completion.
    """

    scope = getattr(event.spec, "scope", None)
    if not isinstance(scope, str):
        return False
    return scope.startswith("montage-stage-plan:")


class PresentationOwner:
    """One daemon thread draining routed completions under the order lock.

    The thread is a daemon so a wedged owner cannot hold interpreter exit
    hostage; :meth:`shutdown` is what closes it deterministically, and tests
    rely on that join rather than on the daemon flag.

    Counters are plain attributes, written only by the owner thread (the lock
    counters also by whichever thread calls :meth:`timed_order_lock`) and read
    without any lock. Torn reads are accepted: these are diagnostics, and a
    lock around them would put GUI-thread contention into the path whose
    contention they exist to measure.
    """

    def __init__(self, kernel: Any, *, error_hook: Callable[[BaseException], None] | None = None):
        self._kernel = kernel
        self._error_hook = error_hook
        #: The ADR 0065 session-order lock. Re-entrant because a GUI-side
        #: section may call into another one; owner sections never nest.
        self.order_lock = threading.RLock()
        self._events: deque = deque()
        #: Posted operations (:meth:`post`). A second deque rather than a tag
        #: on the event deque because ``append`` runs under the kernel lock and
        #: must stay a bare ``deque.append`` — see the module docstring — and
        #: because the two are drained at different points (see :meth:`_drain`).
        self._messages: deque = deque()
        self._wake = threading.Event()
        self._stop = False
        self._thread: threading.Thread | None = None
        self._lifecycle = threading.Lock()
        self.events_dispatched = 0
        self.events_dropped_at_stop = 0
        self.messages_posted = 0
        self.messages_dispatched = 0
        self.messages_dropped_at_stop = 0
        self.worst_event_ms = 0.0
        self.worst_event_key_head = ""
        self.owner_lock_acquisitions = 0
        self.owner_lock_wait_ms_total = 0.0
        self.owner_lock_wait_ms_max = 0.0
        self.gui_lock_acquisitions = 0
        self.gui_lock_wait_ms_total = 0.0
        self.gui_lock_wait_ms_max = 0.0
        self.errors = 0
        self.last_error = ""
        #: How many GUI-side sections are currently waiting for the lock.
        #: Written GUI-side around the acquire; the drain loop reads it
        #: between events and gives up one turn for it. Paired with
        #: ``gui_lock_acquisitions``, which is how the drain knows the turn it
        #: gave up was taken. Both are advisory; see :meth:`timed_order_lock`.
        self._gui_waiting = 0
        #: Turns given up, one per stand-aside episode.
        self.owner_yields_to_gui = 0
        #: Called (from the owner thread) after a drain batch that dispatched
        #: at least one event. The bridge-side reconciliations — capacity
        #: waiters, the rate-limited governor reapply — historically piggybacked
        #: on the GUI drain that every completion used to cause; routed events
        #: no longer cause one, so the composition points this at the bridge's
        #: own ``eventsReady.emit`` (thread-safe, queued) and an empty GUI drain
        #: runs exactly those reconciliations. Measured cost when the base
        #: queue is empty: microseconds. Without this, a task parked on quota
        #: whose capacity was freed by a routed completion waits forever
        #: (observed: ``histogram_aggregate_inflight`` latched in 5 of 6
        #: builds, and a load-gated all-idle settle wedge).
        self.after_dispatch: Callable[[], None] | None = None

    # ------------------------------------------------------------ route sink

    def append(self, event: Any) -> None:
        """Take one routed event. Called under the kernel lock — lock-free."""

        self._events.append(event)

    def wake(self) -> None:
        """Signal the loop. Called under the kernel lock — lock-free."""

        self._wake.set()

    # ---------------------------------------------------------- posted work

    def post(self, callback: Callable[[], None]) -> None:
        """Queue one owner-thread operation. Callable from any thread.

        ADR 0065 stage 2 slice B. The completion deque carries work the kernel
        produced; this carries work a *caller* wants run under the owner's
        order-lock section — the ladder retarget above all. Both end up in the
        same drain, and each posted callback gets its own
        :meth:`timed_order_lock` section and its own stand-aside turn, exactly
        like a dispatched event: the alternation the fill depends on knows
        nothing about where an item came from.

        A posted callback owns its own currency re-check. The owner cannot do
        it — it has no idea which session identity the closure was built
        against — and a message that ran against a superseded session would
        reintroduce the staleness class the ADR refuses to add. Every caller in
        the tree captures ``(session_id, session.key)`` at post time and
        re-checks with ``_is_current_frame_session`` inside.

        ``deque.append`` is atomic and the wake is an ``Event.set``, so this
        takes no lock and is safe to call from a GUI section that already holds
        the order lock — posting must never be able to deadlock against the
        drain it is feeding.
        """

        self._messages.append(callback)
        self.messages_posted += 1
        self._wake.set()

    # ------------------------------------------------------------- lifecycle

    def accepts_posts(self) -> bool:
        """Whether :meth:`post` will actually be drained.

        Asked before posting, never after: a callback handed to a stopped
        owner is dropped, and for the retarget that means a lost wakeup rather
        than a late one. Callers run it inline instead.
        """

        thread = self._thread
        return not self._stop and thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            thread = threading.Thread(target=self._run, name="presentation-owner", daemon=True)
            self._thread = thread
            thread.start()

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop the loop and join. Returns whether the thread is gone.

        Idempotent, and callable before :meth:`start`. Events still queued are
        dropped rather than dispatched: a teardown must not run montage bodies
        against a window that is already unwinding.
        """

        with self._lifecycle:
            self._stop = True
            self._wake.set()
            thread = self._thread
            if thread is not None:
                thread.join(float(timeout))
                if thread.is_alive():
                    return False
                self._thread = None
            # Safe here and only here: no thread is left to pop concurrently.
            self._drop_pending()
            return True

    # ------------------------------------------------------------- the loop

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait()
            self._wake.clear()
            before = self.events_dispatched + self.messages_dispatched
            self._drain()
            hook = self.after_dispatch
            if (
                hook is not None
                and (self.events_dispatched + self.messages_dispatched) > before
                and not self._stop
            ):
                try:
                    hook()
                except Exception:
                    self.last_error = traceback.format_exc()
        self._drop_pending()

    def _drain(self) -> None:
        """Events first, posted messages at the tail of the batch.

        Not a priority statement — a coalescing one, and the reason is the
        replan gate. Its arm/fire pair is coalesced by a flag that ``fire()``
        clears on entry, so the number of retargets a completion streak causes
        is exactly the number of times ``fire()`` runs between arms. The
        GUI-thread predecessor got its coalescing from Qt: the 1 ms timer could
        not fire until the event loop turned, and every admission that landed
        in that window folded into the one armed gate. Interleaving posted
        messages between events reproduces the O(N²) replan storm
        ``request_montage_replan`` exists to prevent — one full ladder retarget
        per admission. Draining them once the event deque is empty is the same
        barrier the timer was, expressed in the owner's own units, and it is
        where stage 2's decision 1 puts transaction preparation as well.

        A message dispatched at the tail can post another message or admit work
        that produces events; the loop re-checks both deques, so the batch ends
        only when there is genuinely nothing left. The starvation this trades
        for is a message behind an event stream that never drains — bounded in
        practice by the fill being finite, and visible as ``pending_messages``
        in :meth:`snapshot` rather than assumed away.
        """

        events = self._events
        messages = self._messages
        while not self._stop:
            try:
                event = events.popleft()
            except IndexError:
                event = None
            if event is not None:
                self._dispatch(event)
                self._yield_to_gui()
                continue
            try:
                message = messages.popleft()
            except IndexError:
                return
            self._dispatch_message(message)
            self._yield_to_gui()

    def _yield_to_gui(self) -> None:
        """Give up **one turn** to a waiting GUI section, between events.

        The receiving half of the announcement made in
        :meth:`timed_order_lock`, and the anti-convoy mechanism: re-acquiring
        the order lock immediately after releasing it wins the race against a
        thread that has been waiting for it, because a Python lock hands off
        by race and not by queue order. Called **between** events, never while
        the lock is held — yielding under the lock would be the convoy with
        extra latency.

        The exit condition is one admission, not an empty waiter queue. That
        distinction is the whole design and it was measured, not assumed: the
        first version stood aside until ``_gui_waiting`` fell to zero, and on
        the 272-tile probe that inverted the starvation instead of removing
        it. GUI waits collapsed (mean 0.94 ms against the p50 55 ms / max
        550 ms it was built to fix) and the *owner* then spent 5.4 s of an
        8.4 s fill waiting for the lock, so admissions arrived in a trickle:
        commit cohorts fell from 55–91 tiles to about 10, ``rung_prepare``
        went 669 → 1181, and the fill still censored. What the single-threaded
        predecessor had was neither priority — it was **alternation**, the
        admission and the commit taking turns on one thread. So the owner
        stands aside until one GUI section has been admitted and then takes
        its turn back, even if more sections are queued behind that one. A
        section announced during the next event triggers the next stand-aside,
        which is one-for-one alternation under contention and full owner
        throughput whenever the GUI is idle.

        Both terms are read without a lock, so both are advisory: a torn read
        costs one event of extra wait in either direction. The wait is a poll
        for the same reason — there is no lock to build a condition on — and
        the ceiling keeps the drain's liveness independent of a section that
        never comes to take its turn.
        """

        if self._gui_waiting <= 0:
            return
        self.owner_yields_to_gui += 1
        target = self.gui_lock_acquisitions + 1
        checks = 0
        while (
            self._gui_waiting > 0
            and self.gui_lock_acquisitions < target
            and not self._stop
            and checks < _YIELD_MAX_CHECKS
        ):
            sleep(_YIELD_SLEEP_S)
            checks += 1

    def _drop_pending(self) -> None:
        events = self._events
        while True:
            try:
                events.popleft()
            except IndexError:
                break
            self.events_dropped_at_stop += 1
        messages = self._messages
        while True:
            try:
                messages.popleft()
            except IndexError:
                return
            # Counted, not silent: a retarget dropped at teardown is correct
            # (the window is unwinding) but a *non-zero* count outside teardown
            # would be a lost wakeup, and only the counter can say which.
            self.messages_dropped_at_stop += 1

    def _dispatch(self, event: Any) -> None:
        failure: BaseException | None = None
        with self.timed_order_lock(_OWNER):
            started = perf_counter()
            try:
                self._kernel.dispatch_event(event)
            except BaseException as exc:  # the owner thread must outlive its bodies
                failure = exc
                self.errors += 1
                self.last_error = traceback.format_exc()
            event_ms = (perf_counter() - started) * 1000.0
            self.events_dispatched += 1
            if event_ms > self.worst_event_ms:
                self.worst_event_ms = event_ms
                self.worst_event_key_head = _key_head(getattr(event.spec, "key", None))
        if failure is not None and self._error_hook is not None:
            # A hook that raises would kill the one thread that reports it.
            try:
                self._error_hook(failure)
            except Exception:
                self.last_error = traceback.format_exc()

    def _dispatch_message(self, callback: Callable[[], None]) -> None:
        """Run one posted callback under its own order-lock section."""

        failure: BaseException | None = None
        with self.timed_order_lock(_OWNER):
            started = perf_counter()
            try:
                callback()
            except BaseException as exc:  # the owner thread must outlive its bodies
                failure = exc
                self.errors += 1
                self.last_error = traceback.format_exc()
            event_ms = (perf_counter() - started) * 1000.0
            self.messages_dispatched += 1
            if event_ms > self.worst_event_ms:
                self.worst_event_ms = event_ms
                self.worst_event_key_head = _key_head(getattr(callback, "owner_key", None))
        if failure is not None and self._error_hook is not None:
            try:
                self._error_hook(failure)
            except Exception:
                self.last_error = traceback.format_exc()

    # ------------------------------------------------------------ the bridge

    @contextmanager
    def timed_order_lock(self, side: str) -> Iterator[None]:
        """Hold the session-order lock, charging the wait to a side.

        ``side`` is ``"gui"`` or ``"owner"``. The two are counted separately
        because they answer different questions: owner waits say the GUI still
        holds session sections, GUI waits say the owner is in the GUI's way —
        and only the second is a regression.
        """

        if side not in (_GUI, _OWNER):
            raise ValueError(f"unknown order-lock side: {side!r}")
        started = perf_counter()
        if side == _GUI:
            # The GUI announces itself before it waits; the drain reads the
            # announcement between events and gives up one turn for it (see
            # `_yield_to_gui`). Without the announcement the fill collapses
            # into a lock convoy: a Python lock hands off by race, not by
            # queue order, so the owner — which releases and immediately
            # re-acquires per event during a completion streak — barges past a
            # waiting GUI thread for hundreds of milliseconds at a stretch
            # (measured on the 272-tile probe: p50 55 ms, max 550 ms).
            #
            # What the announcement buys is a turn, deliberately not priority.
            # Standing aside until every announced section had been admitted
            # was tried and refuted on the same probe: the owner then waited
            # 5.4 s of an 8.4 s fill, commit cohorts fell from 55–91 tiles to
            # about 10, `rung_prepare` went 669 → 1181, and the fill censored
            # anyway. Both extremes starve one side; the single-threaded
            # predecessor's real property was alternation.
            #
            # The counter is a plain int, not a bool: nested GUI sections
            # under Qt re-entrancy must not clear each other's announcement.
            # It is advisory — a torn read costs one event of extra wait — and
            # never a lock of its own.
            self._gui_waiting += 1
        try:
            self.order_lock.acquire()
        finally:
            if side == _GUI:
                self._gui_waiting -= 1
        wait_ms = (perf_counter() - started) * 1000.0
        if side == _GUI:
            self.gui_lock_acquisitions += 1
            self.gui_lock_wait_ms_total += wait_ms
            if wait_ms > self.gui_lock_wait_ms_max:
                self.gui_lock_wait_ms_max = wait_ms
        else:
            self.owner_lock_acquisitions += 1
            self.owner_lock_wait_ms_total += wait_ms
            if wait_ms > self.owner_lock_wait_ms_max:
                self.owner_lock_wait_ms_max = wait_ms
        try:
            yield
        finally:
            self.order_lock.release()

    # ----------------------------------------------------------- diagnostics

    def snapshot(self) -> dict[str, Any]:
        """Every counter, read without the lock. See the class docstring."""

        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "pending_events": len(self._events),
            "pending_messages": len(self._messages),
            "events_dispatched": int(self.events_dispatched),
            "events_dropped_at_stop": int(self.events_dropped_at_stop),
            "messages_posted": int(self.messages_posted),
            "messages_dispatched": int(self.messages_dispatched),
            "messages_dropped_at_stop": int(self.messages_dropped_at_stop),
            "worst_event_ms": float(self.worst_event_ms),
            "worst_event_key_head": str(self.worst_event_key_head),
            "owner_lock_acquisitions": int(self.owner_lock_acquisitions),
            "owner_lock_wait_ms_total": float(self.owner_lock_wait_ms_total),
            "owner_lock_wait_ms_max": float(self.owner_lock_wait_ms_max),
            "gui_lock_acquisitions": int(self.gui_lock_acquisitions),
            "gui_lock_wait_ms_total": float(self.gui_lock_wait_ms_total),
            "gui_lock_wait_ms_max": float(self.gui_lock_wait_ms_max),
            "owner_yields_to_gui": int(self.owner_yields_to_gui),
            "gui_waiting": int(self._gui_waiting),
            "errors": int(self.errors),
            "last_error": str(self.last_error),
        }


def _key_head(key: Any) -> str:
    head = key[0] if isinstance(key, tuple) and key else key
    text = repr(head)
    if len(text) > _KEY_HEAD_LIMIT:
        return text[: _KEY_HEAD_LIMIT - 3] + "..."
    return text


__all__ = ["PresentationOwner", "montage_completion_route"]
