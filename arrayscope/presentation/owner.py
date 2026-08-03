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

Dispatch takes the lock **per bounded batch** — up to :data:`_EVENT_BATCH_LIMIT`
events or :data:`_EVENT_BATCH_MS` of dispatch time, whichever comes first — and
stands aside between batches for an announced GUI section (see
:meth:`PresentationOwner._yield_to_gui`).

That rule was the other way round until 2026-08-03, and the flip is worth
stating because the reasoning it replaces was correct when written. Stage 1's
rule was "per event, never per batch": the GUI held session sections for 20-45
ms at a time (``retarget_frame_pipeline`` at ~89 ms x 52 per fill), so a batch
held end to end was one more multi-frame stall on top of the GUI's own. Slices
D2a-D2c planned and applied those sections away, and slice D3 then measured the
routed fill with **every** GUI-side hold under a frame — and it still censored.
The cost it named was hand-off count: ~700 owner acquisitions per fill on one
``RLock``, the GUI's mean wait per acquisition 3.5-4.2 ms (nothing) and its
total 1.0-1.4 s of an 8 s fill (everything).

The batch is what that reading asks for, and **the measurement then refuted the
reading**: batching cut owner acquisitions to 267-303 and left the GUI's total
wait exactly where it was (1010-1577 ms against 933-1460 ms unbatched, same
box, interleaved arms — see :func:`montage_completion_route`). What costs the
fill is the owner's *occupancy* of the bridge, not the number of times it hands
it over, and no drain policy can shrink an occupancy.

So the rule stands on a smaller claim than the one it was built for: the GUI's
worst wait for the bridge is bounded by **one batch** rather than by one body,
which is under a frame by construction and is the same budget shape the Qt
bridge's own drain uses. With the ``frame:`` clause narrowed there are ~6 events
a fill to batch, so this is scaffolding held for the next attempt, measured
neutral rather than measured good. The stand-aside is unchanged, and still
between batches rather than inside one — yielding under the lock would be a
convoy with extra latency.
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

#: Events dispatched under one order-lock acquisition, and the wall-clock
#: budget that closes a batch early. Both are the Qt bridge drain's own numbers:
#: a batch is bounded by whichever comes first, so a stream of cheap admissions
#: costs one hand-off per eight and a stream of expensive ones still hands the
#: lock back inside a frame.
_EVENT_BATCH_LIMIT = 8
_EVENT_BATCH_MS = 8.0

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

    (**That last sentence is the prediction slice D3 falsified.** The sections
    were shrunk to well under a frame and the fill censored anyway. Read the
    D3 entry at the end of this ledger before acting on any condition stated
    between here and it — the earlier ones are kept because each names what it
    measured, not because any of them is still the open question.)

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

    **Slice D2b took the submit out of the commit's section and split the
    retarget into stage/plan/admit.** Same instrumentation, same probe, three
    runs (9 fills, all ``272/272 within-5s``), per fill:

    ===============================  =====  ========  =======
    GUI section                      calls  total ms  max ms
    ===============================  =====  ========  =======
    ``commit_pending_session``       31-34    605-880   44-171
    ``retarget_frame_pipeline``      31-33  1010-1472      193
    ===============================  =====  ========  =======

    The commit is where the slice landed: 1913 -> ~660 ms and 520 -> ~46 ms
    typical, because the 526 ms backend submit is no longer a section. **The
    retarget did not move**, and the sub-phase breakdown says why the D2b
    design was aimed at the wrong statement: ``ladder.plan`` — the "expensive
    middle" the split was built for — is 8-114 ms *per fill total*. The hold is
    in staging, which cannot leave the lock because it writes:
    ``effects.tile_states`` (2.1-2.2 s per 3 fills; it releases evaluation
    claims and seeds crop rebinds) and
    ``mark_ladder_swaps_for_current_demand`` (1.7-1.8 s, one call to 164 ms).
    Those two are the R5 model rebuild, and getting them off the lock needs the
    planner/applier treatment slice D1 gave the commit — not another split.

    **Slice D3 met that condition and the probe refused a fourth time. This is
    the entry that changes the plan**, because the stated precondition was
    satisfied and it was not enough. D2c planned/applied both staging writes,
    and the GUI-side holds, instrumented per calling funnel on the routed arm
    itself, are now all well under a frame:

    ==============================  =====  ========  =======
    GUI section (per fill)          calls  total ms  max ms
    ==============================  =====  ========  =======
    ``_commit_tile_layer``          74-132   294-604   15-20
    ``_submit_prepared_transaction`` 70-128   194-531   16-20
    ``retarget_frame_pipeline``      12-87     16-162    4-6
    ``commit_pending_session``      74-132     19-31    2-6
    ==============================  =====  ========  =======

    The fill censors anyway. Six fills on the widened clause against six
    control fills, interleaved on one box within the hour: **6/6 censored (36,
    40, 61, 64, 66, 125 of 272 presented) against 6/6 ``272/272 within-5s``**,
    target settle 4.27-4.81 s. Bisected as before — routing alone censors
    (108/29/35 of 272) with ``fire()`` on the timer, and ``fire()`` alone
    without routing storms exactly as slice B measured (18, 5, 5 of 272), so
    the coupling claim survives and the routing does not.

    The counters name a blocker that is **not hold time**, which is why no
    further hold-shrinking slice can reach it:

    ==========================  ===============  =============
    per fill                    routed arm       control arm
    ==========================  ===============  =============
    ``events_dispatched``       637-662          2
    ``owner_lock_acquisitions`` 692-719          3-5
    ``gui_lock_acquisitions``   280-356          1210-1327
    ``gui_lock_wait_ms_total``  984-1407         0.7-192
    ``gui_lock_wait_ms_max``    135-221          0-100
    ``owner_lock_wait_ms_total`` 147-331         0-6
    ``owner_yields_to_gui``     31-41            0-4
    ==========================  ===============  =============

    Neither side starves the way stage 1 (owner 5.4 s) or D2a (owner 1.8 s)
    did. What happens instead is **~700 hand-offs per fill on one RLock**: the
    GUI's mean wait per acquisition is 3.5-4.2 ms, which is nothing, and its
    total is 1.0-1.4 s of an ~8 s fill, which is everything. The stand-aside
    fires 31-41 times and cannot help — it trades a turn, and there is no
    policy under which 700 alternations cost less than the hand-offs
    themselves.

    Two second-order measurements say the same thing in the fill's own units:

    - **cohorts collapse**, control ``1,1,3,4,25,68,68,68,34`` /
      ``1,1,3,91,91,86`` against routed runs of ones and eights
      (``1,1,1,1,1,1,1,2,128,13,16,8,8,4,68,18``). The commit gate's batching
      never came from the gate: it came from the admissions being on the
      *commit's* thread, so none could land inside a commit. Routed, every
      commit re-arms the gate while it runs and presents only what arrived
      during the previous one.
    - **throughput per callback-second** falls 412-472 -> 112-294 items/s, and
      one routed target pass admitted 437 items for 272 tiles: the same tiles
      re-committed. Draw rate is *unchanged* (21-32 tile draws per control fill
      against 31-35 routed, over comparable wall clock), so this is not a
      pacing effect — the draws carry less each.

    So the condition for a fifth attempt is not another section. It is that the
    admission path stop sharing one lock with the commit — a per-family or
    reader/writer bridge, or the admissions writing state the commit reads
    through a snapshot — **plus** a commit-gate barrier that does not depend on
    the admissions running on the commit's thread. Both are new mechanism, not
    a shrink.

    **Slice D3b built both of those and the probe refused a fifth time — and
    this entry retires the D3 diagnosis above rather than extending it.** The
    two mechanisms were: one order-lock section per bounded batch (8 events or
    8 ms) instead of per event, and the gate's Qt post deferred to the tail of
    the owner's drain batch (:meth:`PresentationOwner.defer_to_batch_tail`)
    whenever an admission armed it with more admissions still queued. Nine
    routed fills against nine control fills, interleaved on one box:
    **9/9 censored (59, 75, 91, 100, 108, 108, 112, 137, 161 of 272) against a
    control that presents 272/272 on every fill** (settle 4.59-4.99 s; the
    control's first fill of each process reports its milestone late with all
    272 presented, on both arms, and is environmental).

    The counters are the finding, because M1 did exactly what it was designed
    to do and it bought nothing:

    ===========================  ==============  ==============  ===========
    per fill                     routed, batch 8 routed, batch 1 control
    ===========================  ==============  ==============  ===========
    ``owner_lock_acquisitions``  267-303         682-702         3-5
    ``gui_lock_wait_ms_total``   1010-1577       933-1460        1-132
    ``gui_lock_wait_ms_max``     138-176         148-167         11-80
    ``owner_yields_to_gui``      22-38           32-39           0-3
    ===========================  ==============  ==============  ===========

    Cutting the hand-offs by 2.4x moved the GUI's total wait **not at all**. So
    the quantity that costs 1.0-1.5 s of an ~8 s fill is not the number of
    alternations (D3's reading, refuted here by its own successor) but the
    owner's *occupancy* of the bridge — ~640 admission bodies' worth of work
    holding a lock the GUI needs. No batching, priority or stand-aside policy
    can reduce an occupancy; only taking the admission path off this lock can.

    M2 failed for a reason worth stating separately, because it looks like a
    tuning failure and is not: **the drain batch is not a barrier, because the
    owner is never behind its producers.** The deferral was accepted 9-17 times
    per fill against ~640 admissions — every other gate arm found the event
    deque already empty, i.e. the tail was *now*. What made the single-threaded
    predecessor's cohorts large was the GUI thread being *busy committing* while
    completions queued behind it in the same loop; the owner has no equivalent
    occupancy to accumulate against, and 8 ms of batching does not create one.

    The cohorts show the third-order consequence, and it names where a barrier
    must act. Routed target passes fit ``fixed=0.00ms item=11.2-16.0ms`` with
    ``frame-critical=1``; control target passes fit ``fixed=2.4-8.0ms
    item=0.27-0.79ms`` with ``frame-critical=0``. That degenerate fit is R5's
    documented fill bistability: small cohorts misattribute the commit's fixed
    cost to the item term, the model then prices any larger chunk out of the
    frame budget, and the chunk stays at 8 for the rest of the pass. Cohort
    collapse is therefore not merely a symptom of the routing — it **latches**,
    which is why a barrier that produces a large *first* cohort is a
    precondition and a barrier that merely averages is not. Throughput per
    callback-second: control 404-527 target items/s, routed 47-197.

    Draw rate is unchanged again (control 20-32 tile draws per fill, routed
    23-31, over comparable wall clock), which rules out pacing for the second
    time.

    Both mechanisms are kept — they are inert with this clause narrowed (the
    batch holds the ~6 stage-plan bodies, and the deferral is refused 100% of
    the time, measured) and they are what the next attempt would otherwise
    rebuild. Neither is evidence of anything on its own, and the bisect says so:
    M1 alone and M2 alone censor identically to the pair.

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
        #: Callbacks deferred to the tail of the batch that is running right
        #: now (:meth:`defer_to_batch_tail`). Appended only by the owner thread
        #: and only while it is inside a batch, so it needs no lock and no
        #: cross-thread contract at all.
        self._batch_tail: deque = deque()
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
        self.owner_batch_acquisitions = 0
        self.owner_events_per_batch_max = 0
        self.batch_tail_deferrals = 0
        self.batch_tail_dropped_at_stop = 0
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

    def defer_to_batch_tail(self, callback: Callable[[], None]) -> bool:
        """Run ``callback`` once at the tail of the batch already running.

        ADR 0065 stage 2 slice D3b, and the answer to the second blocker D3
        named: **the commit gate's cohorts were an accident of single-threading
        and need a barrier of their own.** Before the admissions moved here, an
        admission could not land inside a commit because both ran on the GUI
        thread; the gate therefore batched whatever arrived between commits, and
        cohorts of 55-91 tiles were the result. Routed, every admission arms the
        gate from this thread while the GUI is mid-commit, so the next commit
        presents only what arrived during the previous one — measured cohorts of
        ones and eights, and a censored fill.

        The barrier this restores is the drain batch. The *arming* still happens
        immediately (the gate-owner flags are the coalescing key and nothing
        about their single-flight semantics changes); only the **wakeup** waits,
        until the event deque is empty. One gate post per batch, not per
        admission, and the cohort is everything the batch admitted.

        Returns ``False`` — meaning "post it yourself, now" — unless the caller
        is the owner thread *inside a batch with events still queued*. Both
        terms matter: off the owner thread there is no batch to be the tail of,
        and with an empty event deque the tail is now, so deferring would only
        add a hop. A refusal is therefore never a lost wakeup; it is the caller
        keeping the wakeup it already had.

        **Measured, and it does not work**, which is why the ``frame:`` clause
        in :func:`montage_completion_route` is narrow again: with the
        admissions routed, the deferral was accepted 9-17 times per fill against
        ~640 admissions, because the owner is never behind its producers and an
        empty deque means the tail is *now*. The predecessor's cohorts came from
        the GUI thread being busy inside a 20-45 ms commit while completions
        piled up in the same loop — an occupancy the owner has no equivalent of,
        and one that 8 ms of batching does not manufacture. A barrier that
        works has to hold the wakeup on the *gate's* terms (a quantity, a
        deadline), not on the drain's structure. Kept for the next attempt;
        refused 100% of the time as the tree stands.
        """

        if self._stop or threading.current_thread() is not self._thread:
            return False
        if not self._events:
            return False
        self._batch_tail.append(callback)
        self.batch_tail_deferrals += 1
        return True

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

        The deferred **batch tail** (:meth:`defer_to_batch_tail`) runs at the
        same point, and ahead of the messages: it carries the presentation
        gate's wakeup, which is a queued Qt event the GUI cannot act on until
        it next turns its loop, so posting it before the retarget message
        costs the owner nothing and gives the GUI its cohort one hop sooner.
        """

        events = self._events
        messages = self._messages
        while not self._stop:
            if events:
                self._dispatch_event_batch()
                self._yield_to_gui()
                continue
            self._run_batch_tail()
            try:
                message = messages.popleft()
            except IndexError:
                if events:
                    continue
                return
            self._dispatch_message(message)
            self._yield_to_gui()

    def _run_batch_tail(self) -> None:
        """Run what the batch deferred, with the lock down.

        Deliberately outside any order-lock section: the one caller posts a Qt
        event, which needs no session ordering, and running it under the lock
        would put a hand-off back exactly where this mechanism removes one. A
        raising tail is contained the way every other body here is — the owner
        thread must outlive what it runs.
        """

        tail = self._batch_tail
        while True:
            try:
                callback = tail.popleft()
            except IndexError:
                return
            try:
                callback()
            except BaseException:  # the owner thread must outlive its bodies
                self.errors += 1
                self.last_error = traceback.format_exc()

    def _yield_to_gui(self) -> None:
        """Give up **one turn** to a waiting GUI section, between batches.

        The receiving half of the announcement made in
        :meth:`timed_order_lock`, and the anti-convoy mechanism: re-acquiring
        the order lock immediately after releasing it wins the race against a
        thread that has been waiting for it, because a Python lock hands off
        by race and not by queue order. Called **between** batches, never while
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
        section announced during the next batch triggers the next stand-aside,
        which is one-for-one alternation under contention and full owner
        throughput whenever the GUI is idle. Since 2026-08-03 the unit of that
        alternation is a batch rather than an event, which is the whole of what
        makes it affordable at admission volume: the turns are the same shape,
        there are eight times fewer of them.

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
        tail = self._batch_tail
        while True:
            try:
                tail.popleft()
            except IndexError:
                return
            # Same rule, and the same reason it is a count rather than a run: a
            # gate post delivered into a window that is unwinding is a wakeup
            # for a commit that must not happen.
            self.batch_tail_dropped_at_stop += 1

    def _dispatch_event_batch(self) -> None:
        """Dispatch up to one bounded batch under ONE order-lock section.

        The batch closes on whichever of the two bounds arrives first, and on
        ``_stop`` — re-checked per event, so a shutdown never waits out a batch
        it could have abandoned. Failures are collected rather than reported
        inside the section: an error hook is application code and must not run
        with the migration bridge held.
        """

        events = self._events
        failures: list[BaseException] = []
        dispatched = 0
        with self.timed_order_lock(_OWNER):
            self.owner_batch_acquisitions += 1
            batch_started = perf_counter()
            while dispatched < _EVENT_BATCH_LIMIT and not self._stop:
                try:
                    event = events.popleft()
                except IndexError:
                    break
                started = perf_counter()
                try:
                    self._kernel.dispatch_event(event)
                except BaseException as exc:  # the owner thread must outlive its bodies
                    failures.append(exc)
                    self.errors += 1
                    self.last_error = traceback.format_exc()
                now = perf_counter()
                event_ms = (now - started) * 1000.0
                self.events_dispatched += 1
                dispatched += 1
                if event_ms > self.worst_event_ms:
                    self.worst_event_ms = event_ms
                    self.worst_event_key_head = _key_head(getattr(event.spec, "key", None))
                if (now - batch_started) * 1000.0 >= _EVENT_BATCH_MS:
                    break
        if dispatched > self.owner_events_per_batch_max:
            self.owner_events_per_batch_max = dispatched
        if self._error_hook is None:
            return
        for failure in failures:
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
            "pending_batch_tail": len(self._batch_tail),
            "owner_batch_acquisitions": int(self.owner_batch_acquisitions),
            "owner_events_per_batch_max": int(self.owner_events_per_batch_max),
            "batch_tail_deferrals": int(self.batch_tail_deferrals),
            "batch_tail_dropped_at_stop": int(self.batch_tail_dropped_at_stop),
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
