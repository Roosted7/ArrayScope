"""What a window must actually be wired to for ADR 0065 stage 1 to be real.

The owner thread and the routed queue are adjudicated beside themselves
(``tests/presentation``, ``tests/kernel``). What is only checkable from a whole
window is the composition: that the routed queue is the one the *kernel*
publishes to, that the bridge's per-event lock is the *owner's* GUI side, and
that the thread the window starts is a thread the window also joins.

That last one is the reason this file exists at all. A leaked
``presentation-owner`` thread per window is invisible in a green suite and
compounds across the hundreds of windows a pass builds, so the join is pinned
here, against the thread object this window started rather than against the
process's thread list — another test's leaked window must not be able to fail
this one, nor hide its failure.

The routing claim is thread identity taken from inside a handler, and it is
mutation-checked in the same test by an unrouted scope: an assertion that the
handler merely *ran* would pass with the whole composition deleted.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from arrayscope.kernel.completions import RoutedCompletionQueue
from arrayscope.kernel.task import Lane, TaskSpec
from arrayscope.presentation.owner import montage_completion_route
from arrayscope.tools.interaction_budget import interaction_settle_timeout_ms
from arrayscope.window.diagnostics_snapshot import collect_runtime_diagnostics_snapshot

pytest.importorskip("pytestqt")

#: The thread name :class:`PresentationOwner` gives its loop. Named here
#: because the leak pin identifies the thread by it and nothing else can.
OWNER_THREAD_NAME = "presentation-owner"

#: A montage-scoped scope prefix the shipped predicate claims, and one it does
#: not. The pair is the mutation check: swap either and the routing test fails.
#: Tracks the predicate by construction — stage 1 narrowed its claim to the
#: stage-plan family alone (`montage_completion_route`), so `frame:` moved from
#: the routed side to the unrouted one and this constant moved with it.
ROUTED_SCOPE = "montage-stage-plan:presentation-owner-wiring"
UNROUTED_SCOPE = "presentation-owner-wiring"

#: A window's kernel carries lane quotas, and ``TaskSpec``'s default lane
#: (``visible_materialization``) is one a settled window has fully spent: a
#: probe submitted there parks on quota and never completes, which reads
#: exactly like a routing failure. Probes go in a lane the window is not using.
PROBE_LANE = Lane.VISIBLE_PLANNING


def _data():
    return np.zeros((3, 8, 8), dtype=np.float32)


def _owner_threads() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if thread.name == OWNER_THREAD_NAME}


@pytest.fixture
def window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(_data())
    qtbot.addWidget(win)
    yield win
    win.close()


def _wait(qtbot, predicate):
    qtbot.waitUntil(predicate, timeout=interaction_settle_timeout_ms())


def test_a_window_starts_one_owner_thread_and_joins_it_on_close(qtbot):
    from arrayscope.window import ArrayScopeWindow

    before = _owner_threads()
    win = ArrayScopeWindow(_data())
    qtbot.addWidget(win)
    started = _owner_threads() - before

    assert len(started) == 1
    thread = next(iter(started))
    assert thread.is_alive()
    assert win.presentation_owner.snapshot()["running"] is True

    win.close()

    assert not thread.is_alive()
    assert not (_owner_threads() & started)
    assert win.presentation_owner.snapshot()["running"] is False


def test_closing_twice_is_still_one_clean_teardown(window):
    # Qt delivers a close to a window that is already closing (a second
    # ``close()``, an application quit after one). The owner's shutdown is
    # idempotent; this pins that the window's call site keeps it that way.
    window.close()
    window.close()

    assert window.presentation_owner.snapshot()["running"] is False


def test_the_kernel_publishes_into_the_routed_queue(window):
    queue = window.kernel.completions

    assert isinstance(queue, RoutedCompletionQueue)
    # The two ends of the route, read directly: the identity of the sink is
    # the composition claim, and no behaviour distinguishes "routed to this
    # owner" from "routed to some owner".
    assert queue._route_sink is window.presentation_owner
    assert queue._route_predicate is montage_completion_route


def test_the_bridges_per_event_lock_is_the_owners_gui_side(window):
    owner = window.presentation_owner
    acquisitions = owner.gui_lock_acquisitions

    with window.kernel_bridge._order_lock_factory():
        pass

    assert owner.gui_lock_acquisitions == acquisitions + 1


def test_a_routed_completion_dispatches_off_the_gui_thread(window, qtbot):
    gui_thread = threading.current_thread()
    seen: dict[str, threading.Thread] = {}

    for name, scope in (("routed", ROUTED_SCOPE), ("unrouted", UNROUTED_SCOPE)):
        window.kernel.submit(
            TaskSpec(key=f"wiring-{name}", fn=lambda: None, lane=PROBE_LANE, scope=scope),
            on_done=lambda _value, name=name: seen.setdefault(name, threading.current_thread()),
        )

    _wait(qtbot, lambda: {"routed", "unrouted"} <= seen.keys())

    assert seen["routed"].name == OWNER_THREAD_NAME
    assert seen["routed"] is not gui_thread
    assert seen["unrouted"] is gui_thread


def test_the_diagnostics_snapshot_publishes_the_owners_counters(window, qtbot):
    before = collect_runtime_diagnostics_snapshot(window).presentation_owner
    assert before is not None
    assert before["running"] is True

    done: list[int] = []
    for index in range(3):
        window.kernel.submit(
            TaskSpec(
                key=f"wiring-diagnostics-{index}",
                fn=lambda: None,
                lane=PROBE_LANE,
                scope=ROUTED_SCOPE,
            ),
            on_done=lambda _value: done.append(1),
        )
    _wait(qtbot, lambda: len(done) == 3)

    after = collect_runtime_diagnostics_snapshot(window).presentation_owner

    # A lower bound, not an equality: the window's own montage work dispatches
    # through the same owner, and this test does not own that traffic.
    assert after["events_dispatched"] - before["events_dispatched"] >= 3


# ------------------------------------------- the retarget as an owner operation


def _montage_window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    win._set_view_state(
        win.view_state.with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text="0:4")
    )
    win.update_image_view()
    return win


def test_a_posted_retarget_runs_the_ladder_on_the_owner_thread(qtbot):
    """ADR 0065 stage 2 slice B's counting pin, at the composition level.

    ``post_frame_pipeline_retarget`` is what the resize and viewport-only
    tails call now. The claim is not "a retarget happened" -- that would pass
    with the post reverted to a direct call, which is exactly the mutation
    this is checked against -- but *which thread ran the ladder*.

    Membership rather than equality, and deliberately: a live window retargets
    on its own schedule from the GUI thread, and this test owns one of those
    calls, not the traffic around it. The owner thread's name can appear in
    that list only if a ladder retarget ran off the GUI thread.
    """

    from arrayscope.window.frame_runtime import post_frame_pipeline_retarget

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        owner = win.presentation_owner
        session = renderer._frame_session
        assert session is not None
        threads: list[str] = []
        original = type(renderer).retarget_frame_pipeline

        def spy(self, target, **kwargs):
            threads.append(threading.current_thread().name)
            return original(self, target, **kwargs)

        renderer.retarget_frame_pipeline = spy.__get__(renderer, type(renderer))
        assert OWNER_THREAD_NAME not in threads

        before = owner.snapshot()["messages_dispatched"]
        post_frame_pipeline_retarget(renderer, session)

        _wait(qtbot, lambda: owner.snapshot()["messages_dispatched"] > before)
        assert OWNER_THREAD_NAME in threads
    finally:
        win.close()


def test_a_posted_retarget_for_a_superseded_session_does_not_run(qtbot):
    """The re-check inside the closure, not merely its presence.

    A posted operation can land after a rebirth. Slice B's rule is that the
    closure carries the identity pair it was built with and drops itself; the
    mutation is deleting that check, which this test fails. The evidence is
    again the owner thread's absence from the retarget log, because the GUI
    thread keeps retargeting this window for reasons of its own.
    """

    from arrayscope.window.frame_runtime import post_frame_pipeline_retarget

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        owner = win.presentation_owner
        session = renderer._frame_session
        assert session is not None
        threads: list[str] = []
        renderer.retarget_frame_pipeline = lambda *_args, **_kwargs: threads.append(
            threading.current_thread().name
        )

        # Post and supersede inside ONE order-lock section, so the closure
        # cannot be dispatched between the two: without this the test would be
        # racing the owner for its own precondition.
        with owner.timed_order_lock("gui"):
            before = owner.snapshot()["messages_dispatched"]
            # Advance the generation the way an index-window retarget does:
            # same object, new id. The post names the OLD one.
            session_id = int(session.session_id)
            post_frame_pipeline_retarget(renderer, session)
            session.session_id = session_id + 1

        _wait(qtbot, lambda: owner.snapshot()["messages_dispatched"] > before)
        assert OWNER_THREAD_NAME not in threads
    finally:
        win.close()


def test_settle_chrome_decided_on_the_owner_still_lands_on_the_gui_thread(qtbot):
    """The chrome half of slice B's split, pinned as a thread, not an effect.

    ``_settle_montage_visible_plan_if_complete`` decides from session state --
    which evaluates wherever the retarget runs -- and then publishes overlay
    chrome, which is Qt and must not. Asserting only that the overlay went away
    would pass with the split deleted, so what is recorded here is the thread
    the setters ran on, taken from inside them, while the decision itself is
    made on the owner.
    """

    from arrayscope.core.cache_status import CacheStatus

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        owner = win.presentation_owner
        _wait(qtbot, lambda: renderer._frame_session.visible_plan_complete())
        session = renderer._frame_session

        # Put the chrome back into its "still working" state so the settle has
        # something to undo.
        win.img_view.setEvaluationOverlay(True, "Updating image frame...")
        setter_threads: list[str] = []
        original = win.img_view.setEvaluationOverlay

        def spy(visible, text=""):
            setter_threads.append(threading.current_thread().name)
            return original(visible, text)

        win.img_view.setEvaluationOverlay = spy

        before = owner.snapshot()["messages_dispatched"]
        decided: list[str] = []

        def settle():
            decided.append(threading.current_thread().name)
            renderer._settle_montage_visible_plan_if_complete(session)

        owner.post(settle)
        _wait(qtbot, lambda: owner.snapshot()["messages_dispatched"] > before)
        _wait(qtbot, lambda: bool(setter_threads))

        assert decided == [OWNER_THREAD_NAME]
        assert set(setter_threads) == {threading.current_thread().name}
        assert win.operation_evaluator.last_status.status is CacheStatus.READY
    finally:
        win.close()


# ------------------------------------------- the prepare seam's three phases


def test_a_prepare_plans_while_another_thread_holds_the_order_lock(qtbot):
    """ADR 0065 stage 2 slice D2a: the plan phase acquires no lock.

    D2's refutation was that the owner's prepare was ONE lock section spanning
    the montage build — a faulthandler caught the owner inside
    ``shader_mapping.__eq__`` while the GUI waited on the same lock. The answer
    is structural: ``_stage_prepare_inputs`` is the lock's business (the
    private session copy plus every live, renderer and widget read the plan
    needs) and ``_plan_prepared_outcome`` is a computation over what it froze.

    The claim is therefore about *lock acquisition*, not about speed, and it is
    stated the only way that cannot pass vacuously: the session-order lock is
    held for the whole of the middle phase by a thread that is not the planner,
    and the planner still finishes. The mutation is re-adding a
    ``timed_order_lock`` anywhere inside ``_plan_prepared_outcome``, which parks
    the planner until this test's join timeout and reds it.

    The prepared outcome is deliberately dropped rather than taken. That is
    exactly what slice D1 bought: a plan that is never applied writes nothing.
    """

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        owner = win.presentation_owner
        effects = renderer._frame_pipeline.effects
        _wait(qtbot, lambda: effects.direct_tile_layer_presentation() is not None)
        direct_presentation = effects.direct_tile_layer_presentation()
        dirty_tiles = effects.session.peek_dirty_tiles()

        # Phase 1 on the GUI thread, exactly as the inline anchor runs it.
        staging = effects._stage_prepare_inputs()

        held = threading.Event()
        release = threading.Event()

        def hold_the_lock() -> None:
            with owner.timed_order_lock("owner"):
                held.set()
                release.wait(30.0)

        holder = threading.Thread(target=hold_the_lock, name="d2a-lock-holder")
        holder.start()
        assert held.wait(10.0), "the holder never took the order lock"

        planned: list[object] = []

        def plan() -> None:
            planned.append(
                effects._plan_prepared_outcome(
                    staging,
                    direct_presentation,
                    dirty_tiles=dirty_tiles,
                )
            )

        planner = threading.Thread(target=plan, name="d2a-planner")
        planner.start()
        planner.join(20.0)
        blocked = planner.is_alive()

        release.set()
        holder.join(10.0)
        planner.join(10.0)

        assert not blocked, (
            "the plan phase blocked on the session-order lock: it acquires one "
            "somewhere, which is the shape that refuted D2"
        )
        assert len(planned) == 1
    finally:
        win.close()


# ------------------------------------- slice D2b: the GUI's own hold shrinks


def _prepared_commit(qtbot, win):
    """Drive the window to a real prepared transaction, and take it.

    Both D2b move-1 pins need a commit that genuinely reaches the backend, so
    neither may settle for "nothing to submit" — a skip here would be the
    silent pass that makes an assertion meaningless. The wait is on the
    session's own obligation state, and once it returns no further Qt event
    runs, so the gate cannot commit the transaction out from under the test.
    """

    effects = win.renderer._frame_pipeline.effects
    session = effects.session
    _wait(qtbot, lambda: bool(session.dirty_payloads or session.pending_payload_upserts))
    session.flush_pending = True
    session.final_commit_pending = True
    direct_presentation = effects.direct_tile_layer_presentation()
    dirty_tiles = session.peek_dirty_tiles()
    effects._begin_commit_frame(dirty_tiles)
    outcome = effects._prepare_tile_layer_transaction(direct_presentation, dirty_tiles=dirty_tiles)
    transaction = effects._take_prepared_outcome(outcome)
    assert transaction is not None, f"the prepare bailed instead of building a delta: {outcome}"
    assert transaction.tile_delta.upserts, "the prepared delta carries no upserts to submit"
    return effects, session, transaction


def _hold_the_order_lock(owner, name: str):
    """Hold the order lock from another thread until the returned event is set."""

    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with owner.timed_order_lock("owner"):
            held.set()
            release.wait(30.0)

    holder = threading.Thread(target=hold, name=name)
    holder.start()
    assert held.wait(10.0), "the holder never took the order lock"
    return holder, release


def test_a_commit_lets_another_thread_in_across_its_backend_submit(qtbot):
    """ADR 0065 stage 2 slice D2b move 1: the backend submit is not a section.

    ``commit_pending_session`` held the lock across the whole commit — 43
    calls, 2350 ms, 526 ms worst on the 272-tile probe — and the submit inside
    it needs no session ordering at all.

    The claim is about lock *availability* during that span, so it is read from
    the other side: while the GUI thread is inside ``_present_tile_delta``,
    another thread tries to take the session-order lock, and it must get it.
    The commit then finishes and acknowledges normally. Put the submit back
    inside the enclosing section and that acquire times out — the whole test,
    not one assertion, is the mutation check.
    """

    win = _montage_window(qtbot)
    try:
        owner = win.presentation_owner
        effects = win.renderer._frame_pipeline.effects
        session = effects.session
        _wait(qtbot, lambda: bool(session.dirty_payloads or session.pending_payload_upserts))
        session.flush_pending = True
        session.final_commit_pending = True

        acquired: list[bool] = []
        submits: list[bool] = []
        original = type(effects)._present_tile_delta

        def contended_submit(self, transaction):
            held = threading.Event()
            release = threading.Event()

            def take_the_lock() -> None:
                got = owner.order_lock.acquire(timeout=5.0)
                acquired.append(bool(got))
                held.set()
                if got:
                    release.wait(20.0)
                    owner.order_lock.release()

            other = threading.Thread(target=take_the_lock, name="d2b-submit-window")
            other.start()
            assert held.wait(15.0), "the contending thread never reported"
            try:
                applied = original(self, transaction)
            finally:
                release.set()
                other.join(10.0)
            submits.append(bool(applied))
            return applied

        effects._present_tile_delta = contended_submit.__get__(effects, type(effects))
        try:
            effects.commit_pending_session()
        finally:
            del effects._present_tile_delta

        assert submits, "the commit never reached the backend, so nothing was observed"
        assert acquired == [True] * len(submits), (
            "another thread could not take the session-order lock while the "
            "commit was inside its backend submit: the submit is a section "
            "again, which is the 526 ms hold slice D2b removes"
        )
        assert submits == [True] * len(submits)
        assert bool(win.renderer._last_montage_report_acknowledges)
    finally:
        win.close()


def test_an_admission_inside_the_submit_window_is_not_acknowledged_away(qtbot):
    """The audit's answer, as a counting test rather than a claim.

    With the lock down across the submit, an admission can land between the
    plan's *apply* and the *ack* — and the ack pops ``dirty_payloads`` /
    ``pending_payload_upserts`` for every accepted upsert and releases the
    peeked dirty tiles. Both would swallow that admission's obligation.

    So one is staged deliberately: a delta tile is re-dirtied and re-appended
    while the window is open, exactly as ``_admit_evaluation_result`` does, and
    the tile must still be owed a presentation afterwards. The mutation is
    dropping either half of the re-derivation (the payload comparison or the
    prefix release), which leaves the tile clean and the session settled on a
    payload no commit presented.
    """

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        effects, session, transaction = _prepared_commit(qtbot, win)

        from arrayscope.window.frame_effects import _SubmitWindow

        window = _SubmitWindow(
            dirty_prefix=len(session.dirty_tiles),
            payloads=effects._submit_window_payloads(transaction),
        )
        tile = int(next(iter(transaction.tile_delta.upserts)))

        # The interleaving: a completion admits a NEWER payload for a tile this
        # commit is already submitting, and marks it dirty both ways.
        successor = object()
        session.display_tile_payloads[tile] = successor
        session.dirty_payloads[tile] = None
        session.dirty_tiles.append(tile)

        assert effects._present_tile_delta(transaction) is True
        ack = effects._capture_ack_inputs(transaction, commit_start=0.0, submit_span_ms=0.0)
        effects._acknowledge_transaction(ack, window=window)

        # Without this the rest is vacuous: a report that acknowledges nothing
        # pops nothing, so an unguarded ack would keep the tile dirty too.
        assert bool(renderer._last_montage_report_acknowledges)
        assert session.display_tile_payloads.get(tile) is successor
        assert tile in session.dirty_payloads, (
            "the acknowledgement popped an obligation an admission created "
            "while the submit ran with the order lock down"
        )
        assert tile in set(session.dirty_tiles)
        assert bool(session.flush_pending)
        assert int(getattr(renderer, "_montage_commit_interleaved_admissions", 0)) >= 1
    finally:
        win.close()


def test_a_retarget_plans_while_another_thread_holds_the_order_lock(qtbot):
    """Slice D2b move 2: ``ladder.plan`` is a computation, not a section.

    Same shape as D2a's prepare pin and for the same reason —
    ``retarget_frame_pipeline`` was one lock section 28-38 times a fill at
    ~50 ms mean and ~195 ms worst, and no owner-side policy can fit 630
    acquisitions between holds like that. The mutation is planning inside the
    staging section again.
    """

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        owner = win.presentation_owner
        session = renderer._frame_session
        pipeline = renderer._frame_pipeline_for_session(session)
        intent = renderer._montage_render_intent(session)
        scope = renderer._lod_admission_scope(session, intent)
        staged = pipeline.stage_retarget(intent, session.lod_policy_decision.demand, scope)
        assert staged is not None

        holder, release = _hold_the_order_lock(owner, "d2b-retarget-lock-holder")
        planned: list[object] = []

        planner = threading.Thread(
            target=lambda: planned.append(pipeline.plan_retarget(staged)),
            name="d2b-retarget-planner",
        )
        planner.start()
        planner.join(20.0)
        blocked = planner.is_alive()

        release.set()
        holder.join(10.0)
        planner.join(10.0)

        assert not blocked, (
            "the ladder plan blocked on the session-order lock: it acquires "
            "one somewhere, which is the hold this slice removes"
        )
        assert len(planned) == 1
    finally:
        win.close()


def test_the_retargets_whole_middle_runs_while_another_thread_holds_the_lock(qtbot):
    """Slice D2c: the swap marks and the state snapshot join the unlocked middle.

    D2b's pin above covered ``ladder.plan`` alone, and D2b's own numbers said
    that was the cheap part: ``effects.tile_states`` and
    ``mark_ladder_swaps_for_current_demand`` were 4.0 s of the retarget's
    per-fill hold between them, and both stayed under the lock because both
    write the session. They plan against a private copy now, so the claim is
    the same one and it is stated the same way — the order lock is held for the
    whole middle by a thread that is not the planner, and the planner still
    finishes. The mutation, as for D2a's and D2b's pins, is a
    ``timed_order_lock`` acquire anywhere inside the middle (verified: it parks
    the planner until this test's join timeout).
    """

    from arrayscope.render import lod as render_lod

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        owner = win.presentation_owner
        session = renderer._frame_session
        pipeline = renderer._frame_pipeline_for_session(session)
        intent = renderer._montage_render_intent(session)
        scope = renderer._lod_admission_scope(session, intent)
        staged = pipeline.stage_retarget(intent, session.lod_policy_decision.demand, scope)
        assert staged is not None
        staged_session = staged.staged_session
        assert staged_session is not None, "the stage phase took no private copy"
        assert staged_session is not session

        holder, release = _hold_the_order_lock(owner, "d2c-retarget-lock-holder")
        planned: list[object] = []

        def plan() -> None:
            render_lod.mark_ladder_swaps_for_current_demand(staged_session)
            planned.append(pipeline.plan_retarget(staged))

        planner = threading.Thread(target=plan, name="d2c-retarget-planner")
        planner.start()
        planner.join(20.0)
        blocked = planner.is_alive()

        release.set()
        holder.join(10.0)
        planner.join(10.0)

        assert not blocked, (
            "the retarget's middle blocked on the session-order lock: the swap "
            "marks or the state snapshot still take one, which is the 4.0 s of "
            "per-fill hold this slice removes"
        )
        assert len(planned) == 1
        assert planned[0].tile_states_plan is not None
    finally:
        win.close()


def test_a_plan_a_later_retarget_superseded_admits_nothing(qtbot):
    """The currency drop, counted where it happens.

    The middle runs unlocked, so a second retarget can stage while the first
    is still planning. Its staging bumped the admission generation and it owns
    the admissions; the first plan must drop rather than claim lifecycle state
    twice. Without the check the two plans both admit and the counter stays
    zero, which is the mutation.

    Slice D2c widens what "admits nothing" has to mean. The superseded plan now
    also carries the stage phase's whole session write set — the claim releases,
    the crop-rebind seed and the swap marks — so applying it would install
    decisions taken against a snapshot two generations old. The refusal is
    therefore read as a session fingerprint that does not move, not only as a
    submission count of zero.
    """

    from tests.window.test_presentation_plan_split import _session_fingerprint

    win = _montage_window(qtbot)
    try:
        renderer = win.renderer
        session = renderer._frame_session
        pipeline = renderer._frame_pipeline_for_session(session)
        intent = renderer._montage_render_intent(session)
        scope = renderer._lod_admission_scope(session, intent)
        demand = session.lod_policy_decision.demand

        first = pipeline.stage_retarget(intent, demand, scope)
        assert first is not None
        planned = pipeline.plan_retarget(first)
        assert planned.tile_states_plan is not None
        assert dict(planned.tile_states_plan.plan.staged), (
            "the superseded plan carries no session write, so nothing is observed"
        )
        # The successor: a whole retarget, staged and admitted, while the
        # predecessor was planning.
        second = pipeline.stage_retarget(intent, demand, scope)
        assert second is not None
        pipeline.admit_retarget(second, pipeline.plan_retarget(second))

        superseded_before = int(pipeline.counters.retargets_superseded)
        fingerprint_before = _session_fingerprint(session)

        assert pipeline.admit_retarget(first, planned) == 0
        assert int(pipeline.counters.retargets_superseded) == superseded_before + 1
        assert _session_fingerprint(session) == fingerprint_before
    finally:
        win.close()
