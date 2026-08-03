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
