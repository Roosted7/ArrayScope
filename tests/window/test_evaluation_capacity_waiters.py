"""Capacity waiters: a declined admission always leaves a wakeup armed.

ADR 0051 P2 (machine-derived dispatch): when WorkGraph backpressure
declines an admission, the caller arms a one-shot waiter; the next
controller drain that processes any completion fires it, and the caller
re-derives its work from authoritative state.  Without this, the last
in-flight decline freezes the pump (the 2026-07-05 dead-pump defect).
"""

import time


def _make_controller(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    return EvaluationController(name="capacity-test", max_workers=1)


def _drain_until(controller, predicate, *, timeout_s=5.0):
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        controller._drain_queue()
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_waiter_fires_after_any_completion_and_clears(qt_app):
    controller = _make_controller(qt_app)
    fired = []
    controller.notify_when_capacity("pump-a", lambda: fired.append("a"))

    controller.start_latest(
        lambda: 42,
        key="job",
        priority=20,
        replace_group="group",
        on_done=lambda value: None,
    )
    assert _drain_until(controller, lambda: bool(fired))
    assert fired == ["a"]

    # One-shot: a later completion must not re-fire a consumed waiter.
    controller.start_latest(
        lambda: 43,
        key="job2",
        priority=20,
        replace_group="group2",
        on_done=lambda value: None,
    )
    assert _drain_until(controller, lambda: controller._completed_count >= 2)
    assert fired == ["a"]


def test_rearming_replaces_callable_per_key(qt_app):
    controller = _make_controller(qt_app)
    fired = []
    controller.notify_when_capacity("pump", lambda: fired.append("old"))
    controller.notify_when_capacity("pump", lambda: fired.append("new"))

    controller.start_latest(
        lambda: 1,
        key="job",
        priority=20,
        replace_group="group",
        on_done=lambda value: None,
    )
    assert _drain_until(controller, lambda: bool(fired))
    assert fired == ["new"]


def test_waiter_exception_does_not_break_drain(qt_app):
    controller = _make_controller(qt_app)
    fired = []

    def explode():
        raise RuntimeError("waiter boom")

    controller.notify_when_capacity("bad", explode)
    controller.notify_when_capacity("good", lambda: fired.append("ok"))

    controller.start_latest(
        lambda: 1,
        key="job",
        priority=20,
        replace_group="group",
        on_done=lambda value: None,
    )
    assert _drain_until(controller, lambda: bool(fired))
    assert fired == ["ok"]
