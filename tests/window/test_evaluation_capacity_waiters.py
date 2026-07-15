"""Capacity waiters are owned by the single QtKernelBridge after R1."""

import time


def _make_controller(qt_app):
    from arrayscope.kernel.eval_adapter import KernelEvaluationController as EvaluationController

    return EvaluationController(name="capacity-test", max_workers=1)


def _drain_until(predicate, *, timeout_s=5.0):
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_waiter_fires_after_any_completion_and_clears(qt_app):
    controller = _make_controller(qt_app)
    fired = []
    try:
        controller.notify_when_capacity("pump-a", lambda: fired.append("a"))

        controller.start_latest(
            lambda: 42,
            key="job",
            priority=20,
            replace_group="group",
            on_done=lambda value: None,
        )
        assert _drain_until(lambda: bool(fired))
        assert fired == ["a"]

        controller.start_latest(
            lambda: 43,
            key="job2",
            priority=20,
            replace_group="group2",
            on_done=lambda value: None,
        )
        assert _drain_until(lambda: controller.diagnostics().completed >= 2)
        assert fired == ["a"]
    finally:
        controller.shutdown_for_close()


def test_rearming_replaces_callable_per_key(qt_app):
    controller = _make_controller(qt_app)
    fired = []
    try:
        controller.notify_when_capacity("pump", lambda: fired.append("old"))
        controller.notify_when_capacity("pump", lambda: fired.append("new"))

        controller.start_latest(
            lambda: 1,
            key="job",
            priority=20,
            replace_group="group",
            on_done=lambda value: None,
        )
        assert _drain_until(lambda: bool(fired))
        assert fired == ["new"]
    finally:
        controller.shutdown_for_close()


def test_waiter_exception_does_not_break_drain(qt_app):
    controller = _make_controller(qt_app)
    fired = []

    def explode():
        raise RuntimeError("waiter boom")

    try:
        controller.notify_when_capacity("bad", explode)
        controller.notify_when_capacity("good", lambda: fired.append("ok"))

        controller.start_latest(
            lambda: 1,
            key="job",
            priority=20,
            replace_group="group",
            on_done=lambda value: None,
        )
        assert _drain_until(lambda: bool(fired))
        assert fired == ["ok"]
    finally:
        controller.shutdown_for_close()
