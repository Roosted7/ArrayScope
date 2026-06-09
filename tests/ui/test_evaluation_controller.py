import os
import inspect
import time

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_evaluation_controller_ignores_stale_results(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController()
    done = []
    stale = []

    controller.start(lambda: (time.sleep(0.12), "old")[1], on_done=done.append, on_stale=lambda: stale.append("old"))
    controller.start(lambda: "new", on_done=done.append, on_stale=lambda: stale.append("new"))

    QtTest.QTest.qWait(220)
    qt_app.processEvents()

    assert done == ["new"]
    assert stale == ["old"]


def test_evaluation_controller_dedupes_and_limits_prefetch(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController()
    first = controller.start_prefetch(lambda: "a", key=("prefetch", 1))
    duplicate = controller.start_prefetch(lambda: "b", key=("prefetch", 1))
    controller._max_prefetch = 1
    limited = controller.start_prefetch(lambda: "c", key=("prefetch", 2))

    controller.shutdown_for_close()

    assert first.scheduled
    assert not duplicate.scheduled
    assert duplicate.reason == "deduped"
    assert not limited.scheduled
    assert limited.reason == "limited"


def test_start_latest_clears_queued_work_and_only_commits_newest(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []
    stale = []

    controller.start_latest(
        lambda: (time.sleep(0.12), "old")[1],
        key="old",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
        on_stale=lambda: stale.append("old"),
    )
    controller.start_latest(
        lambda: "new",
        key="new",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
        on_stale=lambda: stale.append("new"),
    )

    QtTest.QTest.qWait(240)
    qt_app.processEvents()

    assert done == ["new"]
    assert "old" in stale


def test_visible_pool_max_thread_count_is_one(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1, name="visible")

    assert controller.pool.maxThreadCount() == 1


def test_shutdown_ignores_late_results(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    done = []

    controller.start_latest(
        lambda: (time.sleep(0.08), "late")[1],
        key="late",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=done.append,
    )
    controller.shutdown_for_close()

    QtTest.QTest.qWait(140)
    qt_app.processEvents()

    assert done == []


def test_start_latest_can_pass_cancellation_token(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    seen = []

    controller.start_latest(
        lambda token: seen.append(hasattr(token, "cancelled")) or "ok",
        key="token",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
        pass_token=True,
    )

    QtTest.QTest.qWait(80)
    qt_app.processEvents()

    assert seen == [True]


def test_cancelled_evaluation_does_not_call_error(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.operations.cancellation import EvaluationCancelled
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    errors = []
    stale = []

    controller.start_latest(
        lambda token: (_ for _ in ()).throw(EvaluationCancelled()),
        key="cancel",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
        on_error=errors.append,
        on_stale=lambda: stale.append(True),
        pass_token=True,
    )

    QtTest.QTest.qWait(80)
    qt_app.processEvents()

    assert errors == []
    assert stale == [True]


def test_controller_diagnostics_counts_completed_cancelled_stale(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.operations.cancellation import EvaluationCancelled
    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    controller.start_latest(
        lambda: "done",
        key="done",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
    )
    QtTest.QTest.qWait(60)
    qt_app.processEvents()
    controller.start_latest(
        lambda token: (_ for _ in ()).throw(EvaluationCancelled()),
        key="cancel",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
        on_stale=lambda: None,
        pass_token=True,
    )
    QtTest.QTest.qWait(60)
    qt_app.processEvents()

    diagnostics = controller.diagnostics()
    assert diagnostics.completed == 1
    assert diagnostics.cancelled == 1
    assert diagnostics.stale >= 1


def test_prefetch_can_be_cancelled_separately(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    started = controller.start_prefetch(lambda: "prefetch", key="prefetch")

    controller.cancel_prefetch()

    assert started.scheduled
    assert not controller._prefetch_keys


def test_start_prefetch_idle_elapsed_false_blocks_with_idle_reason(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    started = controller.start_prefetch(lambda: "prefetch", key="prefetch", idle_elapsed=False)

    assert not started.scheduled
    assert started.reason == "idle"
    assert controller.diagnostics().prefetch_idle_blocked == 1


def test_start_prefetch_zero_memory_budget_blocks_with_cost_reason(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    controller = EvaluationController(max_workers=1)
    started = controller.start_prefetch(lambda: "prefetch", key="prefetch", memory_budget_bytes=0)

    assert not started.scheduled
    assert started.reason == "cost"
    assert controller.diagnostics().prefetch_cost_blocked == 1


def test_start_prefetch_no_longer_accepts_idle_deadline_ms(qt_app):
    from arrayscope.window.evaluation_controller import EvaluationController

    signature = inspect.signature(EvaluationController.start_prefetch)

    assert "idle_deadline_ms" not in signature.parameters


def test_is_busy_reflects_pending_or_running_work(qt_app):
    from pyqtgraph.Qt import QtTest

    from arrayscope.window.evaluation_controller import EvalPriority, EvaluationController

    controller = EvaluationController(max_workers=1)
    controller.start_latest(
        lambda: (time.sleep(0.05), "done")[1],
        key="busy",
        priority=EvalPriority.VISIBLE_IMAGE,
        replace_group="visible",
        on_done=lambda _value: None,
    )

    assert controller.is_busy()
    QtTest.QTest.qWait(100)
    qt_app.processEvents()
    assert not controller.is_busy()
