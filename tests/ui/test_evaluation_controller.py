import os
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
