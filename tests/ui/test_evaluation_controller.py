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
