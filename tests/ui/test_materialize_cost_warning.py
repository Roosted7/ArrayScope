import numpy as np

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings, process_events as _process_events


def test_large_fft_pipeline_warning_mentions_estimated_peak_bytes(qtbot, monkeypatch):
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: messages.append(args[2]) or QtWidgets.QMessageBox.StandardButton.No)
    win = ArrayScopeWindow(np.zeros((8, 16), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win.request_operation("centered_fft", 0)
        _process_events(qtbot, count=20)
        assert not win._confirm_expensive_full_array("Materialize", win.data.shape, win.data.dtype)
        assert "estimated peak" in messages[-1]
        assert "CenteredFFT axis 0" in messages[-1]
    finally:
        win.close()


def test_disabled_fft_step_does_not_trigger_fft_warning(qtbot, monkeypatch):
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.operations.pipeline import CenteredFFT, OperationStep
    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected warning")))
    win = ArrayScopeWindow(np.zeros((8, 16), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win.operation_coordinator.load_steps((OperationStep(CenteredFFT(axis=0), enabled=False),))
        win._set_document(win.operation_coordinator.document)
        assert win._confirm_expensive_full_array("Materialize", win.data.shape, win.data.dtype)
    finally:
        win.close()


def test_large_reduction_pipeline_warning_uses_cost_model(qtbot, monkeypatch):
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.window import ArrayScopeWindow

    _clear_arrayscope_settings()
    messages = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: messages.append(args[2]) or QtWidgets.QMessageBox.StandardButton.No)
    win = ArrayScopeWindow(np.zeros((8, 16), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        win.request_operation("rss", 0)
        _process_events(qtbot, count=20)
        win._confirm_expensive_full_array("Export", win.data.shape, win.data.dtype)
        assert "RootSumSquares axis 0" in messages[-1]
        assert "estimated peak" in messages[-1]
    finally:
        win.close()
