import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.clear()
    settings.sync()


def test_nearby_slice_prefetch_uses_prefetch_state_keys(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._active_slice_axis = 2
        win._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.image_cache_diagnostics().prefetch_scheduled > 0
        assert win.operation_evaluator.image_cache_diagnostics().prefetch_stored > 0
        assert win.operation_evaluator.cached_image(win.view_state.with_slice(2, 0)) is not None
    finally:
        win.close()


def test_nearby_slice_prefetch_skips_operation_backed_documents(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.request_operation("reverse", 0)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        before = win.operation_evaluator.image_cache_diagnostics()
        win._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.image_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert after.prefetch_skipped > before.prefetch_skipped
    finally:
        win.close()
