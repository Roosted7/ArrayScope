import os
import time

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


def test_visible_render_controller_uses_latest_only_group(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9 * 10, dtype=float).reshape(8, 9, 10))
    qtbot.addWidget(win)
    calls = []
    original_start_latest = win.visible_evaluation_controller.start_latest

    def recording_start_latest(fn, **kwargs):
        calls.append(kwargs.get("replace_group"))

        def slow_fn():
            time.sleep(0.02)
            return fn()

        return original_start_latest(slow_fn, **kwargs)

    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", recording_start_latest)
    try:
        _process_events(qtbot, count=20)
        win._set_view_state(win.view_state.with_slice(2, 1))
        win.render(reason="test-1")
        win._set_view_state(win.view_state.with_slice(2, 2))
        win.render(reason="test-2")
        _process_events(qtbot, count=40)

        assert "visible-image" in calls
        assert win.visible_evaluation_controller.pool.maxThreadCount() == 1
    finally:
        win.close()


def test_prefetch_never_runs_during_montage(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        state = win.view_state.with_montage_axis(2, indices=(0, 1, 2), text=":")
        before = win.operation_evaluator.image_cache_diagnostics()
        win._prefetch_nearby_slices(state, None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.image_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert after.prefetch_skipped > before.prefetch_skipped
    finally:
        win.close()


def test_stale_visible_result_does_not_clear_updating_overlay(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9 * 10, dtype=float).reshape(8, 9, 10))
    qtbot.addWidget(win)
    captured = {}

    def capture_start_latest(_fn, **kwargs):
        captured.update(kwargs)
        kwargs["on_slow"]()
        return 1

    monkeypatch.setattr(win.visible_evaluation_controller, "start_latest", capture_start_latest)
    try:
        _process_events(qtbot, count=20)
        win.operation_evaluator.clear_cache()
        win.update_image_view()
        assert win.img_view._evaluation_overlay is not None
        assert win.img_view._evaluation_overlay.isVisible()

        captured["on_stale"]()
        assert win.img_view._evaluation_overlay.isVisible()
    finally:
        win.close()
