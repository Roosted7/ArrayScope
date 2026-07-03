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

    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def test_nearby_slice_prefetch_uses_prefetch_state_keys(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 0
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_stored > 0
        assert win.operation_evaluator.cached_display_tile(win.view_state.with_slice(2, 1)) is not None
    finally:
        win.close()


def test_nearby_slice_prefetch_skips_when_setting_disabled(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._active_slice_axis = 2
        before = win.operation_evaluator.display_cache_diagnostics()
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert after.prefetch_skipped > before.prefetch_skipped
    finally:
        win.close()


def test_prefetch_dispatch_is_queued_but_not_timer_admitted(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.renderer._schedule_prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        assert not hasattr(win, "_prefetch_idle_timer")
        _process_events(qtbot, count=10)
        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 0
    finally:
        win.close()


def test_prefetch_limits_to_two_neighbors(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 8, dtype=float).reshape(3, 4, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        momentum = SliceScrubMomentum()
        momentum.observe(2, now=0.0)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 4), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled <= 2
    finally:
        win.close()


def test_prefetch_deepens_with_sustained_directional_scrub(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        momentum = SliceScrubMomentum()
        base = time.monotonic() - 0.3
        for step, index in enumerate((2, 3, 4, 5)):
            momentum.observe(index, now=base + 0.1 * step)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 6), None)
        _process_events(qtbot, count=40)

        scheduled = win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled
        assert scheduled > 2, "sustained same-direction scrubbing should warm deeper ahead of the motion"
    finally:
        win.close()


def test_montage_prefetch_denial_does_not_cap_slice_prefetch(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.prefetch_policy import SliceScrubMomentum
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 16, dtype=float).reshape(3, 4, 16))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.prefetch_evaluation_controller.set_max_prefetch(32)

        # With no montage stage ready, the montage-prefetch decision denies
        # montage speculation. That local decision must not shrink the shared
        # prefetch controller and accidentally throttle ordinary slice prefetch.
        win._apply_resource_governor_decisions()
        assert win.prefetch_evaluation_controller._max_prefetch == 32

        momentum = SliceScrubMomentum()
        base = time.monotonic() - 0.3
        for step, index in enumerate((2, 3, 4, 5)):
            momentum.observe(index, now=base + 0.1 * step)
        win.renderer._prefetch_momentum = momentum
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 6), None)
        _process_events(qtbot, count=40)

        assert win.operation_evaluator.display_cache_diagnostics().prefetch_scheduled > 2
    finally:
        win.close()


def test_prefetch_skips_while_visible_controller_busy(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.core.work_graph import WorkItem, WorkLane
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win._active_slice_axis = 2
        win.visible_evaluation_controller.start_latest(
            lambda: time.sleep(0.08),
            key="busy",
            priority=0,
            replace_group="visible",
            frame_target=FrameTarget("busy", None, "presentation", "exact-visible"),
            supersession_key="visible-image",
            supersession_value="busy",
            work_item=WorkItem(
                key=("visible", "busy"),
                lane=WorkLane.VISIBLE_MATERIALIZATION,
                frame_target=FrameTarget("busy", None, "presentation", "exact-visible"),
                supersession_key="visible-image",
                supersession_value="busy",
            ),
            on_done=lambda _value: None,
        )
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=5)

        assert win.prefetch_evaluation_controller.diagnostics().prefetch_visible_busy_blocked == 1
    finally:
        win.close()


def test_cost_aware_prefetch_allows_cheap_operation_backed_stack(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True)
        win.request_operation("reverse", 0)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 2), None)
        _process_events(qtbot, count=40)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled > 0
        assert after.prefetch_stored > 0
    finally:
        win.close()


def test_cost_aware_prefetch_blocks_expensive_fft_stack(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((96, 96, 96), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = AppSettingsState(theme=win.app_settings.theme, prefetch_nearby_slices=True, render_memory_budget_mb=128)
        win.request_operation("centered_fft", 2)
        _process_events(qtbot, count=20)
        win._active_slice_axis = 2
        before = win.operation_evaluator.display_cache_diagnostics()
        before_blocked = win.prefetch_evaluation_controller.diagnostics().prefetch_cost_blocked
        win.renderer._prefetch_nearby_slices(win.view_state.with_slice(2, 1), None)
        _process_events(qtbot, count=20)
        after = win.operation_evaluator.display_cache_diagnostics()

        assert after.prefetch_scheduled == before.prefetch_scheduled
        assert win.prefetch_evaluation_controller.diagnostics().prefetch_cost_blocked > before_blocked
    finally:
        win.close()
