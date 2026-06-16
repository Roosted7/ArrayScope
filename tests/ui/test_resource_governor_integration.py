import numpy as np

from tests.ui.helpers import clear_arrayscope_settings, process_events


def test_resource_governor_applies_worker_and_callback_limits(qtbot):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        win.resource_governor.min_worker_update_interval_ms = 0
        before = win.montage_tile_evaluation_controller.pool.maxThreadCount()

        win.resource_governor.record_ui_observation("montage_tile_result", 80.0, item_count=1)
        win._apply_resource_governor_decisions()

        after = win.montage_tile_evaluation_controller.pool.maxThreadCount()
        assert after <= before
        assert win.montage_tile_evaluation_controller._max_callback_dispatch_per_drain >= 1
        assert win.prefetch_evaluation_controller._max_prefetch >= 0
    finally:
        win.close()


def test_histogram_preview_interval_is_governor_controlled(qtbot):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        win.resource_governor.record_ui_observation("histogram_preview", 40.0, item_count=1)
        win._apply_resource_governor_decisions()

        controller = win.img_view._histogram_preview_controller
        assert controller.interval_ms >= 1
    finally:
        win.close()
