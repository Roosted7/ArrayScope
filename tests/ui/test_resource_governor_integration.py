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

        win.resource_governor.record_ui_observation("montage_tile_result", 80.0, item_count=1)
        win._apply_resource_governor_decisions()

        after = win.montage_tile_evaluation_controller.diagnostics().max_workers
        decision = next(
            lane
            for lane in win.resource_governor.diagnostics().lane_decisions
            if lane.lane.value == "montage_tile"
        )
        assert after == decision.target_workers
        assert decision.min_workers <= after <= decision.max_workers
        assert win.kernel_bridge._max_items_per_drain >= 1
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


def test_interaction_edge_applies_interactive_budgets_immediately(qtbot):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        qtbot.waitUntil(lambda: not win._interaction_active_now(), timeout=5000)
        # A known drain cost separates the idle and interactive budgets.
        for _ in range(6):
            win.resource_governor.record_ui_observation("kernel_bridge_drain", 10.0, item_count=2)
        win._apply_resource_governor_decisions()
        idle_budget = win.kernel_bridge._budget_ms
        assert idle_budget is not None

        # The sampling timer runs at 250 ms (1 s idle); an interactive
        # request must not run against idle budgets until the next tick.
        win.render_coordinator.request(reason="interaction-edge-test", interactive=True)

        interactive_budget = win.kernel_bridge._budget_ms
        assert win._governor_interactive_applied is True
        assert interactive_budget < idle_budget
    finally:
        win.close()


def test_interaction_stop_edge_restores_idle_budgets_immediately(qtbot):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        qtbot.waitUntil(lambda: not win._interaction_active_now(), timeout=5000)
        for _ in range(6):
            win.resource_governor.record_ui_observation("kernel_bridge_drain", 10.0, item_count=2)
        win._apply_resource_governor_decisions()
        idle_budget = win.kernel_bridge._budget_ms

        win.render_coordinator.request(reason="interaction-stop-edge-test", interactive=True)
        interactive_budget = win.kernel_bridge._budget_ms
        assert interactive_budget < idle_budget

        win.render_coordinator._quiet_timer_elapsed()

        assert win._governor_interactive_applied is False
        assert win.kernel_bridge._budget_ms == idle_budget
    finally:
        win.close()


def test_runtime_diagnostics_lists_kernel_bridge_drain_channel(qtbot):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        channels = {
            channel.channel
            for channel in win.collect_runtime_diagnostics().resource_governor.feedback_channels
        }

        assert "kernel_bridge_drain" in channels
    finally:
        win.close()
