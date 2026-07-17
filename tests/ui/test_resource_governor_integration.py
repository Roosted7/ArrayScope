import numpy as np
from types import SimpleNamespace

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

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


def test_interaction_edge_applies_budget_and_lane_quotas_immediately(qtbot):
    clear_arrayscope_settings()
    from arrayscope.kernel import Lane
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        qtbot.waitUntil(lambda: not win._interaction_active_now(), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        # A known drain cost separates the idle and interactive budgets.
        for _ in range(6):
            win.resource_governor.record_ui_observation("kernel_bridge_drain", 10.0, item_count=2)
        win._apply_resource_governor_decisions()
        idle_budget = win.kernel_bridge._budget_ms
        assert idle_budget is not None

        win.render_coordinator.request(reason="interaction-edge-test", interactive=True)
        interactive_budget = win.kernel_bridge._budget_ms
        assert win._governor_interactive_applied is True
        assert interactive_budget < idle_budget
        assert win.kernel._lane_quotas[Lane.DISPLAY_PREVIEW] == 1
        assert win.kernel._lane_quotas[Lane.DISPLAY_PREPARATION] == 0
    finally:
        win.close()


def test_interaction_stop_edge_restores_idle_budget_and_preparation_lane(qtbot):
    clear_arrayscope_settings()
    from arrayscope.kernel import Lane
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        qtbot.waitUntil(lambda: not win._interaction_active_now(), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        for _ in range(6):
            win.resource_governor.record_ui_observation("kernel_bridge_drain", 10.0, item_count=2)
        win._apply_resource_governor_decisions()
        idle_budget = win.kernel_bridge._budget_ms

        win.render_coordinator.request(reason="interaction-stop-edge-test", interactive=True)
        interactive_budget = win.kernel_bridge._budget_ms
        assert interactive_budget < idle_budget
        assert win._governor_interactive_applied is True
        assert win.kernel._lane_quotas[Lane.DISPLAY_PREPARATION] == 0

        win.render_coordinator._quiet_timer_elapsed()
        assert win._governor_interactive_applied is False
        assert win.kernel_bridge._budget_ms == idle_budget
        assert win.kernel._lane_quotas[Lane.DISPLAY_PREPARATION] > 0
    finally:
        win.close()


def test_interaction_stop_edge_replans_deferred_native_montage_quality(qtbot, monkeypatch):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    calls = []
    try:
        process_events(qtbot)
        qtbot.waitUntil(lambda: not win._interaction_active_now(), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        monkeypatch.setattr(
            win.renderer,
            "replan_deferred_interactive_native_quality",
            lambda: calls.append("replan") or True,
        )

        win._last_interaction_active_state = True
        win._note_interaction_state_changed()

        assert calls == ["replan"]
    finally:
        win.close()


def test_interaction_stop_native_replan_watermark_is_scoped_to_frame_session():
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    first_pipeline = SimpleNamespace(
        counters=SimpleNamespace(interactive_native_deferred=40),
    )
    second_pipeline = SimpleNamespace(
        counters=SimpleNamespace(interactive_native_deferred=1),
    )
    first_session = SimpleNamespace(
        session_id=1,
        pipeline=first_pipeline,
        _interactive_residency_deferred=False,
    )
    second_session = SimpleNamespace(
        session_id=2,
        pipeline=second_pipeline,
        _interactive_residency_deferred=False,
    )
    replanned = []
    owner = SimpleNamespace(
        _frame_session=first_session,
        _frame_session_is_current=lambda candidate: candidate is owner._frame_session,
        retarget_frame_pipeline=lambda session, **_kwargs: replanned.append(session),
    )
    owner.win = owner

    assert FrameRuntimeMixin.replan_deferred_interactive_native_quality(owner)
    owner._frame_session = second_session

    assert FrameRuntimeMixin.replan_deferred_interactive_native_quality(owner)
    assert replanned == [first_session, second_session]


def test_roi_lane_stays_parked_until_visible_first_pixels_are_physical(qtbot):
    clear_arrayscope_settings()
    from arrayscope.core.compute_policy import ComputeLane
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        process_events(qtbot)
        session = win.renderer._frame_session
        session.scheduling_policy.retarget(
            ("test-required-scope",),
            session.required_tile_numbers(),
            progressive=True,
        )

        busy = win._scheduler_busy_state()
        decision = win.resource_governor.decide_lane_workers(
            ComputeLane.ROI,
            interactive=False,
            busy_state=busy,
        )

        assert busy.visible_busy is True
        assert busy.montage_busy is True
        assert decision.target_workers == 0
        assert "inspection parked" in decision.reason
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
