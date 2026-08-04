from types import SimpleNamespace

import numpy as np

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import clear_arrayscope_settings
from tests.ui.helpers import (
    settle_idle as _settle_idle,
)


def test_resource_governor_applies_worker_and_callback_limits(qtbot):
    clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _settle_idle(win)
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
        _settle_idle(win)
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
        _settle_idle(win)
        qtbot.waitUntil(
            lambda: not win._interaction_active_now(), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
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
        _settle_idle(win)
        qtbot.waitUntil(
            lambda: not win._interaction_active_now(), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
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
        _settle_idle(win)
        qtbot.waitUntil(
            lambda: not win._interaction_active_now(), timeout=INTERACTION_SETTLE_HARD_LIMIT_MS
        )
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
    from arrayscope.render.progressive_scheduling import RoundLoadShedding
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
        round_load_shedding=RoundLoadShedding(),
        _interactive_residency_deferred=False,
    )
    second_session = SimpleNamespace(
        session_id=2,
        pipeline=second_pipeline,
        round_load_shedding=RoundLoadShedding(),
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


def test_interaction_stop_rearms_deferred_wgpu_histogram_evidence():
    from arrayscope.render.progressive_scheduling import RoundLoadShedding
    from arrayscope.window.frame_runtime import FrameRuntimeMixin

    pipeline = SimpleNamespace(
        counters=SimpleNamespace(interactive_native_deferred=0),
    )
    acknowledged = {3: object(), 7: object()}
    session = SimpleNamespace(
        session_id=1,
        pipeline=pipeline,
        round_load_shedding=RoundLoadShedding(),
        _interactive_residency_deferred=False,
        _wgpu_histogram_evidence_deferred=True,
        tile_presentation_state=SimpleNamespace(payloads=acknowledged),
    )
    replanned = []
    pumped = []
    owner = SimpleNamespace(
        _frame_session=session,
        _frame_session_is_current=lambda candidate: candidate is session,
        retarget_frame_pipeline=lambda current, **kwargs: replanned.append((current, kwargs)),
        _queue_montage_level_stats_for_payloads=lambda current, payloads: pumped.append(
            (current, payloads)
        ),
    )
    owner.win = owner

    assert FrameRuntimeMixin.replan_deferred_interactive_native_quality(owner)
    assert session._wgpu_histogram_evidence_deferred is False
    # The evidence queue is pumped directly at the settle edge: the forced
    # commit alone cannot re-own a deferral at the fill tail (no further
    # backend acknowledgements arrive while the coverage barrier is latched).
    assert pumped == [(session, acknowledged)]
    assert replanned == [(session, {"force_commit": True})]


def test_roi_lane_stays_parked_until_visible_first_pixels_are_physical(qtbot):
    clear_arrayscope_settings()
    from arrayscope.core.compute_policy import ComputeLane
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _settle_idle(win)
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
        _settle_idle(win)
        channels = {
            channel.channel
            for channel in win.collect_runtime_diagnostics().resource_governor.feedback_channels
        }

        assert "kernel_bridge_drain" in channels
    finally:
        win.close()


def test_hot_path_reconcile_resamples_telemetry_on_the_cadence(qtbot, monkeypatch):
    # Every hot-path caller of `_apply_resource_governor_decisions` passes
    # `refresh_telemetry=False`, and the only unconditional refresher is a
    # settings change. Without a cadence the governor's host telemetry stays
    # frozen at the value sampled during window build for the whole session,
    # which pins `cpu_headroom` at the neutral 0.5 forever.
    from time import perf_counter

    from arrayscope.window import ArrayScopeWindow
    from arrayscope.window import main as window_main

    clear_arrayscope_settings()
    win = ArrayScopeWindow(np.zeros((8, 8, 4), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _settle_idle(win)
        samples: list[int] = []
        real_sampler = window_main.sample_resource_snapshot

        def counting_sampler():
            samples.append(1)
            return real_sampler()

        monkeypatch.setattr(window_main, "sample_resource_snapshot", counting_sampler)

        # Fresh reading: the hot path must not pay for a resample.
        win._last_resource_telemetry_sample = perf_counter()
        win._apply_resource_governor_decisions(refresh_telemetry=False)
        assert samples == []

        # Older than the cadence: the same hot-path call must resample.
        win._last_resource_telemetry_sample = perf_counter() - (
            window_main._RESOURCE_TELEMETRY_REFRESH_SECONDS + 0.05
        )
        win._apply_resource_governor_decisions(refresh_telemetry=False)
        assert len(samples) == 1
    finally:
        win.close()
