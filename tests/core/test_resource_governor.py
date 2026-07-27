from arrayscope.app.settings_state import AppSettingsState
from arrayscope.core.compute_policy import ComputeLane, compute_policy_from_settings
from arrayscope.core.gui_callback_budget import GuiCallbackObservation
from arrayscope.core.memory_policy import (
    MemoryPolicy,
    MemoryProfileChoice,
    SystemMemorySnapshot,
    compute_memory_policy,
)
from arrayscope.core.resource_governor import (
    ResourceGovernor,
    SchedulerBusyState,
    _render_pass_extrapolation_cost_ms,
    _render_pass_latency_cost_ms,
)
from arrayscope.core.resource_telemetry import CpuSnapshot, ResourceSnapshot


def _policy(profile=MemoryProfileChoice.BALANCED):
    return compute_policy_from_settings(AppSettingsState(memory_profile=profile), cpu_count=16)


def _memory(available_fraction=0.5) -> MemoryPolicy:
    total = 16 * 1024**3
    return compute_memory_policy(
        profile=MemoryProfileChoice.BALANCED,
        render_cap_mb=512,
        input_nbytes=128,
        system=SystemMemorySnapshot(total, int(total * available_fraction), 100),
    )


def _snapshot(memory: MemoryPolicy, cpu_percent=25.0) -> ResourceSnapshot:
    return ResourceSnapshot(
        memory=SystemMemorySnapshot(
            memory.system_total_bytes, memory.system_available_bytes, memory.process_rss_bytes
        ),
        cpu=CpuSnapshot(
            16,
            process_cpu_percent=20.0,
            system_cpu_percent=cpu_percent,
            load_average_1m=2.0,
            source="test",
        ),
        timestamp_monotonic=0.0,
    )


def test_bridge_drain_knob_uses_profile_feedback_defaults():
    governor = ResourceGovernor(
        _policy(MemoryProfileChoice.AGGRESSIVE), profile=MemoryProfileChoice.AGGRESSIVE
    )
    governor.update_telemetry(_snapshot(_memory()), _memory())

    decision = governor.decide_bridge_drain(interactive=False)

    assert decision.channel == "kernel_bridge_drain"
    assert decision.batch_limit == 18
    assert decision.budget_ms == 11.0
    assert decision.byte_cap == 0


def test_commit_batch_knob_covers_last_observed_bytes_per_item():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        12.0,
        item_count=6,
        byte_count=6 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    decision = governor.decide_commit_batch(interactive=False)

    assert decision.channel == "montage_present_total"
    assert decision.batch_limit >= 1
    assert decision.byte_cap >= decision.batch_limit * 1024 * 1024
    assert decision.interval_ms == 0
    assert decision.model == "ewma"


def test_render_pass_knob_owns_r5_target_and_adapts_per_item_work():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    cold = governor.decide_render_pass(interactive=False)
    governor.record_ui_observation(
        "montage_render_pass_preview",
        80.0,
        item_count=8,
        byte_count=8 * 1024,
        work_class="presentation_upsert",
        backend="pyqtgraph",
    )
    adapted = governor.decide_render_pass(interactive=False)

    assert cold.channel == "montage_render_pass_preview"
    assert cold.batch_limit == 1
    assert cold.budget_ms == 32.0
    assert adapted.batch_limit == 4
    assert adapted.budget_ms == 32.0
    assert adapted.model == "r5-feedback"


def test_render_pass_knob_still_shrinks_per_item_work_after_overrun():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_target",
        80.0,
        item_count=8,
        byte_count=8 * 1024,
        work_class="presentation_upsert",
        backend="pyqtgraph",
    )

    adapted = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
    )

    assert adapted.batch_limit == 4
    assert adapted.model == "r5-feedback"


def test_render_pass_latches_item_independent_cost_and_recovers_floor():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_preview",
        41.64,
        item_count=2,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    assert governor.decide_render_pass(interactive=False).batch_limit == 1
    governor.record_ui_observation(
        "montage_render_pass_preview",
        41.48,
        item_count=1,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    recovered = governor.decide_render_pass(interactive=False, remaining_items=272)

    assert recovered.batch_limit > 2
    assert any("fixed=" in detail and "item=" in detail for detail in recovered.details)
    governor.record_ui_observation(
        "montage_render_pass_preview",
        47.11,
        item_count=2,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    assert governor.decide_render_pass(interactive=False, remaining_items=272).batch_limit >= 2


def test_render_pass_splits_identifiable_fixed_item_and_byte_costs():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    observations = (
        (1, 1, 15.0),
        (2, 1, 17.0),
        (1, 2, 18.0),
        (2, 2, 20.0),
    )
    for items, mib, elapsed_ms in observations:
        governor.record_ui_observation(
            "montage_render_pass_preview",
            elapsed_ms,
            item_count=items,
            byte_count=mib * 1024 * 1024,
            work_class="presentation_upsert",
            backend="wgpu",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            remaining_items=272,
        )

    assert any(
        "fixed=10.00ms item=2.00ms bytes=3.00ms/MiB" in detail for detail in decision.details
    )
    assert any("fit-rms=0.00ms r2=1.000 samples=4" in detail for detail in decision.details)


def test_render_pass_measures_rising_cohort_cost():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    for items in (1, 4, 8):
        governor.record_ui_observation(
            "montage_render_pass_target",
            10.0 + items + 0.05 * items * items,
            item_count=items,
            work_class="presentation_upsert",
            backend="pyqtgraph",
        )
        decision = governor.decide_render_pass(
            interactive=False,
            pass_kind="target",
            remaining_items=272,
        )

    assert any("cohort2=0.05ms/item2" in detail for detail in decision.details)
    assert decision.batch_limit < 32


def test_render_pass_weights_fill_time_against_continuous_latency_cost():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_target",
        50.0,
        item_count=1,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )
    governor.record_ui_observation(
        "montage_render_pass_target",
        50.1,
        item_count=2,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    decision = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )

    # 50.0 ms at one item and 50.1 ms at two is an almost entirely fixed cost:
    # 49.9 ms fixed, 0.1 ms per item. With 272 remaining, the cohort that
    # actually minimizes the objective is ~136 items at ~63 ms per chunk --
    # deliberately over the 50 ms report line, because two 63 ms chunks beat
    # 272 single-item chunks of 50 ms each by two orders of magnitude of fill
    # time. Collapsing toward 1 here is the pathology, not the safe choice.
    #
    # Evidence only extends to two items, though, so the extrapolation term
    # must still hold the step short of that informed optimum: the governor
    # explores outward across successive decisions rather than trusting a
    # model fitted on two points at 68x the range it has measured.
    assert decision.batch_limit > 3, "fixed-dominated cost must not collapse the cohort"
    assert decision.batch_limit < 136, (
        "a two-point model must not jump straight to the informed optimum"
    )
    assert any(
        "steering=weighted-fill-latency-extrapolation" in detail for detail in decision.details
    )
    assert any("r5-achievable=1" in detail for detail in decision.details)

    # Widening the evidence must move the choice toward that optimum, not away.
    governor.record_ui_observation(
        "montage_render_pass_target",
        50.0 + 0.1 * decision.batch_limit,
        item_count=decision.batch_limit,
        work_class="presentation_upsert",
        backend="wgpu",
    )
    wider = governor.decide_render_pass(
        interactive=False,
        pass_kind="target",
        remaining_items=272,
    )
    assert wider.batch_limit > decision.batch_limit, (
        "the cohort must ratchet outward as the cost model gains range"
    )


def test_render_pass_latency_cost_is_zero_then_smooth_and_quadratic():
    assert _render_pass_latency_cost_ms(18.0) == 0.0
    assert 0.0 < _render_pass_latency_cost_ms(30.0) < _render_pass_latency_cost_ms(45.0)
    epsilon = 1e-4
    left_slope = (
        _render_pass_latency_cost_ms(45.0) - _render_pass_latency_cost_ms(45.0 - epsilon)
    ) / epsilon
    right_slope = (
        _render_pass_latency_cost_ms(45.0 + epsilon) - _render_pass_latency_cost_ms(45.0)
    ) / epsilon
    assert abs(left_slope - right_slope) < 1e-3
    assert _render_pass_latency_cost_ms(60.0) - _render_pass_latency_cost_ms(
        50.0
    ) > _render_pass_latency_cost_ms(50.0) - _render_pass_latency_cost_ms(45.0)


def test_render_pass_extrapolation_cost_is_smoothly_exponential():
    assert _render_pass_extrapolation_cost_ms(1.1) == 0.0
    at_two = _render_pass_extrapolation_cost_ms(2.0)
    at_ten = _render_pass_extrapolation_cost_ms(10.0)
    at_hundred = _render_pass_extrapolation_cost_ms(100.0)
    assert 0.0 < at_two < 5.0
    assert at_ten > 50.0 * at_two
    assert at_hundred > 50.0 * at_ten


def test_render_pass_requirement_does_not_adapt_downward():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_render_pass_target",
        92.0,
        item_count=1,
        work_class="presentation_upsert",
        backend="wgpu",
    )

    diagnostics = governor.diagnostics(channels=("montage_render_pass_target",))

    assert diagnostics.feedback_channels[0].budget_ms == 32.0
    assert diagnostics.recent_ui_work_observations[-1].target_ms == 32.0


def test_callback_observations_are_kept_without_decision_ring():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    observation = GuiCallbackObservation(
        channel="montage_commit",
        work_class="tile_upsert",
        backend="wgpu",
        target_ms=4.0,
        warning_ms=16.0,
        item_cap=12,
        byte_cap=8 * 1024 * 1024,
        elapsed_ms=18.0,
        processed_items=3,
        processed_bytes=4096,
    )

    governor.record_gui_callback_observation(observation)
    diagnostics = governor.diagnostics()

    assert diagnostics.recent_over_warning_callbacks == (observation,)
    assert diagnostics.recent_ui_work_observations[-1] == observation
    assert not hasattr(diagnostics, "ui_decisions")
    assert not hasattr(diagnostics, "recent_ui_work_decisions")


def test_histogram_workers_park_only_behind_runnable_visible_work():
    governor = ResourceGovernor(
        _policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0
    )

    busy = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=True,
        busy_state=SchedulerBusyState(montage_busy=True, result_backlog=2),
    )
    between_bursts = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=True,
        busy_state=SchedulerBusyState(),
    )

    assert busy.target_workers == 0
    assert busy.min_workers == 0
    assert "runnable user-visible rendering" in busy.reason
    assert between_bursts.target_workers == between_bursts.max_workers

    bookkeeping_only = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=False,
        busy_state=SchedulerBusyState(result_backlog=2),
    )
    assert bookkeeping_only.target_workers == bookkeeping_only.max_workers


def test_interaction_immediately_parks_side_lanes_and_limits_montage():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    busy = SchedulerBusyState(montage_busy=True)

    montage = governor.decide_lane_workers(
        ComputeLane.MONTAGE_TILE,
        interactive=True,
        busy_state=busy,
    )
    prefetch = governor.decide_lane_workers(
        ComputeLane.PREFETCH,
        interactive=True,
        busy_state=busy,
    )
    profile = governor.decide_lane_workers(
        ComputeLane.PROFILE,
        interactive=True,
        busy_state=busy,
    )

    assert montage.target_workers == 2
    assert "bounded montage workers" in montage.reason
    assert prefetch.target_workers == 0
    assert profile.target_workers == 0


def test_first_lane_decision_starts_at_policy_target_without_damping_from_maximum():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    decision = governor.decide_lane_workers(
        ComputeLane.PREFETCH,
        interactive=True,
        busy_state=SchedulerBusyState(visible_busy=True),
    )

    assert decision.target_workers == 0


def test_blocking_semantic_evidence_keeps_one_histogram_worker():
    governor = ResourceGovernor(
        _policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0
    )

    decision = governor.decide_lane_workers(
        ComputeLane.HISTOGRAM,
        interactive=True,
        busy_state=SchedulerBusyState(
            visible_busy=True,
            result_backlog=2,
            semantic_evidence_blocking=True,
        ),
    )

    assert decision.target_workers == 1
    assert decision.min_workers == 1
    assert "blocks first presentation" in decision.reason


def test_tile_worker_product_guard_still_holds():
    policy = _policy(MemoryProfileChoice.AGGRESSIVE)

    assert policy.fft_workers_tile == 1
    assert policy.montage_tile_workers * policy.fft_workers_tile <= 14
