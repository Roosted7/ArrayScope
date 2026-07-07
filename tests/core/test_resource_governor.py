from arrayscope.app.settings_state import AppSettingsState
from arrayscope.core.compute_policy import ComputeLane, compute_policy_from_settings
from arrayscope.core.gui_callback_budget import GuiCallbackObservation
from arrayscope.core.memory_policy import MemoryPolicy, MemoryProfileChoice, SystemMemorySnapshot, compute_memory_policy
from arrayscope.core.resource_governor import ResourceGovernor, ResourcePressure, SchedulerBusyState
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
        memory=SystemMemorySnapshot(memory.system_total_bytes, memory.system_available_bytes, memory.process_rss_bytes),
        cpu=CpuSnapshot(16, process_cpu_percent=20.0, system_cpu_percent=cpu_percent, load_average_1m=2.0, source="test"),
        timestamp_monotonic=0.0,
    )


def test_governor_profile_tuning_controls_batch_defaults():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.AGGRESSIVE), profile=MemoryProfileChoice.AGGRESSIVE)
    governor.update_telemetry(_snapshot(_memory()), _memory())

    decision = governor.decide_ui_work("montage_tile_result", interactive=False)

    assert decision.batch_limit == 18
    assert decision.budget_ms == 11.0


def test_vispy_presentation_starts_conservative_until_feedback():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.AGGRESSIVE), profile=MemoryProfileChoice.AGGRESSIVE)
    governor.update_telemetry(_snapshot(_memory()), _memory())

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit == governor.latency_feedback.tuning.max_batch


def test_tile_layer_commit_uses_presentation_upload_feedback_ramp():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.BALANCED), profile=MemoryProfileChoice.BALANCED)
    governor.update_telemetry(_snapshot(_memory()), _memory())

    governor.record_ui_observation("tile_layer_commit", 2.0, item_count=1, byte_count=4096)
    decision = governor.decide_ui_work("tile_layer_commit", interactive=False)

    assert decision.batch_limit > 1
    assert decision.byte_cap >= 4096 * decision.batch_limit
    assert decision.control_budget_ms >= decision.budget_ms
    assert decision.model in {"fallback", "overhead+marginal"}
    assert any("snapshot last=" in detail for detail in decision.details)


def test_tile_layer_commit_reset_can_start_conservative_until_feedback():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.BALANCED), profile=MemoryProfileChoice.BALANCED)
    governor.update_telemetry(_snapshot(_memory()), _memory())
    governor.record_ui_observation("tile_layer_commit", 2.0, item_count=1, byte_count=4096)
    assert governor.decide_ui_work("tile_layer_commit", interactive=False).batch_limit > 1

    governor.reset_ui_work_feedback("tile_layer_commit", conservative_start=True)
    initial = governor.decide_ui_work("tile_layer_commit", interactive=False)
    assert initial.batch_limit == governor.latency_feedback.tuning.min_batch

    governor.record_ui_observation(
        "tile_layer_commit",
        2.0,
        item_count=32,
        byte_count=32 * 4096,
        work_class="tile_layer_commit",
    )
    warmed = governor.decide_ui_work("tile_layer_commit", interactive=False)

    assert warmed.batch_limit <= governor.latency_feedback.tuning.min_batch + 1

    governor.record_ui_observation(
        "tile_layer_commit",
        0.5,
        item_count=32,
        byte_count=0,
        work_class="presentation_upsert",
    )
    still_guarded = governor.decide_ui_work("tile_layer_commit", interactive=False)

    assert still_guarded.batch_limit <= governor.latency_feedback.tuning.min_batch + 1


def test_tile_layer_zero_byte_presentation_observation_is_diagnostics_only():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.BALANCED), profile=MemoryProfileChoice.BALANCED)
    governor.update_telemetry(_snapshot(_memory()), _memory())
    governor.record_ui_observation(
        "tile_layer_commit",
        20.0,
        item_count=4,
        byte_count=4 * 1024 * 1024,
        work_class="tile_layer_commit",
        backend="pyqtgraph",
    )
    before = governor.decide_ui_work("tile_layer_commit", interactive=False)

    governor.record_gui_callback_observation(
        GuiCallbackObservation(
            channel="tile_layer_commit",
            work_class="presentation_upsert",
            backend="pyqtgraph",
            target_ms=8.0,
            warning_ms=16.0,
            item_cap=32,
            byte_cap=0,
            elapsed_ms=0.5,
            processed_items=32,
            processed_bytes=0,
        )
    )
    after = governor.decide_ui_work("tile_layer_commit", interactive=False)
    snapshot = governor.latency_feedback.channel_snapshot("tile_layer_commit")

    assert after.batch_limit == before.batch_limit
    assert snapshot.last_count == 4
    assert governor.diagnostics().recent_ui_work_observations[-1].processed_items == 32


def test_tile_layer_zero_byte_direct_observation_is_diagnostics_only():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.BALANCED), profile=MemoryProfileChoice.BALANCED)
    governor.update_telemetry(_snapshot(_memory()), _memory())
    governor.record_ui_observation(
        "tile_layer_commit",
        20.0,
        item_count=4,
        byte_count=4 * 1024 * 1024,
        work_class="tile_layer_commit",
        backend="pyqtgraph",
    )
    before = governor.decide_ui_work("tile_layer_commit", interactive=False)

    governor.record_ui_observation(
        "tile_layer_commit",
        0.5,
        item_count=32,
        byte_count=0,
        work_class="tile_layer_commit",
        backend="pyqtgraph",
    )
    after = governor.decide_ui_work("tile_layer_commit", interactive=False)
    snapshot = governor.latency_feedback.channel_snapshot("tile_layer_commit")

    assert after.batch_limit == before.batch_limit
    assert snapshot.last_count == 4


def test_presentation_single_tile_gray_zone_explores_larger_upload_batch():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.BALANCED), profile=MemoryProfileChoice.BALANCED)
    governor.update_telemetry(_snapshot(_memory()), _memory())

    governor.record_ui_observation(
        "tile_layer_commit",
        17.0,
        item_count=1,
        byte_count=451_584,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("tile_layer_commit", interactive=False)

    assert decision.batch_limit > 1
    assert decision.byte_cap >= 451_584 * decision.batch_limit


def test_ui_pressure_reduces_batch_and_workers():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    governor.record_ui_observation("montage_tile_result", 40.0, item_count=1)

    ui_decision = governor.decide_ui_work("montage_tile_result", interactive=False)
    worker_decision = governor.decide_lane_workers(ComputeLane.MONTAGE_TILE, interactive=False, busy_state=SchedulerBusyState(stage_ready_or_in_flight=True))

    assert governor.diagnostics().pressure.ui_pressure == ResourcePressure.HIGH
    assert ui_decision.batch_limit < 12
    assert worker_decision.target_workers < worker_decision.max_workers


def test_governor_retains_over_warning_callback_observation_details():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    observation = GuiCallbackObservation(
        channel="montage_commit",
        work_class="tile_upsert",
        backend="vispy",
        target_ms=4.0,
        warning_ms=16.0,
        item_cap=12,
        byte_cap=8 * 1024 * 1024,
        elapsed_ms=18.0,
        processed_items=3,
        processed_bytes=4096,
    )

    governor.record_gui_callback_observation(observation)

    callbacks = governor.diagnostics().recent_over_warning_callbacks
    assert callbacks == (observation,)
    channel = governor.diagnostics().feedback_channels[0]
    assert channel.last_count == 3
    assert channel.last_byte_count == 4096


def test_governor_byte_observations_reduce_byte_cap():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_commit",
        16.0,
        item_count=1,
        byte_count=64 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_commit", interactive=False)

    assert 0 < decision.byte_cap < 32 * 1024 * 1024


def test_presentation_byte_cap_covers_decided_batch_items():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        12.0,
        item_count=6,
        byte_count=6 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.byte_cap >= decision.batch_limit * 1024 * 1024


def test_presentation_over_budget_sample_backs_off_next_decision_immediately():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.AGGRESSIVE), profile=MemoryProfileChoice.AGGRESSIVE)
    governor.record_ui_observation(
        "montage_present_total",
        40.0,
        item_count=20,
        byte_count=20 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit <= 13
    assert decision.byte_cap <= 14 * 1024 * 1024


def test_presentation_over_budget_sample_scales_from_measured_cost_not_warning_threshold():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        12.0,
        item_count=12,
        byte_count=12 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit > 12
    assert decision.byte_cap >= 12 * 1024 * 1024


def test_presentation_probe_above_profile_cap_is_not_blocked_by_unrelated_ui_pressure():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        12.0,
        item_count=12,
        byte_count=12 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )
    governor.record_ui_observation(
        "montage_commit",
        80.0,
        item_count=12,
        byte_count=0,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert governor.diagnostics().pressure.ui_pressure == ResourcePressure.HIGH
    assert decision.batch_limit > 12
    assert decision.reason == "feedback target"


def test_interaction_state_not_global_pressure_controls_idle_presentation_width():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        12.0,
        item_count=12,
        byte_count=12 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )
    governor.record_ui_observation(
        "montage_commit",
        80.0,
        item_count=12,
        byte_count=0,
        work_class="presentation_upsert",
        backend="vispy",
    )

    idle = governor.decide_ui_work("montage_present_total", interactive=False)
    interactive = governor.decide_ui_work("montage_present_total", interactive=True)

    assert idle.batch_limit > interactive.batch_limit
    assert idle.reason == "feedback target"
    assert interactive.reason == "interactive feedback target"


def test_presentation_under_warning_sample_recovers_from_single_item_limit():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        40.0,
        item_count=12,
        byte_count=12 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )
    assert governor.decide_ui_work("montage_present_total", interactive=False).batch_limit == 7

    governor.record_ui_observation(
        "montage_present_total",
        7.0,
        item_count=4,
        byte_count=4 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit > 7
    assert decision.byte_cap > 4 * 1024 * 1024


def test_presentation_single_item_fast_sample_recovers_from_sticky_min_batch():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "tile_layer_commit",
        80.0,
        item_count=12,
        byte_count=12 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="pyqtgraph",
    )
    assert governor.decide_ui_work("tile_layer_commit", interactive=False).batch_limit <= 7

    governor.record_ui_observation(
        "tile_layer_commit",
        2.0,
        item_count=1,
        byte_count=256 * 1024,
        work_class="presentation_upsert",
        backend="pyqtgraph",
    )

    decision = governor.decide_ui_work("tile_layer_commit", interactive=False)

    assert decision.batch_limit >= 3
    assert decision.byte_cap >= 256 * 1024 * decision.batch_limit


def test_presentation_feedback_records_but_filters_isolated_outlier():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        10.0,
        item_count=10,
        byte_count=10 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )
    before = governor.decide_ui_work("montage_present_total", interactive=False)

    governor.record_ui_observation(
        "montage_present_total",
        90.0,
        item_count=10,
        byte_count=10 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    after = governor.decide_ui_work("montage_present_total", interactive=False)
    raw = governor.diagnostics().recent_ui_work_observations[-1]

    assert raw.elapsed_ms == 90.0
    assert after.batch_limit >= before.batch_limit // 2


def test_conservative_presentation_cold_start_releases_on_real_upload_work():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.reset_ui_work_feedback("montage_present_total", conservative_start=True)

    first = governor.decide_ui_work("montage_present_total", interactive=False)
    assert first.batch_limit == 1

    for _ in range(4):
        governor.record_ui_observation(
            "montage_present_total",
            2.0,
            item_count=1,
            byte_count=64 * 1024,
            work_class="presentation_upsert",
            backend="vispy",
        )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit > 1
    assert not any("conservative cold start" in detail for detail in decision.details)


def test_conservative_cold_start_ignores_zero_byte_visibility_work():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.reset_ui_work_feedback("montage_present_total", conservative_start=True)

    for _ in range(4):
        governor.record_ui_observation(
            "montage_present_total",
            0.1,
            item_count=1,
            byte_count=0,
            work_class="presentation_upsert",
            backend="vispy",
        )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit == 1
    assert any("conservative cold start" in detail for detail in decision.details)


def test_repeated_presentation_outliers_are_learned():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "montage_present_total",
        10.0,
        item_count=10,
        byte_count=10 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )
    governor.record_ui_observation(
        "montage_present_total",
        90.0,
        item_count=10,
        byte_count=10 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )
    governor.record_ui_observation(
        "montage_present_total",
        95.0,
        item_count=10,
        byte_count=10 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_present_total", interactive=False)

    assert decision.batch_limit < 10


def test_upload_telemetry_does_not_drive_global_ui_pressure():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    governor.record_ui_observation(
        "montage_present_total",
        60.0,
        item_count=16,
        byte_count=64 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_ui_work("montage_commit", interactive=False)

    assert governor.diagnostics().pressure.ui_pressure == ResourcePressure.NORMAL
    assert decision.batch_limit == 12
    assert decision.interval_ms <= 16
    assert any(channel.channel == "montage_present_total" for channel in governor.diagnostics().feedback_channels)


def test_priority_retarget_metadata_does_not_drive_global_ui_pressure():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    governor.record_ui_observation(
        "montage_priority_retarget",
        60.0,
        item_count=8,
        work_class="queue_metadata",
        backend="qt",
    )

    decision = governor.decide_ui_work("montage_commit", interactive=False)

    assert governor.diagnostics().pressure.ui_pressure == ResourcePressure.NORMAL
    assert decision.batch_limit == 12
    assert any(channel.channel == "montage_priority_retarget" for channel in governor.diagnostics().feedback_channels)


def test_elevated_ui_pressure_preserves_stage_backed_tile_workers():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory, cpu_percent=9.0), memory)
    governor.record_ui_observation("montage_commit", 13.0, item_count=1)

    ui_decision = governor.decide_ui_work("montage_tile_result", interactive=False)
    worker_decision = governor.decide_lane_workers(
        ComputeLane.MONTAGE_TILE,
        interactive=False,
        busy_state=SchedulerBusyState(stage_ready_or_in_flight=True, result_backlog=0),
    )

    assert governor.diagnostics().pressure.ui_pressure == ResourcePressure.ELEVATED
    assert ui_decision.batch_limit >= 4
    assert worker_decision.target_workers == worker_decision.max_workers


def test_worker_recovery_is_bounded_but_not_sticky():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0, max_worker_step=2)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory, cpu_percent=9.0), memory)
    governor._lane_targets[ComputeLane.MONTAGE_TILE] = 1

    decision = governor.decide_lane_workers(
        ComputeLane.MONTAGE_TILE,
        interactive=False,
        busy_state=SchedulerBusyState(stage_ready_or_in_flight=True, result_backlog=0),
    )

    assert decision.target_workers == 3


def test_memory_pressure_disables_prefetch_first():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory(available_fraction=0.05)
    governor.update_telemetry(_snapshot(memory), memory)

    decision = governor.decide_montage_prefetch(stage_ready_or_in_flight=True, visible_busy=False)

    assert not decision.allowed
    assert "memory" in decision.reason


def test_unrelated_ui_pressure_does_not_disable_idle_montage_prefetch():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    governor.record_ui_observation("montage_commit", 80.0, item_count=1)

    decision = governor.decide_montage_prefetch(stage_ready_or_in_flight=True, visible_busy=False)

    assert governor.diagnostics().pressure.ui_pressure == ResourcePressure.HIGH
    assert decision.allowed
    assert decision.reason == "stage ready and idle"


def test_tile_worker_product_guard_still_holds():
    policy = _policy(MemoryProfileChoice.AGGRESSIVE)

    assert policy.fft_workers_tile == 1
    assert policy.montage_tile_workers * policy.fft_workers_tile <= 14


def test_result_drain_feedback_filters_isolated_outlier():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    for _ in range(6):
        governor.record_ui_observation("visible_queue_drain", 2.0, item_count=4)
    healthy = governor.decide_ui_work("visible_queue_drain", interactive=False)

    # One GC pause / incidental relayout should not collapse the batch.
    governor.record_ui_observation("visible_queue_drain", 60.0, item_count=4)
    after_spike = governor.decide_ui_work("visible_queue_drain", interactive=False)

    assert healthy.batch_limit > 4
    # Without suppression the spike drives the per-item EWMA above the whole
    # budget and the batch collapses to the minimum.
    assert after_spike.batch_limit > 4


def test_result_drain_repeated_slow_samples_are_learned():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    for _ in range(6):
        governor.record_ui_observation("visible_queue_drain", 2.0, item_count=4)
    healthy = governor.decide_ui_work("visible_queue_drain", interactive=False)

    governor.record_ui_observation("visible_queue_drain", 60.0, item_count=4)
    governor.record_ui_observation("visible_queue_drain", 60.0, item_count=4)
    governor.record_ui_observation("visible_queue_drain", 60.0, item_count=4)
    after_streak = governor.decide_ui_work("visible_queue_drain", interactive=False)

    assert after_streak.batch_limit < healthy.batch_limit


def test_result_drain_under_budget_batch_recovers_from_measured_rate():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    memory = _memory()
    governor.update_telemetry(_snapshot(memory), memory)
    # Learned slow: repeated genuinely slow drains pin the batch low.
    for _ in range(8):
        governor.record_ui_observation("montage_queue_drain", 40.0, item_count=4)
    slow = governor.decide_ui_work("montage_queue_drain", interactive=False)
    assert slow.batch_limit <= 2

    # Real drains now complete far under budget while hitting their cap;
    # the next decision must grow from the measured rate instead of waiting
    # for the EWMA to decay one small drain at a time.
    governor.record_ui_observation("montage_queue_drain", 1.0, item_count=max(1, slow.batch_limit))
    recovered = governor.decide_ui_work("montage_queue_drain", interactive=False)

    assert recovered.batch_limit > slow.batch_limit


def test_presentation_batches_escape_fixed_overhead_death_spiral():
    # Replays the observed montage fill pathology: commits cost ~15 ms fixed
    # (level sync, histogram, presentation build) plus ~1 ms per tile. With
    # per-item EWMAs, a 2-tile commit measures ~8.5 ms/item, the batch shrinks
    # to 1-2, and every subsequent small commit looks over budget, pinning the
    # whole fill at a trickle. The overhead/marginal model must recover a
    # batch that amortizes the fixed cost.
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)

    def commit(count):
        governor.record_ui_observation("montage_present_total", 15.0 + 1.0 * count, item_count=count)
        return governor.decide_ui_work("montage_present_total", interactive=False).batch_limit

    batches = [commit(2), commit(1), commit(2), commit(1)]
    # Once counts have varied, the model kicks in; drive a few more commits
    # using whatever batch the governor allows.
    for _ in range(6):
        batches.append(commit(max(1, batches[-1])))

    # Balanced idle control budget for presentation channels is
    # max(budget, 16+8) = 24 ms; with 15 ms overhead and 1 ms marginal the
    # sustainable batch is ~9, not 1-2.
    assert batches[-1] >= 6, batches
    # And it converges rather than oscillating back to a trickle.
    assert min(batches[-3:]) >= 6, batches


def test_presentation_model_still_keeps_genuinely_expensive_items_small():
    # 12 ms per item with no meaningful fixed cost: the model must not
    # inflate batches (24 ms control budget / 12 ms per item = 2).
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    for count in (1, 3, 2, 4, 1, 3, 2, 4):
        governor.record_ui_observation("montage_present_total", 12.0 * count, item_count=count)
    decision = governor.decide_ui_work("montage_present_total", interactive=False)
    assert decision.batch_limit <= 3, decision


def test_presentation_byte_cap_escapes_overhead_inflated_per_byte_rate():
    # 900 KB tiles at ~2.5 ms marginal upload cost plus ~15 ms fixed commit
    # overhead: the naive per-byte EWMA folds the overhead into the byte rate
    # and caps uploads at a fraction of what the budget sustains, which
    # throttled montage presentation mid-fill.
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    tile = 900 * 1024
    for count in (2, 5, 3, 7, 2, 6, 4, 7, 3, 5):
        governor.record_ui_observation(
            "montage_present_total", 15.0 + 2.5 * count, item_count=count, byte_count=tile * count
        )
    decision = governor.decide_ui_work("montage_present_total", interactive=False)
    # Non-interactive amortization allows ~2x overhead of marginal work:
    # 2*15ms / (2.5ms/tile) = 12 tiles ~= 10.8 MB. The inflated per-byte rate
    # would have allowed only ~4-6 tiles.
    assert decision.byte_cap >= 8 * tile, decision
