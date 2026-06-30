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
