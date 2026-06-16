from arrayscope.app.settings_state import AppSettingsState
from arrayscope.core.compute_policy import ComputeLane, compute_policy_from_settings
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


def test_tile_worker_product_guard_still_holds():
    policy = _policy(MemoryProfileChoice.AGGRESSIVE)

    assert policy.fft_workers_tile == 1
    assert policy.montage_tile_workers * policy.fft_workers_tile <= 14
