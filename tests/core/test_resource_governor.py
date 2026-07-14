from arrayscope.app.settings_state import AppSettingsState
from arrayscope.core.compute_policy import ComputeLane, compute_policy_from_settings
from arrayscope.core.gui_callback_budget import GuiCallbackObservation
from arrayscope.core.memory_policy import MemoryPolicy, MemoryProfileChoice, SystemMemorySnapshot, compute_memory_policy
from arrayscope.core.resource_governor import ResourceGovernor, SchedulerBusyState
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


def test_bridge_drain_knob_uses_profile_feedback_defaults():
    governor = ResourceGovernor(_policy(MemoryProfileChoice.AGGRESSIVE), profile=MemoryProfileChoice.AGGRESSIVE)
    governor.update_telemetry(_snapshot(_memory()), _memory())

    decision = governor.decide_bridge_drain(interactive=False)

    assert decision.channel == "kernel_bridge_drain"
    assert decision.batch_limit == 18
    assert decision.budget_ms == 11.0
    assert decision.byte_cap == 0


def test_commit_batch_knob_covers_last_observed_bytes_per_item():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED)
    governor.record_ui_observation(
        "presentation_commit",
        12.0,
        item_count=6,
        byte_count=6 * 1024 * 1024,
        work_class="presentation_upsert",
        backend="vispy",
    )

    decision = governor.decide_commit_batch(interactive=False)

    assert decision.channel == "presentation_commit"
    assert decision.batch_limit >= 1
    assert decision.byte_cap >= decision.batch_limit * 1024 * 1024
    assert decision.interval_ms == 0
    assert decision.model == "ewma"


def test_callback_observations_are_kept_without_decision_ring():
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
    diagnostics = governor.diagnostics()

    assert diagnostics.recent_over_warning_callbacks == (observation,)
    assert diagnostics.recent_ui_work_observations[-1] == observation
    assert not hasattr(diagnostics, "ui_decisions")
    assert not hasattr(diagnostics, "recent_ui_work_decisions")


def test_histogram_workers_park_only_behind_runnable_visible_work():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0)

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


def test_blocking_semantic_evidence_keeps_one_histogram_worker():
    governor = ResourceGovernor(_policy(), profile=MemoryProfileChoice.BALANCED, min_worker_update_interval_ms=0)

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
