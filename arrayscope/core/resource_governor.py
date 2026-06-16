"""Pure adaptive resource governor for scheduling and UI fan-in."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic

from arrayscope.core.compute_policy import ComputeLane, ComputePolicy
from arrayscope.core.latency_feedback import LatencyFeedbackController, LatencyFeedbackTuning
from arrayscope.core.memory_policy import MemoryPolicy, MemoryProfileChoice, normalize_memory_profile_choice
from arrayscope.core.resource_telemetry import ResourceSnapshot


class ResourcePressure(Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"


@dataclass(frozen=True)
class SchedulerBusyState:
    visible_busy: bool = False
    montage_busy: bool = False
    stage_busy: bool = False
    prefetch_busy: bool = False
    queued_callbacks: int = 0
    result_backlog: int = 0
    stage_ready_or_in_flight: bool = False


@dataclass(frozen=True)
class ResourcePressureState:
    ui_pressure: ResourcePressure
    cpu_headroom: float
    memory_pressure: ResourcePressure
    cache_pressure: ResourcePressure
    reason: str


@dataclass(frozen=True)
class LaneWorkerDecision:
    lane: ComputeLane
    target_workers: int
    min_workers: int
    max_workers: int
    reason: str


@dataclass(frozen=True)
class UiWorkDecision:
    channel: str
    batch_limit: int
    budget_ms: float
    interval_ms: int
    reason: str


@dataclass(frozen=True)
class PrefetchAdmissionDecision:
    kind: str
    allowed: bool
    max_items: int
    reason: str


@dataclass(frozen=True)
class FeedbackChannelDiagnostics:
    channel: str
    last_elapsed_ms: float
    last_count: int
    elapsed_ewma_ms: float | None
    per_item_ewma_ms: float | None
    budget_ms: float
    batch_limit: int
    interval_ms: int


@dataclass(frozen=True)
class ResourceGovernorDiagnostics:
    pressure: ResourcePressureState
    lane_decisions: tuple[LaneWorkerDecision, ...] = ()
    ui_decisions: tuple[UiWorkDecision, ...] = ()
    feedback_channels: tuple[FeedbackChannelDiagnostics, ...] = ()
    telemetry_source: str = "n/a"
    system_cpu_percent: float | None = None
    process_cpu_percent: float | None = None
    load_average_1m: float | None = None


@dataclass(frozen=True)
class _ProfileTuning:
    interactive_target_ms: float
    idle_target_ms: float
    max_batch: int
    prefetch_idle: int
    prefetch_stage_ready: int


_PROFILE_TUNING = {
    MemoryProfileChoice.CONSERVATIVE: _ProfileTuning(3.5, 7.0, 6, 0, 1),
    MemoryProfileChoice.BALANCED: _ProfileTuning(4.0, 8.0, 12, 1, 2),
    MemoryProfileChoice.AGGRESSIVE: _ProfileTuning(5.5, 11.0, 18, 2, 4),
    MemoryProfileChoice.CUSTOM: _ProfileTuning(4.0, 8.0, 12, 1, 2),
}


@dataclass
class ResourceGovernor:
    compute_policy: ComputePolicy
    profile: MemoryProfileChoice | str = MemoryProfileChoice.BALANCED
    latency_feedback: LatencyFeedbackController | None = None
    min_worker_update_interval_ms: int = 250
    max_worker_step: int = 2
    _memory_policy: MemoryPolicy | None = None
    _telemetry: ResourceSnapshot | None = None
    _pressure: ResourcePressureState = field(
        default_factory=lambda: ResourcePressureState(ResourcePressure.NORMAL, 0.5, ResourcePressure.NORMAL, ResourcePressure.NORMAL, "initial")
    )
    _lane_targets: dict[ComputeLane, int] = field(default_factory=dict)
    _last_lane_update_monotonic: dict[ComputeLane, float] = field(default_factory=dict)
    _lane_decisions: dict[ComputeLane, LaneWorkerDecision] = field(default_factory=dict)
    _ui_decisions: dict[str, UiWorkDecision] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.profile = normalize_memory_profile_choice(self.profile)
        if self.latency_feedback is None:
            self.latency_feedback = LatencyFeedbackController()
        self._apply_latency_tuning()
        for lane in ComputeLane:
            self._lane_targets[lane] = self.compute_policy.workers_for_lane(lane)

    def update_policy(self, compute_policy: ComputePolicy, *, profile: MemoryProfileChoice | str | None = None) -> None:
        self.compute_policy = compute_policy
        if profile is not None:
            new_profile = normalize_memory_profile_choice(profile)
            if new_profile != self.profile:
                self.profile = new_profile
                self._apply_latency_tuning()
        for lane in ComputeLane:
            target = self._lane_targets.get(lane, self.compute_policy.workers_for_lane(lane))
            self._lane_targets[lane] = _clamp_int(target, 1, self.compute_policy.workers_for_lane(lane))

    def _apply_latency_tuning(self) -> None:
        tuning = _PROFILE_TUNING[normalize_memory_profile_choice(self.profile)]
        self.latency_feedback.tuning = LatencyFeedbackTuning(
            target_idle_ms=tuning.idle_target_ms,
            target_interactive_ms=tuning.interactive_target_ms,
            max_batch=tuning.max_batch,
        )

    def update_telemetry(self, snapshot: ResourceSnapshot, memory_policy: MemoryPolicy) -> None:
        self._telemetry = snapshot
        self._memory_policy = memory_policy
        self._pressure = self._compute_pressure(snapshot, memory_policy)

    def record_ui_observation(self, channel: str, elapsed_ms: float, item_count: int = 1) -> None:
        self.latency_feedback.observe(channel, elapsed_ms, count=max(1, int(item_count)))
        self._pressure = self._pressure_with_ui(channel)

    def decide_lane_workers(self, lane: ComputeLane, *, interactive: bool, busy_state: SchedulerBusyState) -> LaneWorkerDecision:
        lane = ComputeLane(lane)
        max_workers = max(1, int(self.compute_policy.workers_for_lane(lane)))
        min_workers = 1
        desired = max_workers
        reasons: list[str] = []
        pressure = self._pressure
        if pressure.memory_pressure == ResourcePressure.HIGH:
            desired = min(desired, 1)
            reasons.append("high memory pressure")
        if lane == ComputeLane.MONTAGE_TILE:
            if not busy_state.stage_ready_or_in_flight and busy_state.stage_busy:
                desired = min(desired, 2)
                reasons.append("waiting for reusable stage")
            if busy_state.result_backlog > max(4, desired * 2):
                desired = min(desired, max(1, desired - 1))
                reasons.append("UI result backlog")
            elif self._channel_pressure("montage_tile_result") == ResourcePressure.HIGH:
                desired = min(desired, max(1, self._lane_targets.get(lane, max_workers) - self.max_worker_step))
                reasons.append("high tile-result fan-in pressure")
        elif lane == ComputeLane.PREFETCH:
            if interactive or busy_state.visible_busy or busy_state.montage_busy or pressure.ui_pressure != ResourcePressure.NORMAL:
                desired = 1
                reasons.append("prefetch kept narrow while user-visible work is active")
        elif lane in {ComputeLane.VISIBLE, ComputeLane.STAGE, ComputeLane.PROFILE, ComputeLane.ROI, ComputeLane.PIXEL}:
            desired = min(desired, max_workers)
        if pressure.cpu_headroom < 0.15 and lane not in {ComputeLane.VISIBLE, ComputeLane.STAGE}:
            desired = min(desired, max(1, self._lane_targets.get(lane, max_workers) - 1))
            reasons.append("low CPU headroom")
        target = self._damped_lane_target(lane, _clamp_int(desired, min_workers, max_workers))
        decision = LaneWorkerDecision(lane, target, min_workers, max_workers, ", ".join(reasons) or "profile baseline")
        self._lane_decisions[lane] = decision
        return decision

    def decide_ui_work(self, channel: str, *, interactive: bool) -> UiWorkDecision:
        channel = str(channel)
        feedback = self.latency_feedback
        budget = float(feedback.work_budget_ms(channel, interactive=interactive))
        snapshot = feedback.channel_snapshot(channel)
        batch = int(feedback.batch_limit(channel, interactive=interactive))
        if snapshot.per_item_ewma_ms is not None and snapshot.per_item_ewma_ms > 0.0:
            batch = max(
                int(feedback.tuning.min_batch),
                min(int(feedback.tuning.max_batch), int(budget // max(0.25, snapshot.per_item_ewma_ms))),
            )
        interval = int(feedback.commit_interval_ms(channel, interactive=interactive))
        if self._pressure.ui_pressure == ResourcePressure.HIGH:
            batch = max(1, batch // 2)
            budget = max(2.0, budget * 0.75)
            interval = min(250, max(interval, int(round(budget * 3.0))))
            reason = "high UI pressure"
        elif self._pressure.ui_pressure == ResourcePressure.ELEVATED:
            if channel in {"montage_commit", "histogram_preview", "roi_refresh", "profile_update", "pixel_hover", "diagnostics_refresh"}:
                interval = min(250, max(interval, int(round(budget * 2.5))))
                reason = "elevated UI pressure; spacing UI commits"
            else:
                reason = "elevated UI pressure; preserving compute/result throughput"
        else:
            reason = "feedback target"
        decision = UiWorkDecision(channel, batch, budget, interval, reason)
        self._ui_decisions[channel] = decision
        return decision

    def decide_stage_warmup(self, candidate, *, app_idle: bool, stage_cached: bool, stage_in_flight: bool) -> PrefetchAdmissionDecision:
        if stage_cached:
            return PrefetchAdmissionDecision("stage_warmup", False, 0, "stage already cached")
        if stage_in_flight:
            return PrefetchAdmissionDecision("stage_warmup", False, 0, "stage already in flight")
        if not app_idle:
            return PrefetchAdmissionDecision("stage_warmup", False, 0, "visible work is busy")
        if self._pressure.memory_pressure in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
            return PrefetchAdmissionDecision("stage_warmup", False, 0, "memory pressure")
        return PrefetchAdmissionDecision("stage_warmup", True, 1, "idle and memory healthy")

    def decide_montage_prefetch(self, *, stage_ready_or_in_flight: bool, visible_busy: bool) -> PrefetchAdmissionDecision:
        tuning = _PROFILE_TUNING[normalize_memory_profile_choice(self.profile)]
        if visible_busy:
            return PrefetchAdmissionDecision("montage_prefetch", False, 0, "visible work is busy")
        if self._pressure.ui_pressure != ResourcePressure.NORMAL:
            return PrefetchAdmissionDecision("montage_prefetch", False, 0, "UI pressure")
        if self._pressure.memory_pressure in {ResourcePressure.ELEVATED, ResourcePressure.HIGH}:
            return PrefetchAdmissionDecision("montage_prefetch", False, 0, "memory pressure")
        if not stage_ready_or_in_flight:
            return PrefetchAdmissionDecision("montage_prefetch", False, 0, "required stage is not cached or in flight")
        return PrefetchAdmissionDecision("montage_prefetch", True, int(tuning.prefetch_stage_ready), "stage ready and idle")

    def diagnostics(self, *, channels: tuple[str, ...] = ()) -> ResourceGovernorDiagnostics:
        channel_names = tuple(dict.fromkeys((*channels, *tuple(snapshot.channel for snapshot in self.latency_feedback.snapshots()))))
        feedback_channels = []
        for channel in channel_names:
            snapshot = self.latency_feedback.channel_snapshot(channel)
            decision = self.decide_ui_work(channel, interactive=False)
            feedback_channels.append(
                FeedbackChannelDiagnostics(
                    channel=channel,
                    last_elapsed_ms=snapshot.last_elapsed_ms,
                    last_count=snapshot.last_count,
                    elapsed_ewma_ms=snapshot.elapsed_ewma_ms,
                    per_item_ewma_ms=snapshot.per_item_ewma_ms,
                    budget_ms=decision.budget_ms,
                    batch_limit=decision.batch_limit,
                    interval_ms=decision.interval_ms,
                )
            )
        telemetry = self._telemetry
        cpu = None if telemetry is None else telemetry.cpu
        return ResourceGovernorDiagnostics(
            pressure=self._pressure,
            lane_decisions=tuple(self._lane_decisions[lane] for lane in ComputeLane if lane in self._lane_decisions),
            ui_decisions=tuple(self._ui_decisions[channel] for channel in sorted(self._ui_decisions)),
            feedback_channels=tuple(feedback_channels),
            telemetry_source="n/a" if cpu is None else cpu.source,
            system_cpu_percent=None if cpu is None else cpu.system_cpu_percent,
            process_cpu_percent=None if cpu is None else cpu.process_cpu_percent,
            load_average_1m=None if cpu is None else cpu.load_average_1m,
        )

    def _compute_pressure(self, snapshot: ResourceSnapshot, memory_policy: MemoryPolicy) -> ResourcePressureState:
        available_fraction = float(memory_policy.system_available_bytes) / max(1.0, float(memory_policy.system_total_bytes))
        if available_fraction < 0.08:
            memory = ResourcePressure.HIGH
        elif available_fraction < 0.15:
            memory = ResourcePressure.ELEVATED
        elif available_fraction > 0.45:
            memory = ResourcePressure.LOW
        else:
            memory = ResourcePressure.NORMAL
        system_cpu = snapshot.cpu.system_cpu_percent
        if system_cpu is None or system_cpu <= 0.0:
            headroom = 0.5
        else:
            headroom = _clamp_float(1.0 - system_cpu / 100.0, 0.0, 1.0)
        ui = self._ui_pressure_from_channels()
        cache = ResourcePressure.NORMAL
        return ResourcePressureState(ui, headroom, memory, cache, f"available={available_fraction:.0%}, cpu_headroom={headroom:.0%}")

    def _pressure_with_ui(self, channel: str) -> ResourcePressureState:
        previous = self._pressure
        return ResourcePressureState(
            self._ui_pressure_from_channels(),
            previous.cpu_headroom,
            previous.memory_pressure,
            previous.cache_pressure,
            previous.reason,
        )

    def _ui_pressure_from_channels(self) -> ResourcePressure:
        worst = ResourcePressure.NORMAL
        for snapshot in self.latency_feedback.snapshots():
            target = self.latency_feedback.tuning.target_idle_ms
            elapsed = snapshot.elapsed_ewma_ms
            if elapsed is None:
                continue
            ratio = float(elapsed) / max(0.25, float(target))
            if ratio >= 2.0:
                return ResourcePressure.HIGH
            if ratio >= 1.25:
                worst = ResourcePressure.ELEVATED
        return worst

    def _channel_pressure(self, channel: str) -> ResourcePressure:
        snapshot = self.latency_feedback.channel_snapshot(channel)
        if snapshot.elapsed_ewma_ms is None:
            return ResourcePressure.NORMAL
        target = self.latency_feedback.tuning.target_idle_ms
        ratio = float(snapshot.elapsed_ewma_ms) / max(0.25, float(target))
        if ratio >= 2.0:
            return ResourcePressure.HIGH
        if ratio >= 1.25:
            return ResourcePressure.ELEVATED
        return ResourcePressure.NORMAL

    def _damped_lane_target(self, lane: ComputeLane, desired: int) -> int:
        current = max(1, int(self._lane_targets.get(lane, self.compute_policy.workers_for_lane(lane))))
        now = monotonic()
        last = float(self._last_lane_update_monotonic.get(lane, 0.0))
        if (now - last) * 1000.0 < self.min_worker_update_interval_ms:
            return current
        if desired < current:
            target = max(desired, current - self.max_worker_step)
        elif desired > current:
            target = min(desired, current + self.max_worker_step)
        else:
            target = current
        self._lane_targets[lane] = int(target)
        self._last_lane_update_monotonic[lane] = now
        return int(target)


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))
