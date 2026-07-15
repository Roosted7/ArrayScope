"""Pure telemetry governor for kernel quotas and bounded GUI fan-in."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from math import ceil
from time import monotonic

from arrayscope.core.compute_policy import ComputeLane, ComputePolicy
from arrayscope.core.gui_callback_budget import GuiCallbackObservation, WARNING_THRESHOLD_MS
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
    semantic_evidence_blocking: bool = False


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
    byte_cap: int = 0
    control_budget_ms: float = 0.0
    model: str = ""
    details: tuple[str, ...] = ()


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
    last_byte_count: int = 0
    per_byte_ewma_ms: float | None = None


@dataclass(frozen=True)
class ResourceGovernorDiagnostics:
    pressure: ResourcePressureState
    lane_decisions: tuple[LaneWorkerDecision, ...] = ()
    feedback_channels: tuple[FeedbackChannelDiagnostics, ...] = ()
    recent_ui_work_observations: tuple[GuiCallbackObservation, ...] = ()
    recent_over_warning_callbacks: tuple[GuiCallbackObservation, ...] = ()
    telemetry_source: str = "n/a"
    system_cpu_percent: float | None = None
    process_cpu_percent: float | None = None
    load_average_1m: float | None = None


@dataclass(frozen=True)
class _ProfileTuning:
    interactive_target_ms: float
    idle_target_ms: float
    max_batch: int


_PROFILE_TUNING = {
    MemoryProfileChoice.CONSERVATIVE: _ProfileTuning(3.5, 7.0, 6),
    MemoryProfileChoice.BALANCED: _ProfileTuning(4.0, 8.0, 12),
    MemoryProfileChoice.AGGRESSIVE: _ProfileTuning(5.5, 11.0, 18),
    MemoryProfileChoice.CUSTOM: _ProfileTuning(4.0, 8.0, 12),
}

_PRESSURE_TELEMETRY_ONLY_CHANNELS = frozenset(
    {
        "montage_cold_commit",
        "montage_layout_commit",
        "montage_level_evidence",
        "montage_level_refinement",
        "montage_present_total",
        "montage_priority_retarget",
        "tile_layer_commit",
    }
)

_PRESENTATION_UPLOAD_CHANNELS = frozenset(
    {
        "montage_cold_commit",
        "montage_present_total",
        "tile_layer_commit",
    }
)


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
    _feedback_outlier_streak: dict[str, int] = field(default_factory=dict)
    _recent_ui_work_observations: deque[GuiCallbackObservation] = field(default_factory=lambda: deque(maxlen=4096))
    _recent_over_warning_callbacks: deque[GuiCallbackObservation] = field(default_factory=lambda: deque(maxlen=32))
    _ui_observation_epoch: int = 0
    _ui_observation_epoch_active: int = 0
    _ui_observation_epoch_count: int = 0
    _ui_observation_epoch_max_ms: float = 0.0

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

    def record_ui_observation(
        self,
        channel: str,
        elapsed_ms: float,
        item_count: int = 1,
        *,
        byte_count: int = 0,
        work_class: str = "",
        backend: str = "",
        details: tuple[str, ...] = (),
    ) -> None:
        if _diagnostics_only_ui_work(channel, work_class=work_class, byte_count=byte_count):
            return
        count = max(1, int(item_count))
        byte_count = max(0, int(byte_count))
        feedback_elapsed_ms = self._feedback_elapsed_ms(channel, elapsed_ms)
        self.latency_feedback.observe(channel, feedback_elapsed_ms, count=count, byte_count=byte_count)
        observation = GuiCallbackObservation(
            channel=str(channel),
            work_class=str(work_class or ""),
            backend=str(backend or ""),
            target_ms=float(self.latency_feedback.work_budget_ms(channel, interactive=False)),
            warning_ms=WARNING_THRESHOLD_MS,
            item_cap=max(1, count),
            byte_cap=byte_count,
            elapsed_ms=max(0.0, float(elapsed_ms)),
            processed_items=count,
            processed_bytes=byte_count,
            details=tuple(str(detail) for detail in details),
        )
        self._recent_ui_work_observations.append(observation)
        self._note_ui_observation_epoch(observation.elapsed_ms)
        if float(elapsed_ms) >= WARNING_THRESHOLD_MS:
            self._recent_over_warning_callbacks.append(observation)
        self._pressure = self._pressure_with_ui(channel)

    def record_gui_callback_observation(self, observation: GuiCallbackObservation) -> None:
        diagnostics_only = _diagnostics_only_ui_observation(observation)
        self._recent_ui_work_observations.append(observation)
        self._note_ui_observation_epoch(observation.elapsed_ms)
        if observation.over_warning:
            self._recent_over_warning_callbacks.append(observation)
        if diagnostics_only:
            return
        count = max(1, int(observation.processed_items))
        byte_count = max(0, int(observation.processed_bytes))
        feedback_elapsed_ms = self._feedback_elapsed_ms(observation.channel, observation.elapsed_ms)
        self.latency_feedback.observe(observation.channel, feedback_elapsed_ms, count=count, byte_count=byte_count)
        self._pressure = self._pressure_with_ui(observation.channel)

    def begin_ui_observation_epoch(self) -> int:
        """Start constant-memory timing evidence for one diagnostic phase."""

        self._ui_observation_epoch += 1
        self._ui_observation_epoch_active = int(self._ui_observation_epoch)
        self._ui_observation_epoch_count = 0
        self._ui_observation_epoch_max_ms = 0.0
        return int(self._ui_observation_epoch_active)

    def ui_observation_epoch_evidence(self, epoch: int) -> tuple[int, float, bool]:
        current = int(epoch) == int(self._ui_observation_epoch_active)
        return (
            int(self._ui_observation_epoch_count) if current else 0,
            float(self._ui_observation_epoch_max_ms) if current else 0.0,
            bool(current),
        )

    def _note_ui_observation_epoch(self, elapsed_ms: float) -> None:
        if int(self._ui_observation_epoch_active) <= 0:
            return
        self._ui_observation_epoch_count += 1
        self._ui_observation_epoch_max_ms = max(
            float(self._ui_observation_epoch_max_ms),
            max(0.0, float(elapsed_ms)),
        )

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
            if interactive:
                desired = min(desired, 2)
                reasons.append("bounded montage workers during interaction")
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
            if interactive or busy_state.visible_busy or busy_state.montage_busy:
                min_workers = 0
                desired = 0
                reasons.append("prefetch parked while user-visible work is active")
        elif lane == ComputeLane.HISTOGRAM:
            if busy_state.semantic_evidence_blocking:
                desired = min(desired, 1)
                reasons.append("semantic level evidence blocks first presentation")
            elif (
                busy_state.visible_busy
                or busy_state.montage_busy
                or busy_state.stage_busy
            ):
                min_workers = 0
                desired = 0
                reasons.append("histogram parked behind runnable user-visible rendering")
            else:
                desired = min(desired, max_workers)
        elif lane in {ComputeLane.PROFILE, ComputeLane.ROI, ComputeLane.PIXEL}:
            if interactive or busy_state.visible_busy or busy_state.montage_busy or busy_state.stage_busy:
                min_workers = 0
                desired = 0
                reasons.append("inspection parked behind visible rendering")
            else:
                desired = min(desired, max_workers)
        elif lane in {ComputeLane.VISIBLE, ComputeLane.STAGE}:
            desired = min(desired, max_workers)
        if desired > 0 and pressure.cpu_headroom < 0.15 and lane not in {ComputeLane.VISIBLE, ComputeLane.STAGE}:
            desired = min(desired, max(1, self._lane_targets.get(lane, max_workers) - 1))
            reasons.append("low CPU headroom")
        clamped_desired = _clamp_int(desired, min_workers, max_workers)
        target = (
            clamped_desired
            if lane == ComputeLane.MONTAGE_TILE and interactive
            else self._damped_lane_target(lane, clamped_desired)
        )
        decision = LaneWorkerDecision(lane, target, min_workers, max_workers, ", ".join(reasons) or "profile baseline")
        self._lane_decisions[lane] = decision
        return decision

    def decide_bridge_drain(self, *, interactive: bool) -> UiWorkDecision:
        """Return the bridge-drain knob: item cap plus elapsed budget."""

        return self._decide_budget("kernel_bridge_drain", interactive=interactive, byte_cap=0)

    def decide_commit_batch(self, *, interactive: bool) -> UiWorkDecision:
        """Return the shared presentation-commit knob: item and byte bounds."""

        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
        return self._decide_budget("montage_present_total", interactive=interactive, byte_cap=byte_cap)

    def _decide_budget(self, channel: str, *, interactive: bool, byte_cap: int) -> UiWorkDecision:
        channel = str(channel)
        feedback = self.latency_feedback
        budget = float(feedback.work_budget_ms(channel, interactive=interactive))
        snapshot = feedback.channel_snapshot(channel)
        batch = int(feedback.batch_limit(channel, interactive=interactive))
        details: tuple[str, ...] = (
            f"snapshot last={snapshot.last_elapsed_ms:.2f}ms/{snapshot.last_count} items/{snapshot.last_byte_count} bytes",
            f"ewma={_optional_ms(snapshot.elapsed_ewma_ms)} per-item={_optional_ms(snapshot.per_item_ewma_ms)}",
            f"budget={budget:.2f}ms interactive={bool(interactive)}",
        )
        if byte_cap > 0 and snapshot.last_count > 0 and snapshot.last_byte_count > 0:
            bytes_per_item = int(ceil(float(snapshot.last_byte_count) / max(1.0, float(snapshot.last_count))))
            byte_cap = max(int(byte_cap), int(bytes_per_item * max(1, batch)))
        interval = 0
        reason = "interactive feedback target" if interactive else "feedback target"
        decision = UiWorkDecision(
            channel,
            batch,
            budget,
            interval,
            reason,
            int(byte_cap),
            float(budget),
            "ewma",
            tuple(details),
        )
        return decision

    def diagnostics(self, *, channels: tuple[str, ...] = ()) -> ResourceGovernorDiagnostics:
        channel_names = tuple(dict.fromkeys((*channels, *tuple(snapshot.channel for snapshot in self.latency_feedback.snapshots()))))
        feedback_channels = []
        for channel in channel_names:
            snapshot = self.latency_feedback.channel_snapshot(channel)
            feedback_channels.append(
                FeedbackChannelDiagnostics(
                    channel=channel,
                    last_elapsed_ms=snapshot.last_elapsed_ms,
                    last_count=snapshot.last_count,
                    last_byte_count=snapshot.last_byte_count,
                    elapsed_ewma_ms=snapshot.elapsed_ewma_ms,
                    per_item_ewma_ms=snapshot.per_item_ewma_ms,
                    per_byte_ewma_ms=snapshot.per_byte_ewma_ms,
                    budget_ms=float(self.latency_feedback.work_budget_ms(channel, interactive=False)),
                    batch_limit=int(self.latency_feedback.batch_limit(channel, interactive=False)),
                    interval_ms=0,
                )
            )
        telemetry = self._telemetry
        cpu = None if telemetry is None else telemetry.cpu
        return ResourceGovernorDiagnostics(
            pressure=self._pressure,
            lane_decisions=tuple(self._lane_decisions[lane] for lane in ComputeLane if lane in self._lane_decisions),
            feedback_channels=tuple(feedback_channels),
            recent_ui_work_observations=tuple(self._recent_ui_work_observations),
            recent_over_warning_callbacks=tuple(self._recent_over_warning_callbacks),
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

    def _feedback_elapsed_ms(self, channel: str, elapsed_ms: float) -> float:
        elapsed = max(0.0, float(elapsed_ms))
        channel = str(channel)
        snapshot = self.latency_feedback.channel_snapshot(channel)
        previous = snapshot.elapsed_ewma_ms
        if previous is None or previous <= 0.0:
            self._feedback_outlier_streak[channel] = 0
            return elapsed
        # Isolated spikes (GC pauses, one-off relayouts, event-loop stalls)
        # measure the environment, not the per-item cost of this channel.
        # Suppress a single outlier for every channel; a repeat is accepted
        # as a genuine cost change. Presentation uploads keep their wider
        # control budget so GPU submission bursts are not treated as spikes.
        if channel in _PRESENTATION_UPLOAD_CHANNELS:
            control_budget = max(
                float(self.latency_feedback.work_budget_ms(channel, interactive=False)),
                WARNING_THRESHOLD_MS + float(self.latency_feedback.tuning.target_idle_ms),
            )
        else:
            control_budget = max(
                float(self.latency_feedback.work_budget_ms(channel, interactive=False)),
                float(self.latency_feedback.tuning.target_idle_ms),
            )
        isolated_spike = elapsed > max(control_budget * 2.0, float(previous) * 3.0)
        if isolated_spike:
            streak = int(self._feedback_outlier_streak.get(channel, 0)) + 1
            self._feedback_outlier_streak[channel] = streak
            if streak < 2:
                return max(float(previous), control_budget)
            return elapsed
        self._feedback_outlier_streak[channel] = 0
        return elapsed

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
            if snapshot.channel in _PRESSURE_TELEMETRY_ONLY_CHANNELS:
                continue
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
        desired = max(0, int(desired))
        if lane not in self._lane_decisions:
            self._lane_targets[lane] = desired
            self._last_lane_update_monotonic[lane] = monotonic()
            return desired
        if desired == 0:
            self._lane_targets[lane] = 0
            self._last_lane_update_monotonic[lane] = monotonic()
            return 0
        current = max(0, int(self._lane_targets.get(lane, self.compute_policy.workers_for_lane(lane))))
        now = monotonic()
        last = float(self._last_lane_update_monotonic.get(lane, 0.0))
        if current == 0:
            target = min(desired, self.max_worker_step)
            self._lane_targets[lane] = int(target)
            self._last_lane_update_monotonic[lane] = now
            return int(target)
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


def _optional_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.2f}ms"


def _diagnostics_only_ui_observation(observation: GuiCallbackObservation) -> bool:
    return _diagnostics_only_ui_work(
        observation.channel,
        work_class=observation.work_class,
        byte_count=observation.processed_bytes,
    )


def _diagnostics_only_ui_work(channel: str, *, work_class: str = "", byte_count: int = 0) -> bool:
    return bool(
        str(channel) == "tile_layer_commit"
        and str(work_class or "") in {"presentation_upsert", "tile_layer_commit"}
        and int(byte_count or 0) <= 0
    )
