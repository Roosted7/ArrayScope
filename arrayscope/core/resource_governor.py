"""Pure telemetry governor for kernel quotas and bounded GUI fan-in."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from math import ceil, exp, log10
from time import monotonic

from arrayscope.core.compute_policy import ComputeLane, ComputePolicy
from arrayscope.core.gui_callback_budget import WARNING_THRESHOLD_MS, GuiCallbackObservation
from arrayscope.core.latency_feedback import LatencyFeedbackController, LatencyFeedbackTuning
from arrayscope.core.memory_policy import (
    MemoryPolicy,
    MemoryProfileChoice,
    normalize_memory_profile_choice,
)
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

_RENDER_PASS_REQUIREMENT_MS = 32.0
_RENDER_PASS_ITEM_INDEPENDENCE_RATIO = 0.9
_RENDER_PASS_HARD_LIMIT_MS = 50.0
_MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class _RenderPassCostModel:
    fixed_ms: float | None = None
    item_ms: float | None = None
    byte_ms_per_mib: float | None = None
    cohort_quadratic_ms: float | None = None
    residual_rms_ms: float | None = None
    r_squared: float | None = None
    samples: int = 0

    @property
    def identifiable(self) -> bool:
        return self.fixed_ms is not None and (
            self.item_ms is not None or self.byte_ms_per_mib is not None
        )


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

# Cap on how many recent UI-work observations `diagnostics()` exposes.  The
# in-memory feedback deque stays 4096 deep for the latency controller and the
# profiling harness (which reads `_recent_ui_work_observations` directly);
# this bound only governs serialized diagnostic evidence.
_DIAGNOSTIC_UI_OBSERVATION_LIMIT = 64


def _deque_tail(values: deque, limit: int) -> tuple:
    if len(values) <= limit:
        return tuple(values)
    return tuple(values)[-limit:]


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
        default_factory=lambda: ResourcePressureState(
            ResourcePressure.NORMAL,
            0.5,
            ResourcePressure.NORMAL,
            ResourcePressure.NORMAL,
            "initial",
        )
    )
    _lane_targets: dict[ComputeLane, int] = field(default_factory=dict)
    _last_lane_update_monotonic: dict[ComputeLane, float] = field(default_factory=dict)
    _lane_decisions: dict[ComputeLane, LaneWorkerDecision] = field(default_factory=dict)
    _feedback_outlier_streak: dict[str, int] = field(default_factory=dict)
    _recent_ui_work_observations: deque[GuiCallbackObservation] = field(
        default_factory=lambda: deque(maxlen=4096)
    )
    _recent_over_warning_callbacks: deque[GuiCallbackObservation] = field(
        default_factory=lambda: deque(maxlen=32)
    )
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

    def update_policy(
        self, compute_policy: ComputePolicy, *, profile: MemoryProfileChoice | str | None = None
    ) -> None:
        self.compute_policy = compute_policy
        if profile is not None:
            new_profile = normalize_memory_profile_choice(profile)
            if new_profile != self.profile:
                self.profile = new_profile
                self._apply_latency_tuning()
        for lane in ComputeLane:
            target = self._lane_targets.get(lane, self.compute_policy.workers_for_lane(lane))
            self._lane_targets[lane] = _clamp_int(
                target, 1, self.compute_policy.workers_for_lane(lane)
            )

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
        self.latency_feedback.observe(
            channel, feedback_elapsed_ms, count=count, byte_count=byte_count
        )
        target_ms = (
            _RENDER_PASS_REQUIREMENT_MS
            if str(channel).startswith("montage_render_pass_")
            else float(self.latency_feedback.work_budget_ms(channel, interactive=False))
        )
        observation = GuiCallbackObservation(
            channel=str(channel),
            work_class=str(work_class or ""),
            backend=str(backend or ""),
            target_ms=target_ms,
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
        self.latency_feedback.observe(
            observation.channel, feedback_elapsed_ms, count=count, byte_count=byte_count
        )
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
            float(self._ui_observation_epoch_max_ms), 0.0, float(elapsed_ms)
        )

    def decide_lane_workers(
        self, lane: ComputeLane, *, interactive: bool, busy_state: SchedulerBusyState
    ) -> LaneWorkerDecision:
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
                desired = min(
                    desired,
                    max(1, self._lane_targets.get(lane, max_workers) - self.max_worker_step),
                )
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
            elif busy_state.visible_busy or busy_state.montage_busy or busy_state.stage_busy:
                min_workers = 0
                desired = 0
                reasons.append("histogram parked behind runnable user-visible rendering")
            else:
                desired = min(desired, max_workers)
        elif lane in {ComputeLane.PROFILE, ComputeLane.ROI, ComputeLane.PIXEL}:
            if (
                interactive
                or busy_state.visible_busy
                or busy_state.montage_busy
                or busy_state.stage_busy
            ):
                min_workers = 0
                desired = 0
                reasons.append("inspection parked behind visible rendering")
            else:
                desired = min(desired, max_workers)
        elif lane in {ComputeLane.VISIBLE, ComputeLane.STAGE}:
            desired = min(desired, max_workers)
        if (
            desired > 0
            and pressure.cpu_headroom < 0.15
            and lane not in {ComputeLane.VISIBLE, ComputeLane.STAGE}
        ):
            desired = min(desired, max(1, self._lane_targets.get(lane, max_workers) - 1))
            reasons.append("low CPU headroom")
        clamped_desired = _clamp_int(desired, min_workers, max_workers)
        target = (
            clamped_desired
            if lane == ComputeLane.MONTAGE_TILE and interactive
            else self._damped_lane_target(lane, clamped_desired)
        )
        decision = LaneWorkerDecision(
            lane, target, min_workers, max_workers, ", ".join(reasons) or "profile baseline"
        )
        self._lane_decisions[lane] = decision
        return decision

    def decide_bridge_drain(self, *, interactive: bool) -> UiWorkDecision:
        """Return the bridge-drain knob: item cap plus elapsed budget."""

        return self._decide_budget("kernel_bridge_drain", interactive=interactive, byte_cap=0)

    def decide_commit_batch(self, *, interactive: bool) -> UiWorkDecision:
        """Return the shared presentation-commit knob: item and byte bounds."""

        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
        return self._decide_budget(
            "montage_present_total", interactive=interactive, byte_cap=byte_cap
        )

    def decide_render_pass(
        self,
        *,
        interactive: bool,
        pass_kind: str = "preview",
        remaining_items: int | None = None,
    ) -> UiWorkDecision:
        """Own R5 chunk size and deadline for preview and target passes.

        The generic UI target is intentionally tight (4/8 ms), but a tiled
        presentation transaction has fixed scene-publication work. Driving
        that channel to 8 ms collapses to one tile without shortening the
        callback. R5 permits a governed chunk up to 50 ms, so the governor
        targets 32 ms and keeps 18 ms of hard-budget margin.
        """

        pass_kind = "target" if str(pass_kind) == "target" else "preview"
        channel = f"montage_render_pass_{pass_kind}"
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
        snapshot = self.latency_feedback.channel_snapshot(channel)
        max_batch = max(1, int(self.latency_feedback.tuning.max_batch))
        previous_rows = getattr(self, "_render_pass_previous_observations", None)
        if previous_rows is None:
            previous_rows = {}
            self._render_pass_previous_observations = previous_rows
        item_independent = getattr(self, "_render_pass_item_independent", None)
        if item_independent is None:
            item_independent = set()
            self._render_pass_item_independent = item_independent
        sample_rows = getattr(self, "_render_pass_cost_samples", None)
        if sample_rows is None:
            sample_rows = {}
            self._render_pass_cost_samples = sample_rows
        samples = sample_rows.setdefault(channel, [])
        previous = previous_rows.get(channel)
        if (
            previous is not None
            and snapshot.last_count < previous[0]
            and snapshot.last_elapsed_ms
            >= float(previous[1]) * _RENDER_PASS_ITEM_INDEPENDENCE_RATIO
        ):
            item_independent.add(channel)
        if snapshot.last_count > 0:
            sample = (
                int(snapshot.last_count),
                float(snapshot.last_elapsed_ms),
                int(snapshot.last_byte_count),
            )
            if not samples or samples[-1] != sample:
                samples.append(sample)
                del samples[:-8]
            previous_rows[channel] = (
                int(snapshot.last_count),
                float(snapshot.last_elapsed_ms),
            )
        model = _render_pass_cost_model(samples)
        fixed_ms = model.fixed_ms
        item_ms = model.item_ms
        byte_ms_per_mib = model.byte_ms_per_mib
        if fixed_ms is not None:
            mean_elapsed = sum(row[1] for row in samples) / len(samples)
            if fixed_ms >= 0.75 * mean_elapsed:
                item_independent.add(channel)
        byte_rows = [row for row in samples if row[2] > 0]
        bytes_per_item = (
            float(sum(row[2] for row in byte_rows))
            / max(1.0, float(sum(row[0] for row in byte_rows)))
            if byte_rows
            else 0.0
        )
        effective_marginal_ms = (
            max(0.0, float(item_ms or 0.0))
            + max(0.0, float(byte_ms_per_mib or 0.0)) * bytes_per_item / _MIB
        )
        remaining = max(
            1,
            int(
                remaining_items
                if remaining_items is not None
                else max((row[0] for row in samples), default=max_batch)
            ),
        )
        steering = "feedback"
        predicted_ms = None
        regularization_ms = None
        extrapolation_ms = None
        control_achievable = None
        hard_achievable = None
        if model.identifiable:
            batch, predicted_ms, regularization_ms = _render_pass_optimal_point(
                fixed_ms=max(0.0, float(fixed_ms or 0.0)),
                marginal_ms=max(0.0, effective_marginal_ms),
                quadratic_ms=max(0.0, float(model.cohort_quadratic_ms or 0.0)),
                remaining_items=remaining,
                observed_item_max=max((int(row[0]) for row in samples), default=1),
                observed_byte_max=(
                    max((int(row[2]) for row in byte_rows), default=0)
                    if byte_ms_per_mib is not None
                    else 0
                ),
                bytes_per_item=bytes_per_item,
            )
            steering = "weighted-fill-latency"
            extrapolation_ms = _render_pass_extrapolation_cost_ms(
                float(batch) / max(1.0, max((float(row[0]) for row in samples), default=1.0))
            )
            if byte_ms_per_mib is not None and byte_rows and bytes_per_item > 0.0:
                extrapolation_ms += _render_pass_extrapolation_cost_ms(
                    float(batch) * bytes_per_item / max(1.0, *(float(row[2]) for row in byte_rows))
                )
            if extrapolation_ms > 0.0:
                steering = "weighted-fill-latency-extrapolation"
            one_item_ms = max(0.0, float(fixed_ms or 0.0)) + max(0.0, effective_marginal_ms)
            control_achievable = one_item_ms <= _RENDER_PASS_REQUIREMENT_MS
            hard_achievable = one_item_ms <= _RENDER_PASS_HARD_LIMIT_MS
            if bytes_per_item > 0.0:
                desired_bytes = max(1, ceil(float(batch) * bytes_per_item))
                byte_cap = min(
                    int(byte_cap),
                    desired_bytes,
                )
        elif snapshot.last_count <= 0:
            # Cold-start at one. The old two-item prediction was already
            # enough to exceed 50 ms for PyQtGraph's FFT windowing on some
            # tiles; growth is earned only by measured sub-20 ms feedback.
            batch = 1
        elif snapshot.last_elapsed_ms >= 50.0:
            batch = max(1, int(snapshot.last_count) // 2)
        elif snapshot.last_elapsed_ms > 32.0:
            batch = max(1, int(snapshot.last_count) - 1)
        elif snapshot.last_elapsed_ms < 20.0:
            batch = min(max_batch, int(snapshot.last_count) + 2)
        else:
            batch = min(max_batch, int(snapshot.last_count) + 1)
        return UiWorkDecision(
            channel,
            batch,
            _RENDER_PASS_REQUIREMENT_MS,
            0,
            "R5 governed render-pass target",
            byte_cap,
            _RENDER_PASS_REQUIREMENT_MS,
            "r5-feedback",
            (
                f"snapshot last={snapshot.last_elapsed_ms:.2f}ms/"
                f"{snapshot.last_count} items/{snapshot.last_byte_count} bytes",
                f"per-item={_optional_ms(snapshot.per_item_ewma_ms)}",
                f"item-independent={int(channel in item_independent)}",
                (
                    f"fixed={_optional_ms(fixed_ms)} "
                    f"item={_optional_ms(item_ms)} "
                    f"bytes={_optional_ms(byte_ms_per_mib)}/MiB "
                    f"cohort2={_optional_ms(model.cohort_quadratic_ms)}/item2"
                ),
                (
                    f"fit-rms={_optional_ms(model.residual_rms_ms)} "
                    f"r2={_optional_float(model.r_squared)} samples={model.samples}"
                ),
                (
                    f"steering={steering} remaining={remaining} "
                    f"predicted={_optional_ms(predicted_ms)} "
                    f"latency-cost={_optional_ms(regularization_ms)} "
                    f"extrapolation-cost={_optional_ms(extrapolation_ms)}"
                ),
                (
                    f"control-achievable={_optional_bool(control_achievable)} "
                    f"r5-achievable={_optional_bool(hard_achievable)}"
                ),
                "target=32.00ms hard=50.00ms",
            ),
        )

    def begin_render_pass(self, token: object, *, pass_kind: str = "preview") -> None:
        """Start feedback for one preview or target pass."""

        pass_kind = "target" if str(pass_kind) == "target" else "preview"
        tokens = getattr(self, "_render_pass_tokens", None)
        if tokens is None:
            tokens = {}
            self._render_pass_tokens = tokens
        if tokens.get(pass_kind) == token:
            return
        tokens[pass_kind] = token
        channel = f"montage_render_pass_{pass_kind}"
        self.latency_feedback.reset_channel(channel)
        getattr(self, "_render_pass_previous_observations", {}).pop(channel, None)
        getattr(self, "_render_pass_item_independent", set()).discard(channel)
        getattr(self, "_render_pass_cost_samples", {}).pop(channel, None)

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
            bytes_per_item = ceil(
                float(snapshot.last_byte_count) / max(1.0, float(snapshot.last_count))
            )
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
        channel_names = tuple(
            dict.fromkeys(
                (
                    *channels,
                    *tuple(snapshot.channel for snapshot in self.latency_feedback.snapshots()),
                )
            )
        )
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
                    budget_ms=(
                        _RENDER_PASS_REQUIREMENT_MS
                        if str(channel).startswith("montage_render_pass_")
                        else float(self.latency_feedback.work_budget_ms(channel, interactive=False))
                    ),
                    batch_limit=int(self.latency_feedback.batch_limit(channel, interactive=False)),
                    interval_ms=0,
                )
            )
        telemetry = self._telemetry
        cpu = None if telemetry is None else telemetry.cpu
        return ResourceGovernorDiagnostics(
            pressure=self._pressure,
            lane_decisions=tuple(
                self._lane_decisions[lane] for lane in ComputeLane if lane in self._lane_decisions
            ),
            feedback_channels=tuple(feedback_channels),
            # Diagnostics snapshots serialize this tuple verbatim into JSONL
            # at 2 Hz; exposing the full 4096-deep feedback deque grew field
            # snapshot lines from ~1 KB to ~2 MB (2026-07-24).  The feedback
            # loop keeps the full deque (profile tooling reads it directly);
            # serialized evidence is bounded to the recent tail.
            recent_ui_work_observations=tuple(
                _deque_tail(self._recent_ui_work_observations, _DIAGNOSTIC_UI_OBSERVATION_LIMIT)
            ),
            recent_over_warning_callbacks=tuple(self._recent_over_warning_callbacks),
            telemetry_source="n/a" if cpu is None else cpu.source,
            system_cpu_percent=None if cpu is None else cpu.system_cpu_percent,
            process_cpu_percent=None if cpu is None else cpu.process_cpu_percent,
            load_average_1m=None if cpu is None else cpu.load_average_1m,
        )

    def _compute_pressure(
        self, snapshot: ResourceSnapshot, memory_policy: MemoryPolicy
    ) -> ResourcePressureState:
        available_fraction = float(memory_policy.system_available_bytes) / max(
            1.0, float(memory_policy.system_total_bytes)
        )
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
        return ResourcePressureState(
            ui,
            headroom,
            memory,
            cache,
            f"available={available_fraction:.0%}, cpu_headroom={headroom:.0%}",
        )

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
        current = max(
            0, int(self._lane_targets.get(lane, self.compute_policy.workers_for_lane(lane)))
        )
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


def _optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _optional_bool(value: bool | None) -> str:
    return "unknown" if value is None else str(int(bool(value)))


def _render_pass_cost_model(
    samples: list[tuple[int, float, int]],
) -> _RenderPassCostModel:
    """Measure fixed, item, and byte-dependent transaction cost.

    Item and byte terms are reported separately only when the observed
    cohorts vary independently enough to identify both. Proportional
    item/byte samples deliberately fall back to the one-dimensional item fit
    instead of inventing a split.
    """

    if len(samples) < 2:
        return _RenderPassCostModel(samples=len(samples))
    counts = [float(row[0]) for row in samples]
    mib = [float(row[2]) / _MIB for row in samples]
    elapsed = [float(row[1]) for row in samples]
    mean_count = sum(counts) / len(counts)
    mean_mib = sum(mib) / len(mib)
    mean_elapsed = sum(elapsed) / len(elapsed)
    item_variance = sum((count - mean_count) ** 2 for count in counts)
    item_covariance = sum(
        (count - mean_count) * (duration - mean_elapsed)
        for count, duration in zip(counts, elapsed, strict=True)
    )
    item_ms = max(0.0, item_covariance / item_variance) if item_variance > 0.0 else None
    byte_ms_per_mib = None
    fixed = max(0.0, mean_elapsed - float(item_ms or 0.0) * mean_count)

    # Centred two-predictor least squares. The determinant is the evidence
    # that item count and bytes varied independently; without it their costs
    # cannot honestly be split.
    byte_variance = sum((value - mean_mib) ** 2 for value in mib)
    cross = sum(
        (count - mean_count) * (value - mean_mib) for count, value in zip(counts, mib, strict=True)
    )
    byte_covariance = sum(
        (value - mean_mib) * (duration - mean_elapsed)
        for value, duration in zip(mib, elapsed, strict=True)
    )
    determinant = item_variance * byte_variance - cross * cross
    scale = max(1.0, item_variance * byte_variance)
    if len(samples) >= 4 and determinant > scale * 1e-6:
        fitted_item = (item_covariance * byte_variance - byte_covariance * cross) / determinant
        fitted_byte = (byte_covariance * item_variance - item_covariance * cross) / determinant
        fitted_fixed = mean_elapsed - fitted_item * mean_count - fitted_byte * mean_mib
        if fitted_item >= 0.0 and fitted_byte >= 0.0 and fitted_fixed >= 0.0:
            fixed = fitted_fixed
            item_ms = fitted_item
            byte_ms_per_mib = fitted_byte
    elif item_variance <= 0.0 and byte_variance > 0.0:
        byte_ms_per_mib = max(0.0, byte_covariance / byte_variance)
        fixed = max(0.0, mean_elapsed - byte_ms_per_mib * mean_mib)

    cohort_quadratic_ms = None
    if byte_ms_per_mib is None and len(set(counts)) >= 3:
        quadratic = _nonnegative_quadratic_fit(counts, elapsed)
        if quadratic is not None:
            fitted_fixed, fitted_item, fitted_quadratic = quadratic
            fixed = fitted_fixed
            item_ms = fitted_item
            cohort_quadratic_ms = fitted_quadratic

    predicted = [
        fixed
        + float(item_ms or 0.0) * count
        + float(cohort_quadratic_ms or 0.0) * count * count
        + float(byte_ms_per_mib or 0.0) * value
        for count, value in zip(counts, mib, strict=True)
    ]
    residual_sum = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(elapsed, predicted, strict=True)
    )
    total_sum = sum((actual - mean_elapsed) ** 2 for actual in elapsed)
    return _RenderPassCostModel(
        fixed_ms=fixed,
        item_ms=item_ms,
        byte_ms_per_mib=byte_ms_per_mib,
        cohort_quadratic_ms=cohort_quadratic_ms,
        residual_rms_ms=(residual_sum / len(samples)) ** 0.5,
        r_squared=(1.0 - residual_sum / total_sum) if total_sum > 0.0 else None,
        samples=len(samples),
    )


def _nonnegative_quadratic_fit(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float, float] | None:
    """Fit ``fixed + linear*x + quadratic*x²`` without negative cost terms."""

    sums = [sum(value**power for value in x_values) for power in range(5)]
    rhs = [
        sum((value**power) * target for value, target in zip(x_values, y_values, strict=True))
        for power in range(3)
    ]
    matrix = [
        [sums[0], sums[1], sums[2], rhs[0]],
        [sums[1], sums[2], sums[3], rhs[1]],
        [sums[2], sums[3], sums[4], rhs[2]],
    ]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) <= 1e-12:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(3):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[column], strict=True)
            ]
    coefficients = tuple(float(matrix[row][3]) for row in range(3))
    if any(value < 0.0 for value in coefficients):
        return None
    return coefficients


def _render_pass_latency_cost_ms(elapsed_ms: float) -> float:
    """Continuous responsiveness price in absolute milliseconds.

    Work below the 18 ms smooth-interaction envelope is unregularized. The
    price rises gently to 45 ms, then joins a stronger quadratic with matching
    value and first derivative. There is deliberately no branch at 50 ms:
    R5 is reported there, while optimization sees the whole smooth curve.
    """

    elapsed = max(0.0, float(elapsed_ms))
    if elapsed <= 18.0:
        return 0.0
    gentle = 0.01
    if elapsed <= 45.0:
        return gentle * (elapsed - 18.0) ** 2
    at_45 = gentle * (45.0 - 18.0) ** 2
    slope_at_45 = 2.0 * gentle * (45.0 - 18.0)
    beyond = elapsed - 45.0
    return at_45 + slope_at_45 * beyond + 0.5 * beyond**2


def _render_pass_extrapolation_cost_ms(ratio: float) -> float:
    """Smooth evidence-distance price: free to 1.1x, exponential thereafter."""

    ratio = max(0.0, float(ratio))
    if ratio <= 1.1:
        return 0.0
    decades = log10(ratio / 1.1)
    return 5.0 * (exp(2.0 * decades) - 1.0) ** 2


def _render_pass_optimal_point(
    *,
    fixed_ms: float,
    marginal_ms: float,
    quadratic_ms: float,
    remaining_items: int,
    observed_item_max: int,
    observed_byte_max: int,
    bytes_per_item: float,
) -> tuple[int, float, float]:
    """Minimize predicted fill time plus continuous callback-latency cost."""

    remaining = max(1, int(remaining_items))
    marginal = max(0.0, float(marginal_ms))
    quadratic = max(0.0, float(quadratic_ms))

    def latency(items: int) -> float:
        count = float(items)
        return float(fixed_ms) + marginal * count + quadratic * count * count

    rows: list[tuple[float, float, int, float]] = []
    for items in range(1, remaining + 1):
        chunks = ceil(float(remaining) / float(items))
        chunk_ms = latency(items)
        latency_cost = _render_pass_latency_cost_ms(chunk_ms)
        extrapolation_cost = _render_pass_extrapolation_cost_ms(
            float(items) / max(1.0, float(observed_item_max))
        )
        if observed_byte_max > 0 and bytes_per_item > 0.0:
            extrapolation_cost += _render_pass_extrapolation_cost_ms(
                float(items) * float(bytes_per_item) / float(observed_byte_max)
            )
        rows.append(
            (
                # Fill time and responsiveness are paid by EVERY chunk, so they
                # scale with the chunk count. The extrapolation term does not:
                # it prices the risk that this cost model is wrong at a size it
                # has never observed, which is a property of taking the step at
                # all, not of each chunk that follows. Multiplying it by
                # `chunks` conflated the two, and did so worst exactly where
                # the chunk count is highest -- a small cohort inflated a few
                # milliseconds of model risk into hundreds, so the objective
                # preferred the size it had already measured no matter what the
                # model predicted. Measured: with a genuinely per-item cost
                # (2 ms fixed, 8 ms/item, 272 remaining) the per-chunk form
                # pinned the cohort at 1 forever -- 272 single-item chunks
                # against an optimum of 3 -- because probing 2 items scored a
                # 630 ms penalty for 4.6 ms of actual risk.
                chunks * (chunk_ms + latency_cost) + extrapolation_cost,
                chunk_ms,
                items,
                latency_cost,
            )
        )
    selected = min(
        rows,
        key=lambda row: (row[0], row[1], -row[2]),
    )
    return int(selected[2]), float(selected[1]), float(selected[3])


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
