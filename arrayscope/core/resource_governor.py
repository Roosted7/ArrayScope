"""Pure telemetry governor for kernel quotas and bounded GUI fan-in."""

from __future__ import annotations

import json
import os
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from functools import wraps
from math import ceil, exp, log2, log10
from statistics import median
from threading import RLock
from time import monotonic
from typing import TypeVar

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
#: Knee of the fill-optimal responsiveness curve: free below, gently priced to
#: the steep point, strongly quadratic beyond it.
_RENDER_PASS_LATENCY_KNEE_MS = 18.0
_RENDER_PASS_LATENCY_STEEP_MS = 45.0
#: Assumed display cadence when no caller reports the real one.
_RENDER_PASS_DEFAULT_REFRESH_HZ = 60.0
_RETAINED_FALLBACK_REBIND_BATCH_LIMIT = 32
_RETAINED_FALLBACK_REFINEMENT_BATCH_LIMIT = 4
_MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class _RenderPassCostModel:
    fixed_ms: float | None = None
    item_ms: float | None = None
    byte_ms_per_mib: float | None = None
    residual_variance_ms2: float | None = None
    residual_rms_ms: float | None = None
    raw_residual_rms_ms: float | None = None
    outlier_samples: int = 0
    r_squared: float | None = None
    design_independence: float | None = None
    samples: int = 0
    design_points: int = 0
    observed_item_min: int = 0
    observed_item_max: int = 0
    mean_elapsed_ms: float | None = None
    seeded: bool = False
    # True only when THIS fit separated the byte term from its own design
    # (independent item/byte variation, or byte-only variation). A byte value
    # merely inherited from a pooled prior keeps False: pooling an inherited
    # value back into the pool is how one misattributed fit became every
    # pass's marginal (the 2026-07-31 slow-fill attractor).
    byte_identified: bool = False

    @property
    def identifiable(self) -> bool:
        return self.fixed_ms is not None and (
            self.item_ms is not None or self.byte_ms_per_mib is not None
        )


# Channels whose cost is evidence, not a symptom of a busy UI. Each one is a
# bounded unit of presentation work whose duration is set by how much it was
# asked to do; feeding them to the pressure model would let a big-but-healthy
# commit or replan cut lane workers. `montage_retarget` joins them for exactly
# that reason: ADR 0065 stage 2 added it to measure the funnel it is about to
# move, and a measurement channel that steers scheduling is not a measurement.
_PRESSURE_TELEMETRY_ONLY_CHANNELS = frozenset(
    {
        "montage_cold_commit",
        "montage_layout_commit",
        "montage_level_evidence",
        "montage_level_refinement",
        "montage_present_total",
        "montage_priority_retarget",
        "montage_retarget",
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

# Diagnostics snapshots keep only a 64-observation tail, and a phase that dies
# mid-run never publishes its snapshot at all — both of which made the
# 2026-07-31 pyqtgraph commit-cost attribution impossible to read from JSONL
# rows alone (the tail was saturated by kernel_bridge_drain).  When this
# environment variable names a file, every observation is appended to it as
# one JSON line at record time, so the stream survives crashes and flooding.
_UI_OBSERVATION_LOG_ENV = "ARRAYSCOPE_UI_OBSERVATION_LOG"


def _maybe_log_ui_observation(observation: GuiCallbackObservation) -> None:
    path = os.environ.get(_UI_OBSERVATION_LOG_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t_monotonic": monotonic(), **asdict(observation)}) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def _deque_tail(values: deque, limit: int) -> tuple:
    if len(values) <= limit:
        return tuple(values)
    return tuple(values)[-limit:]


_R = TypeVar("_R")


def _governed(method: Callable[..., _R]) -> Callable[..., _R]:
    """Serialize one public governor entry point against the other callers.

    ADR 0065 gives the presentation-owner thread the render-pass family and its
    own drain observations, while the GUI thread keeps the lane, pressure, and
    telemetry families.  Both families walk the same dicts and deques, so every
    public entry point that mutates state -- or copies a collection out of it --
    takes the object's one lock.

    The lock is re-entrant because the guard sits on the entry points rather
    than inside them: a caller that takes several decisions as one section (the
    owner's begin/decide/record run), or any future entry point that reaches
    another one, would otherwise deadlock on its own acquisition.

    One acquire per call, never per item inside one: these are per-commit and
    per-drain paths, and an uncontended acquire is ~100 ns against callbacks
    measured in milliseconds.
    """

    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded


@dataclass
class ResourceGovernor:
    compute_policy: ComputePolicy
    profile: MemoryProfileChoice | str = MemoryProfileChoice.BALANCED
    latency_feedback: LatencyFeedbackController | None = None
    responsiveness_weight: float = 1.0
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
    _ui_observation_epoch_max: GuiCallbackObservation | None = None
    _render_pass_sample_banks: dict[
        object, dict[tuple[str, object], list[tuple[int, float, int]]]
    ] = field(default_factory=dict)
    _render_pass_contexts: dict[str, tuple[object, object]] = field(default_factory=dict)
    _render_pass_seed_models: dict[tuple[object, str, object], _RenderPassCostModel] = field(
        default_factory=dict
    )
    _render_pass_last_models: dict[tuple[object, str], _RenderPassCostModel] = field(
        default_factory=dict
    )
    _render_pass_reference_models: dict[tuple[object, str, object], _RenderPassCostModel] = field(
        default_factory=dict
    )
    _render_pass_last_local_models: dict[tuple[object, str, object], _RenderPassCostModel] = field(
        default_factory=dict
    )
    _render_pass_observation_markers: dict[str, GuiCallbackObservation | None] = field(
        default_factory=dict
    )
    # Never a constructor argument and never part of identity: two governors
    # are the same governor by their policy and telemetry, not by which lock
    # object they happen to hold.
    #
    # Outside callers that read a plain scalar attribute -- `profile`,
    # `responsiveness_weight` -- stay lock-free: a single attribute load is
    # already atomic and the reader gets one side of a write, never a torn
    # value.  `latency_feedback` is not such an attribute: the controller has
    # no lock of its own, so every call into it belongs to a guarded method
    # here.  The window alias that let the presentation path call it directly
    # is gone (ADR 0065 stage 2, decision 7) -- do not reintroduce one.
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.profile = normalize_memory_profile_choice(self.profile)
        self.responsiveness_weight = max(0.0, float(self.responsiveness_weight))
        if self.latency_feedback is None:
            self.latency_feedback = LatencyFeedbackController()
        self._apply_latency_tuning()
        for lane in ComputeLane:
            self._lane_targets[lane] = self.compute_policy.workers_for_lane(lane)

    @_governed
    def set_responsiveness_weight(self, value: float) -> None:
        """Move the legal render-pass optimum without changing the R5 requirement."""

        self.responsiveness_weight = max(0.0, float(value))

    @_governed
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

    @_governed
    def update_telemetry(self, snapshot: ResourceSnapshot, memory_policy: MemoryPolicy) -> None:
        self._telemetry = snapshot
        self._memory_policy = memory_policy
        self._pressure = self._compute_pressure(snapshot, memory_policy)

    @_governed
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
        _maybe_log_ui_observation(observation)
        self._note_ui_observation_epoch(observation)
        if float(elapsed_ms) >= WARNING_THRESHOLD_MS:
            self._recent_over_warning_callbacks.append(observation)
        self._pressure = self._pressure_with_ui(channel)

    @_governed
    def record_gui_callback_observation(self, observation: GuiCallbackObservation) -> None:
        diagnostics_only = _diagnostics_only_ui_observation(observation)
        self._recent_ui_work_observations.append(observation)
        _maybe_log_ui_observation(observation)
        self._note_ui_observation_epoch(observation)
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

    @_governed
    def observe_latency(
        self, channel: str, elapsed_ms: float, *, count: int = 1, byte_count: int = 0
    ) -> None:
        """Feed one raw timing into the channel model, under this lock.

        The fallback sink for callers with no ``_record_ui_work`` window. It
        deliberately skips the observation deque and the pressure recompute:
        those describe GUI callbacks, and a caller that reaches here has no
        GUI callback to describe.
        """

        self.latency_feedback.observe(
            str(channel),
            max(0.0, float(elapsed_ms)),
            count=max(1, int(count)),
            byte_count=max(0, int(byte_count)),
        )

    @_governed
    def begin_ui_observation_epoch(self) -> int:
        """Start constant-memory timing evidence for one diagnostic phase."""

        self._ui_observation_epoch += 1
        self._ui_observation_epoch_active = int(self._ui_observation_epoch)
        self._ui_observation_epoch_count = 0
        self._ui_observation_epoch_max_ms = 0.0
        self._ui_observation_epoch_max = None
        return int(self._ui_observation_epoch_active)

    @_governed
    def ui_observation_epoch_evidence(self, epoch: int) -> tuple[int, float, bool]:
        current = int(epoch) == int(self._ui_observation_epoch_active)
        return (
            int(self._ui_observation_epoch_count) if current else 0,
            float(self._ui_observation_epoch_max_ms) if current else 0.0,
            bool(current),
        )

    @_governed
    def ui_observation_epoch_max(self, epoch: int) -> GuiCallbackObservation | None:
        """Return the observation the epoch maximum came from, or ``None``.

        The epoch's maximum is taken over every observation, while the
        diagnostics tail keeps only the last 64 — so on a flooded phase the
        maximum was reported as a bare number with nothing naming it.  That is
        exactly the phase worth attributing, so the one observation that set it
        is retained beside the number.  Still constant memory: one record.
        """

        if int(epoch) != int(self._ui_observation_epoch_active):
            return None
        return self._ui_observation_epoch_max

    def _note_ui_observation_epoch(self, observation: GuiCallbackObservation) -> None:
        if int(self._ui_observation_epoch_active) <= 0:
            return
        self._ui_observation_epoch_count += 1
        elapsed_ms = max(0.0, float(observation.elapsed_ms))
        if self._ui_observation_epoch_max is None or elapsed_ms > float(
            self._ui_observation_epoch_max_ms
        ):
            self._ui_observation_epoch_max_ms = elapsed_ms
            self._ui_observation_epoch_max = observation

    @_governed
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

    @_governed
    def decide_bridge_drain(self, *, interactive: bool) -> UiWorkDecision:
        """Return the bridge-drain knob: item cap plus elapsed budget."""

        return self._decide_budget("kernel_bridge_drain", interactive=interactive, byte_cap=0)

    @_governed
    def decide_commit_batch(self, *, interactive: bool) -> UiWorkDecision:
        """Return the shared presentation-commit knob: item and byte bounds."""

        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
        return self._decide_budget(
            "montage_present_total", interactive=interactive, byte_cap=byte_cap
        )

    @_governed
    def batch_limit_for(self, channel: str, *, interactive: bool) -> int:
        """Return one channel's feedback batch limit, under this lock.

        The presentation path used to read the controller through a window
        alias with no lock at all (ADR 0065 stage 2, decision 7). Callers that
        want the whole knob still ask ``decide_commit_batch``; this is the
        floor they fall back to when a decision yields no limit.
        """

        return int(self.latency_feedback.batch_limit(str(channel), interactive=bool(interactive)))

    @_governed
    def latency_tuning(self) -> LatencyFeedbackTuning:
        """Return the current feedback tuning, under this lock.

        Frozen, so the caller may hold and read it after the lock is released;
        a profile change replaces the object rather than mutating it.
        """

        return self.latency_feedback.tuning

    def decide_ladder_admission(
        self,
        *,
        default_limit: int,
        retained_fallback_refinement: bool,
    ) -> int:
        """Bound GUI-side task admission behind an already-visible fallback.

        Unguarded on purpose, like ``decide_resident_crop_rebind`` below: both
        are named policies over their arguments and module constants and touch
        no governor state, so a lock here would only cost callers contention
        against the threads that do.

        That proof, re-verified 2026-08-03, is also what keeps them unguarded
        when ADR 0065 stage 2 slice B moves their callers onto the presentation
        owner: neither body names ``self`` at all, so it is already thread-safe
        and the lock would buy nothing. Decorate the moment either one does.
        """

        limit = max(1, int(default_limit))
        return (
            min(limit, _RETAINED_FALLBACK_REFINEMENT_BATCH_LIMIT)
            if retained_fallback_refinement
            else limit
        )

    def decide_resident_crop_rebind(self, *, remaining_items: int) -> UiWorkDecision:
        """Bound the retained-crop residency probes in one visible callback."""

        remaining = max(1, int(remaining_items))
        return UiWorkDecision(
            "resident_crop_rebind",
            min(remaining, _RETAINED_FALLBACK_REBIND_BATCH_LIMIT),
            _RENDER_PASS_REQUIREMENT_MS,
            0,
            "R5 retained-crop residency handoff",
            0,
            _RENDER_PASS_REQUIREMENT_MS,
            "named-policy",
            (f"retained crop rebind cap={_RETAINED_FALLBACK_REBIND_BATCH_LIMIT}",),
        )

    @_governed
    def decide_render_pass(
        self,
        *,
        interactive: bool,
        pass_kind: str = "preview",
        remaining_items: int | None = None,
        retained_fallback_refinement: bool = False,
        callback_elapsed_ms: float = 0.0,
        frame_critical: bool = False,
        refresh_hz: float = _RENDER_PASS_DEFAULT_REFRESH_HZ,
    ) -> UiWorkDecision:
        """Own R5 chunk size and deadline for preview and target passes.

        The generic UI target is intentionally tight (4/8 ms), but a tiled
        presentation transaction has fixed scene-publication work. Driving
        that channel to 8 ms collapses to one tile without shortening the
        callback. R5 permits a governed chunk up to 50 ms, so the governor
        targets 32 ms and keeps 18 ms of hard-budget margin.

        That 32 ms target is fill-optimal and frame-rate-hostile: it is two
        frames at 60 Hz and four at 144 Hz, so a fill that keeps the GUI
        thread saturated with such chunks cannot also draw. ``frame_critical``
        (a gesture is running, or the display is owed a presentation draw)
        reprices the same objective against ``refresh_hz`` instead. The rate
        is a plain parameter: the governor never asks Qt what the display
        does, its owner does. An idle fill with nothing owed keeps the
        fill-optimal curve exactly as before.
        """

        pass_kind = "target" if str(pass_kind) == "target" else "preview"
        rate_hz = float(refresh_hz)
        rate_hz = rate_hz if rate_hz >= 1.0 else _RENDER_PASS_DEFAULT_REFRESH_HZ
        frame_budget_ms = 1000.0 / rate_hz if bool(frame_critical) else 0.0
        channel = f"montage_render_pass_{pass_kind}"
        byte_cap = 8 * 1024 * 1024 if interactive else 32 * 1024 * 1024
        snapshot = self.latency_feedback.channel_snapshot(channel)
        marker = self._render_pass_observation_markers.get(pass_kind)
        measured = None
        for observation in reversed(self._recent_ui_work_observations):
            if observation is marker:
                break
            if observation.channel == channel:
                measured = observation
                break
        if measured is not None:
            self._render_pass_observation_markers[pass_kind] = measured
        latest_count = (
            int(measured.processed_items) if measured is not None else int(snapshot.last_count)
        )
        latest_elapsed_ms = (
            float(measured.elapsed_ms) if measured is not None else float(snapshot.last_elapsed_ms)
        )
        latest_byte_count = (
            int(measured.processed_bytes) if measured is not None else int(snapshot.last_byte_count)
        )
        max_batch = max(1, int(self.latency_feedback.tuning.max_batch))
        previous_rows = getattr(self, "_render_pass_previous_observations", None)
        if previous_rows is None:
            previous_rows = {}
            self._render_pass_previous_observations = previous_rows
        item_independent = getattr(self, "_render_pass_item_independent", None)
        if item_independent is None:
            item_independent = set()
            self._render_pass_item_independent = item_independent
        structural_key, representation_key = self._render_pass_contexts.get(
            pass_kind, (("default-commit",), ("default-representation",))
        )
        sample_bank = self._render_pass_sample_banks.setdefault(structural_key, {})
        samples = sample_bank.setdefault((pass_kind, representation_key), [])
        previous = previous_rows.get(channel)
        sample = (latest_count, latest_elapsed_ms, latest_byte_count)
        # Equal-valued callbacks are still independent evidence. Identity,
        # rather than tuple equality, distinguishes a new observation from a
        # second decision made before another callback completed.
        new_sample = measured is not None and latest_count > 0
        if (
            previous is not None
            and new_sample
            and latest_count < previous[0]
            and latest_elapsed_ms >= float(previous[1]) * _RENDER_PASS_ITEM_INDEPENDENCE_RATIO
        ):
            item_independent.add(channel)
        round_item_max = getattr(self, "_render_pass_round_item_max", None)
        if round_item_max is None:
            round_item_max = {}
            self._render_pass_round_item_max = round_item_max
        if new_sample:
            samples.append(sample)
            del samples[:-8]
            previous_rows[channel] = (
                latest_count,
                latest_elapsed_ms,
            )
            # A cohort size this round has already measured stays legal
            # evidence for the whole round. The 8-sample window alone forgot
            # an early 32-item commit within eight callbacks, after which the
            # extrapolation guard forbade ever exploring past ~1.1x the small
            # recent cohorts — the self-locking slow fill (2026-07-31).
            round_item_max[channel] = max(int(round_item_max.get(channel, 0)), int(latest_count))
        raw_models = tuple(
            _render_pass_cost_model(rows) for rows in sample_bank.values() if len(rows) >= 2
        )
        retained_models = tuple(
            retained
            for (
                retained_structure,
                _retained_pass,
            ), retained in self._render_pass_last_models.items()
            if retained_structure == structural_key
        )
        shared_fixed_ms, shared_byte_ms_per_mib = _render_pass_shared_terms(
            (*retained_models, *raw_models)
        )
        model = _render_pass_cost_model(
            samples,
            fixed_prior_ms=shared_fixed_ms,
            byte_prior_ms_per_mib=shared_byte_ms_per_mib,
        )
        seed_key = (structural_key, pass_kind, representation_key)
        seed = self._render_pass_seed_models.get(seed_key)
        if seed is not None and len({row[0] for row in samples}) < 2:
            model = _seeded_render_pass_model(
                seed,
                samples=samples,
                shared_fixed_ms=shared_fixed_ms,
                shared_byte_ms_per_mib=shared_byte_ms_per_mib,
            )
        reference_key = (structural_key, pass_kind, representation_key)
        reference = self._render_pass_reference_models.get(reference_key)
        load_offset_ms = _render_pass_recent_load_offset_ms(samples, reference)
        reference_shape_error_ms = _render_pass_reference_shape_error_ms(samples, reference)
        steering_model = (
            model
            if reference is None
            else replace(
                reference,
                fixed_ms=max(0.0, float(reference.fixed_ms or 0.0) + load_offset_ms),
                seeded=False,
            )
        )
        fixed_ms = steering_model.fixed_ms
        item_ms = steering_model.item_ms
        byte_ms_per_mib = steering_model.byte_ms_per_mib
        if fixed_ms is not None:
            mean_elapsed = (
                sum(row[1] for row in samples) / len(samples)
                if samples
                else float(model.mean_elapsed_ms or fixed_ms)
            )
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
        steering_marginal_ms = _render_pass_optimistic_marginal_ms(
            steering_model,
            effective_marginal_ms,
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
        local_count_span = len({row[0] for row in samples})
        if reference is not None:
            self._render_pass_last_models[(structural_key, pass_kind)] = reference
        elif model.identifiable and local_count_span >= 2:
            self._render_pass_last_models[(structural_key, pass_kind)] = model
        if model.identifiable and local_count_span >= 2:
            self._render_pass_last_local_models[reference_key] = model
        if steering_model.identifiable:
            observed_item_max = max(
                1,
                int(steering_model.observed_item_max),
                int(round_item_max.get(channel, 0)),
            )
            # Fixed-dominated evidence means one more item in the chunk is
            # (measured) nearly free: pacing such a transaction just above its
            # observed cohort re-pays the whole-montage fixed cost per tiny
            # chunk. The step multiple stays finite — outward over successive
            # decisions, never a one-shot jump — but grows 4x per decision
            # instead of 1.1x, so a fill escapes a small seed in a few commits
            # rather than never. Byte extrapolation and the byte cap keep
            # pricing the real upload work either way.
            item_exploration_multiple = 4.0 if channel in item_independent else 1.0
            trusted_byte_rows = [
                row
                for row in byte_rows
                if int(steering_model.observed_item_min)
                <= int(row[0])
                <= int(steering_model.observed_item_max)
            ]
            batch, predicted_ms, regularization_ms = _render_pass_optimal_point(
                fixed_ms=max(0.0, float(fixed_ms or 0.0)),
                marginal_ms=steering_marginal_ms,
                remaining_items=remaining,
                observed_item_max=observed_item_max,
                observed_byte_max=(
                    max((int(row[2]) for row in trusted_byte_rows), default=0)
                    if byte_ms_per_mib is not None
                    else 0
                ),
                bytes_per_item=bytes_per_item,
                model=steering_model,
                responsiveness_weight=self.responsiveness_weight,
                item_exploration_multiple=item_exploration_multiple,
                callback_elapsed_ms=callback_elapsed_ms,
                frame_budget_ms=frame_budget_ms,
            )
            steering = "weighted-fill-latency"
            item_ratio = float(batch) / max(
                1.0, float(observed_item_max) * item_exploration_multiple
            )
            extrapolation_ms = _render_pass_extrapolation_cost_ms(
                item_ratio
            ) * _render_pass_extrapolation_weight(steering_model)
            if byte_ms_per_mib is not None and trusted_byte_rows and bytes_per_item > 0.0:
                byte_ratio = (
                    float(batch)
                    * bytes_per_item
                    / max(1.0, *(float(row[2]) for row in trusted_byte_rows))
                )
                extrapolation_ms += _render_pass_extrapolation_cost_ms(
                    byte_ratio
                ) * _render_pass_extrapolation_weight(steering_model)
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
        elif latest_count <= 0:
            # Measure the smallest legal cohort before extrapolating.
            batch = 1
        elif latest_elapsed_ms >= 50.0:
            batch = max(1, latest_count // 2)
        elif latest_elapsed_ms > 32.0:
            batch = max(1, latest_count - 1)
        elif latest_elapsed_ms < 20.0:
            batch = min(max_batch, latest_count + 2)
        else:
            batch = min(max_batch, latest_count + 1)
        if retained_fallback_refinement:
            batch = min(batch, _RETAINED_FALLBACK_REFINEMENT_BATCH_LIMIT)
        return UiWorkDecision(
            channel,
            batch,
            _RENDER_PASS_REQUIREMENT_MS,
            0,
            (
                "R5 governed render-pass target; retained fallback refinement"
                if retained_fallback_refinement
                else "R5 governed render-pass target"
            ),
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
                    f"load-offset={load_offset_ms:.2f}ms "
                    f"shape-error={reference_shape_error_ms:.2f}ms "
                    f"reference={int(reference is not None)}"
                ),
                (
                    f"candidate-fixed={_optional_ms(model.fixed_ms)} "
                    f"candidate-item={_optional_ms(model.item_ms)} "
                    f"candidate-bytes={_optional_ms(model.byte_ms_per_mib)}/MiB"
                ),
                (
                    f"fit-rms={_optional_ms(model.residual_rms_ms)} "
                    f"raw-rms={_optional_ms(model.raw_residual_rms_ms)} "
                    f"outliers={model.outlier_samples} "
                    f"fit-var={_optional_float(model.residual_variance_ms2)}ms2 "
                    f"r2={_optional_float(model.r_squared)} "
                    f"item-byte-independence={_optional_float(model.design_independence)} "
                    f"samples={model.samples} "
                    f"design-points={model.design_points} "
                    f"fit-span={model.observed_item_min}:{model.observed_item_max} "
                    f"raw-span={min((row[0] for row in samples), default=0)}:"
                    f"{max((row[0] for row in samples), default=0)} "
                    f"uncertainty={_render_pass_model_uncertainty(steering_model):.3f} "
                    f"seeded={int(model.seeded)}"
                ),
                "fit-observations="
                + ",".join(
                    f"{count}@{elapsed_ms:.2f}ms@{byte_count}B"
                    for count, elapsed_ms, byte_count in samples
                ),
                (
                    "measured-last=n/a"
                    if measured is None
                    else (
                        f"measured-last={measured.processed_items}@"
                        f"{measured.elapsed_ms:.2f}ms@{measured.processed_bytes}B"
                    )
                ),
                (
                    f"steering={steering} remaining={remaining} "
                    f"predicted={_optional_ms(predicted_ms)} "
                    f"steering-marginal={steering_marginal_ms:.3f}ms/item "
                    f"residual-risk={float(steering_model.residual_rms_ms or 0.0):.2f}ms "
                    f"latency-cost={_optional_ms(regularization_ms)} "
                    f"extrapolation-cost={_optional_ms(extrapolation_ms)} "
                    f"responsiveness-weight={self.responsiveness_weight:.2f} "
                    f"enclosing-callback={max(0.0, float(callback_elapsed_ms)):.2f}ms"
                ),
                (
                    f"control-achievable={_optional_bool(control_achievable)} "
                    f"r5-achievable={_optional_bool(hard_achievable)}"
                ),
                "target=32.00ms hard=50.00ms",
                (
                    f"frame-critical={int(bool(frame_critical))} "
                    f"refresh={rate_hz:.1f}Hz "
                    f"frame-budget={frame_budget_ms:.2f}ms"
                ),
            )
            + (
                (f"retained fallback refinement cap={_RETAINED_FALLBACK_REFINEMENT_BATCH_LIMIT}",)
                if retained_fallback_refinement
                else ()
            ),
        )

    @_governed
    def begin_render_pass(
        self,
        token: object,
        *,
        pass_kind: str = "preview",
        structural_key: object = ("default-commit",),
        representation_key: object = ("default-representation",),
    ) -> None:
        """Start one pass while retaining cost knowledge at its valid scope.

        ``structural_key`` owns the shared fixed/submission and transport-byte
        terms. ``representation_key`` owns the pass-local item term.
        A representation change warm-starts those local terms with deliberately
        high uncertainty; a structural change does not cross-seed them.
        """

        pass_kind = "target" if str(pass_kind) == "target" else "preview"
        structural_key = _hashable_render_pass_key(structural_key)
        representation_key = _hashable_render_pass_key(representation_key)
        context = (structural_key, representation_key)
        self._render_pass_contexts[pass_kind] = context
        if (
            structural_key not in self._render_pass_sample_banks
            and len(self._render_pass_sample_banks) >= 8
        ):
            active_structures = {value[0] for value in self._render_pass_contexts.values()}
            stale_structure = next(
                (key for key in self._render_pass_sample_banks if key not in active_structures),
                next(iter(self._render_pass_sample_banks)),
            )
            self._render_pass_sample_banks.pop(stale_structure, None)
            self._render_pass_seed_models = {
                key: value
                for key, value in self._render_pass_seed_models.items()
                if key[0] != stale_structure
            }
            self._render_pass_last_models = {
                key: value
                for key, value in self._render_pass_last_models.items()
                if key[0] != stale_structure
            }
            self._render_pass_reference_models = {
                key: value
                for key, value in self._render_pass_reference_models.items()
                if key[0] != stale_structure
            }
            self._render_pass_last_local_models = {
                key: value
                for key, value in self._render_pass_last_local_models.items()
                if key[0] != stale_structure
            }
        sample_bank = self._render_pass_sample_banks.setdefault(structural_key, {})
        group_key = (pass_kind, representation_key)
        if group_key not in sample_bank:
            if len(sample_bank) >= 8:
                active_groups = {
                    (kind, value[1])
                    for kind, value in self._render_pass_contexts.items()
                    if value[0] == structural_key
                }
                stale_group = next(
                    (key for key in sample_bank if key not in active_groups),
                    next(iter(sample_bank)),
                )
                sample_bank.pop(stale_group, None)
                self._render_pass_seed_models.pop(
                    (structural_key, stale_group[0], stale_group[1]), None
                )
                self._render_pass_reference_models.pop(
                    (structural_key, stale_group[0], stale_group[1]), None
                )
                self._render_pass_last_local_models.pop(
                    (structural_key, stale_group[0], stale_group[1]), None
                )
            sample_bank[group_key] = []
        tokens = getattr(self, "_render_pass_tokens", None)
        if tokens is None:
            tokens = {}
            self._render_pass_tokens = tokens
        token_with_context = (token, context)
        if tokens.get(pass_kind) == token_with_context:
            return
        local_key = (structural_key, pass_kind, representation_key)
        previous_local_model = self._render_pass_last_local_models.get(local_key)
        if previous_local_model is not None and _render_pass_reference_model_is_stable(
            previous_local_model
        ):
            self._render_pass_reference_models[local_key] = previous_local_model
        previous_model = self._render_pass_last_models.get((structural_key, pass_kind))
        if previous_model is not None:
            self._render_pass_seed_models[(structural_key, pass_kind, representation_key)] = (
                previous_model
            )
        # Carry learned parameters, not stale raw observations. A new round
        # receives the prior model as a high-uncertainty seed and gathers its
        # own local evidence, so yesterday's eight-row cohort distribution
        # cannot take eight callbacks to age out.
        sample_bank[group_key] = []
        tokens[pass_kind] = token_with_context
        channel = f"montage_render_pass_{pass_kind}"
        self._render_pass_observation_markers[pass_kind] = (
            self._recent_ui_work_observations[-1] if self._recent_ui_work_observations else None
        )
        self.latency_feedback.reset_channel(channel)
        getattr(self, "_render_pass_previous_observations", {}).pop(channel, None)
        getattr(self, "_render_pass_item_independent", set()).discard(channel)
        getattr(self, "_render_pass_round_item_max", {}).pop(channel, None)

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

    @_governed
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


def _hashable_render_pass_key(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _render_pass_cost_model(
    samples: list[tuple[int, float, int]],
    *,
    fixed_prior_ms: float | None = None,
    byte_prior_ms_per_mib: float | None = None,
) -> _RenderPassCostModel:
    """Measure fixed, item, and byte-dependent transaction cost.

    Item and byte terms are reported separately only when the observed
    cohorts vary independently enough to identify both. Proportional
    item/byte samples deliberately fall back to the one-dimensional item fit
    instead of inventing a split.
    """

    if len(samples) < 2:
        if samples and fixed_prior_ms is not None:
            count, elapsed_ms, byte_count = samples[0]
            fixed = max(0.0, float(fixed_prior_ms))
            byte_ms_per_mib = (
                None if byte_prior_ms_per_mib is None else max(0.0, float(byte_prior_ms_per_mib))
            )
            byte_cost_ms = float(byte_ms_per_mib or 0.0) * float(byte_count) / _MIB
            item_ms = max(0.0, (float(elapsed_ms) - fixed - byte_cost_ms) / max(1, count))
            return _RenderPassCostModel(
                fixed_ms=fixed,
                item_ms=item_ms,
                byte_ms_per_mib=byte_ms_per_mib,
                # One observation fits exactly by construction; that is not
                # evidence that the local marginal is known without error.
                residual_variance_ms2=None,
                residual_rms_ms=None,
                samples=1,
                design_points=1,
                observed_item_min=int(count),
                observed_item_max=int(count),
                mean_elapsed_ms=float(elapsed_ms),
                seeded=True,
            )
        return _RenderPassCostModel(
            samples=len(samples),
            design_points=len({(row[0], row[2]) for row in samples}),
            observed_item_min=min((row[0] for row in samples), default=0),
            observed_item_max=max((row[0] for row in samples), default=0),
            mean_elapsed_ms=(sum(row[1] for row in samples) / len(samples) if samples else None),
        )
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
    robust_linear = _robust_linear_fit(counts, elapsed)
    item_ms = None if robust_linear is None else robust_linear[1]
    byte_ms_per_mib = None
    byte_identified = False
    fixed = max(0.0, mean_elapsed) if robust_linear is None else robust_linear[0]
    item_only_residual_sum = sum(
        (duration - (fixed + float(item_ms or 0.0) * count)) ** 2
        for count, duration in zip(counts, elapsed, strict=True)
    )

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
    design_independence = (
        max(0.0, min(1.0, determinant / scale))
        if item_variance > 0.0 and byte_variance > 0.0
        else None
    )
    if len(samples) >= 4 and design_independence is not None and design_independence >= 0.05:
        fitted_item = (item_covariance * byte_variance - byte_covariance * cross) / determinant
        fitted_byte = (byte_covariance * item_variance - item_covariance * cross) / determinant
        fitted_fixed = mean_elapsed - fitted_item * mean_count - fitted_byte * mean_mib
        if fitted_item >= 0.0 and fitted_byte >= 0.0 and fitted_fixed >= 0.0:
            candidate_residual_sum = sum(
                (duration - (fitted_fixed + fitted_item * count + fitted_byte * byte_mib)) ** 2
                for count, byte_mib, duration in zip(counts, mib, elapsed, strict=True)
            )
            if candidate_residual_sum <= 0.9 * item_only_residual_sum:
                fixed = fitted_fixed
                item_ms = fitted_item
                byte_ms_per_mib = fitted_byte
                byte_identified = True
    elif item_variance <= 0.0 and byte_variance > 0.0:
        byte_ms_per_mib = max(0.0, byte_covariance / byte_variance)
        byte_identified = True
        fixed = max(0.0, mean_elapsed - byte_ms_per_mib * mean_mib)

    # Shared structural knowledge is a prior with the correct ownership:
    # fixed/submission and bytes/transport cross pass and representation
    # boundaries, while item cost is re-estimated locally.
    if fixed_prior_ms is not None or byte_prior_ms_per_mib is not None:
        if byte_prior_ms_per_mib is not None:
            byte_ms_per_mib = max(0.0, float(byte_prior_ms_per_mib))
            # The value is now the pool's, not this design's.
            byte_identified = False
        transport_adjusted = [
            duration - float(byte_ms_per_mib or 0.0) * value
            for duration, value in zip(elapsed, mib, strict=True)
        ]
        if len(set(counts)) >= 2:
            local_linear = _robust_linear_fit(counts, transport_adjusted)
            if local_linear is not None:
                fixed, item_ms = local_linear
        else:
            fixed = max(0.0, float(fixed_prior_ms if fixed_prior_ms is not None else fixed))
            residual_targets = [target - fixed for target in transport_adjusted]
            numerator = sum(
                count * target for count, target in zip(counts, residual_targets, strict=True)
            )
            denominator = sum(count * count for count in counts)
            item_ms = max(0.0, numerator / max(1e-12, denominator))
    predicted = [
        fixed + float(item_ms or 0.0) * count + float(byte_ms_per_mib or 0.0) * value
        for count, value in zip(counts, mib, strict=True)
    ]
    raw_residual_sum = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(elapsed, predicted, strict=True)
    )
    absolute_residuals = [
        abs(actual - estimate) for actual, estimate in zip(elapsed, predicted, strict=True)
    ]
    inlier_threshold_ms = max(1.0, 3.0 * float(median(absolute_residuals)))
    inlier_indices = tuple(
        index
        for index, residual in enumerate(absolute_residuals)
        if residual <= inlier_threshold_ms
    )
    if not inlier_indices:
        inlier_indices = tuple(range(len(samples)))
    inlier_counts = [counts[index] for index in inlier_indices]
    inlier_mib = [mib[index] for index in inlier_indices]
    residual_sum = sum((elapsed[index] - predicted[index]) ** 2 for index in inlier_indices)
    total_sum = sum((actual - mean_elapsed) ** 2 for actual in elapsed)
    residual_variance_ms2 = residual_sum / len(inlier_indices)
    return _RenderPassCostModel(
        fixed_ms=fixed,
        item_ms=item_ms,
        byte_ms_per_mib=byte_ms_per_mib,
        byte_identified=byte_identified,
        residual_variance_ms2=residual_variance_ms2,
        residual_rms_ms=residual_variance_ms2**0.5,
        raw_residual_rms_ms=(raw_residual_sum / len(samples)) ** 0.5,
        outlier_samples=len(samples) - len(inlier_indices),
        r_squared=(1.0 - raw_residual_sum / total_sum) if total_sum > 0.0 else None,
        design_independence=design_independence,
        samples=len(samples),
        design_points=len(set(zip(inlier_counts, inlier_mib, strict=True))),
        observed_item_min=int(min(inlier_counts)),
        observed_item_max=int(max(inlier_counts)),
        mean_elapsed_ms=mean_elapsed,
        seeded=bool(
            len(set(counts)) < 2
            and (fixed_prior_ms is not None or byte_prior_ms_per_mib is not None)
        ),
    )


def _render_pass_reference_model_is_stable(model: _RenderPassCostModel) -> bool:
    """Return whether a fit can anchor the intrinsic cost surface.

    Three or more distinct work shapes separate fixed from dynamic cost.  A
    fourth callback prevents the exact three-point interpolation that made a
    single stalled probe look certain.  The residual limit is deliberately
    relative as backend timing noise grows with callback duration.

    The slope must additionally have been measured across a real cohort
    span.  A reference's item slope is frozen for the whole next round, and
    a slope extrapolated from a 1-4 item span steered rounds that run 8-30
    item cohorts: too-large slopes forbade exactly the cohorts that would
    have falsified them, and 3-5 item commits could never re-open the range
    (the 2026-07-31 slow-fill attractor held item=4.3 ms/item for 70
    straight commits from one such reference).  A fit without that span
    still steers its own round locally; it just may not anchor the next.
    """

    return bool(
        model.identifiable
        and model.samples >= 4
        and model.design_points >= 3
        and model.residual_rms_ms is not None
        and float(model.residual_rms_ms) <= max(2.0, 0.05 * float(model.mean_elapsed_ms or 0.0))
        and int(model.observed_item_max) >= 8
        and int(model.observed_item_max) >= 4 * max(1, int(model.observed_item_min))
    )


def _render_pass_recent_load_offset_ms(
    samples: list[tuple[int, float, int]],
    reference: _RenderPassCostModel | None,
) -> float:
    """Measure current machine delay without rewriting intrinsic cost terms.

    A three-observation median is the smallest causal filter that rejects one
    arbitrary scheduling stall yet accepts a sustained regime after the next
    callback.  The same two-of-three rule makes recovery equally fast.
    """

    if reference is None or len(samples) < 3:
        return 0.0
    recent_residuals = []
    for count, elapsed_ms, byte_count in samples[-3:]:
        predicted_ms = (
            float(reference.fixed_ms or 0.0)
            + float(reference.item_ms or 0.0) * float(count)
            + float(reference.byte_ms_per_mib or 0.0) * float(byte_count) / _MIB
        )
        recent_residuals.append(float(elapsed_ms) - predicted_ms)
    return float(median(recent_residuals))


def _render_pass_reference_shape_error_ms(
    samples: list[tuple[int, float, int]],
    reference: _RenderPassCostModel | None,
) -> float:
    """Measure residual variation after removing an item-independent delay."""

    if reference is None or len(samples) < 3:
        return 0.0
    residuals = []
    for count, elapsed_ms, byte_count in samples:
        predicted_ms = (
            float(reference.fixed_ms or 0.0)
            + float(reference.item_ms or 0.0) * float(count)
            + float(reference.byte_ms_per_mib or 0.0) * float(byte_count) / _MIB
        )
        residuals.append(float(elapsed_ms) - predicted_ms)
    centre = float(median(residuals))
    return (sum((residual - centre) ** 2 for residual in residuals) / len(residuals)) ** 0.5


def _robust_linear_fit(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, float] | None:
    """Fit fixed + item cost robustly while following a changing machine.

    The repeated-median seed resists a high-leverage stalled callback, then
    exponentially weighted Cauchy IRLS follows a sustained cost change.  The
    influence of a residual falls smoothly rather than classifying a callback
    as good or bad, and recent evidence can still outweigh an old regime.
    """

    if len(x_values) != len(y_values) or not x_values:
        return None
    if len(set(x_values)) < 2:
        return None
    recency = [0.72 ** (len(x_values) - index - 1) for index in range(len(x_values))]

    point_slopes: list[float] = []
    for index, x_value in enumerate(x_values):
        slopes = [
            (y_values[other] - y_values[index]) / (x_values[other] - x_value)
            for other in range(len(x_values))
            if x_values[other] != x_value
        ]
        if slopes:
            point_slopes.append(float(median(slopes)))
    initial_slope = max(0.0, float(median(point_slopes)))
    estimate = (
        max(
            0.0,
            float(
                median(
                    target - initial_slope * value
                    for value, target in zip(x_values, y_values, strict=True)
                )
            ),
        ),
        initial_slope,
    )

    def solve(weights: list[float]) -> tuple[float, float]:
        total_weight = sum(weights)
        mean_x = sum(weight * value for weight, value in zip(weights, x_values, strict=True))
        mean_x /= max(1e-12, total_weight)
        mean_y = sum(weight * target for weight, target in zip(weights, y_values, strict=True))
        mean_y /= max(1e-12, total_weight)
        variance = sum(
            weight * (value - mean_x) ** 2 for weight, value in zip(weights, x_values, strict=True)
        )
        covariance = sum(
            weight * (value - mean_x) * (target - mean_y)
            for weight, value, target in zip(weights, x_values, y_values, strict=True)
        )
        candidates = [(max(0.0, mean_y), 0.0)]
        if variance > 1e-12:
            slope = covariance / variance
            intercept = mean_y - slope * mean_x
            if slope >= 0.0 and intercept >= 0.0:
                candidates.append((float(intercept), float(slope)))
        weighted_x2 = sum(
            weight * value * value for weight, value in zip(weights, x_values, strict=True)
        )
        if weighted_x2 > 1e-12:
            origin_slope = (
                sum(
                    weight * value * target
                    for weight, value, target in zip(weights, x_values, y_values, strict=True)
                )
                / weighted_x2
            )
            candidates.append((0.0, max(0.0, float(origin_slope))))

        def weighted_residual(candidate: tuple[float, float]) -> float:
            intercept, slope = candidate
            return sum(
                weight * (target - intercept - slope * value) ** 2
                for weight, value, target in zip(
                    weights,
                    x_values,
                    y_values,
                    strict=True,
                )
            )

        return min(candidates, key=weighted_residual)

    for _ in range(12):
        intercept, slope = estimate
        residuals = [
            target - intercept - slope * value
            for value, target in zip(x_values, y_values, strict=True)
        ]
        residual_centre = float(median(residuals))
        scale = max(
            0.5,
            1.4826 * float(median(abs(value - residual_centre) for value in residuals)),
        )
        cauchy_width = 2.385 * scale
        weights = [
            recency_weight / (1.0 + ((residual - residual_centre) / cauchy_width) ** 2)
            for recency_weight, residual in zip(recency, residuals, strict=True)
        ]
        updated = solve(weights)
        if max(abs(updated[index] - estimate[index]) for index in (0, 1)) <= 1e-6:
            estimate = updated
            break
        estimate = updated
    return estimate


def _render_pass_shared_terms(
    models: tuple[_RenderPassCostModel, ...],
) -> tuple[float | None, float | None]:
    """Pool only terms whose physical owner crosses pass/representation."""

    def robust_centre(attribute: str) -> float | None:
        rows = []
        for model in models:
            value = getattr(model, attribute)
            if value is None:
                continue
            if (
                attribute == "fixed_ms"
                and model.observed_item_max <= model.observed_item_min
                and model.byte_ms_per_mib is None
            ):
                continue
            if attribute == "byte_ms_per_mib" and not model.byte_identified:
                # An inherited byte value re-entering the pool is a feedback
                # loop, not evidence: one misattributed split then becomes
                # every later pass's marginal cost and pins small cohorts.
                continue
            rows.append(float(value))
        if not rows:
            return None
        rows.sort()
        midpoint = len(rows) // 2
        if len(rows) % 2:
            return rows[midpoint]
        return 0.5 * (rows[midpoint - 1] + rows[midpoint])

    return robust_centre("fixed_ms"), robust_centre("byte_ms_per_mib")


def _seeded_render_pass_model(
    seed: _RenderPassCostModel,
    *,
    samples: list[tuple[int, float, int]],
    shared_fixed_ms: float | None,
    shared_byte_ms_per_mib: float | None,
) -> _RenderPassCostModel:
    """Warm-start representation-local terms, explicitly at high uncertainty."""

    byte_ms_per_mib = (
        shared_byte_ms_per_mib if shared_byte_ms_per_mib is not None else seed.byte_ms_per_mib
    )
    if samples:
        fixed_candidates = [
            elapsed_ms
            - float(seed.item_ms or 0.0) * count
            - float(byte_ms_per_mib or 0.0) * byte_count / _MIB
            for count, elapsed_ms, byte_count in samples
        ]
        fixed_ms = max(0.0, float(median(fixed_candidates)))
    else:
        fixed_ms = shared_fixed_ms if shared_fixed_ms is not None else seed.fixed_ms
    return _RenderPassCostModel(
        fixed_ms=fixed_ms,
        item_ms=seed.item_ms,
        byte_ms_per_mib=byte_ms_per_mib,
        residual_variance_ms2=None,
        residual_rms_ms=None,
        r_squared=None,
        samples=len(samples),
        design_points=len({(row[0], row[2]) for row in samples}),
        observed_item_min=min((row[0] for row in samples), default=0),
        observed_item_max=max((row[0] for row in samples), default=0),
        mean_elapsed_ms=(sum(row[1] for row in samples) / len(samples) if samples else None),
        seeded=True,
    )


def _render_pass_latency_cost_ms(
    elapsed_ms: float,
    *,
    knee_ms: float = _RENDER_PASS_LATENCY_KNEE_MS,
    steep_ms: float = _RENDER_PASS_LATENCY_STEEP_MS,
) -> float:
    """Continuous responsiveness price in absolute milliseconds.

    Work below the smooth-interaction envelope (``knee_ms``, 18 ms by default)
    is unregularized. The price rises gently to ``steep_ms`` (45 ms), then
    joins a stronger quadratic with matching value and first derivative. There
    is deliberately no branch at 50 ms: R5 is reported there, while
    optimization sees the whole smooth curve.

    The two knees are the only frame-awareness knob. Moving them down to a
    refresh period reprices the SAME milliseconds more steeply rather than
    adding a second charge for them, so a caller that also prices an enclosing
    callback cannot pay twice for one overrun.
    """

    elapsed = max(0.0, float(elapsed_ms))
    knee = max(0.0, float(knee_ms))
    steep = max(knee + 1.0, float(steep_ms))
    if elapsed <= knee:
        return 0.0
    gentle = 0.01
    if elapsed <= steep:
        return gentle * (elapsed - knee) ** 2
    at_steep = gentle * (steep - knee) ** 2
    slope_at_steep = 2.0 * gentle * (steep - knee)
    beyond = elapsed - steep
    return at_steep + slope_at_steep * beyond + 0.5 * beyond**2


def _render_pass_frame_aware_knees(
    frame_budget_ms: float,
    fixed_ms: float,
    minimum_chunk_ms: float,
) -> tuple[float, float] | None:
    """Responsiveness knees for a pass that owes the display a frame.

    A frame-critical pass is priced against the refresh period instead of the
    fill-optimal 18/45 envelope: the knee drops to half a frame and the steep
    branch starts at one frame, so a chunk that costs more than the display's
    own cadence is charged as the missed frames it is.

    ``None`` when a frame is not for sale. Every chunk re-pays the whole fixed
    term, so pricing a pass into frame-sized chunks multiplies that term by the
    chunk count -- R5's cohort collapse, arriving through the responsiveness
    price instead of through the budget. The frame must therefore be able to
    hold at least two of the fixed cost, or shortening the chunk spends more
    fill time than the frame it buys is worth. Measured 2026-08-02 on the cold
    272-tile WGPU montage, whose commit carries ~11 ms of fixed cost against a
    16.7 ms frame: without this refusal the profiler's 5 s window presented 34
    tiles against 266 with the fill-optimal curve.

    ``minimum_chunk_ms`` floors the steep point at the cheapest chunk the model
    can produce, so the smallest legal chunk is never itself priced as an
    overrun; the overrun is reported rather than chunked away.
    """

    budget = max(0.0, float(frame_budget_ms))
    if budget <= 0.0 or budget < 2.0 * max(0.0, float(fixed_ms)):
        return None
    steep = max(budget, float(minimum_chunk_ms))
    return 0.5 * steep, steep


def _render_pass_extrapolation_cost_ms(ratio: float) -> float:
    """Smooth evidence-distance price: free to 1.1x, exponential thereafter."""

    ratio = max(0.0, float(ratio))
    if ratio <= 1.1:
        return 0.0
    decades = log10(ratio / 1.1)
    return 5.0 * (exp(2.0 * decades) - 1.0) ** 2


def _render_pass_model_uncertainty(model: _RenderPassCostModel) -> float:
    """Dimensionless fit uncertainty from sample count, span, and residuals."""

    # Repeating one cohort improves the noise estimate, but supplies no new
    # information about the item/byte slopes. Exploration confidence therefore
    # follows independent design points, not the raw observation count.
    design_points = max(
        1,
        int(
            model.design_points
            or min(model.samples, 2 if model.observed_item_max > model.observed_item_min else 1)
        ),
    )
    sample_uncertainty = max(0.0, (8.0 - float(design_points)) / 2.0)
    if model.observed_item_min > 0 and model.observed_item_max > 0:
        span = float(model.observed_item_max) / float(model.observed_item_min)
    else:
        span = 1.0
    span_uncertainty = max(0.0, 3.0 - log2(max(1.0, span)))
    residual_uncertainty = (
        2.0
        if model.residual_rms_ms is None or design_points < 2
        else min(
            4.0,
            8.0 * float(model.residual_rms_ms) / max(1.0, float(model.mean_elapsed_ms or 0.0)),
        )
    )
    if model.seeded:
        sample_uncertainty = max(sample_uncertainty, 4.0)
    return sample_uncertainty + span_uncertainty + residual_uncertainty


def _render_pass_optimistic_marginal_ms(
    model: _RenderPassCostModel,
    measured_marginal_ms: float,
) -> float:
    """Return the marginal cost justified by the locally measured span.

    A shared fixed/transport prior plus one local observation can explain the
    observation, but it cannot identify how much of the remainder is truly
    per-item. Treating that algebraic remainder as certain pins the analytic
    optimum at one and prevents the experiment that would distinguish fixed
    from dynamic cost. Until two local cohort sizes have been measured, the
    lower confidence bound of that marginal is zero. Once a slope is measured,
    the fitted value owns steering and the smooth extrapolation price owns how
    far beyond its evidence the next decision may move.
    """

    if model.observed_item_max <= model.observed_item_min:
        return 0.0
    return max(0.0, float(measured_marginal_ms))


def _render_pass_extrapolation_weight(model: _RenderPassCostModel) -> float:
    """Optimism under uncertainty without turning extrapolation into a gate."""

    uncertainty = _render_pass_model_uncertainty(model)
    return 1.0 / (1.0 + uncertainty)


def _render_pass_optimal_point(
    *,
    fixed_ms: float,
    marginal_ms: float,
    remaining_items: int,
    observed_item_max: int,
    observed_byte_max: int,
    bytes_per_item: float,
    model: _RenderPassCostModel | None = None,
    responsiveness_weight: float = 1.0,
    item_exploration_multiple: float = 1.0,
    callback_elapsed_ms: float = 0.0,
    frame_budget_ms: float = 0.0,
) -> tuple[int, float, float]:
    """Minimize fill plus latency and model-risk prices in logarithmic work.

    The continuous relaxation is convex for the measured non-negative cost
    terms. A bisection root of its marginal cost locates the knee; a small set
    of neighbouring integer and chunk-boundary candidates restores the exact
    ``ceil(remaining / cohort)`` semantics. This replaces the former
    ``range(1, remaining + 1)`` scan.
    """

    remaining = max(1, int(remaining_items))
    marginal = max(0.0, float(marginal_ms))
    responsiveness = max(0.0, float(responsiveness_weight))
    uncertainty_model = model or _RenderPassCostModel(
        fixed_ms=float(fixed_ms),
        item_ms=marginal,
        samples=8,
        design_points=8,
        observed_item_min=1,
        observed_item_max=max(1, int(observed_item_max)),
        mean_elapsed_ms=float(fixed_ms) + marginal,
    )
    residual_risk_ms = max(0.0, float(uncertainty_model.residual_rms_ms or 0.0))

    def latency(items: float) -> float:
        count = float(items)
        return float(fixed_ms) + marginal * count

    frame_knees = _render_pass_frame_aware_knees(
        frame_budget_ms,
        fixed_ms,
        latency(1.0) + residual_risk_ms,
    )
    knee_ms, steep_ms = frame_knees or (
        _RENDER_PASS_LATENCY_KNEE_MS,
        _RENDER_PASS_LATENCY_STEEP_MS,
    )

    def responsiveness_price_ms(chunk_ms: float) -> float:
        return _render_pass_latency_cost_ms(chunk_ms, knee_ms=knee_ms, steep_ms=steep_ms)

    exploration_multiple = max(1.0, float(item_exploration_multiple))

    def extrapolation(items: float) -> float:
        item_ratio = float(items) / max(1.0, float(observed_item_max) * exploration_multiple)
        result = _render_pass_extrapolation_cost_ms(item_ratio) * _render_pass_extrapolation_weight(
            uncertainty_model
        )
        if observed_byte_max > 0 and bytes_per_item > 0.0:
            byte_ratio = float(items) * float(bytes_per_item) / float(observed_byte_max)
            result += _render_pass_extrapolation_cost_ms(
                byte_ratio
            ) * _render_pass_extrapolation_weight(uncertainty_model)
        return result

    enclosing_ms = max(0.0, float(callback_elapsed_ms))

    def enclosing_cost(risk_adjusted_chunk_ms: float) -> float:
        """Responsiveness the FIRST chunk owes for the callback it sits in.

        R5's ceiling is on what a GUI-thread callback blocks for, and a pass
        that runs inside another callback -- an R6 shed remap presenting from
        inside the retarget that produced it -- does not start its blocking at
        zero. Only the first chunk pays this: the rest run in callbacks of
        their own, which is exactly the choice the price is meant to buy.
        Zero when nothing encloses the pass, so every other caller is
        arithmetically unchanged.

        It reads the same (possibly frame-shifted) curve as the per-chunk
        price and charges only the DIFFERENCE the enclosing milliseconds make,
        so a frame-critical pass prices those milliseconds once, not twice.
        """

        if enclosing_ms <= 0.0:
            return 0.0
        return responsiveness * (
            responsiveness_price_ms(enclosing_ms + risk_adjusted_chunk_ms)
            - responsiveness_price_ms(risk_adjusted_chunk_ms)
        )

    def relaxed_objective(items: float) -> float:
        chunk_ms = latency(items)
        risk_adjusted_chunk_ms = chunk_ms + residual_risk_ms
        return (
            float(remaining)
            / max(1.0, float(items))
            * (
                risk_adjusted_chunk_ms
                + responsiveness * responsiveness_price_ms(risk_adjusted_chunk_ms)
            )
            + enclosing_cost(risk_adjusted_chunk_ms)
            + extrapolation(items)
        )

    def derivative(items: float) -> float:
        step = max(1e-4, float(items) * 1e-4)
        low = max(1.0, float(items) - step)
        high = min(float(remaining), float(items) + step)
        if high <= low:
            return 0.0
        return (relaxed_objective(high) - relaxed_objective(low)) / (high - low)

    low = 1.0
    high = float(remaining)
    if derivative(low) >= 0.0:
        root = low
    elif derivative(high) <= 0.0:
        root = high
    else:
        for _ in range(56):
            midpoint = 0.5 * (low + high)
            if derivative(midpoint) < 0.0:
                low = midpoint
            else:
                high = midpoint
        root = 0.5 * (low + high)

    candidates = {1, remaining, max(1, int(observed_item_max))}
    centre = max(1, min(remaining, round(root)))
    candidates.update(max(1, min(remaining, centre + offset)) for offset in range(-4, 5))
    root_chunks = max(1, round(float(remaining) / max(1.0, root)))
    for chunk_delta in range(-8, 9):
        chunks = max(1, root_chunks + chunk_delta)
        candidates.add(max(1, min(remaining, ceil(float(remaining) / chunks))))

    rows: list[tuple[float, float, int, float]] = []
    for items in candidates:
        chunks = ceil(float(remaining) / float(items))
        chunk_ms = latency(items)
        risk_adjusted_chunk_ms = chunk_ms + residual_risk_ms
        latency_cost = responsiveness * responsiveness_price_ms(risk_adjusted_chunk_ms)
        extrapolation_cost = extrapolation(items)
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
                chunks * (risk_adjusted_chunk_ms + latency_cost)
                + enclosing_cost(risk_adjusted_chunk_ms)
                + extrapolation_cost,
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
