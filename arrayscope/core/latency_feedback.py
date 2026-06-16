"""Small feedback controller for UI-thread work budgets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatencyFeedbackTuning:
    target_idle_ms: float = 8.0
    target_interactive_ms: float = 4.0
    min_budget_ms: float = 2.0
    max_budget_ms: float = 12.0
    min_interval_ms: int = 8
    max_interval_ms: int = 250
    min_batch: int = 1
    max_batch: int = 8
    ewma_alpha: float = 0.25


@dataclass
class LatencyFeedbackChannel:
    elapsed_ewma_ms: float | None = None
    per_item_ewma_ms: float | None = None
    last_elapsed_ms: float = 0.0
    last_count: int = 0


@dataclass(frozen=True)
class LatencyFeedbackChannelSnapshot:
    channel: str
    elapsed_ewma_ms: float | None
    per_item_ewma_ms: float | None
    last_elapsed_ms: float
    last_count: int


@dataclass
class LatencyFeedbackController:
    tuning: LatencyFeedbackTuning = field(default_factory=LatencyFeedbackTuning)
    _channels: dict[str, LatencyFeedbackChannel] = field(default_factory=dict)

    def observe(self, channel: str, elapsed_ms: float, *, count: int = 1) -> None:
        state = self._channels.setdefault(str(channel), LatencyFeedbackChannel())
        elapsed = max(0.0, float(elapsed_ms))
        count = max(1, int(count))
        state.last_elapsed_ms = elapsed
        state.last_count = count
        state.elapsed_ewma_ms = _ewma(state.elapsed_ewma_ms, elapsed, self.tuning.ewma_alpha)
        state.per_item_ewma_ms = _ewma(state.per_item_ewma_ms, elapsed / count, self.tuning.ewma_alpha)

    def work_budget_ms(self, channel: str, *, interactive: bool = False) -> float:
        state = self._channels.get(str(channel))
        target = self._target(interactive)
        if state is None or state.elapsed_ewma_ms is None or state.elapsed_ewma_ms <= 0.0:
            return _clamp(target, self.tuning.min_budget_ms, self.tuning.max_budget_ms)
        ratio = target / max(state.elapsed_ewma_ms, 0.25)
        return _clamp(target * ratio, self.tuning.min_budget_ms, self.tuning.max_budget_ms)

    def batch_limit(self, channel: str, *, interactive: bool = False) -> int:
        state = self._channels.get(str(channel))
        target = self._target(interactive)
        if state is None or state.per_item_ewma_ms is None or state.per_item_ewma_ms <= 0.0:
            return int(self.tuning.max_batch)
        limit = int(max(1, target // max(state.per_item_ewma_ms, 0.25)))
        return max(int(self.tuning.min_batch), min(int(self.tuning.max_batch), limit))

    def commit_interval_ms(self, channel: str, *, force: bool = False, interactive: bool = False) -> int:
        if force:
            return int(self.tuning.min_interval_ms)
        state = self._channels.get(str(channel))
        target = self._target(interactive)
        if state is None or state.elapsed_ewma_ms is None or state.elapsed_ewma_ms <= target:
            return max(int(self.tuning.min_interval_ms), int(round(target * 2.0)))
        interval = int(round(max(target * 2.0, state.elapsed_ewma_ms * 2.0)))
        return max(int(self.tuning.min_interval_ms), min(int(self.tuning.max_interval_ms), interval))

    def channel_snapshot(self, channel: str) -> LatencyFeedbackChannelSnapshot:
        name = str(channel)
        state = self._channels.get(name)
        if state is None:
            return LatencyFeedbackChannelSnapshot(name, None, None, 0.0, 0)
        return LatencyFeedbackChannelSnapshot(
            channel=name,
            elapsed_ewma_ms=state.elapsed_ewma_ms,
            per_item_ewma_ms=state.per_item_ewma_ms,
            last_elapsed_ms=float(state.last_elapsed_ms),
            last_count=int(state.last_count),
        )

    def snapshots(self) -> tuple[LatencyFeedbackChannelSnapshot, ...]:
        return tuple(self.channel_snapshot(channel) for channel in sorted(self._channels))

    def _target(self, interactive: bool) -> float:
        return float(self.tuning.target_interactive_ms if interactive else self.tuning.target_idle_ms)


def _ewma(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return float(value)
    alpha = _clamp(float(alpha), 0.01, 1.0)
    return float(previous) * (1.0 - alpha) + float(value) * alpha


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))
