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
    per_byte_ewma_ms: float | None = None
    last_elapsed_ms: float = 0.0
    last_count: int = 0
    last_byte_count: int = 0
    count_ewma: float | None = None
    count_var_ewma: float = 0.0
    count_elapsed_cov_ewma: float = 0.0
    byte_ewma: float | None = None
    byte_var_ewma: float = 0.0
    byte_elapsed_cov_ewma: float = 0.0
    byte_observations: int = 0
    observations: int = 0


@dataclass(frozen=True)
class LatencyFeedbackChannelSnapshot:
    channel: str
    elapsed_ewma_ms: float | None
    per_item_ewma_ms: float | None
    per_byte_ewma_ms: float | None
    last_elapsed_ms: float
    last_count: int
    last_byte_count: int
    overhead_ewma_ms: float | None = None
    marginal_per_item_ms: float | None = None


@dataclass
class LatencyFeedbackController:
    tuning: LatencyFeedbackTuning = field(default_factory=LatencyFeedbackTuning)
    _channels: dict[str, LatencyFeedbackChannel] = field(default_factory=dict)

    def observe(self, channel: str, elapsed_ms: float, *, count: int = 1, byte_count: int = 0) -> None:
        state = self._channels.setdefault(str(channel), LatencyFeedbackChannel())
        elapsed = max(0.0, float(elapsed_ms))
        count = max(1, int(count))
        byte_count = max(0, int(byte_count))
        state.last_elapsed_ms = elapsed
        state.last_count = count
        state.last_byte_count = byte_count
        alpha = _clamp(float(self.tuning.ewma_alpha), 0.01, 1.0)
        # Exponentially-weighted first and second moments of (count, elapsed)
        # support a per-call-overhead + per-item-marginal cost model; deltas
        # are taken against the pre-update means.
        if state.count_ewma is None or state.elapsed_ewma_ms is None:
            state.count_ewma = float(count)
            state.count_var_ewma = 0.0
            state.count_elapsed_cov_ewma = 0.0
        else:
            delta_count = float(count) - float(state.count_ewma)
            delta_elapsed = elapsed - float(state.elapsed_ewma_ms)
            state.count_ewma = float(state.count_ewma) + alpha * delta_count
            state.count_var_ewma = (1.0 - alpha) * (state.count_var_ewma + alpha * delta_count * delta_count)
            state.count_elapsed_cov_ewma = (1.0 - alpha) * (state.count_elapsed_cov_ewma + alpha * delta_count * delta_elapsed)
        if byte_count > 0:
            if state.byte_ewma is None or state.elapsed_ewma_ms is None:
                state.byte_ewma = float(byte_count)
                state.byte_var_ewma = 0.0
                state.byte_elapsed_cov_ewma = 0.0
            else:
                delta_bytes = float(byte_count) - float(state.byte_ewma)
                delta_elapsed = elapsed - float(state.elapsed_ewma_ms)
                state.byte_ewma = float(state.byte_ewma) + alpha * delta_bytes
                state.byte_var_ewma = (1.0 - alpha) * (state.byte_var_ewma + alpha * delta_bytes * delta_bytes)
                state.byte_elapsed_cov_ewma = (1.0 - alpha) * (state.byte_elapsed_cov_ewma + alpha * delta_bytes * delta_elapsed)
            state.byte_observations += 1
        state.observations += 1
        state.elapsed_ewma_ms = _ewma(state.elapsed_ewma_ms, elapsed, self.tuning.ewma_alpha)
        state.per_item_ewma_ms = _ewma(state.per_item_ewma_ms, elapsed / count, self.tuning.ewma_alpha)
        if byte_count > 0:
            state.per_byte_ewma_ms = _ewma(state.per_byte_ewma_ms, elapsed / byte_count, self.tuning.ewma_alpha)

    def overhead_and_marginal_ms(self, channel: str) -> tuple[float, float] | None:
        """Split a channel's cost into per-call overhead and per-item marginal.

        Per-item EWMAs misattribute fixed per-call overhead to the items: a
        drain with 15 ms of fixed work and 1 ms per item measures 8.5 ms/item
        at batch size 2, which shrinks the next batch and locks the channel
        into tiny, overhead-dominated batches. The regression over the
        exponentially-weighted (count, elapsed) moments recovers the marginal
        rate instead. Returns ``(overhead_ms, marginal_ms_per_item)``, or
        ``None`` until the batch sizes have varied enough to separate the two.
        """
        state = self._channels.get(str(channel))
        if state is None or state.count_ewma is None or state.elapsed_ewma_ms is None:
            return None
        if state.observations < 4 or state.count_var_ewma <= 0.05:
            return None
        marginal = max(0.01, float(state.count_elapsed_cov_ewma) / float(state.count_var_ewma))
        overhead = max(0.0, float(state.elapsed_ewma_ms) - marginal * float(state.count_ewma))
        return overhead, marginal

    def overhead_and_marginal_per_byte_ms(self, channel: str) -> tuple[float, float] | None:
        """Byte-denominated counterpart of :meth:`overhead_and_marginal_ms`.

        Per-byte EWMAs suffer the same misattribution as per-item ones:
        small commits fold the fixed per-call overhead into the byte rate,
        which shrinks the byte cap and locks presentation into tiny,
        overhead-dominated uploads. Returns ``(overhead_ms, marginal_ms_per
        _byte)``, or ``None`` until byte counts have varied enough.
        """
        state = self._channels.get(str(channel))
        if state is None or state.byte_ewma is None or state.elapsed_ewma_ms is None:
            return None
        if state.byte_observations < 4 or state.byte_var_ewma <= max(1.0, 0.0001 * float(state.byte_ewma) ** 2):
            return None
        marginal = max(1e-12, float(state.byte_elapsed_cov_ewma) / float(state.byte_var_ewma))
        overhead = max(0.0, float(state.elapsed_ewma_ms) - marginal * float(state.byte_ewma))
        return overhead, marginal

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
            return LatencyFeedbackChannelSnapshot(name, None, None, None, 0.0, 0, 0)
        model = self.overhead_and_marginal_ms(name)
        return LatencyFeedbackChannelSnapshot(
            channel=name,
            elapsed_ewma_ms=state.elapsed_ewma_ms,
            per_item_ewma_ms=state.per_item_ewma_ms,
            per_byte_ewma_ms=state.per_byte_ewma_ms,
            last_elapsed_ms=float(state.last_elapsed_ms),
            last_count=int(state.last_count),
            last_byte_count=int(state.last_byte_count),
            overhead_ewma_ms=None if model is None else model[0],
            marginal_per_item_ms=None if model is None else model[1],
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
