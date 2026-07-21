"""Frame-timing observation for the wgpu screen present path.

The screen path's diagnostics kept only *last* and *max* for acquire and
present, which cannot answer the question phase 1 of
``docs/redesign/wgpu-frame-pacing-2026-07-21.md`` asks: are presented frames
aligned in time with scan-out, or merely emitted at the right average rate?
A frequency-matched, phase-free pacer reports a perfect fps while every
frame lands at a different offset from vblank.

The headline metric is ``phase_lock_r``, the circular resultant length of
each present's phase within the display period:

* ``R -> 0`` — phases are uniformly distributed: no relationship to
  scan-out at all.  This is the signature of a free-running rate limiter.
* ``R -> 1`` — every frame presents at the same point in the refresh
  cycle: phase-locked.

``R`` alone cannot say *why* a pacer is unlocked, and the two reasons want
different fixes: an interval that is slightly wrong walks steadily through
the phase (dossier defect 1), while correct-interval-but-noisy wakeups
scatter without going anywhere (defect 2).  Both collapse ``R`` to ~0.
``phase_drift_ms_per_s`` tells them apart — but only when
``phase_advance_spread_ms`` is small.  Drift is a median over per-frame
phase advances, so once those advances are spread across a large fraction
of the period the median's own sampling error, rescaled to per-second,
swamps the answer.  **Read the drift only as far as the spread allows**;
the pair is the metric, not the drift alone.

Everything is computed on read from a bounded ring; the per-frame cost is
appending a handful of floats.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import pairwise

#: Frames retained per series.  At 60 Hz this is ~8.5 s of history — long
#: enough to see drift, short enough that a snapshot reflects the current
#: interaction rather than a minute-old one.
DEFAULT_CAPACITY = 512


@dataclass
class FrameTimingRecorder:
    """Bounded observation of the screen path's frame cadence.

    One recorder per canvas.  The canvas owns pacing, so it owns the
    timings too: splitting them between canvas (scheduling) and view
    (present edge) is what left the existing counters unable to relate a
    slipped wakeup to a late frame.
    """

    capacity: int = DEFAULT_CAPACITY
    _present_at: deque[float] = field(default_factory=deque, init=False)
    _intervals_ms: deque[float] = field(default_factory=deque, init=False)
    _frame_ms: deque[float] = field(default_factory=deque, init=False)
    _slip_ms: deque[float] = field(default_factory=deque, init=False)
    _acquire_ms: deque[float] = field(default_factory=deque, init=False)
    _present_ms: deque[float] = field(default_factory=deque, init=False)
    _draw_started_at: float | None = field(default=None, init=False)
    _deadline_at: float | None = field(default=None, init=False)
    _presents: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        # The field defaults build unbounded deques; rebuild them with the
        # window pinned so every series ages together.
        capacity = max(8, int(self.capacity))
        self.capacity = capacity
        self._present_at = deque(maxlen=capacity)
        self._intervals_ms = deque(maxlen=capacity)
        self._frame_ms = deque(maxlen=capacity)
        self._slip_ms = deque(maxlen=capacity)
        self._acquire_ms = deque(maxlen=capacity)
        self._present_ms = deque(maxlen=capacity)

    def _series(self) -> tuple[deque[float], ...]:
        return (
            self._present_at,
            self._intervals_ms,
            self._frame_ms,
            self._slip_ms,
            self._acquire_ms,
            self._present_ms,
        )

    # ---- recording ----------------------------------------------------------

    def note_scheduled(self, deadline_at: float) -> None:
        """The wakeup this pacer intends to happen at ``deadline_at``."""

        self._deadline_at = float(deadline_at)

    def note_draw_started(self, started_at: float) -> None:
        """A paced draw actually began.

        The gap against the intended deadline is timer slip — the amount by
        which the platform missed the pacer's request.  It is recorded even
        when the pacer asked for "now" (delay 0), because an event loop busy
        with commit work delays those too.
        """

        started_at = float(started_at)
        self._draw_started_at = started_at
        deadline = self._deadline_at
        if deadline is not None:
            self._slip_ms.append((started_at - deadline) * 1000.0)
            self._deadline_at = None

    def note_presented(self, presented_at: float, *, acquire_ms: float, present_ms: float) -> None:
        """A frame reached the real ``wgpuSurfacePresent`` edge."""

        presented_at = float(presented_at)
        previous = self._present_at[-1] if self._present_at else None
        if previous is not None:
            self._intervals_ms.append((presented_at - previous) * 1000.0)
        self._present_at.append(presented_at)
        if self._draw_started_at is not None:
            self._frame_ms.append((presented_at - self._draw_started_at) * 1000.0)
            self._draw_started_at = None
        self._acquire_ms.append(float(acquire_ms))
        self._present_ms.append(float(present_ms))
        self._presents += 1

    def reset(self) -> None:
        """Drop history — a present-mode or display change invalidates it."""

        for series in self._series():
            series.clear()
        self._draw_started_at = None
        self._deadline_at = None

    # ---- readout ------------------------------------------------------------

    def snapshot(self, *, period_ms: float) -> dict[str, object]:
        """Percentiles plus the phase statistics, against a display period.

        ``period_ms`` is the refresh period the pacer is aiming at.  Phase is
        only meaningful relative to an assumed period, so it is passed in
        rather than guessed here.
        """

        period_ms = float(period_ms)
        lock_r, drift, advance_spread = _phase_statistics(self._present_at, period_ms)
        snapshot: dict[str, object] = {
            "frames_observed": int(self._presents),
            "window_frames": len(self._present_at),
            "period_ms": period_ms,
            "phase_lock_r": lock_r,
            "phase_spread_ms": _circular_spread_ms(lock_r, period_ms),
            "phase_drift_ms_per_s": drift,
            # Read drift only as far as this lets you: see _phase_statistics.
            "phase_advance_spread_ms": advance_spread,
        }
        snapshot.update(_distribution("interval_ms", self._intervals_ms))
        snapshot.update(_distribution("frame_ms", self._frame_ms))
        snapshot.update(_distribution("schedule_slip_ms", self._slip_ms))
        snapshot.update(_distribution("acquire_ms", self._acquire_ms))
        snapshot.update(_distribution("present_ms", self._present_ms))
        return snapshot


def _distribution(name: str, values) -> dict[str, float]:
    ordered = sorted(values)
    return {
        f"{name}_p50": _percentile(ordered, 50.0),
        f"{name}_p95": _percentile(ordered, 95.0),
        f"{name}_max": float(ordered[-1]) if ordered else 0.0,
    }


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""

    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (max(0.0, min(100.0, float(q))) / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[int(position)])
    weight = position - low
    return float(ordered[low]) * (1.0 - weight) + float(ordered[high]) * weight


def _phase_statistics(present_at, period_ms: float) -> tuple[float, float, float]:
    """Circular resultant length, phase drift, and the drift's own spread.

    Returns ``(R, drift_ms_per_s, advance_spread_ms)``.  ``R`` is the
    standard circular concentration measure: the length of the mean unit
    vector of each present's angle within the refresh cycle.  Uniformly
    scattered phases cancel to ~0; a locked pacer's align to ~1.

    ``advance_spread_ms`` is the interquartile range of the per-frame phase
    advances, and it exists because drift alone can be confidently wrong.
    A pacer whose every frame advances the phase by the same 0.33 ms gives
    a drift number that means exactly what it says; a pacer scattered by
    half a period gives a *median* advance with roughly 0.3 ms of standard
    error, which the rescaling to per-second multiplies into ~18 ms/s of
    pure noise.  Both report ``R ≈ 0``, so ``R`` cannot separate them —
    only the spread of the advances can.  Read drift as meaningful when
    this is small relative to the period, and as unresolved when it is not.

    Drift is the **median** per-frame phase advance, rescaled to
    milliseconds of phase per second of wall clock.  Each frame's advance is
    its interval reduced into ``[-period/2, +period/2)``, so that a dropped
    frame — an interval of two periods, landing on the *same* phase —
    reduces to ~0 rather than to a full period of imaginary movement.

    Both halves of that sentence were arrived at by being wrong first, and
    the tests pin each:

    * Fitting a slope to an *unwrapped* phase series needs a "did it wrap?"
      test, and jitter approaching half a period makes that test guess
      wrong often enough to manufacture ~11 ms/s of drift out of a pacer
      that is not drifting at all.
    * Summing (or averaging) the reduced advances then random-walks: every
      fold injects ±period, so a scattered pacer accumulated ~21 ms/s of
      pure noise.  The median is indifferent to how often the fold fires
      and reports the *typical* frame's advance, which is the quantity the
      word "drift" actually means here.
    """

    if period_ms <= 0.0:
        return 0.0, 0.0, 0.0
    samples = list(present_at)
    if len(samples) < 2:
        return 0.0, 0.0, 0.0
    origin = samples[0]
    period_s = period_ms / 1000.0

    total_cos = 0.0
    total_sin = 0.0
    for timestamp in samples:
        phase = math.fmod(timestamp - origin, period_s) / period_s
        if phase < 0.0:
            phase += 1.0
        angle = 2.0 * math.pi * phase
        total_cos += math.cos(angle)
        total_sin += math.sin(angle)
    lock_r = math.hypot(total_cos, total_sin) / float(len(samples))

    intervals_ms = [(timestamp - previous) * 1000.0 for previous, timestamp in pairwise(samples)]
    advances = sorted(_centered_remainder(interval, period_ms) for interval in intervals_ms)
    spread_ms = _percentile(advances, 75.0) - _percentile(advances, 25.0)
    typical_interval_ms = _percentile(sorted(intervals_ms), 50.0)
    if typical_interval_ms <= 0.0:
        return lock_r, 0.0, spread_ms
    drift = _percentile(advances, 50.0) * (1000.0 / typical_interval_ms)
    return lock_r, drift, spread_ms


def _centered_remainder(value: float, period: float) -> float:
    """``value`` reduced into ``[-period/2, +period/2)``."""

    remainder = math.fmod(value, period)
    if remainder >= period / 2.0:
        remainder -= period
    elif remainder < -period / 2.0:
        remainder += period
    return remainder


def _circular_spread_ms(lock_r: float, period_ms: float) -> float:
    """Circular standard deviation in milliseconds.

    Reported alongside ``R`` because milliseconds are the unit the rest of
    the timing budget is argued in.  Saturates at a quarter period, which is
    where "unlocked" stops being a useful gradient.
    """

    if period_ms <= 0.0:
        return 0.0
    lock_r = max(0.0, min(1.0, float(lock_r)))
    if lock_r <= 1e-6:
        return period_ms / 4.0
    spread = math.sqrt(max(0.0, -2.0 * math.log(lock_r))) / (2.0 * math.pi) * period_ms
    return min(spread, period_ms / 4.0)
