"""Input-momentum policy for speculative slice prefetch.

The idle prefetcher warms neighboring slices so the next user step lands on
a cache hit. How far ahead it is worth warming depends on what the user is
doing: a single step warrants one neighbor on each side, while a sustained
scrub in one direction makes slices ahead of the motion far more likely to
be requested than slices behind it.

This module is Qt-free. It only turns an observed sequence of slice indices
into a bounded, ordered plan; all admission gates (visible-work busy, memory
budgets, cost estimates, dedupe, in-flight caps, work-graph admission) stay
with the scheduler and are applied per candidate after planning.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlicePrefetchPlan:
    """Bounded, priority-ordered speculation plan for one slice axis."""

    direction: int
    depth: int
    deltas: tuple[int, ...]


def prefetch_deltas(direction: int, depth: int) -> tuple[int, ...]:
    """Candidate slice offsets in priority order.

    Directional plans put every ahead-of-motion offset first and keep a
    single behind-the-motion guard so one reversal step still hits cache.
    Non-directional plans alternate around the current slice.
    """

    depth = max(1, int(depth))
    if direction > 0:
        return tuple(range(1, depth + 1)) + (-1,)
    if direction < 0:
        return tuple(range(-1, -depth - 1, -1)) + (1,)
    return tuple(delta for radius in range(1, depth + 1) for delta in (-radius, radius))


@dataclass
class SliceScrubMomentum:
    """Tracks direction and persistence of slice scrubbing.

    ``observe`` each committed slice index; ``plan`` returns how deep and in
    which direction speculation is currently justified. A pause longer than
    ``quiet_window_s`` or a direction change resets the streak, so depth
    grows only under sustained same-direction motion and decays immediately
    when the user stops or turns around.
    """

    quiet_window_s: float = 0.5
    directional_depth: int = 2
    max_depth: int = 4
    direction: int = 0
    streak: int = 0
    _last_index: int | None = None
    _last_monotonic: float | None = None

    def observe(self, index: int, *, now: float) -> None:
        last_index = self._last_index
        last_time = self._last_monotonic
        self._last_index = int(index)
        self._last_monotonic = float(now)
        if last_index is None:
            self.direction = 0
            self.streak = 0
            return
        step = int(index) - int(last_index)
        if step == 0:
            return
        stale = last_time is None or (float(now) - float(last_time)) > float(self.quiet_window_s)
        direction = 1 if step > 0 else -1
        if stale or direction != self.direction:
            self.direction = direction
            self.streak = 1
        else:
            self.streak += 1

    def plan(self, *, size: int | None = None) -> SlicePrefetchPlan:
        if self.direction == 0 or self.streak <= 0:
            direction = 0
            depth = 1
        else:
            direction = int(self.direction)
            depth = min(int(self.max_depth), int(self.directional_depth) + max(0, self.streak - 1))
        if size is not None:
            depth = max(1, min(depth, max(1, int(size) - 1)))
        return SlicePrefetchPlan(direction, depth, prefetch_deltas(direction, depth))
