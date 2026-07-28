"""Say out loud when the suite ran against a busy machine.

This repository's testing rules lean on timing in two places that a competing
workload quietly invalidates:

* the parallel-execution rule (``docs/testing/strategy.md``) — timing-fragile
  tests flake under CPU saturation, and the standing advice on a surprise red
  is "rerun and compare against a baseline of the same command";
* every performance bar and benchmark, where a wall-clock number measured while
  something else owned the cores is not evidence of anything.

Both are usually discovered *after* someone has already spent an hour bisecting
a phantom. So the load is reported before the first test and again after the
last one, in the header and the terminal summary — the two places a human or an
agent reads without being told to look.

The end-of-run figure subtracts this run's own parallelism (its children's CPU
time over wall time), because a 16-way suite drives the load average up by
itself and would otherwise always look like contention.
"""

from __future__ import annotations

import os
import resource
import time
from dataclasses import dataclass

#: A machine is "busy" once other work occupies this share of the cores. Below
#: it, xdist's own oversubscription is the dominant effect and a warning would
#: be noise.
_BUSY_SHARE = 0.25
_BUSY_FLOOR = 2.0


def _load_average() -> float | None:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):  # not available on every platform
        return None


def _cpu_count() -> int:
    return os.cpu_count() or 1


def _busy_threshold() -> float:
    return max(_BUSY_FLOOR, _BUSY_SHARE * _cpu_count())


@dataclass(frozen=True)
class LoadWindow:
    """Load average and this process tree's CPU use, sampled once."""

    load: float | None
    wall: float
    cpu_seconds: float

    @classmethod
    def sample(cls) -> LoadWindow:
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = resource.getrusage(resource.RUSAGE_SELF)
        return cls(
            load=_load_average(),
            wall=time.monotonic(),
            cpu_seconds=(children.ru_utime + children.ru_stime + own.ru_utime + own.ru_stime),
        )

    def own_parallelism_since(self, start: LoadWindow) -> float:
        elapsed = max(self.wall - start.wall, 1e-6)
        return max(0.0, (self.cpu_seconds - start.cpu_seconds) / elapsed)


def opening_line(start: LoadWindow) -> str | None:
    """A warning for load that was already there before this run started."""

    if start.load is None or start.load < _busy_threshold():
        return None
    return (
        f"WARNING: system load {start.load:.1f} on {_cpu_count()} cores before this run — "
        "expect timing-sensitive tests to flake, and do not read any duration here as evidence."
    )


def closing_line(start: LoadWindow, end: LoadWindow) -> str | None:
    """A warning for load this run did not create.

    Under-reports rather than over-reports: the one-minute load average lags, so
    a burst of foreign work that ended mid-run is partly forgotten by now. A
    quiet line here is weaker evidence than a loud one.
    """

    if end.load is None:
        return None
    foreign = end.load - end.own_parallelism_since(start)
    if foreign < _busy_threshold():
        return None
    return (
        f"WARNING: ~{foreign:.1f} of the load on {_cpu_count()} cores during this run came from "
        "elsewhere. Re-run before attributing a failure or a duration to your change."
    )
