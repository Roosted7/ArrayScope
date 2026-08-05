"""Refuse to start a wide run on a machine that is already at its limit.

``load_report.py`` next door *warns* about a busy machine, because its subject
is evidence: a duration measured while something else owned the cores is not
evidence of anything. This file's subject is survival. On 2026-08-05 several
agents ran their suites concurrently in sibling worktrees, each sizing its pool
from ``os.cpu_count()`` as though it were alone, and the machine froze hard
enough to need a power cycle. No warning printed into a scrollback that nobody
was watching would have changed that outcome; only declining to start would.

So this is a gate, not a report. Three signals, each with the same shape — a
ladder from "fine" through two degraded steps to "not at all":

* **CPU.** Runnable tasks against the logical thread count, but *only* once
  tasks are actually waiting for a core (:data:`CPU_STALL_ENGAGES`, from the
  kernel's PSI). Both halves are load-bearing, and both were chosen against
  measurement rather than from first principles:

  - **The magnitude comes from field 4 of ``/proc/loadavg``, not from the
    average.** The one-minute average is uselessly slow here. Sixteen spinning
    processes on this sixteen-thread machine — a true 1.0× — moved it to 0.52×
    after twenty-eight seconds; thirty-two spinners reached 0.96×, not 2.0×,
    after twenty-four. A ladder reading that number would essentially never
    reach its own refusing band, while the instantaneous runnable count showed
    32 within four seconds. This matters because the incident *was* several
    suites starting within seconds of each other, which is exactly the window
    the average cannot see.
  - **The confirmation is PSI, which is also what makes the fast number safe.**
    An instantaneous count is spiky, and a single unlucky read must not abort a
    run. ``/proc/pressure/cpu``'s ``some avg10`` is the share of the last ten
    seconds in which at least one runnable task was stalled waiting for CPU — a
    momentary burst cannot move a ten-second average, so requiring it removes
    the spike without sampling anything. It also answers the question the
    original "cores ≥ 90% busy" check was a proxy for, and answers it better:
    Linux counts uninterruptible sleep in the load average, so a machine
    blocked on a slow disk reads as loaded while its cores idle, and a task
    blocked on I/O is stalled on ``/proc/pressure/io``, never here.

* **Memory.** How much of RAM is spent, from ``MemAvailable``. ``free(1)`` is a
  formatter over ``/proc/meminfo``; this reads the same source directly rather
  than shelling out and parsing a localised table.

* **Swap.** Deliberately the most aggressive ladder of the three, because on
  this machine swap is ``zram``: a compressed block device *in RAM*. Swapping to
  it does not buy capacity the way a disk does — the pages still occupy memory,
  just smaller — so swap use is not headroom being drawn on, it is headroom that
  is already gone, plus a compression tax on every touch. Half a full zram
  device is where the thrash begins.

The most restrictive signal wins, and every signal that bound is named in the
refusal: "which of the three stopped me" is the first thing the reader needs.

**The memory and swap refusals cannot be overridden.** Not by a flag, not by
``CI``, not by an environment variable, and there is deliberately nowhere to add
one — see :data:`NON_NEGOTIABLE`. The CPU ladder can be, because "this will be
slow and I accept that" is a coherent position and one experiment here depends
on it. "This will not fit in memory and I accept that" is not a position; it is
the sequence of events that ended in a power cycle.

**Unknown is never pressure.** Every reading degrades to ``None`` and a ``None``
signal abstains. This is Linux-and-``/proc`` by construction, and a harness that
refused to run because it could not find ``/proc/meminfo`` would be a worse
failure than the one it prevents.

Nothing here judges *this* run's own load. It is called once, before the first
worker starts, when the only load on the machine is by definition somebody
else's.

**Nothing here sleeps**, which is why four ``/proc`` reads answer all of it in
~0.1 ms and there is no fast path to maintain. Every window this needs — ten
seconds of CPU stall, one minute of load — is one the kernel is already
keeping.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# The ladders
# --------------------------------------------------------------------------- #

#: Runnable-tasks-to-thread ratios. At 1.0 the machine is nominally full: every
#: thread has a runnable task and this run's workers would be taking turns
#: rather than running. Halving there, one worker at 1.5, and nothing at 2.0 —
#: twice oversubscribed is the regime the freeze happened in.
LOAD_HALVE = 1.0
LOAD_SINGLE = 1.5
LOAD_REFUSE = 2.0

#: ...but only when tasks are genuinely waiting for a core. Measured rather
#: than chosen, on this 16-thread machine (spinning processes, PSI ``some
#: avg10``, sampled every 4 s):
#:
#: ===================== ==================
#: state                 cpu-some avg10
#: ===================== ==================
#: idle                  0.08 – 1.1%
#: 16 spinners (1.0×)    1.9% at t+4s, 8 – 12% sustained
#: 32 spinners (2.0×)    14% at t+4s, rising to 59%
#: ===================== ==================
#:
#: 5% sits five times above the idle ceiling and is crossed within a few
#: seconds of real saturation, which is the separation the number is for.
CPU_STALL_ENGAGES = 0.05

#: Share of RAM spent (``1 - MemAvailable/MemTotal``). Below 75% there is room
#: for a pool of Qt workers; the last 10% is reserve the kernel needs to not
#: start reclaiming under everything.
MEMORY_HALVE = 0.75
MEMORY_SINGLE = 0.85
MEMORY_REFUSE = 0.90

#: Share of swap spent. The 25% and 50% edges are the stated policy; the 40%
#: step exists so this ladder degrades in the same three stages as the other
#: two rather than falling off a cliff at 50%.
SWAP_HALVE = 0.25
SWAP_SINGLE = 0.40
SWAP_REFUSE = 0.50

_KIB_PER_GIB = 1024 * 1024


@dataclass(frozen=True)
class Reading:
    """What the machine looked like just before this run asked for workers."""

    threads: int
    #: The larger of the one-minute average and the instantaneous runnable
    #: count. Almost always the latter, and measurement is why: 32 spinners on
    #: this 16-thread machine were visible as 32 runnable within four seconds,
    #: while the one-minute average had reached 0.96 — not 2.0 — after
    #: twenty-four. The average is kept only because it is the one of the two
    #: that cannot read low by chance. ``None`` if neither could be read.
    load: float | None
    #: True when :attr:`load` came from the instantaneous count — worth saying,
    #: since a reader who checks ``uptime`` afterwards will see a much smaller
    #: number and conclude the gate invented this one.
    load_is_instantaneous: bool
    #: PSI ``some avg10`` for CPU as ``0.0``–``1.0``: the share of the last ten
    #: seconds in which some task was waiting for a core. ``None`` on a kernel
    #: without PSI, and then the CPU ladder abstains.
    cpu_stall: float | None
    memory_used: float | None
    memory_total_gib: float | None
    memory_available_gib: float | None
    swap_used: float | None
    swap_total_gib: float | None
    #: Swap is a compressed in-RAM device, so its use is spent RAM.
    swap_is_zram: bool

    @property
    def load_ratio(self) -> float | None:
        if self.load is None or self.threads <= 0:
            return None
        return self.load / self.threads


#: The signals no override, permit or environment may switch off.
#:
#: The CPU ladder is a performance argument: too many workers make everything
#: slower, and somebody who has decided to accept that — the load-robustness
#: experiment does exactly this — is entitled to. Memory is not that kind of
#: argument. The machine that prompted this file did not get slow and then
#: recover; it went unresponsive and needed a power cycle, because the last of
#: RAM and half a zram device is a state a desktop does not come back from on
#: its own. There is no version of "I know what I am doing" that makes an
#: eight-Qt-worker pool fit in memory that is not there, so the override does
#: not reach these two and there is deliberately nowhere to make it.
#:
#: Swap is here rather than beside CPU because on this machine swap *is*
#: memory: zram holds compressed pages in RAM, so a filling swap device is not
#: capacity being drawn on, it is capacity already gone.
NON_NEGOTIABLE = ("memory", "swap")

#: The full set, in ladder order.
SIGNALS = ("cpu", *NON_NEGOTIABLE)


@dataclass(frozen=True)
class Allowance:
    """How many workers this machine can take, and which signal said so."""

    #: ``0`` means no run at all, at any width — not "run serially".
    workers: int
    #: One line per signal that bound, in the order the ladders are declared.
    #: Empty exactly when :attr:`workers` is the unrestricted ceiling.
    reasons: tuple[str, ...]
    #: The names from :data:`SIGNALS` behind :attr:`reasons`, positionally.
    #: Callers need this to tell a refusal an override could have lifted from
    #: one it could not.
    kinds: tuple[str, ...]

    @property
    def refuses(self) -> bool:
        return self.workers == 0

    @property
    def negotiable(self) -> bool:
        """Whether every binding signal is one an override is allowed to lift."""

        return all(kind not in NON_NEGOTIABLE for kind in self.kinds)


def _band(value: float, halve: float, single: float, refuse: float, ceiling: int) -> int:
    if value >= refuse:
        return 0
    if value >= single:
        return 1
    if value >= halve:
        return max(1, ceiling // 2)
    return ceiling


def allow(reading: Reading, ceiling: int, only: tuple[str, ...] = SIGNALS) -> Allowance:
    """The worker count this machine can take, from the most restrictive ladder.

    Pure: everything it judges is in ``reading``, which is what lets the guards
    in ``tests/app/test_machine_pressure.py`` drive every band without needing a
    loaded machine to do it on.

    ``only`` narrows which ladders are consulted, and exists for exactly one
    caller: the harness re-asks with :data:`NON_NEGOTIABLE` when a run has
    earned a permit, so that a permit can excuse contention without being able
    to excuse an empty machine. It is a whitelist rather than a "skip these" so
    that a permit cannot be spelled as ``only=SIGNALS`` minus nothing and
    quietly grow to cover memory later.

    The halved step is half of the *ceiling*, not half of what the run asked
    for. The refusal names a specific number and tells the reader to type it, so
    that number has to be a property of the machine rather than of the command
    that happened to trip the gate — otherwise `-n 8` and `-n 6` are told to
    retry at different widths under identical conditions, and neither can tell
    why.
    """

    limits: list[tuple[str, int, str]] = []

    ratio = reading.load_ratio
    if (
        "cpu" in only
        and ratio is not None
        and reading.cpu_stall is not None
        and reading.cpu_stall >= CPU_STALL_ENGAGES
    ):
        workers = _band(ratio, LOAD_HALVE, LOAD_SINGLE, LOAD_REFUSE, ceiling)
        if workers < ceiling:
            source = "runnable now" if reading.load_is_instantaneous else "1-minute average"
            limits.append(
                (
                    "cpu",
                    workers,
                    f"CPU ({ratio:.2f}× oversubscribed by {source}, tasks waiting "
                    f"for a core {reading.cpu_stall * 100:.0f}% of the last 10 s)",
                )
            )

    if "memory" in only and reading.memory_used is not None:
        workers = _band(reading.memory_used, MEMORY_HALVE, MEMORY_SINGLE, MEMORY_REFUSE, ceiling)
        if workers < ceiling:
            limits.append(
                (
                    "memory",
                    workers,
                    f"memory ({reading.memory_used * 100:.0f}% of RAM spent, "
                    f"{reading.memory_available_gib:.1f} GiB left)",
                )
            )

    if "swap" in only and reading.swap_used is not None:
        workers = _band(reading.swap_used, SWAP_HALVE, SWAP_SINGLE, SWAP_REFUSE, ceiling)
        if workers < ceiling:
            kind = " — zram, which is RAM" if reading.swap_is_zram else ""
            limits.append(
                (
                    "swap",
                    workers,
                    f"swap ({reading.swap_used * 100:.0f}% of "
                    f"{reading.swap_total_gib:.1f} GiB in use{kind})",
                )
            )

    if not limits:
        return Allowance(ceiling, (), ())
    workers = min(count for _, count, _ in limits)
    # Every signal at the binding level is named, not just the first: two
    # ladders arriving at "1" together is a materially different machine from
    # one, and the reader is about to decide whether to wait or to override.
    binding = [(kind, line) for kind, count, line in limits if count == workers]
    return Allowance(
        workers,
        tuple(line for _, line in binding),
        tuple(kind for kind, _ in binding),
    )


# --------------------------------------------------------------------------- #
# Reading the machine
# --------------------------------------------------------------------------- #


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def _loadavg() -> tuple[float | None, int | None]:
    """``(one-minute average, runnable tasks right now)`` from one file read.

    The runnable count includes the reader, which is not competition with
    itself; a machine with nothing else to do must read as 0, not 1.
    """

    raw = _read("/proc/loadavg")
    if not raw:
        return None, None
    fields = raw.split()
    try:
        average = float(fields[0])
    except (IndexError, ValueError):
        average = None
    runnable = None
    if len(fields) >= 4 and "/" in fields[3]:
        try:
            runnable = max(0, int(fields[3].split("/", 1)[0]) - 1)
        except ValueError:
            runnable = None
    return average, runnable


def _cpu_stall() -> float | None:
    """Share of the last 10 s in which some task waited for a core, or ``None``.

    The kernel's Pressure Stall Information, ``some avg10`` on
    ``/proc/pressure/cpu``: the fraction of the window in which at least one
    runnable task was stalled waiting for CPU. It needs no interval of our own
    because the kernel has already kept one, which is why this whole file
    contains no ``sleep``.

    Absent on a kernel built without ``CONFIG_PSI``, or with ``psi=0``. That
    reads as ``None`` and the CPU ladder abstains — unknown is never pressure.
    """

    raw = _read("/proc/pressure/cpu")
    if not raw:
        return None
    for line in raw.splitlines():
        if not line.startswith("some "):
            continue
        for field in line.split()[1:]:
            name, _, value = field.partition("=")
            if name == "avg10":
                try:
                    return min(1.0, max(0.0, float(value) / 100.0))
                except ValueError:
                    return None
    return None


def _meminfo() -> dict[str, float]:
    """``/proc/meminfo`` in KiB. Missing keys are simply absent."""

    raw = _read("/proc/meminfo")
    if not raw:
        return {}
    values: dict[str, float] = {}
    for line in raw.splitlines():
        name, _, rest = line.partition(":")
        try:
            values[name] = float(rest.split()[0])
        except (IndexError, ValueError):
            continue
    return values


def _swap_is_zram() -> bool:
    raw = _read("/proc/swaps") or ""
    return any("zram" in line for line in raw.splitlines()[1:])


def read_machine(threads: int | None = None) -> Reading:
    """Sample CPU, memory and swap once. Four file reads, ~0.1 ms, no sleeping.

    There is no fast path and no slow path, because after PSI replaced the
    sampled busy share there is nothing left that costs anything. That is worth
    stating: the first version of this function bought two timed measurements
    (~200 ms) once a cheap look suggested they might matter, and the whole
    structure existed to keep a bare ``pytest`` on an idle machine free. The
    kernel already keeps the average, so the structure was deleted with the
    sleeps.
    """

    threads = threads or os.cpu_count() or 1
    average, runnable = _loadavg()

    memory = _meminfo()
    total = memory.get("MemTotal")
    available = memory.get("MemAvailable")
    memory_used = None
    if total and available is not None:
        memory_used = max(0.0, min(1.0, 1.0 - available / total))

    swap_total = memory.get("SwapTotal")
    swap_free = memory.get("SwapFree")
    swap_used = None
    if swap_total and swap_free is not None:
        swap_used = max(0.0, min(1.0, 1.0 - swap_free / swap_total))

    known = [float(value) for value in (average, runnable) if value is not None]
    return Reading(
        threads=threads,
        load=max(known) if known else None,
        load_is_instantaneous=(runnable is not None and (average is None or runnable > average)),
        cpu_stall=_cpu_stall(),
        memory_used=memory_used,
        memory_total_gib=total / _KIB_PER_GIB if total else None,
        memory_available_gib=available / _KIB_PER_GIB if available is not None else None,
        swap_used=swap_used,
        swap_total_gib=swap_total / _KIB_PER_GIB if swap_total else None,
        swap_is_zram=_swap_is_zram(),
    )


def situation(reading: Reading) -> tuple[str, ...]:
    """Every signal, bound or not, for the block above a refusal.

    All three are printed even when only one of them fired. The reader's next
    question after "why was I stopped" is "how bad is it, and is it clearing",
    and a block that showed only the offending line would have them run ``free``
    and ``uptime`` by hand to answer it.
    """

    lines = []
    ratio = reading.load_ratio
    if ratio is None:
        lines.append("load: unreadable")
    else:
        stall = (
            "CPU pressure unreadable"
            if reading.cpu_stall is None
            else f"{reading.cpu_stall * 100:.0f}% of the last 10 s waiting for a core"
        )
        lines.append(
            f"load   {reading.load:.1f} runnable on {reading.threads} threads "
            f"({ratio:.2f}×), {stall}"
        )
    if reading.memory_used is None:
        lines.append("memory: unreadable")
    else:
        lines.append(
            f"memory {reading.memory_used * 100:.0f}% spent, "
            f"{reading.memory_available_gib:.1f} GiB available of "
            f"{reading.memory_total_gib:.1f} GiB"
        )
    if reading.swap_used is None:
        lines.append("swap   none configured")
    else:
        kind = "zram — compressed, and still in RAM" if reading.swap_is_zram else "swap"
        lines.append(
            f"swap   {reading.swap_used * 100:.0f}% of {reading.swap_total_gib:.1f} GiB "
            f"in use ({kind})"
        )
    return tuple(lines)
