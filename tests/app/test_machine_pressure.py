"""Guards for the gate that refuses to start on a machine at its limit.

Ring 0 — no Qt, no rendering, no loaded machine required. ``tests/machine_pressure.py``
is split so that everything with a judgement in it is pure: the ladders are
driven here with synthetic :class:`~tests.machine_pressure.Reading` values, which
is the only way to guard the refusing band at all. The alternative — arranging
for the real thing — is the incident this file exists because of.

The wording is guarded as heavily as the arithmetic, for the same reason as in
``test_run_shape_gates.py``: a refusal is read exactly once, by somebody who
believes the harness is broken and is deciding whether to work around it. The
two failures that matter are naming an escape hatch that will not work, and
naming a width that will be refused again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import machine_pressure

CEILING = 8


def _reading(**overrides):
    """A quiet 16-thread machine with 62 GiB of RAM and an idle zram device."""

    base = {
        "threads": 16,
        "load": 0.5,
        "load_is_instantaneous": False,
        "cpu_stall": None,
        "memory_used": 0.20,
        "memory_total_gib": 62.5,
        "memory_available_gib": 50.0,
        "swap_used": 0.0,
        "swap_total_gib": 62.5,
        "swap_is_zram": True,
    }
    base.update(overrides)
    return machine_pressure.Reading(**base)


def _busy(ratio, stall=0.30, **overrides):
    return _reading(load=ratio * 16, cpu_stall=stall, **overrides)


# --------------------------------------------------------------------------- #
# The three ladders
# --------------------------------------------------------------------------- #


def test_a_quiet_machine_is_allowed_the_whole_ceiling():
    allowance = machine_pressure.allow(_reading(), CEILING)

    assert allowance.workers == CEILING
    assert allowance.reasons == ()
    assert not allowance.refuses


def test_the_cpu_ladder_halves_at_full_then_singles_then_refuses():
    """1.0x / 1.5x / 2.0x oversubscription, against the logical thread count.

    The bands are edges, not ranges: 1.0 exactly must already halve, or a
    machine sitting precisely full is treated as having room.
    """

    assert machine_pressure.allow(_busy(0.99), CEILING).workers == CEILING
    assert machine_pressure.allow(_busy(1.00), CEILING).workers == 4
    assert machine_pressure.allow(_busy(1.49), CEILING).workers == 4
    assert machine_pressure.allow(_busy(1.50), CEILING).workers == 1
    assert machine_pressure.allow(_busy(1.99), CEILING).workers == 1
    assert machine_pressure.allow(_busy(2.00), CEILING).refuses


def test_a_high_load_with_nobody_waiting_is_not_treated_as_contention():
    """The half of the CPU rule that keeps it from firing on the wrong machine.

    Linux counts uninterruptible sleep in the load average, so a machine blocked
    on slow I/O reads as heavily loaded while its cores sit idle — precisely the
    shape where more workers cost nothing. Requiring PSI to agree is what makes
    the count mean "work that wants a core": a task blocked on I/O stalls on
    ``/proc/pressure/io``, never on ``/proc/pressure/cpu``.
    """

    assert machine_pressure.allow(_busy(3.0, stall=0.01), CEILING).workers == CEILING
    assert machine_pressure.allow(_busy(3.0, stall=0.049), CEILING).workers == CEILING
    assert machine_pressure.allow(_busy(3.0, stall=0.05), CEILING).refuses


def test_an_unreadable_cpu_pressure_abstains_rather_than_assuming_the_worst():
    """PSI is absent on a kernel built without ``CONFIG_PSI``, or with ``psi=0``.

    That must abstain: a harness refusing to run because it could not find
    ``/proc/pressure/cpu`` would be a worse failure than the one it prevents,
    and the memory ladder still guards the state that actually froze the box.
    """

    assert machine_pressure.allow(_reading(load=48.0, cpu_stall=None), CEILING).workers == CEILING


def test_the_memory_ladder_keeps_the_last_tenth_of_ram_out_of_reach():
    assert machine_pressure.allow(_reading(memory_used=0.74), CEILING).workers == CEILING
    assert machine_pressure.allow(_reading(memory_used=0.75), CEILING).workers == 4
    assert machine_pressure.allow(_reading(memory_used=0.85), CEILING).workers == 1
    assert machine_pressure.allow(_reading(memory_used=0.90), CEILING).refuses


def test_the_swap_ladder_stops_at_half_a_zram_device():
    """The stated policy, and the most aggressive of the three deliberately.

    On this machine swap is zram: compressed pages held *in RAM*. Swapping to it
    does not buy capacity the way a disk does, so a filling swap device is not
    headroom being drawn on — it is headroom already gone, plus a compression
    tax on every touch.
    """

    assert machine_pressure.allow(_reading(swap_used=0.24), CEILING).workers == CEILING
    assert machine_pressure.allow(_reading(swap_used=0.25), CEILING).workers == 4
    assert machine_pressure.allow(_reading(swap_used=0.40), CEILING).workers == 1
    assert machine_pressure.allow(_reading(swap_used=0.50), CEILING).refuses


def test_a_machine_that_reads_as_nothing_is_never_refused():
    """Every signal degrades to ``None``, and ``None`` is never pressure."""

    blank = _reading(load=None, cpu_stall=None, memory_used=None, swap_used=None)

    assert machine_pressure.allow(blank, CEILING).workers == CEILING


# --------------------------------------------------------------------------- #
# Combining them
# --------------------------------------------------------------------------- #


def test_the_most_restrictive_signal_wins_and_every_binding_one_is_named():
    """Two ladders arriving at the same floor is a different machine from one.

    The reader is about to decide whether to wait or to override, and "load and
    memory" is a machine that will still be full in a minute where "load" alone
    may not be.
    """

    allowance = machine_pressure.allow(_busy(1.6, memory_used=0.86, swap_used=0.10), CEILING)

    assert allowance.workers == 1
    assert set(allowance.kinds) == {"cpu", "memory"}
    assert len(allowance.reasons) == 2

    one = machine_pressure.allow(_busy(1.6, memory_used=0.80), CEILING)
    assert one.workers == 1
    assert one.kinds == ("cpu",), "the halved memory signal is not the binding one"


def test_the_ladders_are_declared_in_a_consistent_order():
    """A guard on the constants themselves, not on a code path.

    Each ladder is three numbers that have to stay ordered for ``_band`` to mean
    anything; a transposed pair would silently make a band unreachable rather
    than fail anywhere.
    """

    for halve, single, refuse in (
        (machine_pressure.LOAD_HALVE, machine_pressure.LOAD_SINGLE, machine_pressure.LOAD_REFUSE),
        (
            machine_pressure.MEMORY_HALVE,
            machine_pressure.MEMORY_SINGLE,
            machine_pressure.MEMORY_REFUSE,
        ),
        (machine_pressure.SWAP_HALVE, machine_pressure.SWAP_SINGLE, machine_pressure.SWAP_REFUSE),
    ):
        assert halve < single < refuse


# --------------------------------------------------------------------------- #
# The refusal that has no override
# --------------------------------------------------------------------------- #


def test_no_override_can_reach_the_memory_or_swap_refusal():
    """The rule Thomas asked for, guarded where it is enforced.

    The permit path re-asks with :data:`NON_NEGOTIABLE`, so this is not a
    convention a caller could forget: excluding memory would mean passing a
    tuple that does not exist.
    """

    assert "cpu" not in machine_pressure.NON_NEGOTIABLE
    assert set(machine_pressure.NON_NEGOTIABLE) == {"memory", "swap"}

    permitted = machine_pressure.allow(
        _busy(3.0, memory_used=0.95, swap_used=0.60),
        CEILING,
        only=machine_pressure.NON_NEGOTIABLE,
    )
    assert permitted.refuses, "an override must not be able to buy memory that is not there"
    assert set(permitted.kinds) == {"memory", "swap"}
    assert not permitted.negotiable


def test_an_override_still_excuses_contention_for_the_cores():
    """The half that must keep working: the load-robustness experiment saturates
    the machine on purpose and would otherwise be unrunnable.
    """

    permitted = machine_pressure.allow(_busy(3.0), CEILING, only=machine_pressure.NON_NEGOTIABLE)

    assert permitted.workers == CEILING
    assert permitted.reasons == ()


def test_a_cpu_only_refusal_is_the_one_that_may_be_overridden():
    allowance = machine_pressure.allow(_busy(2.5), CEILING)

    assert allowance.refuses
    assert allowance.negotiable


# --------------------------------------------------------------------------- #
# What the harness does with it
# --------------------------------------------------------------------------- #


def test_a_serial_run_is_not_a_way_around_the_refusing_band():
    """`-n 0` still imports Qt and still allocates.

    ``numprocesses`` is 0 for an in-process run, and counting that as "zero
    processes" would let the reflex reply to the refusal -- "fine, I'll run it
    serially" -- walk straight past the band the freeze happened in.
    """

    from tests.conftest import _overloaded_run_refusal

    refusing = machine_pressure.Allowance(0, ("memory: full",), ("memory",))
    assert _overloaded_run_refusal(refusing, 0)
    assert _overloaded_run_refusal(refusing, 1)

    room_for_four = machine_pressure.Allowance(4, ("cpu: busy",), ("cpu",))
    assert _overloaded_run_refusal(room_for_four, 8)
    assert not _overloaded_run_refusal(room_for_four, 4)
    assert not _overloaded_run_refusal(room_for_four, 0), "an in-process run fits under four"


def test_the_refusal_names_the_width_to_retype(monkeypatch):
    """The number in the message must be one that will actually be allowed.

    It is half the *ceiling*, never half of what was asked for: otherwise `-n 8`
    and `-n 6` are told to retry at different widths on an identical machine,
    and the second reader cannot tell why they were given a different answer.
    """

    message = _refusal(monkeypatch, _busy(1.2), numprocesses=8)

    assert "pytest -n 4 tests/ui" in message
    assert "bound by CPU" in message
    # Every signal is shown, not only the offending one: the reader's next
    # question is "how bad, and is it clearing".
    assert "memory" in message
    assert "zram" in message


def test_the_memory_refusal_does_not_name_a_flag_that_will_not_work(monkeypatch):
    """Naming the override here and letting it fail would be worse than silence.

    The reader would type the long flag, be refused again, and conclude the
    harness is broken -- which is the point at which people start deleting
    gates. So the message says there is nothing to type, and says why.
    """

    message = _refusal(monkeypatch, _reading(memory_used=0.93), numprocesses=8)

    assert "no override for this one" in message
    assert "93% of RAM spent" in message
    # It may still be *mentioned* -- to say what it does not cover -- but never
    # as the reader's next action.
    assert "If you are certain" not in message


def test_the_refusal_points_at_the_other_suites_rather_than_at_the_tests(monkeypatch):
    """The cause is almost never this run. Naming where to look is the fix.

    The incident was several agents each sizing a pool from ``os.cpu_count()``
    as though it were alone on the machine.
    """

    message = _refusal(monkeypatch, _reading(swap_used=0.55), numprocesses=8)

    assert "pgrep" in message
    assert "worktree" in message.lower()


def test_the_suggested_command_is_the_one_that_was_typed(monkeypatch):
    """A refusal that widens the run it just declined would be worse than none.

    ``config.args`` holds only the positional paths, so answering
    ``pytest tests/ui -k montage`` with "run `pytest -n 4 tests/ui`" hands back
    a *wider* command — to somebody who is copying rather than reading, which
    is the whole reason to name one.
    """

    from tests import conftest

    config = SimpleNamespace(
        invocation_params=SimpleNamespace(args=("tests/ui", "-k", "montage or crop", "-q"))
    )

    assert conftest._retry_command(config, 4) == "pytest -n 4 tests/ui -k 'montage or crop' -q"


def test_the_suggested_command_carries_no_second_worker_count(monkeypatch):
    """Argparse would take the last spelling; the reader should not need to know.

    Every ``-n`` spelling has to go, including the joined and ``=`` forms, or a
    refusal of `-n 16` answers with a line containing both `-n 4` and `-n 16`.
    """

    from tests import conftest

    for typed in (("-n", "16"), ("-n16",), ("--numprocesses=16",), ("--numprocesses", "16")):
        config = SimpleNamespace(invocation_params=SimpleNamespace(args=(*typed, "tests/ui")))
        assert conftest._retry_command(config, 4) == "pytest -n 4 tests/ui"


def _refusal(monkeypatch, reading, *, numprocesses):
    """The text ``_refuse_overloaded_run`` raises for ``reading``, or fail.

    Pinned against a fixed ceiling rather than the host's: these assertions are
    about the sentence, and a 2-core runner would otherwise read a different one.
    """

    from tests import conftest

    monkeypatch.setattr(conftest, "_XDIST_WORKER", None)
    monkeypatch.setattr(conftest, "worker_ceiling", lambda _cpus: CEILING)
    monkeypatch.setattr(machine_pressure, "read_machine", lambda: reading)
    monkeypatch.delenv("CI", raising=False)

    config = SimpleNamespace(
        option=SimpleNamespace(numprocesses=numprocesses),
        args=["tests/ui"],
        invocation_params=SimpleNamespace(args=("tests/ui",)),
    )
    with pytest.raises(pytest.UsageError) as refused:
        conftest._refuse_overloaded_run(config)
    return str(refused.value)


def test_an_inspection_run_is_answerable_on_a_machine_in_any_state(monkeypatch):
    """`pytest --help` is where a refused reader goes to find the flag.

    A gate able to refuse the explanation of itself is a trap, not a guard --
    and an inspection run executes nothing, so there is nothing to make room for.
    """

    from tests import conftest

    monkeypatch.setattr(conftest, "_XDIST_WORKER", None)
    monkeypatch.setattr(
        machine_pressure,
        "read_machine",
        lambda: pytest.fail("an inspection run must not even look at the machine"),
    )
    for option in ("help", "version", "collectonly"):
        config = SimpleNamespace(option=SimpleNamespace(**{option: True}), args=[])
        conftest._refuse_overloaded_run(config)


def test_the_ci_permit_and_the_override_reach_the_cores_only(monkeypatch):
    from tests import conftest

    forced = SimpleNamespace(
        option=SimpleNamespace(**{conftest.LOAD_FLAG.lstrip("-").replace("-", "_"): True}),
        args=[],
    )
    monkeypatch.delenv("CI", raising=False)
    assert conftest._load_gate_permitted(forced) == "explicitly forced"

    plain = SimpleNamespace(option=SimpleNamespace(), args=[])
    assert conftest._load_gate_permitted(plain) == ""
    monkeypatch.setenv("CI", "1")
    assert conftest._load_gate_permitted(plain) == "CI, which owns its runner"


# --------------------------------------------------------------------------- #
# What it costs, and why the fast signal is safe
# --------------------------------------------------------------------------- #


def test_nothing_in_the_reading_sleeps():
    """A bare `pytest` on an idle machine is the overwhelmingly common case.

    The first version of this bought two timed measurements (~200 ms) once a
    cheap look suggested they might matter, and carried a whole fast-path/
    slow-path structure to keep the idle case free. PSI made both unnecessary:
    the kernel is already keeping the ten-second window, so four ``/proc`` reads
    answer everything. This repository deletes pytest plugins to save 58 ms; a
    gate that spent 200 ms on every no-op run would be removed, correctly.

    Asked of the syntax rather than of the clock: a timing assertion here would
    be exactly the flaky wall-clock test the sibling ``tools/timing_guard.py``
    exists to refuse. Asked of the *syntax* rather than of the text, too — the
    first version of this grepped for "sleep" and failed on the docstring
    phrase "uninterruptible sleep", which is prose about the load average.
    """

    import ast
    from pathlib import Path

    tree = ast.parse(Path(machine_pressure.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "sleep" not in called
    assert "time" not in imported, "nothing here needs a clock at all"


def test_the_fast_signal_is_what_sees_a_second_suite_starting(monkeypatch):
    """The one-minute average is far too slow to be the magnitude here.

    Measured on this 16-thread machine: 32 spinning processes -- a true 2.0x --
    were visible as 32 runnable within four seconds, while the one-minute
    average had reached 0.96x, not 2.0x, after twenty-four. A ladder reading
    the average would essentially never reach its own refusing band, and the
    incident *was* several suites starting within seconds of each other.
    """

    monkeypatch.setattr(machine_pressure, "_loadavg", lambda: (0.96 * 16, 32))
    monkeypatch.setattr(machine_pressure, "_cpu_stall", lambda: 0.59)

    reading = machine_pressure.read_machine(threads=16)

    assert reading.load == 32
    assert reading.load_is_instantaneous
    assert machine_pressure.allow(reading, CEILING).refuses


def test_psi_is_what_keeps_a_spike_from_aborting_a_run(monkeypatch):
    """The instantaneous count is spiky, and PSI is the de-spiking mechanism.

    A momentary burst of runnable tasks cannot move a ten-second average, so
    requiring PSI to agree removes the spike without this file sampling
    anything itself. That is why the confirmation replaced a median of five
    reads taken 20 ms apart, and why nothing here sleeps any more.
    """

    monkeypatch.setattr(machine_pressure, "_loadavg", lambda: (0.2, 40))
    monkeypatch.setattr(machine_pressure, "_cpu_stall", lambda: 0.004)

    spike = machine_pressure.read_machine(threads=16)

    assert spike.load == 40, "the spike is read"
    assert machine_pressure.allow(spike, CEILING).workers == CEILING, "and not acted on"


def test_the_engagement_threshold_sits_between_idle_and_saturated():
    """Measured, not chosen -- so the separation is what has to hold.

    On this 16-thread machine, PSI cpu ``some avg10`` read 0.08-1.1% idle,
    1.9% four seconds into a true 1.0x load rising to 8-12% sustained, and
    14-59% at 2.0x. The threshold has to sit above the idle ceiling and below
    what real saturation reaches within seconds, or it is either always on or
    never on.
    """

    idle_ceiling = 0.011
    saturated_within_seconds = 0.019

    assert idle_ceiling < machine_pressure.CPU_STALL_ENGAGES < saturated_within_seconds * 3
