"""The gate for :mod:`tests.timing_guard`, and the proof its rules can fail.

Ring 0 -- no Qt, no ``arrayscope`` import, nothing built. Both gates read the
tree and nothing else, the same shape as
``test_every_declared_entry_point_is_one_testmon_can_key``: whatever a new pause
does to a later run, it does it to this test first, with the offending path and
line in the message. ~2 s on a loaded box, against 4.4 s for the
interaction-timeout guard that already lives in the suite for the same reason.

One limit, stated rather than left to be discovered. This guard reads the suite
as *data*, so testmon does not select it when one of the files it reads changes
-- the same gap ``tests/app/test_dead_code.py`` and ``test_architecture_guards``
have, and the reason ``testmon_policy.forgivable_checksums`` names those two in
its docstring. A commit that adds a pause therefore meets this gate on the next full
or ``--since`` run rather than at the moment it is written. Closing it properly
means a line in ``.githooks/pre-commit``, which is the gate that actually fires
here, at ~2 s added to every commit; that is a trade for the repository to make
and not one to slip in.

Half of this file is the falsifiability half, and it is the half that matters.
A guard nobody can see fail is the failure mode this repository keeps meeting
-- five times by ``docs/reviews/2026-07-27``, and again in the assertion this
week's work deleted, which was green with the defect forced back in. So every
rule below has a case that would pass silently without it, and every exemption
has a case showing what it lets through and a neighbouring case showing what it
still refuses.
"""

from __future__ import annotations

from tests import timing_guard

#: Fixed pauses in the tree whose duration was typed rather than read from the
#: throttle bounding it. Measured 2026-08-02, and **only ever revised
#: downwards**: a commit that raises it is precisely what this guard exists to
#: refuse. The equality is exact in both directions, so converting a site
#: without lowering the number fails too -- otherwise the ratchet goes slack and
#: the next pause slips into the gap. ``tests.timing_guard.report`` prints the
#: whole inventory on either failure. When it reaches nought, delete it and
#: assert the empty list.
#:
#: The one thing it cannot do is notice a commit that converts one site and adds
#: another in the same breath. That is why the failure message is the inventory
#: and not the number.
UNJUSTIFIED_PAUSE_BACKLOG = 55


def test_no_test_adjudicates_a_clock_reading():
    """A duration may be recorded anywhere and decide nothing, anywhere.

    There is no allowance and no backlog here, because the tree already has
    none: ``ba53fbc8`` restated the last one on counters. The rule is about the
    reading rather than what is done with it, because the last survivor was
    ``empty_ms <= working_ms`` -- threshold-free, ratio-free, and both flaky and
    vacuous at once (``docs/testing/strategy.md``).
    """

    stopwatches = [finding for finding in timing_guard.findings() if finding.kind == "stopwatch"]
    assert stopwatches == [], (
        "these tests decide something from a clock reading, and a worker sharing "
        "sixteen cores cannot measure a duration:\n" + timing_guard.report(stopwatches)
    )


def test_fixed_pauses_are_no_more_numerous_than_the_recorded_backlog():
    """A pause whose duration was typed must not be added to this tree.

    The survivors are a backlog, not an allowlist: they are counted, never
    named, so nothing here grandfathers a particular file and no entry can rot
    into permission. The fix for one of them is to read the interval from the
    throttle it is bounded by -- ``qtbot.wait(2 * PUBLISH_COALESCE_MS)`` -- or
    to replace it with a wait denominated in the product's own progress
    (``tests/settle.py``).
    """

    pauses = [finding for finding in timing_guard.findings() if finding.kind == "pause"]
    assert len(pauses) == UNJUSTIFIED_PAUSE_BACKLOG, (
        f"{len(pauses)} fixed pauses name no throttle, against a recorded backlog of "
        f"{UNJUSTIFIED_PAUSE_BACKLOG}. If you removed some, set "
        "UNJUSTIFIED_PAUSE_BACKLOG to the new number in this file -- it only ever "
        "goes down. If you added one, read its duration from the interval that "
        "bounds it, or wait on the product instead.\n" + timing_guard.report(pauses)
    )


def _kinds(source):
    return [(finding.line, finding.kind) for finding in timing_guard.scan_source(source)]


def test_a_bare_pause_is_named_by_path_and_line():
    """The mutation the guard exists for: a hurried ``qtbot.wait(250)``."""

    findings = timing_guard.scan_source(
        """
def test_thing(qtbot, win):
    win.render()
    qtbot.wait(250)
    assert win.tiles
"""
    )

    assert [str(finding) for finding in findings] == [
        "tests/ui/test_probe.py:4: qtbot.wait(250) pauses for a duration that was "
        "typed, not read from the throttle it is bounded by"
    ]


def test_a_pause_read_from_a_published_throttle_is_accepted():
    """The reason has to be structural, so this is what stating one looks like.

    Naming the constant is not decoration: delete ``PUBLISH_COALESCE_MS`` and
    this test stops importing, where a comment saying the same thing would
    survive the throttle it describes.
    """

    assert (
        _kinds(
            """
from arrayscope.sync.controller import PUBLISH_COALESCE_MS

def test_thing(qtbot, win):
    qtbot.wait(2 * PUBLISH_COALESCE_MS)
"""
        )
        == []
    )


def test_a_pause_read_from_the_live_timer_is_accepted():
    """Not every throttle is a module constant; some are constructor defaults."""

    assert (
        _kinds(
            """
def test_thing(qtbot, win):
    qtbot.wait(2 * win.render_coordinator.interactive_interval_ms)
"""
        )
        == []
    )


def test_a_module_constant_that_is_just_a_number_is_still_a_number():
    """The obvious way to launder a literal: give it capitals and a name."""

    assert _kinds(
        """
COALESCE_MS = 40

def test_thing(qtbot, win):
    qtbot.wait(2 * COALESCE_MS)
"""
    ) == [(5, "pause")]


def test_the_settlement_budget_does_not_bound_a_pause():
    """It is a deadline, so a pause bounded by it is a five-second sleep.

    The import would otherwise satisfy the rule, which is why the one budget
    owner is excluded by name rather than left to look like any other constant.
    """

    assert _kinds(
        """
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

def test_thing(qtbot, win):
    qtbot.wait(INTERACTION_SETTLE_HARD_LIMIT_MS)
"""
    ) == [(5, "pause")]


def test_a_poll_cadence_inside_a_predicate_wait_is_not_a_pause():
    """The wait is the marker; the sleep only decides how often it is asked."""

    assert (
        _kinds(
            """
import time

def _wait_for(marker, deadline):
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    return marker.exists()
"""
        )
        == []
    )


def test_a_loop_that_can_only_end_on_the_clock_is_still_a_pause():
    """A fixed pause with a loop drawn around it, which is the honest reading.

    Without this the poll exemption would swallow the sampling windows: 120 ms
    of ``qtbot.wait(1)`` is 120 ms of sleep however it is spelled.
    """

    assert _kinds(
        """
from time import perf_counter

def test_thing(qtbot, win):
    stop_at = perf_counter() + 0.12
    while perf_counter() < stop_at:
        qtbot.wait(1)
"""
    ) == [(7, "pause")]


def test_simulated_slow_work_is_the_material_and_not_the_wait():
    """A sleep in something handed to the product makes the *product* slow."""

    assert (
        _kinds(
            """
import time

def test_thing(monkeypatch, loader):
    def slowed(self, index):
        time.sleep(0.05)
        return index

    monkeypatch.setattr(loader, "_next_slice", slowed)
"""
        )
        == []
    )


def test_a_helper_holding_the_bot_is_never_simulated_work():
    """The other way into the workload exemption, and it is deliberate.

    Put the pause in a helper, mention the helper once as a value so it is
    never seen being called, and the sleep inside it stops being reported --
    except that a callable the product invokes is never handed ``qtbot``.
    """

    assert _kinds(
        """
import time

def _breathe(qtbot, win):
    win.render()
    time.sleep(0.3)

HELPERS = [_breathe]
"""
    ) == [(6, "pause")]


def test_an_immediately_invoked_lambda_is_the_test_s_own_path():
    """Otherwise the workload exemption is one pair of parentheses wide."""

    assert _kinds(
        """
import time

def test_thing(win):
    (lambda: time.sleep(0.3))()
"""
    ) == [(5, "pause")]


def test_a_bounded_wait_on_a_real_event_is_not_a_pause():
    """``event.wait(timeout=5)`` returns when the thing happens.

    Whether its deadline is capped is a real question, and
    ``test_interaction_gates_have_one_bounded_timeout_owner`` is where it is
    asked. Answering it twice, differently, would be worse than not answering.
    """

    assert (
        _kinds(
            """
def test_thing(started, proc):
    started.wait(timeout=5.0)
    proc.wait(timeout=5)
"""
        )
        == []
    )


def test_hiding_a_pause_behind_a_fresh_helper_reports_the_callers(tmp_path):
    """A wrapper does not launder a pause; it becomes a primitive itself.

    This is the difference between a guard that names four call sites and one
    that names the helper they share and then goes quiet when somebody writes a
    second helper.
    """

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "helpers.py").write_text(
        "def breathe(qtbot, count=8):\n    for _ in range(count):\n        qtbot.wait(10)\n"
    )
    (tests_dir / "test_thing.py").write_text(
        "from tests.helpers import breathe\n\n\n"
        "def test_one(qtbot):\n    breathe(qtbot)\n\n\n"
        "def test_two(qtbot):\n    breathe(qtbot, count=20)\n"
    )

    found = {str(finding).split(" ")[0] for finding in timing_guard.findings(tmp_path)}

    assert found == {"tests/test_thing.py:5:", "tests/test_thing.py:9:"}


def test_an_assert_on_a_duration_is_named():
    assert _kinds(
        """
from time import perf_counter

def test_thing(win):
    started = perf_counter()
    win.render()
    assert perf_counter() - started < 1.0
"""
    ) == [(7, "stopwatch")]


def test_the_threshold_free_comparison_that_survived_everything_else_is_named():
    """``empty_ms <= working_ms``: no constant in it, and still a stopwatch.

    This exact shape passed every rule aimed at thresholds and ratios, was
    flaky under three concurrent suites, and stayed green with the defect it
    named forced back in -- because the regained fixed cost landed on both arms
    of the ``<=`` and cancelled. It is the reason this rule is about the
    reading rather than the comparison.
    """

    assert _kinds(
        """
from time import perf_counter

def test_thing(win):
    start = perf_counter()
    win.commit_empty()
    empty_ms = (perf_counter() - start) * 1000.0
    start = perf_counter()
    win.commit_work()
    working_ms = (perf_counter() - start) * 1000.0
    assert empty_ms <= working_ms
"""
    ) == [(11, "stopwatch")]


def test_a_duration_that_fails_without_saying_assert_is_named():
    """An assert-only rule steps straight over this, so it is not assert-only."""

    assert _kinds(
        """
import pytest
from time import perf_counter

def test_thing(win):
    started = perf_counter()
    win.render()
    if perf_counter() - started > 1.0:
        pytest.fail("too slow")
"""
    ) == [(8, "stopwatch")]


def test_a_deadline_bounded_by_the_budget_owner_is_not_a_stopwatch():
    """The lost-wakeup detector in ``tests/settle.py`` must stay expressible.

    It reads a clock and raises, which is the shape above; what makes it a
    different thing is that its budget is the one owned deadline and only burns
    while the product is idle.
    """

    assert (
        _kinds(
            """
from time import perf_counter
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_S

def settle_until(predicate):
    deadline = perf_counter() + INTERACTION_SETTLE_HARD_LIMIT_S
    while not predicate():
        if perf_counter() > deadline:
            raise AssertionError("lost wakeup")
"""
        )
        == []
    )


def test_recording_a_duration_beside_a_result_is_not_adjudicating_one():
    """``tests/seams.py`` and ``tests/load_report.py`` do exactly this."""

    assert (
        _kinds(
            """
from time import perf_counter

def record(win, out):
    started = perf_counter()
    win.render()
    out.append({"upload_ms": (perf_counter() - started) * 1000.0})
    assert out[-1]["upload_ms"] is not None
"""
        )
        == []
    )


def test_storing_a_reading_on_an_object_does_not_make_the_object_a_clock():
    """The over-approximation has to stop at the attribute it was stored in.

    Taking the root instead cost a false positive on the first run: ``config``
    became a clock reading at ``config.started = monotonic()``, and every later
    ``since = config.getoption(...)`` inherited it, so an unrelated ``if since
    is not None: ... raise`` was reported as a duration gate.
    """

    assert (
        _kinds(
            """
from time import monotonic

def start(config):
    config.started = monotonic()

def finish(config):
    since = config.getoption("since")
    if since is not None:
        raise RuntimeError("no")
"""
        )
        == []
    )
