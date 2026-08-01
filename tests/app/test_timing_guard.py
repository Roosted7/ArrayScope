"""Proof that the timing rules in :mod:`tests.timing_guard` can actually fail.

Ring 0 -- no Qt, no ``arrayscope`` import, nothing built, no real tree read but
one synthetic package. The whole-tree scan is NOT run here: it is a lint, gated
at commit time by ``.githooks/pre-commit`` through ``tools/timing_guard.py``,
for the reason ``tests/app/test_dead_code.py`` gives about its own scan and one
more that is specific to this guard. A pause is added by editing a *test* file,
and testmon selects that file rather than the guard that reads it -- the gap
``testmon_policy.forgivable_checksums`` already names for ``test_dead_code`` and
``test_architecture_guards``. Under pytest the pause would therefore be met on
some later full or ``--since`` run instead of at the moment somebody wrote it,
which is the entire point of having the guard. At the hook it is refused as it
is written, and the ~2.5 s comes off every suite run.

What is left is the part pytest is actually for. A guard nobody can see fail is
the failure mode this repository keeps meeting -- five times by
``docs/reviews/2026-07-27``, again in the assertion this week's work deleted
(green with the defect forced back in), and once more in these very rules: a
test whose whole body was a pause matched "function that does nothing but
pause", so the guard classified it as a *wrapper* and reported nothing, green on
the real tree and green with the mutation in it. Only
``test_a_bare_pause_is_named_by_path_and_line`` below asked the question.

So every rule has a case that would pass silently without it, and every
exemption has a case showing what it lets through beside one showing what it
still refuses. These are all this file has to keep those rules falsifiable, and
they are instant.
"""

from __future__ import annotations

from tests import timing_guard


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
