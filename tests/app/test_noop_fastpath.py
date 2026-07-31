"""Guards for the run that has nothing to run (``tests/conftest.py``).

Ring 0 — no Qt, no rendering, no map.

A no-op ``pytest`` is 100% fixed cost, and two thirds of it is pytest building a
session that then collects nothing. The fast path answers from the map instead
and exits before the session exists. That is worth 0.5 s of every iteration and
it is also the single most dangerous shortcut in this harness: a fast path that
fired one condition too eagerly would report an edit as fully tested having run
nothing at all — the scope-erosion failure class.

So each condition that must defeat it gets its own case here, and the one that
matters most gets two: a brand-new ``test_*.py`` has no map entry, so nothing
marks it affected and nothing marks it changed, and only the *unmapped* count
notices it exists.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests import conftest as harness
from tests import coverage_map, testmon_policy


def _refusal(**overrides) -> str:
    """The refusal for a run that is otherwise a clean, empty, bare ``pytest``."""

    inputs = {
        "inspecting": False,
        "active": True,
        "selects": True,
        "since": False,
        "whole_suite": True,
        "readable_map": True,
        "workload_files": 0,
        "live_reading": True,
    }
    inputs.update(overrides)
    return harness.noop_fast_path_refusal(**inputs)


def test_a_clean_tree_with_nothing_affected_takes_the_fast_path():
    assert _refusal() == ""


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        # The trap, and the reason the discriminator is a *count* of files with
        # work in them rather than the affected set: a new test file is neither
        # affected nor changed, because the map has never heard of it.
        ({"workload_files": 1}, "1 test file(s) have work in them"),
        ({"since": True}, "--since widens the run"),
        ({"selects": False}, "a selector"),
        ({"whole_suite": False}, "a scoped run"),
        ({"active": False}, "selection is not active"),
        ({"readable_map": False}, "the map is missing"),
        ({"inspecting": True}, "asked to see collection"),
        ({"workload_files": None}, "could not be priced"),
        ({"live_reading": False}, "the map reading was not kept"),
    ],
)
def test_each_condition_falls_back_to_a_full_run(override, expected):
    reason = _refusal(**override)
    assert reason, f"{override} must defeat the fast path"
    assert expected in reason


def test_a_new_test_file_is_counted_as_work_even_though_nothing_marks_it(tmp_path, monkeypatch):
    """The condition the whole design turns on, at the level it is decided.

    ``_selection_workload`` unions affected files, forced reds and *unmapped*
    test files. Only the last of the three can see a file the map has never
    recorded — and a fast path that trusted "0 affected" would run nothing and
    call the new test's file tested.
    """

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_brand_new.py").write_text("def test_x(): pass\n")

    peek = testmon_policy.MapPeek(
        environment="offscreen",
        mapped_tests=10,
        mapped_files=frozenset({"tests/test_old.py"}),
        affected_tests=frozenset(),
        affected_files=frozenset(),
        affected_seconds=0.0,
    )
    config = _workload_config(tmp_path)
    monkeypatch.setattr(harness, "_map_peek", lambda config, **_: peek)
    monkeypatch.setattr(harness, "_new_reds", lambda config: set())
    monkeypatch.setattr(testmon_policy, "decide", lambda config: testmon_policy.Decision(True, "d"))

    files, _seconds, unmapped = harness._selection_workload(config)
    assert unmapped == 1
    assert files == 1
    assert _refusal(workload_files=files) != ""


def test_a_tree_the_map_has_seen_every_file_of_leaves_no_work(tmp_path, monkeypatch):
    """The mirror of the case above, so "1" is not simply what it always says."""

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_old.py").write_text("def test_x(): pass\n")

    peek = testmon_policy.MapPeek(
        environment="offscreen",
        mapped_tests=10,
        mapped_files=frozenset({"tests/test_old.py"}),
        affected_tests=frozenset(),
        affected_files=frozenset(),
        affected_seconds=0.0,
    )
    config = _workload_config(tmp_path)
    monkeypatch.setattr(harness, "_map_peek", lambda config, **_: peek)
    monkeypatch.setattr(harness, "_new_reds", lambda config: set())
    monkeypatch.setattr(testmon_policy, "decide", lambda config: testmon_policy.Decision(True, "d"))

    assert harness._selection_workload(config) == (0, 0.0, 0)


def test_a_red_this_checkout_broke_is_work_however_stable_the_map_says_it_is(tmp_path, monkeypatch):
    """The map cannot vouch for a test the run that wrote the map broke."""

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_old.py").write_text("def test_x(): pass\n")

    peek = testmon_policy.MapPeek(
        environment="offscreen",
        mapped_tests=10,
        mapped_files=frozenset({"tests/test_old.py"}),
        affected_tests=frozenset(),
        affected_files=frozenset(),
        affected_seconds=0.0,
    )
    config = _workload_config(tmp_path)
    monkeypatch.setattr(harness, "_map_peek", lambda config, **_: peek)
    monkeypatch.setattr(harness, "_new_reds", lambda config: {"tests/test_old.py::test_x"})
    monkeypatch.setattr(testmon_policy, "decide", lambda config: testmon_policy.Decision(True, "d"))

    files, _seconds, unmapped = harness._selection_workload(config)
    assert (files, unmapped) == (1, 0)
    assert _refusal(workload_files=files) != ""


def _workload_config(rootdir):
    return SimpleNamespace(
        rootdir=rootdir,
        args=["tests"],
        invocation_params=SimpleNamespace(dir=rootdir),
        option=SimpleNamespace(
            testmon_forceselect=False, testmon_noselect=False, keyword=None, markexpr=None, lf=False
        ),
        getini={
            "testpaths": ["tests"],
            "environment_expression": "'offscreen'",
            "testmon_ignore_dependencies": [],
        }.get,
        getoption=lambda name, default=None: None,
        arrayscope_selection_seed=None,
    )


# --------------------------------------------------------------------------- #
# What the fast path prints
# --------------------------------------------------------------------------- #


class _FakeData:
    def __init__(self, environment="offscreen", mapped=4028):
        self.environment = environment
        self.all_tests = {f"tests/test_a.py::test_{index}": {} for index in range(mapped)}


def _printing_config(tmp_path, mapped=4028):
    peek = testmon_policy.MapPeek(
        environment="offscreen",
        mapped_tests=mapped,
        mapped_files=frozenset(),
        affected_tests=frozenset(),
        affected_files=frozenset(),
        affected_seconds=0.0,
        forced_tests=frozenset({"tests/test_a.py::test_red"}),
        data=_FakeData(mapped=mapped),
    )
    config = SimpleNamespace(
        rootdir=tmp_path,
        args=["tests"],
        invocation_params=SimpleNamespace(dir=tmp_path),
        option=SimpleNamespace(color="no", code_highlight="no"),
        getini={"testpaths": ["tests"], "environment_expression": "'offscreen'"}.get,
        getoption=lambda name, default=None: None,
        arrayscope_selection=testmon_policy.Decision(True, "default"),
        arrayscope_selection_seed=None,
    )
    return config, peek


def test_the_fast_path_prints_what_the_full_path_prints(
    tmp_path, capsys, monkeypatch, as_the_controller
):
    """The lines, in the order ``pytest_terminal_summary`` emits them.

    Not a golden file: each assertion names the line it is there to keep, so a
    deliberate rewording moves one assertion rather than a blob. The point is
    that the two paths cannot silently diverge into "the quiet one and the
    informative one".
    """

    config, peek = _printing_config(tmp_path)
    monkeypatch.setattr(testmon_policy, "decide", lambda config: config.arrayscope_selection)

    assert harness._report_nothing_to_run(config, peek) == 0

    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert "test session starts" in lines[0]
    assert "test selection: on (default) — 0 of 4028 mapped tests affected" in printed
    assert f"rootdir: {tmp_path}" in printed
    assert "collected 0 items" in printed
    # The sentence that stops a green selected run turning into a sweep.
    assert "test selection: nothing to run" in printed
    assert "That is a result, not a gap." in printed
    assert "Branch-sized question, before merging:  pytest --since" in printed
    assert "1 inherited reds not re-run" in printed
    assert "no tests ran in" in lines[-1]


def test_the_exit_code_is_the_one_the_full_path_produces(
    tmp_path, monkeypatch, capsys, as_the_controller
):
    """Measured, not assumed: a no-op selected run exits 0 today. A fast path
    exiting 5 (``NO_TESTS_COLLECTED``) would break every script that treats a
    clean loop as success."""

    config, peek = _printing_config(tmp_path)
    monkeypatch.setattr(testmon_policy, "decide", lambda config: config.arrayscope_selection)
    assert harness._report_nothing_to_run(config, peek) == 0
    capsys.readouterr()


def test_the_coverage_measure_still_runs_on_a_run_that_ran_nothing(
    tmp_path, monkeypatch, capsys, as_the_controller
):
    """The one thing that must not be skipped for speed.

    An edit can delete or rename a covered function while affecting no test, and
    this figure is the only thing in the run that would say so. Rejected as a
    saving on its merits, not on cost — see ``docs/testing/test-selection.md``.
    """

    config, peek = _printing_config(tmp_path)
    monkeypatch.setattr(testmon_policy, "decide", lambda config: config.arrayscope_selection)
    measured = []
    monkeypatch.setattr(
        coverage_map, "measure", lambda *args, **kwargs: measured.append(args) or None
    )

    harness._report_nothing_to_run(config, peek)
    capsys.readouterr()
    assert measured, "the fast path skipped the coverage measure"


def test_the_erosion_canary_still_runs_on_a_run_that_ran_nothing(
    tmp_path, monkeypatch, capsys, as_the_controller
):
    """A map that lost a chunk of itself reports "nothing affected" — which is
    exactly the shape of run this path is for, so the canary has to be on it."""

    config, peek = _printing_config(tmp_path)
    monkeypatch.setattr(testmon_policy, "decide", lambda config: config.arrayscope_selection)
    coverage_map.write_sidecar(
        testmon_policy.map_path(tmp_path),
        structure={},
        baselines={},
        map_sizes={"offscreen": 9000},
    )

    harness._report_nothing_to_run(config, peek)
    printed = capsys.readouterr().out
    assert "the map lost 4972 of 9000 recorded tests" in printed


def test_the_header_line_is_the_one_the_full_path_uses():
    """One spelling, shared, because it is the line an agent reads to decide
    whether the run it got is the run it asked for."""

    assert harness.selection_header_line("default", 0, 4028) == (
        "test selection: on (default) — 0 of 4028 mapped tests affected"
    )
    assert "records the map" in harness.selection_header_line("default", 0, 0)


def test_the_cadence_nudge_still_fires_on_a_run_that_ran_nothing(
    tmp_path, monkeypatch, capsys, as_the_controller
):
    """A settled tree answered from the map is the run shape a quiet fortnight
    is made of — a checkout living entirely on the fast path is exactly the one
    whose blind spots age out unnoticed, so the nudge must survive the path or
    it exists only for the runs that least need it."""

    config, peek = _printing_config(tmp_path)
    monkeypatch.setattr(testmon_policy, "decide", lambda config: config.arrayscope_selection)
    coverage_map.write_sidecar(
        testmon_policy.map_path(tmp_path),
        structure={},
        baselines={},
        # Fully stamped because this test exercises the cadence CLOCK: a record
        # missing either stamp takes the attribution branch instead, which has
        # its own fast-path test in test_since_verification.py.
        sweep={
            "sha": "0" * 40,
            "at": "2026-05-01T00:00:00+00:00",
            "swept": True,
            "attributed": True,
            "skips": True,
        },
    )

    harness._report_nothing_to_run(config, peek)
    printed = capsys.readouterr().out
    assert "swept exhaustively" in printed, "the fast path dropped the cadence nudge"


def test_the_fast_path_withdraws_the_claim_when_a_record_came_from_a_skip(
    tmp_path, capsys, monkeypatch, as_the_controller
):
    """This is *the* run that said "nothing to run — that is a result, not a gap"
    for a rewritten `screen_canvas.py` whose only real coverage was eleven
    skip-recorded ring-4 tests.

    It reads the ledger from the file rather than from memory, and that is not
    incidental: this path never reaches ``pytest_sessionstart``, so the live
    ledger every other caller uses does not exist here. A remainder that only
    worked in the full path would miss the exact shape that caused the miss.
    """

    from tests import skip_ledger

    config, peek = _printing_config(tmp_path)
    monkeypatch.setattr(testmon_policy, "decide", lambda config: config.arrayscope_selection)
    # What production has here: the fast path returns from `pytest_cmdline_main`
    # before `pytest_sessionstart`, so no ledger is open. Patched rather than
    # assumed, because the session *running this assertion* has one of its own.
    monkeypatch.setattr(harness, "_skip_ledger", None)
    ledger = skip_ledger.SkipLedger.load(tmp_path / ".testmondata", "offscreen")
    for index in range(11):
        ledger.record(f"tests/test_a.py::test_{index}", skipped=True)
    ledger.save()

    assert harness._report_nothing_to_run(config, peek) == 0

    printed = capsys.readouterr().out
    assert "test selection: nothing to run" in printed
    assert "That is a result, not a gap." not in printed, "a remainder is exactly a gap"
    assert "Unmeasured remainder: 11 tests" in printed
    assert "tests/test_a.py (11)" in printed
