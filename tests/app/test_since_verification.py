"""``--since`` as a verification report, not a re-run.

Ring 0 — no Qt, no rendering. Three features meet here and they are one idea:

* **The partition.** The map is an evidence ledger. A test whose fingerprint is
  stable executed byte-identical code last time it ran, so a recorded green is
  a green *for this tree*. ``--since`` used to un-deselect the whole reached
  set, which on a faithfully-iterated branch is ~95% re-confirmation.
* **The flake ledger.** The one case where a stable green is not evidence: a
  test observed to flip outcome with no code change at all.
* **The cadence nudge.** Selection's blind spots were documented as CI's
  problem. CI has run seven times ever, so they had no owner.

Every test here is written to fail if the behaviour reverts to un-deselecting
everything, or to trusting a green it should not.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests import coverage_map, red_ledger, testmon_policy
from tests.conftest import (
    _refuse_fresh_without_since,
    _run_swept_exhaustively,
    cadence_nudge,
)

REPO_ROOT = testmon_policy.REPO_ROOT

VERIFIED = "tests/a/test_a.py::test_verified"
UNSTABLE = "tests/a/test_a.py::test_unstable"
NEW = "tests/b/test_new.py::test_never_recorded"
BROKEN = "tests/c/test_c.py::test_broken_here"
INHERITED = "tests/d/test_d.py::test_inherited_red"
FLAKY = "tests/e/test_e.py::test_flaky"


def _partition(**overrides):
    """The five reached tests, one per bucket, so every branch is exercised."""

    arguments = {
        "reached": [VERIFIED, UNSTABLE, NEW, BROKEN, INHERITED, FLAKY],
        "recorded": {VERIFIED, UNSTABLE, BROKEN, INHERITED, FLAKY},
        "stable": {VERIFIED, BROKEN, INHERITED, FLAKY},
        "failed": {BROKEN, INHERITED},
        "broken_here": {BROKEN},
        "flaky": {FLAKY},
    }
    arguments.update(overrides)
    reached = arguments.pop("reached")
    return testmon_policy.partition_reached(reached, **arguments)


# --------------------------------------------------------------------------- #
# The partition
# --------------------------------------------------------------------------- #


def test_a_recorded_green_on_unchanged_code_is_not_re_run():
    """The whole feature in one assertion.

    Content-addressing is what makes this sound: a stable fingerprint means the
    last recorded run executed the same bytes that are on disk now. Re-running
    such a test cannot produce information, only a copy of the record — and
    measured on a settled branch that is ~95% of what ``--since`` used to do.
    """

    buckets = _partition()

    assert buckets[testmon_policy.VERIFIED] == frozenset({VERIFIED})
    assert VERIFIED not in frozenset().union(
        *(buckets[reason] for reason in testmon_policy.RUN_REASONS)
    )


def test_every_kind_of_evidence_gap_runs():
    """Four reasons the map cannot vouch for a test, four buckets, no overlap."""

    buckets = _partition()

    assert buckets["without evidence"] == frozenset({UNSTABLE, NEW})
    assert buckets["broken here"] == frozenset({BROKEN})
    assert buckets["inherited red"] == frozenset({INHERITED})
    assert buckets["known flake"] == frozenset({FLAKY})


def test_the_partition_is_total_so_the_printed_counts_add_up():
    """A report whose parts do not sum is a report nobody can check."""

    buckets = _partition()
    everything = list(buckets.values())

    assert sum(len(names) for names in everything) == 6
    assert frozenset().union(*everything) == frozenset(
        {VERIFIED, UNSTABLE, NEW, BROKEN, INHERITED, FLAKY}
    )


def test_a_known_flake_runs_even_though_it_is_stable_and_green():
    """The one case where a recorded green is not evidence.

    Nothing else in the partition would catch it: the flake is stable, its last
    outcome was a pass, and its dependencies are untouched — it is
    indistinguishable from a verified test *except* for the observation.
    """

    assert _partition(flaky=set())[testmon_policy.VERIFIED] == frozenset({VERIFIED, FLAKY})
    assert _partition()["known flake"] == frozenset({FLAKY})


def test_an_inherited_red_the_branch_reaches_still_runs():
    """`--since` is a merge gate; the moment the branch lands this is main's red."""

    assert _partition()["inherited red"] == frozenset({INHERITED})


def test_a_test_the_map_has_never_seen_can_never_be_verified():
    """New tests are the reason `--since` exists; skipping one would be the defect."""

    assert NEW in _partition()["without evidence"]
    assert NEW not in _partition()[testmon_policy.VERIFIED]


# --------------------------------------------------------------------------- #
# What the run does with it
# --------------------------------------------------------------------------- #


class _Plugin:
    def __init__(self, deselected_tests, deselected_files):
        self.deselected_tests = list(deselected_tests)
        self.deselected_files = list(deselected_files)


class _PluginManager:
    def __init__(self, plugin):
        self._plugin = plugin

    def get_plugin(self, name):
        return self._plugin if name == "TestmonSelect" else None


def _config(tmp_path, monkeypatch, *, flakes=(), fresh=False):
    """A run whose map holds the six tests above, with the ledgers on disk."""

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    map_path = tmp_path / "themap"
    map_path.write_bytes(b"")
    if flakes:
        red_ledger.flake_path(map_path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "environments": {
                        "env": {name: {"count": 1, "last_seen": "x"} for name in flakes}
                    },
                }
            ),
            encoding="utf-8",
        )
    plugin = _Plugin(
        deselected_tests=[VERIFIED, INHERITED, FLAKY, "tests/z/test_z.py::test_elsewhere"],
        deselected_files=["tests/a/test_a.py", "tests/d/test_d.py", "tests/e/test_e.py"],
    )
    data = SimpleNamespace(
        environment="env",
        all_tests={
            VERIFIED: {"failed": False},
            UNSTABLE: {"failed": False},
            BROKEN: {"failed": True},
            INHERITED: {"failed": True},
            FLAKY: {"failed": False},
        },
        stable_test_names={VERIFIED, BROKEN, INHERITED, FLAKY},
    )
    config = SimpleNamespace(
        rootdir=tmp_path,
        pluginmanager=_PluginManager(plugin),
        testmon_data=data,
        testmon_config=SimpleNamespace(select=True, collect=True),
        getoption=lambda name, default=None: {"fresh": fresh}.get(name, default),
    )
    return config, plugin


def _reached_is(monkeypatch, names):
    monkeypatch.setattr(testmon_policy, "resolve_baseline", lambda root, ref: ("main", "base"))
    monkeypatch.setattr(
        testmon_policy, "baseline_method_checksums", lambda root, base: {"x.py": []}
    )
    monkeypatch.setattr(
        testmon_policy, "tests_reached_since_baseline", lambda data, sums: frozenset(names)
    )


REACHED = (VERIFIED, UNSTABLE, NEW, BROKEN, INHERITED, FLAKY)


def test_since_un_deselects_only_the_evidence_gaps(tmp_path, monkeypatch):
    """The behavioural change: the verified test stays deselected.

    Its file stays deselected too — but only because nothing else in that file
    has to run. ``tests/a`` holds an unstable test as well, so the file comes
    back and the verified test inside it does not.
    """

    config, plugin = _config(tmp_path, monkeypatch, flakes=[FLAKY])
    _reached_is(monkeypatch, REACHED)

    report = testmon_policy.apply_since_baseline(config, None, {BROKEN})

    assert report.reached == 6
    assert report.verified == 1
    assert VERIFIED in plugin.deselected_tests, "a proven test must not be re-run"
    assert set(report.run_tests) == {UNSTABLE, NEW, BROKEN, INHERITED, FLAKY}
    for name in (INHERITED, FLAKY):
        assert name not in plugin.deselected_tests
    # Untouched: a test this branch never reached is none of --since's business.
    assert "tests/z/test_z.py::test_elsewhere" in plugin.deselected_tests


def test_a_file_stays_collectable_when_anything_in_it_must_run(tmp_path, monkeypatch):
    config, plugin = _config(tmp_path, monkeypatch, flakes=[FLAKY])
    _reached_is(monkeypatch, REACHED)

    testmon_policy.apply_since_baseline(config, None, {BROKEN})

    assert "tests/a/test_a.py" not in plugin.deselected_files  # holds the unstable one
    assert "tests/e/test_e.py" not in plugin.deselected_files  # holds the flake


def test_a_branch_that_proves_everything_runs_nothing(tmp_path, monkeypatch):
    """The end state of a faithfully-iterated branch, and the point of the feature."""

    config, plugin = _config(tmp_path, monkeypatch)
    _reached_is(monkeypatch, [VERIFIED])
    before = list(plugin.deselected_tests)

    report = testmon_policy.apply_since_baseline(config, None, set())

    assert (report.reached, report.verified, report.run) == (1, 1, 0)
    assert set(plugin.deselected_tests) == set(before)
    assert "verified green" in report.sentence()


def test_fresh_re_acquires_all_evidence(tmp_path, monkeypatch):
    """`--fresh` is the old behaviour, kept for the runs that distrust the map."""

    config, plugin = _config(tmp_path, monkeypatch, flakes=[FLAKY], fresh=True)
    _reached_is(monkeypatch, REACHED)

    report = testmon_policy.apply_since_baseline(config, None, {BROKEN}, fresh=True)

    assert set(report.run_tests) == set(REACHED)
    assert report.verified == 0
    assert VERIFIED not in plugin.deselected_tests
    assert "--fresh" in report.sentence()


def test_an_unreadable_ledger_widens_the_run_it_never_narrows_it(tmp_path, monkeypatch):
    """Degradation has one legal direction, and this is it.

    Everything in this layer fails open. A partition that cannot be computed —
    a testmon upgrade, a corrupt ledger, a map that will not answer — has to
    cost time, never coverage.
    """

    config, plugin = _config(tmp_path, monkeypatch)
    _reached_is(monkeypatch, REACHED)
    monkeypatch.setattr(testmon_policy, "since_partition", lambda *a, **k: None)

    report = testmon_policy.apply_since_baseline(config, None, set())

    assert set(report.run_tests) == set(REACHED)
    assert VERIFIED not in plugin.deselected_tests
    assert report.verified == 0


def test_the_controller_reads_the_partition_before_anything_runs(monkeypatch):
    """*When* the controller computes it changes the answer.

    Under xdist the controller never collects, so it recomputes the report for
    its own summary — and it used to do that at summary time, after every test
    had re-recorded its fingerprint. Measured: a ``--since --fresh`` run printed
    "ran the 233 tests your tree reaches" directly above "branch reaches 185".
    Same paragraph, same run, two different trees.

    The call site is pinned by source, not by driving the hook: this file runs
    inside a live session, and calling ``pytest_sessionstart`` from a test
    resets the very ledgers that session is keeping — which it did, silently
    swallowing this test's own result on the first attempt.
    """

    import inspect

    from tests import conftest

    monkeypatch.setattr(conftest, "_since_on_the_controller", lambda config: "before-the-run")
    off = SimpleNamespace(getoption=lambda name, default=None: default)
    on = SimpleNamespace(getoption=lambda name, default=None: {"since": ""}.get(name, default))

    assert conftest.since_before_the_run(off) is None
    assert conftest.since_before_the_run(on) == "before-the-run"
    assert "since_before_the_run" in inspect.getsource(conftest.pytest_sessionstart), (
        "asking at summary time describes a tree the run has already moved"
    )
    assert "arrayscope_since_controller" in inspect.getsource(conftest.pytest_terminal_summary)


def test_the_report_sentence_states_numbers_that_add_up():
    report = testmon_policy.SinceReport(
        "main",
        reached=65,
        verified=60,
        running=(("broken here", 1), ("without evidence", 3), ("known flake", 1)),
    )
    sentence = report.sentence()

    assert "--since main" in sentence
    assert "65 tests" in sentence
    assert "60 verified green" in sentence
    assert "5 run now" in sentence
    assert "3 without evidence" in sentence
    assert "1 broken here" in sentence
    assert "1 known flake" in sentence
    assert report.run == 5


def test_the_closing_summary_carries_the_report_on_every_since_run():
    """It used to print only when the run *grew*, so the best answer was silent."""

    from tests.conftest import unaffected_summary

    proven = testmon_policy.SinceReport("main", reached=65, verified=65)
    assert "verified green" in "\n".join(unaffected_summary(0, 3941, 3941, 222.0, since=proven))


def test_fresh_without_since_is_an_error_not_a_no_op():
    """Alone it would silently do nothing, which is how a phantom sweep happens."""

    _refuse_fresh_without_since(
        SimpleNamespace(getoption=lambda name, default=None: {"fresh": False}.get(name, default))
    )
    _refuse_fresh_without_since(
        SimpleNamespace(
            getoption=lambda name, default=None: {"fresh": True, "since": ""}.get(name, default)
        )
    )
    with pytest.raises(pytest.UsageError, match="only means something with --since"):
        _refuse_fresh_without_since(
            SimpleNamespace(getoption=lambda name, default=None: {"fresh": True}.get(name, default))
        )


# --------------------------------------------------------------------------- #
# The flake ledger
# --------------------------------------------------------------------------- #


def test_a_forced_red_that_passes_is_written_down(tmp_path):
    """The observation every run used to throw away.

    It re-ran *only* because it was red — nothing it executes changed — and it
    passed. There is no code-shaped explanation for that.
    """

    ledger = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    ledger.observe(FLAKY, when="2026-07-30T10:00:00+00:00")
    ledger.save()

    marks = red_ledger.read_flakes(tmp_path / "themap", "env")
    assert marks[FLAKY]["count"] == 1
    assert marks[FLAKY]["last_seen"] == "2026-07-30T10:00:00+00:00"
    assert red_ledger.read_flakes(tmp_path / "themap", "other") == {}


def test_a_second_observation_counts_rather_than_replaces(tmp_path):
    ledger = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    ledger.observe(FLAKY, when="a")
    ledger.save()
    again = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    again.observe(FLAKY, when="b")
    again.save()

    # The two fields every reader of this file has always used, unchanged by
    # the per-observation history that now sits beside them
    # (tests/app/test_flake_context.py owns that half).
    marks = red_ledger.read_flakes(tmp_path / "themap", "env")
    assert marks[FLAKY]["count"] == 2
    assert marks[FLAKY]["last_seen"] == "b"


def test_the_mark_clears_when_the_dependencies_actually_change(tmp_path):
    """New code is new evidence; the old flakiness was never about it.

    Without this the mark is permanent, and a test that was flaky once would be
    re-run by every ``--since`` for the rest of the repository's life.
    """

    ledger = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    ledger.observe(FLAKY, when="a")
    ledger.observe(VERIFIED, when="a")
    ledger.save()

    reopened = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    reopened.forget({FLAKY})
    reopened.save()

    assert set(red_ledger.read_flakes(tmp_path / "themap", "env")) == {VERIFIED}


def test_a_corrupt_flake_ledger_reads_as_empty(tmp_path):
    """It must cost a trusted green, never a crashed run."""

    red_ledger.flake_path(tmp_path / "themap").write_text("{ not json", encoding="utf-8")
    assert red_ledger.read_flakes(tmp_path / "themap", "env") == {}


def test_the_two_ledgers_do_not_share_a_file(tmp_path):
    """Separate writers, separate files — no merge, so no field to drop."""

    assert red_ledger.flake_path(tmp_path / "themap") != red_ledger.ledger_path(tmp_path / "themap")


# --------------------------------------------------------------------------- #
# The cadence nudge
# --------------------------------------------------------------------------- #


def test_a_fresh_record_says_nothing():
    record = {"sha": "abc", "at": "x", "swept": True}

    assert cadence_nudge(record, days=3.0, commits=5) == ""
    assert cadence_nudge({}, days=900.0, commits=900) == ""


def test_either_clock_running_out_is_enough():
    """A checkout can sit for three weeks, or take forty commits in an afternoon."""

    record = {"sha": "abc", "at": "x", "swept": True}

    assert "27 days ago" in cadence_nudge(record, days=27.0, commits=5)
    assert "63 commits" in cadence_nudge(record, days=3.0, commits=63)
    assert "--testmon-noselect" in cadence_nudge(record, days=27.0, commits=63)


def test_an_anchor_is_not_reported_as_a_sweep():
    """A checkout that has never swept must not be told when it last did."""

    anchor = {"sha": "abc", "at": "x", "swept": False}
    line = cadence_nudge(anchor, days=27.0, commits=63)

    assert "never been swept" in line
    assert "last swept" not in line


def test_a_sha_git_no_longer_knows_still_reports_the_days():
    """A rebase can orphan the recorded commit; the clock keeps running."""

    record = {"sha": "gone", "at": "x", "swept": True}
    line = cadence_nudge(record, days=27.0, commits=None)

    assert "27 days ago" in line
    assert "commits" not in line


def _sweep_config(**options):
    defaults = {"testmon_noselect": False, "keyword": "", "markexpr": "", "lf": False}
    return SimpleNamespace(
        option=SimpleNamespace(**{**defaults, **options}),
        rootdir=REPO_ROOT,
        args=[],
        invocation_params=SimpleNamespace(dir=str(REPO_ROOT)),
        getini=lambda name: ["tests"] if name == "testpaths" else [],
        arrayscope_selection=testmon_policy.Decision(True, "default"),
    )


def test_only_a_run_that_really_executed_everything_counts_as_a_sweep():
    noselect = _sweep_config(testmon_noselect=True)

    assert _run_swept_exhaustively(noselect, ran=3941) is True
    # Nothing ran: --collect-only, or an aborted session.
    assert _run_swept_exhaustively(noselect, ran=0) is False
    # A manual selector narrows the run without narrowing its path arguments.
    assert _run_swept_exhaustively(
        _sweep_config(testmon_noselect=True, keyword="render"), 3941
    ) is (False)
    # A selected run is exactly the one that did not sweep.
    assert _run_swept_exhaustively(_sweep_config(), ran=3941) is False


def test_a_run_with_selection_switched_off_counts_too():
    """`--no-testmon`, `--cov` and CI all execute the whole suite."""

    config = _sweep_config()
    config.arrayscope_selection = testmon_policy.Decision(False, "--no-testmon")

    assert _run_swept_exhaustively(config, ran=3941) is True


def test_the_sweep_record_survives_another_feature_writing_the_sidecar(tmp_path):
    """The preservation contract, in the direction this field cares about."""

    map_path = tmp_path / "themap"
    coverage_map.write_sidecar(
        map_path,
        structure={},
        baselines={},
        sweep={"sha": "abc", "at": "2026-07-01T00:00:00+00:00", "swept": True},
    )
    coverage_map.write_sidecar(map_path, structure={"a": 1}, baselines={"env": {}})

    assert coverage_map.last_sweep(coverage_map.read_sidecar(map_path))["sha"] == "abc"
