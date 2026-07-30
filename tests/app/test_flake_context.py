"""A flake observation, and the conditions it happened under.

Ring 0 — no Qt, no rendering. The flake ledger itself lives in
``tests/app/test_since_verification.py``: what makes a mark, what clears it,
what ``--since`` does with one. This file owns the two things added around it,
which are one idea in two halves:

* **The observation survives its checkout.** Work here happens in throwaway
  worktrees, so an observation used to be deleted with the worktree that made
  it — while the map that observation is *about* went on to be adopted by the
  main checkout, arriving able to vouch for a green somebody had watched flip.
* **The observation says under what.** A count answers "how often"; the next
  question is always "under what?" — a loaded machine, eight workers, a
  ``--since`` — and nothing was writing that down.

The boundary is load-bearing and is asserted here rather than only documented:
the harness **records** confounders and **prints** them, and never uses one to
discount an observation. Load is a trigger for real flakes at least as often as
it is an alibi, so an automatic "explained by load" would hide exactly the
defects worth finding. Every judgement stays with the human reading the line.

They are together in one file because they are one contract about one record,
and split across the modules that implement it (the ledger, the carrier, the
observation site) it could not be read in one sitting.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from tests import load_report, red_ledger, testmon_policy
from tests.conftest import _run_conditions, _worker_count

FLAKY = "tests/e/test_e.py::test_flaky"
OTHER = "tests/f/test_f.py::test_other"

REPO_ROOT = testmon_policy.REPO_ROOT


def _write_ledger(map_path, marks, environment="env"):
    red_ledger.flake_path(map_path).write_text(
        json.dumps({"version": 1, "environments": {environment: marks}}),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Context per observation
# --------------------------------------------------------------------------- #


def test_an_observation_records_what_the_run_looked_like(tmp_path):
    """The count says how often. This says under what — which is the next question.

    Without it a flake ledger is a list of names: nothing in it can distinguish
    "flips on a quiet machine, serially" (a real defect, and a cheap one to
    reproduce) from "seen four times, always at load 9 on eight workers".
    """

    ledger = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    ledger.observe(
        FLAKY,
        when="2026-07-30T21:12:00+00:00",
        context={"load": 6.2, "foreign_load": 4.1, "workers": 8, "shape": "loop"},
    )
    ledger.save()

    mark = red_ledger.read_flakes(tmp_path / "themap", "env")[FLAKY]
    assert mark["recent"] == [
        {
            "at": "2026-07-30T21:12:00+00:00",
            "load": 6.2,
            "foreign_load": 4.1,
            "workers": 8,
            "shape": "loop",
        }
    ]


def test_the_history_is_bounded_and_newest_first(tmp_path):
    """A ledger nobody prunes is a ledger that grows until nobody opens it."""

    ledger = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    for index in range(8):
        ledger.observe(FLAKY, when=f"2026-07-30T00:0{index}:00+00:00", context={"workers": index})
    ledger.save()

    mark = red_ledger.read_flakes(tmp_path / "themap", "env")[FLAKY]
    assert mark["count"] == 8, "the count is not capped; only the conditions are"
    assert [entry["workers"] for entry in mark["recent"]] == [7, 6, 5, 4, 3]


def test_a_ledger_written_before_this_still_reads(tmp_path):
    """Additive, so nothing has to be migrated and no version had to move.

    A file whose marks predate the history reads as "no conditions recorded",
    which is the truth about it, and the next observation starts a history
    without losing the count that was already there.
    """

    _write_ledger(tmp_path / "themap", {FLAKY: {"count": 3, "last_seen": "2026-07-01T00:00:00"}})

    ledger = red_ledger.FlakeLedger.load(tmp_path / "themap", "env")
    assert ledger.marks[FLAKY]["count"] == 3
    assert "recent" not in ledger.marks[FLAKY]
    ledger.observe(FLAKY, when="2026-07-30T00:00:00+00:00", context={"workers": 8})
    ledger.save()

    mark = red_ledger.read_flakes(tmp_path / "themap", "env")[FLAKY]
    assert mark["count"] == 4
    assert [entry["workers"] for entry in mark["recent"]] == [8]


def test_the_load_context_names_which_sample_it_got(tmp_path):
    """An early observation has no honest foreign-load estimate, and says so.

    Subtracting this run's own parallelism needs two samples separated by
    enough wall clock to divide by. Milliseconds in, the arithmetic returns a
    parallelism of hundreds and a foreign load that is large and negative — a
    number that looks measured and is not. So the opening sample is reported
    under a name that cannot be mistaken for the settled one.
    """

    start = load_report.LoadWindow.sample()
    early = load_report.observation_context(start)
    assert "foreign_load" not in early
    assert early == {} or "load_at_start" in early

    settled = load_report.observation_context(
        load_report.LoadWindow(load=start.load, wall=start.wall - 60.0, cpu_seconds=0.0)
    )
    assert settled == {} or {"load", "foreign_load"} <= set(settled)


def test_the_run_shape_distinguishes_the_four_commands():
    """Which command it was changes how much the observation is worth reading.

    The same flip under a plain loop and under ``--testmon-noselect`` are not
    the same fact: the loop re-ran the test *because* it was red, while a
    noselect run re-ran everything and the "nothing it executes changed" half
    comes from the map instead. Recorded as a label; nothing branches on it.
    """

    def config(**options):
        return SimpleNamespace(
            getoption=lambda name, default=None: options.get(name, default),
            option=SimpleNamespace(numprocesses=options.get("numprocesses", 0)),
        )

    assert _run_conditions(config())["shape"] == "loop"
    assert _run_conditions(config(rerun_reds=True))["shape"] == "rerun-reds"
    assert _run_conditions(config(since="main"))["shape"] == "since"
    assert _run_conditions(config(since="main", testmon_noselect=True))["shape"] == "noselect"

    assert _run_conditions(config(numprocesses=8))["workers"] == 8
    assert _run_conditions(config(numprocesses=None))["workers"] == 0, "in-process is zero"
    # An unresolved `-n auto` is unknown, and a guessed "serial" would be
    # indistinguishable from a measured one to anyone reading the ledger.
    assert _worker_count(config(numprocesses="auto")) is None
    assert "workers" not in _run_conditions(config(numprocesses="auto"))


# --------------------------------------------------------------------------- #
# Carrying the ledger between checkouts
# --------------------------------------------------------------------------- #


def _donor_and_worktree(tmp_path, monkeypatch):
    donor_checkout, worktree = tmp_path / "main", tmp_path / "worktree"
    donor_checkout.mkdir()
    worktree.mkdir()
    donor = donor_checkout / "themap"
    donor.write_bytes(b"donor-map")
    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    monkeypatch.setattr(testmon_policy, "_best_donor_map", lambda rootdir, filename: donor)
    return donor, worktree


def test_seeding_carries_the_donors_observations(tmp_path, monkeypatch):
    """A fresh worktree inherits the map *and* what is known not to be evidence.

    The map arrives claiming a stable green for every recorded test. Without
    the ledger beside it, the greens the donor had already watched flip arrive
    indistinguishable from the trustworthy ones, and `--since` verifies against
    them.
    """

    donor, worktree = _donor_and_worktree(tmp_path, monkeypatch)
    _write_ledger(donor, {FLAKY: {"count": 2, "last_seen": "2026-07-29T00:00:00+00:00"}})

    assert testmon_policy.seed_map(worktree) == str(donor)
    assert red_ledger.read_flakes(worktree / "themap", "env")[FLAKY]["count"] == 2


def test_adoption_carries_the_observations_home(tmp_path, monkeypatch):
    """The other direction, and the one that was actually losing them.

    Almost every run happens in a worktree that is then deleted. The map it
    recorded is adopted by the main checkout on a fast-forward; the flakes it
    witnessed used to be thrown away with the worktree, so the main checkout
    re-learned the same ones for as long as they existed.
    """

    monkeypatch.setenv("TESTMON_DATAFILE", "themap")
    root, branch = tmp_path / "main", tmp_path / "wt"
    root.mkdir()
    branch.mkdir()
    (root / "themap").write_bytes(b"stale")
    (branch / "themap").write_bytes(b"recorded-the-merge")
    os.utime(root / "themap", (1, 1))
    os.utime(branch / "themap", (2, 2))  # the newer map wins
    _write_ledger(branch / "themap", {FLAKY: {"count": 1, "last_seen": "2026-07-30T00:00"}})
    monkeypatch.setattr(
        testmon_policy, "_worktree_heads", lambda rootdir: [(root, "abc123"), (branch, "abc123")]
    )

    assert testmon_policy.adopt_map_from_identical_worktree(root) == str(branch / "themap")
    assert set(red_ledger.read_flakes(root / "themap", "env")) == {FLAKY}


def test_carrying_merges_rather_than_replaces(tmp_path):
    """Both sides observed something real; a copy would delete one side's.

    Counts add and both node ids survive. The count is "how often this kept
    happening" rather than an audited total — an observation seeded out and
    adopted back is counted twice, which is why nothing branches on its value.
    """

    donor, destination = tmp_path / "donor", tmp_path / "destination"
    _write_ledger(
        donor,
        {
            FLAKY: {
                "count": 2,
                "last_seen": "2026-07-29T00:00:00+00:00",
                "recent": [{"at": "2026-07-29T00:00:00+00:00", "workers": 8}],
            }
        },
    )
    _write_ledger(
        destination,
        {
            FLAKY: {
                "count": 1,
                "last_seen": "2026-07-30T00:00:00+00:00",
                "recent": [{"at": "2026-07-30T00:00:00+00:00", "workers": 0}],
            },
            OTHER: {"count": 5, "last_seen": "2026-07-28T00:00:00+00:00"},
        },
    )

    red_ledger.carry_flakes(donor, destination)

    marks = red_ledger.read_flakes(destination, "env")
    assert marks[FLAKY]["count"] == 3
    assert marks[FLAKY]["last_seen"] == "2026-07-30T00:00:00+00:00", "the newer sighting wins"
    assert [entry["workers"] for entry in marks[FLAKY]["recent"]] == [0, 8]
    assert marks[OTHER]["count"] == 5, "a mark only this side had must survive the merge"


def test_a_ledger_that_cannot_be_written_costs_observations_and_nothing_else(tmp_path, monkeypatch):
    """Same tolerance as the coverage sidecar, for the same reason.

    The map is already in place by the time the ledger travels. A failure here
    leaves the destination exactly as trustworthy as it was a moment ago — one
    fewer flake it knows about, no wrong answer — so it must not take the seed
    down with it.
    """

    donor, worktree = _donor_and_worktree(tmp_path, monkeypatch)
    _write_ledger(donor, {FLAKY: {"count": 1, "last_seen": "x"}})
    # A directory where the file wants to be: every write raises IsADirectoryError.
    red_ledger.flake_path(worktree / "themap").mkdir()

    assert testmon_policy.seed_map(worktree) == str(donor)
    assert (worktree / "themap").read_bytes() == b"donor-map"
    assert red_ledger.read_flakes(worktree / "themap", "env") == {}
