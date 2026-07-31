"""Remember which recorded tests never actually ran.

The map records, per test, the source lines that test executed — and it records
them for a test that *skipped* exactly as it records them for one that ran. A
skip still imports the test module and still evaluates whatever gate turned it
away, so it produces a small, entirely real-looking dependency set, with
``failed`` false and a duration of about a millisecond. Nothing in testmon's
schema distinguishes it from a fast green.

That is the whole defect. Selection reads "this test's recorded dependencies do
not include your change" and concludes *this test is unaffected*, when the only
thing the record supports is *this test did not run*. Measured on
2026-07-31, ``tests/gpu_interaction/test_wgpu_screen_present.py`` under the
default ``offscreen`` environment: 10 tests, 2-3 dependency files each, none of
them ``screen_canvas.py``. The same ten tests under ``xcb+gpu`` and
``wayland+gpu``, where the display gate opens and they really run, record 26-224
files each with ``screen_canvas.py`` among them. A branch rewriting that file
was told "nothing to run — no line your working tree changed is executed by any
of the 11 recorded tests. That is a result, not a gap", and ``--since`` called
the branch fully verified. It was the only real coverage the change had.

So the outcome is observed and written down here, because it is a fact only a
run can see and every run used to throw it away — the same argument, and the
same shape, as ``tests/red_ledger.py``. An entry means "this test's recorded
dependency set is not evidence about anything". What the mark buys:

* ``--since`` never counts that test's recorded green as verification, and
  reaches it instead. The gate is the run that makes a completeness claim, so
  the gate is the run that pays to make it true. A skip costs about a
  millisecond to re-attempt, so this is affordable in a way that re-running the
  whole opt-in ring would not be.
* The inner loop's closing sentence stops folding it into "unaffected by
  construction" and names it as an unmeasured remainder instead. The loop makes
  no completeness claim, so it does not pay; it just stops lying.

**Self-healing, in the direction that matters.** The mark clears on exactly one
event: the test runs for real. That is the moment its record stops being a
statement about a gate and becomes a statement about code, which is precisely
when it becomes evidence again. Nothing has to expire, and no entry can outlive
the condition that justified it.

**Per environment**, because the skip is a property of the environment and not
of the test: the ring-4 tests skip under ``offscreen`` and run under
``xcb+gpu``. Sharing one set across environments would mark them unmeasured in
the very environment that measures them.

One thing this cannot do is speak for a test it has never observed. A map
recorded before this file existed holds skips it cannot identify, and the
absence of a mark there means "never seen", not "ran". That is why the sweep
record carries a ``skips`` stamp and ``testmon_policy.attribution_gap`` reports
its absence — the same treatment, for the same reason, as the per-test
attribution fix earlier the same day.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

#: Sits next to the map, shares its lifetime, gitignored like it. Deliberately
#: its own file rather than another key in the red or flake ledger: the three
#: records are written on different events and cleared on different ones, and a
#: single writer that had to merge them is one careless rewrite away from
#: dropping the field it did not think about — the defect ``cbf3f254`` fixed in
#: the coverage sidecar. Three files, three owners, no merge.
SKIP_SUFFIX = "-skips.json"

_VERSION = 1


def skip_path(map_path: Path) -> Path:
    return map_path.with_name(map_path.name + SKIP_SUFFIX)


class SkipLedger:
    """Node ids whose most recent recorded run ended in a skip, per environment."""

    def __init__(self, path: Path, environment: str) -> None:
        self.path = path
        self.environment = environment
        self._environments = _read(path)
        self.skipped: set[str] = set(self._environments.get(environment, ()))
        self._initial = frozenset(self.skipped)

    @classmethod
    def load(cls, map_path, environment: str) -> SkipLedger:
        return cls(skip_path(Path(map_path)), environment)

    def record(self, nodeid: str, skipped: bool) -> None:
        """Fold one test's outcome in. Idempotent within a run.

        ``skipped`` false is as load-bearing as true: it is the one event that
        retires a mark, and it has to be recorded from the run that witnessed
        the test execute rather than inferred later from a map that cannot tell
        the two apart.
        """

        if skipped:
            self.skipped.add(nodeid)
        else:
            self.skipped.discard(nodeid)

    def observe(self, report) -> None:
        """Fold one pytest report in, if it says anything about the body running.

        The rule lives here rather than in the caller because it is the whole
        subtlety of this file and it has exactly one correct spelling. Three
        cases, and the two ends of the distinction are the point:

        * **Skipped in setup** — the body was never entered, so the record is
          about the gate and nothing else. Marked.
        * **Skipped during the call** — a runtime ``pytest.skip()`` stopped the
          body part-way, so the record is truncated at the skip. That is the
          same shape testmon already refuses to trust in a *failed* test, whose
          fingerprint is "the lines of a run that ended early", and it earns the
          same answer here. Marked.
        * **xfail** — reported as skipped with ``wasxfail``, but the body ran to
          completion and recorded everything it touched. Its record is ordinary
          evidence and must not be marked, or every xfail in the suite would
          carry a permanent mark and widen the merge gate forever.

        A setup *error* produces no call report and leaves any existing mark
        alone. That is right: its record is truncated too, and the red ledger
        already owns it.

        Reports that say nothing either way are ignored, so this is safe to call
        for every phase of every test.
        """

        if getattr(report, "wasxfail", None) is not None:
            return
        if report.when == "setup" and report.skipped:
            self.record(report.nodeid, skipped=True)
        elif report.when == "call":
            self.record(report.nodeid, skipped=bool(report.skipped))

    def forget_missing(self, collected: set[str]) -> None:
        """Drop entries for tests that no longer exist.

        A renamed or deleted test would otherwise sit here forever, permanently
        widening ``--since`` by a node id nothing can collect. An entry is only
        dropped when its *file* was collected, so a scoped run that never looked
        at ``tests/gpu_interaction`` cannot quietly forget the marks there —
        exactly the rule ``RedLedger.forget_missing`` uses, and for the same
        reason.
        """

        seen_files = {nodeid.split("::", 1)[0] for nodeid in collected}
        self.skipped = {
            nodeid
            for nodeid in self.skipped
            if nodeid in collected or nodeid.split("::", 1)[0] not in seen_files
        }

    def save(self) -> None:
        if self.skipped == self._initial:
            return
        if self.skipped:
            self._environments[self.environment] = sorted(self.skipped)
        else:
            self._environments.pop(self.environment, None)
        payload = {"version": _VERSION, "environments": self._environments}
        # A lost ledger costs a trusted record, never a wrong result: the tests
        # it would have widened the gate by are the ones the gate has always
        # been silently skipping, so failing to write it leaves the harness
        # exactly as trustworthy as it was before.
        with contextlib.suppress(OSError):
            self.path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _read(path: Path) -> dict[str, list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        return {}
    environments = payload.get("environments")
    if not isinstance(environments, dict):
        return {}
    return {
        str(name): [str(node) for node in nodes]
        for name, nodes in environments.items()
        if isinstance(nodes, list)
    }


#: How many test files a remainder names before it stops counting. Enough to
#: recognise which ring or which opt-in family is unmeasured, short enough that
#: the sentence stays a sentence.
REMAINDER_FILES_NAMED = 3


def remainder_by_file(unmeasured) -> list[tuple[str, int]]:
    """``[(test file, marks in it)]``, worst first. The shape every reader wants.

    Grouped by file because that is the unit somebody acts on: a remainder is
    almost always a whole opt-in family that one environment variable and one
    display would turn back on, and thirty node ids bury that.

    Lives here rather than in either caller because there are three places that
    withdraw the completeness claim — the run's closing sentence, the no-op fast
    path, and ``tools/test_selection.py`` — and a second copy of a rule in this
    file has already drifted once inside a single session (see
    :meth:`SkipLedger.observe`). One owner, and a tool that must not import a Qt
    conftest can still reach it.
    """

    files: dict[str, int] = {}
    for nodeid in unmeasured:
        name = nodeid.split("::", 1)[0]
        files[name] = files.get(name, 0) + 1
    return sorted(files.items(), key=lambda entry: (-entry[1], entry[0]))


def read_skips(map_path, environment: str) -> set[str]:
    """The recorded set, without loading a whole ledger. For read-only callers.

    Three of them, and none may write: the ``--since`` partition, which runs on
    every xdist worker; the closing summary on the controller; and
    ``tools/test_selection.py``.
    """

    return set(_read(skip_path(Path(map_path))).get(environment, ()))


def carry_skips(donor_map, destination_map) -> None:
    """Merge the donor map's skip marks into the destination map's.

    The marks belong to the map, not to the checkout: they are statements about
    node ids whose records the map holds, so a map that moves without them
    arrives able to vouch for records that were witnessed never running. A
    worktree is throwaway here, which makes that the normal case rather than the
    exotic one — see ``red_ledger.carry_flakes``, which this mirrors.

    **Union, not replace**, per environment. Both sides are records of things
    that actually happened, and neither side's absence of a mark is a positive
    statement that the test ran — only the *presence* of a mark carries
    information, so the union is the reading that loses nothing. The cost of
    being wrong in this direction is one cheap re-attempted skip.

    Failure is silent and costs marks, never a result: the map it rides with has
    already been copied by the time this runs.
    """

    theirs = _read(skip_path(Path(donor_map)))
    if not theirs:
        return
    destination_file = skip_path(Path(destination_map))
    ours = _read(destination_file)
    merged = {
        environment: sorted(set(theirs.get(environment) or ()) | set(ours.get(environment) or ()))
        for environment in set(theirs) | set(ours)
    }
    payload = {"version": _VERSION, "environments": merged}
    with contextlib.suppress(OSError):
        destination_file.write_text(json.dumps(payload, indent=1), encoding="utf-8")
