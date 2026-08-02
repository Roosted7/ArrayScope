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

**The clearing is itself an observation, and it used to be thrown away.** A gate
is meant to be a property of the environment, so within one environment a test
either skips every time or runs every time. One that does both is a test whose
gate turned it away for a reason the environment does not explain — and that is
the shape of the defect this ledger was extended for: ``make_backend_window``
skipped three different tests across eleven passes with a message the run had
already disproved, and because the mark cleared on the very next pass there was
nothing afterwards to say it had ever happened. The transition survives now, in
``intermittent``, with the conditions it happened under. Precisely the argument
:class:`tests.red_ledger.FlakeLedger` makes for a red that goes green, and the
same mark shape, so one reader learns both.

Two boundaries hold that apart from the mark above, and both are deliberate:

* **It changes no selection.** A test that ran recorded real dependencies, and
  those are evidence whatever its last run did — so ``--since`` must not widen
  by one node here. The mark above is the thing selection reads; this is the
  thing a person reads. Recorded and printed, never consulted.
* **It clears only when the test stops existing.** There is no event that makes
  an intermittent gate un-happen, so there is nothing else honest to clear on.

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
        payload = _payload(path)
        self._environments = _environments_of(payload)
        self._intermittent_environments = _intermittent_of(payload)
        self.skipped: set[str] = set(self._environments.get(environment, ()))
        self._initial = frozenset(self.skipped)
        self.intermittent: dict[str, dict] = dict(
            self._intermittent_environments.get(environment, {})
        )
        self._initial_intermittent = json.dumps(self.intermittent, sort_keys=True)

    @classmethod
    def load(cls, map_path, environment: str) -> SkipLedger:
        return cls(skip_path(Path(map_path)), environment)

    def record(self, nodeid: str, skipped: bool, *, when=None, context=None) -> None:
        """Fold one test's outcome in. Idempotent within a run.

        ``skipped`` false is as load-bearing as true: it is the one event that
        retires a mark, and it has to be recorded from the run that witnessed
        the test execute rather than inferred later from a map that cannot tell
        the two apart.

        Retiring a mark this run *found* is also the one moment the intermittent
        record can be written, so it is written here. ``when``/``context`` are
        the conditions, kept the way the flake ledger keeps them; without them
        the transition is still counted, because the count is the fact and the
        conditions are commentary on it.
        """

        if skipped:
            self.skipped.add(nodeid)
            return
        # Ran. Only a mark that was already in the file when this run opened it
        # is a transition: one this same run added and then cleared would be an
        # xfail or a re-attempt, not a gate that answered differently twice.
        if nodeid in self._initial and nodeid in self.skipped:
            previous = self.intermittent.get(nodeid) or {}
            self.intermittent[nodeid] = {
                "count": int(previous.get("count") or 0) + 1,
                "last_seen": when or "?",
                "recent": [{"at": when or "?", **(context or {})}, *_recent_of(previous)][
                    :_RECENT_LIMIT
                ],
            }
        self.skipped.discard(nodeid)

    def observe(self, report, *, when=None, context=None) -> None:
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
            self.record(
                report.nodeid,
                skipped=bool(report.skipped),
                when=when,
                context=context,
            )

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

        def survives(nodeid: str) -> bool:
            return nodeid in collected or nodeid.split("::", 1)[0] not in seen_files

        self.skipped = {nodeid for nodeid in self.skipped if survives(nodeid)}
        # The same rule, and it is the *only* clearing rule the intermittent
        # record has: nothing makes an intermittent gate un-happen, so a test
        # ceasing to exist is the one honest reason to forget that it did.
        self.intermittent = {
            nodeid: mark for nodeid, mark in self.intermittent.items() if survives(nodeid)
        }

    def save(self) -> None:
        intermittent_changed = json.dumps(self.intermittent, sort_keys=True) != (
            self._initial_intermittent
        )
        if self.skipped == self._initial and not intermittent_changed:
            return
        if self.skipped:
            self._environments[self.environment] = sorted(self.skipped)
        else:
            self._environments.pop(self.environment, None)
        if self.intermittent:
            self._intermittent_environments[self.environment] = self.intermittent
        else:
            self._intermittent_environments.pop(self.environment, None)
        payload = {"version": _VERSION, "environments": self._environments}
        # Additive, and only when there is something to say. An older reader
        # ignores the key; an older *file* without it reads as "none observed",
        # which is true of every run before this record existed.
        if self._intermittent_environments:
            payload["intermittent"] = self._intermittent_environments
        # A lost ledger costs a trusted record, never a wrong result: the tests
        # it would have widened the gate by are the ones the gate has always
        # been silently skipping, so failing to write it leaves the harness
        # exactly as trustworthy as it was before.
        with contextlib.suppress(OSError):
            self.path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


#: Observations kept with their conditions, per test, newest first — the flake
#: ledger's limit, for the same reason and so the two records read alike.
_RECENT_LIMIT = 5


def _payload(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        return {}
    return payload


def _environments_of(payload: dict) -> dict[str, list[str]]:
    environments = payload.get("environments")
    if not isinstance(environments, dict):
        return {}
    return {
        str(name): [str(node) for node in nodes]
        for name, nodes in environments.items()
        if isinstance(nodes, list)
    }


def _intermittent_of(payload: dict) -> dict[str, dict[str, dict]]:
    """Tolerant, like ``_recent_of``: a shape this code did not expect reads as
    "none observed" rather than raising. Commentary may never cost a result."""

    intermittent = payload.get("intermittent")
    if not isinstance(intermittent, dict):
        return {}
    return {
        str(name): {str(node): dict(mark) for node, mark in marks.items() if isinstance(mark, dict)}
        for name, marks in intermittent.items()
        if isinstance(marks, dict)
    }


def _recent_of(mark) -> list[dict]:
    recent = mark.get("recent") if isinstance(mark, dict) else None
    if not isinstance(recent, list):
        return []
    return [dict(entry) for entry in recent if isinstance(entry, dict)]


def _read(path: Path) -> dict[str, list[str]]:
    return _environments_of(_payload(path))


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


def read_intermittent(map_path, environment: str) -> dict[str, dict]:
    """The durable transitions, for the one reader that prints them.

    Separate from :func:`read_skips` because the two answer different
    questions and exactly one of them may reach selection. A caller that wanted
    both would be a caller about to widen the gate by a record that is
    deliberately not evidence — see this module's docstring.
    """

    return dict(_intermittent_of(_payload(skip_path(Path(map_path)))).get(environment, {}))


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

    donor_payload = _payload(skip_path(Path(donor_map)))
    theirs = _environments_of(donor_payload)
    their_intermittent = _intermittent_of(donor_payload)
    if not theirs and not their_intermittent:
        return
    destination_file = skip_path(Path(destination_map))
    our_payload = _payload(destination_file)
    ours = _environments_of(our_payload)
    merged = {
        environment: sorted(set(theirs.get(environment) or ()) | set(ours.get(environment) or ()))
        for environment in set(theirs) | set(ours)
    }
    payload = {"version": _VERSION, "environments": merged}
    # The transitions travel with the map for the same reason the marks do: they
    # are statements about node ids whose records the map holds. Merged by
    # highest count rather than unioned, because two checkouts' marks for one
    # test are observations of the same intermittent gate, not of two.
    merged_intermittent = _merge_intermittent(their_intermittent, _intermittent_of(our_payload))
    if merged_intermittent:
        payload["intermittent"] = merged_intermittent
    with contextlib.suppress(OSError):
        destination_file.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _merge_intermittent(theirs: dict, ours: dict) -> dict[str, dict[str, dict]]:
    merged: dict[str, dict[str, dict]] = {}
    for environment in set(theirs) | set(ours):
        their_marks = theirs.get(environment) or {}
        our_marks = ours.get(environment) or {}
        marks = {}
        for nodeid in set(their_marks) | set(our_marks):
            candidates = [mark for mark in (their_marks.get(nodeid), our_marks.get(nodeid)) if mark]
            marks[nodeid] = max(candidates, key=lambda mark: int(mark.get("count") or 0))
        if marks:
            merged[environment] = marks
    return merged
