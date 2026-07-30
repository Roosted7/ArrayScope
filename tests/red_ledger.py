"""Remember which failing tests are this checkout's own doing.

Change selection stops re-running a test that failed last time and whose
dependencies have not moved since. Without something like this ledger that rule
has a hole big enough to drive the whole feature into:

    edit foo.py  ->  test T runs, fails, and the map records T as failed
    edit bar.py  ->  T's dependencies are unchanged *since the map was written*,
                     so T is skipped — two minutes after you broke it

The map cannot answer "was this red already there?", because the map is
rewritten by every run, including the run that introduced the red. So the answer
is observed instead of inferred: every time a test *transitions* into failing —
from passing, or from not existing — it is written down here, and a test in this
ledger is never skipped and is printed at the end of every run until it passes
again.

That makes the baseline exactly what it should be: whatever was already failing
when this checkout's map arrived (usually seeded from ``main``). Those are
inherited and stay quiet. Everything that broke afterwards is yours and stays
loud, with no git archaeology, no snapshot to maintain, and nothing to bootstrap
— an absent ledger means "nothing has broken here yet", which is the correct
reading of a checkout that has not run anything.

One deliberate asymmetry: a test that appears in the map for the first time
*already failing* counts as new only if the map was populated when the run
started. On the very first recording pass there is no such thing as a
regression, only an inventory.

The second ledger here — :class:`FlakeLedger` — writes down the opposite
observation, and for the same reason: it is a fact only a run can see, and every
run used to throw it away. See its docstring.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

#: Sits next to the map, shares its lifetime, gitignored like it.
LEDGER_SUFFIX = "-newreds.json"

#: The flake ledger's own file. Deliberately *not* another key in the file
#: above: the two records are written on different events and cleared on
#: different ones, and a single writer that had to merge them is one careless
#: rewrite away from dropping the field it did not think about — which is
#: exactly the defect ``cbf3f254`` fixed in the coverage sidecar. Two files, two
#: owners, no merge.
FLAKE_SUFFIX = "-flakes.json"

_VERSION = 1

#: Observations kept with their conditions, per test, newest first. Five is
#: enough to see a pattern or the absence of one, and small enough that the
#: ledger stays a file a human can read. The ``count`` above it is not capped:
#: how often is a different question from under what.
_RECENT_LIMIT = 5


def ledger_path(map_path: Path) -> Path:
    return map_path.with_name(map_path.name + LEDGER_SUFFIX)


class RedLedger:
    """The set of tests that have broken in this checkout, per environment."""

    def __init__(
        self,
        path: Path,
        environment: str,
        previous_outcomes: dict[str, bool],
        map_was_populated: bool,
    ) -> None:
        self.path = path
        self.environment = environment
        self._previous = previous_outcomes
        self._map_was_populated = map_was_populated
        self._environments = _read(path)
        self.new_reds: set[str] = set(self._environments.get(environment, ()))
        self._initial = frozenset(self.new_reds)

    @classmethod
    def load(cls, map_path, environment, previous_outcomes, map_was_populated) -> RedLedger:
        return cls(
            ledger_path(Path(map_path)),
            environment,
            previous_outcomes,
            map_was_populated,
        )

    def record(self, nodeid: str, failed: bool) -> None:
        """Fold one test's outcome in. Idempotent within a run."""

        if not failed:
            self.new_reds.discard(nodeid)
            return
        was = self._previous.get(nodeid)
        if was is False:
            self.new_reds.add(nodeid)  # it used to pass here
        elif was is None and self._map_was_populated:
            self.new_reds.add(nodeid)  # brand new, and born failing

    def forget_missing(self, collected: set[str]) -> None:
        """Drop entries for tests that no longer exist.

        A renamed or deleted test would otherwise sit here forever: it can never
        pass, so nothing removes it, and its file keeps being collected to run a
        test that is not there. ``collected`` is what the run actually found, and
        an entry is only dropped when its *file* was collected — so a scoped run
        that never looked at ``tests/ui`` cannot quietly forget a red there.
        """

        seen_files = {nodeid.split("::", 1)[0] for nodeid in collected}
        self.new_reds = {
            nodeid
            for nodeid in self.new_reds
            if nodeid in collected or nodeid.split("::", 1)[0] not in seen_files
        }

    def save(self) -> None:
        if self.new_reds == self._initial:
            return
        self._environments[self.environment] = sorted(self.new_reds)
        payload = {"version": _VERSION, "environments": self._environments}
        # A lost ledger costs a re-run, never a wrong result.
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
    return environments if isinstance(environments, dict) else {}


def read_new_reds(map_path, environment: str) -> set[str]:
    """The recorded set, without loading a whole ledger. For read-only callers."""

    return set(_read(ledger_path(Path(map_path))).get(environment, ()))


def flake_path(map_path: Path) -> Path:
    return map_path.with_name(map_path.name + FLAKE_SUFFIX)


class FlakeLedger:
    """Tests observed to flip from red to green with no code change at all.

    The map is an evidence ledger: a recorded green is trusted for as long as
    the test's fingerprint holds, because content-addressing means an unchanged
    fingerprint is an unchanged *body of executed code*. That inference has one
    counter-example, and the suite produces it regularly — a test whose outcome
    depends on something the fingerprint does not describe (a timer, a worker
    pool, the order two threads happened to reach a queue).

    Such a test is observable, and every run used to throw the observation away.
    A red that re-ran **only because it was red** — its dependencies unchanged,
    so it was in the forced set rather than the affected set — and then *passed*
    has, by construction, flipped its outcome with no code change. That is the
    definition of a flake, witnessed rather than guessed, and it costs nothing
    to notice because the run had already paid to execute the test.

    What the mark buys: ``--since`` never counts that test's recorded green as
    evidence. It re-runs it instead, which is the one honest thing to do with a
    result the map cannot vouch for.

    Deliberately minimal *as a decision*. No score, no decay, no threshold: one
    observation is enough to know the fingerprint does not determine the
    outcome, and a count that had to be tuned would be a second thing to get
    wrong. The mark clears on exactly one event — the test's dependencies
    actually changed, so it re-recorded. New code is new evidence, and the old
    flakiness is no longer about it.

    Not minimal as a *record*. Each observation keeps what the run looked like
    (:meth:`observe`), because the mark answers "this test flipped" and the
    first question after it is always "under what?" — a loaded machine, eight
    workers, a ``--since`` — and that is exactly what nothing was writing down.
    The two halves are separate on purpose: the conditions are printed, never
    consulted. A harness that discounted an observation because the machine was
    busy would be a harness that hides the flakes worth finding, since load is
    a trigger for the real ones as often as it is an alibi.
    """

    def __init__(self, path: Path, environment: str) -> None:
        self.path = path
        self.environment = environment
        self._environments = _read_flakes(path)
        self.marks: dict[str, dict] = dict(self._environments.get(environment, {}))
        self._initial = json.dumps(self.marks, sort_keys=True)

    @classmethod
    def load(cls, map_path, environment: str) -> FlakeLedger:
        return cls(flake_path(Path(map_path)), environment)

    def observe(self, nodeid: str, when: str, context: dict | None = None) -> None:
        """Record one red-that-was-not-changed passing. Idempotent-ish: it counts.

        ``context`` describes the run this happened on — machine load, worker
        count, run shape — and is kept per observation under ``recent``, newest
        first, capped at :data:`_RECENT_LIMIT`. ``count`` and ``last_seen`` keep
        their exact previous meaning and position, so every existing reader of
        this file is untouched by the addition; ``recent`` is additive and its
        absence in an older file reads as "no conditions were recorded", which
        is true.

        The keys inside ``context`` are the caller's business. This class stores
        and prints them and does nothing else with them — see the class
        docstring for why that boundary is the point rather than a limitation.
        """

        previous = self.marks.get(nodeid) or {}
        self.marks[nodeid] = {
            "count": int(previous.get("count") or 0) + 1,
            "last_seen": when,
            "recent": [{"at": when, **(context or {})}, *_recent_of(previous)][:_RECENT_LIMIT],
        }

    def forget(self, nodeids) -> None:
        """Drop marks that have stopped being evidence about anything.

        Two reasons, both meaning "this mark is no longer about the test in
        front of us": the test's dependencies changed, so it re-recorded against
        code the flake was never observed on; or the test is gone from the map
        entirely, and a mark nothing can ever clear would force a re-run of a
        node id that does not exist.
        """

        for nodeid in nodeids:
            self.marks.pop(nodeid, None)

    def save(self) -> None:
        if json.dumps(self.marks, sort_keys=True) == self._initial:
            return
        if self.marks:
            self._environments[self.environment] = self.marks
        else:
            self._environments.pop(self.environment, None)
        payload = {"version": _VERSION, "environments": self._environments}
        # A lost ledger costs a trusted green, never a wrong result.
        with contextlib.suppress(OSError):
            self.path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _recent_of(mark) -> list[dict]:
    """The per-observation history of one mark, or none. Tolerant by design.

    A ledger written before ``recent`` existed, or one hand-edited into a shape
    this code did not expect, must read as "no conditions recorded" rather than
    raise: the conditions are commentary on the mark, and losing the commentary
    may never cost the mark.
    """

    recent = mark.get("recent") if isinstance(mark, dict) else None
    if not isinstance(recent, list):
        return []
    return [dict(entry) for entry in recent if isinstance(entry, dict)]


def carry_flakes(donor_map, destination_map) -> None:
    """Merge the donor map's flake observations into the destination map's.

    The observations belong to the map, not to the checkout: they are statements
    about node ids whose fingerprints the map holds, so a map that moves without
    them arrives claiming it can vouch for greens that were witnessed flipping.
    Worktrees are throwaway here, which made that the normal case — every
    observation a branch made died with it, and the main checkout kept
    re-learning the same flakes.

    **Union, not replace.** Both sides are records of things that actually
    happened, and a plain copy would silently discard whichever set of them the
    destination had made. So per environment and per node id: counts add, the
    newer ``last_seen`` wins, and the two histories interleave by timestamp back
    to :data:`_RECENT_LIMIT`. The one thing this cannot do is recognise an
    observation it is being handed *back* — a mark seeded from main, observed
    once more in the worktree, and adopted home counts as both. So the count is
    "how often this kept happening", not an audited total, and nothing here
    branches on its value.

    Failure is silent and costs observations, never a result: the map it rides
    with has already been copied by the time this runs, and a ledger that fails
    to arrive leaves the destination exactly as trustworthy as it was before.
    """

    donor_file = flake_path(Path(donor_map))
    destination_file = flake_path(Path(destination_map))
    theirs = _read_flakes(donor_file)
    if not theirs:
        return
    merged = _merge_environments(theirs, _read_flakes(destination_file))
    payload = {"version": _VERSION, "environments": merged}
    with contextlib.suppress(OSError):
        destination_file.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _merge_environments(theirs: dict, ours: dict) -> dict:
    return {
        environment: _merge_marks(theirs.get(environment) or {}, ours.get(environment) or {})
        for environment in set(theirs) | set(ours)
    }


def _merge_marks(theirs: dict, ours: dict) -> dict:
    merged = {}
    for nodeid in set(theirs) | set(ours):
        mine, yours = ours.get(nodeid) or {}, theirs.get(nodeid) or {}
        if not mine or not yours:
            merged[nodeid] = dict(mine or yours)
            continue
        merged[nodeid] = {
            "count": int(mine.get("count") or 0) + int(yours.get("count") or 0),
            "last_seen": max(str(mine.get("last_seen") or ""), str(yours.get("last_seen") or "")),
            "recent": sorted(
                _recent_of(mine) + _recent_of(yours),
                key=lambda entry: str(entry.get("at") or ""),
                reverse=True,
            )[:_RECENT_LIMIT],
        }
    return merged


def _read_flakes(path: Path) -> dict[str, dict[str, dict]]:
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
        str(name): {str(node): dict(mark) for node, mark in marks.items() if isinstance(mark, dict)}
        for name, marks in environments.items()
        if isinstance(marks, dict)
    }


def read_flakes(map_path, environment: str) -> dict[str, dict]:
    """``{node id: {count, last_seen, recent}}``, without loading a whole ledger.

    The read-only half, for the two callers that must not write: the ``--since``
    partition, which runs on every xdist worker, and ``tools/test_selection.py``.
    """

    return _read_flakes(flake_path(Path(map_path))).get(environment, {})
