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

    Deliberately minimal. No score, no decay, no threshold: one observation is
    enough to know the fingerprint does not determine the outcome, and a count
    that had to be tuned would be a second thing to get wrong. The mark clears
    on exactly one event — the test's dependencies actually changed, so it
    re-recorded. New code is new evidence, and the old flakiness is no longer
    about it.
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

    def observe(self, nodeid: str, when: str) -> None:
        """Record one red-that-was-not-changed passing. Idempotent-ish: it counts."""

        previous = self.marks.get(nodeid) or {}
        self.marks[nodeid] = {
            "count": int(previous.get("count") or 0) + 1,
            "last_seen": when,
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
    """``{node id: {count, last_seen}}``, without loading a whole ledger.

    The read-only half, for the two callers that must not write: the ``--since``
    partition, which runs on every xdist worker, and ``tools/test_selection.py``.
    """

    return _read_flakes(flake_path(Path(map_path))).get(environment, {})
