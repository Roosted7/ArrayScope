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
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

#: Sits next to the map, shares its lifetime, gitignored like it.
LEDGER_SUFFIX = "-newreds.json"

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
