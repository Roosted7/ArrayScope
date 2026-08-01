"""Report fixed pauses and clock-backed verdicts that the suite must not gain.

    python tools/timing_guard.py            # refuse the tree if either rule fires

This is the gate, run from ``.githooks/pre-commit`` alongside ruff and the
dead-code scan, and for the same reason: it asks a whole-tree question that only
has an answer once the tree is written, and CI is not a gate in this repository.

It is also the reason the guard is *not* a pytest test. A commit that adds
``qtbot.wait(300)`` to a test file is selected by testmon as that file, never as
the guard, so pytest would meet the pause on some later full run rather than at
the moment somebody wrote it -- which is the whole point of having the guard.
Here it is refused as it is written, and the ~2 s comes off every suite run and
is paid once per commit. ``tests/app/test_timing_guard.py`` keeps only the
synthetic proof that these rules can fail.

Two rules, in :mod:`tests.timing_guard`; this file is only the command line.
**A clock reading may not reach a verdict** -- no ``assert`` on a duration, and
no ``if elapsed > ...: pytest.fail()`` either -- with no allowance, because the
tree has needed none since ``ba53fbc8``. **A fixed pause must read its duration
from the throttle that bounds it** rather than having it typed, with the
survivors carried as a counted backlog that only ever goes down.

There is no allowlist and nowhere to put one. A pause is either bounded by
something the product publishes, or it is a wait on the product's own progress
(``tests/settle.py``), or it is the thing this refuses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import timing_guard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    refusals = timing_guard.problems()
    if not refusals:
        print(
            "timing guard: none — no test adjudicates a clock, and every fixed "
            f"pause is accounted for ({timing_guard.UNJUSTIFIED_PAUSE_BACKLOG} "
            "still name no throttle)."
        )
        return 0
    print("\n".join(refusals), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
