"""Report production code that no shipped code path reaches.

    python tools/dead_code.py            # refuse the tree if anything is dead
    python tools/dead_code.py --list     # ... and name what is already excused

This is the gate, run from ``.githooks/pre-commit`` alongside ruff, for the
same reason ruff is there: it asks a whole-tree question that only has an
answer once the tree is written, and CI is not a gate in this repository.

Two classes are refused. **Unreachable** code nothing names at all, and
**test-only** code whose only callers are under ``tests/`` -- the class a
coverage number cannot see, because the function still executes and so still
reads as covered. The rules, the allowlist, and the unadjudicated backlog live
in :mod:`tests.dead_code`; this file is only the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import dead_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list",
        action="store_true",
        help="also print the excused definitions: allowlist and pending backlog",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(f"allowlisted ({len(dead_code._ALLOWLIST)}) — real entry points:")
        for path, name, reason in dead_code._ALLOWLIST:
            print(f"  {path}::{name}\n      {reason}")
        pending = dead_code._PENDING_ADJUDICATION
        print(
            f"\npending adjudication ({len(pending)}/{dead_code._PENDING_CEILING}) — "
            "queue row 12, delete or restore a caller:"
        )
        for path, name in pending:
            print(f"  {path}::{name}")
        print()

    problems = dead_code.problems()
    if not problems:
        print("dead code: none — every definition has a production caller or a stated reason.")
        return 0
    print("\n".join(problems), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
