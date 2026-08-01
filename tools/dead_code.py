"""Report production code that no shipped code path reaches.

    python tools/dead_code.py            # refuse the tree if anything is dead

This is the gate, run from ``.githooks/pre-commit`` alongside ruff, for the
same reason ruff is there: it asks a whole-tree question that only has an
answer once the tree is written, and CI is not a gate in this repository.

Three classes are refused. **Unreachable** code nothing names at all,
**test-only** code whose only callers are under ``tests/`` -- the class a
coverage number cannot see, because the function still executes and so still
reads as covered -- and **dead test helpers**, the same question asked of the
suite's own support modules, so that moving a seam into ``tests/`` cannot
launder it. The first two are asked of module-level definitions and of methods
of classes no framework can dispatch into. The rules live in
:mod:`tests.dead_code`; this file is only the command line.

There is no allowlist and no backlog. A definition the product cannot reach is
either the product's, or the suite's, or nobody's, and the fix is to move it or
delete it rather than to excuse it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import dead_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    problems = dead_code.problems()
    if not problems:
        print("dead code: none — every definition has a caller on the side that owns it.")
        return 0
    print("\n".join(problems), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
