#!/usr/bin/env python
"""Report what the change-selection map says, without running anything.

`pytest` already selects by default (see `tests/testmon_policy.py` and
`docs/testing/test-selection.md`). This tool answers the two questions running
the tests does not:

* **What is the blast radius of my working tree?** — `impact` names the tests
  a change reaches, grouped by directory, in about a quarter of a second. That
  is the number to put in a handoff, and the one that catches "this render fix
  touches thirteen `tests/ui` files".
* **Is the map trustworthy right now?** — `status` names what it does *not*
  cover: test files it has never recorded, and declared child-process entry
  points that have moved. A map with a hole in it selects confidently and
  wrongly, so the hole has to be visible.

Usage:
    python tools/test_selection.py                  # impact of the working tree
    python tools/test_selection.py impact --tests   # ... listing every node id
    python tools/test_selection.py impact --json    # ... as JSON
    python tools/test_selection.py status

Both read the map for one environment — offscreen by default, matching ring 1.
Pass `--environment` (or set the same variables the ring uses, e.g.
`ARRAYSCOPE_GPU_TESTS=1`) to read another.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# `python tools/test_selection.py` otherwise searches an editable install
# before this checkout, and would report on the wrong tree entirely.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import testmon_policy

_DEFAULT_QT_PLATFORM = "offscreen"


def _environment(explicit: str | None) -> str:
    if explicit:
        return explicit
    # tests/conftest.py pins this at import time for every ring-0..2 run; match
    # it here so the tool reads the slice of the map the suite writes.
    os.environ.setdefault("QT_QPA_PLATFORM", _DEFAULT_QT_PLATFORM)
    expression = _ini_value("environment_expression", "")
    return testmon_policy.resolve_environment(expression)


def _ini_value(key: str, default):
    import tomllib

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    return config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get(key, default)


def _read_map(environment: str, with_changed_files: bool):
    ignore = tuple(_ini_value("testmon_ignore_dependencies", []))
    peek = testmon_policy.peek(
        REPO_ROOT,
        environment,
        ignore,
        with_changed_files=with_changed_files,
    )
    if peek is None:
        raise SystemExit(
            f"No usable selection map at {testmon_policy.map_path(REPO_ROOT)}.\n"
            "Record one by running the suite once: pytest"
        )
    if peek.empty:
        raise SystemExit(
            f"The map holds no tests for environment {environment!r}.\n"
            "Either the suite has never run in this regime, or the environment "
            "key differs from the one the suite writes — check QT_QPA_PLATFORM "
            "and the ARRAYSCOPE_* ring variables."
        )
    return peek


def _unmapped_test_files(peek) -> set[str]:
    return {
        name
        for name in testmon_policy.collect_test_files(REPO_ROOT)
        if name not in peek.mapped_files
    }


def command_impact(args) -> int:
    environment = _environment(args.environment)
    peek = _read_map(environment, with_changed_files=True)
    per_file = Counter(name.split("::", 1)[0] for name in peek.affected_tests)
    unmapped = _unmapped_test_files(peek)

    if args.json:
        print(
            json.dumps(
                {
                    "environment": environment,
                    "mapped_tests": peek.mapped_tests,
                    "changed_files": sorted(peek.changed_files),
                    "affected_tests": sorted(peek.affected_tests),
                    "affected_files": dict(sorted(per_file.items())),
                    "affected_seconds": round(peek.affected_seconds, 1),
                    "unmapped_test_files": sorted(unmapped),
                },
                indent=2,
            )
        )
        return 0

    print(f"environment  {environment}")
    print(f"map          {testmon_policy.map_path(REPO_ROOT)}  ({peek.mapped_tests} tests)")
    print()
    if peek.changed_files:
        print(f"differs from the map ({len(peek.changed_files)}):")
        for name in sorted(peek.changed_files):
            print(f"    {name}")
        print()
    if not peek.affected_tests:
        print("affected: nothing — the recorded tests all still hold for this tree.")
    else:
        print(
            f"affected: {len(peek.affected_tests)} tests in {len(per_file)} files, "
            f"~{peek.affected_seconds:.0f} s recorded"
        )
        by_directory = Counter()
        for name, count in per_file.items():
            by_directory["/".join(name.split("/")[:2])] += count
        print()
        for directory, count in sorted(by_directory.items(), key=lambda item: -item[1]):
            print(f"    {directory:<28} {count:>5} tests")
        print()
        for name, count in sorted(per_file.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {name:<70} {count:>4}")
        if args.tests:
            print()
            for name in sorted(peek.affected_tests):
                print(f"    {name}")
    if unmapped:
        print()
        print(f"never recorded, so always run ({len(unmapped)}):")
        for name in sorted(unmapped):
            print(f"    {name}")
    print()
    print("run them:  pytest        (selection is the default)")
    return 0


def command_status(args) -> int:
    environment = _environment(args.environment)
    peek = _read_map(environment, with_changed_files=False)
    unmapped = _unmapped_test_files(peek)

    print(f"environment  {environment}")
    print(f"map          {testmon_policy.map_path(REPO_ROOT)}")
    print(f"recorded     {peek.mapped_tests} tests over {len(peek.mapped_files)} files")
    print(f"unmapped     {len(unmapped)} test files have never been recorded")
    for name in sorted(unmapped):
        print(f"    {name}")

    print()
    if peek.forced_tests:
        rerun = "re-run" if testmon_policy.rerun_known_red_tests() else "NOT re-run"
        print(
            f"known red ({len(peek.forced_tests)}, ~{peek.forced_seconds:.0f} s) — "
            f"failing at their last recorded run, dependencies unchanged, {rerun}:"
        )
        for name in sorted(peek.forced_tests):
            print(f"    {name}")
        print()
        print("    ARRAYSCOPE_TESTMON_RERUN_FAILING=1 pytest   re-runs them")
        print("    pytest --no-testmon                         runs everything")
        print()
    print()
    print("declared child-process dependencies (invisible to coverage):")
    missing = []
    for test_file, declared in sorted(testmon_policy.OUT_OF_PROCESS_DEPENDENCIES.items()):
        for entry_point in declared:
            exists = (REPO_ROOT / entry_point).exists()
            if not exists:
                missing.append((test_file, entry_point))
            print(f"    {'ok ' if exists else 'GONE'}  {test_file}  ->  {entry_point}")
    if missing:
        print()
        print("A declared entry point no longer exists. Its test is now selected by")
        print("nothing, which reads as 'unaffected' forever — fix the declaration in")
        print("tests/testmon_policy.py.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--environment", help="read another regime's slice of the map")
    subparsers = parser.add_subparsers(dest="command")

    impact = subparsers.add_parser("impact", help="tests this working tree affects")
    impact.add_argument("--tests", action="store_true", help="list every affected node id")
    impact.add_argument("--json", action="store_true", help="machine-readable output")
    impact.set_defaults(func=command_impact)

    status = subparsers.add_parser("status", help="what the map does not cover")
    status.set_defaults(func=command_status)

    arguments = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(arguments)
    if args.command is None:
        args = parser.parse_args([*arguments, "impact"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
