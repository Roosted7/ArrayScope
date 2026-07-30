"""Read executed-function coverage out of the selection map, for free.

The map already holds, per test, the AST blocks that test executed. Unioned
across every test it has recorded, it answers a question nobody was asking it:
**which functions in ``arrayscope/`` does the suite execute?** No coverage run,
no extra tracing, no wall clock — one SQL query plus the block structure of the
source tree.

It is legitimate on a *selected* run, which is the whole point: a test the map
deselected keeps its recorded fingerprint, and that fingerprint is still valid by
construction — the same invariant selection itself rests on. It self-heals, too:
when a file changes, every test that recorded it is affected, so it re-runs and
re-records in the same breath, and stale checksums stop matching the current tree
and drop out of the numerator.

What it is **not** is `coverage.py`'s line percentage, and the two must never be
quoted as if they were. Measured on identical execution data (``tests/core`` plus
``tests/kernel``, coverage.py's own ``executed_lines`` folded through testmon's
own ``create_fingerprint``):

    coverage.py lines   14.8%
    testmon    blocks   13.1%
    per file            median +0.0, but the spread runs -36.1 to +17.4 points

The aggregate agreement is two large biases cancelling, not accuracy. Per file
they diverge violently, in both directions, for one structural reason each:

* **Upward** — a block counts as covered when *any* line in it ran, so a
  function entered and abandoned at its first guard reads exactly like one run
  to completion.
* **Downward** — a module's top-level statements (imports, class bodies, every
  ``def``, every constant) are a *single* block spanning the whole file, so
  merely importing a module makes coverage.py call most of its statements
  covered while this counts one block.

The second is severe enough that the module block is excluded from the metric
entirely: it spans line one to EOF, so "the module block is covered" only ever
means "some test touched this file at all". That is reported separately, as
reach, and never mixed into the function figure.

So this measures **executed-function coverage**, a different metric from CI's
Codecov line figure, and a *floor* rather than an estimate — a test the map has
never recorded contributes nothing, so :attr:`Reach.recorded_tests` is the figure
that says how tight the floor is, and is printed with it.

Validated against a full ``pytest --cov`` run on the same tree, with a completely
recorded map: 5312 of 6204 functions from the map against 5271 ground truth,
**0.66 points apart**. Of the difference, 99 were functions the map knew about and
that day's coverage run skipped, and 58 were seen by coverage.py and invisible
here — mostly child processes, testmon's first blind spot.

Cost: one SQL query (~6 ms over 5.7k fingerprint rows) plus block structure for
the tree. Parsing all 295 files costs ~2.3 s, more than a small selected run
takes end to end, so the structure is cached beside the map keyed by each file's
git blob sha — the same content addressing testmon uses, so it is valid across
checkouts and invalidated by exactly the files that changed. Steady state is a
read-and-hash of the tree, ~20 ms.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

#: Sits next to the map, shares its lifetime, gitignored like it. Holds the parse
#: cache *and* this checkout's inherited baseline, so seeding a worktree's map
#: alongside this file makes the default comparison "since where I branched
#: from" with nothing to bootstrap. See ``testmon_policy.seed_map``.
COVERAGE_SUFFIX = "-coverage.json"

_VERSION = 1


def sidecar_path(map_path) -> Path:
    map_path = Path(map_path)
    return map_path.with_name(map_path.name + COVERAGE_SUFFIX)


def _digest(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


@dataclass(frozen=True)
class Function:
    """One function or method body, as the map identifies it."""

    path: str
    line: int
    name: str
    #: testmon's own block checksum — what the map stores, and therefore the only
    #: thing that can be looked up *in* it. It includes the block's index within
    #: the file, so inserting a function above this one changes it.
    checksum: int
    #: Digest of the block's representation with that index removed. Stable
    #: across edits elsewhere in the file, which is what comparing two
    #: revisions needs: otherwise one insertion at the top of a file reports
    #: every function below it as changed.
    body: str
    #: Occurrence index, to keep two byte-identical same-named bodies in one
    #: file from collapsing into a single entry and unbalancing the totals.
    occurrence: int = 0

    @property
    def identity(self) -> str:
        """Cross-run identity: this body, in this file, this many times over."""

        return f"{self.path}#{self.body}#{self.occurrence}"

    def __str__(self) -> str:
        return f"{self.path}:{self.line}  {self.name}"


@dataclass(frozen=True)
class Reach:
    """What the map says the suite executes, as of now."""

    environment: str
    covered: int
    total: int
    files_reached: int
    files_total: int
    #: Tests the map holds for this environment. The figure above is a floor of
    #: exactly this tightness: a test the map has never recorded contributes
    #: nothing, and there are two ordinary ways for that to happen — a ring
    #: never run here, and testmon's own housekeeping, which drops a map entry
    #: for an affected test that a *scoped* run did not collect.
    recorded_tests: int = 0
    #: ``{"arrayscope/render": (covered, total)}``, two path components deep.
    per_package: dict[str, tuple[int, int]] = field(default_factory=dict)
    uncovered: tuple[Function, ...] = ()
    #: Files no recorded test touched at all. Note the systematic false
    #: positive: a package ``__init__`` is imported at *collection* time, before
    #: any test's tracing starts, so it never appears under a test. 19 of this
    #: tree's 31 unreached files are that, not dead code.
    unreached_files: tuple[str, ...] = ()
    #: Files whose content still differs from the map when this was measured, so
    #: their functions read as uncovered until the tests using them re-run.
    #: Non-zero means the figure is provisional — normal mid-edit, and after a
    #: run scoped by path or ``-k``, which leaves deselected tests the map would
    #: otherwise have counted.
    provisional_files: int = 0
    #: Identities of the covered functions, for comparison against a baseline.
    identities: frozenset[str] = frozenset()

    @property
    def percent(self) -> float:
        return 100.0 * self.covered / self.total if self.total else 0.0


def _stripped(code: str) -> str:
    """A block's representation without testmon's per-file block index.

    ``Block.code`` is ``f"{index}:{ast_repr}"``. The index is what makes an
    inserted function invalidate every block below it — right for selection,
    where over-running is safe, and wrong for "what did this branch change",
    which would otherwise name a whole file after a one-line insertion.
    """

    _, separator, rest = code.partition(":")
    return rest if separator else code


def _module_of(rootdir, relative: str):
    from testmon.process_code import Module, read_source_sha

    source, fsha = read_source_sha(Path(rootdir) / relative)
    if source is None:
        return None, None
    return (
        Module(source_code=source, filename=relative, rootdir=str(rootdir), fs_fsha=fsha),
        fsha,
    )


def functions_of(rootdir, relative: str) -> tuple[list[Function], int | None]:
    """``(functions, module_block_checksum)`` for one file at its current revision.

    testmon appends a body's block *after* the bodies nested inside it, so the
    module block — which encloses everything — is always the last one. Taken
    positionally rather than by span: a file holding a single function starting
    at line one would tie on width.
    """

    module, _ = _module_of(rootdir, relative)
    if module is None:
        return [], None
    blocks = module.blocks
    if not blocks:
        return [], None
    checksums = module.method_checksums
    function_blocks, module_checksum = blocks[:-1], checksums[-1]

    functions: list[Function] = []
    seen: dict[str, int] = {}
    # Not strict: ``checksums`` carries the module block's too, and it is the one
    # entry deliberately left out of the function list.
    for block, checksum in zip(function_blocks, checksums, strict=False):
        body = _digest(_stripped(block.code))
        occurrence = seen.get(body, 0)
        seen[body] = occurrence + 1
        functions.append(
            Function(
                path=relative,
                line=block.start,
                name=block.name,
                checksum=checksum,
                body=body,
                occurrence=occurrence,
            )
        )
    return functions, module_checksum


def bodies_at(rootdir, relative: str, revision_source: bytes) -> set[str]:
    """Body digests of one file as it stood at another revision.

    Same digest as :attr:`Function.body`, so the two are directly comparable.
    An unparseable revision yields the empty set, which reads as "all of it is
    new" — the safe direction for a report about uncovered new code.
    """

    from testmon.process_code import Module

    try:
        module = Module(
            source_code=revision_source.decode("utf-8", "replace"),
            filename=relative,
            rootdir=str(rootdir),
        )
        return {_digest(_stripped(block.code)) for block in module.blocks}
    except Exception:
        return set()


def source_roots(rootdir) -> tuple[str, ...]:
    """Whatever ``[tool.coverage.run] source`` names, so the two agree.

    One denominator, declared once: if CI's coverage job measures a package,
    this measures the same package.
    """

    import tomllib

    try:
        with (Path(rootdir) / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, ValueError):
        return ("arrayscope",)
    source = config.get("tool", {}).get("coverage", {}).get("run", {}).get("source")
    if isinstance(source, list) and source:
        return tuple(str(entry) for entry in source)
    return ("arrayscope",)


def source_files(rootdir, roots: tuple[str, ...]) -> list[str]:
    rootdir = Path(rootdir)
    files: list[str] = []
    for root in roots:
        base = rootdir / root
        if base.is_file():
            files.append(base.relative_to(rootdir).as_posix())
        elif base.is_dir():
            files.extend(path.relative_to(rootdir).as_posix() for path in base.rglob("*.py"))
    return sorted(files)


def _read_only(map_path):
    """Our own read-only connection, or ``None``.

    Deliberately not testmon's: this runs at terminal-summary time, after xdist
    workers have written their fingerprints to the same file, and testmon's
    connection may still be holding an older snapshot.
    """

    map_path = Path(map_path)
    if not map_path.exists():
        return None
    try:
        # ``as_uri`` rather than an f-string: a checkout path containing ``?`` or
        # ``#`` would otherwise be parsed as query or fragment and read as a
        # missing database.
        connection = sqlite3.connect(f"{map_path.as_uri()}?mode=ro", uri=True)
    except (sqlite3.Error, ValueError):
        return None
    connection.row_factory = sqlite3.Row
    return connection


def read_map(map_path, environment: str) -> tuple[dict[str, set[int]], int] | None:
    """``({path: checksums some recorded test executed}, recorded test count)``.

    ``None`` if the map is missing or its schema has moved;
    ``tests/app/test_test_selection.py`` fails on the latter deliberately.
    """

    from testmon.process_code import blob_to_checksums

    connection = _read_only(map_path)
    if connection is None:
        return None
    try:
        rows = connection.execute(
            """
            SELECT f.filename, f.method_checksums
            FROM environment e, test_execution te, test_execution_file_fp tf, file_fp f
            WHERE e.environment_name = ?
              AND te.environment_id = e.id
              AND te.id = tf.test_execution_id
              AND tf.fingerprint_id = f.id
            """,
            (environment,),
        ).fetchall()
        tests = connection.execute(
            """
            SELECT count(*) FROM environment e, test_execution te
            WHERE e.environment_name = ? AND te.environment_id = e.id
            """,
            (environment,),
        ).fetchone()[0]
    except (sqlite3.Error, IndexError, TypeError):
        return None
    finally:
        with contextlib.suppress(sqlite3.Error):
            connection.close()

    covered: dict[str, set[int]] = {}
    for row in rows:
        blob = row["method_checksums"]
        if blob is not None:
            covered.setdefault(row["filename"], set()).update(blob_to_checksums(blob))
    return covered, int(tests)


class Structure:
    """Block structure for the source tree, cached by content.

    Parsing 295 files costs ~2.3 s, which is more than a small selected run
    takes end to end, so it cannot be paid every time. Keyed by git blob sha, so
    the cost after the first run is proportional to the diff — the same shape as
    everything else here.
    """

    def __init__(self, cached: dict | None = None) -> None:
        self._cached = cached if isinstance(cached, dict) else {}
        self._fresh: dict[str, dict] = {}
        self.parsed = 0

    def for_file(self, rootdir, relative: str) -> dict | None:
        from testmon.process_code import read_source_sha

        source, fsha = read_source_sha(Path(rootdir) / relative)
        if source is None:
            return None
        entry = self._cached.get(relative)
        if isinstance(entry, dict) and entry.get("fsha") == fsha:
            self._fresh[relative] = entry
            return entry
        functions, module_checksum = functions_of(rootdir, relative)
        self.parsed += 1
        entry = {
            "fsha": fsha,
            "module": module_checksum,
            "functions": [
                [function.checksum, function.line, function.name, function.body]
                for function in functions
            ],
        }
        self._fresh[relative] = entry
        return entry

    def payload(self) -> dict:
        """Only the files seen this time, so deletions expire the cache entry."""

        return self._fresh


def functions_in(entry: dict, relative: str) -> list[Function]:
    functions: list[Function] = []
    seen: dict[str, int] = {}
    for checksum, line, name, body in entry.get("functions", ()):
        occurrence = seen.get(body, 0)
        seen[body] = occurrence + 1
        functions.append(
            Function(
                path=relative,
                line=line,
                name=name,
                checksum=checksum,
                body=body,
                occurrence=occurrence,
            )
        )
    return functions


def measure(
    rootdir,
    map_path,
    environment: str,
    *,
    structure: Structure | None = None,
    provisional_files: int = 0,
) -> Reach | None:
    """Executed-function coverage for one environment. ``None`` if unreadable.

    Per environment, never unioned across them: the map is keyed by ring (see
    ``environment_expression`` in ``pyproject.toml``), and merging an offscreen
    recording with a compositor one would credit code to a ring that has never
    executed it. A ring never run here reads as zero, which is the truth.
    """

    read = read_map(map_path, environment)
    if read is None:
        return None
    covered, recorded = read
    structure = structure if structure is not None else Structure()

    total = hit = files_total = files_reached = 0
    per_package: dict[str, list[int]] = {}
    uncovered: list[Function] = []
    unreached: list[str] = []
    identities: set[str] = set()

    for relative in source_files(rootdir, source_roots(rootdir)):
        entry = structure.for_file(rootdir, relative)
        if entry is None:
            continue
        seen = covered.get(relative, set())
        files_total += 1
        covered_here = 0
        functions = functions_in(entry, relative)
        # Reach has to be checksum-matched too, not merely "the map has rows for
        # this path". A file whose every recorded fingerprint is from an older
        # revision would otherwise read as reached while nothing in it matches.
        if entry.get("module") in seen or any(function.checksum in seen for function in functions):
            files_reached += 1
        else:
            unreached.append(relative)
        for function in functions:
            if function.checksum in seen:
                covered_here += 1
                identities.add(function.identity)
            else:
                uncovered.append(function)
        total += len(functions)
        hit += covered_here
        bucket = per_package.setdefault("/".join(relative.split("/")[:2]), [0, 0])
        bucket[0] += covered_here
        bucket[1] += len(functions)

    return Reach(
        environment=environment,
        covered=hit,
        total=total,
        files_reached=files_reached,
        files_total=files_total,
        recorded_tests=recorded,
        per_package={name: (values[0], values[1]) for name, values in per_package.items()},
        uncovered=tuple(uncovered),
        unreached_files=tuple(unreached),
        provisional_files=provisional_files,
        identities=frozenset(identities),
    )


# --------------------------------------------------------------------------- #
# What the figure is compared against
# --------------------------------------------------------------------------- #
#
# The map answers "what does the suite execute *now*". A delta needs a second
# point, and there are only three honest candidates:
#
#   1. The branch point, for the functions this branch added or changed. Exact,
#      and the one that matters before merging -- see `new_code` below.
#   2. What this checkout inherited when its map arrived. Observed rather than
#      inferred, exactly like tests/red_ledger.py and for the same reason: after
#      a worktree seeds its map from `main`, "what did I add or lose" is the
#      question, and it has a recorded answer.
#   3. Nothing -- a fresh clone gets the absolute figure and no delta, because
#      inventing a comparison is worse than not having one.
#
# What is NOT a candidate: the whole tree as it stood at the branch point. For a
# file this branch edited, the map holds fingerprints of the *current* revision
# only, because the tests re-recorded when they re-ran, so the baseline
# revision's coverage is not recoverable from it. A global "percent since main"
# would silently read every edited file as newly uncovered. That measurement is
# refused rather than approximated.


def read_sidecar(map_path) -> dict:
    try:
        payload = json.loads(sidecar_path(map_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        return {}
    return payload


def write_sidecar(map_path, *, structure: dict, baselines: dict) -> None:
    payload = {"version": _VERSION, "structure": structure, "baselines": baselines}
    # A lost sidecar costs one re-parse and one baseline, never a wrong result.
    with contextlib.suppress(OSError, TypeError, ValueError):
        sidecar_path(map_path).write_text(json.dumps(payload), encoding="utf-8")


def cached_structure(payload: dict) -> dict:
    structure = payload.get("structure")
    return structure if isinstance(structure, dict) else {}


def baselines(payload: dict) -> dict:
    recorded = payload.get("baselines")
    return dict(recorded) if isinstance(recorded, dict) else {}


def baseline_from(reach: Reach) -> dict:
    return {
        "covered": reach.covered,
        "total": reach.total,
        "identities": sorted(reach.identities),
    }


@dataclass(frozen=True)
class Drift:
    """How this checkout's coverage moved since the baseline it inherited."""

    gained: tuple[str, ...]
    lost: tuple[str, ...]
    baseline_covered: int
    baseline_total: int

    @property
    def moved(self) -> bool:
        return bool(self.gained or self.lost)


def drift(reach: Reach, baseline: dict | None) -> Drift | None:
    if not baseline:
        return None
    was = frozenset(baseline.get("identities", ()))
    return Drift(
        gained=tuple(sorted(reach.identities - was)),
        lost=tuple(sorted(was - reach.identities)),
        baseline_covered=int(baseline.get("covered", 0)),
        baseline_total=int(baseline.get("total", 0)),
    )


@dataclass(frozen=True)
class NewCode:
    """Functions this branch added or changed, split by whether a test runs them."""

    ref: str
    covered: tuple[Function, ...]
    uncovered: tuple[Function, ...]

    @property
    def total(self) -> int:
        return len(self.covered) + len(self.uncovered)


def new_code(
    rootdir,
    map_path,
    environment: str,
    ref: str,
    merge_base: str,
    *,
    structure: Structure | None = None,
) -> NewCode | None:
    """What this branch added or changed, and whether the suite executes it.

    A percentage is the wrong answer here (see the note above); this is the
    computable one, and it is exact in the direction that matters: a function
    whose body this branch touched necessarily re-recorded in the run that just
    happened, so "no test executes it" cannot be an artefact of a stale map.

    Comparison is on the index-free body digest, so inserting a function at the
    top of a file does not report every function below it as changed — which the
    raw block checksum would, and which is why testmon over-selects there.
    """

    from tests import testmon_policy

    read = read_map(map_path, environment)
    if read is None:
        return None
    covered, _ = read
    structure = structure if structure is not None else Structure()
    roots = source_roots(rootdir)

    tracked = testmon_policy._git(rootdir, "diff", "--name-only", merge_base)
    untracked = testmon_policy._git(rootdir, "ls-files", "--others", "--exclude-standard")
    if tracked is None or untracked is None:
        return None
    paths = sorted(
        {
            candidate
            for line in (tracked + untracked).splitlines()
            if (candidate := line.strip()).endswith(".py")
            and any(candidate.startswith(f"{root}/") or candidate == root for root in roots)
        }
    )

    fresh: list[Function] = []
    for relative in paths:
        entry = structure.for_file(rootdir, relative)
        if entry is None:
            continue  # deleted in this tree; nothing of it left to cover
        source = testmon_policy._git(rootdir, "show", f"{merge_base}:{relative}", binary=True)
        before = bodies_at(rootdir, relative, source) if source is not None else set()
        fresh.extend(
            function for function in functions_in(entry, relative) if function.body not in before
        )

    executed, missing = [], []
    for function in fresh:
        seen = covered.get(function.path, set())
        (executed if function.checksum in seen else missing).append(function)
    return NewCode(ref=ref, covered=tuple(executed), uncovered=tuple(missing))
