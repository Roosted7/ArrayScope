"""Change-driven test selection policy (pytest-testmon).

Why the suite selects by default
--------------------------------
The recurring defect is not "the suite was red and we shipped it". It is *we
ran the directory the change lived in, and the regression surfaced in a
directory nobody thought to run* — a ``kernel/`` edit breaking a ``tests/ui``
scheduling assertion, a ``window/`` edit breaking a ``tests/display`` identity
assertion. Directory intuition is the thing that fails, and it fails silently.

pytest-testmon records, per test, the *executed* lines of every source file it
touched, so it answers the question from evidence instead of intuition. Reverting
``987d6bdc`` (a 22-line change to ``arrayscope/window/frame_effects.py``, shipped
with one new ``tests/window`` file) selects 20 tests across 16 files — 13 of them
in ``tests/ui``, one in ``tests/display``. Selection is therefore on by default:
it makes the honest set of affected tests the *cheap* thing to run, so nobody has
to choose between a 4-minute suite and a guess.

What this module owns
---------------------
* :func:`decide` — whether selection is active for one pytest run, and why.
* :func:`peek` — one cheap read of the recorded map *before* xdist sizes its
  worker pool, so a two-file selection does not pay to boot eight Qt workers.
* :data:`OUT_OF_PROCESS_DEPENDENCIES` and
  :func:`inject_out_of_process_dependencies` — the one dependency class testmon
  cannot observe, declared and fed back into its own map.

What it deliberately does not own: the dependency map, or the decision of which
tests a change affects. That is testmon's; this module calls
``TestmonData.determine_stable`` rather than re-deriving it.

Blind spots — stated, not papered over
--------------------------------------
testmon traces Python lines executed **in the pytest process**. So:

1. **Child processes.** A test that shells out gets nothing from the child.
   The tests that really do spawn one are declared in
   :data:`OUT_OF_PROCESS_DEPENDENCIES`, which puts the entry point back in the
   map; the modules that entry point *imports* remain invisible, and CI's
   exhaustive sweep is what covers them.
2. **Non-Python inputs.** Icons and fixture arrays have no fingerprint.
3. **Real rendering.** Rings 3–4 (``docs/testing/README.md``) prove things
   about pixels a compositor drew. Selection has nothing to say there and the
   ring rules are unchanged.

``pytest --since`` is the pre-merge command: it asks what the whole *branch*
changed rather than what moved since the last run, and still selects through
the map. Blind spots 1 and 2 are swept exhaustively by CI on every push;
blind spot 3 has no offscreen answer at all and belongs to rings 3–4.
``pytest --no-testmon`` stays available for the narrow cases — regenerating
the canonical artifacts, or settling a run this map is suspected of getting
wrong — and is not a routine sweep.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Recorded map, one per checkout (worktrees get their own). Gitignored.
MAP_FILENAME = ".testmondata"

#: Tests whose real work happens in a child process, mapped to the entry points
#: that child runs. Coverage never sees those files, so they are re-attached to
#: the test's recorded dependencies by hand (whole-file granularity) — which
#: means these tests are selected *exactly* when their entry point changes, and
#: skipped otherwise. That precision is the point: the set below costs ~46 s to
#: run, far too much to attach to every iteration.
#:
#: Only genuine spawns belong here. A test that monkeypatches ``subprocess.run``
#: (``test_qt_platform``, ``test_free_threading``, ``test_headless_display``,
#: ``test_profile_montage_workflow``, ``test_bart_pack``,
#: ``test_execution_environments``) runs entirely in-process and is already
#: traced; adding it would only slow the loop down.
#: ``tests/app/test_test_selection.py`` fails when a new real spawn appears
#: without an entry here.
OUT_OF_PROCESS_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    # `python tools/demo_recorder.py --smoke ...`
    "tests/app/test_demo_recorder_smoke.py": ("tools/demo_recorder.py",),
    # `python -c "import arrayscope, pyqtgraph.Qt as Qt"` — the binding
    # preference has to hold on a *fresh* interpreter, which is the whole point
    # of the test and the reason it cannot be traced.
    "tests/app/test_qt_binding.py": (
        "arrayscope/__init__.py",
        "arrayscope/app/qt_binding.py",
    ),
    # `python -c "import arrayscope.io.numpy_save"` (asserts pyqtgraph stays out
    # of sys.modules).
    "tests/io/test_numpy_save.py": ("arrayscope/io/numpy_save.py",),
    # `python -m arrayscope.tools.profile_montage_workflow ...` (ring 3).
    "tests/stress/test_synthetic_stress_matrix.py": (
        "arrayscope/tools/profile_montage_workflow.py",
    ),
    # One opt-in test in this file runs the profiler under py-spy
    # (ARRAYSCOPE_RUN_PY_SPY_SMOKE=1); every other subprocess here is
    # monkeypatched and traced normally.
    "tests/app/test_profile_montage_workflow.py": ("arrayscope/tools/profile_montage_workflow.py",),
    # Spawns, but the child is a client script generated from a template inside
    # this same test file, and the compositor launcher it drives runs
    # in-process. Nothing to re-attach — declared so the guard stays honest
    # rather than silent.
    "tests/gpu_interaction/test_headless_exact_window.py": (),
    # `python experiments/wgpu_gate_b/probe_native_wayland.py` (ring 4).
    "tests/gpu_interaction/test_wgpu_native_wayland_pin.py": (
        "experiments/wgpu_gate_b/probe_native_wayland.py",
    ),
}

#: Command-line flags that mean "the developer already chose"; the default
#: policy never overrides them.
_EXPLICIT_OPTIONS = (
    "testmon",
    "testmon_noselect",
    "testmon_nocollect",
    "testmon_forceselect",
    "tmnet",
)

_whole_file_line_cache: dict[str, frozenset[int]] = {}


@dataclass(frozen=True)
class Decision:
    """Whether selection runs for one pytest invocation, and why."""

    active: bool
    reason: str
    #: True when the developer asked for it themselves; the default policy then
    #: leaves every testmon option exactly as given.
    explicit: bool = False


def testmon_installed() -> bool:
    from importlib.util import find_spec

    return find_spec("testmon") is not None


def decide(config) -> Decision:
    """Resolve the selection policy for ``config``.

    Ordered so that an explicit request always wins over the default, and the
    two runs that must execute every test — a coverage run and CI — can never
    be truncated by a stale map.
    """

    if not testmon_installed():
        return Decision(False, "pytest-testmon is not installed")

    options = vars(config.option)
    if options.get("no-testmon"):
        return Decision(False, "--no-testmon")
    if any(options.get(name) for name in _EXPLICIT_OPTIONS):
        return Decision(True, "requested on the command line", explicit=True)

    override = os.environ.get("ARRAYSCOPE_TESTMON", "").strip().lower()
    if override in {"0", "off", "no", "false"}:
        return Decision(False, "ARRAYSCOPE_TESTMON is off")
    if override in {"1", "on", "yes", "true"}:
        # Not ``explicit``: that flag means "the developer set the testmon
        # options themselves, leave them alone". This override still needs the
        # options configured for it — it only overrules the refusals below.
        return Decision(True, "ARRAYSCOPE_TESTMON is on")

    # A coverage report describes the tests that ran. Selecting a subset and
    # publishing its coverage as the project's would be a false number, so the
    # two are mutually exclusive rather than merely discouraged.
    if options.get("cov_source") or options.get("cov_report"):
        return Decision(False, "coverage must execute every test")
    if os.environ.get("CI"):
        return Decision(False, "CI runs the whole suite")

    return Decision(True, "default")


def selects(config) -> bool:
    """Whether this run will actually deselect anything.

    Mirrors testmon's own rule that an explicit selector wins: with ``-k``,
    ``-m``, ``--lf`` or a ``file.py::test`` argument it keeps recording but runs
    what you asked for. Worker sizing has to know the difference, because those
    runs are not narrowed by the map.
    """

    options = vars(config.option)
    if options.get("testmon_forceselect"):
        return True
    if options.get("testmon_noselect"):
        return False
    if options.get("keyword") or options.get("markexpr") or options.get("lf"):
        return False
    return not any(".py::" in str(argument) for argument in config.args)


@dataclass(frozen=True)
class MapPeek:
    """What the recorded map says about the current working tree."""

    environment: str
    mapped_tests: int
    mapped_files: frozenset[str]
    affected_tests: frozenset[str]
    affected_files: frozenset[str]
    #: Recorded wall time of the affected tests, in seconds. Tests with no
    #: recorded duration count as zero, so this is a lower bound.
    affected_seconds: float
    #: Tests whose last recorded outcome was a failure and whose dependencies
    #: did *not* change. testmon re-runs these unconditionally; whether this
    #: repository does is :func:`rerun_known_red_tests`.
    forced_tests: frozenset[str] = frozenset()
    forced_files: frozenset[str] = frozenset()
    forced_seconds: float = 0.0
    #: Files whose content differs from what the map recorded. Only filled in
    #: when ``with_changed_files`` was asked for; note this is "different from
    #: the map", not "different from git HEAD" — an old map makes an untouched
    #: file changed, which is exactly the truth selection acts on.
    changed_files: frozenset[str] = frozenset()

    @property
    def empty(self) -> bool:
        return self.mapped_tests == 0


def map_path(rootdir: os.PathLike[str] | str) -> Path:
    return Path(rootdir) / os.environ.get("TESTMON_DATAFILE", MAP_FILENAME)


def rerun_known_red_tests(config=None) -> bool:
    """Whether *every* previously-failing test re-runs, changed or not.

    testmon's own answer is always yes: a failed test stops at the failure, so
    the lines it recorded are the lines of a run that ended early, and treating
    that truncated set as its full dependencies would be unsound.

    ArrayScope's answer is "only the ones that broke here" — see
    ``tests/red_ledger.py``. Re-running *all* of them measured **124 s on an
    otherwise clean tree**, and an inner loop that costs two minutes to
    re-confirm what everybody already knows is an inner loop nobody uses. A red
    that was already failing when this checkout's map arrived stays quiet; a red
    that this checkout broke is never skipped, because "its dependencies have
    not moved since the map was written" is worthless evidence when the map was
    written by the run that broke it.

    ``--rerun-reds`` re-runs the inherited ones as well — the right choice while
    you are fixing one of them, since the fix can land outside the truncated
    dependency set of the run that failed. It is a flag rather than only an
    environment variable because that is the case where somebody who cannot find
    it reaches for ``--no-testmon`` instead and trades a few-second loop for the
    whole suite. ``ARRAYSCOPE_TESTMON_RERUN_FAILING=1`` still works and is the
    form to use from a script or a CI step, and it is what an xdist worker reads
    when it is handed no command line of its own.
    """

    if config is not None:
        try:
            if bool(config.getoption("rerun_reds", False)):
                return True
        except (AttributeError, ValueError):
            # Not a pytest run, or the option was never registered — the
            # environment variable is the answer for both.
            pass
    return os.environ.get("ARRAYSCOPE_TESTMON_RERUN_FAILING", "").strip().lower() in {
        "1",
        "on",
        "yes",
        "true",
    }


def known_red_tests(config) -> set[str]:
    """Mapped tests that failed last time and whose dependencies are unchanged.

    Derived from the map rather than from what any process deselected, so the
    controller and a serial run report the same number. This is the *candidate*
    set; the ledger in ``tests/red_ledger.py`` decides which of them are
    inherited and may actually be skipped.
    """

    data = getattr(config, "testmon_data", None)
    if data is None:
        return set()
    try:
        all_tests = data.all_tests
        return {
            name
            for name in (data.stable_test_names or ())
            if (all_tests.get(name) or {}).get("failed")
        }
    except Exception:  # a report line is not worth failing a run over
        return set()


def apply_known_red_policy(config, new_reds: set[str]) -> int:
    """Stop re-running the *inherited* reds. Returns how many were dropped.

    testmon decides its deselection lists once, in ``TestmonSelect.__init__``,
    exempting every previously-failing test from deselection. Most of that
    exemption is worth keeping and one third of it is not:

    * A red whose dependencies changed is affected like anything else and runs.
    * A red this checkout broke (``new_reds``) runs whatever the map says. The
      map cannot vouch for it — the map was written by the run that broke it.
    * What is left failed on code that was already failing when this checkout's
      map arrived. Re-running it says nothing new and measured 124 s.

    Runs before collection, so both the file-level ignore and the item-level
    deselection see it.

    Does nothing when the run is not deselecting. Under ``--testmon-noselect``
    the lists only decide *ordering* (testmon runs the deselected group last),
    so touching them there would reshuffle an exhaustive run for no benefit —
    and this suite has enough latent order sensitivity that a reshuffle costs
    about one spurious failure per run.
    """

    if rerun_known_red_tests(config):
        return 0
    testmon_config = getattr(config, "testmon_config", None)
    if testmon_config is None or not testmon_config.select:
        return 0
    plugin = config.pluginmanager.get_plugin("TestmonSelect")
    data = getattr(config, "testmon_data", None)
    if plugin is None or data is None:
        return 0

    inherited = known_red_tests(config) - set(new_reds)
    if not inherited:
        return 0
    keep_collecting = {red.split("::", 1)[0] for red in new_reds}
    try:
        # Sets, not the lists testmon builds: both are membership-tested once
        # per collected item, which is 3500 linear scans over 3500 names.
        plugin.deselected_tests = set(plugin.deselected_tests) | inherited
        # A file only becomes skippable wholesale once it holds no red this
        # checkout owns; otherwise its collection has to happen so that red
        # can run.
        plugin.deselected_files = {
            name
            for name in (set(data.stable_files or ()) | set(plugin.deselected_files))
            if name not in keep_collecting
        }
    except (AttributeError, TypeError):  # a testmon upgrade renamed them
        return 0
    return len(inherited)


def protect_map_outside_the_scope(config) -> int:
    """Stop a scoped run from deleting the map entries it did not look at.

    testmon garbage-collects the map on every run by deleting each recorded test
    that the run neither collected nor called unaffected
    (``TestmonData.sync_db_fs_tests``). That is right for a whole-suite run,
    where "not collected" really does mean "gone". For a scoped one it is a
    silent, permanent hole, and it hits exactly the test you most need:

        edit arrayscope/display/scene.py   -> affects one test, in tests/ui
        pytest tests/render                -> that test is affected, was not
                                              collected, and is deleted
        pytest                             -> "affected: nothing", 0 tests run

    The last step is the damage. Its map entry is gone, so nothing marks it
    affected; and testmon's file-level ``pytest_ignore_collect`` skips its whole
    file because every *remaining* test in it is unaffected, so collection never
    rediscovers it and ``sync_db_fs_tests`` never re-adds it. The edit is then
    untested and reported as fully tested — by ``pytest``, by ``pytest --since``,
    and by ``tools/test_selection.py``, all three. Only ``--no-testmon`` catches
    it. Measured, on a complete map: 3941 recorded tests, ``pytest tests/render``
    takes it to 3940, and the working tree's one affected test evaporates.

    So the deletion is narrowed to the scope the run actually examined: every
    mapped test whose file lies outside the run's path arguments is retained
    regardless. In-scope cleanup still happens, which is the part that has a
    legitimate job to do (a renamed or deleted test), and it is still exact —
    that scope *was* collected.

    A run that collected the whole suite is left alone: testmon's own bookkeeping
    is correct there and this must not second-guess it. That includes the ordinary
    bare ``pytest``, whose ``testpaths`` argument *is* a path argument — see
    :func:`collected_the_whole_suite`. A node-id run counts as narrow, because it
    is: ``pytest file.py::test`` collects that file and nothing else, and it is
    the sharpest form of the bug rather than an exception to it.

    Returns the number of mapped tests it will protect, for the report line.
    """

    data = getattr(config, "testmon_data", None)
    if data is None or getattr(data, "_arrayscope_scope_guard", False):
        return 0
    paths = collected_scope(config)
    if collected_the_whole_suite(config):
        return 0

    try:
        recorded = tuple(data.all_tests or ())
    except Exception:  # an unreadable map degrades to testmon's own behaviour
        return 0
    outside = frozenset(name for name in recorded if not under(name.split("::", 1)[0], paths))
    if not outside:
        return 0

    original = data.sync_db_fs_tests

    def sync_db_fs_tests(retain, _original=original, _outside=outside):
        return _original(retain=set(retain) | set(_outside))

    try:
        data.sync_db_fs_tests = sync_db_fs_tests
        data._arrayscope_scope_guard = True
    except AttributeError:  # a testmon upgrade made the object immutable
        return 0
    return len(outside)


def seed_map(rootdir: os.PathLike[str] | str) -> str | None:
    """Copy a sibling checkout's map in when this one has none. Returns the donor.

    A fresh worktree would otherwise pay a full traced run (~4 min) before
    selection helps at all, which is precisely when it would help most: an agent
    branch touches a handful of files.

    Copying is safe because every fingerprint in the map is content-addressed
    and path-relative — the donor's records are compared against *this* tree's
    files, so whatever differs is invalidated and re-run. The failure mode is
    "runs more tests than necessary", never "runs fewer". Package set and Python
    version are part of the environment key, so a map from a different
    environment simply reads as empty.

    Copy, never link. Two checkouts sharing one map would fight: each run
    records its own tree's fingerprints over the other's and vacuums what the
    other still references, so both would see everything as changed forever —
    and SQLite would serialize concurrent runs on top of that.
    """

    if os.environ.get("ARRAYSCOPE_TESTMON_SEED", "").strip().lower() in {"0", "off", "no", "false"}:
        return None
    destination = map_path(rootdir)
    if destination.exists():
        return None

    donor = _best_donor_map(Path(rootdir).resolve(), destination.name)
    if donor is None:
        return None
    # Copy through a temporary name in the destination directory: a half-written
    # SQLite file read by a concurrently starting run would look like a corrupt
    # map rather than a missing one.
    import shutil
    from tempfile import mkstemp

    handle, staged = mkstemp(dir=str(destination.parent), prefix=f".{destination.name}.")
    os.close(handle)
    try:
        shutil.copyfile(donor, staged)
        os.replace(staged, destination)
    except OSError:
        Path(staged).unlink(missing_ok=True)
        return None
    _seed_coverage_sidecar(donor, destination)
    return str(donor)


def _seed_coverage_sidecar(donor: Path, destination: Path) -> None:
    """Carry the donor's coverage sidecar across with its map.

    Two things ride in that file, and both want to come: the block-structure
    parse cache, which is content-addressed and therefore valid anywhere, and
    the donor's coverage baseline — which is exactly the right baseline for a
    checkout branched off the donor. It makes "since this checkout's baseline"
    mean "since main" in a worktree, with nothing to bootstrap, the same way
    seeding the map itself makes the inherited reds the donor's reds.
    """

    from tests import coverage_map

    source = coverage_map.sidecar_path(donor)
    if not source.exists():
        return
    import shutil

    with contextlib.suppress(OSError):
        shutil.copyfile(source, coverage_map.sidecar_path(destination))


def _best_donor_map(rootdir: Path, filename: str) -> Path | None:
    """The main checkout's map, else the largest sibling worktree's.

    The main checkout is preferred because a worktree is usually branched from
    it, so its map is the closest match. Size is the tie-breaker: a bigger map
    recorded more of the suite, and a stale-but-complete map costs one
    invalidation pass while a fresh-but-partial one leaves real holes.
    """

    import subprocess

    try:
        listing = subprocess.run(
            ("git", "worktree", "list", "--porcelain"),
            cwd=str(rootdir),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listing.returncode != 0:
        return None

    candidates: list[tuple[int, int, Path]] = []
    seen = 0
    for line in listing.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        # `git worktree list --porcelain` reports the main checkout first.
        is_main = seen == 0
        seen += 1
        checkout = Path(line[len("worktree ") :]).resolve()
        if checkout == rootdir:
            continue
        candidate = checkout / filename
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        if size == 0:
            continue
        candidates.append((1 if is_main else 0, size, candidate))
    if not candidates:
        return None
    return max(candidates)[2]


def resolve_environment(expression: str) -> str:
    """Evaluate the ``environment_expression`` ini value the way testmon does.

    Callers outside pytest (``tools/test_selection.py``) must land on the same
    key as the suite; otherwise they read a different — usually empty — slice of
    the map and report "nothing affected" for entirely the wrong reason.
    """

    from testmon.testmon_core import eval_environment

    return eval_environment(expression)


def peek(
    rootdir: os.PathLike[str] | str,
    environment: str,
    ignore_dependencies: tuple[str, ...] = (),
    with_changed_files: bool = False,
) -> MapPeek | None:
    """Read the map and diff it against the working tree. ``None`` if unusable.

    This runs before testmon initializes itself (xdist fixes its worker count
    first), so it has to reproduce testmon's inputs exactly — same environment
    key, same ignored dependencies. A mismatch would not merely mis-measure:
    testmon replaces an environment row whose package set changed, so peeking
    with different inputs would *discard* the map it is reading.
    """

    if not testmon_installed() or not map_path(rootdir).exists():
        return None

    from testmon.common import get_system_packages
    from testmon.testmon_core import TestmonData

    try:
        data = TestmonData.for_local_run(
            rootdir=str(rootdir),
            environment=environment,
            system_packages=get_system_packages(ignore=list(ignore_dependencies)),
        )
        data.determine_stable()
        all_tests = data.all_tests
    except Exception:  # a broken map must degrade to running everything
        return None

    affected = frozenset(data.unstable_test_names or ())
    seconds = sum((all_tests.get(name) or {}).get("duration") or 0.0 for name in affected)
    forced = frozenset(
        name
        for name, report in all_tests.items()
        if name not in affected and (report or {}).get("failed")
    )
    forced_seconds = sum((all_tests.get(name) or {}).get("duration") or 0.0 for name in forced)
    changed: frozenset[str] = frozenset()
    if with_changed_files:
        try:
            fshas = {}
            for name in data.files_of_interest:
                module = data.source_tree.get_file(name)
                if module is not None:
                    fshas[name] = module.fs_fsha
            changed = frozenset(data.db.fetch_unknown_files(fshas, data.exec_id))
        except Exception:  # a report line is not worth failing a run over
            changed = frozenset()

    return MapPeek(
        environment=environment,
        mapped_tests=len(all_tests),
        mapped_files=frozenset(data.all_files or ()),
        affected_tests=affected,
        affected_files=frozenset(data.unstable_files or ()),
        affected_seconds=seconds,
        forced_tests=forced,
        forced_files=frozenset(name.split("::", 1)[0] for name in forced),
        forced_seconds=forced_seconds,
        changed_files=changed,
    )


def _whole_file_lines(rootdir: os.PathLike[str] | str, relative: str) -> frozenset[int]:
    """Every line number of ``relative``, i.e. "this test depends on all of it".

    testmon fingerprints a dependency by hashing the AST blocks that cover the
    lines a test executed. Handing it the whole file therefore records the whole
    file's method checksums — the correct granularity for code we know ran, but
    only in a process we could not trace.
    """

    cached = _whole_file_line_cache.get(relative)
    if cached is not None:
        return cached
    path = Path(rootdir) / relative
    try:
        count = len(path.read_bytes().splitlines())
    except OSError:
        count = 0
    lines = frozenset(range(1, count + 1))
    _whole_file_line_cache[relative] = lines
    return lines


def inject_out_of_process_dependencies(report, rootdir: os.PathLike[str] | str) -> None:
    """Add the declared child-process entry points to a report's coverage.

    Called before testmon turns ``report.nodes_files_lines`` into fingerprints,
    so the declaration lands in the map as an ordinary dependency and every
    later run compares it the same way testmon compares everything else. No
    forcing, no always-run list: these tests are selected when their entry point
    changes and stay out of the way otherwise.
    """

    nodes_files_lines = getattr(report, "nodes_files_lines", None)
    if not nodes_files_lines:
        return
    for test_name, files_lines in nodes_files_lines.items():
        declared = OUT_OF_PROCESS_DEPENDENCIES.get(test_name.split("::", 1)[0])
        if not declared:
            continue
        for relative in declared:
            lines = _whole_file_lines(rootdir, relative)
            if lines:
                files_lines[relative] = set(lines)


class UsageError(Exception):
    """Raised for a flag the developer can fix; conftest turns it into pytest's."""


#: Refs tried, in order, when ``--since`` is given without one. The upstream
#: comes first so a branch stacked on another branch measures itself against its
#: parent rather than against ``main`` — otherwise every stacked branch inherits
#: its parent's whole diff. ``ARRAYSCOPE_BASELINE_REF`` overrides the lot, which
#: is the answer for a worktree whose parent git cannot infer (a branch split in
#: two, a temporary branch cut from somewhere unusual).
_BASELINE_REFS = ("@{upstream}", "main", "origin/main")

#: Above this many changed files, reading and AST-parsing each one's baseline
#: revision stops being cheap — and a sweep that large reaches everything
#: anyway, so the honest answer is to run the whole suite.
_MAX_BASELINE_FILES = 400


def _git(rootdir, *arguments, binary: bool = False):
    """Run one git command in ``rootdir``. ``None`` on any failure."""

    import subprocess

    try:
        finished = subprocess.run(
            ("git", *arguments),
            cwd=str(rootdir),
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout if binary else finished.stdout.decode("utf-8", "replace")


def resolve_baseline(rootdir, explicit: str | None) -> tuple[str, str]:
    """``(ref, merge_base_sha)`` for the branch point. Raises ``UsageError`` if none.

    Reported by the caller rather than assumed, because "what counts as the
    baseline" is a real choice on a stacked or split branch and a silent default
    would be the wrong kind of convenient.
    """

    candidates = [explicit] if explicit else []
    if not explicit:
        env_ref = os.environ.get("ARRAYSCOPE_BASELINE_REF", "").strip()
        candidates = ([env_ref] if env_ref else []) + list(_BASELINE_REFS)
    for ref in candidates:
        merge_base = _git(rootdir, "merge-base", ref, "HEAD")
        if merge_base and merge_base.strip():
            return ref, merge_base.strip()
    tried = ", ".join(candidates)
    raise UsageError(
        f"--since: no baseline could be resolved (tried {tried}). Name one "
        "explicitly, e.g. `--since origin/main`, or set ARRAYSCOPE_BASELINE_REF."
    )


def baseline_method_checksums(rootdir, merge_base: str) -> dict[str, list | None] | None:
    """``{path: checksums at the branch point}`` for everything the branch touched.

    ``None`` for a path that is not Python, or did not exist at the branch
    point, which testmon's comparison reads as "all of it changed" — the right
    fallback for a fixture array or an icon, which have no method structure to
    compare.
    """

    from testmon.process_code import Module

    tracked = _git(rootdir, "diff", "--name-only", merge_base)
    untracked = _git(rootdir, "ls-files", "--others", "--exclude-standard")
    if tracked is None or untracked is None:
        return None

    changed = {line.strip() for line in (tracked + untracked).splitlines() if line.strip()}
    if len(changed) > _MAX_BASELINE_FILES:
        return None

    checksums: dict[str, list | None] = {}
    for path in changed:
        source = (
            _git(rootdir, "show", f"{merge_base}:{path}", binary=True)
            if path.endswith(".py")
            else None
        )
        if source is None:
            checksums[path] = None
            continue
        try:
            checksums[path] = Module(
                source_code=source.decode("utf-8", "replace"),
                filename=path,
                rootdir=str(rootdir),
            ).method_checksums
        except Exception:  # unparseable back there: treat as fully changed
            checksums[path] = None
    return checksums


def tests_reached_since_baseline(data, checksums) -> frozenset[str] | None:
    """Recorded tests whose executed code this branch actually changed.

    Method-level, using testmon's own fingerprint comparison against the branch
    point, rather than "any test that recorded a changed file". Measured on a
    four-file diff that includes ``tests/conftest.py``: 41 tests method-level
    against 55 file-level. The gap grows with how widely the touched code is
    executed, and it is the difference between a targeted pre-merge run and a
    flag nobody uses.

    (It is *not* the difference between 41 and the whole suite: a test's record
    for ``conftest.py`` covers only the fixture blocks it actually executed, so
    even the file-level answer is far short of everything. The earlier version
    of this comment claimed otherwise and the measurement refuted it.)

    Read-only, so it works on an xdist worker too, whose database connection is
    not writable and where testmon's own ``determine_tests`` would fail.
    """

    if not checksums:
        return frozenset()
    from testmon.db import blob_to_checksums, check_fingerprint_db

    names = sorted(checksums)
    try:
        column = data.db._test_execution_fk_column()
        placeholders = ",".join("?" * len(names))
        rows = data.db.con.execute(
            f"""
            SELECT te.test_name, f.filename, f.method_checksums
            FROM test_execution te, test_execution_file_fp te_ffp, file_fp f
            WHERE te.{column} = ?
              AND te.id = te_ffp.test_execution_id
              AND te_ffp.fingerprint_id = f.id
              AND f.filename IN ({placeholders})
            """,
            [data.exec_id, *names],
        ).fetchall()
    except Exception:  # schema moved; tests/app/test_test_selection.py guards this
        return None
    return frozenset(
        row["test_name"]
        for row in rows
        if not check_fingerprint_db(
            checksums, row["filename"], blob_to_checksums(row["method_checksums"])
        )
    )


def apply_since_baseline(config, explicit_ref: str | None) -> tuple[str, int]:
    """Un-deselect everything this branch changed since its baseline.

    The map answers "what changed since the last run", which is the right
    question during an edit loop and the wrong one before merging: by then every
    file has been recorded, so the map reports *nothing* affected while the
    branch has changed twenty files. ``--since`` asks the other question, and it
    is the pre-merge command precisely because it still selects: it is the
    branch-sized answer at map speed, not a sweep.
    """

    ref, merge_base = resolve_baseline(config.rootdir, explicit_ref)
    plugin = config.pluginmanager.get_plugin("TestmonSelect")
    data = getattr(config, "testmon_data", None)
    if plugin is None or data is None:
        return ref, 0

    checksums = baseline_method_checksums(config.rootdir, merge_base)
    if checksums is None:
        raise UsageError(
            f"--since {ref}: this branch changed more than {_MAX_BASELINE_FILES} files, "
            "which reaches everything anyway. Run `pytest --testmon-noselect` — same "
            "tests, and it re-records the map on the way through."
        )
    reached = tests_reached_since_baseline(data, checksums)
    if reached is None:
        return ref, 0
    try:
        plugin.deselected_tests = set(plugin.deselected_tests) - reached
        keep = {name.split("::", 1)[0] for name in reached}
        plugin.deselected_files = {
            name for name in set(plugin.deselected_files) if name not in keep
        }
    except (AttributeError, TypeError):  # a testmon upgrade renamed them
        return ref, 0
    return ref, len(reached)


COUPLED_MARKER = "coupled_order"


def file_seconds(config) -> dict[str, float]:
    """Recorded wall time per test file, from the map. Absent files are omitted."""

    data = getattr(config, "testmon_data", None)
    if data is None:
        return {}
    try:
        totals: dict[str, float] = {}
        for name, report in data.all_tests.items():
            duration = (report or {}).get("duration")
            if duration:
                totals[name.split("::", 1)[0]] = totals.get(name.split("::", 1)[0], 0.0) + duration
        return totals
    except Exception:  # ordering is an optimization; never fail a run for it
        return {}


#: A test file at or below this recorded cost goes in the fast prefix (below).
#: 500 ms covers ~72% of the files for 15 s of work — 0.5 s of makespan once
#: spread over the workers, against 28 s of slack. Raising it much further
#: starts eating the slack that absorbs bad duration estimates.
FAST_FILE_SECONDS = 0.5


def order_for_makespan(items: list, seconds: dict[str, float], declaration_order: dict[str, int]):
    """Fast files first, then longest-first; declaration order within each file.

    **The ordering is for determinism and first-feedback latency, not for wall
    time.** A queue model over this suite's own recorded durations (256 files, 8
    workers, floor 196 s) predicted a large makespan win for longest-first —
    296.5 s shortest-first against 196.4 s longest-first. Measured on the real
    suite, three runs each, the two orders are indistinguishable: 225.7 / 226.1 /
    226.2 s longest-first against 226.7 / 227.4 s shortest-first. The model is
    wrong because xdist prefetches the next work unit as a worker's current file
    drains, so the tail it predicts never forms, and because durations recorded
    under 8-way parallel load do not compose the way a serial model assumes.
    Do not restate the modeled numbers as a result.

    What the order is chosen for, then:

    * **Immediate feedback.** Pure longest-first starts every worker on
      something enormous and reports nothing for 49 s. Leading with every
      sub-500 ms file costs nothing measurable and puts the first results on
      screen in about a millisecond.
    * **Determinism inside a file** (below), which shortest-first cannot give.
    * Longest-first for the remainder because it is the right shape if the
      balance ever does matter — it is free, and it is not a regression.

    Files the map has never timed lead: unknown cost is more safely assumed
    large than small.

    Declaration order *within* a file is restored on the way past. testmon's
    sort reshuffles it, and since recorded durations change every run the
    reshuffle differs every run — which turned three latent order dependencies
    in this suite into failures that moved from file to file between runs
    (``tests/gpu/test_chunk_codec.py``, ``tests/ui/test_window_sync.py``,
    ``tests/operations/test_operation_library.py``). Those are real bugs worth
    fixing, but a run order that changes underneath them is not how to surface
    them: it makes every red ambiguous. A deliberate shuffle would be.
    """

    grouped: dict[str, list] = {}
    for item in items:
        grouped.setdefault(item.nodeid.split("::", 1)[0], []).append(item)
    for members in grouped.values():
        members.sort(key=lambda item: declaration_order.get(item.nodeid, 0))

    def rank(name: str):
        cost = seconds.get(name)
        if cost is None:
            return (0, 0.0, name)  # never timed: lead, and assume it is large
        if cost <= FAST_FILE_SECONDS:
            return (1, cost, name)  # the fast prefix, cheapest first
        return (2, -cost, name)  # everything else, longest first

    ordered = []
    for name in sorted(grouped, key=rank):
        ordered.extend(grouped[name])
    return ordered


def regroup_coupled_items(items: list, declaration_order: dict[str, int], groups: dict[str, str]):
    """Reassemble ``coupled_order`` groups after testmon reorders ``items``.

    testmon sorts what it selected by recorded duration so the quickest tests
    fail first, which is worth keeping. What is not survivable is that sort
    splitting a pair of tests that only means anything in sequence:
    ``tests/core/test_test_state_isolation.py`` installs a module in one test
    and asserts on its identity in the next, and the split reported a phantom
    teardown regression.

    So the ordering preference stays global and the exception is explicit: every
    member of a marked group moves to where the group's earliest member landed,
    in declaration order. A group is otherwise sorted like anything else, so an
    expensive group does not drag itself to the front.

    Pure, and separate from the hook, so the guard test can prove it reorders.
    """

    first_seen: dict[str, int] = {}
    members: dict[str, list] = {}
    for position, item in enumerate(items):
        group = groups.get(item.nodeid)
        if group is None:
            continue
        first_seen.setdefault(group, position)
        members.setdefault(group, []).append(item)
    if not members:
        return list(items)

    for group, group_items in members.items():
        group_items.sort(key=lambda item: declaration_order.get(item.nodeid, 0))
        members[group] = group_items

    emitted: set[str] = set()
    result = []
    for position, item in enumerate(items):
        group = groups.get(item.nodeid)
        if group is None:
            result.append(item)
        elif first_seen[group] == position:
            result.extend(members[group])
            emitted.add(group)
    return result


def collect_test_files(rootdir: os.PathLike[str] | str, paths: tuple[str, ...] = ()) -> set[str]:
    """Every ``test_*.py`` under ``paths`` (default: the whole suite), relative."""

    root = Path(rootdir)
    bases = [root / path for path in paths] if paths else [root / "tests"]
    found: set[str] = set()
    for base in bases:
        if base.is_file():
            candidates: list[Path] = [base]
        elif base.is_dir():
            candidates = list(base.rglob("test_*.py"))
        else:
            continue
        for candidate in candidates:
            if candidate.name.startswith("test_") and candidate.suffix == ".py":
                found.add(candidate.resolve().relative_to(root).as_posix())
    return found


def path_arguments(config) -> tuple[str, ...]:
    """The plain path arguments of a run, relative to rootdir.

    Node-id arguments (``file.py::test``) are dropped: testmon turns selection
    off for those anyway, so they must not be allowed to narrow the estimate.
    For the other question — what this run could *collect*, where a node id is
    every bit as narrow as a directory — see :func:`collected_scope`.
    """

    return _resolved_arguments(config, node_ids=False)


def collected_scope(config) -> tuple[str, ...]:
    """The files and directories this run's arguments could collect.

    Like :func:`path_arguments`, except a node id contributes its *file*:
    ``pytest file.py::test`` collects nothing outside that file, so anything the
    run concludes about the rest of the suite is a conclusion about tests it
    never looked at. Different question from the impact estimate, different
    answer — which matters, because getting it wrong here would let
    :func:`protect_map_outside_the_scope` treat a one-test run as exhaustive and
    hand testmon the whole map to garbage-collect.
    """

    return _resolved_arguments(config, node_ids=True)


def _resolved_arguments(config, *, node_ids: bool) -> tuple[str, ...]:
    root = Path(config.rootdir)
    invocation_dir = Path(getattr(config.invocation_params, "dir", os.getcwd()))
    resolved: list[str] = []
    for argument in config.args:
        if "::" in argument:
            if not node_ids:
                continue
            argument = argument.split("::", 1)[0]
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = invocation_dir / argument
        if not candidate.exists():
            continue
        try:
            resolved.append(candidate.resolve().relative_to(root).as_posix())
        except ValueError:
            continue
    return tuple(resolved)


def collected_the_whole_suite(config) -> bool:
    """Whether this run's arguments cover every configured test root.

    Not the same as "no path arguments": ``testpaths`` resolves a bare ``pytest``
    to ``tests``, so the ordinary inner-loop run *does* carry one, and a caller
    that treated any path argument as a narrowing scope would classify every run
    as scoped. Both callers here — the map's scope guard and the coverage report
    — need "did this look at the whole suite", which is this.
    """

    paths = collected_scope(config)
    if not paths:
        return True
    try:
        roots = tuple(str(entry) for entry in (config.getini("testpaths") or ()))
    except (ValueError, KeyError):
        roots = ()
    if not roots:
        return False  # nothing to compare against: assume it narrowed
    return all(under(root, paths) for root in roots)


def under(relative_path: str, paths: tuple[str, ...]) -> bool:
    """Whether ``relative_path`` sits inside one of ``paths`` (empty = the suite)."""

    if not paths:
        return True
    return any(relative_path == path or relative_path.startswith(f"{path}/") for path in paths)
