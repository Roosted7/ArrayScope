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
   map; the modules that entry point *imports* remain invisible.
2. **Non-Python inputs.** Icons and fixture arrays have no fingerprint.
3. **Real rendering.** Rings 3–4 (``docs/testing/README.md``) prove things
   about pixels a compositor drew. Selection has nothing to say there and the
   ring rules are unchanged.

Blind spots 1 and 2 used to be handed to CI, "swept exhaustively on every
push". Measured on 2026-07-30 that was false: CI has run seven times in this
repository's life, most recently with ``main`` 247 commits ahead of it, and
there is no required check. The local backstop is an occasional
``pytest --testmon-noselect`` and the cadence line in ``tests/conftest.py``
that asks for one when the last is over three weeks or forty commits old. CI,
when it runs, is a bonus. Blind spot 3 has no offscreen answer at all and
belongs to rings 3–4.

``pytest --since`` is the pre-merge command: it asks what the whole *branch*
changed rather than what moved since the last run, and reports the answer as
evidence — the map already holds a passing run against byte-identical code for
most of what a settled branch reaches, so only the gaps run. ``--fresh``
re-acquires the lot. ``pytest --no-testmon`` stays available for the narrow
cases — regenerating the canonical artifacts, or settling a run this map is
suspected of getting wrong — and is not a routine sweep.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
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
    # The per-test-batch oracle runs a real pytest+testmon session over files
    # generated in tmp_path: a second live testmon inside this process would
    # fight the one recording the oracle itself. The child's only repo-tracked
    # code is `tests/testmon_policy.py`, which the tests here already execute
    # in-process, so there is nothing to re-attach.
    "tests/app/test_test_selection.py": (),
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


def pin_per_test_dependency_batches() -> bool:
    """Give every test its own coverage batch, so its record is its own.

    testmon batches up to 250 tests into one live ``Coverage`` object and
    switches a dynamic context per test; the map's dependency edges are read
    back from ``contexts_by_lineno``. coverage's ``sysmon`` core — its default
    on CPython 3.14+ — uses ``sys.monitoring``'s DISABLE optimization: each
    line location reports once per ``Coverage`` lifetime, so a line an earlier
    test in the batch already executed is never attributed to a later one.
    Every test after the first then records a *subset* of what it ran, and a
    missing edge is the silent kind of wrong: the file changes, the test is
    not selected, and ``--since`` calls the branch verified without ever
    reaching it. Measured on
    ``test_session_fixture_shape_mismatch_raises_actionable_error``: 69
    dependency files recorded alone, 8 when its file ran in one process —
    neither ``view_session.py`` nor ``view_recipe.py`` among them, which is
    how the 2026-07-31 ``view_recipe`` regression (``28140661``) passed every
    ``--since`` on its branch and surfaced only under ``--testmon-noselect``.

    The remedy is *not* pinning a trace-based core: ``COVERAGE_CORE=ctrace``
    attributes correctly but was measured to turn a 267 s full run into 388 s
    and to flip 13 timing-sensitive ``tests/ui`` settle/quiescence tests red
    (2 red under sysmon, different families) — the tracer's per-line cost is
    itself a scheduling perturbation. Batch size 1 keeps sysmon's near-zero
    per-line cost and is sufficient: testmon ends a batch with
    ``stop → collect → erase → start``, and a fresh start resets the DISABLE
    state, so attribution within a one-test batch is complete (probed against
    both cores; the guard in ``tests/app/test_test_selection.py`` records two
    real tests through a real testmon run and fails if either loses its shared
    dependency). The cycle costs ~5 ms per test, noise against this suite.

    Pinned from ``tests/conftest.py`` at import time, which precedes testmon's
    first ``Coverage`` construction (first-test setup) and reaches xdist
    workers the same way. Returns whether testmon was importable; without it
    there is nothing to record and :func:`decide` already reports why.

    A map recorded before this pin under-reports dependencies for most of the
    suite and cannot heal itself — the tests with missing edges are exactly
    the ones selection no longer runs. Re-record it:
    ``pytest --testmon-noselect``.
    """

    try:
        from testmon import testmon_core
    except ImportError:
        return False
    testmon_core.TEST_BATCH_SIZE = 1
    return True


def attribution_gap(sweep_record: dict) -> str:
    """Why this map's records cannot back an evidence claim, or ``""``.

    A sweep record carrying ``attributed`` was written by code that pins
    per-test coverage batches (see :func:`pin_per_test_dependency_batches`),
    so every entry in the map behind it was recorded with complete dependency
    edges. Any other state means the map may hold records made under batched
    sysmon coverage — each a *subset* of the truth — and a missing edge is the
    invisible kind of wrong: the test is not selected, not "reached", and
    reported verified. A map with no full record at all is the same hole with
    a different history: its entries came from a donor whose records predate
    the fix, and nothing since has re-recorded them wholesale.

    (A map recorded from scratch post-fix stamps itself: its first run
    executes everything, which the sweep detector counts as exhaustive.)

    The gap heals exactly one way — ``pytest --testmon-noselect`` — and the
    callers make that the path of least resistance rather than a footnote:
    the inner loop names it on every run, and ``--since`` refuses until it
    has happened, because a gate whose "reached" set is itself under-scoped
    cannot make the claim the gate exists to make.
    """

    if not isinstance(sweep_record, dict):
        sweep_record = {}
    if sweep_record.get("swept") and sweep_record.get("attributed"):
        return ""
    if sweep_record.get("swept"):
        return (
            "this map's last full record predates per-test dependency "
            "attribution (2026-07-31: batched sysmon coverage credited each "
            "line only to the first test in a 250-test batch)"
        )
    return (
        "this map has never been fully recorded under per-test dependency "
        "attribution (its entries were seeded from records that predate the "
        "2026-07-31 fix)"
    )


@dataclass(frozen=True)
class Decision:
    """Whether selection runs for one pytest invocation, and why."""

    active: bool
    reason: str
    #: True when the developer asked for it themselves; the default policy then
    #: leaves every testmon option exactly as given.
    explicit: bool = False


def testmon_installed(config=None) -> bool:  # noqa: PT028 - not a test, it only reads like one
    """Whether testmon is here *for this run*, not merely importable.

    With a ``config`` the question is answered by the pluginmanager, because
    that is the object that would have done the work. ``find_spec`` answers
    "the package is installed", which is a different question and was
    measurably the wrong one: ``-p no:pytest-testmon`` leaves the package
    perfectly importable while blocking the entry point, so selection reported
    itself active, deselected nothing, recorded nothing, and the header printed
    ``test selection: on ... this run records the map`` — false in both halves,
    with no gate anywhere. Reading registration turns that into an ordinary
    inactive decision, which the bypass gate then prices like any other.

    Without a ``config`` the package answer is the only one available, and it is
    the right one for the callers outside a pytest run (``tools/test_selection.py``).
    """

    if config is not None:
        with contextlib.suppress(AttributeError):
            return bool(config.pluginmanager.hasplugin("pytest-testmon"))
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
    if not testmon_installed(config):
        return Decision(False, "pytest-testmon is installed but not registered for this run")

    options = vars(config.option)
    if options.get("no-testmon"):
        return Decision(False, "--no-testmon")
    if any(options.get(name) for name in _EXPLICIT_OPTIONS):
        return Decision(True, "requested on the command line", explicit=True)

    # There is deliberately no environment override here. It duplicated the
    # command-line flags in both directions -- off is ``--no-testmon``, on is
    # ``--testmon`` -- and a second spelling of "bypass selection" is exactly
    # the kind of thing that escapes a refusal built around the first.

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
    #: Recorded wall time per test node id. Used to price a run *before* it
    #: starts — the only way to refuse an expensive one while it can still be
    #: retyped. Absent or never-recorded tests are simply missing, so a sum
    #: over this is a lower bound, which is the safe direction for a refusal.
    test_seconds: Mapping[str, float] = field(default_factory=dict)
    #: Files whose content differs from what the map recorded. Only filled in
    #: when ``with_changed_files`` was asked for; note this is "different from
    #: the map", not "different from git HEAD" — an old map makes an untouched
    #: file changed, which is exactly the truth selection acts on.
    changed_files: frozenset[str] = frozenset()
    #: The live ``TestmonData`` this reading was taken from, when the caller
    #: asked to keep it (``keep_data``). It is the object testmon's own
    #: ``init_testmon_data`` would otherwise build a second copy of — see
    #: :func:`reuse_map_read`. Excluded from equality and repr: it is a database
    #: handle, not part of what this value *says*.
    data: object | None = field(default=None, compare=False, repr=False)

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
    dependency set of the run that failed.

    **``--since`` implies it.** That run means "everything this branch changed,
    ready to merge", and a red it declined to re-run is a hole in exactly that
    claim: the moment the branch lands, an inherited red is main's red. It used
    to only *print* the skip, one grey line under a bold ``900 passed``, which
    is a line that gets read as noise -- it was, for a whole session, by the
    author of this paragraph. The inner loop keeps the cheap behaviour, because
    two minutes to re-confirm what everybody knows is an inner loop nobody uses.

    The flag is the only spelling. An ``ARRAYSCOPE_TESTMON_RERUN_FAILING``
    variable used to shadow it, justified by workers being "handed no command
    line of their own"; that was measured false on 2026-07-30 — an xdist worker
    reads ``rerun_reds=True`` from the controller's own command line — so the
    variable was a duplicate and is gone. Scripts and CI steps pass the flag.
    """

    if config is not None:
        try:
            if bool(config.getoption("rerun_reds", False)):
                return True
            # --since is a merge gate, so it owns the inherited reds too.
            if config.getoption("since", None) is not None:
                return True
        except (AttributeError, ValueError):
            # Not a pytest run, or the option was never registered — the
            # environment variable is the answer for both.
            pass
    return False


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


def seed_map(rootdir: os.PathLike[str] | str, *, seeding: bool = True) -> str | None:
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

    if not seeding:
        return None
    destination = map_path(rootdir)
    if destination.exists():
        return None

    donor = _best_donor_map(Path(rootdir).resolve(), destination.name)
    if donor is None:
        return None
    if not _copy_map(donor, destination):
        return None
    return str(donor)


def _copy_map(donor: Path, destination: Path) -> bool:
    """Atomically put ``donor`` at ``destination``, sidecar and flakes included.

    Through a temporary name in the destination directory: a half-written
    SQLite file read by a concurrently starting run would look like a corrupt
    map rather than a missing one.

    Three files travel, and the map is the only one that has to arrive. The two
    passengers are copied — never linked, for the same reason the map is not —
    after the map is in place, so a failure in either costs what it carries and
    nothing else.
    """

    import shutil
    from tempfile import mkstemp

    handle, staged = mkstemp(dir=str(destination.parent), prefix=f".{destination.name}.")
    os.close(handle)
    try:
        shutil.copyfile(donor, staged)
        os.replace(staged, destination)
    except OSError:
        Path(staged).unlink(missing_ok=True)
        return False
    _seed_coverage_sidecar(donor, destination)
    _seed_flake_ledger(donor, destination)
    return True


def _seed_flake_ledger(donor: Path, destination: Path) -> None:
    """Carry the donor's flake observations across with its map.

    Without this an observation lives and dies inside one checkout, which is the
    wrong lifetime for it twice over: a worktree is throwaway, so a branch's
    observations were being deleted with it, and the map it made them about
    goes on to be adopted by the main checkout — arriving able to vouch for
    greens somebody watched flip. Merge rather than replace, because the
    destination's observations are as real as the donor's; see
    ``tests/red_ledger.py:carry_flakes``.
    """

    from tests import red_ledger

    with contextlib.suppress(Exception):  # observations, never a result
        red_ledger.carry_flakes(donor, destination)


def _worktree_heads(rootdir: Path) -> list[tuple[Path, str]]:
    """``(checkout, HEAD)`` for every worktree of this repository."""

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
        return []
    if listing.returncode != 0:
        return []
    found: list[tuple[Path, str]] = []
    checkout: Path | None = None
    for line in listing.stdout.splitlines():
        if line.startswith("worktree "):
            checkout = Path(line[len("worktree ") :]).resolve()
        elif line.startswith("HEAD ") and checkout is not None:
            found.append((checkout, line[len("HEAD ") :].strip()))
            checkout = None
    return found


def adopt_map_from_identical_worktree(rootdir: os.PathLike[str] | str) -> str | None:
    """Take the map of a sibling worktree checked out at this exact commit.

    Almost all work here happens in a throwaway worktree, so the map that
    recorded it is the *branch's*, and it is deleted with the worktree. The main
    checkout meanwhile keeps a map from before the merge: it reports reds
    inherited from whatever seeded it, and re-records work that was already
    traced minutes earlier somewhere else. Landing a branch therefore used to
    cost a `--rerun-reds` and a `--since` in the main checkout purely to quiet a
    map nobody had updated.

    After a fast-forward the two HEADs are equal, which is stronger than the
    "close enough donor" :func:`seed_map` settles for: identical commits mean
    identical fingerprints, so the adopted map needs no invalidation pass at
    all. Anything that does differ is still content-addressed and re-runs, so
    the failure mode stays "runs more tests than necessary".

    Newest wins, because a merge is not the only way two checkouts sit on one
    commit and the map that ran most recently is the one that knows most.

    When adoption is suspected of smuggling in a wrong record, look at the
    *recording* first: the 2026-07-31 ``view_recipe`` miss (``28140661``) was
    blamed on adoption across agent worktrees, and refuted — the donor's record
    was already missing its edges the moment it was written, because the sysmon
    coverage core credits a shared line only to the first test in a batch. See
    :func:`pin_per_test_dependency_batches`. The same-HEAD gate above held.
    """

    rootdir = Path(rootdir).resolve()
    destination = map_path(rootdir)
    heads = dict(_worktree_heads(rootdir))
    head = heads.get(rootdir)
    if head is None:
        return None

    best: tuple[float, Path] | None = None
    for checkout, other in heads.items():
        if checkout == rootdir or other != head:
            continue
        candidate = checkout / destination.name
        if not candidate.is_file() or candidate.stat().st_size == 0:
            continue
        stamp = candidate.stat().st_mtime
        if best is None or stamp > best[0]:
            best = (stamp, candidate)
    if best is None:
        return None
    if destination.is_file() and destination.stat().st_mtime >= best[0]:
        return None  # ours already ran later; adopting would lose work
    if not _copy_map(best[1], destination):
        return None
    return str(best[1])


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


# --------------------------------------------------------------------------- #
# Module-block forgiveness
# --------------------------------------------------------------------------- #
#
# testmon collapses to whole-file granularity the moment a module's top-level
# statement list changes at all, because the entire module body is one AST block
# spanning line 1 to EOF and that block is therefore in 100% of the file's
# fingerprints. Adding one inert function to `arrayscope/window/frame_session.py`
# selected 258 of 3933 tests and cost a measured 123.16 s of wall clock, with
# zero tests whose executed code had changed. Editing a function *body* is
# exactly precise (32 of 166 on `render/lod.py`), so this is not "testmon is
# coarse" — it is one checksum, and it is the whole story.
#
# The fix appends provably-equivalent retired checksums to the list
# `db.check_fingerprint_db` subset-tests each fingerprint against. That is the
# safety floor: appending to the right-hand side of a subset test can only turn
# "affected" into "unaffected", never the reverse, and it cannot change anything
# else testmon does. The entire risk surface is "is each forgiven checksum
# really equivalent?", one checksum at a time, and it is answered by
# `coverage_map.forgives_module_block` (a pure insertion of statements nothing
# already there can observe) and by the index-free body digest (identical digest
# means byte-identical AST; only the block's ordinal moved).
#
# Everything degrades to today's behaviour: no sidecar, no retained revision, a
# testmon upgrade that moved a name, an unparseable file — no forgiveness. The
# failure mode stays "runs more tests than necessary", which is the invariant
# `seed_map` already documents.


def declared_whole_file_paths() -> frozenset[str]:
    """Entry points whose honest dependency is the whole file, so never forgiven.

    :data:`OUT_OF_PROCESS_DEPENDENCIES` re-attaches these by hand at whole-file
    granularity precisely because *nothing* about the child process is traced.
    Forgiveness rests on "no execution we recorded can observe the new name",
    and for an untraced child there is no recording to reason from —
    ``tests/app/test_qt_binding.py`` asserts on a fresh interpreter's
    ``sys.modules`` after importing ``arrayscope/__init__.py``, which a new name
    in that file can move.

    Note this exclusion is load-bearing rather than decorative: the argument that
    a whole-file fingerprint "contains every block, so any real change removes
    one" does **not** hold for a pure addition, where every old block survives
    and every one of them would be forgiven. ``tests/app/test_selection_forgiveness.py``
    proves it.
    """

    return frozenset(path for paths in OUT_OF_PROCESS_DEPENDENCIES.values() for path in paths)


def forgivable_checksums(
    rootdir, relative: str, revisions, parsed: dict | None = None
) -> list[int]:
    """Retired checksums of ``relative`` that still describe its current revision.

    Two rules, both against each retained previous revision:

    * a block whose **index-free body digest** still exists in the current
      revision — identical digest means the AST of the code a test executed is
      byte-identical and only its ordinal within the file moved, which nothing
      can observe;
    * the **module block**, when the current revision is the previous one with
      inert definitions inserted (see ``coverage_map.forgives_module_block``).

    Skipped entirely for a file that mentions ``__all__``: adding a name there
    changes an ``import *`` surface, and the check is textual because it costs
    nothing and errs toward selecting. The residual hole — a test that
    introspects the module with ``dir()`` or ``getattr()`` — is bounded by a
    measured fact rather than argued away: this tree's introspective guards
    (``tests/app/test_dead_code.py``, ``test_architecture_guards.py``) read
    source as *data*, so selection never runs them for an ``arrayscope/`` edit
    in the first place. See ``docs/testing/test-selection.md``.

    ``parsed`` is the sidecar's cached decomposition of this same revision, when
    the caller already had it. It carries the body digests and the element
    digests, which is everything the first rule needs and the *cheap* half of the
    second — so the AST parse is deferred to the one branch that genuinely needs
    ``Element`` objects (a retained module block whose statements differ from the
    current ones), and skipped entirely otherwise.
    """

    from testmon.process_code import read_source_sha

    from tests import coverage_map

    if not relative.endswith(".py") or relative in declared_whole_file_paths():
        return []
    # Read even when ``parsed`` was supplied: the ``__all__`` gate is textual,
    # and a read is a thousandth of what a parse costs.
    source, _ = read_source_sha(Path(rootdir) / relative)
    if source is None or "__all__" in source:
        return []
    elements = None
    if parsed is not None and parsed.get("elements") is not None:
        current_digests = list(parsed["elements"])
        bodies = {
            record[-1]
            for record in parsed.get("functions") or ()
            if isinstance(record, list | tuple) and len(record) >= 2
        }
    else:
        functions, _module, elements = coverage_map.revision_of(source, relative, rootdir)
        if elements is None:
            return []
        current_digests = coverage_map.digests_of(elements)
        bodies = {function.body for function in functions}

    forgiven: list[int] = []
    for entry in revisions:
        if not isinstance(entry, dict):
            continue
        for record in entry.get("functions") or ():
            # ``[checksum, …, body]``: the current revision keeps a line and a
            # name for reporting, a retained one is stripped to the two ends.
            if not isinstance(record, list | tuple) or len(record) < 2:
                continue
            if record[-1] in bodies:
                forgiven.append(record[0])
        module = entry.get("module")
        retained = entry.get("elements")
        if module is None or not isinstance(retained, list) or retained == current_digests:
            continue  # identical: its checksum is already there
        if elements is None:
            elements = coverage_map.elements_of(source, relative, rootdir)
            if elements is None:
                continue
        if coverage_map.forgives_module_block(retained, elements):
            forgiven.append(module)
    return forgiven


def install_forgiveness(rootdir: os.PathLike[str] | str) -> bool:
    """Patch ``collect_mhashes`` to serve from cache and forgive. Idempotent.

    One patch covers every caller: ``TestmonData.determine_stable`` reaches
    ``collect_mhashes`` through the module global, so the pytest plugin,
    :func:`peek` and ``tools/test_selection.py`` all go through it, and the
    derivation of the affected set — which this module deliberately does not own
    — is untouched.

    Two things happen here, and they share one lookup.

    **Serving the checksums from the sidecar.** testmon re-parses every file
    whose recorded fsha row does not match, and on this tree that is a *standing*
    set: a file no affected test depends on is never re-recorded, so it never
    stops differing, and its AST is rebuilt on every run forever — 0.084 s for
    nine files, growing linearly with the branch's diff. The sidecar already
    holds that file's blocks, keyed by content, because the coverage figure needs
    their identities; ``coverage_map.method_checksums_of`` reads testmon's own
    list straight back out of it. Verified equal to testmon's on all 574 cached
    files of this tree.

    **Forgiveness**, unchanged in effect, now handed the same cached
    decomposition so its own parse is skipped too.

    Nothing here can produce a wrong answer, only a slower run. The cache key is
    the file's content hash, so a stale entry cannot match; a miss, a sidecar
    from an older shape, or any exception falls back to testmon parsing the file
    exactly as it does today.

    Returns whether the patch is in place. ``False`` means there was no sidecar
    to read yet, and the next call retries: a fresh checkout seeds one with its
    map, and until then selection is exactly what it was.
    """

    try:
        import testmon.testmon_core as core

        from tests import coverage_map
    except ImportError:
        return False

    original = core.collect_mhashes
    # The marker is the function it replaced, so "already installed?" and "what
    # was here before?" are one attribute.
    if getattr(original, "arrayscope_forgiveness", None) is not None:
        return True
    payload = coverage_map.read_sidecar(map_path(rootdir))
    if not coverage_map.cached_structure(payload):
        return False

    def collect_mhashes(source_tree, new_changed_file_data):
        cached: dict[str, dict] = {}
        for relative in new_changed_file_data:
            try:
                entry = coverage_map.revision_at(payload, rootdir, relative)
            except Exception:  # never let the cache be why a run cannot start
                entry = None
            if entry is not None and coverage_map.method_checksums_of(entry) is not None:
                cached[relative] = entry
        unparsed = [name for name in new_changed_file_data if name not in cached]
        files_mhashes = original(source_tree, unparsed) if unparsed else {}
        for relative, entry in cached.items():
            files_mhashes[relative] = coverage_map.method_checksums_of(entry)

        for relative, checksums in list(files_mhashes.items()):
            if not checksums:
                continue
            try:
                extra = forgivable_checksums(
                    rootdir,
                    relative,
                    coverage_map.previous_revisions(payload, relative),
                    parsed=cached.get(relative),
                )
            except Exception:  # never let this be the reason a run cannot start
                continue
            known = set(checksums)
            addition = [checksum for checksum in extra if checksum not in known]
            if addition:
                files_mhashes[relative] = list(checksums) + addition
        return files_mhashes

    collect_mhashes.arrayscope_forgiveness = original
    core.collect_mhashes = collect_mhashes
    return True


def peek(
    rootdir: os.PathLike[str] | str,
    environment: str,
    ignore_dependencies: tuple[str, ...] = (),
    with_changed_files: bool = False,
    keep_data: bool = False,
) -> MapPeek | None:
    """Read the map and diff it against the working tree. ``None`` if unusable.

    This runs before testmon initializes itself (xdist fixes its worker count
    first), so it has to reproduce testmon's inputs exactly — same environment
    key, same ignored dependencies. A mismatch would not merely mis-measure:
    testmon replaces an environment row whose package set changed, so peeking
    with different inputs would *discard* the map it is reading.

    ``keep_data`` hands the live ``TestmonData`` back on the result instead of
    dropping it. Reproducing testmon's inputs exactly is the condition this
    function already had to meet, so the object it built *is* the one testmon is
    about to build again; :func:`reuse_map_read` turns that from a coincidence
    into the thing that happens.
    """

    if not testmon_installed() or not map_path(rootdir).exists():
        return None

    install_forgiveness(rootdir)
    # Before the import below, which binds whatever the name points at now.
    memoise_system_packages()

    from testmon.common import get_system_packages
    from testmon.testmon_core import TestmonData

    try:
        packages = get_system_packages(ignore=list(ignore_dependencies))
        data = TestmonData.for_local_run(
            rootdir=str(rootdir),
            environment=environment,
            system_packages=packages,
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
        test_seconds={
            name: (report or {}).get("duration") or 0.0
            for name, report in all_tests.items()
            if (report or {}).get("duration")
        },
        forced_tests=forced,
        forced_files=frozenset(name.split("::", 1)[0] for name in forced),
        forced_seconds=forced_seconds,
        changed_files=changed,
        data=data if keep_data else None,
    )


# --------------------------------------------------------------------------- #
# Reading the map once instead of twice
# --------------------------------------------------------------------------- #


def map_read_identity(rootdir, environment: str, ignore_dependencies) -> tuple[str, str, str]:
    """The three inputs that decide *which* map a reading is a reading of.

    Everything else ``TestmonData.for_local_run`` takes is derived (the Python
    version) or unused here (``database``, which is the tmnet RPC proxy and is
    ``None`` for every local run). If these three agree, the two objects are
    reading the same rows of the same file with the same environment row — which
    is the condition :func:`peek` has always had to meet anyway, stated as a
    value that can be compared instead of as a comment that can rot.
    """

    from testmon.common import get_system_packages

    return (
        str(rootdir),
        str(environment),
        get_system_packages(ignore=list(ignore_dependencies or ())),
    )


def memoise_system_packages() -> bool:
    """Cache ``get_system_packages`` for the life of the process. Idempotent.

    It walks ``importlib.metadata.distributions()`` over the whole environment —
    0.045 s measured, and it was paid twice per run because :func:`peek` and
    testmon's own ``init_testmon_data`` each ask for it. The installed
    distributions cannot change inside one pytest process, so a second walk can
    only produce the answer the first one did.

    Patched in all three modules that hold a binding: ``testmon.common`` owns it,
    and ``testmon.pytest_testmon`` and ``testmon.testmon_core`` each imported the
    name at module import time, so patching only the owner would leave both
    callers on the original.
    """

    try:
        import testmon.common as common
        import testmon.pytest_testmon as plugin
        import testmon.testmon_core as core
    except ImportError:
        return False

    original = common.get_system_packages
    if getattr(original, "arrayscope_memoised", False):
        return True

    cache: dict[tuple, str] = {}

    def get_system_packages(ignore=None):
        key = tuple(ignore) if ignore else ()
        if key not in cache:
            cache[key] = original(ignore=ignore)
        return cache[key]

    get_system_packages.arrayscope_memoised = True
    for module in (common, plugin, core):
        if getattr(module, "get_system_packages", None) is original:
            module.get_system_packages = get_system_packages
    return True


def reuse_map_read(config, peeked: MapPeek | None) -> bool:
    """Make testmon adopt the ``TestmonData`` :func:`peek` already built.

    testmon's ``init_testmon_data`` constructs a second ``TestmonData`` and runs
    ``determine_stable()`` over it, having no way to know one was built minutes
    of CPU earlier — ``peek`` exists precisely because xdist fixes the worker
    pool in ``pytest_cmdline_main``, before testmon's ``pytest_configure`` has
    anything to ask. Measured, the pair cost 0.33 s of a 1.15 s no-op run.

    Interception rather than pre-setting ``config.testmon_data``: testmon calls
    ``init_testmon_data`` unconditionally, so anything set beforehand is
    overwritten. The call goes through the module global, so replacing that name
    reaches it.

    Refused, falling back to today's double build, when:

    * there is no peeked reading, or it was taken without ``keep_data``;
    * this is an xdist **worker** — a worker's data comes from
      ``TestmonData.for_worker`` with the controller's ``exec_id``, which is a
      different object built from different inputs, and handing it a local-run
      reading would have it record against the wrong execution;
    * ``(rootdir, environment, system packages)`` differ from what testmon is
      about to ask for.

    Returns whether the interception is in place. Idempotent.

    The offer is carried on ``config``, never captured in the patch. The patch is
    a module global and outlives any one run, so a closure over one run's data
    would hand a *stale* reading to the next ``pytest.main()`` in the same
    process. Keyed on the config, a run that made no offer simply gets testmon's
    own behaviour.
    """

    try:
        import testmon.pytest_testmon as plugin
    except ImportError:
        return False

    data = getattr(peeked, "data", None)
    if data is None or hasattr(config, "workerinput"):
        return False
    try:
        identity = map_read_identity(
            config.rootdir,
            getattr(data, "environment", None),
            config.getini("testmon_ignore_dependencies"),
        )
    except Exception:  # a peek we cannot identify is one we must not reuse
        return False
    config.arrayscope_map_read = (identity, data)

    original = plugin.init_testmon_data
    if getattr(original, "arrayscope_reuse", None) is not None:
        return True

    def init_testmon_data(target):
        offered = adopted_map_read(target)
        if offered is None:
            return original(target)
        target.testmon_data = offered
        return None

    init_testmon_data.arrayscope_reuse = original
    plugin.init_testmon_data = init_testmon_data
    return True


def adopted_map_read(config):
    """The ``TestmonData`` this config offered, if it is the one testmon wants.

    Separate from the patch so the identity rule is testable without a pytest
    run, and pure in the sense that matters: it never builds anything, and any
    doubt at all resolves to ``None``, which is today's double build.
    """

    offer = getattr(config, "arrayscope_map_read", None)
    if not offer or hasattr(config, "workerinput"):
        return None
    identity, data = offer
    # tmnet routes every read through an RPC proxy instead of the local file.
    testmon_config = getattr(config, "testmon_config", None)
    if getattr(testmon_config, "tmnet", False) or getattr(config, "tmnet", None):
        return None
    try:
        from testmon.testmon_core import eval_environment

        environment = config.getoption("environment_expression") or eval_environment(
            config.getini("environment_expression")
        )
        wanted = map_read_identity(
            config.rootdir, environment, config.getini("testmon_ignore_dependencies")
        )
    except Exception:  # cannot prove they match, so they do not
        return None
    if wanted != identity or getattr(data, "environment", None) != identity[1]:
        return None
    return data


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


def head_sha(rootdir) -> str:
    """The current commit, or ``""``. For pinning a record to a point in history."""

    return (_git(rootdir, "rev-parse", "HEAD") or "").strip()


def commits_since(rootdir, sha: str) -> int | None:
    """How far ``HEAD`` has moved past ``sha``. ``None`` if git cannot say.

    ``None`` is a real answer and not a zero: a rebase or a fresh clone can
    leave a recorded sha that no longer exists, and reporting "0 commits" for a
    commit git has never heard of would be a confident lie about distance.
    """

    if not sha:
        return None
    counted = _git(rootdir, "rev-list", "--count", f"{sha}..HEAD")
    if counted is None:
        return None
    try:
        return int(counted.strip())
    except ValueError:
        return None


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


def describe_baseline(rootdir, explicit: str | None) -> str:
    """One line naming what a bare ``--since`` decided to compare against.

    A bare ``--since`` picks from three sources in order --
    ``ARRAYSCOPE_BASELINE_REF``, the upstream, then ``main`` -- and which one it
    finds changes the answer completely. Standing on ``main`` in this
    repository it resolves to ``origin/main``, which nothing has been pushed to
    for 261 commits, so the "branch-sized" question quietly becomes a
    2000-test one. The ref alone does not show that; the distance does, which
    is why it is counted here and printed before the run rather than after.
    """

    try:
        ref, merge_base = resolve_baseline(rootdir, explicit)
    except UsageError as error:
        return f"test selection: --since could not resolve a baseline ({error})"
    chosen = "named" if explicit else "resolved"
    behind = (_git(rootdir, "rev-list", "--count", f"{merge_base}..HEAD") or "").strip()
    ahead = (_git(rootdir, "rev-list", "--count", f"HEAD..{ref}") or "").strip()
    parts = [f"test selection: --since {ref} ({chosen}), branch point {merge_base[:8]}"]
    if behind:
        parts.append(f"{behind} commit(s) of yours since it")
    if ahead and ahead != "0":
        parts.append(f"{ref} is {ahead} ahead of you")
    return " — ".join(parts)


def baseline_method_checksums(rootdir, merge_base: str) -> dict[str, list | None] | None:
    """``{path: checksums at the branch point}`` for everything the branch touched.

    ``None`` for a path that is not Python, or did not exist at the branch
    point, which testmon's comparison reads as "all of it changed" — the right
    fallback for a fixture array or an icon, which have no method structure to
    compare.

    Each list is extended by :func:`since_forgivable_checksums`, which is module-
    block forgiveness in mirror image: here the fingerprints are *current* and
    the checksum list is the *baseline*'s, so the equivalent checksums to append
    are the current ones. Without it ``--since`` degrades to file-level for every
    file the branch added a function to — measured 166 → 32, 258 → 0, 157 → 16,
    297 → 0 on this tree.
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
            baseline = Module(
                source_code=source.decode("utf-8", "replace"),
                filename=path,
                rootdir=str(rootdir),
            ).method_checksums
        except Exception:  # unparseable back there: treat as fully changed
            checksums[path] = None
            continue
        # Forgiveness never fails a run; it only ever declines.
        with contextlib.suppress(Exception):
            baseline = list(baseline) + since_forgivable_checksums(rootdir, path, source)
        checksums[path] = baseline
    return checksums


def since_forgivable_checksums(rootdir, relative: str, baseline_source: bytes) -> list[int]:
    """Current checksums of ``relative`` that the branch point already describes.

    The mirror of :func:`forgivable_checksums`, and cheaper: ``git show`` has
    already produced the baseline revision, so nothing has to be retained
    anywhere. Same two rules, same gate, same degradation — an unreadable or
    unparseable revision on either side forgives nothing.

    Appending to the baseline's checksum list is again a subset test's
    right-hand side, so the worst a mistake here can do is leave one test out of
    a pre-merge run, and each appended checksum is one of: a block whose
    index-free body digest already existed at the branch point, or the module
    block of a file the branch only inserted inert definitions into.
    """

    from testmon.process_code import read_source_sha

    from tests import coverage_map

    if not relative.endswith(".py") or relative in declared_whole_file_paths():
        return []
    current_source, _ = read_source_sha(Path(rootdir) / relative)
    if current_source is None:
        return []
    baseline_text = baseline_source.decode("utf-8", "replace")
    if "__all__" in current_source or "__all__" in baseline_text:
        return []
    baseline_elements = coverage_map.elements_of(baseline_text, relative, rootdir)
    functions, module_checksum, current_elements = coverage_map.revision_of(
        current_source, relative, rootdir
    )
    if baseline_elements is None or current_elements is None:
        return []

    before = coverage_map.bodies_at(rootdir, relative, baseline_source)
    forgiven = [function.checksum for function in functions if function.body in before]
    if module_checksum is not None and coverage_map.forgives_module_block(
        coverage_map.digests_of(baseline_elements), current_elements
    ):
        forgiven.append(module_checksum)
    return forgiven


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


#: Why a test this branch reaches cannot be skipped, most-specific first. The
#: order is both the tie-break (a test qualifying twice gets the first label)
#: and the order the closing sentence prints them in.
RUN_REASONS = ("broken here", "without evidence", "inherited red", "known flake")

#: The other half of the partition: the map already holds a green for this exact
#: code, so running it again could only reproduce it.
VERIFIED = "verified"


def partition_reached(
    reached,
    *,
    recorded: frozenset[str] | set[str],
    stable: frozenset[str] | set[str],
    failed: frozenset[str] | set[str],
    broken_here: frozenset[str] | set[str],
    flaky: frozenset[str] | set[str],
) -> dict[str, frozenset[str]]:
    """Split the tests a branch reaches into "already proven" and "must run".

    The map is an evidence ledger, and this is the reading of it. A test is
    **verified for the current tree** when its fingerprint is stable — which,
    content-addressing being what it is, means its last recorded run executed
    byte-identical code to what is on disk — and that run passed. Re-running it
    cannot produce new information; it can only reproduce the record.

    Everything else reaches the run, for one of four reasons:

    * ``broken here`` — the red ledger owns it. The map cannot vouch for a red
      it recorded, because the map was written by the run that broke it.
    * ``without evidence`` — unstable (a bare ``pytest`` would run it anyway) or
      never recorded at all. New tests land here, which is the point.
    * ``inherited red`` — stable, recorded, and its last outcome was a failure.
      ``--since`` is a merge gate and the moment the branch lands an inherited
      red is ``main``'s red, so the gate owns it.
    * ``known flake`` — stable and green, but observed to flip outcome with no
      code change (``tests/red_ledger.py``), so its green is not evidence.

    Pure, and total: every reached name lands in exactly one bucket, and the
    counts therefore add up in the sentence the run prints. Membership in
    ``recorded``/``stable``/``failed`` is read the way :func:`known_red_tests`
    reads it, from the map, so a serial run and an xdist controller agree.
    """

    buckets: dict[str, set[str]] = {name: set() for name in (VERIFIED, *RUN_REASONS)}
    for name in reached:
        if name in broken_here:
            buckets["broken here"].add(name)
        elif name not in recorded or name not in stable:
            buckets["without evidence"].add(name)
        elif name in failed:
            buckets["inherited red"].add(name)
        elif name in flaky:
            buckets["known flake"].add(name)
        else:
            buckets[VERIFIED].add(name)
    return {reason: frozenset(names) for reason, names in buckets.items()}


@dataclass(frozen=True)
class SinceReport:
    """What one ``--since`` run found, and what it decided to do about it."""

    ref: str
    reached: int
    verified: int
    #: ``(reason, count)`` in :data:`RUN_REASONS` order, empty reasons dropped.
    running: tuple[tuple[str, int], ...] = ()
    #: The node ids to un-deselect. Only the applying process needs these; the
    #: controller recomputes the counts read-only for its summary.
    run_tests: frozenset[str] = frozenset()
    #: ``--fresh``: evidence was deliberately not consulted.
    fresh: bool = False

    @property
    def run(self) -> int:
        return sum(count for _, count in self.running)

    def sentence(self) -> str:
        """The one line every ``--since`` run prints, whatever its shape."""

        tests = "test" if self.reached == 1 else "tests"
        head = f"--since {self.ref}: branch reaches {self.reached} {tests}"
        if self.fresh:
            return f"{head} — --fresh, so all {self.reached} run and re-acquire their evidence."
        if not self.reached:
            return f"{head} — nothing to verify."
        why = ", ".join(f"{count} {reason}" for reason, count in self.running)
        return (
            f"{head} — {self.verified} verified green against this exact tree from "
            f"recorded runs, {self.run} run now ({why})."
            if why
            else f"{head} — all {self.verified} verified green against this exact tree "
            "from recorded runs, nothing left to run."
        )


def since_partition(config, reached, new_reds) -> dict[str, frozenset[str]] | None:
    """:func:`partition_reached` fed from this run's map and ledgers.

    ``None`` when any input cannot be read. Every caller turns that into "run
    everything reached", which is the pre-partition behaviour — the degradation
    direction is always *more* tests, never fewer.
    """

    from tests import red_ledger

    data = getattr(config, "testmon_data", None)
    if data is None:
        return None
    try:
        all_tests = dict(data.all_tests or {})
        flaky = red_ledger.read_flakes(map_path(config.rootdir), data.environment)
        return partition_reached(
            reached,
            recorded=frozenset(all_tests),
            stable=frozenset(data.stable_test_names or ()),
            failed=frozenset(
                name for name, report in all_tests.items() if (report or {}).get("failed")
            ),
            broken_here=frozenset(new_reds or ()),
            flaky=frozenset(flaky),
        )
    except Exception:  # an unreadable map or ledger must widen the run, not narrow it
        return None


def apply_since_baseline(
    config, explicit_ref: str | None, new_reds=(), *, fresh: bool = False
) -> SinceReport:
    """Run the evidence gaps this branch opened, and *report* the rest as proven.

    The map answers "what changed since the last run", which is the right
    question during an edit loop and the wrong one before merging: by then every
    file has been recorded, so the map reports *nothing* affected while the
    branch has changed twenty files. ``--since`` asks the other question, and it
    is the pre-merge command precisely because it still selects: it is the
    branch-sized answer at map speed, not a sweep.

    This used to un-deselect **everything** the branch reached, which on a
    faithfully-iterated branch is ~95% re-confirmation: those tests already ran,
    against byte-identical code, and passed. The map is an evidence ledger, so
    the honest pre-merge act is to *read* it — report the reached set, state
    which part of it is already proven for this exact tree, and run only the
    gaps. See :func:`partition_reached` for what counts as a gap.

    ``fresh`` restores the old behaviour verbatim: re-acquire all evidence.
    That is the right call for a flake hunt, or for a map you distrust.

    **Degradation.** Any failure to compute the partition — an unreadable
    ledger, a testmon upgrade, a map that will not answer — falls back to
    running everything reached. Never to skipping more: a wrong answer here has
    to cost time, not coverage.
    """

    ref, merge_base = resolve_baseline(config.rootdir, explicit_ref)
    plugin = config.pluginmanager.get_plugin("TestmonSelect")
    data = getattr(config, "testmon_data", None)
    if plugin is None or data is None:
        return SinceReport(ref, 0, 0)

    checksums = baseline_method_checksums(config.rootdir, merge_base)
    if checksums is None:
        # The command named here has to be one that actually runs. It used to be
        # `pytest --testmon-noselect`, which the bypass gate refused at 1320 s --
        # a documented remedy that exited 4. It is now priced on whether it
        # repairs anything, and a branch this size always differs from the map,
        # so the recommendation is sound again. Should it ever refuse, the
        # refusal is itself the answer: nothing differs, so nothing to find.
        raise UsageError(
            f"--since {ref}: this branch changed more than {_MAX_BASELINE_FILES} files, "
            "which reaches everything anyway. Run `pytest --testmon-noselect` — same "
            "tests, and it re-records the map on the way through."
        )
    reached = tests_reached_since_baseline(data, checksums)
    if reached is None:
        return SinceReport(ref, 0, 0)
    report = since_verification(ref, reached, config, new_reds, fresh=fresh)
    try:
        plugin.deselected_tests = set(plugin.deselected_tests) - report.run_tests
        keep = {name.split("::", 1)[0] for name in report.run_tests}
        plugin.deselected_files = {
            name for name in set(plugin.deselected_files) if name not in keep
        }
    except (AttributeError, TypeError):  # a testmon upgrade renamed them
        return SinceReport(ref, len(reached), 0)
    return report


def since_verification(ref, reached, config, new_reds=(), *, fresh: bool = False) -> SinceReport:
    """The read-only half of :func:`apply_since_baseline`, decided once.

    Split out because two processes need the same answer from the same inputs:
    the one that mutates testmon's deselection lists (an xdist *worker*, where
    collection happens) and the one that owns the terminal (the controller,
    which never collects and so recomputes this instead). A number derived twice
    by two code paths is a number that eventually disagrees with itself.
    """

    reached = frozenset(reached)
    partition = None if fresh else since_partition(config, reached, new_reds)
    if partition is None:
        # Degraded, or asked for: run everything reached. Reported as one bucket
        # rather than guessed at, because a fabricated breakdown of an unread
        # ledger is worse than no breakdown.
        return SinceReport(
            ref,
            len(reached),
            0,
            (("without evidence", len(reached)),) if not fresh and reached else (),
            reached,
            fresh=fresh,
        )
    running = tuple((reason, len(partition[reason])) for reason in RUN_REASONS if partition[reason])
    return SinceReport(
        ref,
        len(reached),
        len(partition[VERIFIED]),
        running,
        frozenset().union(*(partition[reason] for reason in RUN_REASONS)),
    )


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
