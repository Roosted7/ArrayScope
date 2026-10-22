import os
import shlex
import sys
import tempfile
from time import monotonic

import pytest

from tests.filesystem_isolation import install_artifact_directory, isolated_process_root

# --- Test filesystem isolation (QSettings + artifacts) -----------------------
# Two shared on-disk resources leak developer/cross-process state into tests:
#   * QSettings — the test QApplication resolves a UserScope store under
#     ``XDG_CONFIG_HOME``. Under pytest-qt's ``qapp`` the organizationName is
#     empty, so Qt merges in the org-level fallback ``Unknown Organization.conf``
#     — a file the autouse ``_clear_qt_settings`` / ``clear_arrayscope_settings``
#     helpers CANNOT wipe (``QSettings().clear()`` only touches the app-scoped
#     file, never the org fallback). On a developer box that fallback holds real
#     persisted keys (e.g. ``image_rendering_backend=wgpu``), so a serial
#     ``-n 0`` run reads the developer's live config and builds the wrong image
#     backend, while xdist workers (isolated below) silently pass — the exact
#     pass/fail-by-parallelism signature that made ``-n 0`` unusable for
#     debugging ``tests/ui/test_diagnostics_dialog.py``. The only reliable reset
#     is to point QSettings at a private, empty config dir; ``.clear()`` cannot
#     reach the fallback, so we must never let tests see it in the first place.
#   * ``tests/artifacts/`` — the Qt smoke test writes fixed filenames there, so
#     concurrent xdist workers would clobber each other.
# QSettings isolation applies to EVERY run (serial and parallel): tests must
# never read the developer's real ~/.config. Artifact isolation stays
# xdist-only, so serial CI still generates into the canonical ``tests/artifacts/``.
# This must happen at import time, before Qt (and thus QSettings) resolves any
# path.
# Default the whole suite to the offscreen Qt platform at import time (rings
# 0-2).  Ring-3/4 real-display runs pass QT_QPA_PLATFORM explicitly, which
# wins over this setdefault.  This also makes the Linux/Wayland display-server
# policy (arrayscope.app.qt_platform) env-overridden and therefore inert in
# tests — no test that calls the CLI main() may spawn a supervised child.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
# Per-run private config root: one per xdist worker, one per process for serial.
_config_root = str(
    isolated_process_root(
        tempfile.gettempdir(),
        worker=_XDIST_WORKER,
        run_uid=os.environ.get("PYTEST_XDIST_TESTRUNUID"),
        process_id=os.getpid(),
    )
)
_worker_config = os.path.join(_config_root, "config")
os.makedirs(_worker_config, exist_ok=True)
# On Linux/Unix QSettings (UserScope) resolves under XDG_CONFIG_HOME; pointing
# it at a private empty dir keeps every run's settings store hermetic and free
# of the un-clearable ``Unknown Organization.conf`` org fallback.
os.environ["XDG_CONFIG_HOME"] = _worker_config
if _XDIST_WORKER:
    install_artifact_directory(os.environ, _config_root)


from tests import (
    coverage_map,
    load_report,
    machine_pressure,
    red_ledger,
    skip_ledger,
    testmon_policy,
)

# testmon reads each test's dependency edges out of coverage dynamic contexts,
# and coverage's sysmon core (its default on CPython 3.14+) credits a line only
# to the FIRST test in a batch that executes it — silently slimming every later
# test's record until selection stops reaching it. Give each test its own
# batch here, at import time, before testmon builds its first Coverage at
# first-test setup. See testmon_policy.pin_per_test_dependency_batches for the
# incident write-up and why a trace-based core is NOT the remedy.
testmon_policy.pin_per_test_dependency_batches()

#: Written by the controller (or a serial run) only; workers read the file.
_red_ledger = None
#: Same writer, same lifetime: the flake ledger and the two map facts it needs
#: from *before* anything ran. See :func:`pytest_sessionstart`.
_flake_ledger = None
#: Same writer again: which recorded tests ended in a skip, so their dependency
#: sets stop being read as evidence of independence. See ``tests/skip_ledger.py``.
_skip_ledger = None
_forced_reds: frozenset[str] = frozenset()
_changed_since_recorded: frozenset[str] = frozenset()
#: What this run *is*, for anything it observes: settled once at sessionstart
#: because neither the worker count nor the command changes mid-run, unlike the
#: machine load beside it. See :func:`_flake_observation_context`.
_flake_run_shape: dict = {}
_flake_load_start = None
#: Node ids this run collected, gathered from wherever collection happened.
_collected: set[str] = set()

# A worker process here is not free: it imports Qt, pyqtgraph and the whole
# arrayscope package before it runs anything, which measures ~1.3 s wall. Below
# this much recorded work, starting even one of them costs more than the tests
# it would run, so a small selection runs in-process instead.
_SERIAL_BELOW_SECONDS = 2.5

#: Deliberately unpleasant, and deliberately *not* the bypass override. See the
#: help text on the option, and :func:`serial_run_refusal`.
SERIAL_FLAG = "--serial-force-please-i-really-consider-it-needed"

#: The third of the three, and separate from both for the same reason they are
#: separate from each other: "run this on a machine that is already busy" is a
#: claim about the machine, not about the tracer or the process count. The one
#: documented use is the load-robustness experiment in
#: ``docs/testing/strategy.md``, which saturates the machine on purpose.
#:
#: Unlike the other two, this one is *partial* — it excuses contention for the
#: cores and nothing else. See :data:`machine_pressure.NON_NEGOTIABLE`.
LOAD_FLAG = "--load-force-please-i-really-consider-it-needed"


#: The one line ``pytest --help`` prints above these options, directly under
#: testmon's own group. It used to be a bare pointer at a document, which is
#: worth what the reader's willingness to go and read it is worth -- and the
#: reader here has, by construction, just typed something that document told
#: them not to. So it states the policy instead, in one breath.
SELECTION_GROUP = (
    "arrayscope test selection — `pytest` runs what your tree affects; "
    "`pytest --since` before merging. Wide `--no-testmon` / `--testmon-noselect` / "
    "`-n 0` runs are refused by price. Reference: docs/testing/test-selection.md"
)


def pytest_addoption(parser):
    # One group, so the policy line is printed once for all of these rather than
    # repeated in every option's help.
    group = parser.getgroup("arrayscope selection", SELECTION_GROUP)
    group.addoption(
        "--since",
        metavar="REF",
        nargs="?",
        const="",
        default=None,
        help=(
            "Run everything this branch changed since REF (merge-base). Bare --since "
            "resolves the baseline: ARRAYSCOPE_BASELINE_REF, the upstream, then main. "
            "Use it before merging, after a rebase, or in a borrowed checkout."
        ),
    )
    group.addoption(
        "--fresh",
        action="store_true",
        default=False,
        help=(
            "With --since: re-acquire all evidence. Runs every test this branch "
            "reaches, including the ones the map already holds a green for against "
            "this exact code. For a flake hunt, or a map you distrust."
        ),
    )
    group.addoption(
        "--rerun-reds",
        action="store_true",
        default=False,
        help=(
            "Also re-run the reds this checkout inherited; selection skips them by "
            "default. Use it while fixing one. This is the only spelling -- xdist "
            "workers read it from the controller's command line."
        ),
    )
    group.addoption(
        "--no-seed-map",
        action="store_true",
        default=False,
        help=(
            "Do not copy a sibling checkout's map into a fresh worktree. Seeding is "
            "safe (fingerprints are content-addressed, so anything that differs is "
            "re-run) and saves a full traced run; this exists to test that."
        ),
    )
    group.addoption(
        "--no-testmon-force-please-i-really-consider-it-needed",
        action="store_true",
        default=False,
        help=(
            "Override the refusal of a selection-bypassing run. Long on purpose: "
            "needing it means selection has a blind spot, which is a defect to "
            "report against tests/testmon_policy.py, not a workflow. Implies "
            "--no-testmon."
        ),
    )
    group.addoption(
        SERIAL_FLAG,
        action="store_true",
        default=False,
        help=(
            "Override the refusal of a wide `-n 0` run. Separate from the bypass "
            "override on purpose: 'I need the tracer out of the way' and 'I need "
            "this in one process' are unrelated claims, and one flag granting "
            "both would hand each exemption to somebody who only meant the other."
        ),
    )
    group.addoption(
        LOAD_FLAG,
        action="store_true",
        default=False,
        help=(
            "Compete for a busy machine's cores anyway. For the load-robustness "
            "experiment, which saturates it deliberately. It does NOT reach the memory "
            "or swap refusals -- nothing does; that is the state this machine froze in."
        ),
    )


def worker_ceiling(cpus: int) -> int:
    """Half the logical cores, floored at two. The GL-context stability cap.

    Extracted so the ``-n auto`` path and the clamp on an explicit ``-n N``
    cannot drift apart: the cap only ever applied to the former, which is how
    an explicit ``-n 16`` came to defeat it silently.
    """

    return max(2, cpus // 2)


def pytest_xdist_auto_num_workers(config):
    """Size the ``-n auto`` pool for the work this run will actually do.

    Two caps, for different reasons:

    * **Half the logical cores, always.** Many tests create real rendering
      contexts and Qt surfaces. One worker per core has every worker building
      GL contexts against the same offscreen/software-GL stack at once, which
      intermittently segfaults the driver and starves the timing-sensitive UI
      tests. At half, workers are stable and the speedup is still large. On a
      2-core CI runner this floors at 2, so CI parallelism is unaffected.
    * **The number of test files change-selection left.** ``--dist loadfile``
      keeps a file on one worker, so files — not tests — bound the useful
      parallelism. Booting eight Qt workers to run two files is pure latency,
      and booting any at all to run one is worse than running it here.

    The second cap has to be decided before collection (xdist fixes the pool in
    ``pytest_cmdline_main``), so it comes from the recorded map rather than from
    collected items, and it counts test files the map has never seen as work —
    unknown means "assume it runs", never "assume it doesn't".
    """

    ceiling = worker_ceiling(os.cpu_count() or 2)

    workload = _selection_workload(config)
    if workload is None:
        return ceiling
    files, seconds, unmapped = workload
    if files == 0:
        return 0
    if files == 1 or (unmapped == 0 and seconds < _SERIAL_BELOW_SECONDS):
        return 0
    return max(2, min(ceiling, files))


def _selection_workload(config):
    """``(test files, recorded seconds, unmapped files)``, or ``None`` if unknown.

    ``None`` means "size for the whole suite": selection is off, the map is
    missing or empty, or the run carries a selector (``-k``, ``-m``, ``--lf``,
    a node id) that makes testmon stop deselecting.
    """

    if not testmon_policy.decide(config).active:
        return None
    _ensure_map_seeded(config)
    if not testmon_policy.selects(config):
        return None
    if config.getoption("since") is not None:
        # --since widens the run by an amount only collection can know, so the
        # map's answer would under-provision the pool.
        return None
    # Through the shared cache, not a peek of its own. This hook runs in
    # ``pytest_cmdline_main``, ahead of every pricing gate, so it is the run's
    # *first* reader — and a private read here meant a bare `pytest` opened and
    # stability-checked the map three times over (here, in the gates, and again
    # in testmon's own configure) rather than twice.
    peek = _map_peek(config)
    if peek is None or peek.empty:
        return None

    paths = testmon_policy.path_arguments(config)
    affected = {name for name in peek.affected_files if testmon_policy.under(name, paths)}
    seconds = peek.affected_seconds
    # Reds run whether or not anything they use changed, so they are part of the
    # workload the pool has to cover — leaving them out is how a thirteen-file
    # re-run ends up running in one process. Which reds run depends on the
    # policy: the ones this checkout broke always, the inherited ones only under
    # the flag.
    forced_files = {red.split("::", 1)[0] for red in _new_reds(config)}
    if testmon_policy.rerun_known_red_tests(config):
        forced_files |= set(peek.forced_files)
        seconds += peek.forced_seconds
    affected |= {name for name in forced_files if testmon_policy.under(name, paths)}
    unmapped = {
        name
        for name in testmon_policy.collect_test_files(config.rootdir, paths)
        if name not in peek.mapped_files
    }
    return len(affected | unmapped), seconds, len(unmapped)


def _new_reds(config):
    """Tests this checkout broke, from the ledger. Cached per run.

    Read from disk rather than passed down, because the two places that need it
    — worker sizing on the controller, and the deselection that happens on each
    xdist worker — are in different processes. The controller writes the file
    once, after every test has finished, so it is stable for the whole run.
    """

    cached = getattr(config, "arrayscope_new_reds", None)
    if cached is not None:
        return cached
    try:
        environment = testmon_policy.resolve_environment(config.getini("environment_expression"))
        found = red_ledger.read_new_reds(testmon_policy.map_path(config.rootdir), environment)
    except Exception:  # an unreadable ledger costs a re-run, never a wrong result
        found = set()
    config.arrayscope_new_reds = found
    return found


def _ensure_map_seeded(config):
    """Give a fresh checkout a map to work from, once, and remember the donor.

    Idempotent and safe to call from any of the three entry points that need
    it — xdist's pool sizing, ``pytest_cmdline_main`` and ``pytest_configure``,
    in that order, none of which is guaranteed to be the one that fires. The
    first to arrive does the copy; the rest cost a single attribute read.

    The health check comes first and is the reason this is the right moment for
    it: a map that fails it is taken out of the way, which leaves the checkout
    looking exactly like a fresh one — so the seeding two lines down is also
    the rebuild, and a corrupt map costs a copy rather than a traced suite.
    """

    if getattr(config, "arrayscope_selection_seed", "unset") != "unset" or _XDIST_WORKER:
        return
    _refuse_to_open_a_broken_map(config)
    try:
        config.arrayscope_selection_seed = testmon_policy.seed_map(
            config.rootdir, seeding=not config.getoption("no_seed_map", False)
        )
    except Exception:  # an unseeded run is slow, never wrong
        config.arrayscope_selection_seed = None


def _refuse_to_open_a_broken_map(config):
    """Take an unreadable map away before anything tries to read it.

    A malformed map is the one failure the harness had no answer for. Every
    reader here degrades politely — ``peek`` catches the error and reports "no
    usable map" — and then testmon's own ``init_testmon_data`` opens the same
    file seconds later and takes the session down with ``INTERNALERROR>
    sqlite3.DatabaseError: database disk image is malformed``, out of a
    traceback that names neither the map nor the checkout that copied it. Twice
    in two integration sessions that cost a full ``--testmon-noselect``
    rebuild, and, worse, the `--since` gates run in between had been answering
    from those same records.

    Printed rather than raised, straight to stderr rather than through the
    terminal reporter: this runs from ``pytest_cmdline_main`` as well as
    ``pytest_configure``, and the earlier of those is before there is a
    reporter to write to. The run continues — recording from scratch is the
    documented cost of having no map, and it is the cost this converts the
    crash into.
    """

    try:
        quarantined = testmon_policy.quarantine_broken_map(config.rootdir)
    except Exception:  # a health check may never be the thing that fails a run
        return
    if quarantined is None:
        return
    complaint, where = quarantined
    config.arrayscope_map_rebuilt = complaint
    print(
        f"test selection: this checkout's map is corrupt ({complaint}) — rebuilding.\n"
        f"test selection: the unreadable map is at {where} if it is worth a look.",
        file=sys.stderr,
    )


#: A run that bypasses selection may cost at most this. Above it, retyping the
#: command is cheaper than the run, so there is always time to be told to.
_UNSELECTED_ABORT_SECONDS = 5.0

#: Deliberately unpleasant. It is typed by someone claiming the harness left
#: them no choice, and that claim should cost something and be visible in a
#: shell history. See :func:`unselected_run_refusal`.
BYPASS_FLAG = "--no-testmon-force-please-i-really-consider-it-needed"

#: Printed after any run that actually bypassed selection.
BYPASS_BANNER = "!!! IF THIS WAS NEEDED, REPORT AND FIX THE TEST HARNESS !!!"


def unselected_run_refusal(seconds, permitted_because) -> str:
    """Why a selection-bypassing run is refused, or ``""`` to allow it.

    The rule is about **time**, not about which flag spelled it. ``pytest
    --no-testmon`` and ``pytest --testmon-noselect`` are the same act -- running tests the change provably cannot reach -- and
    the cost is the thing worth refusing, so it is the thing measured. A run
    that still selects is never touched here however long it takes: the map has
    already narrowed it to what the change touches, so that time is earned.

    ``seconds`` is priced from the map before collection, which is the only
    moment a refusal can still save the time it is refusing. It is summed
    *recorded test time*, not predicted wall clock -- the suite runs eight ways
    parallel, so wall clock is a fraction of it. The comparison that matters is
    against a selected run of the same tree, which is measured the same way.

    A bypass that was genuinely necessary is a **defect report**, not a
    workflow. If selection cannot see something, the fix belongs in
    ``testmon_policy`` (as the declared child-process class already is), not in
    a flag each person remembers separately -- hence the banner every bypassed
    run ends with.
    """

    if permitted_because:
        return ""
    if seconds <= _UNSELECTED_ABORT_SECONDS:
        return ""
    return (
        f"it skips selection and would run ~{seconds:.0f} s of recorded test time "
        f"the change cannot reach"
    )


#: A ``--no-testmon`` run may name at most this much recorded work. Past it,
#: a list of node ids is a sweep with names on it. Six times the general bypass
#: allowance, because this shape is the one legitimate question and answering it
#: on a slow test should not need an override.
_UNTRACED_ABORT_SECONDS = 30.0


def untraced_run_refusal(named_only, seconds, permitted_because) -> str:
    """Why a ``--no-testmon`` run is refused, or ``""``. Shape first, then price.

    ``--no-testmon`` records nothing, so unlike every other bypass its cost is
    the whole story — and price alone could not express the part that matters.
    Under the old price-only rule ``pytest --no-testmon tests/app/test_test_selection.py``
    ran, because that file is under 5 s, while ``docs/testing/test-selection.md``
    had always claimed a file was refused. That gap is how the habit gets
    taught: one narrowed sweep costs seconds, reads as a clean pass, nothing
    pushes back, and the same shape is later aimed at the whole suite.

    So the allow is written around the one question ``--no-testmon`` uniquely
    answers — *is the tracer itself causing this?* — which is asked of named
    tests and answered in seconds. Anything wider is somebody reaching past
    selection and paying twice: once in wall clock, and again because the run
    leaves the map exactly as it found it.
    """

    if permitted_because:
        return ""
    if not named_only:
        return (
            "--no-testmon is for named tests only. It records nothing, so a bare run, "
            "a directory or a whole file leaves the map exactly as it found it"
        )
    if seconds > _UNTRACED_ABORT_SECONDS:
        return (
            f"--no-testmon names ~{seconds:.0f} s of recorded tests, which is a sweep "
            "with a list of names on it"
        )
    return ""


def noselect_run_refusal(mapped_tests, unmapped_files, changed_files, permitted_because) -> str:
    """Why a ``--testmon-noselect`` sweep is refused, or ``""``. Product, not price.

    The bypass gate's principle is "the cost is the thing worth refusing", and
    that is right for ``--no-testmon``, which leaves nothing behind.
    ``--testmon-noselect`` leaves a *repaired map*, so its cost is an
    investment — and the honest rule is that an investment is refusable exactly
    when there is nothing to repair.

    Pricing it instead was a real dead end, not a theoretical one: bare
    ``--testmon-noselect`` was refused at 1320 s while three separate places
    named it as the sanctioned repair — this page, ``tools/test_selection.py``'s
    epilog, and the ``--since`` >400-file error, which told you to run a command
    that exits 4.

    What the three branches buy:

    * **Nothing mapped** — there is nothing to compare against, so let it record.
    * **Holes, or a tree that differs from the map** — a big rebase, a >400-file
      ``--since``, a refactored directory. This run repairs something, which is
      the only reason to type it.
    * **Complete and current** — 222 s, about one spurious ordering failure (the
      reorder this suite's latent couplings cost), and a byte-identical map.
      That is ceremony, and the refusal is itself the answer to the question the
      sweep was asking.

    One measured caveat on the middle branch. ``changed_files`` comes from
    testmon's ``fetch_unknown_files``, which reports every file whose recorded
    fsha row does not match — a *byte-level* question, not "which tests are
    affected". On this repository's map that is a standing set of six files that
    re-recording does not clear, so in practice this rule nearly always permits.
    That is the safe direction (the refused command was the documented repair,
    and over-permitting it costs four minutes rather than a wrong answer), but
    the third branch should not be described as well-exercised.
    """

    if permitted_because:
        return ""
    if not mapped_tests:
        return ""
    if unmapped_files or changed_files:
        return ""
    return (
        f"the map is complete for this tree ({mapped_tests} tests, 0 unmapped files, "
        "0 files differ), so this sweep would re-run everything and change nothing"
    )


def unselected_run_seconds(test_seconds, scope, under) -> float:
    """Recorded cost of the tests a bypassing run would execute."""

    if not scope:
        return sum((test_seconds or {}).values())
    return sum(
        duration
        for name, duration in (test_seconds or {}).items()
        if under(name.split("::", 1)[0], scope)
    )


def node_ids_only(arguments) -> bool:
    """Whether every argument names one test, so the run cannot be wide.

    The exemption that keeps :func:`unpriceable_bypass_refusal` from being a
    wall: ``pytest --no-testmon file.py::test`` runs what it names and nothing
    else, whatever the map holds, so it needs no price to be safe. A bare run
    names nothing, which is the *widest* shape rather than the narrowest — hence
    the emptiness check.
    """

    named = tuple(str(argument) for argument in (arguments or ()))
    return bool(named) and all(".py::" in argument for argument in named)


def unpriceable_bypass_refusal(mapped_tests, permitted_because, *, node_ids_only) -> str:
    """Why an *unpriceable* bypass is refused, or ``""`` to allow it.

    The price rule below reads an unreadable map as 0 s, and 0 s opens the gate.
    Four separate routes land there, and every one of them was a working bypass
    on 2026-07-30: a fresh clone, a worktree that has not recorded yet,
    ``TESTMON_DATAFILE`` pointing at a path that does not exist (measured —
    price 0, allowed), and a map recorded under a different environment key.
    Each makes the gate a no-op at exactly the moment there is no evidence to
    defend it with.

    So for a bypass, and only for a bypass, unknown cost is assumed large. The
    direction is deliberately the opposite of the one the serial gate takes
    (see :func:`serial_run_refusal`), because the two are protecting against
    different mistakes: here an unpriceable run is the whole untraced suite,
    there it is a ring that has never been recorded and must be serial.
    """

    if permitted_because or node_ids_only:
        return ""
    if mapped_tests:
        return ""
    return (
        "the map holds nothing for this environment, so a run that skips selection "
        "cannot be priced -- and for a bypass, unknown cost is assumed large"
    )


def selection_materialised(config) -> bool:
    """Whether the plugin that was supposed to select actually turned up.

    Every degradation path in this layer fails **open** — an unreadable map, a
    renamed testmon attribute, a blocked entry point — and each one leaves the
    header printing ``test selection: on`` over a run that selected nothing. The
    invariant that closes the class is that a run may only *claim* selection if
    it can prove it: the plugin is registered, and it built the configuration
    object it was supposed to build. Anything else is a bypass, and is priced
    like one.
    """

    try:
        if not config.pluginmanager.hasplugin("pytest-testmon"):
            return False
    except AttributeError:
        return False
    return bool(getattr(config, "testmon_config", None))


def _bypass_permitted(config) -> str:
    """Why an unselected run is legitimate here, or ``""``."""

    options = vars(config.option)
    if options.get(BYPASS_FLAG.lstrip("-").replace("-", "_")):
        return "explicitly forced"
    if os.environ.get("CI"):
        return "CI, where every job is a gate"
    # ``cov_source``, not ``cov``: pytest-cov registers ``--cov`` with
    # ``dest="cov_source"`` and there is no ``cov`` key at all, so the old
    # spelling read ``None`` for every coverage run and priced it. Measured:
    # ``pytest --cov <scope>`` exited 4 — a documented, legitimate command
    # refused by a typo. This has to read what ``testmon_policy.decide`` reads,
    # or the two disagree about what kind of run this is.
    if options.get("cov_source") or options.get("cov_report"):
        return "coverage, which is false over a subset"
    if not testmon_policy.testmon_installed():
        return "pytest-testmon is not installed"
    return ""


def _bypasses_selection(config) -> bool:
    options = vars(config.option)
    if options.get("testmon_noselect"):
        return True
    return not testmon_policy.decide(config).active


def _refuse_unselected_run(config) -> None:
    if vars(config.option).get(BYPASS_FLAG.lstrip("-").replace("-", "_")):
        # The override implies the thing it overrides, so the one long flag is
        # the whole command -- and so the run still counts as a bypass, and
        # still ends with the banner.
        setattr(config.option, "no-testmon", True)
        return
    if not _bypasses_selection(config):
        return
    permitted = _bypass_permitted(config)
    named_only = node_ids_only(config.args)

    if vars(config.option).get("testmon_noselect"):
        # The one bypass with a product. Priced on whether it has one, not on
        # what it costs — and exempt from the fail-closed rule below, because an
        # empty map is the single best reason to run it.
        peek = _map_peek(config, with_changed_files=True)
        reason = noselect_run_refusal(
            0 if peek is None else peek.mapped_tests,
            unmapped_files=0 if peek is None else len(_unmapped_test_files(config, peek)),
            changed_files=0 if peek is None else len(peek.changed_files),
            permitted_because=permitted,
        )
        if reason:
            _refuse(config, reason, _NOTHING_TO_REPAIR)
        return

    peek = _map_peek(config)
    unpriceable = unpriceable_bypass_refusal(
        None if peek is None else peek.mapped_tests, permitted, node_ids_only=named_only
    )
    if unpriceable:
        _refuse(config, unpriceable, _RECORD_ONE_FIRST)
        return

    scope = _run_scope(config)
    recorded = _recorded_test_seconds(config)
    if vars(config.option).get("no-testmon"):
        seconds = sum(recorded.get(str(name), 0.0) for name in (config.args or ()))
        reason = untraced_run_refusal(named_only, seconds, permitted)
        if reason:
            _refuse(config, reason, _NAME_THE_TESTS)
        return

    seconds = unselected_run_seconds(recorded, scope, testmon_policy.under)
    reason = unselected_run_refusal(seconds, permitted)
    if reason:
        _refuse(config, reason, _RUN_IT_SELECTED)


def _unmapped_test_files(config, peek) -> set[str]:
    """Test files under this run's scope that the map has never recorded."""

    paths = testmon_policy.path_arguments(config)
    return {
        name
        for name in testmon_policy.collect_test_files(config.rootdir, paths)
        if name not in peek.mapped_files
    }


#: What to say instead, when the run was refused on price: the same tests are
#: available, selected, for a fraction of the cost.
_RUN_IT_SELECTED = (
    "runs the same thing, selected, in a fraction of the time. Selection",
    "is not a shortcut -- it is the honest set, from recorded evidence.",
    "",
    "  Everything this branch changed, before merging:  pytest --since",
    "  A red this checkout inherited:                   pytest --rerun-reds",
    "  A map you distrust:                              rm .testmondata && pytest --no-seed-map",
)

#: What to say instead, when there was no map to price against. Naming the
#: reasons matters more than usual here: three of the four are environmental,
#: and an agent that reads "no map" as "the harness is broken" reaches for a
#: wider command, which is the failure this whole gate exists to stop.
_RECORD_ONE_FIRST = (
    "records one first, and costs about what this run would have. Every command",
    "after it is priced against real evidence instead of against silence.",
    "",
    "  A map reads as empty when:",
    "    nothing has been recorded in this checkout yet (a fresh clone or worktree)",
    "    TESTMON_DATAFILE points somewhere that does not exist",
    "    the environment key differs -- QT_QPA_PLATFORM, or an ARRAYSCOPE_* ring",
    "      variable, keys the map by regime and a ring you never ran holds nothing",
    "",
    "  Naming the tests you actually want is never refused, map or no map:",
    "    pytest --no-testmon tests/ui/test_thing.py::test_one",
)

#: What to say instead, when ``--no-testmon`` named more than a few tests. The
#: point is not that the run is expensive -- it is that it records nothing, so
#: the alternative has to be the spelling that leaves the map better.
_NAME_THE_TESTS = (
    "records nothing. Whatever it tells you, the next run plans from the same",
    "map this one started with, so a wide --no-testmon pays twice.",
    "",
    "  It answers exactly one question nothing else can -- is the TRACER doing",
    "  this? -- and that question is asked of named tests:",
    "    pytest --no-testmon tests/ui/test_thing.py::test_one",
    "",
    "  For everything else:",
    "    pytest                     what your tree affects",
    "    pytest --since             everything this branch changed, before merging",
    "    pytest --testmon-noselect  run it all AND leave the map repaired",
)

#: What to say instead, when a ``--testmon-noselect`` sweep would repair nothing.
#: This refusal is unusual in that it is itself the answer: the question the
#: sweep was asking has already been settled by the map it would rewrite.
_NOTHING_TO_REPAIR = (
    "would spend ~222 s re-running every test to write a byte-identical map --",
    "and reorder the run on the way through, which costs this suite about one",
    "spurious failure per sweep. This refusal is the answer: nothing in this",
    "tree differs from what was recorded, so there is nothing a sweep could",
    "find that the map does not already know.",
    "",
    "  A map with holes or a tree that moved is never refused here -- that is",
    "  what this command is for, and it is the right call after a big rebase.",
    "",
    "  If you distrust the map itself rather than its coverage, rebuild it:",
    "    rm .testmondata && pytest --no-seed-map",
    "  (plain `rm .testmondata && pytest` re-seeds from a sibling checkout, so",
    "   it hands you back the very map you distrusted.)",
)


def _refuse(config, reason: str, instead: tuple[str, ...]) -> None:
    named = " ".join(str(argument) for argument in (config.args or ())) or ""
    raise pytest.UsageError(
        "\n".join(
            (
                f"Refused: {reason}.",
                "",
                f"    pytest {named}".rstrip(),
                "",
                *instead,
                "",
                "If you are certain the harness has left you no option, the override is",
                f"    {BYPASS_FLAG}",
                "and it is that long on purpose. Needing it means selection has a blind",
                "spot worth fixing in tests/testmon_policy.py, so report it.",
                "",
                "Read docs/testing/test-selection.md.",
            )
        )
    )


def _run_scope(config) -> tuple[str, ...]:
    """This run's path scope, with "the whole suite" spelled as ``()``."""

    scope = testmon_policy.collected_scope(config)
    testpaths = tuple(config.getini("testpaths") or ())
    return () if set(scope) == set(testpaths) else scope


def _map_peek(config, *, with_changed_files: bool = False):
    """One cached read of the map for every gate. ``None`` if it is unusable.

    Cached on the config because four gates now price the same run — the bypass
    gate, the fail-closed rule above it, the ``--no-testmon`` shape rule and the
    serial gate — and each ``peek`` opens the SQLite map and runs testmon's whole
    stability determination. Sharing the read is also what keeps them from
    disagreeing about what a run costs.

    ``with_changed_files`` is a separate slot rather than the default because
    answering it hashes every file of interest in the tree. Only the
    ``--testmon-noselect`` rule needs it, and that run is about to spend 222 s.
    """

    attribute = "arrayscope_map_peek_full" if with_changed_files else "arrayscope_map_peek"
    cached = getattr(config, attribute, "unset")
    if cached != "unset":
        return cached
    try:
        peek = testmon_policy.peek(
            config.rootdir,
            testmon_policy.resolve_environment(config.getini("environment_expression")),
            tuple(config.getini("testmon_ignore_dependencies")),
            with_changed_files=with_changed_files,
            # Keep the live object so testmon can adopt it instead of building a
            # second one — see ``testmon_policy.reuse_map_read``. Only from the
            # plain slot: the ``--testmon-noselect`` reading is a second, extra
            # read taken for one gate, and two candidates would make "which one
            # does testmon get" a question rather than an answer.
            keep_data=not with_changed_files,
        )
    except Exception:  # a broken map must never be the reason a run cannot start
        peek = None
    setattr(config, attribute, peek)
    return peek


def _recorded_test_seconds(config) -> dict:
    """Per-node-id durations from the map, or ``{}`` if it cannot be read."""

    peek = _map_peek(config)
    return dict(peek.test_seconds) if peek is not None else {}


# --------------------------------------------------------------------------- #
# `-n 0`: the other way to run far more than selection would
# --------------------------------------------------------------------------- #

#: A serial run may cost at most this in recorded test time. Calibrated, not
#: chosen: the most expensive command any ring documents as serial is the
#: ring-2 artifact file at 27.5 s, and the cheapest shape that has ever hurt
#: anybody is ``tests/app`` at 105.9 s (``tests/ui`` is 1164.8 s, the suite
#: 1320 s). 60 s sits in the middle of that empty 3.8x gap, so the threshold is
#: not a judgement call about any real command.
_SERIAL_ABORT_SECONDS = 60.0

#: Targets whose whole point is a serial run, never refused at any price. Note
#: their *recorded* price is 0 by construction — the map is keyed by regime and
#: has never held ``wayland+gpu`` or ``offscreen+stress`` — which is exactly why
#: they are named rather than priced.
_SERIAL_RINGS = (
    "tests/gpu_interaction",  # ring 4, real display
    "tests/stress",  # ring 3
    "tests/ui/test_qt_smoke_artifacts.py",  # ring 2: writes canonical tests/artifacts/
)


def explicit_worker_count(argv):
    """The ``-n`` a person actually typed, or ``None``.

    Read from the command line rather than from ``config.option.numprocesses``,
    and that is the whole design. ``pytest_xdist_auto_num_workers`` returns 0
    for any selection in one file or under 2.5 s, and xdist writes that straight
    into the option before ``pytest_configure`` — so by the time a gate can
    look, the option cannot tell the harness's own in-process decision from a
    person's. A gate that could not tell them apart would refuse exactly the
    runs the harness makes cheap.

    ``auto`` and ``logical`` read as "not a person": ``addopts`` carries
    ``-n auto`` for every run here. Last spelling wins, as argparse would.
    """

    arguments = [str(argument) for argument in (argv or ())]
    found = None
    for index, argument in enumerate(arguments):
        if argument in ("-n", "--numprocesses"):
            value = arguments[index + 1] if index + 1 < len(arguments) else ""
        elif argument.startswith("--numprocesses="):
            value = argument.split("=", 1)[1]
        elif argument.startswith("-n") and len(argument) > 2:
            value = argument[2:]
        else:
            continue
        try:
            found = int(value)
        except ValueError:
            found = None
    return found


def serial_run_refusal(seconds, priceable, *, whole_suite, permitted_because) -> str:
    """Why an explicitly typed ``-n 0`` is refused, or ``""`` to allow it.

    Three things make a *wide* serial run worse than merely slow, and none of
    them apply to the narrow ones:

    * **There is no per-test timeout.** Under xdist a test that never settles
      hangs one worker and the other seven keep printing; serially it is the
      whole run, indefinitely, at 100% CPU. (``faulthandler_timeout`` now turns
      that into a dumped traceback naming the test, which is a floor, not a fix.)
    * **A segfault takes the run, not a worker.** xdist reports a crashed node
      and carries on; ``-n 0`` has nothing to carry on with.
    * **Serial is a different regime with its own reds**, ~25 of them standing.
      A wide ``-n 0`` run is not a de-noised version of the parallel run; it is
      a second, less-tested one, and most of what it produces is not yours.

    Note the direction on unknown cost is the **opposite** of
    :func:`unpriceable_bypass_refusal`, deliberately. Rings 3 and 4 have zero
    recorded seconds by construction and are precisely the runs that must be
    serial, so "unknown = large" would refuse the two rings ground rule 1
    depends on. Scope is the backstop instead: an unpriceable serial run is
    refused only when it names the whole suite, which is the fresh-clone
    ``pytest -n 0`` — 1320 s of first-run tracing in one process.
    """

    if permitted_because:
        return ""
    if seconds > _SERIAL_ABORT_SECONDS:
        return f"`-n 0` would run ~{seconds:.0f} s of recorded test time in one process"
    if not priceable and whole_suite:
        return (
            "`-n 0` names the whole suite and this environment has nothing recorded, "
            "so this is a first traced run of everything, in one process"
        )
    return ""


def _non_offscreen_regime() -> str:
    """Why this run is not the regime the map priced, or ``""``."""

    for name in ("ARRAYSCOPE_GPU_TESTS", "ARRAYSCOPE_STRESS", "ARRAYSCOPE_STRICT_UI"):
        if os.environ.get(name):
            return f"{name} is set, so this is not the offscreen regime the map priced"
    platform = os.environ.get("QT_QPA_PLATFORM", "offscreen")
    if platform != "offscreen":
        return f"QT_QPA_PLATFORM={platform} is not the offscreen regime the map priced"
    return ""


def _serial_permitted(config) -> str:
    """Why an explicit ``-n 0`` is legitimate here, or ``""``.

    Every documented serial use in ``docs/testing/README.md`` has to survive
    this, or the gate breaks the ring rules it is supposed to leave alone.
    ``-s`` is deliberately absent: "I want to see prints" is the reflex, not a
    reason to spend ten minutes.
    """

    options = vars(config.option)
    if options.get(SERIAL_FLAG.lstrip("-").replace("-", "_")):
        return "explicitly forced"
    if os.environ.get("CI"):
        return "CI, whose serial steps are the artifact gate"
    regime = _non_offscreen_regime()
    if regime:
        return regime
    if options.get("usepdb") or options.get("trace"):
        return "a debugger, which xdist cannot host"
    if options.get("collectonly"):
        return "collect-only, which executes nothing"
    scope = _run_scope(config)
    if scope and all(testmon_policy.under(path, _SERIAL_RINGS) for path in scope):
        return "a ring that is serial by design"
    return ""


def _serial_run_seconds(config) -> tuple[float, bool]:
    """``(recorded seconds, priceable)`` for what an explicit ``-n 0`` will run.

    Priced over what the run will *execute*, not over its scope. The middle
    branch is the one that matters: ``pytest -n 0 tests/ui`` after a one-line
    edit runs three tests, and refusing it because ``tests/ui`` *could* cost
    1164 s would be the "unknown = large" mistake pointed the other way.
    """

    peek = _map_peek(config)
    priceable = peek is not None and not peek.empty
    recorded = dict(peek.test_seconds) if peek is not None else {}

    named = [str(argument) for argument in (config.args or ()) if ".py::" in str(argument)]
    if named:
        # Named tests are never the disaster, whatever the map remembers of them.
        return sum(recorded.get(name, 0.0) for name in named), priceable

    scope = _run_scope(config)
    if priceable and testmon_policy.decide(config).active and testmon_policy.selects(config):
        return (
            sum(
                recorded.get(name, 0.0)
                for name in peek.affected_tests
                if testmon_policy.under(name.split("::", 1)[0], scope)
            ),
            priceable,
        )
    return unselected_run_seconds(recorded, scope, testmon_policy.under), priceable


def _refuse_serial_run(config) -> None:
    # Never on a worker: it is in-process by definition, its command line is
    # rebuilt from the controller's paths, and the controller already answered
    # this question for the whole run.
    if _XDIST_WORKER:
        return
    if explicit_worker_count(getattr(config.invocation_params, "args", ())) != 0:
        return
    seconds, priceable = _serial_run_seconds(config)
    reason = serial_run_refusal(
        seconds,
        priceable,
        whole_suite=not _run_scope(config),
        permitted_because=_serial_permitted(config),
    )
    if not reason:
        return
    named = " ".join(str(argument) for argument in (config.args or ())) or ""
    raise pytest.UsageError(
        "\n".join(
            (
                f"Refused: {reason}.",
                "",
                f"    pytest {named}".rstrip(),
                "",
                "runs the same tests eight ways parallel. Serial is not a safer mode",
                "here -- it is a different regime with its own ~25 standing failures, no",
                "per-test timeout, and no worker to survive a segfault, so a wide `-n 0`",
                "run mostly produces reds that are not yours.",
                "",
                "  Serial is for one thing at a time:",
                "    pytest -n 0 tests/ui/test_thing.py::test_one     debug one test",
                "    pytest -n 0 --pdb tests/ui/test_thing.py::test_x drop into pdb",
                "    pytest --rerun-reds                              re-check an inherited red",
                "",
                "  These rings are serial by design and are never refused:",
                "    ARRAYSCOPE_GPU_TESTS=1 ... pytest tests/gpu_interaction -n 0   (ring 4)",
                "    ARRAYSCOPE_STRESS=1 pytest tests/stress -n 0                   (ring 3)",
                "    pytest tests/ui/test_qt_smoke_artifacts.py -n 0                (ring 2)",
                "",
                "If you are certain you need a wide serial run, the override is",
                f"    {SERIAL_FLAG}",
                "and it is that long on purpose.",
                "",
                "Read docs/testing/test-selection.md.",
            )
        )
    )


def _cap_explicit_workers(config) -> None:
    """Clamp an explicit ``-n N`` to the GL-stability ceiling, and say so.

    ``pytest_xdist_auto_num_workers`` owns every worker-count decision and is
    consulted only for ``-n auto``, so an explicit ``-n 16`` walked straight
    past the half-cores cap that keeps concurrent GL context creation from
    segfaulting the driver. Clamped rather than refused, because "more parallel"
    is a benign intent — unlike ``-n 0``, which is a claim about the regime.

    Both ``numprocesses`` and ``tx`` are rewritten: xdist fixes ``tx`` in
    ``pytest_cmdline_main``, which has already run by now, and ``tx`` is what
    the node manager actually builds the pool from.
    """

    if _XDIST_WORKER:
        return
    requested = explicit_worker_count(getattr(config.invocation_params, "args", ()))
    ceiling = worker_ceiling(os.cpu_count() or 2)
    if requested is None or requested <= ceiling:
        return
    config.option.numprocesses = ceiling
    config.option.tx = ["popen"] * ceiling
    config.arrayscope_worker_cap = (
        f"worker sizing: -n {requested} capped to {ceiling} (half the logical cores). "
        "Concurrent GL context creation past that intermittently segfaults the driver "
        "and starves the timing-sensitive UI tests."
    )


def _retry_command(config, workers: int) -> str:
    """The command that was typed, re-sized to ``workers``.

    Rebuilt from the whole command line rather than from ``config.args``, which
    holds only the positional paths. A refusal of ``pytest tests/ui -k montage``
    that answered "run `pytest -n 4 tests/ui`" would be handing back a *wider*
    run than the one it just declined, to somebody who is copying rather than
    reading — and the whole point of naming a command is that it gets copied.

    Any ``-n`` already there is dropped rather than left to fight the new one:
    argparse takes the last spelling, but the reader should not have to know
    that to trust the line.
    """

    arguments = [str(argument) for argument in getattr(config.invocation_params, "args", ())]
    kept: list[str] = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument in ("-n", "--numprocesses"):
            skip_next = True
            continue
        if argument.startswith("--numprocesses=") or (
            argument.startswith("-n") and len(argument) > 2
        ):
            continue
        kept.append(argument)
    return " ".join(["pytest", "-n", str(workers), *(shlex.quote(part) for part in kept)])


def _load_gate_skipped(config) -> bool:
    """Runs the gate does not look at, at all.

    Only the ones that execute no tests. ``--help`` in particular *must* stay
    answerable on a machine in any state: it is where a reader who has just been
    refused goes to find the flag, and a gate that could refuse the explanation
    of itself would be a trap rather than a guard.
    """

    options = vars(config.option)
    return any(options.get(name) for name in _INSPECTION_OPTIONS)


def _load_gate_permitted(config) -> str:
    """Why this run may compete for a busy machine's *cores*, or ``""``.

    Cores only. Whatever this returns, the memory and swap ladders are still
    consulted — see :data:`machine_pressure.NON_NEGOTIABLE`, which is where that
    is enforced rather than here. Being loaded is a speed problem somebody may
    knowingly accept; being out of memory is not a problem anybody accepts, it
    is the thing that took the machine down.

    Two entries, each with a reason it cannot simply be told to wait:

    * the override, whose whole purpose is the load-robustness experiment that
      saturates the machine deliberately;
    * CI, which owns its runner outright — a 2-core runner sitting at a load of
      2.0 is a runner doing its job, not a contended desktop.

    ``ARRAYSCOPE_STRESS`` and the other rings are deliberately *absent*, unlike
    :func:`_serial_permitted`. A ring being serial by design says nothing about
    whether the machine can currently host it, and ring 3 is the heaviest thing
    here — exempting it would exempt the run most able to finish the job.
    """

    options = vars(config.option)
    if options.get(LOAD_FLAG.lstrip("-").replace("-", "_")):
        return "explicitly forced"
    if os.environ.get("CI"):
        return "CI, which owns its runner"
    return ""


def _overloaded_run_refusal(allowance, requested: int) -> bool:
    """Whether a run of ``requested`` processes is over what the machine allows.

    ``requested`` is *processes*, so an in-process run counts as one rather than
    as zero: `-n 0` still imports Qt and still allocates. That matters only in
    the refusing band, where the answer is that nothing may start — which is the
    band the freeze happened in, and the one where "just run it serially" is the
    reflex that would have made no difference.
    """

    if allowance.refuses:
        return True
    return max(1, requested) > allowance.workers


def _refuse_overloaded_run(config) -> None:
    """Decline to add workers to a machine that is already at its limit.

    Placed after :func:`_cap_explicit_workers` on purpose: that clamp is the
    last writer of ``numprocesses``, so this is the first point at which the
    number being judged is the number the pool will actually be built from.

    The refusal names the width to retype rather than silently reducing it.
    Silent reduction was considered and rejected — the failure it is protecting
    against is *several* processes each independently deciding it is alone, and
    a harness that quietly halves gives every one of them the same false
    reassurance that it handled things. A stop that has to be read is also the
    only version that tells the reader the other suites exist.
    """

    if _XDIST_WORKER or _load_gate_skipped(config):
        return
    permitted = _load_gate_permitted(config)
    if permitted == "explicitly forced":
        # Not silently: the override exists for one experiment, and a stale
        # `--load-force-...` left in a shell alias would otherwise excuse
        # contention for weeks without ever saying so.
        config.arrayscope_load_gate = (
            f"machine-pressure gate: CPU ladder off ({LOAD_FLAG}); memory and swap "
            "still apply. You are responsible for what else is running."
        )

    reading = machine_pressure.read_machine()
    ceiling = worker_ceiling(os.cpu_count() or 2)
    allowance = machine_pressure.allow(
        reading,
        ceiling,
        only=machine_pressure.NON_NEGOTIABLE if permitted else machine_pressure.SIGNALS,
    )
    if not allowance.reasons:
        return

    requested = int(getattr(config.option, "numprocesses", 0) or 0)
    if not _overloaded_run_refusal(allowance, requested):
        # Under pressure but within budget — a small selection sized itself
        # below what is left. Worth one line, because the next command in the
        # same shell may not be.
        config.arrayscope_load_gate = (
            f"machine-pressure gate: {allowance.workers} workers is all this machine "
            f"has room for right now — bound by {allowance.reasons[0]}. This run asked "
            f"for {max(1, requested)}, so it stands."
        )
        return

    situation = [f"    {line}" for line in machine_pressure.situation(reading)]
    # Both refusing branches end at "nothing may start", and they are still
    # written separately: the reason is the reader's whole next decision. Out of
    # memory means stop something; twice-oversubscribed cores mean wait. Telling
    # someone with 24 GiB free that a serial run "still allocates" is an
    # argument they can see is wrong, and a gate caught being wrong once is a
    # gate that gets deleted.
    finder = [
        "  Find it:",
        "    pgrep -af 'pytest|python -m pytest'      other suites, including agents'",
        "    ls ~/.claude/worktrees .worktrees        where they are probably running",
    ]
    if allowance.refuses and not allowance.negotiable:
        headline = "Refused: this machine is out of memory, and a test run is how it freezes."
        remedy = [
            "No worker count fits, including `-n 0` -- a serial run still imports Qt",
            "and still allocates. Something has to give the memory back first.",
            "",
            *finder,
        ]
    elif allowance.refuses:
        headline = "Refused: this machine's cores are more than twice oversubscribed already."
        remedy = [
            "Even one more worker here would be taking turns rather than running.",
            "There is nothing to re-run at a narrower width; wait for what is",
            "already going, and this clears on its own.",
            "",
            *finder,
        ]
    else:
        # Same band, two different machines. "Already loaded" in front of a
        # memory figure reads as a harness that did not look at what it printed.
        state = "already loaded" if allowance.negotiable else "low on memory"
        headline = (
            f"Refused: this machine is {state}, and {max(1, requested)} test workers "
            f"is more than it can take."
        )
        remedy = [
            f"Re-run at the width the machine has left, which is {allowance.workers}:",
            "",
            f"    {_retry_command(config, allowance.workers)}",
            "",
            "Most often this is another agent's suite in a sibling worktree. Waiting",
            "for it and then re-running at full width is usually faster than this.",
            "",
            *finder,
        ]

    if allowance.negotiable:
        escape = [
            "This gate exists because concurrent suites froze this machine hard",
            "enough to need a power cycle. If you are certain, the override is",
            f"    {LOAD_FLAG}",
            "and it is that long on purpose.",
        ]
    else:
        # Naming the flag here and letting it not work would be worse than not
        # naming it: the reader would spend the next five minutes deciding the
        # harness is broken. Say instead that there is nothing to type.
        escape = [
            "There is no override for this one, and deliberately nowhere to add it.",
            f"    {LOAD_FLAG}",
            "excuses a busy machine, not an empty one: no worker count fits in memory",
            "that is not there, and this is the state the machine froze in.",
        ]

    raise pytest.UsageError(
        "\n".join(
            (
                headline,
                "",
                *situation,
                "",
                *(f"  bound by {reason}" for reason in allowance.reasons),
                "",
                *remedy,
                "",
                *escape,
                "",
                "Read docs/testing/strategy.md.",
            )
        )
    )


# --------------------------------------------------------------------------- #
# The run with nothing to do
# --------------------------------------------------------------------------- #

#: Options that asked to *see* collection. The fast path answers a question
#: about what would run, not about what exists, so none of them may take it.
_INSPECTION_OPTIONS = (
    "collectonly",
    "showfixtures",
    "show_fixtures_per_test",
    "markers",
    "help",
    "version",
)


def noop_fast_path_refusal(
    *,
    inspecting: bool,
    active: bool,
    selects: bool,
    since: bool,
    whole_suite: bool,
    readable_map: bool,
    workload_files: int | None,
    live_reading: bool,
) -> str:
    """Why this run has to build a session, or ``""`` when it provably need not.

    Every clause is a way a run can still have work in it, and one of them —
    ``workload_files`` — is the one that could hurt. It is
    ``_selection_workload``'s own count, deliberately *not* re-derived here,
    because it already unions the four things that mean "something must run":
    affected tests, the reds this checkout broke, the inherited reds when
    ``--rerun-reds`` asked for them, and **test files the map has never seen**.

    That last one is the trap. A brand-new ``tests/**/test_*.py`` has no map
    entry, so nothing marks it affected and nothing marks it changed; a fast
    path that only asked "is anything affected?" would report the file as fully
    tested having never run it. That is the scope-erosion failure class, the
    worst this repository has (``docs/testing/test-selection.md``), and it is why
    the count rather than the affected set is the discriminator.

    Written as a *reason* rather than a boolean so each clause can be seen
    firing, and so the one that is hard to reason about is not a silent ``and``
    in the middle of an expression.
    """

    if inspecting:
        return "the run asked to see collection"
    if not active:
        return "selection is not active, so the map has nothing to say"
    if not selects:
        return "a selector (-k, -m, --lf, a node id) decides this run, not the map"
    if since:
        return "--since widens the run by an amount only collection can know"
    if not whole_suite:
        return "a scoped run still cleans up the map inside its own scope"
    if not readable_map:
        return "the map is missing, unreadable, or holds nothing for this environment"
    if workload_files is None:
        return "the workload could not be priced"
    if workload_files:
        return f"{workload_files} test file(s) have work in them"
    if not live_reading:
        return "the map reading was not kept, so there is nothing to report from"
    return ""


def pytest_cmdline_main(config):
    """Skip building a session for a run that provably has nothing to run.

    A no-op ``pytest`` is 100% fixed cost — measured, there is no test work in it
    at all — and two thirds of that cost is pytest constructing a session,
    importing 266 test files' worth of conftests and letting testmon's
    ``pytest_ignore_collect`` discard every one of them. Answering from the map
    instead measured **1.25 s → 0.55 s** on this worktree. For scale,
    ``tools/test_selection.py`` — a separate process answering the same question
    without pytest at all — costs 0.35 s, so this lands close to the floor.

    Placed in ``pytest_cmdline_main`` and deliberately *not* ``tryfirst``: xdist
    sizes its worker pool in a ``tryfirst`` implementation of this same hook, and
    that is the one map read this decision is made from. Running after it means
    the read is already paid for.

    What a normal run does here that this one does not, and why each is safe:

    * **testmon's ``sync_db_fs_tests``** garbage-collects the map. Its retain set
      on a zero-affected whole-suite run is every recorded test, so it deletes
      nothing. (A *deleted* test file cannot hide behind this either: its
      fingerprint stops matching, it is reported affected, and the fast path does
      not fire.)
    * **``db.finish_execution``** updates statistics and vacuums orphaned
      fingerprint rows. Nothing was written, so nothing was orphaned.
    * **The red ledger** is asked to forget tests that were not collected; with
      nothing collected it drops nothing, and ``save()`` no-ops on an unchanged
      set.
    * **``protect_map_outside_the_scope``** returns 0 for a whole-suite run.

    What it does **not** skip is the coverage measure and the erosion canary.
    Both cost ~0.07 s and both are load-bearing on exactly this run: an edit that
    affects no test can still delete or rename a covered function, and the
    coverage line is the only thing that says so.

    The two plugin-disable refusals run here too. They normally live in
    ``pytest_configure``, which this path never reaches, and one of them
    (``-p no:cacheprovider``) is the difference between a sentence and an
    ``INTERNALERROR`` — a bare ``pytest`` carrying it must not quietly exit 0.
    The other two refusals are unreachable rather than skipped: the bypass gate
    fires on ``--testmon-noselect`` or on selection being off, and this path
    requires both to be false; and the serial gate prices what an explicit
    ``-n 0`` would *execute*, which here is zero seconds of a priceable map, so
    it permits.
    """

    if _XDIST_WORKER:
        return None
    _refuse_harmful_plugin_disables(config)
    _refuse_pointless_flags(config)
    # Before the ``_map_peek`` below, which is this run's first read of the
    # map. ``-n auto`` already reaches this through
    # ``pytest_xdist_auto_num_workers`` and pays nothing twice; an explicit
    # ``-n`` never fired that hook, so the peek came first and cached "no
    # usable map" for the rest of the run — leaving a checkout that was about
    # to be handed one unable to use it until the *next* run. That is the case
    # a corrupt map lands in, and it is the run that most wants its rebuild to
    # count now.
    _ensure_map_seeded(config)

    options = vars(config.option)
    inspecting = any(options.get(name) for name in _INSPECTION_OPTIONS)
    active = testmon_policy.decide(config).active
    peek = _map_peek(config) if active and not inspecting else None
    workload = _selection_workload(config) if peek is not None else None
    reason = noop_fast_path_refusal(
        inspecting=inspecting,
        active=active,
        selects=active and testmon_policy.selects(config),
        since=config.getoption("since", None) is not None,
        whole_suite=testmon_policy.collected_the_whole_suite(config),
        readable_map=peek is not None and not peek.empty,
        workload_files=None if workload is None else workload[0],
        live_reading=getattr(peek, "data", None) is not None,
    )
    if reason:
        return None
    return _report_nothing_to_run(config, peek)


class _FastPathReporter:
    """The slice of ``TerminalReporter`` the two summary writers actually use.

    ``stats`` is empty and stays empty, which is not a stub: it is the truth this
    run is reporting, and ``tests_executed`` reading zero from it is what keeps
    the coverage *drift* line — a statement about damage a run did — off a run
    that did nothing.
    """

    def __init__(self, writer):
        self._writer = writer
        self.stats: dict = {}

    def write_line(self, line, **markup):
        self._writer.line(line, **markup)


def _report_nothing_to_run(config, peek) -> int:
    """Print what the full path prints for this run, and return its exit code.

    The exit code is the measured one: a no-op selected run exits 0 today, and
    a fast path that exited 5 (``NO_TESTS_COLLECTED``) would break every script
    that treats a clean loop as success.
    """

    from _pytest.config import create_terminal_writer

    config.arrayscope_load_start = load_report.LoadWindow.sample()
    config.arrayscope_started = monotonic()
    config.arrayscope_selection = testmon_policy.decide(config)
    # The reading the gates already took, so the coverage and erosion reports
    # below read the same object testmon's own configure would have handed them.
    config.testmon_data = peek.data
    # This run never reaches testmon's configure, so predicates that ask
    # ``testmon_config`` whether the run narrowed need the answer another way.
    config.arrayscope_fast_path = True

    writer = create_terminal_writer(config)
    reporter = _FastPathReporter(writer)
    writer.sep("=", "test session starts", bold=True)
    for line in _fast_path_header(config, peek):
        writer.line(line)
    writer.line("")
    writer.line("collected 0 items")
    writer.line("")

    # Same order as ``pytest_terminal_summary``, so the two paths read alike.
    # ``_report_gate_run_cost`` is the one omission and it is unreachable here:
    # this path requires selection to be active *and* selecting, which is the
    # negation of what makes a run a bypass.
    busy = load_report.closing_line(config.arrayscope_load_start, load_report.LoadWindow.sample())
    if busy:
        writer.line(busy, bold=True, yellow=True)
    _report_map_erosion(reporter, config)
    _report_function_coverage(reporter, config)
    # The cadence nudge must survive the fast path: a settled tree answered
    # from the map is the run shape a quiet fortnight is made of, so skipping
    # it here would hide the nudge from exactly the checkout that needs it.
    _report_sweep_cadence(reporter, config)
    # The remainder belongs here more than anywhere: this is the run that says
    # "nothing to run", and it is the exact shape that sent a rewritten
    # screen_canvas.py to merge with its only real coverage never collected.
    for line in unaffected_summary(
        0,
        peek.mapped_tests,
        peek.mapped_tests,
        0.0,
        since=None,
        unmeasured=_unmeasured_remainder(config),
    ):
        writer.line(line, bold=True)
    inherited = peek.forced_tests
    if inherited and not testmon_policy.rerun_known_red_tests(config):
        writer.line(
            f"test selection: {len(inherited)} inherited reds not re-run; "
            "`--rerun-reds` runs them.",
            bold=True,
            yellow=True,
        )
    elapsed = monotonic() - config.arrayscope_started
    writer.sep("=", f"no tests ran in {elapsed:.2f}s", bold=True)
    return 0


def _fast_path_header(config, peek) -> list[str]:
    """The header lines a run with nothing to do can still honestly print.

    Everything conditioned on collection is gone, and everything that describes
    *this tree against the map* stays — which is the whole content of the run.
    """

    lines = []
    busy = load_report.opening_line(config.arrayscope_load_start)
    if busy:
        lines.append(busy)
    donor = getattr(config, "arrayscope_selection_seed", None)
    if donor:
        lines.append(f"test selection: seeded this checkout's map from {donor}")
    lines.append(selection_header_line(config.arrayscope_selection.reason, 0, peek.mapped_tests))
    notice = harness_self_edit_notice(_harness_files_changed(config))
    if notice:
        lines.append(notice)
    lines.append(f"rootdir: {config.rootdir}")
    return lines


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Turn change-selection on by default, before testmon reads its options.

    ``tryfirst`` is load-bearing, not decoration: testmon resolves its whole
    configuration from ``config.option`` in its own ``pytest_configure``, so the
    default has to be in place by then. (``pytest_configure`` is a historic
    hook and cannot be wrapped, so anything that needs testmon's *result* reads
    it lazily further down instead.)
    """

    # Harmful first: one of these entries is a crash, and a crash that beats us
    # to the punch is a crash nobody gets a sentence about.
    _refuse_harmful_plugin_disables(config)
    _refuse_pointless_flags(config)
    _refuse_fresh_without_since(config)
    # Seed *before* pricing, and for every run rather than only a selected one.
    # Seeding is a file copy from a sibling checkout, and a bypass run is
    # precisely the run that most needs a real map to be priced against: gating
    # it on ``decision.active`` meant a fresh worktree priced every bypass at
    # zero and let it straight through.
    _ensure_map_seeded(config)
    _refuse_unselected_run(config)
    _refuse_serial_run(config)
    _refuse_since_on_unattributed_map(config)
    _cap_explicit_workers(config)
    # Last of the run-shape gates, and last for a reason: it is the only one
    # that judges the pool as finally sized, and the only one whose verdict can
    # change between two identical commands a minute apart.
    _refuse_overloaded_run(config)
    decision = testmon_policy.decide(config)
    config.arrayscope_selection = decision
    if decision.active:
        # Before testmon's own configure, which is where it resolves the
        # affected set through the module global this patches.
        testmon_policy.install_forgiveness(config.rootdir)
        # Same window, different reason: without this a declared child-process
        # entry point with no filename extension takes the run down from inside
        # testmon at fingerprint time. ``_map_peek`` above has usually installed
        # it already; a run with no map yet has not, and that run still records.
        testmon_policy.install_extensionless_dependency_support()
        # Also before it: this hands testmon the map reading the gates above
        # already took, instead of letting it build and stability-check a second
        # copy of the same thing.
        testmon_policy.reuse_map_read(config, _map_peek(config))
    if decision.active and not decision.explicit:
        config.option.testmon = True
    if decision.active and not _XDIST_WORKER:
        # A selected run is never the run that regenerates the canonical
        # artifacts (that is ``--no-testmon``, or CI's serial step), and it may
        # well execute in-process — see pytest_xdist_auto_num_workers. Keep it
        # off the shared directory the way an xdist worker already is.
        install_artifact_directory(os.environ, _config_root)
    config.arrayscope_load_start = load_report.LoadWindow.sample()
    config.arrayscope_started = monotonic()


def pytest_collection(session):
    """Narrow testmon's "always re-run what failed" exemption to the normal rule.

    Has to happen here: testmon fixes its deselection lists during configure
    (this conftest's own ``pytest_configure`` is too early — testmon's plugin is
    not registered yet), and collection is the last point before both the
    file-level ignore and the item-level deselection consume them. Under xdist
    that means the workers, which is where collection actually happens; the
    controller reports the count from the map instead of from the mutation.
    """

    config = session.config
    testmon_policy.apply_known_red_policy(config, _new_reds(config))
    since = config.getoption("since")
    if since is not None and _selection_is_narrowing(config):
        try:
            report = testmon_policy.apply_since_baseline(
                config,
                since or None,
                _new_reds(config),
                fresh=bool(config.getoption("fresh", False)),
            )
        except testmon_policy.UsageError as error:
            raise pytest.UsageError(str(error)) from error
        config.arrayscope_since = report


def _freeze_collection_seconds(config):
    """Keep one immutable cost map for every worker in this run."""
    snapshot = getattr(config, "_arrayscope_collection_seconds", None)
    if snapshot is None:
        snapshot = tuple(sorted(testmon_policy.file_seconds(config).items()))
        config._arrayscope_collection_seconds = snapshot
    return snapshot


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """Give initial and crash-replacement workers the same pre-run timings."""
    node.workerinput["arrayscope_file_seconds"] = _freeze_collection_seconds(node.config)


def _collection_seconds(config):
    if hasattr(config, "workerinput"):
        # A missing payload still has a deterministic order; reading the live
        # map here could differ from workers that have already started tests.
        return dict(config.workerinput.get("arrayscope_file_seconds", ()))
    return testmon_policy.file_seconds(config)


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    """Order the run for wall time, then put coupled groups back together.

    testmon reorders what it selected shortest-first, for an early first
    failure. This replaces that with longest-file-first, which is what actually
    shortens a `--dist loadfile` run, and restores declaration order inside each
    file — see ``testmon_policy.order_for_makespan`` for both reasons.

    A hook wrapper so this runs after every implementation, testmon's included.
    """

    declaration_order = {item.nodeid: index for index, item in enumerate(items)}
    groups = {}
    for item in items:
        marker = item.get_closest_marker(testmon_policy.COUPLED_MARKER)
        if marker is not None and marker.args:
            groups[item.nodeid] = str(marker.args[0])

    yield

    if not getattr(config, "testmon_config", None):
        return
    items[:] = testmon_policy.order_for_makespan(
        items, _collection_seconds(config), declaration_order
    )
    if groups:
        items[:] = testmon_policy.regroup_coupled_items(items, declaration_order, groups)


def _selection_is_narrowing(config):
    """Whether this run actually left tests out.

    Not merely "testmon is on": under ``--testmon-noselect``, or with ``-k`` /
    a node id, testmon records but deselects nothing, and reporting a gap that
    does not exist would be its own kind of false statement.
    """

    testmon_config = getattr(config, "testmon_config", None)
    return bool(testmon_config) and bool(testmon_config.select)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """Open the red ledger, once testmon's map is readable and before any test.

    Not in ``pytest_configure``: that hook runs ``tryfirst`` here so the default
    is in place before testmon reads it, which means testmon's own configure —
    and therefore ``config.testmon_data`` — does not exist yet.

    Only the controller (or a serial run) opens it. Worker reports are forwarded
    to the controller's ``pytest_runtest_logreport``, so it sees every outcome,
    and one writer means no lock and no lost update.
    """

    global _red_ledger, _flake_ledger, _skip_ledger, _forced_reds, _changed_since_recorded
    global _flake_run_shape, _flake_load_start
    _red_ledger = None
    _flake_ledger = None
    _skip_ledger = None
    _forced_reds = frozenset()
    _changed_since_recorded = frozenset()
    _flake_run_shape = {}
    _flake_load_start = None
    config = session.config
    options = getattr(config, "option", None)
    if not hasattr(config, "workerinput") and (
        getattr(options, "numprocesses", 0) or getattr(options, "tx", ())
    ):
        # xdist starts workers in its trylast sessionstart. Capture before any
        # worker can record durations, including runs using explicit --tx.
        _freeze_collection_seconds(config)
    testmon_config = getattr(config, "testmon_config", None)
    # Both the controller and every worker delete from the map during their own
    # collection, so both need the scope guard — unlike the ledger below, which
    # deliberately has one writer. Installed here because sessionstart is the
    # last point that reliably precedes collection on both.
    if testmon_config:
        testmon_policy.protect_map_outside_the_scope(config)
    if _XDIST_WORKER or not testmon_config or not testmon_config.collect:
        return
    config.arrayscope_since_controller = since_before_the_run(config)
    data = getattr(config, "testmon_data", None)
    if data is None:
        return
    try:
        previous = {
            name: bool((report or {}).get("failed")) for name, report in data.all_tests.items()
        }
        _red_ledger = red_ledger.RedLedger.load(
            testmon_policy.map_path(config.rootdir),
            data.environment,
            previous,
            map_was_populated=bool(previous),
        )
    except Exception:  # a missing ledger costs a re-run, never a wrong result
        _red_ledger = None
    try:
        _skip_ledger = skip_ledger.SkipLedger.load(
            testmon_policy.map_path(config.rootdir), data.environment
        )
    except Exception:  # an unreadable ledger costs a cheap re-attempt, not a result
        _skip_ledger = None
    try:
        _flake_ledger = red_ledger.FlakeLedger.load(
            testmon_policy.map_path(config.rootdir), data.environment
        )
        # Both read now, before a single test can move them. ``_forced_reds`` is
        # the set that will re-run *only* because it is red — one of them
        # passing is the flake observation. ``_changed_since_recorded`` is what
        # the tree already moved under, and a flake mark on any of those stops
        # being evidence the moment this run re-records it.
        _forced_reds = frozenset(testmon_policy.known_red_tests(config))
        _changed_since_recorded = frozenset(data.unstable_test_names or ()) | (
            frozenset(_flake_ledger.marks) - frozenset(data.all_tests or ())
        )
        _flake_run_shape = _run_conditions(config)
        _flake_load_start = getattr(config, "arrayscope_load_start", None)
    except Exception:  # an unreadable flake ledger costs a trusted green, not a result
        _flake_ledger = None


def pytest_collection_finish(session):
    """Serial runs collect here; xdist collects on the workers (see below)."""

    if _red_ledger is not None or _skip_ledger is not None:
        _collected.update(item.nodeid for item in session.items)


def pytest_xdist_node_collection_finished(node, ids):
    """The controller's only view of what was collected, since it never collects."""

    if _red_ledger is not None or _skip_ledger is not None:
        _collected.update(ids)


def pytest_sessionfinish(session):
    if _red_ledger is not None:
        _red_ledger.forget_missing(_collected)
        _red_ledger.save()
    if _flake_ledger is not None:
        _flake_ledger.forget(_changed_since_recorded)
        _flake_ledger.save()
    if _skip_ledger is not None:
        _skip_ledger.forget_missing(_collected)
        _skip_ledger.save()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_logreport(report):
    """Record the dependencies coverage cannot see, before testmon stores them.

    ``report.nodes_files_lines`` is the batch of per-test coverage testmon is
    about to turn into fingerprints. Declared child-process entry points are
    added here so they land in the map as ordinary dependencies — which is what
    lets those tests be *selected on change* instead of run unconditionally.

    A hook wrapper, because plain implementations of this hook run in
    registration order and testmon's (registered during configure) would
    otherwise consume the reports first.
    """

    testmon_policy.inject_out_of_process_dependencies(report, testmon_policy.REPO_ROOT)
    if _red_ledger is not None:
        if report.failed:
            # Any phase counts: a fixture that raises in setup breaks the test
            # just as thoroughly as a failing assertion.
            _red_ledger.record(report.nodeid, failed=True)
        elif report.when == "call" and report.passed:
            _red_ledger.record(report.nodeid, failed=False)
    if (
        _flake_ledger is not None
        and report.when == "call"
        and report.passed
        and report.nodeid in _forced_reds
    ):
        # It was red, nothing it executes changed, and it just passed. There is
        # no code-shaped explanation for that, so the map's green is not
        # evidence about it — see tests/red_ledger.py:FlakeLedger.
        _flake_ledger.observe(report.nodeid, when=_now_iso(), context=_flake_observation_context())
    if _skip_ledger is not None:
        # The rule for what an outcome says about the body having run is the
        # ledger's own, not this hook's: it is the whole subtlety of that file,
        # and a second copy here drifted from it inside one session (an xfail
        # marked unmeasured, which its own guard test caught). One owner.
        #
        # The conditions are this hook's, and they are deliberately the flake
        # ledger's own sampler: a test that skipped last run and ran this one is
        # the same kind of observation as a red that goes green, so "under
        # what?" is the same question and deserves the same answer shape.
        _skip_ledger.observe(report, when=_now_iso(), context=_flake_observation_context())
    yield


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _flake_observation_context() -> dict:
    """The conditions one flake observation happened under.

    Sampled here rather than summarised at the end because the interesting part
    is per-observation: two flips in one run under very different load are two
    different facts, and a single end-of-run figure would flatten them into one.

    Recorded and printed, never acted on. See
    ``tests/red_ledger.py:FlakeLedger`` for why that line is where it is.
    """

    return {**_flake_run_shape, **load_report.observation_context(_flake_load_start)}


def _run_conditions(config) -> dict:
    """The half of an observation's context that cannot change under it."""

    conditions = {"shape": _run_shape(config)}
    workers = _worker_count(config)
    if workers is not None:
        conditions["workers"] = workers
    return conditions


def _worker_count(config) -> int | None:
    """Workers this run has, ``0`` for in-process, ``None`` when unknowable.

    The *effective* number, read after the harness's own cap and xdist's auto
    decision have both landed (see :func:`_cap_explicit_workers`) — the flake
    ledger records what the test actually ran alongside, not what somebody
    typed. An unresolved ``auto`` is reported as unknown rather than as zero: a
    guessed "serial" in this field is worse than a gap, because a reader of the
    ledger cannot tell it from a measurement.
    """

    try:
        return int(getattr(config.option, "numprocesses", 0) or 0)
    except (TypeError, ValueError):
        return None


def _run_shape(config) -> str:
    """Which of the four documented commands this run is.

    Ordered by how much each one weakens the observation, strongest qualifier
    first. ``noselect`` means the test did not re-run *because* it was red —
    everything ran — so the "nothing it executes changed" half of the
    observation came from the map rather than from the run's own shape. Then
    ``--since``, which re-runs reds as a merge gate; then ``--rerun-reds``,
    which asks for exactly them; then the plain loop.

    A label, not a judgement: all four record the same observation.
    """

    if config.getoption("testmon_noselect", False):
        return "noselect"
    if config.getoption("since", None) is not None:
        return "since"
    if config.getoption("rerun_reds", False):
        return "rerun-reds"
    return "loop"


def pytest_report_header(config):
    lines = []
    start = getattr(config, "arrayscope_load_start", None)
    if start is not None:
        busy = load_report.opening_line(start)
        if busy:
            lines.append(busy)
    capped = getattr(config, "arrayscope_worker_cap", None)
    if capped:
        lines.append(capped)
    pressure = getattr(config, "arrayscope_load_gate", None)
    if pressure:
        lines.append(pressure)

    decision = getattr(config, "arrayscope_selection", None)
    if decision is None:
        return lines or None
    if not decision.active:
        lines.append(f"test selection: off ({decision.reason}) — running everything")
        return lines
    if not selection_materialised(config):
        # A run may only claim selection if it can prove it. Everything that
        # degrades in this layer degrades open, so the absence of proof is
        # reported rather than assumed away.
        lines.append(
            "test selection: claimed on, but pytest-testmon did not materialise — "
            "this run selects nothing and records nothing. Report it."
        )
        return lines

    since = config.getoption("since", None)
    if since is not None:
        lines.append(testmon_policy.describe_baseline(config.rootdir, since or None))

    rebuilt = getattr(config, "arrayscope_map_rebuilt", None)
    if rebuilt:
        # Also on stderr, from before there was a reporter to say it to. Said
        # twice on purpose: this is the line that explains why a run that
        # should have selected four tests is running the suite.
        lines.append(f"test selection: the map was corrupt ({rebuilt}) and was rebuilt")
    donor = getattr(config, "arrayscope_selection_seed", None)
    if donor:
        lines.append(f"test selection: seeded this checkout's map from {donor}")
    data = getattr(config, "testmon_data", None)
    lines.append(
        selection_header_line(
            decision.reason,
            len(getattr(data, "unstable_test_names", ()) or ()),
            len(getattr(data, "all_tests", ()) or ()),
        )
    )
    notice = harness_self_edit_notice(_harness_files_changed(config))
    if notice:
        lines.append(notice)
    return lines


def selection_header_line(reason: str, affected: int, mapped: int) -> str:
    """The one line every run opens with about the size of its selection.

    Shared with the no-op fast path (:func:`pytest_cmdline_main`), which prints
    its own header because it never builds a session. Two spellings of this
    sentence would drift, and it is the line an agent reads to decide whether
    the run it is about to get is the run it asked for.
    """

    if not mapped:
        return "test selection: on, but nothing is mapped yet — this run records the map"
    return f"test selection: on ({reason}) — {affected} of {mapped} mapped tests affected"


#: Editing either of these makes selection's own answer suspect, which is a
#: rational doubt and deserves an answer rather than a shrug.
_HARNESS_FILES = ("tests/conftest.py", "tests/testmon_policy.py")


def _harness_files_changed(config) -> frozenset[str]:
    """Which of :data:`_HARNESS_FILES` this working tree has uncommitted edits to.

    From git, not from the map, after trying the map and measuring it. Two
    map-shaped answers were candidates and both are wrong here:

    * ``unstable_files`` is the set of *test* files the affected tests live in.
      ``tests/conftest.py`` holds no tests, so a harness edit never appears
      there however much it changes.
    * ``db.fetch_unknown_files`` reports every file whose recorded fsha row does
      not match, which on this map is a standing set of six — four of them in
      ``arrayscope/`` and untouched by this branch — and re-recording does not
      clear it. A notice that fires on every run forever is worse than no notice.

    Uncommitted is also the sharper question. The doubt this answers is "I am
    editing the thing that decides what runs, right now"; once the edit is
    committed, ``pytest --since`` is the answer and this line names it.
    """

    try:
        import subprocess

        finished = subprocess.run(
            ("git", "status", "--porcelain", "--", *_HARNESS_FILES),
            cwd=str(config.rootdir),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # not a checkout, or no git
        return frozenset()
    if finished.returncode != 0:
        return frozenset()
    return frozenset(
        name
        for name in _HARNESS_FILES
        if any(name in line for line in finished.stdout.splitlines())
    )


def harness_self_edit_notice(changed_files) -> str:
    """One line for the run that changed the thing doing the selecting.

    Editing ``tests/conftest.py`` makes an agent doubt the harness that just
    told it 3941 tests were fine — and that doubt is *correct* in shape, wrong
    only in remedy. Selection fingerprints these files like anything else, so
    every test that recorded them is already in this run; saying so is what
    keeps the doubt from becoming a sweep.
    """

    touched = sorted(name for name in (changed_files or ()) if name in _HARNESS_FILES)
    if not touched:
        return ""
    return (
        f"test selection: you changed the harness itself ({', '.join(touched)}). Selection "
        "fingerprints it like anything else, so every test that recorded it is in the affected "
        "set above. Branch-sized proof: pytest --since"
    )


#: Below this, an exhaustive run cost nothing worth naming an alternative for.
_GATE_NUDGE_SECONDS = 5.0

#: Plugins it is pointless to disable here, and what to say about each. These
#: arrive by habit from other repositories; the flag then makes a command look
#: like it is stabilising something when it is doing nothing at all.
#:
#: Fires while the named plugin is **absent** — the complaint is inertness, so
#: installing the plugin retires the entry by itself.
_POINTLESS_PLUGIN_DISABLES = {
    "randomly": "pytest-randomly is not installed",
}

#: Plugins that this harness genuinely depends on, and what to say about each:
#: ``(module whose presence makes the disable bite — ``None`` for a pytest
#: builtin, which is always there —, what the disable does, what to do instead)``.
#:
#: The predicate is the **opposite** of the table above. That one fires while
#: the plugin is absent, because an inert flag is its whole complaint. These
#: fire while it is present, because here the effect *is* the damage.
#:
#: Both entries were measured on 2026-07-30, and both come from the same
#: imported incantation (``-n 0 -q -p no:cacheprovider``).
_HARMFUL_PLUGIN_DISABLES = {
    "cacheprovider": (
        None,
        "\n".join(
            (
                "it crashes the run.",
                "",
                "pytest-testmon reads config.option['lf'] directly, and that option only",
                "exists while the cache plugin is registered. Disabling it does not make",
                "the run cleaner -- it makes pytest exit 3 with an INTERNALERROR",
                "traceback that looks like this repository is broken. It is not.",
            )
        ),
        "\n".join(
            (
                "`-n 0 -q -p no:cacheprovider` is a habit from repositories where cached",
                "state makes runs unreproducible. Here the cached state IS the feature:",
                "`.testmondata` is what lets `pytest` run the handful of tests your change",
                "reaches instead of all 3941. Clearing it costs ~222 s and buys nothing.",
            )
        ),
    ),
    "pytest-testmon": (
        "testmon",
        "\n".join(
            (
                "it turns off change selection AND recording, with nothing to say so.",
                "",
                "The header would still print `test selection: on ... this run records",
                "the map`, and both halves of that would be false. A run nobody can tell",
                "apart from a selected one is worse than a slow one.",
            )
        ),
        "\n".join(
            (
                "Drop it. `pytest` is already the narrow run. If you want everything and",
                "want the map left repaired, `pytest --testmon-noselect` is the spelling",
                "that says so out loud and has a gate to answer to.",
            )
        ),
    ),
}


def pointless_plugin_disables(plugins, installed) -> list[tuple[str, str]]:
    """``(flag, why)`` for each ``-p no:X`` that could not have had an effect.

    ``installed`` answers "is ``pytest_X`` importable?", injected so the rule
    can be tested without depending on what this environment happens to have.
    A plugin that really does get added makes its entry stop firing by itself.
    """

    found = []
    for plugin in tuple(plugins or ()):
        name = str(plugin)
        if not name.startswith("no:"):
            continue
        disabled = name[3:]
        why = _POINTLESS_PLUGIN_DISABLES.get(disabled)
        if why is None or installed(disabled):
            continue
        found.append((name, why))
    return found


def harmful_plugin_disables(plugins, installed) -> list[tuple[str, str, str]]:
    """``(flag, what it does, what to do instead)`` for each ``-p no:X`` that bites.

    ``installed`` answers "is this module here, so that blocking its plugin
    would actually take something away?", injected for the same reason as in
    :func:`pointless_plugin_disables`. An entry with no module to probe is a
    pytest builtin: there is no environment in which it is absent, so the
    disable always takes effect.

    Deliberately short. Only the *silent* disables belong here — ``-p
    no:xdist`` turns ``addopts`` into a usage error, ``-p no:mark`` is refused
    by pytest itself, and ``-p no:testmon`` misses the entry point name
    entirely. Each of those answers instantly and unambiguously, so listing
    them would buy nothing but a list to keep in sync.
    """

    found = []
    for plugin in tuple(plugins or ()):
        name = str(plugin)
        if not name.startswith("no:"):
            continue
        entry = _HARMFUL_PLUGIN_DISABLES.get(name[3:])
        if entry is None:
            continue
        module, what, instead = entry
        if module is not None and not installed(module):
            continue
        found.append((name, what, instead))
    return found


def _refuse_harmful_plugin_disables(config) -> None:
    """Replace a crash, or a silent bypass, with a sentence.

    This has to run before testmon's own ``pytest_configure``, and it does:
    ``testpaths`` makes this an *initial* conftest, so its ``tryfirst``
    configure is ahead of every plugin's. Verified by measurement — ``-p
    no:cacheprovider -p no:randomly`` printed the repository's refusal rather
    than testmon's ``KeyError``, which is what proves the interception point is
    early enough.

    Why a refusal rather than a warning: the run these flags come from is
    already being typed by somebody who believes the harness is broken. An
    INTERNALERROR confirms that belief, and the next command is wider. This
    message is the only string guaranteed to be read at that moment.
    """

    from importlib.util import find_spec

    harmful = harmful_plugin_disables(
        getattr(config.option, "plugins", ()),
        lambda module: find_spec(module) is not None,
    )
    if not harmful:
        return
    lines: list[str] = []
    for flag, what, instead in harmful:
        lines.extend((f"-p {flag} is refused here: {what}", "", instead, ""))
    lines.extend(
        (
            "    pytest                   the loop -- already narrow, already reproducible",
            "    pytest --since           everything this branch changed, before merging",
            "    pytest -n 0 f.py::test   serial, for the one test you are debugging",
            "",
            "Read docs/testing/test-selection.md rule 5.",
        )
    )
    raise pytest.UsageError("\n".join(lines))


def _refuse_pointless_flags(config) -> None:
    """Abort on a flag that cannot do anything, rather than noting it after.

    This used to print a line in the terminal summary. Escalated to an abort on
    2026-07-30 for a reason specific to who types it: ``-p no:randomly`` is not
    a thing a person reaches for here, because there is nothing in this repo
    that suggests it. It arrives from another codebase's habit, which in
    practice means an agent pattern-matching a command it has seen elsewhere --
    and an agent that types it has demonstrably not read rule 5, so a line it
    would read *afterwards* is aimed at the wrong moment.

    The flag is also worse than inert. It reads as "I stabilised the ordering",
    so a run carrying it looks more controlled than it is, and a flake it did
    nothing to prevent gets attributed to something else.
    """

    from importlib.util import find_spec

    pointless = pointless_plugin_disables(
        getattr(config.option, "plugins", ()),
        lambda name: find_spec(f"pytest_{name}") is not None,
    )
    if not pointless:
        return
    raise pytest.UsageError(
        "\n".join(
            [
                *(f"-p {flag} does nothing here: {why}." for flag, why in pointless),
                "",
                "Drop it. It is not merely inert -- it makes a run look like the",
                "ordering was pinned when it was not, so a flake it never had a",
                "chance of preventing gets blamed on something else.",
                "",
                "Read docs/testing/test-selection.md before the next command: if this",
                "flag came from another repository's habit rather than from this one,",
                "the rest of that habit is probably wrong here too.",
            ]
        )
    )


def _refuse_fresh_without_since(config) -> None:
    """``--fresh`` alone is a request the loop cannot honour, so it is an error.

    It reads as "ignore the map and run everything", and a plain ``pytest`` has
    no reached set to be fresh *about* — the flag would silently do nothing,
    which is the failure mode where somebody believes they ran a sweep. The
    commands that really do run more are named instead.
    """

    if not config.getoption("fresh", False) or config.getoption("since", None) is not None:
        return
    raise pytest.UsageError(
        "\n".join(
            (
                "--fresh only means something with --since: it re-acquires the evidence",
                "for the tests this branch reaches. On its own there is no reached set,",
                "so the flag would quietly do nothing.",
                "",
                "    pytest --since --fresh    re-run everything this branch changed",
                "    pytest --testmon-noselect run and re-record the whole suite",
                "",
                "Read docs/testing/test-selection.md.",
            )
        )
    )


def _refuse_since_on_unattributed_map(config) -> None:
    """A merge gate may not rest on records with missing dependency edges.

    ``--since`` derives "reached" from each test's *recorded* dependencies, so
    a map written under batched sysmon coverage under-scopes the question
    itself: the tests missing an edge to a changed file are exactly the ones
    the gate neither runs nor counts, and it then says "verified". ``--fresh``
    inherits the same hole — it re-runs the reached set, and the reached set is
    the broken part. So the gate refuses outright until the map has one full
    record under per-test attribution, which is the one state in which its
    claim means what it says. The inner loop is deliberately *not* refused:
    it makes no claim beyond the tests it ran, and its closing line names the
    gap on every run (see ``_report_sweep_cadence``).
    """

    if _XDIST_WORKER or config.getoption("since", None) is None:
        return
    if not testmon_policy.decide(config).active:
        return  # nothing selected through the map, nothing to over-trust
    peek = _map_peek(config)
    if peek is None or peek.empty:
        return  # an absent map already degrades to running everything
    record = coverage_map.last_sweep(
        coverage_map.read_sidecar(testmon_policy.map_path(config.rootdir))
    )
    gap = testmon_policy.attribution_gap(record)
    if not gap:
        return
    raise pytest.UsageError(
        "\n".join(
            (
                f"Refused: {gap}.",
                "",
                "A missing dependency edge is invisible to this gate: the test is not",
                "selected, not counted as reached, and reported verified. One re-record",
                "repairs the map and retires this refusal:",
                "",
                "    pytest --testmon-noselect     (~4 min, records every edge)",
                "",
                "then re-run:  pytest --since",
                "",
                "Read docs/testing/test-selection.md (per-test dependency batches).",
            )
        )
    )


def _report_gate_run_cost(terminalreporter, config) -> None:
    """Close every selection-bypassing run with what it cost and a defect ask.

    Fires for the ones that got through — the explicitly forced override, CI,
    and a coverage run — because the point is not to scold the flag but to keep
    the bypass *visible*. A bypass that was genuinely necessary means selection
    could not see something, which is a defect in ``testmon_policy`` and worth
    a report; a bypass that was habit is worth noticing at the moment its price
    is on screen.
    """

    if not _bypasses_selection(config):
        return
    started = getattr(config, "arrayscope_started", None)
    if started is None:
        return
    elapsed = monotonic() - started
    reason = bypass_reason(
        _bypass_permitted(config), records=bool(vars(config.option).get("testmon_noselect"))
    )
    terminalreporter.write_line(
        f"Bypassed selection ({reason}) and ran {elapsed:.0f} s.",
        bold=True,
        yellow=True,
    )
    terminalreporter.write_line(BYPASS_BANNER, bold=True, red=True)


def bypass_reason(permitted_because: str, *, records: bool = False) -> str:
    """The parenthesised half of the closing line, never empty.

    A run allowed by a *rule* rather than by a permission had no string to put
    there, so the line read ``Bypassed selection ()`` — which looks like a
    formatting bug rather than the statement it is meant to be, and the whole
    point of the line is to keep the bypass visible. The two rules say different
    things: a cheap ``--no-testmon`` was not worth the argument, and a
    ``--testmon-noselect`` sweep bought a repaired map with its four minutes.
    """

    if permitted_because:
        return permitted_because
    if records:
        return "it re-records the map, so the time bought something"
    return "cheap enough that retyping the command would cost more"


def pytest_terminal_summary(terminalreporter, config):
    """State the size of the gap between "this run" and "the suite".

    A selected run that ends green has not said the suite is green, and the
    difference is invisible unless something says it out loud.

    Both counts come from the map, not from collected items or from the
    deselection the workers performed: under xdist the controller neither
    collects nor applies the known-red policy, so anything read from this
    process's plugin state would report zero here and something else on a
    serial run.
    """

    start = getattr(config, "arrayscope_load_start", None)
    if start is not None:
        busy = load_report.closing_line(start, load_report.LoadWindow.sample())
        if busy:
            terminalreporter.write_line(busy, bold=True, yellow=True)

    # Before the coverage report, which re-reads the sidecar this may write.
    _report_map_erosion(terminalreporter, config)
    _report_function_coverage(terminalreporter, config)
    _report_sweep_cadence(terminalreporter, config)
    _report_gate_run_cost(terminalreporter, config)

    if not _selection_is_narrowing(config):
        return
    data = getattr(config, "testmon_data", None)
    mapped = len(getattr(data, "all_tests", ()) or ())
    unaffected = len(getattr(data, "stable_test_names", ()) or ())
    if not mapped or not unaffected:
        return
    since = (
        # In order of authority: what this process actually applied, what the
        # controller read before anything ran, and a late recompute for a shape
        # neither of the first two covers.
        getattr(config, "arrayscope_since", None)
        or getattr(config, "arrayscope_since_controller", None)
        or _since_on_the_controller(config)
    )

    new_reds = _red_ledger.new_reds if _red_ledger is not None else _new_reds(config)
    inherited = testmon_policy.known_red_tests(config) - new_reds
    # ``unaffected`` counts the *stable* tests, so the correction has to be over
    # stable tests too. Two things pull one back in: --since running it despite
    # the map, and the flag that re-runs inherited reds. Intersecting is not
    # decoration — --since reaches unstable tests as well, and subtracting those
    # from a count they were never in used to underreport the gap.
    stable = frozenset(getattr(data, "stable_test_names", ()) or ())
    ran_anyway: set[str] = set()
    if since is not None:
        ran_anyway |= set(since.run_tests) & stable
    if testmon_policy.rerun_known_red_tests(config):
        ran_anyway |= inherited & stable
    skipped = max(0, unaffected - len(ran_anyway))
    sweep = sum(_recorded_test_seconds(config).values())
    for line in unaffected_summary(
        tests_executed(terminalreporter),
        skipped,
        mapped,
        sweep,
        since=since,
        unmeasured=_unmeasured_remainder(config),
    ):
        terminalreporter.write_line(line, bold=True)
    if inherited and not testmon_policy.rerun_known_red_tests(config):
        terminalreporter.write_line(
            f"test selection: {len(inherited)} inherited reds not re-run; "
            "`--rerun-reds` runs them.",
            bold=True,
            yellow=True,
        )
    for line in broken_here_lines(new_reds):
        terminalreporter.write_line(line, bold=True, red=True)


def tests_executed(terminalreporter) -> int:
    """How many tests this session actually ran, from the reporter's own tally.

    Not from collected items: under xdist the controller never collects, and not
    from the map either, which describes the tests that *exist*. The tally is
    what the "N passed" line is built from, so the two cannot disagree.
    """

    stats = getattr(terminalreporter, "stats", None) or {}
    return sum(len(stats.get(key, ()) or ()) for key in ("passed", "failed", "error"))


def _unmeasured_remainder(config) -> tuple[str, ...]:
    """Mapped tests whose record still describes a gate rather than their code.

    These are the ones the closing sentence may not count as "unaffected by
    construction": their recorded dependency set says what the gate touched, so
    it cannot say whether your change is in the code they would execute. Naming
    them is what turns a false completeness claim into a true statement with a
    stated remainder — see ``tests/skip_ledger.py``.

    Read from the **live** ledger where there is one, and that is the whole
    subtlety. "Reached" is not "measured": ``--since`` reaches every mark, and
    offscreen every one of them skips again, which leaves the change exactly as
    unmeasured as before — an earlier draft of this subtracted what the run
    reached and reported an empty remainder for that case, which is the same
    false claim one step along. The live ledger is the honest set because it is
    the outcome ledger: a test that ran for real this run has already had its
    mark cleared by ``SkipLedger.observe``, and one that skipped again still
    carries it. So the remainder empties exactly when the tests are
    measured, on the display where their gate opens, and not a run before.
    """

    data = getattr(config, "testmon_data", None)
    if data is None:
        return ()
    try:
        environment = data.environment
        live = _skip_ledger
        if live is not None and live.environment == environment:
            marks = set(live.skipped)
        else:
            # No live ledger, or one belonging to a different environment — the
            # no-op fast path is the first case, and it is the exact run this
            # remainder exists for: it returns from ``pytest_cmdline_main``
            # before ``pytest_sessionstart``, so nothing has opened a ledger.
            # The environment check is what stops a second in-process
            # ``pytest.main()`` reading the previous session's marks, the way
            # ``reuse_map_read`` guards its own carried state.
            marks = skip_ledger.read_skips(testmon_policy.map_path(config.rootdir), environment)
        return tuple(sorted(marks & set(data.all_tests or ())))
    except Exception:  # a report line is not worth failing a run over
        return ()


def unmeasured_lines(unmeasured) -> list[str]:
    """Name the tests whose record cannot answer the question, or say nothing.

    Grouping is ``skip_ledger.remainder_by_file``'s, not this function's: three
    places withdraw the completeness claim and none of them may disagree about
    what the remainder *is*.

    This one names the *command* too. A remainder line that only reported a
    number would be one more thing to read past — which is precisely what the
    grey "inherited reds not re-run" line was measured to become — and the whole
    reason this exists is that the previous wording sent a branch to merge
    without its only real coverage.
    """

    if not unmeasured:
        return []
    ranked = skip_ledger.remainder_by_file(unmeasured)
    limit = skip_ledger.REMAINDER_FILES_NAMED
    named = ", ".join(f"{name} ({count})" for name, count in ranked[:limit])
    if len(ranked) > limit:
        named += f", +{len(ranked) - limit} more file(s)"
    tests = "test" if len(unmeasured) == 1 else "tests"
    return [
        f"  Unmeasured remainder: {len(unmeasured)} {tests} skipped when the map recorded them, "
        "so their recorded",
        f"  dependencies describe a gate, not their code — {named}.",
        "  `pytest --since` reaches them; running them where their gate opens is what measures them.",
    ]


def unaffected_summary(ran, skipped, mapped, sweep_seconds, since, unmeasured=()) -> list[str]:
    """The closing sentence about the gap between this run and the suite.

    This is the highest-leverage string in the file, and it used to say the
    wrong thing about the right number::

        test selection: 3941/3941 mapped unaffected, not run. Pre-merge: `--since`.

    That states a **deficit**. An agent reading "3941 of 3941 not run" concludes
    it has been handed an incomplete result and that finishing the job is its
    responsibility -- which is how a selected run ends in a sweep. The number is
    unchanged here; the claim is inverted. Nothing was withheld: the tests that
    did not run could only have re-confirmed fingerprints that still hold, and
    saying what that would have cost is what makes the trade visible.

    Kept as a pure function because the sentence is the feature. It is the only
    thing between a green selected run and somebody deciding to "make sure".

    ``unmeasured`` is the one thing that can withdraw the completeness half of
    the claim, and it had to exist because the claim was being made falsely. A
    test that *skipped* when the map recorded it has a dependency set describing
    the gate that turned it away, so "your change is not in it" is not a
    statement about that test's code at all — see ``tests/skip_ledger.py`` for
    the branch this cost. Both sentences below therefore stop at what the
    evidence supports and name the remainder, rather than rounding it into
    "unaffected by construction" or "a result, not a gap".
    """

    lines = []
    remainder = len(unmeasured)
    if ran:
        tests = "test" if ran == 1 else "tests"
        lines.append(
            f"test selection: ran the {ran} {tests} your tree reaches. The other {skipped} of "
            f"{mapped} are unaffected by construction — their recorded"
        )
        closing = (
            "This is the complete affected set, not a sample."
            if not remainder
            else "That covers every test the map can speak for."
        )
        lines.append(
            f"  fingerprints still hold, so running them could only re-confirm them "
            f"(~{sweep_seconds:.0f} s). {closing}"
        )
    else:
        lines.append(
            "test selection: nothing to run — no line your working tree changed is executed "
            f"by any of the {mapped} recorded tests."
        )
        if not remainder:
            lines.append("  That is a result, not a gap.")
    lines.extend(unmeasured_lines(unmeasured))
    if since is not None:
        # The verification report, always — including when it widened by zero.
        # "Your branch reaches 65 tests and 65 are already proven" is the most
        # useful thing --since can say, and the old line, printed only when the
        # run grew, could not say it at all.
        lines.append(f"  {since.sentence()}")
    else:
        lines.append("  Branch-sized question, before merging:  pytest --since")
    return lines


def broken_here_lines(new_reds) -> list[str]:
    """Name the tests this checkout broke, and the two commands that follow.

    The heading was already good; what was missing was the next command. A red
    an agent cannot explain is exactly the moment it reaches for a sweep, and a
    sweep answers neither question -- "is it still broken" is one node id, and
    "is it even mine" is a ledger question, not a test-running one.
    """

    reds = sorted(new_reds or ())
    if not reds:
        return []
    lines = [
        f"BROKEN HERE ({len(reds)}): these were passing in this checkout and are not; "
        "they run on every selected run until they pass again.",
    ]
    lines.extend(f"    {nodeid}" for nodeid in reds)
    lines.append(f"    Re-check just this one:  pytest {reds[0]}")
    lines.append(
        "    Not yours (stale worktree, stacked branch)?  "
        "python tools/test_selection.py accept-reds"
    )
    return lines


def since_before_the_run(config):
    """The ``--since`` report as the tree stood when the session opened.

    Called from ``pytest_sessionstart``, and *when* is the whole point. Tests
    re-record their fingerprints as they finish, so the same query asked at
    summary time describes a tree that no longer exists — measured, a
    ``--since --fresh`` run printed "ran the 233 tests your tree reaches"
    directly above "branch reaches 185": one paragraph, one run, two trees.

    This is the controller's only chance to ask, because xdist owns
    ``pytest_collection`` and this conftest's implementation never runs there.
    """

    if config.getoption("since", default=None) is None:
        return None
    return _since_on_the_controller(config)


def _since_on_the_controller(config):
    """The ``SinceReport`` for a run whose collection happened somewhere else.

    ``config.arrayscope_since`` is written during ``pytest_collection``, which
    under xdist runs on the workers — so the controller, the only process with a
    terminal, had neither the ``--since`` line nor its correction to the
    "unaffected and did not run" count. The gap the run had already closed was
    reported as still open, and the flag looked like it had done nothing.

    Recomputed rather than forwarded: the query is read-only and the workers
    have no channel to hand a number back on. It costs a couple of seconds, and
    ``--since`` is the pre-merge step, not the inner loop. Called through
    :func:`since_before_the_run`, because *when* it runs changes the answer.

    The partition has to be recomputed the same way too, through the same
    function the workers applied — the counts in the printed sentence are the
    ones the run acted on, or they are fiction.
    """

    if config.getoption("since", default=None) is None:
        return None
    data = getattr(config, "testmon_data", None)
    if data is None:
        return None
    try:
        ref, merge_base = testmon_policy.resolve_baseline(
            config.rootdir, config.getoption("since") or None
        )
        checksums = testmon_policy.baseline_method_checksums(config.rootdir, merge_base)
        if checksums is None:
            return None
        reached = testmon_policy.tests_reached_since_baseline(data, checksums)
        if reached is None:
            return None
        return testmon_policy.since_verification(
            ref,
            reached,
            config,
            _new_reds(config),
            fresh=bool(config.getoption("fresh", False)),
        )
    except Exception:  # a summary line is not worth failing a finished run over
        return None


#: A map that loses more than this share of its recorded tests between runs has
#: almost certainly eroded rather than shrunk honestly. Deliberately loose: a
#: normal shrink is one deleted test file, and the defect this watches for took
#: 1261 of 3941 in a single scoped run.
_MAP_EROSION_FRACTION = 0.05


def map_high_water(mapped, recorded, authoritative) -> int:
    """The mark to keep for this environment, given what this run saw.

    Strictly rising, except after a run that re-recorded the whole suite without
    selecting. That run's count *is* the truth — there is no scope for it to
    have missed anything — so it is the one thing allowed to lower the mark, and
    it is what stops an honest deletion from warning forever.
    """

    if authoritative and mapped:
        return int(mapped)
    return max(int(mapped or 0), int(recorded or 0))


def map_erosion_warning(mapped, high_water) -> str:
    """Say it out loud when the map has quietly lost a chunk of itself.

    This is the only thing that would catch the fail-open degradations in
    ``testmon_policy`` — ``protect_map_outside_the_scope`` silently returning 0
    after a testmon upgrade, say. That is not hypothetical: the same defect
    class eroded ``main``'s map to 2680 of 3941 tests, which then read as a
    15-point phantom coverage regression, because the map's size is the
    denominator of that figure.

    Nothing else notices. The map does not complain, the header still says
    ``test selection: on``, and every report — ``pytest``, ``pytest --since``,
    ``tools/test_selection.py`` — agrees that the missing tests are unaffected.
    """

    if not high_water or not mapped:
        return ""
    lost = int(high_water) - int(mapped)
    if lost <= int(high_water) * _MAP_EROSION_FRACTION:
        return ""
    return (
        f"test selection: the map lost {lost} of {high_water} recorded tests since this "
        "checkout's high-water mark. That is the erosion signature "
        "(docs/testing/test-selection.md). Repair: pytest --testmon-noselect"
    )


def _report_map_erosion(terminalreporter, config) -> None:
    """Keep, and check, the map's high-water mark in the coverage sidecar."""

    if _XDIST_WORKER:
        return
    data = getattr(config, "testmon_data", None)
    environment = getattr(data, "environment", None)
    mapped = len(getattr(data, "all_tests", ()) or ())
    if not environment or not mapped:
        return

    map_path = testmon_policy.map_path(config.rootdir)
    sidecar = coverage_map.read_sidecar(map_path)
    sizes = coverage_map.map_sizes(sidecar)
    recorded = sizes.get(environment, 0)

    warning = map_erosion_warning(mapped, recorded)
    if warning:
        terminalreporter.write_line(warning, bold=True, red=True)

    authoritative = bool(vars(config.option).get("testmon_noselect")) and (
        testmon_policy.collected_the_whole_suite(config)
    )
    mark = map_high_water(mapped, recorded, authoritative)
    if mark != recorded:
        coverage_map.write_sidecar(
            map_path,
            structure=coverage_map.cached_structure(sidecar),
            baselines=coverage_map.baselines(sidecar),
            # Preserved, not recomputed: dropping the key here would silently
            # strip every retained revision and turn forgiveness off.
            revisions=coverage_map.retained_revisions(sidecar),
            map_sizes={**sizes, environment: mark},
        )


#: How stale an exhaustive sweep may get before the loop mentions it. Two
#: clocks, because either one can run out on its own: a checkout can sit for
#: three weeks with four commits, or take forty in an afternoon.
_SWEEP_STALE_DAYS = 21
_SWEEP_STALE_COMMITS = 40


def cadence_nudge(record: dict, days: float | None, commits: int | None) -> str:
    """One quiet line when the blind spots have gone unowned too long, else ``""``.

    Selection is honest about what it cannot see — child processes, non-Python
    inputs, Python read as data rather than executed — and the documentation
    used to hand all three to CI, "swept exhaustively on every push". Measured
    on 2026-07-30: CI has run **seven times in the life of this repository**,
    last on 2026-07-25 with ``main`` 247 commits ahead of it. There is no branch
    protection and no required check. So those blind spots have had no owner at
    all, and nothing anywhere said so.

    This is the local backstop, and it is deliberately the smallest one that
    works: no gate, no refusal, no colour. A sweep is four minutes and the
    person reading this is mid-loop — the line has to be droppable, or it
    becomes noise and then becomes invisible. It states a fact and a command,
    once the fact is worth stating.

    An anchor with ``swept`` false says a different thing from a real record and
    is worded as such: the checkout has never swept, and the clock runs from
    when it first looked rather than from a sweep that never happened.
    """

    if not record:
        return ""
    stale_days = days is not None and days > _SWEEP_STALE_DAYS
    stale_commits = commits is not None and commits > _SWEEP_STALE_COMMITS
    if not (stale_days or stale_commits):
        return ""
    age = f"{days:.0f} days" if days is not None else "an unknown time"
    distance = f", {commits} commits" if commits is not None else ""
    if record.get("swept"):
        what = f"blind spots last swept exhaustively {age} ago{distance}"
    else:
        what = f"blind spots have never been swept exhaustively here ({age}{distance})"
    return f"{what} — pytest --testmon-noselect (~4 min) when convenient."


def _run_swept_exhaustively(config, ran: int) -> bool:
    """Whether this run really did execute every test there is.

    Both shapes count and nothing else does: ``--testmon-noselect`` (which runs
    everything *and* re-records) and a run where the policy turned selection off
    altogether (``--no-testmon``, ``--cov``, CI). A manual selector narrows the
    run without narrowing its path arguments, so it is excluded explicitly —
    ``pytest -k render`` collects the whole suite and runs a handful of it.
    """

    if not ran or not testmon_policy.collected_the_whole_suite(config):
        return False
    options = vars(config.option)
    if options.get("keyword") or options.get("markexpr") or options.get("lf"):
        return False
    if options.get("testmon_noselect"):
        return True
    decision = getattr(config, "arrayscope_selection", None)
    return decision is not None and not decision.active


def _report_sweep_cadence(terminalreporter, config) -> None:
    """Record every exhaustive run, and mention it when the last one aged out."""

    if _XDIST_WORKER:
        return
    map_path = testmon_policy.map_path(config.rootdir)
    sidecar = coverage_map.read_sidecar(map_path)
    record = coverage_map.last_sweep(sidecar)

    def remember(swept: bool) -> None:
        # Every other field passed through: this writer owns one key, and a
        # write that dropped another feature's state is the defect cbf3f254
        # fixed. See ``coverage_map.write_sidecar``.
        coverage_map.write_sidecar(
            map_path,
            structure=coverage_map.cached_structure(sidecar),
            baselines=coverage_map.baselines(sidecar),
            revisions=coverage_map.retained_revisions(sidecar),
            map_sizes=coverage_map.map_sizes(sidecar),
            sweep={
                "sha": testmon_policy.head_sha(config.rootdir),
                "at": _now_iso(),
                "swept": swept,
                # Only code that carries the per-test-batch pin can write this
                # key, so its presence on a real sweep is proof the whole map
                # was recorded with complete dependency attribution. Absent on
                # the anchor: swept=False already says "no full record exists".
                **({"attributed": True} if swept else {}),
                # Narrower than its neighbour on purpose. A sweep also counts
                # when the policy turned selection *off* (``--no-testmon``,
                # ``--cov``, CI), and those runs open no ledger — so they
                # witness every skip and write none of them down. Stamping them
                # would claim exactly the knowledge this key exists to certify.
                **({"skips": True} if swept and _skip_ledger is not None else {}),
            },
        )

    if _run_swept_exhaustively(config, tests_executed(terminalreporter)):
        remember(swept=True)
        return
    # Only the inner loop nags. --since is already the wider question, and a
    # bypassed run has its own closing paragraph. The fast path counts as
    # narrowing by definition — it exists because selection narrowed the run to
    # nothing — but it exits before testmon configures, so the
    # ``testmon_config`` this predicate reads never comes to exist there.
    narrowing = _selection_is_narrowing(config) or getattr(config, "arrayscope_fast_path", False)
    if config.getoption("since", None) is not None or not narrowing:
        return
    if not record:
        # A new checkout must not open with a nag, so its first bare run starts
        # the clock instead of complaining about a sweep it never had time for.
        remember(swept=False)
        return
    gap = testmon_policy.attribution_gap(record)
    if gap:
        # No clock: an under-attributed map is under-selecting *now*, on every
        # run, and the loop is where the missing edges silently skip tests.
        terminalreporter.write_line(
            f"test selection: {gap} — selection is under-scoped until one "
            "re-record: pytest --testmon-noselect (~4 min).",
            bold=True,
            yellow=True,
        )
        return
    line = cadence_nudge(
        record,
        _days_since(record.get("at")),
        testmon_policy.commits_since(config.rootdir, str(record.get("sha") or "")),
    )
    if line:
        terminalreporter.write_line(line)


def _days_since(when) -> float | None:
    from datetime import datetime, timezone

    try:
        recorded = datetime.fromisoformat(str(when))
    except (TypeError, ValueError):
        return None
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - recorded).total_seconds() / 86400.0


def _report_function_coverage(terminalreporter, config):
    """Say what the map now knows about executed functions — only when it moved.

    The map records which AST blocks each test executed, so the union over every
    recorded test is a coverage figure this run has already paid for. See
    ``tests/coverage_map.py`` for what the number is and, more importantly, what
    it is not: it is not `coverage.py`'s line percentage, the two disagree by up
    to 36 points per file in both directions, and they must never be quoted as
    the same thing.

    Printed only when the covered set *changed*, because on a clean tree it
    cannot: an inner loop that touches nothing coverage-relevant would otherwise
    grow one more line saying so every time.

    Skipped for a run that did not collect the whole suite, and only for that.
    A scoped run leaves out-of-scope affected tests unrecorded, so both the
    figure and its delta would move for a reason that has nothing to do with the
    code. Deselection itself is *not* a reason to withhold it: under ``-k`` or
    ``--testmon-noselect`` the whole suite is still collected, nothing is dropped
    from the map, and ``--testmon-noselect`` is precisely the run that leaves the
    map — and therefore this figure — at its most complete.
    """

    if _XDIST_WORKER:
        return
    decision = getattr(config, "arrayscope_selection", None)
    if decision is None or not decision.active:
        return  # no map to read: --no-testmon, --cov, or testmon not installed
    if not testmon_policy.collected_the_whole_suite(config):
        return
    environment = getattr(getattr(config, "testmon_data", None), "environment", None)
    if not environment:
        return

    map_path = testmon_policy.map_path(config.rootdir)
    sidecar = coverage_map.read_sidecar(map_path)
    structure = coverage_map.Structure(
        coverage_map.cached_structure(sidecar), coverage_map.retained_revisions(sidecar)
    )
    reach = coverage_map.measure(config.rootdir, map_path, environment, structure=structure)
    if reach is None or not reach.total:
        return
    # Not for the figure — for selection. Every file the map depends on needs its
    # revision retained here, because this is the moment the tree and the map
    # agree, and the next edit is compared against it. See
    # ``testmon_policy.install_forgiveness``.
    coverage_map.track_revisions(config.rootdir, map_path, environment, structure)

    baselines = coverage_map.baselines(sidecar)
    drift = coverage_map.drift(reach, baselines.get(environment))
    roots = "/, ".join(coverage_map.source_roots(config.rootdir))
    if drift is None:
        # First run in this checkout: record what it inherited and say so once.
        # Anything that moves from here is this checkout's own doing — the same
        # baseline rule tests/red_ledger.py uses for reds.
        baselines[environment] = coverage_map.baseline_from(reach)
        terminalreporter.write_line(
            f"coverage: baseline recorded for this checkout ({roots}/).",
            bold=True,
        )
    elif drift.moved and tests_executed(terminalreporter):
        # Only when something ran. A drift line under `no tests ran in 0.38s`
        # reads as damage this run did, and nothing this run did could have
        # caused it -- it is drift against a baseline the checkout inherited,
        # and it was the second of four invitations to sweep that a bare
        # `pytest` used to print in five lines.
        points = reach.percent - (100.0 * drift.baseline_covered / max(1, reach.total))
        line = (
            f"coverage: +{len(drift.gained)} -{len(drift.lost)} functions ({points:+.1f} pts) "
            "vs the baseline this checkout inherited"
        )
        if drift.lost:
            line += " — `tools/test_selection.py coverage` names the lost ones"
        terminalreporter.write_line(f"{line}.", bold=True, yellow=bool(drift.lost))

    # The option, not ``config.arrayscope_since``: that is set during collection,
    # which under xdist happens on the workers, so the controller — the only
    # process with a terminal — never sees it.
    since = config.getoption("since", default=None)
    if since is not None:
        _report_new_code_coverage(terminalreporter, config, environment, map_path, structure, since)

    if structure.parsed or environment not in coverage_map.baselines(sidecar):
        # ``map_sizes`` is passed through rather than recomputed: the erosion
        # canary ran first and may already have moved the mark, and a write that
        # dropped the key would quietly retire the canary.
        coverage_map.write_sidecar(
            map_path,
            structure=structure.payload(),
            baselines=baselines,
            revisions=structure.retained(),
            map_sizes=coverage_map.map_sizes(sidecar),
        )


def _report_new_code_coverage(terminalreporter, config, environment, map_path, structure, ref):
    """On a ``--since`` run, name the new code no test executes.

    This is the one comparison against the branch point that is exact. A global
    "percent since main" is not available and is deliberately not faked: for a
    file this branch edited, the map holds only the current revision's
    fingerprints, so the baseline revision's coverage cannot be recovered from it
    (see the note in ``tests/coverage_map.py``).
    """

    try:
        ref, merge_base = testmon_policy.resolve_baseline(config.rootdir, ref or None)
    except testmon_policy.UsageError:
        return
    fresh = coverage_map.new_code(
        config.rootdir, map_path, environment, ref, merge_base, structure=structure
    )
    if fresh is None or not fresh.total:
        return
    if not fresh.uncovered:
        terminalreporter.write_line(
            f"coverage: {fresh.total}/{fresh.total} new functions since {ref} are tested.",
            bold=True,
        )
        return
    terminalreporter.write_line(
        f"coverage: {len(fresh.uncovered)}/{fresh.total} new functions since {ref} have no test:",
        bold=True,
        yellow=True,
    )
    for function in fresh.uncovered[:12]:
        terminalreporter.write_line(f"    {function}", yellow=True)
    if len(fresh.uncovered) > 12:
        terminalreporter.write_line(
            f"    ... +{len(fresh.uncovered) - 12} more "
            "(`tools/test_selection.py coverage --since`)",
            yellow=True,
        )


# Keep direct-import test modules from replacing the real package in sys.modules.
import contextlib

import arrayscope  # noqa: F401


def _arrayscope_modules():
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "arrayscope" or name.startswith("arrayscope.")
    }


def _has_replaced_ancestor(name, replaced_names):
    parent_name = name
    while "." in parent_name:
        parent_name = parent_name.rpartition(".")[0]
        if parent_name in replaced_names:
            return True
    return False


@pytest.fixture(autouse=True)
def _restore_arrayscope_module_identity():
    """Keep ``arrayscope`` module (and class) identities stable across tests.

    A few tests purge ``arrayscope.*`` from ``sys.modules`` and re-import to
    exercise fresh-import behavior (package callability, lazy Qt binding, no
    eager pyqtgraph import). Without restoration the re-imported modules define
    brand-new class objects, so a later ``isinstance`` or ``pytest.raises``
    check in an unrelated test compares against the *old* class identity and
    fails even though the production code is correct (e.g. ``LazySourceArray``,
    ``SourceReadRefused``). Snapshot the arrayscope module objects and their
    parent-package bindings before each test. Restore replaced modules while
    retaining legitimate first imports so module identity cannot leak forward.
    """

    snapshot = _arrayscope_modules()
    package_attributes = {
        name: vars(module).copy()
        for name, module in snapshot.items()
        if hasattr(module, "__path__")
    }
    yield

    current = _arrayscope_modules()
    replaced_names = {name for name, module in snapshot.items() if current.get(name) is not module}
    discard_names = replaced_names | {
        name
        for name in current.keys() - snapshot.keys()
        if _has_replaced_ancestor(name, replaced_names)
    }
    for name in discard_names:
        sys.modules.pop(name, None)
    sys.modules.update(snapshot)

    # Import machinery also caches every child module as an attribute of its
    # parent package. Restore those bindings together with sys.modules; fixing
    # only one side leaves two module/class identities alive in later tests.
    for name in discard_names:
        parent_name, separator, child_name = name.rpartition(".")
        if not separator or parent_name not in package_attributes:
            continue
        parent = snapshot[parent_name]
        previous_attributes = package_attributes[parent_name]
        if child_name in previous_attributes:
            setattr(parent, child_name, previous_attributes[child_name])
        else:
            vars(parent).pop(child_name, None)


@pytest.fixture(autouse=True)
def _pin_system_memory_snapshot(request, monkeypatch):
    """Make memory policy deterministic across hosts.

    Budgets derive from sampled system memory; without pinning, the same test
    passes on a 32 GiB workstation and fails on a 4 GiB CI runner. Tests that
    intentionally exercise host sampling can opt out with
    ``@pytest.mark.real_system_memory``.
    """

    if request.node.get_closest_marker("real_system_memory") is not None:
        yield
        return
    from arrayscope.core import memory_policy

    pinned = memory_policy.SystemMemorySnapshot(
        total_bytes=32 * memory_policy.GiB,
        available_bytes=24 * memory_policy.GiB,
        process_rss_bytes=1 * memory_policy.GiB,
        source="pinned-test-snapshot",
    )
    monkeypatch.setattr(memory_policy, "sample_system_memory", lambda **_: pinned)
    yield


@pytest.fixture(autouse=True)
def _restore_fft_runtime_options():
    from arrayscope.operations import fft_backend

    previous = fft_backend.get_fft_runtime_options()
    yield
    fft_backend.set_fft_runtime_options(backend=previous[0], workers=previous[1])


def _clear_qt_settings(QtCore) -> None:
    settings = QtCore.QSettings()
    settings.clear()
    settings.sync()


def _drain_qt_events(QtCore, QtWidgets) -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


@pytest.fixture(autouse=True)
def _isolate_qt_test_state(request):
    """Keep Qt settings and stray top-level widgets from leaking between tests."""

    if not {"qtbot", "qt_app"}.intersection(request.fixturenames):
        yield
        return

    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtCore, QtWidgets

    _clear_qt_settings(QtCore)
    yield
    app = QtWidgets.QApplication.instance()
    if app is not None:
        for widget in tuple(app.topLevelWidgets()):
            with contextlib.suppress(RuntimeError):
                widget.close()
        _drain_qt_events(QtCore, QtWidgets)
    _clear_qt_settings(QtCore)


@pytest.fixture(scope="session")
def qt_app():
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg

    app = pg.mkQApp()
    app.setOrganizationName("ArrayScopeTests")
    app.setApplicationName("ArrayScopeTests")
    return app
