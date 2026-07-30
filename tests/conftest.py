import os
import sys
import tempfile
from time import monotonic

import pytest

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
_config_tag = _XDIST_WORKER or f"serial-{os.getpid()}"
_config_root = os.path.join(tempfile.gettempdir(), f"arrayscope-test-{_config_tag}")
_worker_config = os.path.join(_config_root, "config")
os.makedirs(_worker_config, exist_ok=True)
# On Linux/Unix QSettings (UserScope) resolves under XDG_CONFIG_HOME; pointing
# it at a private empty dir keeps every run's settings store hermetic and free
# of the un-clearable ``Unknown Organization.conf`` org fallback.
os.environ["XDG_CONFIG_HOME"] = _worker_config
if _XDIST_WORKER:
    os.environ.setdefault("ARRAYSCOPE_ARTIFACT_DIR", os.path.join(_config_root, "artifacts"))


from tests import coverage_map, load_report, red_ledger, testmon_policy

#: Written by the controller (or a serial run) only; workers read the file.
_red_ledger = None
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


def pytest_addoption(parser):
    # One group, so the pointer to the reference is printed once for all of
    # these rather than repeated in every option's help.
    group = parser.getgroup(
        "arrayscope selection",
        "arrayscope test selection — docs/testing/test-selection.md",
    )
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
    try:
        environment = testmon_policy.resolve_environment(config.getini("environment_expression"))
        peek = testmon_policy.peek(
            config.rootdir,
            environment,
            tuple(config.getini("testmon_ignore_dependencies")),
        )
    except Exception:  # a broken or half-written map must never fail a run
        return None
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

    Idempotent and safe to call from either of the two entry points that need
    it: xdist sizes its pool before ``pytest_configure`` runs, so whichever
    comes first does the copy.
    """

    if getattr(config, "arrayscope_selection_seed", "unset") != "unset" or _XDIST_WORKER:
        return
    try:
        config.arrayscope_selection_seed = testmon_policy.seed_map(
            config.rootdir, seeding=not config.getoption("no_seed_map", False)
        )
    except Exception:  # an unseeded run is slow, never wrong
        config.arrayscope_selection_seed = None


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
    peek = _map_peek(config)
    named_only = node_ids_only(config.args)
    unpriceable = unpriceable_bypass_refusal(
        None if peek is None else peek.mapped_tests, permitted, node_ids_only=named_only
    )
    if unpriceable:
        _refuse(config, unpriceable, _RECORD_ONE_FIRST)
        return
    scope = _run_scope(config)
    seconds = unselected_run_seconds(_recorded_test_seconds(config), scope, testmon_policy.under)
    reason = unselected_run_refusal(seconds, permitted)
    if reason:
        _refuse(config, reason, _RUN_IT_SELECTED)


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


def _map_peek(config):
    """One cached read of the map for every gate. ``None`` if it is unusable.

    Cached on the config because three gates now price the same run — the
    bypass gate, the fail-closed rule above it, and the serial gate — and each
    ``peek`` opens the SQLite map and runs testmon's whole stability
    determination. Sharing the read is also what keeps them from disagreeing
    about what a run costs.
    """

    cached = getattr(config, "arrayscope_map_peek", "unset")
    if cached != "unset":
        return cached
    try:
        peek = testmon_policy.peek(
            config.rootdir,
            testmon_policy.resolve_environment(config.getini("environment_expression")),
            tuple(config.getini("testmon_ignore_dependencies")),
        )
    except Exception:  # a broken map must never be the reason a run cannot start
        peek = None
    config.arrayscope_map_peek = peek
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
    # Seed *before* pricing, and for every run rather than only a selected one.
    # Seeding is a file copy from a sibling checkout, and a bypass run is
    # precisely the run that most needs a real map to be priced against: gating
    # it on ``decision.active`` meant a fresh worktree priced every bypass at
    # zero and let it straight through.
    _ensure_map_seeded(config)
    _refuse_unselected_run(config)
    _refuse_serial_run(config)
    _cap_explicit_workers(config)
    decision = testmon_policy.decide(config)
    config.arrayscope_selection = decision
    if decision.active:
        # Before testmon's own configure, which is where it resolves the
        # affected set through the module global this patches.
        testmon_policy.install_forgiveness(config.rootdir)
    if decision.active and not decision.explicit:
        config.option.testmon = True
    if decision.active and not _XDIST_WORKER:
        # A selected run is never the run that regenerates the canonical
        # artifacts (that is ``--no-testmon``, or CI's serial step), and it may
        # well execute in-process — see pytest_xdist_auto_num_workers. Keep it
        # off the shared directory the way an xdist worker already is.
        os.environ.setdefault("ARRAYSCOPE_ARTIFACT_DIR", os.path.join(_config_root, "artifacts"))
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
            ref, reached = testmon_policy.apply_since_baseline(config, since or None)
        except testmon_policy.UsageError as error:
            raise pytest.UsageError(str(error)) from error
        config.arrayscope_since = (ref, reached)


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
        items, testmon_policy.file_seconds(config), declaration_order
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


def pytest_sessionstart(session):
    """Open the red ledger, once testmon's map is readable and before any test.

    Not in ``pytest_configure``: that hook runs ``tryfirst`` here so the default
    is in place before testmon reads it, which means testmon's own configure —
    and therefore ``config.testmon_data`` — does not exist yet.

    Only the controller (or a serial run) opens it. Worker reports are forwarded
    to the controller's ``pytest_runtest_logreport``, so it sees every outcome,
    and one writer means no lock and no lost update.
    """

    global _red_ledger
    _red_ledger = None
    config = session.config
    testmon_config = getattr(config, "testmon_config", None)
    # Both the controller and every worker delete from the map during their own
    # collection, so both need the scope guard — unlike the ledger below, which
    # deliberately has one writer. Installed here because sessionstart is the
    # last point that reliably precedes collection on both.
    if testmon_config:
        testmon_policy.protect_map_outside_the_scope(config)
    if _XDIST_WORKER or not testmon_config or not testmon_config.collect:
        return
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


def pytest_collection_finish(session):
    """Serial runs collect here; xdist collects on the workers (see below)."""

    if _red_ledger is not None:
        _collected.update(item.nodeid for item in session.items)


def pytest_xdist_node_collection_finished(node, ids):
    """The controller's only view of what was collected, since it never collects."""

    if _red_ledger is not None:
        _collected.update(ids)


def pytest_sessionfinish(session):
    if _red_ledger is not None:
        _red_ledger.forget_missing(_collected)
        _red_ledger.save()


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
    yield


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

    donor = getattr(config, "arrayscope_selection_seed", None)
    if donor:
        lines.append(f"test selection: seeded this checkout's map from {donor}")
    data = getattr(config, "testmon_data", None)
    mapped = len(getattr(data, "all_tests", ()) or ())
    if not mapped:
        lines.append("test selection: on, but nothing is mapped yet — this run records the map")
    else:
        affected = len(getattr(data, "unstable_test_names", ()) or ())
        lines.append(
            f"test selection: on ({decision.reason}) — {affected} of {mapped} mapped tests affected"
        )
    return lines


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
    terminalreporter.write_line(
        f"Bypassed selection ({_bypass_permitted(config)}) and ran {elapsed:.0f} s.",
        bold=True,
        yellow=True,
    )
    terminalreporter.write_line(BYPASS_BANNER, bold=True, red=True)


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

    _report_function_coverage(terminalreporter, config)
    _report_gate_run_cost(terminalreporter, config)

    if not _selection_is_narrowing(config):
        return
    data = getattr(config, "testmon_data", None)
    mapped = len(getattr(data, "all_tests", ()) or ())
    unaffected = len(getattr(data, "stable_test_names", ()) or ())
    if not mapped or not unaffected:
        return
    since = getattr(config, "arrayscope_since", None) or _since_on_the_controller(config)
    reached = 0
    if since is not None:
        reached = since[1]

    new_reds = _red_ledger.new_reds if _red_ledger is not None else _new_reds(config)
    inherited = testmon_policy.known_red_tests(config) - new_reds
    # Every count here is "of the tests the map called unaffected", so anything
    # pulled back in — by --since, or by the flag that re-runs inherited reds —
    # has to come back out, or the line claims a gap that was already closed.
    skipped = unaffected - reached
    if testmon_policy.rerun_known_red_tests(config):
        skipped -= len(inherited)
    skipped = max(0, skipped)
    widened = f" +{reached} --since {since[0]}" if since is not None and reached else ""
    hint = "" if since is not None else " Pre-merge: `--since`."
    terminalreporter.write_line(
        f"test selection: {skipped}/{mapped} mapped unaffected, not run{widened}.{hint}",
        bold=True,
    )
    if inherited and not testmon_policy.rerun_known_red_tests(config):
        terminalreporter.write_line(
            f"test selection: {len(inherited)} inherited reds not re-run; "
            "`--rerun-reds` runs them.",
            bold=True,
            yellow=True,
        )
    if new_reds:
        # Printed whether or not they ran this time. They are never deselected,
        # so normally they are in the failure list above — but a run scoped with
        # -k or a path can still miss them, and then this is the only place the
        # regression is named.
        terminalreporter.write_line(
            f"BROKEN HERE ({len(new_reds)}): these were passing in this checkout and are "
            "not; they run on every selected run until they pass again.",
            bold=True,
            red=True,
        )
        for nodeid in sorted(new_reds):
            terminalreporter.write_line(f"    {nodeid}", red=True)


def _since_on_the_controller(config):
    """``(ref, count)`` for a ``--since`` run whose collection happened elsewhere.

    ``config.arrayscope_since`` is written during ``pytest_collection``, which
    under xdist runs on the workers — so the controller, the only process with a
    terminal, had neither the ``--since`` line nor its correction to the
    "unaffected and did not run" count. The gap the run had already closed was
    reported as still open, and the flag looked like it had done nothing.

    Recomputed here rather than forwarded: the query is read-only and the workers
    have no channel to hand a number back on. It costs a couple of seconds, and
    ``--since`` is the pre-merge step, not the inner loop.
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
    except Exception:  # a summary line is not worth failing a finished run over
        return None
    return None if reached is None else (ref, len(reached))


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
    elif drift.moved:
        points = reach.percent - (100.0 * drift.baseline_covered / max(1, reach.total))
        line = f"coverage: +{len(drift.gained)} -{len(drift.lost)} functions ({points:+.1f} pts)"
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
        coverage_map.write_sidecar(
            map_path,
            structure=structure.payload(),
            baselines=baselines,
            revisions=structure.retained(),
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
