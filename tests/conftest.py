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
            "default. Use it while fixing one. Same as "
            "ARRAYSCOPE_TESTMON_RERUN_FAILING=1."
        ),
    )


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

    cpus = os.cpu_count() or 2
    ceiling = max(2, cpus // 2)

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
        config.arrayscope_selection_seed = testmon_policy.seed_map(config.rootdir)
    except Exception:  # an unseeded run is slow, never wrong
        config.arrayscope_selection_seed = None


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Turn change-selection on by default, before testmon reads its options.

    ``tryfirst`` is load-bearing, not decoration: testmon resolves its whole
    configuration from ``config.option`` in its own ``pytest_configure``, so the
    default has to be in place by then. (``pytest_configure`` is a historic
    hook and cannot be wrapped, so anything that needs testmon's *result* reads
    it lazily further down instead.)
    """

    decision = testmon_policy.decide(config)
    config.arrayscope_selection = decision
    if decision.active:
        _ensure_map_seeded(config)
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

    decision = getattr(config, "arrayscope_selection", None)
    if decision is None:
        return lines or None
    if not decision.active:
        lines.append(f"test selection: off ({decision.reason}) — running everything")
        return lines

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
_POINTLESS_PLUGIN_DISABLES = {
    "randomly": "pytest-randomly is not installed",
}


def _report_pointless_flags(terminalreporter, config) -> None:
    """Say so when a run carried a flag that could not have had an effect.

    Cheaper than a documentation rule nobody reads at the moment they type it,
    and it cannot go stale: if the plugin is ever really added, the check stops
    firing by itself.
    """

    from importlib.util import find_spec

    for plugin in tuple(getattr(config.option, "plugins", ()) or ()):
        name = str(plugin)
        if not name.startswith("no:"):
            continue
        disabled = name[3:]
        why = _POINTLESS_PLUGIN_DISABLES.get(disabled)
        if why is None:
            continue
        if find_spec(f"pytest_{disabled}") is not None:
            continue
        terminalreporter.write_line(f"-p {name}: no effect ({why}).", bold=True, yellow=True)


def _report_gate_run_cost(terminalreporter, config) -> None:
    """Name the focused answers after a wide ``--no-testmon`` sweep.

    ``--no-testmon`` has legitimate narrow uses — regenerating the canonical
    artifacts, or settling a run you suspect the map got wrong. It is not the
    pre-merge gate, and this fires when it was used as one.

    What an exhaustive *offscreen* run adds over selection is narrower than it
    looks: the map-erosion hole that once made it the only trustworthy answer is
    closed (:func:`protect_map_outside_the_scope`), the child-process class is
    declared and guarded, and rings 3–4 are not in it either way. CI sets
    ``CI``, which turns selection off, so every push is already swept
    exhaustively by a machine. ``--since`` is the branch-sized question a person
    should be asking before merging, and it costs a fraction of this.

    The trap this exists for is specific and hard to see from inside: while
    fixing an inherited red, a targeted ``pytest path::test`` answers
    "deselected", because selection deliberately does not re-run reds this
    checkout did not break — and ``--no-testmon`` is the first thing that
    visibly works. It costs ~222 s against a few-second loop, every iteration.
    Whoever just paid that is the only person who can be told at the moment it
    is useful.

    Deliberately not fired for the other exhaustive runs (CI, ``--cov``): those
    chose nothing and have no faster alternative.
    """

    if not vars(config.option).get("no-testmon"):
        return
    started = getattr(config, "arrayscope_started", None)
    if started is None:
        return
    elapsed = monotonic() - started
    if elapsed < _GATE_NUDGE_SECONDS:
        return
    terminalreporter.write_line(
        f"--no-testmon: swept {elapsed:.0f} s, recorded nothing. Pre-merge is "
        "`--since`; see `pytest --help`.",
        bold=True,
        yellow=True,
    )


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
    _report_pointless_flags(terminalreporter, config)
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
    structure = coverage_map.Structure(coverage_map.cached_structure(sidecar))
    reach = coverage_map.measure(config.rootdir, map_path, environment, structure=structure)
    if reach is None or not reach.total:
        return

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
        coverage_map.write_sidecar(map_path, structure=structure.payload(), baselines=baselines)


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
