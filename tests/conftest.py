import os
import sys
import tempfile

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


from tests import load_report, testmon_policy

# A worker process here is not free: it imports Qt, pyqtgraph and the whole
# arrayscope package before it runs anything, which measures ~1.3 s wall. Below
# this much recorded work, starting even one of them costs more than the tests
# it would run, so a small selection runs in-process instead.
_SERIAL_BELOW_SECONDS = 2.5


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
    if testmon_policy.rerun_known_red_tests():
        # These run whether or not anything they use changed, so they are part
        # of the workload the pool has to cover — leaving them out is how a
        # thirteen-file re-run ends up running in one process.
        affected |= {name for name in peek.forced_files if testmon_policy.under(name, paths)}
        seconds += peek.forced_seconds
    unmapped = {
        name
        for name in testmon_policy.collect_test_files(config.rootdir, paths)
        if name not in peek.mapped_files
    }
    return len(affected | unmapped), seconds, len(unmapped)


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


def pytest_collection(session):
    """Narrow testmon's "always re-run what failed" exemption to the normal rule.

    Has to happen here: testmon fixes its deselection lists during configure
    (this conftest's own ``pytest_configure`` is too early — testmon's plugin is
    not registered yet), and collection is the last point before both the
    file-level ignore and the item-level deselection consume them. Under xdist
    that means the workers, which is where collection actually happens; the
    controller reports the count from the map instead of from the mutation.
    """

    testmon_policy.apply_known_red_policy(session.config)


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

    if not _selection_is_narrowing(config):
        return
    data = getattr(config, "testmon_data", None)
    mapped = len(getattr(data, "all_tests", ()) or ())
    unaffected = len(getattr(data, "stable_test_names", ()) or ())
    if not mapped or not unaffected:
        return
    known_red = len(testmon_policy.known_red_tests(config))
    skipped = unaffected if not testmon_policy.rerun_known_red_tests() else unaffected - known_red
    terminalreporter.write_line(
        f"test selection: {skipped} of {mapped} mapped tests were unaffected by this "
        "working tree and did not run. The whole suite is `pytest --no-testmon`.",
        bold=True,
    )
    if known_red and not testmon_policy.rerun_known_red_tests():
        terminalreporter.write_line(
            f"test selection: {known_red} of those were already failing before this run and "
            "nothing they use changed, so they were not re-run "
            "(`ARRAYSCOPE_TESTMON_RERUN_FAILING=1` re-runs them; "
            "`python tools/test_selection.py status` names them).",
            bold=True,
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
