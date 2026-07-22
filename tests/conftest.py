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


def pytest_xdist_auto_num_workers(config):
    """Cap ``-n auto`` at half the logical cores.

    Many tests create real GL contexts (vispy/pyqtgraph surfaces). Running one
    worker per core saturates the CPU and, more importantly, has every worker
    building GL contexts against the same offscreen/software-GL stack at once,
    which intermittently segfaults the driver and starves the timing-sensitive
    UI tests. Leaving half the cores free for each worker's Qt/GL threads keeps
    workers stable while still giving a large speedup. On small CI runners
    (2 cores) this floors at 2, so CI parallelism is unaffected.
    """

    cpus = os.cpu_count() or 2
    return max(2, cpus // 2)


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
