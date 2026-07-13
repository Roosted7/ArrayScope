import os
import sys
import tempfile

import pytest

# --- Parallel-worker filesystem isolation (pytest-xdist) ---------------------
# xdist workers are separate processes but share one filesystem. Two shared
# on-disk resources would otherwise race between workers:
#   * QSettings — the test QApplication uses a fixed organization/application
#     name, so every worker reads and writes the same on-disk store, and the
#     autouse ``_clear_qt_settings`` fixture in one worker wipes another
#     worker's writes mid-test.
#   * ``tests/artifacts/`` — the Qt smoke test writes fixed filenames there.
# Point each worker at its own config + artifact directory. This must happen at
# import time, before Qt (and thus QSettings) resolves any path. It is a no-op
# for serial runs (``PYTEST_XDIST_WORKER`` is unset under ``-n 0`` / no xdist),
# so single-process runs keep their normal paths — which is what CI relies on
# when generating artifacts into the canonical ``tests/artifacts/``.
_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER")
if _XDIST_WORKER:
    _worker_root = os.path.join(tempfile.gettempdir(), f"arrayscope-xdist-{_XDIST_WORKER}")
    _worker_config = os.path.join(_worker_root, "config")
    os.makedirs(_worker_config, exist_ok=True)
    # On Linux/Unix QSettings (UserScope) resolves under XDG_CONFIG_HOME;
    # isolating it per worker keeps each worker's settings store private.
    os.environ["XDG_CONFIG_HOME"] = _worker_config
    os.environ.setdefault(
        "ARRAYSCOPE_ARTIFACT_DIR", os.path.join(_worker_root, "artifacts")
    )


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
import arrayscope  # noqa: F401

# Modules that tests monkeypatch but production code imports lazily must be
# in the identity snapshot below from the start; otherwise the teardown
# deletes the never-snapshotted sys.modules entry while the parent package
# attribute keeps the old object, and later `from arrayscope.display import
# colormap_library` resolves a *different* module than the running code uses
# (patched writes become invisible to the UI under test).
import arrayscope.display.colormap_library  # noqa: F401


def _arrayscope_module_names():
    return [
        name
        for name in sys.modules
        if name == "arrayscope" or name.startswith("arrayscope.")
    ]


@pytest.fixture(autouse=True)
def _restore_arrayscope_module_identity():
    """Keep ``arrayscope`` module (and class) identities stable across tests.

    A few tests purge ``arrayscope.*`` from ``sys.modules`` and re-import to
    exercise fresh-import behavior (package callability, lazy Qt binding, no
    eager pyqtgraph import). Without restoration the re-imported modules define
    brand-new class objects, so a later ``isinstance`` or ``pytest.raises``
    check in an unrelated test compares against the *old* class identity and
    fails even though the production code is correct (e.g. ``LazySourceArray``,
    ``SourceReadRefused``). Snapshot the arrayscope module objects before each
    test and restore them afterward so module identity cannot leak forward.
    """

    snapshot = {name: sys.modules[name] for name in _arrayscope_module_names()}
    yield
    for name in _arrayscope_module_names():
        if sys.modules.get(name) is not snapshot.get(name):
            del sys.modules[name]
    sys.modules.update(snapshot)


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
            try:
                widget.close()
            except RuntimeError:
                pass
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
    yield app
