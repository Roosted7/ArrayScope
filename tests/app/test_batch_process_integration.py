"""Actual module-entry batch processes keep user files and commit outcomes intact."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

_STARTUP_GUARD = """
import atexit
import importlib.abc
import json
import os
from pathlib import Path
import signal
import sys

class HeadlessImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'PySide2', 'PyQt6', 'PyQt5', 'pyqtgraph',
                                      'wgpu', 'rendercanvas'} or fullname == \
            'arrayscope.operations.library':
            raise AssertionError('batch module entry imported ' + fullname)

sys.meta_path.insert(0, HeadlessImports())
interrupt_sent = False

from arrayscope.app import free_threading, qt_platform
from arrayscope.core import numba_runtime

def refuse_interactive_startup(*args, **kwargs):
    raise AssertionError('batch invoked interactive startup')

# These modules also define Qt-free enums used by numerical policies. Their
# ordinary imports are valid; actual GUI supervision and prewarm are not.
free_threading.supervise_free_threading_if_needed = refuse_interactive_startup
qt_platform.supervise_cli_if_needed = refuse_interactive_startup
numba_runtime.prewarm_all_async = refuse_interactive_startup

def record_imports():
    Path(os.environ['ARRAYSCOPE_BATCH_IMPORT_AUDIT']).write_text(json.dumps({
        'numba_imported': 'numba' in sys.modules,
        'library_imported': 'arrayscope.operations.library' in sys.modules,
        'interrupt_sent': interrupt_sent,
    }))

atexit.register(record_imports)

if os.environ.get('ARRAYSCOPE_BATCH_INTERRUPT_ON_RESTORE') == '1':
    original_signal = signal.signal
    initial_handler = signal.getsignal(signal.SIGINT)
    saw_cooperative_handler = False
    def interrupt_restoration(signum, handler):
        global saw_cooperative_handler, interrupt_sent
        previous = original_signal(signum, handler)
        if signum == signal.SIGINT:
            if saw_cooperative_handler and handler is initial_handler:
                interrupt_sent = True
                os.kill(os.getpid(), signal.SIGINT)
            else:
                saw_cooperative_handler = True
        return previous
    signal.signal = interrupt_restoration
"""


def _module_process(tmp_path, arguments, *, interrupt_on_restore=False):
    startup = tmp_path / "startup"
    startup.mkdir(exist_ok=True)
    (startup / "sitecustomize.py").write_text(_STARTUP_GUARD)
    audit = tmp_path / "imports.json"
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    environment["PYTHONPATH"] = os.pathsep.join((str(startup), environment.get("PYTHONPATH", "")))
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "profile" / "config")
    environment["XDG_CACHE_HOME"] = str(tmp_path / "profile" / "cache")
    environment["ARRAYSCOPE_BATCH_IMPORT_AUDIT"] = str(audit)
    environment["ARRAYSCOPE_BATCH_INTERRUPT_ON_RESTORE"] = str(int(interrupt_on_restore))
    completed = subprocess.run(
        [sys.executable, "-m", "arrayscope", *map(str, arguments)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert audit.exists(), completed.stderr
    assert json.loads(audit.read_text()) == {
        "numba_imported": False,
        "library_imported": False,
        "interrupt_sent": interrupt_on_restore,
    }
    return completed


def test_module_entry_exports_without_display_supervisors_or_library_writes(tmp_path):
    source = np.arange(4 * 3, dtype=np.float32).reshape(4, 3, 1)
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "selected.npz"
    np.save(source_path, source)
    original = source_path.read_bytes()
    profile = tmp_path / "profile"
    profile.mkdir()
    marker = profile / "user-operation.json"
    marker.write_text('{"review": {"required": true}}')
    before = marker.stat().st_mtime_ns, marker.read_bytes()

    completed = _module_process(
        tmp_path,
        [source_path, "--export", output_path, "--range", "0=1:3", "--no-squeeze"],
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 1
    summary = json.loads(completed.stdout)
    assert summary["output_shape"] == [2, 3, 1]
    assert summary["paths"] == [str(output_path)]
    with np.load(output_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["array"], source[1:3])
    assert source_path.read_bytes() == original
    assert (marker.stat().st_mtime_ns, marker.read_bytes()) == before
    assert list(profile.rglob("*")) == [marker]


def test_module_help_exposes_headless_options_without_loading_the_viewer(tmp_path):
    completed = _module_process(tmp_path, ["--help"])

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "--export" in completed.stdout
    assert "--recipe" in completed.stdout
    assert "--dry-run" in completed.stdout


@pytest.mark.parametrize("alias", ["same-path", "symlink", "hardlink"])
def test_module_entry_refuses_input_output_alias_without_changing_input(tmp_path, alias):
    source_path = tmp_path / "input.npy"
    output_path = source_path if alias == "same-path" else tmp_path / "alias.npy"
    source = np.arange(24).reshape(4, 6)
    np.save(source_path, source)
    original = source_path.read_bytes()
    if alias == "symlink":
        output_path.symlink_to(source_path)
    elif alias == "hardlink":
        os.link(source_path, output_path)
    identity = source_path.stat().st_ino

    completed = _module_process(
        tmp_path, [source_path, "--export", output_path, "--allow-expensive"]
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert "same file" in completed.stderr
    assert source_path.read_bytes() == original
    assert source_path.stat().st_ino == identity
    assert output_path.samefile(source_path)
    if alias == "symlink":
        assert output_path.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT at the handler-restoration boundary")
def test_interrupt_restoring_previous_handler_reports_committed_export_as_success(tmp_path):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "result.npy"
    source = np.arange(30, dtype=np.float64).reshape(5, 6)
    np.save(source_path, source)

    completed = _module_process(
        tmp_path, [source_path, "--export", output_path], interrupt_on_restore=True
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 1
    assert json.loads(completed.stdout)["paths"] == [str(output_path)]
    assert "cancelled" not in completed.stderr.lower()
    np.testing.assert_array_equal(np.load(output_path, allow_pickle=False), source)


@pytest.mark.parametrize("alias", ["same-path", "symlink", "hardlink"])
def test_module_entry_refuses_to_overwrite_the_recipe_through_an_output_alias(tmp_path, alias):
    source_path = tmp_path / "input.npy"
    np.save(source_path, np.arange(6).reshape(2, 3))
    recipe_path = tmp_path / ("steps.npz" if alias == "same-path" else "steps.json")
    recipe_path.write_text(json.dumps({"version": 2, "operations": []}))
    output_path = recipe_path if alias == "same-path" else tmp_path / "result.npz"
    if alias == "symlink":
        output_path.symlink_to(recipe_path)
    elif alias == "hardlink":
        os.link(recipe_path, output_path)
    original_source = source_path.read_bytes()
    original_recipe = recipe_path.read_bytes()
    recipe_identity = recipe_path.stat().st_ino

    completed = _module_process(
        tmp_path, [source_path, "--recipe", recipe_path, "--export", output_path]
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert "same file" in completed.stderr
    assert source_path.read_bytes() == original_source
    assert recipe_path.read_bytes() == original_recipe
    assert recipe_path.stat().st_ino == recipe_identity
    assert output_path.samefile(recipe_path)
    if alias == "symlink":
        assert output_path.is_symlink()
