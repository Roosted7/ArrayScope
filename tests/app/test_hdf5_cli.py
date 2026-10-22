"""Explicit HDF5 selection through the public command-line entrypoint."""

import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

_HEADLESS_MAIN = """
import importlib.abc
import runpy
import sys
class NoGUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'PySide2', 'PyQt6', 'PyQt5', 'pyqtgraph', 'wgpu', 'rendercanvas'}:
            raise AssertionError('headless HDF5 imported ' + fullname)
sys.meta_path.insert(0, NoGUI())
"""


def _batch(tmp_path, arguments, *, refuse_payload=False):
    guard = ""
    if refuse_payload:
        guard = """
import h5py
def refuse_read(*args, **kwargs):
    raise AssertionError('HDF5 dry run read payload values')
h5py.Dataset.__getitem__ = refuse_read
"""
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    environment["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _HEADLESS_MAIN + guard + "runpy.run_module('arrayscope', run_name='__main__')",
            *map(str, arguments),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_headless_hdf5_selected_recipe_export_matches_numpy_and_records_identity(tmp_path):
    path = tmp_path / "input.hdf5"
    values = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    values = (values + 1j * values[::-1]).astype(np.complex64)
    with h5py.File(path, "w") as output:
        output.create_dataset("group/data", data=values, chunks=(2, 3), compression="gzip")
    recipe = tmp_path / "steps.json"
    recipe.write_text(
        json.dumps(
            {
                "version": 2,
                "operations": [
                    {"id": "reverse", "axis": 0},
                    {"id": "scale", "parameters": {"factor": 2}},
                ],
            }
        )
    )
    output = tmp_path / "selected.npz"

    completed = _batch(
        tmp_path,
        [
            path,
            "--dataset",
            "/group/data",
            "--recipe",
            recipe,
            "--export",
            output,
            "--range",
            "0=1:3",
            "--range",
            "1=2:5",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 1
    summary = json.loads(completed.stdout)
    assert summary["dataset_path"] == "group/data"
    assert summary["input_path"] == str(path)
    assert summary["output_shape"] == [2, 3]
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["array"], (values[::-1] * 2)[1:3, 2:5])
        selection = json.loads(archive["selection_json"].item())
    assert selection["source"] == {"source_path": str(path), "dataset_path": "group/data"}
    assert not (tmp_path / "config").exists()


def test_headless_hdf5_dry_run_never_reads_payload_or_modifies_output(tmp_path):
    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("data", data=np.ones((4, 5, 1), dtype=np.float32))
    output = tmp_path / "previous.npy"
    output.write_bytes(b"previous complete export")

    completed = _batch(
        tmp_path,
        [
            path,
            "--dataset",
            "data",
            "--export",
            output,
            "--range",
            "0=1:3",
            "--no-squeeze",
            "--dry-run",
        ],
        refuse_payload=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    summary = json.loads(completed.stdout)
    assert summary["dataset_path"] == "data"
    assert summary["output_shape"] == [2, 5, 1]
    assert summary["dry_run"] is True
    assert summary["paths"] == []
    assert output.read_bytes() == b"previous complete export"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dataset", "data"],
        ["input.npy", "--dataset", "data"],
        ["first.h5", "second.h5", "--dataset", "data"],
        ["input.h5", "--dataset", ""],
        ["input.h5", "--dataset", "data", "--consume"],
        ["input.h5", "--export", "result.npy"],
    ],
)
def test_invalid_dataset_cli_usage_fails_before_gui_dispatch(monkeypatch, arguments):
    from arrayscope import __main__ as cli

    def refuse_gui(*_args, **_kwargs):
        pytest.fail("invalid dataset arguments reached GUI dispatch")

    monkeypatch.setattr(cli, "_open_file_async", refuse_gui)
    monkeypatch.setattr(cli, "_show_launcher", refuse_gui)
    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)
    assert stopped.value.code == 2


def test_interactive_dataset_option_is_forwarded_to_async_file_open(tmp_path, monkeypatch):
    from arrayscope import __main__ as cli
    from arrayscope.app import free_threading, qt_platform
    from arrayscope.core import numba_runtime

    path = tmp_path / "input.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("group/data", data=np.zeros((2, 3)))
    calls = []
    monkeypatch.setattr(free_threading, "supervise_free_threading_if_needed", lambda: None)
    monkeypatch.setattr(qt_platform, "supervise_cli_if_needed", lambda: None)
    monkeypatch.setattr(numba_runtime, "prewarm_all_async", lambda: None)
    monkeypatch.setattr(
        cli, "_open_file_async", lambda path, **options: calls.append((path, options))
    )

    cli.main([str(path), "--dataset", "/group/data"])

    assert calls == [
        (path, {"mmap": False, "consume": False, "title": None, "dataset_path": "/group/data"})
    ]
