"""Public batch entrypoint, terminal contract, and actual process cancellation."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from arrayscope import __main__ as cli

_NO_GUI = """
import importlib.abc
import sys

class NoGUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'PySide6', 'PySide2', 'PyQt6', 'PyQt5', 'pyqtgraph',
            'wgpu', 'rendercanvas',
        }:
            raise AssertionError('headless export imported ' + fullname)

sys.meta_path.insert(0, NoGUI())
"""
_CLI = """
from arrayscope.__main__ import main
try:
    main()
finally:
    assert 'numba' not in sys.modules, 'headless export eagerly imported numba'
"""


def _subprocess_cli(arguments, *, before=""):
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    return subprocess.run(
        [sys.executable, "-c", _NO_GUI + before + _CLI, *map(str, arguments)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--dry-run"], "require --export"),
        (["--recipe", ""], "require --export"),
        (["--export", "out.npy"], "exactly one"),
        (["a.npy", "b.npy", "--export", "out.npy"], "exactly one"),
        (["a.npz", "--export", "out.npy"], ".npy input"),
        (["a.npy", "--export", "out.csv"], "end in .npy or .npz"),
        (["a.npy", "--export", "out.npy", "--consume"], "--consume"),
        (["a.npy", "--export", "out.npy", "--mmap"], "--mmap"),
        (["a.npy", "--export", "out.npy", "--title", "demo"], "--title"),
        (["a.npy", "--export", "out.npy", "--trace", "trace.jsonl"], "--trace"),
        (["a.npy", "--export", "out.npy", "--install-desktop"], "--install-desktop"),
        (["a.npy", "--export", "out.npy", "--range", "0=3:1"], "STOP"),
        (["a.npy", "--export", "out.npy", "--range", "0=1:3:2"], "AXIS=START:STOP"),
        (["a.npy", "--export", "out.npy", "--range", "0=1:2", "--range", "0=2:3"], "only once"),
    ],
)
def test_invalid_batch_arguments_fail_before_any_gui_dispatch(
    monkeypatch, capsys, arguments, message
):
    def fail_gui(*_args, **_kwargs):
        raise AssertionError("invalid batch arguments reached GUI dispatch")

    monkeypatch.setattr(cli, "_show_launcher", fail_gui)
    monkeypatch.setattr(cli, "_open_file_async", fail_gui)
    monkeypatch.setattr(cli, "_run_desktop_integration", fail_gui)

    with pytest.raises(SystemExit) as stopped:
        cli.main(arguments)

    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


@pytest.mark.parametrize("suffix", [".npy", ".npz"])
def test_actual_headless_export_matches_complex_numpy_oracle_without_gui_imports(tmp_path, suffix):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / f"result{suffix}"
    recipe_path = tmp_path / "operations.json"
    values = np.arange(4 * 6 * 8, dtype=np.float64).reshape(4, 6, 8)
    source = values + 1j * values[::-1]
    np.save(source_path, source)
    recipe_path.write_text(
        json.dumps(
            {
                "version": 2,
                "operations": [
                    {
                        "id": "crop",
                        "axis": 1,
                        "parameters": {"start": 1, "stop": 5},
                        "enabled": True,
                    },
                    {"id": "centered_fft", "axis": 2, "enabled": True},
                    {"id": "mean", "axis": 0, "enabled": True},
                ],
            }
        )
    )

    completed = _subprocess_cli(
        [
            source_path,
            "--export",
            output_path,
            "--recipe",
            recipe_path,
            "--range",
            "0=1:3",
            "--range",
            "1=2:5",
            "--allow-expensive",
        ]
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["output_shape"] == [2, 3]
    assert summary["dtype"] == "complex128"
    expected = np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(source[:, 1:5, :], axes=2), axis=2, norm="ortho"), axes=2
    ).mean(axis=0)[1:3, 2:5]
    if suffix == ".npz":
        with np.load(output_path, allow_pickle=False) as archive:
            exported = archive["array"]
    else:
        exported = np.load(output_path, allow_pickle=False)
    np.testing.assert_allclose(exported, expected, rtol=1e-12, atol=1e-12)
    assert len(completed.stdout.splitlines()) == 1
    assert "Writing" in completed.stderr


def test_dry_run_is_json_only_without_payload_reads_or_destination_changes(tmp_path):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "previous.npy"
    np.save(source_path, np.arange(3 * 4 * 1, dtype=np.float32).reshape(3, 4, 1))
    output_path.write_bytes(b"previous export")
    before = """
from arrayscope.core.array_source import LazySourceArray, NdArraySource
def refuse_read(*args, **kwargs):
    raise AssertionError('dry run read source values')
LazySourceArray.__array__ = refuse_read
NdArraySource.read_region = refuse_read
"""

    completed = _subprocess_cli(
        [
            source_path,
            "--export",
            output_path,
            "--range",
            "0=1:3",
            "--dry-run",
            "--no-squeeze",
        ],
        before=before,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["output_shape"] == [2, 4, 1]
    assert summary["source_read_bytes"] == summary["output_bytes"] == 32
    assert summary["dry_run"] is True
    assert summary["paths"] == []
    assert completed.stderr == ""
    assert output_path.read_bytes() == b"previous export"
    assert set(tmp_path.iterdir()) == {source_path, output_path}


def test_expensive_dry_run_succeeds_but_execution_requires_consent(tmp_path):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "result.npy"
    recipe_path = tmp_path / "fft.json"
    np.save(source_path, np.ones((3, 8), dtype=np.float64))
    recipe_path.write_text(
        json.dumps(
            {
                "version": 2,
                "operations": [
                    {"id": "centered_fft", "axis": 1, "enabled": True},
                ],
            }
        )
    )
    arguments = [source_path, "--export", output_path, "--recipe", recipe_path]

    preview = _subprocess_cli([*arguments, "--dry-run"])
    refused = _subprocess_cli(arguments)

    assert preview.returncode == 0, preview.stderr
    assert json.loads(preview.stdout)["needs_confirmation"] is True
    assert refused.returncode == 3
    assert refused.stdout == ""
    assert "--allow-expensive" in refused.stderr
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("case", "expected_code", "message"),
    [
        ("missing", 1, "Export failed"),
        ("axis", 2, "axis"),
        ("bounds", 2, "range"),
        ("unsupported", 2, "built-in"),
    ],
)
def test_runtime_failures_have_actionable_stderr_and_no_json(
    tmp_path, case, expected_code, message
):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "result.npy"
    if case != "missing":
        np.save(source_path, np.arange(6).reshape(2, 3))
    arguments = [source_path, "--export", output_path]
    if case == "axis":
        arguments += ["--range", "2=0:1"]
    elif case == "bounds":
        arguments += ["--range", "0=0:3"]
    elif case == "unsupported":
        recipe_path = tmp_path / "unsupported.json"
        recipe_path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "operations": [
                        {"id": "user:unreviewed-command", "enabled": True},
                    ],
                }
            )
        )
        arguments += ["--recipe", recipe_path, "--allow-expensive"]

    completed = _subprocess_cli(arguments)

    assert completed.returncode == expected_code, completed.stderr
    assert completed.stdout == ""
    assert message in completed.stderr
    assert not output_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT delivery in the actual child process")
@pytest.mark.parametrize("interrupt_at", ["staged-write", "committed-replace"])
def test_actual_sigint_preserves_previous_file_before_commit_and_reports_success_after(
    tmp_path, interrupt_at
):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "previous.npy"
    source = np.arange(20 * 1024**2, dtype=np.uint8)
    np.save(source_path, source)
    output_path.write_bytes(b"previous export")
    if interrupt_at == "staged-write":
        before = """
import os
import signal
from arrayscope.io import numpy_save
original_write = numpy_save._ExportStream.write
def interrupt_write(stream, value):
    written = original_write(stream, value)
    if len(value) > 1024:
        os.kill(os.getpid(), signal.SIGINT)
    return written
numpy_save._ExportStream.write = interrupt_write
"""
    else:
        before = """
import os
import signal
from arrayscope.io import numpy_save
original_replace = numpy_save.os.replace
def interrupt_replace(staged, target):
    original_replace(staged, target)
    os.kill(os.getpid(), signal.SIGINT)
numpy_save.os.replace = interrupt_replace
"""

    completed = _subprocess_cli([source_path, "--export", output_path], before=before)

    if interrupt_at == "staged-write":
        assert completed.returncode == 130, completed.stderr
        assert completed.stdout == ""
        assert "cancelled" in completed.stderr
        assert output_path.read_bytes() == b"previous export"
    else:
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["paths"] == [str(output_path)]
        np.testing.assert_array_equal(np.load(output_path, allow_pickle=False), source)
    assert set(tmp_path.iterdir()) == {source_path, output_path}


def test_batch_restores_previous_interrupt_handler_on_success_and_validation_failure(tmp_path):
    source_path = tmp_path / "input.npy"
    output_path = tmp_path / "result.npy"
    np.save(source_path, np.arange(4))
    previous = signal.getsignal(signal.SIGINT)

    def handler(_signal, _frame):
        raise AssertionError("unexpected interrupt")

    signal.signal(signal.SIGINT, handler)
    try:
        for tail, expected in (([], 0), (["--range", "2=0:1"], 2)):
            with pytest.raises(SystemExit) as stopped:
                cli.main([str(source_path), "--export", str(output_path), *tail])
            assert stopped.value.code == expected
            assert signal.getsignal(signal.SIGINT) is handler
    finally:
        signal.signal(signal.SIGINT, previous)
