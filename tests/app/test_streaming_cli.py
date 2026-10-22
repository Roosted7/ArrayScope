"""Opt-in streaming through actual headless processes and real SIGINT delivery."""

import json
import os

import h5py
import numpy as np
import pytest

from tests.app.test_batch_cli import _subprocess_cli

_SMALL_CHUNKS = """
from arrayscope.io import recipe_replay
original_plan = recipe_replay.plan_streaming_export
def small_chunks(request, **kwargs):
    return original_plan(request, chunk_target_bytes=256 * 1024, **kwargs)
recipe_replay.plan_streaming_export = small_chunks
"""


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--stream"], "require --export"),
        (["missing.npy", "--export", "output.npz", "--stream"], ".npy output"),
    ],
)
def test_streaming_usage_errors_are_early_and_headless(arguments, message):
    completed = _subprocess_cli(arguments)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert message in completed.stderr


@pytest.mark.parametrize("source_kind", ["npy", "npy-fortran", "hdf5"])
def test_actual_streamed_complex_recipe_matches_numpy_without_gui_imports(tmp_path, source_kind):
    grid = np.arange(17 * 65 * 9, dtype=np.float64).reshape(17, 65, 9)
    values = grid + 1j * grid[::-1, :, ::-1]
    path = tmp_path / ("source.h5" if source_kind == "hdf5" else "source.npy")
    dataset = []
    if source_kind != "hdf5":
        np.save(path, np.asfortranarray(values) if source_kind == "npy-fortran" else values)
    else:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("scan/data", data=values, chunks=(2, 8, 3), compression="gzip")
        dataset = ["--dataset", "scan/data"]
    output = tmp_path / "selected.NPY"
    recipe = tmp_path / "operations.json"
    recipe.write_text(
        json.dumps(
            {
                "version": 2,
                "operations": [
                    {"id": "crop", "axis": 1, "parameters": {"start": 1, "stop": 65}},
                    {"id": "mean", "axis": 0, "enabled": False},
                    {"id": "reverse", "axis": 0},
                    {"id": "scale", "parameters": {"factor": 2}},
                ],
            }
        )
    )

    completed = _subprocess_cli(
        [
            path,
            *dataset,
            "--export",
            output,
            "--stream",
            "--recipe",
            recipe,
            "--range",
            "0=1:16",
            "--range",
            "1=1:64",
        ],
        before=_SMALL_CHUNKS,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(completed.stdout.splitlines()) == 1
    summary = json.loads(completed.stdout)
    expected = (values[:, 1:65][::-1] * 2)[1:16, 1:64]
    saved = np.load(output, allow_pickle=False)
    assert saved.dtype == np.dtype("complex128")
    np.testing.assert_array_equal(saved, expected)
    assert summary["execution_mode"] == "streaming"
    assert summary["chunk_count"] > 1
    assert summary["estimated_chunk_peak_bytes"] <= 256 * 1024
    assert summary["output_bytes"] == summary["source_read_bytes"] == expected.nbytes
    assert len(completed.stderr.splitlines()) <= 14
    assert "Streaming exact" in completed.stderr
    assert set(tmp_path.iterdir()) == {path, output, recipe}


def test_actual_streaming_dry_run_does_not_read_payload_or_write_files(tmp_path):
    path = tmp_path / "source.npy"
    values = np.arange(10 * 12, dtype=np.float32).reshape(10, 12, 1)
    np.save(path, values)
    output = tmp_path / "previous.npy"
    output.write_bytes(b"previous complete export")
    before = """
from arrayscope.core.array_source import LazySourceArray, NdArraySource
def refuse_read(*args, **kwargs):
    raise AssertionError('streaming dry-run read source values')
LazySourceArray.__array__ = refuse_read
NdArraySource.read_region = refuse_read
"""

    completed = _subprocess_cli(
        [path, "--export", output, "--stream", "--range", "0=2:8", "--dry-run"],
        before=before,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["execution_mode"] == "streaming"
    assert summary["output_shape"] == [6, 12]
    assert summary["output_bytes"] == 6 * 12 * values.dtype.itemsize
    assert summary["paths"] == []
    assert summary["dry_run"]
    assert completed.stderr == ""
    assert output.read_bytes() == b"previous complete export"
    assert set(tmp_path.iterdir()) == {path, output}


def test_unstreamable_transform_refuses_before_source_values_even_with_consent(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, np.arange(16).reshape(4, 4))
    recipe = tmp_path / "fft.json"
    recipe.write_text(
        json.dumps({"version": 2, "operations": [{"id": "centered_fftn", "axes": [0, 1]}]})
    )
    output = tmp_path / "output.npy"
    before = """
from arrayscope.io import recipe_replay
from arrayscope.core.array_source import NdArraySource
original_plan = recipe_replay.plan_streaming_export
def tiny_chunks(request, **kwargs):
    return original_plan(request, chunk_target_bytes=256, **kwargs)
def refuse_read(*args, **kwargs):
    raise AssertionError('unsupported streaming plan read source values')
recipe_replay.plan_streaming_export = tiny_chunks
NdArraySource.read_region = refuse_read
"""

    completed = _subprocess_cli(
        [path, "--export", output, "--stream", "--recipe", recipe, "--allow-expensive"],
        before=before,
    )

    assert completed.returncode == 2, completed.stderr
    assert completed.stdout == ""
    assert "streaming" in completed.stderr
    assert not output.exists()


def test_streaming_still_honors_the_shared_source_read_budget(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, np.arange(100).reshape(10, 10))
    output = tmp_path / "previous.npy"
    output.write_bytes(b"previous complete export")
    before = """
from arrayscope.operations import source_read
from arrayscope.core.array_source import NdArraySource
source_read.DEFAULT_SOURCE_READ_BUDGET_BYTES = 16
def refuse_read(*args, **kwargs):
    raise AssertionError('over-budget streaming reached source values')
NdArraySource.read_region = refuse_read
"""

    completed = _subprocess_cli(
        [path, "--export", output, "--stream", "--allow-expensive"], before=before
    )

    assert completed.returncode == 3, completed.stderr
    assert completed.stdout == ""
    assert "cannot override this budget" in completed.stderr
    assert output.read_bytes() == b"previous complete export"
    assert set(tmp_path.iterdir()) == {path, output}


@pytest.mark.skipif(os.name == "nt", reason="POSIX SIGINT delivered inside the actual child")
@pytest.mark.parametrize("interrupt_at", ["staged-write", "committed-replace"])
def test_actual_streaming_sigint_preserves_old_file_until_commit(tmp_path, interrupt_at):
    path = tmp_path / "source.npy"
    values = np.arange(64 * 32, dtype=np.float64).reshape(64, 32)
    np.save(path, values)
    output = tmp_path / "previous.npy"
    output.write_bytes(b"previous complete export")
    if interrupt_at == "staged-write":
        before = """
import os, signal
from arrayscope.io import numpy_save
original = numpy_save._ExportStream.write
def interrupt(stream, value):
    count = original(stream, value)
    if len(value) > 1024:
        os.kill(os.getpid(), signal.SIGINT)
    return count
numpy_save._ExportStream.write = interrupt
"""
    else:
        before = """
import os, signal
from arrayscope.io import numpy_save
original = numpy_save.os.replace
def interrupt(staged, target):
    original(staged, target)
    os.kill(os.getpid(), signal.SIGINT)
numpy_save.os.replace = interrupt
"""

    completed = _subprocess_cli([path, "--export", output, "--stream"], before=before)

    if interrupt_at == "staged-write":
        assert completed.returncode == 130, completed.stderr
        assert completed.stdout == ""
        assert "cancelled" in completed.stderr
        assert output.read_bytes() == b"previous complete export"
    else:
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["paths"] == [str(output)]
        assert "cancelled" not in completed.stderr
        np.testing.assert_array_equal(np.load(output, allow_pickle=False), values)
    assert set(tmp_path.iterdir()) == {path, output}
