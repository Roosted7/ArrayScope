"""Headless streaming owns its source and preserves exact scientific values."""

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from arrayscope.core.array_source import NdArraySource
from arrayscope.io import recipe_replay, streaming_export
from arrayscope.io.recipe_replay import (
    BatchExportNeedsConfirmation,
    BatchExportRequest,
    run_batch_export,
)
from arrayscope.operations.cancellation import EvaluationCancelled


@pytest.fixture
def source_observations(monkeypatch):
    maps, reads = [], []
    original_load = np.load
    original_read = NdArraySource.read_region

    def load(*args, **kwargs):
        result = original_load(*args, **kwargs)
        if isinstance(result, np.memmap):
            maps.append(result._mmap)
        return result

    def read(source, spec, **kwargs):
        reads.append(tuple(spec))
        return original_read(source, spec, **kwargs)

    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(NdArraySource, "read_region", read)
    yield maps, reads
    assert all(mapping.closed for mapping in maps)


def _small_chunks(monkeypatch):
    original = recipe_replay.plan_streaming_export

    def plan(request, **kwargs):
        return original(request, chunk_target_bytes=2 * 1024**2, **kwargs)

    monkeypatch.setattr(recipe_replay, "plan_streaming_export", plan)


@pytest.mark.parametrize("source_kind", ["npy", "hdf5"])
def test_streaming_dry_run_has_no_payload_reads_or_output_changes(
    tmp_path, monkeypatch, source_kind
):
    values = np.arange(11 * 6, dtype=np.float32).reshape(11, 6, 1)
    path = tmp_path / ("source.npy" if source_kind == "npy" else "source.h5")
    if source_kind == "npy":
        np.save(path, values)
    else:
        with h5py.File(path, "w") as handle:
            handle.create_dataset("scan/data", data=values, chunks=(2, 3, 1))
    output = tmp_path / "previous.npy"
    output.write_bytes(b"previous complete output")

    def reject_read(*_args, **_kwargs):
        pytest.fail("streaming dry-run read source values")

    monkeypatch.setattr(NdArraySource, "read_region", reject_read)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_read)
    result = run_batch_export(
        BatchExportRequest(
            path,
            output,
            ranges=((0, 1, 9), (1, 2, 5)),
            streaming=True,
            dataset_path="scan/data" if source_kind == "hdf5" else None,
        ),
        dry_run=True,
    )

    summary = result.summary
    assert summary["execution_mode"] == "streaming"
    assert summary["output_shape"] == [8, 3]
    assert summary["output_bytes"] == summary["source_read_bytes"] == 96
    assert summary["estimated_peak_bytes"] == summary["estimated_chunk_peak_bytes"]
    assert summary["chunk_shape"] == [8, 3]
    assert summary["chunk_count"] == 1
    assert summary["paths"] == []
    assert summary["dry_run"]
    assert output.read_bytes() == b"previous complete output"
    assert set(tmp_path.iterdir()) == {path, output}


def test_streaming_complex_recipe_uses_multiple_bounded_reads_and_closes_its_map(
    tmp_path, monkeypatch, source_observations
):
    _small_chunks(monkeypatch)
    grid = np.arange(33 * 257 * 17, dtype=np.float64).reshape(33, 257, 17)
    values = grid + 1j * grid[::-1, :, ::-1]
    path = tmp_path / "source.npy"
    np.save(path, values)
    recipe = tmp_path / "operations.json"
    recipe.write_text(
        json.dumps(
            {
                "version": 2,
                "operations": [
                    {"id": "crop", "axis": 1, "parameters": {"start": 1, "stop": 257}},
                    {"id": "scale", "parameters": {"factor": 2}},
                    {"id": "mean", "axis": 0, "enabled": False},
                    {"id": "reverse", "axis": 0},
                ],
            }
        )
    )
    output = tmp_path / "selected.npy"

    result = run_batch_export(
        BatchExportRequest(path, output, recipe, ranges=((0, 1, 32),), streaming=True)
    )

    expected = (values[:, 1:257] * 2)[::-1][1:32]
    np.testing.assert_array_equal(np.load(output, allow_pickle=False), expected)
    maps, reads = source_observations
    assert maps
    assert all(mapping.closed for mapping in maps)
    assert result.dtype == "complex128"
    assert result.chunk_count == len(reads) > 1
    read_bytes = [
        np.prod(
            [len(range(*item.indices(size))) for item, size in zip(spec, values.shape, strict=True)]
        )
        * values.dtype.itemsize
        for spec in reads
    ]
    assert max(read_bytes) < expected.nbytes
    assert sum(read_bytes) == result.source_read_bytes == expected.nbytes
    assert result.estimated_chunk_peak_bytes <= 2 * 1024**2
    assert set(tmp_path.iterdir()) == {path, recipe, output}


@pytest.mark.parametrize("shape", [(), (0, 3), (1, 1)])
def test_streaming_preserves_empty_and_scalar_values(tmp_path, shape, source_observations):
    values = np.full(shape, 3 + 4j, dtype=np.complex64)
    path = tmp_path / "source.npy"
    output = tmp_path / "selected.npy"
    np.save(path, values)

    result = run_batch_export(BatchExportRequest(path, output, streaming=True))

    expected = values.squeeze()
    saved = np.load(output, allow_pickle=False)
    assert saved.dtype == expected.dtype
    assert saved.shape == expected.shape == result.output_shape
    np.testing.assert_array_equal(saved, expected)


def test_default_summary_remains_unchanged_when_streaming_is_not_requested(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, np.arange(12).reshape(3, 4))
    request = BatchExportRequest(path, tmp_path / "output.npy")

    ordinary = run_batch_export(request, dry_run=True).summary
    streamed = run_batch_export(replace(request, streaming=True), dry_run=True).summary

    extra = {"execution_mode", "chunk_shape", "chunk_count", "estimated_chunk_peak_bytes"}
    assert extra.isdisjoint(ordinary)
    assert set(streamed) == set(ordinary) | extra
    for key in (
        "input_shape",
        "derived_shape",
        "ranges",
        "squeeze",
        "output_shape",
        "dtype",
        "output_bytes",
    ):
        assert streamed[key] == ordinary[key]


def test_large_logical_hdf5_dry_run_is_priced_per_chunk_before_any_source_io(tmp_path, monkeypatch):
    path = tmp_path / "large.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", shape=(4097, 65536), dtype="f4", chunks=(1, 4096))
    output = tmp_path / "output.npy"

    def reject_read(*_args, **_kwargs):
        pytest.fail("large streaming dry-run read source values")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_read)
    result = run_batch_export(
        BatchExportRequest(path, output, dataset_path="data", streaming=True), dry_run=True
    )

    assert result.source_read_bytes == result.output_bytes > 1024**3
    assert result.chunk_count > 1
    assert result.estimated_chunk_peak_bytes <= 64 * 1024**2
    assert result.needs_confirmation
    assert path.stat().st_size < 65536
    assert not output.exists()


def test_streaming_cost_consent_precedes_all_data_reads(tmp_path, monkeypatch, source_observations):
    path = tmp_path / "source.npy"
    np.save(path, np.arange(12).reshape(3, 4))
    output = tmp_path / "output.npy"
    monkeypatch.setattr(streaming_export, "DEFAULT_VISIBLE_RENDER_BUDGET_BYTES", 1)
    request = BatchExportRequest(path, output, streaming=True)

    preview = run_batch_export(request, dry_run=True)
    assert preview.needs_confirmation
    with pytest.raises(BatchExportNeedsConfirmation):
        run_batch_export(request)
    assert source_observations[1] == []
    assert not output.exists()

    result = run_batch_export(replace(request, allow_expensive=True))
    assert result.paths == (output,)
    np.testing.assert_array_equal(np.load(output), np.arange(12).reshape(3, 4))


@pytest.mark.parametrize("interrupt_at", ["writing", "complete"])
def test_streaming_cancellation_keeps_atomic_commit_semantics_and_closes_source(
    tmp_path, source_observations, interrupt_at
):
    path = tmp_path / "source.npy"
    np.save(path, np.arange(100).reshape(10, 10))
    output = tmp_path / "output.npy"
    output.write_bytes(b"previous complete export")
    token = type("Token", (), {"cancelled": False})()

    def cancel(progress):
        if progress.phase == interrupt_at and progress.completed_bytes:
            token.cancelled = True

    request = BatchExportRequest(path, output, streaming=True)
    if interrupt_at == "writing":
        with pytest.raises(EvaluationCancelled):
            run_batch_export(request, cancellation_token=token, progress_callback=cancel)
        assert output.read_bytes() == b"previous complete export"
    else:
        result = run_batch_export(request, cancellation_token=token, progress_callback=cancel)
        assert result.paths == (output,)
        np.testing.assert_array_equal(np.load(output), np.arange(100).reshape(10, 10))
    assert token.cancelled
    assert set(tmp_path.iterdir()) == {path, output}


@pytest.mark.parametrize("output_kind", ["npz", "source-alias"])
def test_streaming_rejects_invalid_destinations_before_opening_source(
    tmp_path, source_observations, output_kind
):
    path = tmp_path / "source.npy"
    np.save(path, np.arange(12).reshape(3, 4))
    before = path.read_bytes()
    output = path if output_kind == "source-alias" else tmp_path / "output.npz"

    with pytest.raises(ValueError, match=r"npy output|same file"):
        run_batch_export(BatchExportRequest(path, output, streaming=True))

    assert source_observations == ([], [])
    assert path.read_bytes() == before
