"""Explicit HDF5 selection stays lazy through public loading and batch replay."""

import json

import h5py
import numpy as np
import pytest

from arrayscope.core.array_source import LazySourceArray, SourceReadRefused
from arrayscope.io.file_interpreters import load_path
from arrayscope.io.recipe_replay import BatchExportRequest, run_batch_export


def _input(tmp_path, *, leading=4):
    path = tmp_path / "input.h5"
    values = np.arange(leading * 5, dtype=np.float32).reshape(leading, 5, 1)
    with h5py.File(path, "w") as output:
        output.create_dataset(
            "group/complex_image", data=values, chunks=(2, 5, 1), compression="gzip"
        )
    return path, values


def test_load_path_selects_metadata_without_payload_reads_or_guessed_axes(tmp_path, monkeypatch):
    path, values = _input(tmp_path, leading=2)

    def refuse_read(*_args, **_kwargs):
        pytest.fail("opening a lazy HDF5 source read payload values")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", refuse_read)
    loaded = load_path(path, dataset_path="/group/complex_image")
    try:
        assert isinstance(loaded.data, LazySourceArray)
        assert loaded.data.shape == values.shape
        assert loaded.data.dtype == values.dtype
        assert loaded.axes is None
        assert loaded.metadata == {
            "source_path": str(path),
            "dataset_path": "group/complex_image",
            "selector_class_name": "H5DatasetSelector",
            "detected_format": "HDF5",
            "shape": values.shape,
            "dtype": "float32",
            "lazy": True,
        }
    finally:
        loaded.data.close()


def test_explicit_eager_hdf5_load_preserves_shape_and_detaches_values(tmp_path):
    path, values = _input(tmp_path)

    loaded = load_path(path, dataset_path="group/complex_image", lazy=False)

    assert isinstance(loaded.data, np.ndarray)
    assert loaded.data.shape == (4, 5, 1)
    assert loaded.metadata["lazy"] is False
    assert loaded.axes is None
    np.testing.assert_array_equal(loaded.data, values)
    replacement = tmp_path / "replacement.h5"
    with h5py.File(replacement, "w") as output:
        output.create_dataset("group/complex_image", data=values + 100)
    replacement.replace(path)
    np.testing.assert_array_equal(loaded.data, values)


def test_eager_hdf5_load_refuses_decode_working_memory_before_payload_io(tmp_path, monkeypatch):
    path = tmp_path / "large-chunk.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset(
            "data", shape=(65536, 1024), dtype=np.float64, chunks=(65536, 1024), compression="gzip"
        )

    def refuse_read(*_args, **_kwargs):
        pytest.fail("over-budget HDF5 eager loading reached payload I/O")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", refuse_read)
    with pytest.raises(SourceReadRefused) as caught:
        load_path(path, dataset_path="data", lazy=False)
    assert caught.value.requested_nbytes > caught.value.budget_bytes
    assert path.stat().st_size < 65536


@pytest.mark.parametrize("kind", ["missing-hdf5-key", "dataset-on-npy"])
def test_explicit_dataset_selection_is_required_and_never_ignored(tmp_path, kind):
    if kind == "missing-hdf5-key":
        path, _values = _input(tmp_path)
        with pytest.raises(ValueError, match="explicit dataset_path"):
            load_path(path)
    else:
        path = tmp_path / "input.npy"
        np.save(path, np.ones((2, 3)))
        with pytest.raises(ValueError, match="only supported"):
            load_path(path, dataset_path="data")


def test_hdf5_batch_dry_run_reads_only_metadata_and_keeps_previous_output(tmp_path, monkeypatch):
    path, values = _input(tmp_path)
    output = tmp_path / "previous.npy"
    output.write_bytes(b"previous complete export")

    def refuse_read(*_args, **_kwargs):
        pytest.fail("HDF5 dry-run read source payload values")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", refuse_read)
    result = run_batch_export(
        BatchExportRequest(
            path,
            output,
            dataset_path="/group/complex_image",
            ranges=((0, 1, 3), (1, 2, 5)),
            squeeze=False,
        ),
        dry_run=True,
    )

    assert result.input_shape == values.shape
    assert result.output_shape == (2, 3, 1)
    assert result.summary["dataset_path"] == "group/complex_image"
    assert result.summary["input_path"] == str(path)
    assert result.paths == ()
    assert result.dry_run
    assert output.read_bytes() == b"previous complete export"


def test_hdf5_batch_recipe_export_keeps_source_dataset_in_provenance(tmp_path):
    path, values = _input(tmp_path)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "version": 2,
                "operations": [
                    {"id": "reverse", "axis": 1},
                    {"id": "mean", "axis": 0},
                ],
            }
        )
    )
    output = tmp_path / "result.npz"

    result = run_batch_export(
        BatchExportRequest(
            path,
            output,
            recipe_path=recipe,
            dataset_path="/group/complex_image",
            ranges=((0, 1, 4),),
            squeeze=False,
            allow_expensive=True,
        )
    )

    assert result.summary["dataset_path"] == "group/complex_image"
    with np.load(output, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["array"], values[:, ::-1].mean(axis=0)[1:4])
        selection = json.loads(archive["selection_json"].item())
    assert selection["source"] == {
        "source_path": str(path),
        "dataset_path": "group/complex_image",
    }


@pytest.mark.parametrize("wrong", ["hdf5-without-key", "npy-with-key"])
def test_batch_input_and_dataset_must_match_explicitly(tmp_path, wrong):
    path, _values = _input(tmp_path)
    options = {}
    if wrong == "npy-with-key":
        path = tmp_path / "input.npy"
        np.save(path, np.zeros((2, 3)))
        options["dataset_path"] = "data"
    with pytest.raises(ValueError, match="dataset_path"):
        run_batch_export(BatchExportRequest(path, tmp_path / "out.npy", **options), dry_run=True)
