import json
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.array_source import LazySourceArray, NdArraySource
from arrayscope.io import numpy_save_qt
from arrayscope.io.region_export import export_region
from arrayscope.operations.pipeline import ArrayDocument, Crop, Mean, OperationStep, Scale


def _choose_ranges(monkeypatch, ranges, *, squeeze=True, on_exec=None, accepted=True):
    class RangeDialog:
        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            if on_exec is not None:
                on_exec()
            return (
                numpy_save_qt.QtWidgets.QDialog.DialogCode.Accepted
                if accepted
                else numpy_save_qt.QtWidgets.QDialog.DialogCode.Rejected
            )

        def get_ranges(self):
            return ranges

        def should_squeeze(self):
            return squeeze

    monkeypatch.setattr(numpy_save_qt, "SaveRangeDialog", RangeDialog)


def test_collecting_export_preserves_uppercase_path_and_ranges_without_writing(
    tmp_path, monkeypatch
):
    path = tmp_path / "selection.NPY"
    document = ArrayDocument(np.arange(12).reshape(3, 4))

    class RangeDialog:
        def __init__(self, _parent, shape, **kwargs):
            assert tuple(shape) == (3, 4)
            assert kwargs["action_text"] == "Next…"

        def exec(self):
            return numpy_save_qt.QtWidgets.QDialog.DialogCode.Accepted

        def get_ranges(self):
            return [(1, 2), (1, 4)]

        def should_squeeze(self):
            return True

    monkeypatch.setattr(numpy_save_qt, "SaveRangeDialog", RangeDialog)
    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", lambda *_args: (str(path), ""))

    result = numpy_save_qt.collect_numpy_save_request(None, document)

    assert result.path == path
    assert result.document is document
    assert result.ranges == ((1, 2), (1, 4))
    assert result.output_shape == (3,)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("filename", "selected_filter", "expected", "archive"),
    [
        ("values.NPY", "NumPy archive (*.npz)", "values.NPY", False),
        ("values.npz", "NumPy array (*.npy)", "values.npz", True),
        ("values.NPZ", "NumPy archive (*.npz)", "values.NPZ", True),
        ("values", "", "values.npy", False),
        ("values", "NumPy archive (*.npz)", "values.npz", True),
        ("values.custom", "NumPy archive (*.npz)", "values.custom.npz", True),
    ],
)
def test_numpy_export_format_intent_preserves_explicit_extensions_and_default(
    tmp_path, monkeypatch, filename, selected_filter, expected, archive
):
    document = ArrayDocument(np.arange(6).reshape(2, 3))
    _choose_ranges(monkeypatch, ((0, 2), (0, 3)))

    def choose_file(_parent, _title, _default, name_filter):
        assert name_filter.startswith("NumPy array (*.npy)")
        assert "*.npz" in name_filter
        return str(tmp_path / filename), selected_filter

    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", choose_file)

    request = numpy_save_qt.collect_numpy_save_request(None, document)

    assert request.path == tmp_path / expected
    assert (request.recipe_json is not None) is archive
    assert request.view_recipe_json is None
    assert request.sidecar is False
    assert not tuple(tmp_path.iterdir())


def test_selected_npz_contains_exact_values_and_frozen_recipe_selection_and_source(
    tmp_path, monkeypatch
):
    class CountedSource(NdArraySource):
        def __init__(self, values):
            super().__init__(values)
            self.reads = []

        def read_region(self, spec, *, cancellation_token=None):
            self.reads.append(spec)
            return super().read_region(spec, cancellation_token=cancellation_token)

    values = np.arange(4 * 6 * 2, dtype=np.float64).reshape(4, 6, 2)
    values = values + 1j * values[::-1]
    source = CountedSource(values)
    document = ArrayDocument(
        LazySourceArray(source),
        steps=(
            OperationStep(Scale(2)),
            OperationStep(Mean(0), enabled=False),
            OperationStep(Crop(1, 1, 5)),
        ),
    )
    source_path = tmp_path / "original.h5"
    parent = SimpleNamespace(
        _filepath=source_path, _dataset_path="original/data", document=document
    )

    def edit_during_modal():
        parent._filepath = tmp_path / "new-source.h5"
        parent._dataset_path = "changed/data"
        parent.document = ArrayDocument(np.full((4, 4, 2), 999))

    _choose_ranges(monkeypatch, ((1, 3), (0, 2), (1, 2)), on_exec=edit_during_modal)
    output_path = tmp_path / "selected.NPZ"
    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", lambda *_args: (str(output_path), ""))

    request = numpy_save_qt.collect_numpy_save_request(parent, document)

    assert source.reads == []
    assert request.document is document
    assert not output_path.exists()
    result = export_region(request)
    with np.load(result.paths[0], allow_pickle=False) as saved:
        assert set(saved.files) == {"array", "recipe_json", "selection_json"}
        np.testing.assert_array_equal(
            saved["array"], (values * 2)[:, 1:5, :][1:3, 0:2, 1:2].squeeze()
        )
        assert saved["array"].dtype == np.dtype("complex128")
        recipe = json.loads(str(saved["recipe_json"]))
        manifest = json.loads(str(saved["selection_json"]))
    assert [operation["id"] for operation in recipe["operations"]] == ["scale", "mean", "crop"]
    assert [operation["enabled"] for operation in recipe["operations"]] == [True, False, True]
    assert recipe["operations"][0]["parameters"] == {"factor": 2}
    assert manifest["derived_shape"] == [4, 4, 2]
    assert manifest["ranges"] == [[1, 3], [0, 2], [1, 2]]
    assert manifest["output_shape"] == [2, 2]
    assert manifest["output_axes"] == [0, 1]
    assert manifest["source"] == {"source_path": str(source_path), "dataset_path": "original/data"}
    assert source.reads == [(slice(1, 3, 1), slice(1, 3, 1), slice(1, 2, 1))]
    assert set(tmp_path.iterdir()) == {output_path}


@pytest.mark.parametrize("cancel_at", [None, "ranges", "path"])
def test_raw_numpy_or_cancelled_collection_never_serializes_or_evaluates_custom_operations(
    tmp_path, monkeypatch, cancel_at
):
    class UnregisteredOperation:
        def apply(self, _data):
            raise AssertionError("request collection evaluated an operation")

        def output_shape(self, shape):
            return shape

        def output_dtype(self, dtype):
            return dtype

    document = ArrayDocument(np.arange(6).reshape(2, 3), operations=(UnregisteredOperation(),))
    _choose_ranges(monkeypatch, ((0, 2), (0, 3)), accepted=cancel_at != "ranges")

    def choose_file(*_args):
        assert cancel_at != "ranges"
        return ("" if cancel_at == "path" else str(tmp_path / "raw.npy")), ""

    def fail_recipe(*_args):
        raise AssertionError("raw NumPy or cancellation tried to serialize an operation")

    monkeypatch.setattr(numpy_save_qt, "get_save_file_name", choose_file)
    monkeypatch.setattr(numpy_save_qt, "dumps_recipe", fail_recipe)

    request = numpy_save_qt.collect_numpy_save_request(None, document)

    if cancel_at is not None:
        assert request is None
    else:
        assert request.recipe_json is request.source_metadata_json is None
    assert not tuple(tmp_path.iterdir())


def test_npz_metadata_uses_captured_hdf5_source_identity_without_payload_reads(
    tmp_path, monkeypatch
):
    import h5py

    from arrayscope.io.hdf5_source import open_hdf5_source

    source_path = tmp_path / "input.h5"
    with h5py.File(source_path, "w") as handle:
        handle.create_dataset("group/data", data=np.arange(12).reshape(3, 4))
    source = open_hdf5_source(source_path, "group/data")
    document = ArrayDocument(source)
    parent = SimpleNamespace(_filepath=tmp_path / "stale.h5", _dataset_path="stale/data")
    _choose_ranges(monkeypatch, ((0, 2), (1, 3)))
    monkeypatch.setattr(
        numpy_save_qt, "get_save_file_name", lambda *_args: (str(tmp_path / "selected.npz"), "")
    )

    def fail_payload(*_args, **_kwargs):
        raise AssertionError("collecting source metadata read HDF5 values")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", fail_payload)
    monkeypatch.setattr(h5py.Dataset, "read_direct", fail_payload)
    try:
        request = numpy_save_qt.collect_numpy_save_request(parent, document)
    finally:
        source.close()

    assert json.loads(request.source_metadata_json) == {
        "source_path": str(source_path),
        "dataset_path": "group/data",
    }
    assert not (tmp_path / "selected.npz").exists()
