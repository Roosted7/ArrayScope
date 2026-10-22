"""HDF5 selector metadata and actual selected-source loading."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from arrayscope.core.array_source import LazySourceArray
from arrayscope.io.selectors import H5DatasetSelector


def _write_records(path, value=0):
    records = np.zeros(1, dtype=[("samples", "f4", (2, 3))])
    records["samples"] = value
    with h5py.File(path, "w") as output:
        output.create_dataset("records", data=records)


def test_single_dataset_opens_a_lazy_source_without_reading_values(tmp_path, monkeypatch):
    path = tmp_path / "volume.h5"
    expected = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    with h5py.File(path, "w") as output:
        output.create_dataset("scan/volume", data=expected, chunks=(2, 5, 1), compression="gzip")
    reads = []
    original = h5py.Dataset.__getitem__

    def record(dataset, index):
        reads.append(index)
        return original(dataset, index)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    selector = H5DatasetSelector(path)
    name, source = selector.get_single_data()
    selector.close()

    assert isinstance(source, LazySourceArray)
    assert name == "scan/volume"
    assert source.shape == expected.shape
    assert not reads
    np.testing.assert_array_equal(source.read_region((slice(1, 3), 2, 1)), expected[1:3, 2, 1])
    assert reads
    source.close()


def test_eager_api_remains_detached_but_viewing_uses_a_lazy_source(tmp_path):
    path = tmp_path / "image.h5"
    expected = np.arange(12, dtype=np.complex64).reshape(3, 4) * (1 + 2j)
    with h5py.File(path, "w") as output:
        output.create_dataset("image", data=expected)
    before = h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)
    selector = H5DatasetSelector(path)
    eager = selector.load_data("image")
    source = selector.load_selected("image")
    selector.close()

    assert isinstance(eager, np.ndarray)
    assert isinstance(source, LazySourceArray)
    np.testing.assert_array_equal(eager, expected)
    np.testing.assert_array_equal(source.read_region((1, slice(None))), expected[1])
    assert h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE) == before
    source.close()


def test_compound_subarray_fallback_has_a_visible_cost_before_opening(tmp_path, qtbot):
    path = tmp_path / "records.h5"
    records = np.zeros(1, dtype=[("samples", "f4", (3, 4))])
    with h5py.File(path, "w") as output:
        output.create_dataset("records", data=records)
    selector = H5DatasetSelector(path)
    assert selector.requires_gui()
    dialog = selector._create_dialog()
    qtbot.addWidget(dialog)
    dialog.show()
    item = selector.tree.topLevelItem(0).child(0)
    selector.tree.setCurrentItem(item)

    assert item.text(4) == "Load all"
    assert "48 B" in selector.selection_hint.text()
    assert "entire selected array" in selector.selection_hint.text()
    assert "do not yet support lazy reads" in selector.open_button.accessibleDescription()
    assert isinstance(selector.load_selected("records/samples"), np.ndarray)


def test_lazy_access_is_visible_and_browsing_retains_no_file(tmp_path, qtbot, monkeypatch):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("scan/image", (50, 60, 7), dtype="f4", chunks=(10, 20, 1))
    before = h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)

    def no_read(*_args, **_kwargs):
        pytest.fail("Browsing must use immutable metadata only")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_read)
    selector = H5DatasetSelector(path)
    dialog = selector._create_dialog()
    qtbot.addWidget(dialog)
    dialog.show()
    item = selector.tree.topLevelItem(0).child(0)
    selector.tree.setCurrentItem(item)

    assert item.text(4) == "Lazy"
    assert "only requested array regions" in selector.selection_hint.text()
    assert h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE) == before


def test_nonblocking_view_passes_file_intent_instead_of_a_live_source(tmp_path, monkeypatch):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("image", data=np.ones((3, 4)))
    started = []

    class Process:
        def __init__(self, *, target, args):
            self.target = target
            self.args = args

        def start(self):
            started.append(self)

    monkeypatch.setattr("multiprocessing.Process", Process)
    selector = H5DatasetSelector(path)
    monkeypatch.setattr(selector, "load_selected", lambda _path: pytest.fail("loaded in parent"))

    assert selector.view(block=False)
    assert len(started) == 1
    assert started[0].args == (path.resolve(), "H5DatasetSelector", None)


@pytest.mark.parametrize("method", ["load_data", "load_selected"])
@pytest.mark.parametrize("mutation", ["write", "replace"])
def test_eager_compound_read_refuses_stale_selection_before_payload(
    tmp_path, monkeypatch, method, mutation
):
    from arrayscope.io.hdf5_catalog import HDF5SourceChanged

    path = tmp_path / "records.h5"
    _write_records(path)
    selector = H5DatasetSelector(path)
    if mutation == "write":
        with h5py.File(path, "r+") as output:
            values = np.zeros(1, dtype=output["records"].dtype)
            values["samples"] = 9
            output["records"][:] = values
    else:
        replacement = tmp_path / "replacement.h5"
        _write_records(replacement, 9)
        replacement.replace(path)

    def no_read(*_args, **_kwargs):
        pytest.fail("A stale eager selection must be refused before reading values")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_read)
    handles = h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)
    with pytest.raises(HDF5SourceChanged):
        getattr(selector, method)("records/samples")
    assert h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE) == handles


@pytest.mark.parametrize("kind", ["external_link", "virtual", "external_storage"])
def test_explicit_eager_api_refuses_unsupported_storage_before_payload(tmp_path, monkeypatch, kind):
    target = tmp_path / "target.h5"
    with h5py.File(target, "w") as output:
        output.create_dataset("image", data=np.ones((2, 3)))
    path = tmp_path / "archive.h5"
    with h5py.File(path, "w") as output:
        if kind == "external_link":
            output["outside"] = h5py.ExternalLink(str(target), "/image")
        elif kind == "virtual":
            layout = h5py.VirtualLayout((2, 3), dtype="f8")
            layout[:] = h5py.VirtualSource(str(target), "/image", shape=(2, 3))
            output.create_virtual_dataset("outside", layout)
        else:
            output.create_dataset(
                "outside", (2, 3), dtype="f8", external=[(str(tmp_path / "external.raw"), 0, 48)]
            )
    selector = H5DatasetSelector(path)
    assert not selector.catalog.dataset("outside").eager_supported

    def no_read(*_args, **_kwargs):
        pytest.fail("Unsupported storage must not be followed by the eager API")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_read)
    monkeypatch.setattr(h5py.Dataset, "__array__", no_read)
    with pytest.raises(ValueError, match="not supported"):
        selector.load_data("outside")


def test_eager_read_checks_identity_again_before_returning_values(tmp_path, monkeypatch):
    from arrayscope.io.hdf5_catalog import HDF5SourceChanged

    path = tmp_path / "records.h5"
    _write_records(path, 4)
    selector = H5DatasetSelector(path)
    original = h5py.Dataset.__getitem__

    def change_after_read(dataset, index, **kwargs):
        values = original(dataset, index, **kwargs)
        if isinstance(index, str):
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return values

    monkeypatch.setattr(h5py.Dataset, "__getitem__", change_after_read)
    handles = h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE)
    with pytest.raises(HDF5SourceChanged):
        selector.load_data("records/samples")
    assert h5py.h5f.get_obj_count(types=h5py.h5f.OBJ_FILE) == handles
