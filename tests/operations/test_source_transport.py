"""ADR 0064 family 6 prerequisite: base data that names itself across processes.

``id(document.base_data)`` is a pointer. Two things break on it: a worker
cannot compute the same cache key the parent did, and two documents over one
file look like two different arrays. The replacement has to keep the property
the old key really relied on — two distinct in-memory arrays with equal
contents are still two documents — while gaining the two the boundary needs:
picklability and agreement over a file.
"""

from __future__ import annotations

import base64
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from arrayscope.core.array_source import (
    LazySourceArray,
    NdArraySource,
    ScaledArraySource,
)
from arrayscope.operations import source_transport
from arrayscope.operations.evaluator import _document_key, stage_document_key
from arrayscope.operations.pipeline import ArrayDocument, Crop
from arrayscope.operations.source_transport import (
    HDF5Leaf,
    LazyArrayNode,
    MemmapLeaf,
    SharedMemoryLeaf,
    SourceTransportUnavailable,
    base_data_token,
    describe_base_data,
)


@pytest.fixture
def npy_path(tmp_path):
    path = tmp_path / "volume.npy"
    np.save(path, np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6))
    return path


def test_two_documents_over_one_memmap_file_share_the_stage_document_key(npy_path):
    """The key a worker recomputes has to be the key the parent stored under."""

    first = ArrayDocument(np.load(npy_path, mmap_mode="r"), operations=(Crop(0, 1, 3),))
    second = ArrayDocument(np.load(npy_path, mmap_mode="r"), operations=(Crop(0, 1, 3),))

    assert first.base_data is not second.base_data
    assert stage_document_key(first) == stage_document_key(second)
    # Which is worth nothing unless it survives the pipe.
    assert pickle.loads(pickle.dumps(stage_document_key(first))) == stage_document_key(second)


def test_the_full_document_key_agrees_once_the_steps_are_the_same_objects(npy_path):
    """``_document_key`` folds ``document.steps``, and a step carries a uuid.

    So the base-data token is not the only process-local term in it: two
    independently *built* pipelines over one file still key apart, because
    ``OperationStep`` mints a ``step_id`` per instance. That is why the key
    that crosses the boundary is ``stage_document_key`` — steps travel as a
    recipe, and a rebuilt step is deliberately a new step.
    """

    steps = ArrayDocument(np.zeros((4, 5, 6), np.float32), operations=(Crop(0, 1, 3),)).steps
    first = ArrayDocument(np.load(npy_path, mmap_mode="r"), steps=steps)
    second = ArrayDocument(np.load(npy_path, mmap_mode="r"), steps=steps)

    assert _document_key(first) == _document_key(second)
    assert _document_key(first) != _document_key(
        ArrayDocument(np.load(npy_path, mmap_mode="r"), operations=(Crop(0, 1, 3),))
    )


def test_an_eager_array_keeps_one_token_and_two_arrays_never_share_one():
    """A uuid per object: stable while it lives, never confused with a twin."""

    values = np.arange(12, dtype=np.float32)
    twin = values.copy()

    assert base_data_token(values) == base_data_token(values)
    assert base_data_token(values) != base_data_token(twin)
    assert pickle.loads(pickle.dumps(base_data_token(values))) == base_data_token(values)


def test_two_equal_eager_arrays_still_key_apart():
    """The invariant the old ``id`` key existed for, restated on the new token."""

    left = ArrayDocument(np.zeros((3, 3), dtype=np.float32))
    right = ArrayDocument(np.zeros((3, 3), dtype=np.float32))

    assert _document_key(left) != _document_key(right)


def test_a_dead_array_does_not_hand_its_token_to_the_next_one():
    """An id is recycled; a token registered against a dead object must not be."""

    first = np.zeros(4, dtype=np.float32)
    token = base_data_token(first)
    identity = id(first)
    del first

    replacement = np.zeros(4, dtype=np.float32)
    if id(replacement) != identity:
        pytest.skip("the allocator did not reuse the address, so there is nothing to confuse")
    assert base_data_token(replacement) != token


def test_a_memmap_view_describes_its_own_offset_not_the_maps(npy_path):
    """numpy never updates ``memmap.offset`` on a slice; the descriptor must."""

    mapped = np.load(npy_path, mmap_mode="r")
    view = mapped[2:]

    leaf = describe_base_data(view)

    assert isinstance(leaf, MemmapLeaf)
    assert leaf.offset > int(mapped.offset)
    np.testing.assert_array_equal(np.asarray(leaf.reconstruct()), np.asarray(view))


def test_a_fortran_order_map_round_trips(tmp_path):
    """The ``.cfl`` shape: an F-order map, reshaped, still rebuilds identically."""

    path = tmp_path / "raw.bin"
    values = np.asfortranarray(np.arange(24, dtype=np.complex64).reshape(2, 3, 4))
    path.write_bytes(values.tobytes(order="F"))
    mapped = np.memmap(path, dtype=np.complex64, mode="r", shape=(2, 3, 4), order="F")

    leaf = describe_base_data(mapped)

    assert isinstance(leaf, MemmapLeaf)
    assert leaf.order == "F"
    np.testing.assert_array_equal(np.asarray(leaf.reconstruct()), np.asarray(mapped))


def test_a_scaled_lazy_source_round_trips_through_its_descriptor(npy_path):
    """The uncompressed-NIfTI shape: scale node over a map, behind a lazy proxy."""

    mapped = np.load(npy_path, mmap_mode="r")
    source = LazySourceArray(
        ScaledArraySource(mapped, slope=2.5, inter=-1.0, out_dtype=np.float32, label="nii")
    )

    descriptor = describe_base_data(source)
    rebuilt = descriptor.reconstruct()

    index = (slice(1, 3), slice(None), 2)
    np.testing.assert_array_equal(rebuilt.read_region(index), source.read_region(index))
    assert rebuilt.dtype == source.dtype
    assert rebuilt.shape == source.shape


def test_a_composite_source_round_trips(npy_path):
    mapped = np.load(npy_path, mmap_mode="r")
    from arrayscope.core.array_source import CompositeArraySource

    source = CompositeArraySource(
        NdArraySource(mapped, label="a"), NdArraySource(mapped[::1], label="b")
    )

    rebuilt = describe_base_data(source).reconstruct()

    index = (slice(None), slice(1, 4), 0)
    np.testing.assert_array_equal(rebuilt.read_region(index), source.read_region(index))


def test_an_eager_array_is_not_published_until_this_process_runs_children(monkeypatch):
    """Publication doubles memory; a threads-only viewer must never pay it."""

    values = np.arange(64, dtype=np.float32)

    assert describe_base_data(values) is None
    assert describe_base_data(values, publish=True) is None

    monkeypatch.setattr(source_transport, "_publication_enabled", True)
    leaf = describe_base_data(values, publish=True)

    assert isinstance(leaf, SharedMemoryLeaf)
    np.testing.assert_array_equal(np.asarray(leaf.reconstruct()), values)
    # And the token does not move under it: a published array is the same array.
    assert base_data_token(values) != leaf


def test_a_published_array_is_refused_above_its_ceiling(monkeypatch):
    monkeypatch.setattr(source_transport, "_publication_enabled", True)
    monkeypatch.setattr(source_transport, "MAX_PUBLISHED_BYTES", 16)

    assert describe_base_data(np.zeros(64, dtype=np.float32), publish=True) is None


def test_a_file_that_changed_refuses_to_rebuild(npy_path):
    """Serving other bytes under the same descriptor is the failure to avoid."""

    leaf = describe_base_data(np.load(npy_path, mmap_mode="r"))
    np.save(npy_path, np.zeros((4, 5, 6), dtype=np.float32) + 7.0)

    with pytest.raises(SourceTransportUnavailable):
        leaf.reconstruct()


@pytest.fixture
def hdf5_sources(tmp_path):
    import h5py

    from arrayscope.io.hdf5_catalog import catalog_hdf5
    from arrayscope.io.hdf5_source import HDF5ArraySource

    path = tmp_path / "datasets.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "first", data=np.arange(60, dtype=np.float32).reshape(3, 4, 5), chunks=(1, 2, 5)
        )
        handle.create_dataset("second", data=np.arange(60, dtype=np.float32).reshape(3, 4, 5))
    catalog = catalog_hdf5(path)
    sources = [HDF5ArraySource(path, catalog.dataset(key)) for key in ("first", "second")]
    try:
        yield path, catalog, sources
    finally:
        for source in sources:
            source.close()


def test_hdf5_descriptor_is_metadata_only_and_preserves_dataset_identity(hdf5_sources, monkeypatch):
    import h5py

    from arrayscope.io.hdf5_source import HDF5ArraySource

    path, catalog, (source, other) = hdf5_sources

    def fail_payload(*_args, **_kwargs):
        raise AssertionError("description or reconstruction read HDF5 payload")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", fail_payload)
    monkeypatch.setattr(h5py.Dataset, "read_direct", fail_payload)
    lazy = LazySourceArray(source, materialize_budget_bytes=77)
    descriptor = describe_base_data(lazy)
    payload = pickle.dumps(descriptor)
    rebuilt = pickle.loads(payload).reconstruct()
    twin = HDF5ArraySource(path, catalog.dataset("first"))
    try:
        assert isinstance(descriptor, LazyArrayNode)
        assert isinstance(descriptor.child, HDF5Leaf)
        assert rebuilt._materialize_budget_bytes == 77
        assert len(payload) < 8192
        assert base_data_token(source) == base_data_token(twin)
        assert base_data_token(rebuilt) == base_data_token(lazy)
        assert base_data_token(source) != base_data_token(other)
    finally:
        rebuilt.close()
        twin.close()


def test_hdf5_descriptor_keeps_open_identity_if_file_replaced_before_description(hdf5_sources):
    import h5py

    path, _catalog, (source, _other) = hdf5_sources
    replacement = path.with_name("replacement.h5")
    with h5py.File(replacement, "w") as handle:
        handle.create_dataset("first", data=np.full((3, 4, 5), 99, dtype=np.float32))
    replacement.replace(path)

    descriptor = describe_base_data(source)

    assert descriptor.file_identity == source.file_identity
    with pytest.raises(SourceTransportUnavailable, match="changed"):
        descriptor.reconstruct()


def test_hdf5_reconstructed_source_revalidates_cached_file_on_each_read(hdf5_sources):
    import h5py

    from arrayscope.io.hdf5_catalog import HDF5SourceChanged

    path, _catalog, (source, _other) = hdf5_sources
    rebuilt = describe_base_data(source).reconstruct()
    try:
        expected = np.arange(60, dtype=np.float32).reshape(3, 4, 5)[1, 1:3, 2:4]
        np.testing.assert_array_equal(rebuilt.read_region((1, slice(1, 3), slice(2, 4))), expected)
        with h5py.File(path, "r+") as handle:
            handle["first"][1, 1, 2] = 1000
        with pytest.raises(HDF5SourceChanged, match="changed"):
            rebuilt.read_region((1, slice(1, 3), slice(2, 4)))
    finally:
        rebuilt.close()


def test_hdf5_descriptor_crosses_real_process_boundary_without_gui_or_payload_pickle(hdf5_sources):
    _path, _catalog, (source, _other) = hdf5_sources
    descriptor = describe_base_data(LazySourceArray(source))
    code = """
import base64, json, pickle, sys
from arrayscope.operations.source_read import read_base_region
from arrayscope.operations.regions import region_from_index_spec
source = pickle.loads(base64.b64decode(sys.argv[1])).reconstruct()
try:
    region = region_from_index_spec(source.shape, (1, slice(1, 3), slice(2, 4)))
    values = read_base_region(source, region)
    assert not any(name.startswith(('PySide', 'PyQt', 'pyqtgraph')) for name in sys.modules)
    print(json.dumps(values.tolist()))
finally:
    source.close()
"""

    completed = subprocess.run(
        [sys.executable, "-c", code, base64.b64encode(pickle.dumps(descriptor)).decode("ascii")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    np.testing.assert_array_equal(
        json.loads(completed.stdout), np.arange(60, dtype=np.float32).reshape(3, 4, 5)[1, 1:3, 2:4]
    )
