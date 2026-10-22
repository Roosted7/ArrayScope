"""Independent scientific oracles for named HDF5 sources and exact exports."""

from __future__ import annotations

import os
from contextlib import contextmanager
from math import prod

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from arrayscope.core.array_source import LazySourceArray
from arrayscope.io.hdf5_catalog import catalog_hdf5
from arrayscope.io.hdf5_source import HDF5ArraySource
from arrayscope.io.region_export import RegionExportRequest, export_region, plan_region_export
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    Crop,
    Mean,
    ReverseAxis,
    SplitAxis,
)


@contextmanager
def _source(path, key):
    catalog = catalog_hdf5(path)
    source = HDF5ArraySource(path, catalog.dataset(key))
    try:
        yield source
    finally:
        source.close()


def _cartesian_numpy(data, selection):
    # NumPy's ix_ makes the independent Cartesian meaning explicit. A tuple
    # on two axes must not become NumPy's paired advanced-indexing semantics.
    indices = []
    drop = []
    for axis, (size, item) in enumerate(zip(data.shape, selection, strict=True)):
        selected = np.arange(size)[list(item) if isinstance(item, tuple) else item]
        indices.append(np.atleast_1d(selected))
        if isinstance(item, int):
            drop.append(axis)
    result = data[np.ix_(*indices)]
    return np.squeeze(result, axis=tuple(drop)) if drop else result


@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None), slice(None)),
        (slice(None, None, -2), slice(6, 0, -2), slice(8, 1, -3)),
        ((4, 0, 4, 1), (6, 2, 6, 0), (8, 1, 8, 3)),
        ((-1, 1, 1, 0), 2, slice(None, None, -2)),
        (-1, (6, 0, 6), slice(1, None, 2)),
        ((), (6, 1), slice(None)),
        (slice(0, 0, -1), slice(None), slice(None)),
    ],
    ids=[
        "whole",
        "negative-strides",
        "cartesian-duplicates",
        "mixed",
        "dropped-axis",
        "empty-fancy",
        "empty-slice",
    ],
)
def test_regions_preserve_numpy_values_order_duplicates_and_axis_rank(tmp_path, selection):
    path = tmp_path / "cartesian.h5"
    data = np.arange(5 * 7 * 9, dtype=np.int32).reshape(5, 7, 9)
    with h5py.File(path, "w") as file:
        file.create_dataset("group/signal", data=data, chunks=(2, 3, 4), compression="gzip")

    with _source(path, "/group/signal") as source:
        result = source.read_region(selection)

    expected = _cartesian_numpy(data, selection)
    assert isinstance(result, np.ndarray)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_sparse_corner_gather_never_materializes_the_huge_dataset_or_bounding_box(
    tmp_path, monkeypatch
):
    path = tmp_path / "sparse.h5"
    rows = 1_000_000_000
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset(
            "signal", shape=(rows, 3), dtype="i8", chunks=(1, 3), fillvalue=-5
        )
        dataset[0] = (11, 12, 13)
        dataset[rows - 1] = (71, 72, 73)
    requests = []
    original_getitem = h5py.Dataset.__getitem__
    original_read_direct = h5py.Dataset.read_direct

    def guard(dataset, selection):
        if (
            selection is None
            or selection is Ellipsis
            or (isinstance(selection, tuple) and not selection)
        ):
            items = (slice(None),) * dataset.ndim
        else:
            items = selection if isinstance(selection, tuple) else (selection,)
            if any(item is Ellipsis for item in items):
                axis = next(index for index, item in enumerate(items) if item is Ellipsis)
                items = (
                    *items[:axis],
                    *((slice(None),) * (dataset.ndim - len(items) + 1)),
                    *items[axis + 1 :],
                )
            items = (*items, *((slice(None),) * (dataset.ndim - len(items))))
        counts = []
        for size, item in zip(dataset.shape, items, strict=True):
            if isinstance(item, (int, np.integer)):
                counts.append(1)
            elif isinstance(item, slice):
                counts.append(len(range(*item.indices(size))))
            else:
                counts.append(len(item))
        elements = prod(counts)
        assert elements <= 64, f"a nine-value gather attempted to read {elements} stored values"
        assert sum(requests) + elements <= 64, "the sparse gather scanned too many stored values"
        requests.append(elements)

    def getitem(dataset, selection, *args, **kwargs):
        guard(dataset, selection)
        return original_getitem(dataset, selection, *args, **kwargs)

    def read_direct(dataset, destination, source_sel=None, dest_sel=None):
        guard(dataset, source_sel)
        return original_read_direct(dataset, destination, source_sel=source_sel, dest_sel=dest_sel)

    def reject_array(*_args, **_kwargs):
        pytest.fail("the huge HDF5 dataset was converted to an eager array")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
    monkeypatch.setattr(h5py.Dataset, "read_direct", read_direct)
    monkeypatch.setattr(h5py.Dataset, "__array__", reject_array)

    with _source(path, "signal") as source:
        assert source.nbytes == 24_000_000_000
        actual = source.read_region(((rows - 1, 0, rows - 1), (2, 0, 2)))

    np.testing.assert_array_equal(actual, [[73, 71, 73], [13, 11, 13], [73, 71, 73]])
    assert actual.nbytes == 72
    assert requests


def test_catalog_is_payload_free_and_does_not_guess_complex_values_from_names(
    tmp_path, monkeypatch
):
    path = tmp_path / "metadata.h5"
    real = np.arange(6, dtype=np.float32).reshape(3, 2)
    record = np.zeros((), dtype=[("real", "<f4"), ("imag", "<f4")])
    record["real"], record["imag"] = 3.25, -4.5
    with h5py.File(path, "w") as file:
        file.create_dataset("looks_complex", data=real)
        file.create_dataset("neutral_name", data=record)
        file.create_dataset("native", data=np.asarray(2 + 5j, dtype=np.complex128))

    def reject_payload(*_args, **_kwargs):
        pytest.fail("catalog/source construction read HDF5 payload values")

    for method in ("__getitem__", "__array__", "read_direct"):
        monkeypatch.setattr(h5py.Dataset, method, reject_payload)

    catalog = catalog_hdf5(path)
    named_real = catalog.dataset("looks_complex")
    assert named_real.shape == (3, 2)
    assert named_real.dtype == np.dtype(np.float32)
    encoded = catalog.dataset("neutral_name")
    assert encoded.shape == ()
    assert encoded.dtype == np.dtype(np.complex64)
    native = catalog.dataset("native")
    assert native.shape == ()
    assert native.dtype == np.dtype(np.complex128)
    source = HDF5ArraySource(path, named_real)
    try:
        assert source.shape == (3, 2)
        assert source.dtype == np.dtype(np.float32)
    finally:
        source.close()


@pytest.mark.parametrize("shape", [(), (3, 4)], ids=["scalar-record", "record-array"])
def test_real_imaginary_records_decode_by_dtype_without_dropping_scalar_axes(tmp_path, shape):
    path = tmp_path / "encoded.h5"
    count = int(np.prod(shape)) if shape else 1
    real = np.arange(count, dtype=np.float64).reshape(shape) + 0.25
    imaginary = real * -2 + 0.5
    record = np.empty(shape, dtype=[("real", ">f8"), ("imag", "<f8")])
    record["real"], record["imag"] = real, imaginary
    expected = real + 1j * imaginary
    with h5py.File(path, "w") as file:
        file.create_dataset("signal", data=record)

    with _source(path, "signal") as source:
        assert source.shape == shape
        assert source.dtype == np.dtype(np.complex128)
        result = source.read_region(tuple(slice(None) for _ in shape))

    assert result.shape == shape
    assert result.dtype == np.dtype(np.complex128)
    np.testing.assert_array_equal(result, expected)


def test_compound_subarray_field_is_explicitly_unsupported_for_lazy_reads(tmp_path, monkeypatch):
    path = tmp_path / "subarray.h5"
    records = np.zeros(4, dtype=[("samples", "f4", (2, 3))])
    with h5py.File(path, "w") as file:
        file.create_dataset("records", data=records)

    def reject_payload(*_args, **_kwargs):
        pytest.fail("an unsupported field eagerly loaded its payload")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject_payload)
    catalog = catalog_hdf5(path)
    descriptor = next(
        item
        for item in catalog.datasets
        if item.dataset_path == "/records" and item.field_name == "samples"
    )
    assert descriptor.shape == (4, 2, 3)
    assert not descriptor.lazy_supported
    assert "subarray" in descriptor.unsupported_reason.lower().replace("-", "")
    with pytest.raises(ValueError, match=r"[Ss]ub.?array"):
        HDF5ArraySource(path, descriptor)


def test_derived_output_selection_matches_independent_odd_mri_fft_and_rank_changes(tmp_path):
    path = tmp_path / "processing.h5"
    grid = np.arange(6 * 7 * 4, dtype=np.float64).reshape(6, 7, 4)
    data = grid + 1j * grid[::-1, :, ::-1]
    with h5py.File(path, "w") as file:
        file.create_dataset("acquisition", data=data, chunks=(2, 3, 2))
    output = tmp_path / "selected.npz"

    with _source(path, "acquisition") as source:
        document = ArrayDocument(
            LazySourceArray(source),
            operations=(Crop(0, 1, 5), ReverseAxis(1), CenteredFFT(1), SplitAxis(2, 2), Mean(0)),
        )
        request = RegionExportRequest(document, output, ((1, 6), (0, 2), (1, 2)), squeeze=False)
        prepared = plan_region_export(request)
        result = export_region(request, prepared=prepared)

    # ArrayScope's MRI forward transform is centered, unitary NumPy IFFT.
    # Seven samples makes fftshift/ifftshift ordering observable.
    transformed = np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(data[1:5, ::-1], axes=1), axis=1, norm="ortho"), axes=1
    )
    expected = transformed.reshape(4, 7, 2, 2).mean(axis=0)[1:6, :2, 1:2]
    with np.load(output, allow_pickle=False) as archive:
        actual = archive["array"]
        assert actual.shape == expected.shape == (5, 2, 1)
        assert actual.dtype == expected.dtype == np.dtype(np.complex128)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert result.shape == expected.shape


def test_source_change_after_evaluation_refuses_publication_and_preserves_old_export(tmp_path):
    path = tmp_path / "changing.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("signal", data=np.arange(20).reshape(4, 5))
    output = tmp_path / "previous.npy"
    original = b"previous complete export"
    output.write_bytes(original)
    changed = []

    def mutate_after_evaluation(progress):
        if progress.phase != "writing" or changed:
            return
        before = path.stat()
        with h5py.File(path, "r+") as file:
            file["signal"][0, 0] = -999
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        changed.append(True)

    with _source(path, "signal") as source:
        request = RegionExportRequest(
            ArrayDocument(LazySourceArray(source)), output, ((0, 4), (0, 5)), squeeze=False
        )
        with pytest.raises((RuntimeError, OSError), match=r"changed|Changed"):
            export_region(request, progress_callback=mutate_after_evaluation)

    assert changed == [True]
    assert output.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path, output}
