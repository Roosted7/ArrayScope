"""HDF5 metadata selection and working estimates never decode source payloads."""

import os

import h5py
import numpy as np
import pytest

from arrayscope.io.hdf5_catalog import HDF5SourceChanged, catalog_hdf5, file_identity
from arrayscope.io.hdf5_source import HDF5ArraySource, open_hdf5_source


def _reject_payload(monkeypatch):
    def reject(*_args, **_kwargs):
        pytest.fail("metadata operation decoded HDF5 payload")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", reject)
    monkeypatch.setattr(h5py.Dataset, "__array__", reject)
    monkeypatch.setattr(h5py.Dataset, "read_direct", reject)


def test_catalog_preserves_groups_null_strings_and_field_projection_metadata(tmp_path, monkeypatch):
    path = tmp_path / "catalog.h5"
    records = np.zeros(1, dtype=[("samples", ">f4", (2, 3)), ("quality", "i2")])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("group/records", data=records)
        handle.create_dataset("group/empty", shape=(0, 3), dtype="c8")
        handle.create_dataset("group/null", dtype="f4")
        handle.create_dataset("group/label", data="measurement")
    _reject_payload(monkeypatch)

    catalog = catalog_hdf5(path)

    assert catalog.groups == ("group",)
    field = catalog.dataset("/group/records/samples")
    assert field.dataset_path == "/group/records"
    assert field.field_name == "samples"
    assert field.shape == (2, 3)
    assert field.dtype == np.dtype(">f4")
    assert field.eager_supported
    assert not field.lazy_supported
    assert "subarray" in field.unsupported_reason.lower()
    assert catalog.dataset("group/empty").lazy_supported
    assert catalog.dataset("group/empty").nbytes == 0
    assert catalog.dataset("group/null").shape is None
    assert not catalog.dataset("group/null").eager_supported
    assert not catalog.dataset("group/label").lazy_supported
    with pytest.raises(ValueError, match="not found"):
        catalog.dataset("missing")


def test_scalar_compound_fields_support_exact_native_rank_and_dtype(tmp_path):
    path = tmp_path / "fields.h5"
    records = np.empty((3, 4), dtype=[("signal", ">i4"), ("quality", "u1")])
    records["signal"] = np.arange(12).reshape(3, 4)
    records["quality"] = 7
    with h5py.File(path, "w") as handle:
        handle.create_dataset("records", data=records)

    source = open_hdf5_source(path, "records/signal")
    try:
        result = source.read_region(((2, 0, 2), slice(None, None, -1)))
    finally:
        source.close()

    assert result.dtype == records["signal"].dtype
    assert result.flags.owndata
    np.testing.assert_array_equal(result, records["signal"][[2, 0, 2], ::-1])


def test_links_virtual_and_external_storage_are_blocked_without_eager_bypass(tmp_path, monkeypatch):
    path = tmp_path / "blocked.h5"
    external = tmp_path / "outside.h5"
    with h5py.File(external, "w") as handle:
        handle.create_dataset("secret", data=np.arange(4))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("local", data=np.arange(4))
        handle["soft"] = h5py.SoftLink("/local")
        handle["external"] = h5py.ExternalLink(str(external), "/secret")
        handle.create_dataset(
            "external_storage", shape=(4,), dtype="i4", external=[("raw.bin", 0, 16)]
        )
        layout = h5py.VirtualLayout(shape=(4,), dtype="i8")
        layout[:] = h5py.VirtualSource(str(external), "/secret", shape=(4,))
        handle.create_virtual_dataset("virtual", layout)
    _reject_payload(monkeypatch)

    catalog = catalog_hdf5(path)

    for key in ("soft", "external", "external_storage", "virtual"):
        descriptor = catalog.dataset(key)
        assert descriptor.encoding == "unsupported"
        assert not descriptor.lazy_supported
        assert not descriptor.eager_supported
        with pytest.raises(ValueError, match="not supported"):
            HDF5ArraySource(path, descriptor)


def test_group_hardlink_cycle_does_not_recurse_indefinitely(tmp_path):
    path = tmp_path / "cycle.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("group")
        group["self"] = group
        group.create_dataset("signal", data=np.arange(4))

    catalog = catalog_hdf5(path)

    assert catalog.dataset("group/signal").shape == (4,)
    assert len(catalog.datasets) == 1


def test_stale_catalog_cannot_silently_capture_replaced_file(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("signal", data=np.arange(4))
    descriptor = catalog_hdf5(path).dataset("signal")
    replacement = tmp_path / "replacement.h5"
    with h5py.File(replacement, "w") as handle:
        handle.create_dataset("signal", data=np.arange(4) + 100)
    os.replace(replacement, path)

    with pytest.raises(HDF5SourceChanged, match="changed"):
        HDF5ArraySource(path, descriptor)


def test_dataset_identity_is_checked_even_when_current_file_identity_is_supplied(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("signal", shape=(4,), maxshape=(None,), dtype="i8")
    descriptor = catalog_hdf5(path).dataset("signal")
    with h5py.File(path, "r+") as handle:
        handle["signal"].resize((7,))

    with pytest.raises(HDF5SourceChanged, match="metadata changed"):
        HDF5ArraySource(path, descriptor, file_identity=file_identity(path))


def test_compressed_chunk_working_estimate_is_metadata_only_and_exceeds_tiny_output(
    tmp_path, monkeypatch
):
    path = tmp_path / "chunk.h5"
    shape = (4096, 4096)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("signal", shape=shape, chunks=shape, compression="gzip", dtype="c16")
    source = open_hdf5_source(path, "signal")
    _reject_payload(monkeypatch)
    try:
        estimated = source.source.estimate_read_nbytes(((4095, 0, 4095), 1))
        assert estimated >= np.prod(shape) * 16 + 3 * 16
        assert source.source.estimate_read_nbytes(((), slice(None))) == 0
    finally:
        source.close()


def test_custom_filter_working_estimate_is_unknown_without_decoding(tmp_path, monkeypatch):
    path = tmp_path / "custom.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "signal",
            shape=(10,),
            chunks=(5,),
            dtype="f4",
            compression=32008,
            allow_unknown_filter=True,
        )
    source = open_hdf5_source(path, "signal")
    _reject_payload(monkeypatch)
    try:
        assert source.source.estimate_read_nbytes((slice(0, 2),)) is None
    finally:
        source.close()


def test_named_open_preserves_local_hardlinked_group_alias_paths(tmp_path):
    path = tmp_path / "aliases.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("z_original")
        group.create_dataset("signal", data=np.arange(4))
        handle["a_alias"] = group

    source = open_hdf5_source(path, "z_original/signal")
    try:
        np.testing.assert_array_equal(source.read_region((slice(None),)), np.arange(4))
    finally:
        source.close()


def test_named_open_classifies_only_the_selected_dataset(tmp_path, monkeypatch):
    from arrayscope.io import hdf5_catalog

    path = tmp_path / "many.h5"
    with h5py.File(path, "w") as handle:
        for number in range(100):
            handle.create_dataset(f"dataset-{number}", shape=(3, 4), dtype="f4")
    classified = []
    original = hdf5_catalog._descriptors

    def count(dataset, key, identity):
        classified.append(key)
        return original(dataset, key, identity)

    monkeypatch.setattr(hdf5_catalog, "_descriptors", count)
    _reject_payload(monkeypatch)

    source = open_hdf5_source(path, "dataset-42")
    source.close()

    assert classified == ["dataset-42"]


def test_bundled_lzf_has_a_known_working_estimate_and_preserves_tiny_reads(tmp_path, monkeypatch):
    from arrayscope.core.array_source import LazySourceArray
    from arrayscope.operations.regions import region_from_index_spec
    from arrayscope.operations.source_read import read_base_region

    path = tmp_path / "lzf.h5"
    values = np.arange(32 * 48, dtype=np.float64).reshape(32, 48)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("signal", data=values, chunks=(16, 24), compression="lzf")
    with monkeypatch.context() as metadata_only:
        _reject_payload(metadata_only)
        descriptor = catalog_hdf5(path).dataset("signal")
        source = HDF5ArraySource(path, descriptor)
        estimated = source.estimate_read_nbytes(((31, 0, 31), slice(2, 5)))
    try:
        assert 32000 in descriptor.filter_ids
        assert estimated is not None
        assert estimated >= 16 * 24 * values.dtype.itemsize
        region = region_from_index_spec(values.shape, ((31, 0, 31), slice(2, 5)))
        actual = read_base_region(LazySourceArray(source), region)
    finally:
        source.close()

    np.testing.assert_array_equal(actual, values[[31, 0, 31], 2:5])
