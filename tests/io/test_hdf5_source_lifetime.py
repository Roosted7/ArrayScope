"""HDF5 calls own their handles, including cancellation and stale exports."""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from arrayscope.io.hdf5_catalog import HDF5SourceChanged, catalog_hdf5
from arrayscope.io.hdf5_source import open_hdf5_source
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS

_WAIT_SECONDS = INTERACTION_SETTLE_HARD_LIMIT_MS / 1000


def _file(tmp_path, name="source.h5"):
    path = tmp_path / name
    values = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    with h5py.File(path, "w") as output:
        output.create_dataset("data", data=values, chunks=(3, 4), compression="gzip")
    return path, values


def test_catalog_and_repeated_reads_leave_no_live_hdf5_handles(tmp_path, monkeypatch):
    path, values = _file(tmp_path)
    identities = []
    enter = h5py.File.__enter__

    def entered(handle):
        identities.append(handle.id)
        return enter(handle)

    monkeypatch.setattr(h5py.File, "__enter__", entered)
    descriptor = catalog_hdf5(path).dataset("data")
    source = open_hdf5_source(path, descriptor)
    assert identities
    assert all(not identity.valid for identity in identities)
    for row in (0, 2, 5):
        result = source.read_region((row, slice(1, 6)))
        np.testing.assert_array_equal(result, values[row, 1:6])
        assert all(not identity.valid for identity in identities)
    source.close()
    source.close()
    with pytest.raises(ValueError, match="closed"):
        source.read_region((0, slice(None)))


def test_cancelled_read_is_rejected_before_opening_any_hdf5_file(tmp_path, monkeypatch):
    path, _values = _file(tmp_path)
    source = open_hdf5_source(path, "data")

    def refuse_open(*_args, **_kwargs):
        pytest.fail("a cancelled read opened an HDF5 file")

    monkeypatch.setattr(h5py.File, "__init__", refuse_open)
    try:
        with pytest.raises(EvaluationCancelled):
            source.read_region(
                (slice(None), slice(None)), cancellation_token=SimpleNamespace(cancelled=True)
            )
    finally:
        source.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_close_does_not_invalidate_an_active_reads_owned_handle(tmp_path, monkeypatch, cancel):
    path, values = _file(tmp_path)
    source = open_hdf5_source(path, "data")
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    token = SimpleNamespace(cancelled=False)
    active_ids = []
    results, errors = [], []
    getitem = h5py.Dataset.__getitem__

    def paused(dataset, spec, *args, **kwargs):
        active_ids.append(dataset.id)
        entered.set()
        assert release.wait(_WAIT_SECONDS), "test did not release the active read"
        return getitem(dataset, spec, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", paused)

    def read():
        try:
            results.append(source.read_region((slice(1, 3), slice(2, 6)), cancellation_token=token))
        except BaseException as error:
            errors.append(error)

    def close():
        source.close()
        closed.set()

    reader = threading.Thread(target=read)
    closer = threading.Thread(target=close)
    reader.start()
    try:
        assert entered.wait(_WAIT_SECONDS)
        closer.start()
        assert closed.wait(_WAIT_SECONDS), "close blocked on an independently owned HDF5 read"
        assert all(identity.valid for identity in active_ids)
        with pytest.raises(ValueError, match="closed"):
            source.read_region((0, slice(None)))
        token.cancelled = cancel
    finally:
        release.set()
        reader.join(_WAIT_SECONDS)
        if closer.ident is not None:
            closer.join(_WAIT_SECONDS)
        source.close()
    assert not reader.is_alive()
    assert not closer.is_alive()
    assert all(not identity.valid for identity in active_ids)
    if cancel:
        assert results == []
        assert len(errors) == 1
        assert isinstance(errors[0], EvaluationCancelled)
    else:
        assert errors == []
        assert len(results) == 1
        np.testing.assert_array_equal(results[0], values[1:3, 2:6])


def test_failed_hdf5_payload_read_closes_its_file_and_allows_an_independent_retry(
    tmp_path, monkeypatch
):
    path, values = _file(tmp_path)
    source = open_hdf5_source(path, "data")
    identities = []
    getitem = h5py.Dataset.__getitem__

    def failed(dataset, *_args, **_kwargs):
        identities.append(dataset.id)
        raise OSError("injected HDF5 read failure")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", failed)
    with pytest.raises(OSError, match="injected HDF5"):
        source.read_region((slice(None), 2))
    assert identities
    assert all(not identity.valid for identity in identities)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
    try:
        np.testing.assert_array_equal(source.read_region((slice(None), 2)), values[:, 2])
    finally:
        source.close()


@pytest.mark.parametrize("change", ["inode", "mtime", "dataset-metadata"])
def test_changed_source_is_refused_and_a_fresh_open_uses_the_new_snapshot(tmp_path, change):
    path, values = _file(tmp_path)
    source = open_hdf5_source(path, "data")
    if change == "inode":
        replacement, _values = _file(tmp_path, "replacement.h5")
        replacement.replace(path)
    elif change == "mtime":
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    else:
        with h5py.File(path, "r+") as changed:
            del changed["data"]
            changed.create_dataset("data", data=values.astype(np.float64) + 100)
    try:
        with pytest.raises(HDF5SourceChanged, match="changed"):
            source.read_region((slice(1, 3), slice(2, 4)))
    finally:
        source.close()
    fresh = open_hdf5_source(path, "data")
    try:
        expected = values[1:3, 2:4] + (100 if change == "dataset-metadata" else 0)
        np.testing.assert_array_equal(fresh.read_region((slice(1, 3), slice(2, 4))), expected)
        assert fresh.dtype == (
            np.dtype("float64") if change == "dataset-metadata" else values.dtype
        )
    finally:
        fresh.close()


@pytest.mark.parametrize("kind", ["external-link", "soft-link", "virtual", "external-storage"])
def test_indirect_hdf5_storage_is_refused_without_any_payload_read(tmp_path, monkeypatch, kind):
    path = tmp_path / "indirect.h5"
    with h5py.File(path, "w") as output:
        if kind == "external-link":
            output["indirect"] = h5py.ExternalLink("missing.h5", "/data")
        elif kind == "soft-link":
            output["indirect"] = h5py.SoftLink("/missing")
        elif kind == "virtual":
            layout = h5py.VirtualLayout((4, 5), dtype=np.float32)
            layout[:] = h5py.VirtualSource("missing.h5", "/data", shape=(4, 5))
            output.create_virtual_dataset("indirect", layout)
        else:
            output.create_dataset(
                "indirect",
                shape=(4, 5),
                dtype=np.float32,
                external=[("missing.raw", 0, h5py.h5f.UNLIMITED)],
            )

    def refuse_read(*_args, **_kwargs):
        pytest.fail("unsupported indirect storage was read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", refuse_read)
    descriptor = catalog_hdf5(path).dataset("indirect")
    assert not descriptor.lazy_supported
    assert descriptor.unsupported_reason
    with pytest.raises(ValueError, match="not supported"):
        open_hdf5_source(path, descriptor)
