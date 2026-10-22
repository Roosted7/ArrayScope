"""Streamed NPY files retain exact array semantics and atomic publication."""

import gc
import os
import stat
import weakref
from contextlib import nullcontext

import numpy as np
import pytest

from arrayscope.io import numpy_save
from arrayscope.io.numpy_save import save_derived_array_chunks
from arrayscope.kernel.task import CancellationToken
from arrayscope.operations.cancellation import EvaluationCancelled


def _save(path, array, chunks=None, **kwargs):
    return save_derived_array_chunks(
        path,
        shape=array.shape,
        dtype=array.dtype,
        chunks=[array] if chunks is None else chunks,
        **kwargs,
    )


@pytest.mark.parametrize("layout", ["C", "F", "strided", "complex", "big-endian"])
def test_chunks_roundtrip_in_global_c_order(tmp_path, layout):
    array = np.arange(11 * 7 * 5, dtype=np.float64).reshape(11, 7, 5)
    if layout == "F":
        array = np.asfortranarray(array)
    elif layout == "strided":
        array = array[::-1, ::-2, ::2]
    elif layout == "complex":
        array = (array + 1j * (array + 3))[:, ::-1]
    elif layout == "big-endian":
        array = array.astype(">f8")[:, ::-1]
    path = tmp_path / "output.NPY"

    assert _save(path, array, (array[start : start + 3] for start in range(0, len(array), 3))) == (
        path,
    )

    result = np.load(path, allow_pickle=False)
    assert result.dtype == array.dtype
    assert result.shape == array.shape
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, array)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype([("position", ">f8", (2,)), ("label", "S5")]),
        np.dtype([("温度", "<f4"), ("測定値", "c16")]),
        np.dtype("datetime64[ns]"),
        np.dtype("V0"),
    ],
)
def test_fixed_numpy_dtypes_keep_their_descriptor(tmp_path, dtype):
    array = np.zeros((4, 3), dtype=dtype)
    path = tmp_path / "records.npy"

    # NumPy reports when Unicode field names require its v3 header.
    with (
        pytest.warns(UserWarning, match="format 3.0")
        if dtype.names and "温度" in dtype.names
        else nullcontext()
    ):
        _save(path, array, [array[:1], array[1:]])

    result = np.load(path, allow_pickle=False)
    assert result.dtype == array.dtype
    np.testing.assert_array_equal(result, array)


def test_large_structured_descriptor_uses_numpy_large_header(tmp_path):
    array = np.zeros(2, dtype=[(f"field_{index:04d}", "u1") for index in range(4000)])
    path = tmp_path / "many-fields.npy"

    with pytest.warns(UserWarning, match="format 2.0"):
        _save(path, array, [array[:1], array[1:]])

    result = np.load(path, allow_pickle=False, max_header_size=200_000)
    assert result.dtype == array.dtype
    np.testing.assert_array_equal(result, array)


@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0, 4)])
def test_empty_global_outputs_need_no_data_chunks(tmp_path, shape):
    path = tmp_path / "empty.npy"
    save_derived_array_chunks(path, shape=shape, dtype="c16", chunks=())
    result = np.load(path, allow_pickle=False)
    assert result.shape == shape
    assert result.dtype == np.dtype("c16")


def test_zero_byte_chunks_still_count_rows(tmp_path):
    array = np.empty((3, 0, 4), dtype="f8")
    path = tmp_path / "empty-rows.npy"
    _save(path, array, [array[:1], array[1:]])
    assert np.load(path).shape == (3, 0, 4)

    with pytest.raises(ValueError, match="fewer rows"):
        _save(path, array, [array[:1]])
    assert np.load(path).shape == (3, 0, 4)


def test_scalar_requires_exactly_one_scalar_chunk(tmp_path):
    scalar = np.array(3 + 4j, dtype=">c16")
    path = tmp_path / "scalar.npy"
    _save(path, scalar)
    result = np.load(path)
    assert result.shape == ()
    assert result.dtype == scalar.dtype
    assert result.item() == scalar.item()


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([np.zeros((1, 3), dtype="i4")], "fewer rows"),
        ([np.zeros((3, 3), dtype="i4")], "more rows"),
        ([np.zeros((2, 3), dtype="f4")], "dtype"),
        ([np.zeros((2, 4), dtype="i4")], "shape"),
        ([np.zeros((2, 3, 1), dtype="i4")], "shape"),
        ([[[0, 0, 0], [0, 0, 0]]], "ndarrays"),
    ],
)
def test_invalid_chunks_preserve_destination_and_sidecars(tmp_path, chunks, message):
    path = tmp_path / "output.npy"
    path.write_bytes(b"old complete export")
    sidecar = path.with_suffix(".recipe.json")
    sidecar.write_text("old recipe")
    publications = []

    with pytest.raises((TypeError, ValueError), match=message):
        save_derived_array_chunks(
            path,
            shape=(2, 3),
            dtype="i4",
            chunks=chunks,
            before_publish=lambda: publications.append(True),
        )

    assert path.read_bytes() == b"old complete export"
    assert sidecar.read_text() == "old recipe"
    assert publications == []
    assert set(tmp_path.iterdir()) == {path, sidecar}


@pytest.mark.parametrize("chunks", [[], [np.array(1), np.array(2)], [np.array([1])]])
def test_invalid_scalar_chunks_never_publish(tmp_path, chunks):
    path = tmp_path / "scalar.npy"
    with pytest.raises(ValueError):
        save_derived_array_chunks(path, shape=(), dtype=np.dtype("int64"), chunks=chunks)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_producer_failure_cleans_partial_staged_output(tmp_path):
    path = tmp_path / "output.npy"
    path.write_bytes(b"old export")

    def chunks():
        yield np.arange(3)
        raise RuntimeError("source read failed")

    with pytest.raises(RuntimeError, match="source read failed"):
        save_derived_array_chunks(path, shape=(6,), dtype="int64", chunks=chunks())
    assert path.read_bytes() == b"old export"
    assert set(tmp_path.iterdir()) == {path}


def test_consumed_chunks_are_released_before_producing_the_next(tmp_path):
    def chunks():
        block = np.ones((2, 3))
        reference = weakref.ref(block)
        yield block
        del block
        gc.collect()
        assert reference() is None, "writer retained the preceding output block"
        yield np.full((2, 3), 2.0)

    path = tmp_path / "released.npy"
    save_derived_array_chunks(path, shape=(4, 3), dtype="float64", chunks=chunks())
    np.testing.assert_array_equal(np.load(path), [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]])


def test_each_serialized_write_is_bounded_and_progress_counts_exact_bytes(tmp_path, monkeypatch):
    array = np.asfortranarray(np.arange(257 * 523).reshape(257, 523) * (1 + 2j))[:, ::-1]
    sizes = []
    progress = []
    original = numpy_save._ExportStream.write

    def observe_write(self, value):
        sizes.append(len(value))
        return original(self, value)

    monkeypatch.setattr(numpy_save._ExportStream, "write", observe_write)
    path = tmp_path / "bounded.npy"
    _save(path, array, progress_callback=progress.append)

    assert len(sizes) > 2
    assert max(sizes) <= 1024 * 1024
    assert sum(sizes) == path.stat().st_size == progress[-1]
    assert progress == sorted(set(progress))
    np.testing.assert_array_equal(np.load(path), array)


@pytest.mark.parametrize("boundary", ["before-start", "write", "before-publish"])
def test_cancellation_never_replaces_previous_export(tmp_path, boundary):
    token = CancellationToken()
    array = np.arange(400_000, dtype="i8").reshape(1000, 400)
    path = tmp_path / "cancelled.npy"
    path.write_bytes(b"old export")

    def progress(count):
        if boundary == "write" and count > 1024 * 1024:
            token.cancel()

    def before_publish():
        if boundary == "before-publish":
            token.cancel()

    if boundary == "before-start":
        token.cancel()
    with pytest.raises(EvaluationCancelled):
        _save(
            path,
            array,
            cancellation_token=token,
            progress_callback=progress,
            before_publish=before_publish,
        )
    assert path.read_bytes() == b"old export"
    assert set(tmp_path.iterdir()) == {path}


def test_snapshot_validation_failure_preserves_previous_export(tmp_path):
    path = tmp_path / "changed.npy"
    path.write_bytes(b"old export")

    def before_publish():
        raise RuntimeError("source changed")

    with pytest.raises(RuntimeError, match="source changed"):
        _save(path, np.arange(4), before_publish=before_publish)
    assert path.read_bytes() == b"old export"
    assert set(tmp_path.iterdir()) == {path}


def test_success_wins_cancellation_during_atomic_replace(tmp_path, monkeypatch):
    token = CancellationToken()
    original = os.replace

    def replace_then_cancel(source, destination):
        original(source, destination)
        token.cancel()

    monkeypatch.setattr(numpy_save.os, "replace", replace_then_cancel)
    path = tmp_path / "committed.npy"
    assert _save(path, np.arange(4), cancellation_token=token) == (path,)
    assert token.cancelled
    np.testing.assert_array_equal(np.load(path), np.arange(4))


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks and permission modes")
def test_streamed_publication_follows_symlink_and_preserves_mode(tmp_path):
    target = tmp_path / "actual.npy"
    target.write_bytes(b"old export")
    target.chmod(0o640)
    link = tmp_path / "output.npy"
    link.symlink_to(target.name)

    assert _save(link, np.arange(5)) == (link,)
    assert link.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    np.testing.assert_array_equal(np.load(target), np.arange(5))
    assert set(tmp_path.iterdir()) == {target, link}


@pytest.mark.parametrize("shape", [(-1, 3), (2.0, 3), (True, 3), 4])
def test_invalid_declared_shapes_are_rejected_before_staging(tmp_path, shape):
    with pytest.raises(ValueError, match="shape"):
        save_derived_array_chunks(tmp_path / "invalid.npy", shape=shape, dtype="f8", chunks=())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dtype", [object, np.dtype([("nested", object)]), np.dtype(("f8", (2,)))])
def test_nonserializable_or_top_level_subarray_dtype_is_rejected_before_staging(tmp_path, dtype):
    with pytest.raises(ValueError, match="dtype"):
        save_derived_array_chunks(tmp_path / "invalid.npy", shape=(0,), dtype=dtype, chunks=())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["output", "output.npz", "output.npy.gz"])
def test_streaming_requires_explicit_npy_output(tmp_path, name):
    with pytest.raises(ValueError, match=r"explicit \.npy"):
        _save(tmp_path / name, np.arange(3))
    assert list(tmp_path.iterdir()) == []
