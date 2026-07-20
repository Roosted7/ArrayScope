"""Contract tests for the Julia/MATLAB invocation-wrapper handoff.

The wrappers in ``wrappers/julia`` and ``wrappers/matlab`` write raw,
uncompressed ``.npy`` (format 1.0) files declared ``fortran_order: True`` and
launch the CLI with ``--mmap --consume``. These tests pin the Python side of
that contract: byte-level acceptance of wrapper-style files, lazy
copy-on-write loading, and safe handoff-file consumption. The writer below
mirrors the wrapper implementations line for line so a drift in either
breaks here first.
"""

import sys

import numpy as np
import pytest

from arrayscope.io.file_interpreters import consume_handoff_file, load_path

_DESCR = {
    np.dtype(np.bool_): "|b1",
    np.dtype(np.int8): "|i1",
    np.dtype(np.uint8): "|u1",
    np.dtype(np.int16): "<i2",
    np.dtype(np.uint16): "<u2",
    np.dtype(np.int32): "<i4",
    np.dtype(np.uint32): "<u4",
    np.dtype(np.int64): "<i8",
    np.dtype(np.uint64): "<u8",
    np.dtype(np.float16): "<f2",
    np.dtype(np.float32): "<f4",
    np.dtype(np.float64): "<f8",
    np.dtype(np.complex64): "<c8",
    np.dtype(np.complex128): "<c16",
}


def wrapper_npy_bytes(arr):
    """Emit .npy bytes exactly the way the Julia/MATLAB wrappers do.

    Column-major (Fortran) data section written verbatim, format 1.0 header
    space-padded so the data section starts on a 64-byte boundary.
    """
    if arr.ndim == 1:
        shape_str = f"({arr.shape[0]},)"
    else:
        shape_str = "(" + ", ".join(str(s) for s in arr.shape) + ")"
    dict_str = f"{{'descr': '{_DESCR[arr.dtype]}', 'fortran_order': True, 'shape': {shape_str}, }}"
    unpadded = 6 + 2 + 2 + len(dict_str) + 1
    pad = (64 - unpadded % 64) % 64
    header = (dict_str + " " * pad + "\n").encode("latin1")
    out = b"\x93NUMPY" + bytes([1, 0])
    out += len(header).to_bytes(2, "little")
    out += header
    out += np.asfortranarray(arr).tobytes(order="F")
    return out


def write_wrapper_npy(tmp_path, arr, name="handoff.npy"):
    path = tmp_path / name
    path.write_bytes(wrapper_npy_bytes(arr))
    return path


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        (np.arange(60, dtype=np.float32) - 30).reshape(3, 5, 2, 2),
        (np.arange(12) + 1j * np.arange(12)[::-1]).astype(np.complex64).reshape(3, 4),
        (np.arange(8) - 4.5j * np.arange(8)).astype(np.complex128).reshape(2, 2, 2),
        np.array([True, False, True, True]),
        np.arange(-5, 5, dtype=np.int16),
        np.arange(10, dtype=np.uint64).reshape(5, 2),
        np.linspace(-1, 1, 6, dtype=np.float16).reshape(2, 3),
    ],
    ids=lambda a: f"{a.dtype}-{'x'.join(map(str, a.shape))}",
)
def test_wrapper_npy_roundtrip(tmp_path, arr):
    path = write_wrapper_npy(tmp_path, arr)

    loaded = load_path(path)

    assert loaded.data.dtype == arr.dtype
    assert loaded.data.shape == arr.shape
    np.testing.assert_array_equal(loaded.data, arr)
    assert loaded.metadata["detected_format"] == "numpy"


def test_wrapper_npy_header_layout(tmp_path):
    raw = wrapper_npy_bytes(np.zeros((7, 3), dtype=np.complex128))

    assert raw[:6] == b"\x93NUMPY"
    assert raw[6:8] == bytes([1, 0])  # format 1.0
    header_len = int.from_bytes(raw[8:10], "little")
    data_start = 10 + header_len
    assert data_start % 64 == 0  # aligned data section (mmap-friendly)
    header = raw[10:data_start].decode("latin1")
    assert header.endswith("\n")
    assert "'fortran_order': True" in header


def test_wrapper_npy_matches_native_numpy_reader(tmp_path):
    """np.load itself must agree with what load_path returns."""
    arr = np.random.default_rng(0).normal(size=(4, 5, 6)).astype(np.float32)
    path = write_wrapper_npy(tmp_path, arr)

    native = np.load(path)
    via_load_path = load_path(path).data

    np.testing.assert_array_equal(native, arr)
    np.testing.assert_array_equal(via_load_path, arr)
    assert native.flags["F_CONTIGUOUS"]


def test_load_path_mmap_is_lazy_copy_on_write(tmp_path):
    arr = np.arange(30, dtype=np.float64).reshape(5, 6)
    path = write_wrapper_npy(tmp_path, arr)

    loaded = load_path(path, mmap=True)

    assert isinstance(loaded.data, np.memmap)
    np.testing.assert_array_equal(loaded.data, arr)

    # Copy-on-write: in-process edits must not reach the file.
    loaded.data[0, 0] = -999.0
    np.testing.assert_array_equal(load_path(path).data, arr)


def test_load_path_mmap_false_returns_plain_array(tmp_path):
    path = write_wrapper_npy(tmp_path, np.ones(4))

    loaded = load_path(path)

    assert not isinstance(loaded.data, np.memmap)


def test_consume_handoff_file(tmp_path):
    path = write_wrapper_npy(tmp_path, np.ones(3))

    assert consume_handoff_file(path) is True
    assert not path.exists()
    assert consume_handoff_file(path) is False  # already gone: best effort


@pytest.mark.skipif(sys.platform == "win32", reason="Windows cannot unlink a memory-mapped file")
def test_mmap_survives_consume_on_posix(tmp_path):
    """--mmap --consume: the mapping must stay readable after the unlink."""
    arr = np.arange(1000, dtype=np.float32).reshape(10, 100)
    path = write_wrapper_npy(tmp_path, arr)

    loaded = load_path(path, mmap=True)
    assert consume_handoff_file(path) is True
    assert not path.exists()

    np.testing.assert_array_equal(np.asarray(loaded.data), arr)
