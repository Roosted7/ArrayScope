"""Progressive file reading: byte/slice-level progress and streaming fills.

This module owns the chunked readers behind ``load_path(progress=...)``.
Design goals:

- **Progress**: every reader that can know its byte or slice budget reports
  monotonic ``LoadProgress`` updates through a plain callable. Formats whose
  libraries only offer a monolithic read (NIfTI, single DICOM, text) report
  indeterminate stages instead of fake percentages.
- **Streaming**: formats whose on-disk layout lets us pre-allocate the final
  array after a cheap header probe (.npy, .cfl, Philips .rec) expose the
  destination buffer *before* the bytes arrive, so a viewer can open
  immediately and watch the fill. The buffer handed out by
  :class:`StreamingProbe` is the same memory the reader fills in place.
- **Cancellation**: readers poll a ``threading.Event``-like object between
  chunks and raise :class:`LoadCancelled`.

Everything here is Qt-free; GUI marshalling lives in ``arrayscope.app``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.lib.format as _npformat

# Chunk size for sequential reads/copies. Large enough to reach disk
# throughput, small enough that progress updates and cancellation polls
# stay responsive (a few dozen per second on typical SSDs).
READ_CHUNK_BYTES = 16 * 1024 * 1024


class LoadCancelled(RuntimeError):
    """Raised by progressive readers when the cancel event is set."""


@dataclass(frozen=True)
class LoadProgress:
    """One progress observation from a reader.

    ``fraction`` is in [0, 1] when the total work is known, else None
    (indeterminate). ``stage`` is a coarse machine-readable phase:
    "probing", "reading", "converting", "finalizing".
    """

    stage: str
    fraction: float | None
    message: str = ""
    bytes_done: int | None = None
    bytes_total: int | None = None


@dataclass
class StreamingProbe:
    """Early result of a streaming-capable load.

    ``data`` is the final-shaped destination array, pre-allocated and being
    filled in place by the reader thread. ``axes`` / ``metadata`` carry what
    is already known from the header. Readers only ever *write* regions they
    have fully decoded, so partially loaded regions read as zeros.
    """

    data: np.ndarray
    axes: tuple | None
    metadata: dict


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise LoadCancelled("file load cancelled")


def _emit(progress, *, stage, fraction, message="", bytes_done=None, bytes_total=None):
    if progress is not None:
        progress(
            LoadProgress(
                stage=stage,
                fraction=fraction,
                message=message,
                bytes_done=bytes_done,
                bytes_total=bytes_total,
            )
        )


def _read_npy_header(fp):
    """Return (shape, fortran_order, dtype) using numpy's format helpers."""
    version = _npformat.read_magic(fp)
    try:
        return _npformat._read_array_header(fp, version)
    except AttributeError:  # numpy without the private helper
        if version == (1, 0):
            return _npformat.read_array_header_1_0(fp)
        if version == (2, 0):
            return _npformat.read_array_header_2_0(fp)
        raise ValueError(f"unsupported .npy format version: {version}") from None


def _flat_byte_view(data):
    """A uint8 view over ``data`` in memory order (no copy)."""
    flat = data.ravel(order="K")
    return flat.view(np.uint8)


def _readinto_chunked(fp, data, *, progress, cancel, bytes_offset=0, bytes_total=None):
    """Fill ``data`` from ``fp`` sequentially, reporting byte progress."""
    view = _flat_byte_view(data)
    total = view.nbytes if bytes_total is None else bytes_total
    done = 0
    while done < view.nbytes:
        _check_cancel(cancel)
        end = min(done + READ_CHUNK_BYTES, view.nbytes)
        read = fp.readinto(memoryview(view[done:end]))
        if not read:
            raise ValueError("file ended before the array data was complete (truncated file?)")
        done += read
        _emit(
            progress,
            stage="reading",
            fraction=(bytes_offset + done) / total if total else 1.0,
            bytes_done=bytes_offset + done,
            bytes_total=total,
        )


def load_npy_progressive(filepath, *, progress=None, cancel=None, on_streaming_probe=None):
    """Chunked eager .npy read; equivalent to ``np.load`` for plain arrays."""
    filepath = Path(filepath)
    with open(filepath, "rb") as fp:
        _emit(progress, stage="probing", fraction=None, message="Reading header")
        shape, fortran_order, dtype = _read_npy_header(fp)
        if dtype.hasobject:
            # Match np.load(allow_pickle=False) behaviour for object arrays.
            raise ValueError(f"Object arrays cannot be loaded progressively: {filepath}")
        data = np.empty(shape, dtype=dtype, order="F" if fortran_order else "C")
        if on_streaming_probe is not None:
            on_streaming_probe(
                StreamingProbe(
                    data=data,
                    axes=None,
                    metadata={
                        "source_path": str(filepath),
                        "detected_format": "numpy",
                        "shape": tuple(data.shape),
                        "dtype": str(data.dtype),
                    },
                )
            )
        _readinto_chunked(fp, data, progress=progress, cancel=cancel)
    _emit(progress, stage="finalizing", fraction=1.0)
    return data


def load_cfl_progressive(filepath, *, progress=None, cancel=None, on_streaming_probe=None):
    """Chunked eager BART .cfl read into a squeezed, F-ordered array."""
    from arrayscope.io.file_interpreters import BartLoader, remove_trailing_singletons

    filepath = Path(filepath)
    _emit(progress, stage="probing", fraction=None, message="Reading header")
    loader = BartLoader(filepath)
    mapped = np.memmap(filepath, dtype=np.complex64, mode="r", shape=loader.dims, order="F")
    mapped = remove_trailing_singletons(mapped)

    data = np.empty(mapped.shape, dtype=np.complex64, order="F")
    if on_streaming_probe is not None:
        on_streaming_probe(
            StreamingProbe(
                data=data,
                axes=None,
                metadata={
                    "source_path": str(filepath),
                    "detected_format": "cfl",
                    "shape": tuple(data.shape),
                    "dtype": str(data.dtype),
                },
            )
        )

    dst = data.ravel(order="K")
    src = mapped.ravel(order="K")
    total = dst.nbytes
    chunk_elements = max(1, READ_CHUNK_BYTES // data.dtype.itemsize)
    for start in range(0, dst.size, chunk_elements):
        _check_cancel(cancel)
        end = min(start + chunk_elements, dst.size)
        dst[start:end] = src[start:end]
        _emit(
            progress,
            stage="reading",
            fraction=(end * data.dtype.itemsize) / total if total else 1.0,
            bytes_done=end * data.dtype.itemsize,
            bytes_total=total,
        )
    del src, mapped
    _emit(progress, stage="finalizing", fraction=1.0)
    return data


__all__ = [
    "READ_CHUNK_BYTES",
    "LoadCancelled",
    "LoadProgress",
    "StreamingProbe",
    "load_cfl_progressive",
    "load_npy_progressive",
]
