"""Progressive file reading: byte/slice-level progress and streaming fills.

This module owns the chunked readers behind ``load_path(progress=...)``.
Design goals:

- **Progress**: every reader that can know its byte or slice budget reports
  monotonic ``LoadProgress`` updates through a plain callable. Formats whose
  libraries only offer a monolithic read (NIfTI, single DICOM, text) report
  indeterminate stages instead of fake percentages.
- **Streaming**: formats whose on-disk layout lets us pre-allocate the final
  array after a cheap header probe (.npy, .cfl, Philips .rec) expose a
  synchronized array source before the bytes arrive, so a viewer can open
  immediately and watch completed writes appear without reading a buffer
  while the loader mutates it.
- **Cancellation**: readers poll a ``threading.Event``-like object between
  chunks and raise :class:`LoadCancelled`.

Everything here is Qt-free; GUI marshalling lives in ``arrayscope.app``.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.lib.format as _npformat

from arrayscope.core.array_source import LazySourceArray, NdArraySource

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


class ProgressiveArraySource(NdArraySource):
    """One-writer array source with atomic, detached region publication.

    The loader owns the backing array and mutates it only inside
    :meth:`write_transaction`. Evaluation reads use :meth:`read_region`,
    which holds the same lock while copying the requested region. A reader
    therefore observes either the old zero-filled region or a completed
    write, never an in-place mutation.
    """

    def __init__(self, array, *, label="progressive load"):
        super().__init__(array, label=label)
        self._lock = threading.RLock()

    def read_region(self, index_spec, *, cancellation_token=None):
        with self._lock:
            return np.array(
                super().read_region(index_spec, cancellation_token=cancellation_token), copy=True
            )

    @contextmanager
    def write_transaction(self):
        with self._lock:
            yield self._array

    def write_bytes(self, start, payload):
        values = np.frombuffer(payload, dtype=np.uint8)
        with self.write_transaction() as array:
            view = _flat_byte_view(array)
            view[int(start) : int(start) + values.size] = values

    def write_flat(self, start, values):
        with self.write_transaction() as array:
            flat = array.ravel(order="K")
            flat[int(start) : int(start) + len(values)] = values


@dataclass
class StreamingProbe:
    """Early result of a streaming-capable load.

    ``data`` is the final-shaped synchronized source over a zero-filled
    destination owned by the reader thread. Region reads are detached and
    atomic with respect to loader writes. ``axes`` / ``metadata`` carry what
    is already known from the header.
    """

    data: LazySourceArray
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


def _readinto_chunked(
    fp,
    data,
    *,
    progress,
    cancel,
    bytes_offset=0,
    bytes_total=None,
    published_source=None,
):
    """Fill ``data`` from ``fp`` sequentially, reporting byte progress."""
    view = _flat_byte_view(data)
    total = view.nbytes if bytes_total is None else bytes_total
    done = 0
    while done < view.nbytes:
        _check_cancel(cancel)
        end = min(done + READ_CHUNK_BYTES, view.nbytes)
        if published_source is None:
            read = fp.readinto(memoryview(view[done:end]))
        else:
            payload = fp.read(end - done)
            read = len(payload)
            if read:
                published_source.write_bytes(done, payload)
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
        order = "F" if fortran_order else "C"
        data = np.empty(shape, dtype=dtype, order=order)
        published_source = None
        if on_streaming_probe is not None:
            data.fill(0)
            published_source = ProgressiveArraySource(data, label=str(filepath))
            on_streaming_probe(
                StreamingProbe(
                    data=LazySourceArray(published_source, materialize_budget_bytes=None),
                    axes=None,
                    metadata={
                        "source_path": str(filepath),
                        "detected_format": "numpy",
                        "shape": tuple(data.shape),
                        "dtype": str(data.dtype),
                    },
                )
            )
        _readinto_chunked(
            fp,
            data,
            progress=progress,
            cancel=cancel,
            published_source=published_source,
        )
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
    published_source = None
    if on_streaming_probe is not None:
        data.fill(0)
        published_source = ProgressiveArraySource(data, label=str(filepath))
        on_streaming_probe(
            StreamingProbe(
                data=LazySourceArray(published_source, materialize_budget_bytes=None),
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
        if published_source is None:
            dst[start:end] = src[start:end]
        else:
            published_source.write_flat(start, src[start:end])
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
    "ProgressiveArraySource",
    "StreamingProbe",
    "load_cfl_progressive",
    "load_npy_progressive",
]
