"""Out-of-core array source protocol and adapters.

An :class:`ArraySource` provides explicit, bounded region reads over array
data that may live outside process memory (memory-mapped files today,
chunked stores later). Request planning, cancellation, and memory budgets
live *above* the adapter (see ``operations/source_read.py``); an adapter
only transports and decodes the exact region it is asked for.

This module is Qt-free and must stay importable without the GUI stack.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from arrayscope.core.memory_budget import format_bytes

GiB = 1024 * 1024 * 1024

DEFAULT_SOURCE_READ_BUDGET_BYTES = 1 * GiB
DEFAULT_SOURCE_MATERIALIZE_BUDGET_BYTES = 2 * GiB


class SourceReadRefused(RuntimeError):
    """A lazy-source read was refused because it would exceed its byte budget.

    Raised by the budget guard above the adapter, never by transport failures.
    """

    def __init__(
        self, message: str, *, requested_nbytes: int | None = None, budget_bytes: int | None = None
    ) -> None:
        super().__init__(message)
        self.requested_nbytes = None if requested_nbytes is None else int(requested_nbytes)
        self.budget_bytes = None if budget_bytes is None else int(budget_bytes)


@runtime_checkable
class ArraySource(Protocol):
    """Explicit-region reads over out-of-core array data.

    ``index_spec`` is one item per axis of :attr:`shape`: an ``int`` (drop the
    axis), a ``slice``, or a tuple of ints (fancy selection along that axis).
    ``read_region`` returns an in-memory ``np.ndarray`` that does not alias
    the backing store. Adapters may honor ``cancellation_token`` between
    internal chunks; single-read adapters are free to ignore it because the
    caller checks it around the read.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def chunk_shape(self) -> tuple[int, ...] | None: ...

    @property
    def label(self) -> str: ...

    def read_region(
        self, index_spec: tuple, *, cancellation_token: object | None = None
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class NdArraySource:
    """Adapt an ndarray-like backing store (including ``np.memmap``) to :class:`ArraySource`."""

    def __init__(
        self,
        array,
        *,
        label: str | None = None,
        chunk_shape: tuple[int, ...] | None = None,
        close=None,
    ) -> None:
        self._array = array
        self._label = str(label) if label is not None else type(array).__name__
        self._chunk_shape = (
            None if chunk_shape is None else tuple(int(size) for size in chunk_shape)
        )
        self._close = close
        self._closed = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def chunk_shape(self) -> tuple[int, ...] | None:
        return self._chunk_shape

    @property
    def label(self) -> str:
        return self._label

    def read_region(
        self, index_spec: tuple, *, cancellation_token: object | None = None
    ) -> np.ndarray:
        del cancellation_token  # single bounded read; the caller checks around it
        result = read_index_spec(self._array, index_spec)
        if isinstance(result, np.memmap):
            result = np.array(result)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            self._close()


class ScaledArraySource:
    """Adapt an integer (or otherwise narrow) backing store that carries an
    affine ``value = raw * slope + inter`` rescaling.

    The backing store — typically a ``np.memmap`` of the on-disk voxels — stays
    resident at its compact dtype (int16 is 2 bytes vs. 4 for float32, 8 for
    float64). Only the region actually read is expanded, and it is expanded
    straight into ``out_dtype`` (default float32) so no float64 temporary is
    built along the way. :attr:`nbytes` reports the *expanded* footprint so
    memory planning budgets the real cost of materializing.
    """

    def __init__(
        self,
        array,
        *,
        slope: float = 1.0,
        inter: float = 0.0,
        out_dtype=np.float32,
        label: str | None = None,
        chunk_shape: tuple[int, ...] | None = None,
        close=None,
    ) -> None:
        self._array = array
        self._slope = float(slope)
        self._inter = float(inter)
        self._out_dtype = np.dtype(out_dtype)
        self._label = str(label) if label is not None else type(array).__name__
        self._chunk_shape = None if chunk_shape is None else tuple(int(size) for size in chunk_shape)
        self._close = close
        self._closed = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self._array.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._out_dtype

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self._out_dtype.itemsize

    @property
    def chunk_shape(self) -> tuple[int, ...] | None:
        return self._chunk_shape

    @property
    def label(self) -> str:
        return self._label

    def read_region(self, index_spec: tuple, *, cancellation_token: object | None = None) -> np.ndarray:
        del cancellation_token  # single bounded read; the caller checks around it
        raw = read_index_spec(self._array, index_spec)
        # Expand directly into the output dtype: astype makes the one copy we
        # need (never aliasing the memmap), then scale in place.
        result = np.asarray(raw).astype(self._out_dtype, copy=True)
        if self._slope != 1.0:
            result *= self._out_dtype.type(self._slope)
        if self._inter != 0.0:
            result += self._out_dtype.type(self._inter)
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            self._close()


class LazySourceArray:
    """Document base-data proxy backed by an :class:`ArraySource`.

    Exposes ``shape``/``dtype``/``nbytes`` so planning code can plan without
    reading, and delegates explicit region reads to the source. Implicit full
    materialization (``np.asarray`` and friends) goes through
    :meth:`materialize`, which refuses beyond its byte budget so "lazy" can
    never silently mean "decode everything".
    """

    def __init__(
        self,
        source: ArraySource,
        *,
        materialize_budget_bytes: int = DEFAULT_SOURCE_MATERIALIZE_BUDGET_BYTES,
    ) -> None:
        self._source = source
        self._materialize_budget_bytes = (
            None if materialize_budget_bytes is None else int(materialize_budget_bytes)
        )

    @property
    def source(self) -> ArraySource:
        return self._source

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(size) for size in self._source.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._source.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return int(self._source.nbytes)

    @property
    def chunk_shape(self) -> tuple[int, ...] | None:
        return self._source.chunk_shape

    @property
    def label(self) -> str:
        return self._source.label

    def __len__(self) -> int:
        shape = self.shape
        if not shape:
            raise TypeError("len() of unsized lazy source array")
        return int(shape[0])

    def __repr__(self) -> str:
        return f"LazySourceArray(label={self.label!r}, shape={self.shape}, dtype={self.dtype}, nbytes={format_bytes(self.nbytes)})"

    def read_region(
        self, index_spec: tuple, *, cancellation_token: object | None = None
    ) -> np.ndarray:
        return self._source.read_region(tuple(index_spec), cancellation_token=cancellation_token)

    def materialize(self, *, budget_bytes: int | None | str = "default") -> np.ndarray:
        if budget_bytes == "default":
            budget_bytes = self._materialize_budget_bytes
        if budget_bytes is not None and self.nbytes > int(budget_bytes):
            raise SourceReadRefused(
                f"refusing to materialize lazy source {self.label!r}: "
                f"{format_bytes(self.nbytes)} exceeds the {format_bytes(int(budget_bytes))} materialization budget",
                requested_nbytes=self.nbytes,
                budget_bytes=int(budget_bytes),
            )
        return np.asarray(self.read_region(tuple(slice(None) for _ in self.shape)))

    def __array__(self, dtype=None):
        data = self.materialize()
        return data if dtype is None else np.asarray(data, dtype=dtype)

    def close(self) -> None:
        self._source.close()


def is_lazy_source_array(obj) -> bool:
    return isinstance(obj, LazySourceArray)


def read_index_spec(array, index_spec: tuple):
    """Apply a plain per-axis index spec (int | slice | tuple of ints) to ``array``."""

    spec = tuple(index_spec)
    if len(spec) != int(np.ndim(array)):
        raise ValueError(
            f"index spec length {len(spec)} must match array ndim {int(np.ndim(array))}"
        )
    for item in spec:
        if not isinstance(item, (int, np.integer, slice, tuple, list, np.ndarray)):
            raise TypeError(f"unsupported index item: {item!r}")
    if not any(isinstance(item, (tuple, list, np.ndarray)) for item in spec):
        return array[spec]
    result = array
    result_axis = 0
    for item in spec:
        if isinstance(item, (int, np.integer)):
            result = np.take(result, int(item), axis=result_axis)
            continue
        if isinstance(item, (tuple, list, np.ndarray)):
            result = np.take(result, np.asarray(item, dtype=np.int64), axis=result_axis)
        elif isinstance(item, slice):
            slicer = [slice(None)] * int(np.ndim(result))
            slicer[result_axis] = item
            result = result[tuple(slicer)]
        else:
            raise TypeError(f"unsupported index item: {item!r}")
        result_axis += 1
    return result


__all__ = [
    "DEFAULT_SOURCE_MATERIALIZE_BUDGET_BYTES",
    "DEFAULT_SOURCE_READ_BUDGET_BYTES",
    "ArraySource",
    "LazySourceArray",
    "NdArraySource",
    "ScaledArraySource",
    "SourceReadRefused",
    "is_lazy_source_array",
    "read_index_spec",
]
