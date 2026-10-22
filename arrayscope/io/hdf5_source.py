"""Detached Cartesian HDF5 reads with per-call, read-only file ownership."""

from __future__ import annotations

from math import prod
from operator import index
from pathlib import Path
from threading import Lock

import numpy as np

from arrayscope.core.array_source import LazySourceArray
from arrayscope.io.hdf5_catalog import (
    HDF5DatasetDescriptor,
    HDF5SourceChanged,
    _resolve_selection,
    dataset_identity,
    describe_hdf5_dataset,
    file_identity,
)
from arrayscope.operations.cancellation import EvaluationCancelled

_READ_BLOCK_BYTES = 8 * 1024 * 1024
_STANDARD_FILTERS = frozenset({1, 2, 3, 4, 5, 6, 32000})  # Includes h5py's bundled LZF codec.


def _check_cancelled(token):
    if token is not None and getattr(token, "cancelled", False):
        raise EvaluationCancelled()


def _point(value, size):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("HDF5 indices must be integers, not booleans")
    value = index(value)
    value = value + size if value < 0 else value
    if not 0 <= value < size:
        raise IndexError(f"HDF5 index is outside axis of size {size}")
    return value


def _selection(shape, spec, *, materialize):
    spec = tuple(spec)
    if len(spec) != len(shape):
        raise ValueError(f"index spec length {len(spec)} must match array ndim {len(shape)}")
    axes = []
    output_shape = []
    index_bytes = 0
    for size, item in zip(shape, spec, strict=True):
        if isinstance(item, (int, np.integer)):
            axes.append(_point(item, size))
        elif isinstance(item, slice):
            values = range(*item.indices(size))
            axes.append(values)
            output_shape.append(len(values))
        elif isinstance(item, (tuple, list, np.ndarray)):
            if isinstance(item, np.ndarray) and item.ndim != 1:
                raise ValueError("HDF5 fancy indices must be one-dimensional")
            values = tuple(_point(value, size) for value in item) if materialize else None
            if not materialize:
                for value in item:
                    _point(value, size)
            axes.append(values)
            output_shape.append(len(item))
            index_bytes += len(item) * 64
        else:
            raise TypeError(f"unsupported HDF5 index item: {item!r}")
    return tuple(axes), tuple(output_shape), index_bytes


def _axis_blocks(values, maximum):
    """Yield positive hyperslabs and their exact output positions/reversal."""
    if isinstance(values, int):
        yield values, None, False
        return
    position = 0
    while position < len(values):
        stop = min(len(values), position + maximum)
        if isinstance(values, range):
            step = values.step
        else:
            step = values[position + 1] - values[position] if position + 1 < stop else 1
            if step not in (-1, 1):
                step = 1
                stop = position + 1
            else:
                end = position + 1
                while end < stop and values[end] - values[end - 1] == step:
                    end += 1
                stop = end
        first, last = values[position], values[stop - 1]
        slab = slice(min(first, last), max(first, last) + 1, abs(step))
        yield slab, slice(position, stop), step < 0
        position = stop


def _blocks(axes, capacities, axis=0, inputs=(), outputs=(), reversals=()):
    if axis == len(axes):
        yield inputs, outputs, reversals
        return
    values = axes[axis]
    capacity = capacities[axis]
    for source, target, reverse in _axis_blocks(values, capacity):
        yield from _blocks(
            axes,
            capacities,
            axis + 1,
            (*inputs, source),
            outputs if target is None else (*outputs, target),
            reversals if target is None else (*reversals, reverse),
        )


class HDF5ArraySource:
    """An immutable dataset selection; no HDF5 handle survives a method call."""

    def __init__(self, path, descriptor, *, file_identity=None):
        if not isinstance(descriptor, HDF5DatasetDescriptor):
            raise TypeError("HDF5 source requires a captured dataset descriptor")
        if not descriptor.lazy_supported:
            raise ValueError(
                descriptor.unsupported_reason or "HDF5 dataset is not supported lazily"
            )
        self.path = Path(path).expanduser().absolute()
        self.descriptor = descriptor
        self.file_identity = (
            descriptor.file_identity if file_identity is None else tuple(file_identity)
        )
        self._closed = False
        self._lock = Lock()
        self.validate_snapshot()

    @property
    def shape(self):
        return self.descriptor.shape

    @property
    def dtype(self):
        return self.descriptor.dtype

    @property
    def nbytes(self):
        return self.descriptor.nbytes

    @property
    def chunk_shape(self):
        return self.descriptor.chunk_shape

    @property
    def label(self):
        return f"hdf5:{self.path.name}:{self.descriptor.key}"

    def _check_open(self):
        with self._lock:
            if self._closed:
                raise ValueError("HDF5 source is closed")

    def _check_file(self):
        try:
            unchanged = file_identity(self.path) == self.file_identity
        except OSError as exc:
            raise HDF5SourceChanged("HDF5 source is unavailable; open it again") from exc
        if not unchanged:
            raise HDF5SourceChanged(
                "HDF5 source changed; open it again before reading or exporting"
            )

    def _dataset(self, handle):
        try:
            current, _key, field = _resolve_selection(handle, self.descriptor.dataset_path)
            if field is not None or dataset_identity(current) != self.descriptor.dataset_identity:
                raise HDF5SourceChanged("HDF5 dataset metadata changed; open it again")
        except (KeyError, ValueError) as exc:
            raise HDF5SourceChanged("HDF5 dataset is unavailable; open it again") from exc
        return current

    def validate_snapshot(self):
        import h5py

        self._check_open()
        self._check_file()
        with h5py.File(self.path, "r", rdcc_nbytes=0) as handle:
            self._dataset(handle)
        self._check_file()

    def estimate_read_nbytes(self, index_spec):
        """Conservative peak buffers, with standard filter decode and no cache."""
        self._check_open()
        _axes, shape, indices = _selection(self.shape, index_spec, materialize=False)
        count = prod(shape)
        if not count:
            return indices
        if any(value not in _STANDARD_FILTERS for value in self.descriptor.filter_ids):
            return None
        result = count * self.dtype.itemsize
        block = min(count * self.descriptor.storage_dtype.itemsize, _READ_BLOCK_BYTES)
        block = max(block, self.descriptor.storage_dtype.itemsize)
        decoded = (
            0
            if self.chunk_shape is None
            else prod(self.chunk_shape) * self.descriptor.storage_dtype.itemsize
        )
        filter_buffers = 4 * decoded + (65536 if self.descriptor.filter_ids else 0)
        assembly = (
            2 * min(result, _READ_BLOCK_BYTES) if self.descriptor.encoding == "complex" else 0
        )
        return result + 2 * block + filter_buffers + assembly + indices

    def read_region(self, index_spec, *, cancellation_token=None):
        import h5py

        self._check_open()  # An accepted call owns its local handle even if close() follows.
        _check_cancelled(cancellation_token)
        axes, shape, _indices = _selection(self.shape, index_spec, materialize=True)
        self._check_file()
        remaining = max(1, _READ_BLOCK_BYTES // self.descriptor.storage_dtype.itemsize)
        capacities = []
        for values in reversed(axes):
            capacity = 1 if isinstance(values, int) else min(max(1, len(values)), remaining)
            capacities.append(capacity)
            remaining = max(1, remaining // capacity)
        capacities.reverse()
        with h5py.File(self.path, "r", rdcc_nbytes=0) as handle:
            dataset = self._dataset(handle)
            result = np.empty(shape, dtype=self.dtype)
            if result.size:
                for inputs, outputs, reversals in _blocks(axes, capacities):
                    _check_cancelled(cancellation_token)
                    if self.descriptor.encoding == "complex":
                        real, imag = self.descriptor.real_field, self.descriptor.imag_field
                        records = dataset.fields([real, imag])[inputs]
                        block = records[real] + 1j * records[imag]
                    elif self.descriptor.encoding == "field":
                        block = dataset.fields(self.descriptor.field_name)[inputs]
                    else:
                        block = dataset[inputs]
                    _check_cancelled(cancellation_token)
                    for axis, reverse in enumerate(reversals):
                        if reverse:
                            block = np.flip(block, axis)
                    result[outputs] = block
            self._dataset(handle)
            self._check_file()
            _check_cancelled(cancellation_token)
        return result

    def close(self):
        # There is no shared handle to invalidate. In-flight calls close their
        # own contexts; subsequent calls fail before opening a file.
        with self._lock:
            self._closed = True


def open_hdf5_source(path, dataset):
    """Open a selected key or previously captured descriptor as a lazy proxy."""
    descriptor = (
        dataset
        if isinstance(dataset, HDF5DatasetDescriptor)
        else describe_hdf5_dataset(path, dataset)
    )
    return LazySourceArray(HDF5ArraySource(path, descriptor))
