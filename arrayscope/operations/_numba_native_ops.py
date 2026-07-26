"""Measured optional accelerator for native per-axis normalization.

Log-magnitude and soft-threshold benchmarked faster too, but mixing their Numba
whole-array path with NumPy strided-region fallbacks failed the exact
ELEMENTWISE conformance oracle by a few ULP.  They therefore do not live here.
Normalization is OPAQUE over its selected axis and earned a 6-7x win for the
contiguous last-axis case and a 2.1-2.3x win when a contiguous input needed one
axis-move copy. The shared lazy runtime still returns ``None`` until compilation
finishes, and already-strided inputs fall back rather than stacking another
copy on unknown source layout.
"""

from __future__ import annotations

import numpy as np

from arrayscope.core import numba_runtime

NUMBA_AVAILABLE = numba_runtime.NUMBA_AVAILABLE
_SUPPORTED_DTYPES = frozenset((np.dtype(np.float32), np.dtype(np.complex64)))


def _build_kernels():
    from numba import njit, prange

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def normalize_last_axis(data):
        rows = data.reshape((-1, data.shape[-1]))
        output = np.empty_like(rows)
        for row in prange(rows.shape[0]):
            norm_squared = np.float32(0.0)
            for column in range(rows.shape[1]):
                magnitude = abs(rows[row, column])
                norm_squared += magnitude * magnitude
            norm = np.sqrt(norm_squared)
            if norm == 0:
                for column in range(rows.shape[1]):
                    output[row, column] = 0
            else:
                for column in range(rows.shape[1]):
                    output[row, column] = rows[row, column] / norm
        return output.reshape(data.shape)

    kernels = {"normalize_last_axis": normalize_last_axis}
    for dtype in _SUPPORTED_DTYPES:
        sample = np.ones((2, 3), dtype=dtype)
        normalize_last_axis(sample)
    return kernels


_GROUP = numba_runtime.register("native_ops", _build_kernels)


def prewarm() -> None:
    """Compile all measured kernels synchronously; tests/benchmarks only."""

    _GROUP.prewarm()


def is_ready() -> bool:
    return NUMBA_AVAILABLE and _GROUP.ready()


def _kernels_for(data):
    if not NUMBA_AVAILABLE:
        return None
    array = np.asarray(data)
    if array.dtype not in _SUPPORTED_DTYPES or not array.flags["C_CONTIGUOUS"]:
        return None
    return _GROUP.get()


def normalize_if_ready(data, axis):
    array = np.asarray(data)
    if array.ndim == 0:
        return None
    axis = int(axis) % array.ndim
    kernels = _kernels_for(array)
    if kernels is None:
        return None
    if axis == array.ndim - 1:
        return kernels["normalize_last_axis"](array)
    moved = np.ascontiguousarray(np.moveaxis(array, axis, -1))
    normalized = kernels["normalize_last_axis"](moved)
    return np.moveaxis(normalized, -1, axis)
