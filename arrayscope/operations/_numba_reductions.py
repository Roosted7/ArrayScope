"""Optional numba-accelerated reduction kernels for the operation pipeline.

This module provides a fused root-sum-squares (RSS) reduction that avoids the
full-volume ``np.abs(data) ** 2`` temporary allocated by the numpy reference in
``pipeline.RootSumSquares``.  It plugs into the shared accelerator runtime
(:mod:`arrayscope.core.numba_runtime`) under the group name ``"reductions"``,
which supplies the lazy numba import, off-the-hot-path compile, and selective
prewarm; the numpy reference in ``pipeline`` stays intact as the always-correct
fallback.

Accuracy / policy notes:

* Integer widening is performed by the caller (``pipeline``) before we ever see
  the data, so kernels only handle real/complex floating dtypes.
* Accumulation dtype matches the numpy reference output dtype: single-precision
  inputs (float32 / complex64) accumulate and return float32; double-precision
  inputs (float64 / complex128) accumulate and return float64.
* Only reductions over the *last axis of a C-contiguous array* are accelerated.
  Reducing another axis would require ``moveaxis`` to materialise a
  strided->contiguous copy whose cost cancels the fused-kernel win (measured);
  those cases return ``None`` and fall back to numpy.
"""

from __future__ import annotations

import numpy as np

from arrayscope.core import numba_runtime

# Mirror the runtime's cheap probe as a module constant so tests can monkeypatch
# it to force the numpy fallback path.
NUMBA_AVAILABLE = numba_runtime.NUMBA_AVAILABLE


def _build_kernels() -> dict[np.dtype, object]:
    """Import numba and construct + force-compile the RSS kernels (once)."""
    from numba import njit, prange

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _rss_last_axis_f32(flat):  # (M, N) float32 -> (M,) float32
        rows, cols = flat.shape
        out = np.empty(rows, dtype=np.float32)
        for m in prange(rows):
            acc = np.float32(0.0)
            for n in range(cols):
                v = flat[m, n]
                acc += v * v
            out[m] = np.sqrt(acc)
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _rss_last_axis_f64(flat):  # (M, N) float64 -> (M,) float64
        rows, cols = flat.shape
        out = np.empty(rows, dtype=np.float64)
        for m in prange(rows):
            acc = np.float64(0.0)
            for n in range(cols):
                v = flat[m, n]
                acc += v * v
            out[m] = np.sqrt(acc)
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _rss_last_axis_c64(flat):  # (M, N) complex64 -> (M,) float32
        rows, cols = flat.shape
        out = np.empty(rows, dtype=np.float32)
        for m in prange(rows):
            acc = np.float32(0.0)
            for n in range(cols):
                z = flat[m, n]
                re = np.float32(z.real)
                im = np.float32(z.imag)
                acc += re * re + im * im
            out[m] = np.sqrt(acc)
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _rss_last_axis_c128(flat):  # (M, N) complex128 -> (M,) float64
        rows, cols = flat.shape
        out = np.empty(rows, dtype=np.float64)
        for m in prange(rows):
            acc = np.float64(0.0)
            for n in range(cols):
                z = flat[m, n]
                re = z.real
                im = z.imag
                acc += re * re + im * im
            out[m] = np.sqrt(acc)
        return out

    kernels = {
        np.dtype(np.float32): _rss_last_axis_f32,
        np.dtype(np.float64): _rss_last_axis_f64,
        np.dtype(np.complex64): _rss_last_axis_c64,
        np.dtype(np.complex128): _rss_last_axis_c128,
    }
    # Force compilation of every specialization off the hot path.
    for dtype, kernel in kernels.items():
        kernel(np.ones((2, 2), dtype=dtype))
    return kernels


_GROUP = numba_runtime.register("reductions", _build_kernels)


def prewarm() -> None:
    """Compile the RSS kernels (blocking). Prefer :func:`prewarm_async`."""

    _GROUP.prewarm()


def prewarm_async() -> None:
    """Kick off compilation on a background daemon thread (non-blocking)."""

    _GROUP.prewarm_async()


def is_ready() -> bool:
    """Return True when the numba kernels are compiled and usable."""

    return NUMBA_AVAILABLE and _GROUP.ready()


def rss_if_ready(data, axis: int):
    """Return the fused numba RSS result, or ``None`` for the numpy fallback.

    ``None`` is returned when numba is unavailable, not yet warm (a background
    prewarm is kicked off in that case), or the input is not the accelerated
    contiguous last-axis case.  The caller must then use the numpy reference.

    ``data`` must already be a real/complex floating array (integers are widened
    by the caller); ``axis`` is a validated axis index.
    """
    if not NUMBA_AVAILABLE:
        return None
    kernels = _GROUP.get()  # kicks a background warm and returns None until ready
    if kernels is None:
        return None

    array = np.asarray(data)
    kernel = kernels.get(array.dtype)
    if kernel is None:
        return None

    ndim = array.ndim
    if ndim == 0:
        return None
    if axis < 0:
        axis += ndim
    if axis != ndim - 1:
        # Non-last axis: moveaxis would force a strided->contiguous copy whose
        # cost cancels the fused-kernel win (measured). Fall back to numpy.
        return None
    if not array.flags["C_CONTIGUOUS"]:
        # A non-contiguous last axis (e.g. a transposed view) also needs a copy
        # to reshape into (M, N). Not worth it -- fall back to numpy.
        return None

    n = array.shape[-1]
    out_shape = array.shape[:-1]
    flat = array.reshape(-1, n)  # free view: array is C-contiguous
    result = kernel(flat)
    return result.reshape(out_shape)
