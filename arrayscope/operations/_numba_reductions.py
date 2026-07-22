"""Optional numba-accelerated reduction kernels for the operation pipeline.

This module provides a fused root-sum-squares (RSS) reduction that avoids the
full-volume ``np.abs(data) ** 2`` temporary allocated by the numpy reference in
``pipeline.RootSumSquares``.  It follows the repository's optional-accelerator
pattern:

* the presence of ``numba`` is detected cheaply (via ``find_spec``) and the
  heavy ``import numba`` is deferred until :func:`prewarm` -- so importing this
  module (and therefore ``pipeline``) never pays numba's ~0.4 s import cost,
* the numpy reference in ``pipeline`` stays intact as the always-correct fallback,
* the JIT is compiled off the hot path via :func:`prewarm` (kicked off in a
  background daemon thread on first use), so the visible path never blocks on
  either the numba import or kernel compilation,
* :func:`rss_if_ready` returns ``None`` whenever the accelerator is unavailable,
  not yet warm, or the input is not in the contiguous-friendly case, and the
  caller then uses the numpy reference.

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

import importlib.util
import threading

import numpy as np

# Cheap availability probe: does not import numba (that costs ~0.4 s and would
# be paid on every ``pipeline`` import). The real import happens in prewarm().
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None

_READY = threading.Event()
_WARM_LOCK = threading.Lock()
_WARM_THREAD_LOCK = threading.Lock()
_warm_thread: threading.Thread | None = None

# Populated by prewarm(): {np.dtype: compiled njit kernel}.
_KERNELS: dict[np.dtype, object] = {}


def _build_kernels() -> dict[np.dtype, object]:
    """Import numba and construct the compiled RSS kernels (called once)."""
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

    return {
        np.dtype(np.float32): _rss_last_axis_f32,
        np.dtype(np.float64): _rss_last_axis_f64,
        np.dtype(np.complex64): _rss_last_axis_c64,
        np.dtype(np.complex128): _rss_last_axis_c128,
    }


def prewarm() -> None:
    """Import numba and compile the RSS kernels; safe to call repeatedly.

    Blocks the calling thread until compilation finishes, so callers on the
    visible path should use :func:`prewarm_async` (or rely on the lazy trigger
    inside :func:`rss_if_ready`) instead of calling this directly.
    """
    if not NUMBA_AVAILABLE or _READY.is_set():
        return
    with _WARM_LOCK:
        if _READY.is_set():
            return
        try:
            kernels = _build_kernels()
        except Exception:
            # numba present but unusable (e.g. broken LLVM); stay on numpy.
            globals()["NUMBA_AVAILABLE"] = False
            return
        # Force compilation of every specialization off the hot path.
        for dtype, kernel in kernels.items():
            kernel(np.ones((2, 2), dtype=dtype))
        _KERNELS.update(kernels)
        _READY.set()


def prewarm_async() -> None:
    """Kick off :func:`prewarm` in a background daemon thread (non-blocking).

    Idempotent: at most one warm-up thread runs at a time, and calls after the
    kernels are ready are no-ops.
    """
    global _warm_thread
    if not NUMBA_AVAILABLE or _READY.is_set():
        return
    with _WARM_THREAD_LOCK:
        if _READY.is_set():
            return
        if _warm_thread is not None and _warm_thread.is_alive():
            return
        _warm_thread = threading.Thread(
            target=prewarm, name="arrayscope-numba-rss-prewarm", daemon=True
        )
        _warm_thread.start()


def is_ready() -> bool:
    """Return True when the numba kernels are compiled and usable."""
    return NUMBA_AVAILABLE and _READY.is_set()


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
    if not _READY.is_set():
        # Never block the visible path on the numba import or compilation; warm
        # in the background and let this call fall back to numpy.
        prewarm_async()
        return None

    array = np.asarray(data)
    kernel = _KERNELS.get(array.dtype)
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
