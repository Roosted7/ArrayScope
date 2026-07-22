"""Optional numba accelerator for the LOD page bin reduction.

This module fuses the two ``np.add.reduceat`` passes that dominate
:func:`arrayscope.display.pyramid._reduce_planned_bins` into a single
parallel kernel.  The pure-NumPy implementation there remains the source of
truth and the always-available fallback; this accelerator is engaged only
when numba is importable *and* the kernels have been JIT-compiled off the
hot path.

The lazy import / off-the-hot-path compile / numpy-fallback machinery lives in
the shared :mod:`arrayscope.core.numba_runtime` (group name ``"pyramid"``), so
importing this module (and therefore ``pyramid``) never pays numba's ~0.4-0.6 s
import cost -- that happens on the background prewarm thread.

The kernel reproduces the exact two-pass accumulation *order* of the NumPy
code -- inner sum down the rows of each source column (axis-0 ``reduceat``),
then across the columns of the bin (axis-1 ``reduceat``) -- so real inputs
match bit-for-bit and complex inputs match to float32 rounding.
"""

from __future__ import annotations

import numpy as np

from arrayscope.core import numba_runtime

NUMBA_AVAILABLE = numba_runtime.NUMBA_AVAILABLE


def _build_kernels() -> dict[str, object]:
    """Import numba and construct + force-compile the reduction kernels (once)."""
    from numba import njit, prange

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _reduce_bins_real(src, y_starts, y_counts, x_starts, x_counts):
        ny = y_starts.shape[0]
        nx = x_starts.shape[0]
        height = src.shape[0]
        width = src.shape[1]
        out = np.empty((ny, nx), dtype=np.float32)
        for k in prange(ny):
            y0 = y_starts[k]
            y1 = y_starts[k + 1] if k + 1 < ny else height
            yc = y_counts[k]
            for m in range(nx):
                x0 = x_starts[m]
                x1 = x_starts[m + 1] if m + 1 < nx else width
                total = np.float32(0.0)
                for x in range(x0, x1):
                    column = np.float32(0.0)
                    for y in range(y0, y1):
                        column += src[y, x]
                    total += column
                out[k, m] = total / (yc * x_counts[m])
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _reduce_bins_complex(src, y_starts, y_counts, x_starts, x_counts):
        ny = y_starts.shape[0]
        nx = x_starts.shape[0]
        height = src.shape[0]
        width = src.shape[1]
        out = np.empty((ny, nx), dtype=np.complex64)
        for k in prange(ny):
            y0 = y_starts[k]
            y1 = y_starts[k + 1] if k + 1 < ny else height
            yc = y_counts[k]
            for m in range(nx):
                x0 = x_starts[m]
                x1 = x_starts[m + 1] if m + 1 < nx else width
                total = np.complex64(0.0)
                for x in range(x0, x1):
                    column = np.complex64(0.0)
                    for y in range(y0, y1):
                        column += src[y, x]
                    total += column
                out[k, m] = total / np.float32(yc * x_counts[m])
        return out

    # Force compilation on tiny arrays, off the hot path.
    y_starts = np.array([0, 2], dtype=np.intp)
    x_starts = np.array([0, 2], dtype=np.intp)
    counts = np.array([2.0, 2.0], dtype=np.float32)
    real = np.arange(16, dtype=np.float32).reshape(4, 4)
    cplx = (real + 1j * real[::-1]).astype(np.complex64)
    _reduce_bins_real(real, y_starts, counts, x_starts, counts)
    _reduce_bins_complex(np.ascontiguousarray(cplx), y_starts, counts, x_starts, counts)
    return {"real": _reduce_bins_real, "complex": _reduce_bins_complex}


_GROUP = numba_runtime.register("pyramid", _build_kernels)


def prewarm() -> None:
    """Compile the reduction kernels off the hot path (blocking).

    Idempotent and safe to call from any thread; a no-op when numba is not
    installed.  Callers should run this on a background thread (or use the
    lazy warm inside :func:`reduce_accumulated_if_ready`) so the first real
    reduction never pays the compilation cost.
    """

    _GROUP.prewarm()


def is_ready() -> bool:
    """Whether the accelerated path is compiled and available."""

    return NUMBA_AVAILABLE and _GROUP.ready()


def reduce_accumulated_if_ready(
    accumulated: np.ndarray,
    y_starts: np.ndarray,
    y_counts: np.ndarray,
    x_starts: np.ndarray,
    x_counts: np.ndarray,
):
    """Fused equivalent of the two ``reduceat`` passes plus ``/ counts``.

    ``accumulated`` is the already-transformed page (float32 or complex64,
    exactly as :func:`_reduce_planned_bins` builds it).  Returns the reduced
    ``(len(y_starts), len(x_starts))`` array in the same dtype, or ``None``
    when the fast path is unavailable or the input is not a supported 2D
    real/complex plane -- in which case the caller runs the NumPy reference.

    Any post-reduction step (``sqrt`` for RMS, the phase-vector zero clamp)
    stays in the NumPy caller and is unaffected.
    """

    if not NUMBA_AVAILABLE:
        return None
    kernels = _GROUP.get()  # kicks a background warm and returns None until ready
    if kernels is None:
        return None
    if accumulated.ndim != 2:
        return None
    dtype = accumulated.dtype
    if dtype == np.float32:
        kernel = kernels["real"]
    elif dtype == np.complex64:
        kernel = kernels["complex"]
    else:
        return None
    src = np.ascontiguousarray(accumulated)
    ys = np.ascontiguousarray(y_starts, dtype=np.intp)
    xs = np.ascontiguousarray(x_starts, dtype=np.intp)
    yc = np.ascontiguousarray(y_counts, dtype=np.float32)
    xc = np.ascontiguousarray(x_counts, dtype=np.float32)
    return kernel(src, ys, yc, xs, xc)
