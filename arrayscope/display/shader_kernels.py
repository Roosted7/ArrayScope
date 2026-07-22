"""Optional, prewarmed Numba accelerators for the CPU (pyqtgraph) display path.

These kernels fuse the scalar-window-LUT and RGB-window pixel chains that
``shader_mapping.cpu_display_rgba`` and ``image_upload.rgb_display_for_levels``
otherwise express as ~6-12 separate NumPy passes, each allocating a
full-frame temporary.  They matter only on the CPU display backend
(pyqtgraph): the wgpu backend runs the identical math in its WGSL fragment
shader on the GPU, so it never touches this module.

Discipline mirrors ``gpu/bc_numba.py``, with one addition: importing this
module costs nothing beyond NumPy -- **numba itself is imported only on the
background prewarm thread** (a bare ``import numba`` is ~570 ms here, and
``shader_mapping`` is imported broadly, including by the wgpu-only path).  The
visible display path never blocks on the import or on JIT compilation: until
``ready()`` is True, ``scalar_rgba``/``rgb_window`` are unavailable and callers
use their exact NumPy reference.  The kernels are written to be
**bit-identical** to those references (float32 throughout, ``np.rint`` LUT
rounding, truncating RGB cast) so displayed pixels never depend on whether the
JIT happens to be warm.

Scale codes match ``ShaderScale``: 0=linear, 1=log, 2=symlog.
"""

from __future__ import annotations

import threading

import numpy as np

_SCALE_LINEAR = 0
_SCALE_LOG = 1
_SCALE_SYMLOG = 2

_READY = threading.Event()
_WARM_LOCK = threading.Lock()
_WARM_STARTED = False
_NUMBA_UNAVAILABLE = False  # set True if the import fails, to stop retrying

# Compiled kernels, populated by _compile() on the prewarm thread.
_scalar_kernel = None
_rgb_kernel = None


def _compile() -> None:
    """Import numba and compile the kernels. Runs off the visible path."""

    global _scalar_kernel, _rgb_kernel, _NUMBA_UNAVAILABLE
    try:
        from numba import njit, prange
    except Exception:  # pragma: no cover - only where numba is absent
        _NUMBA_UNAVAILABLE = True
        return

    @njit(cache=True, nogil=True, fastmath=False, inline="always")
    def _apply_scale_scalar(value, scale_code, symlog_c):
        if np.isnan(value):
            # NumPy's log10/maximum/sign all propagate NaN; mirror that so the
            # ~isfinite alpha mask and window clamp agree bit-for-bit.
            return np.float32(np.nan)
        if scale_code == _SCALE_LINEAR:
            return np.float32(value)
        if scale_code == _SCALE_LOG:
            v = value if value > np.float32(0.0) else np.float32(0.0)
            return np.float32(np.log10(v))  # log10(0) -> -inf, matches np.log10
        denom = np.float32(10.0) ** np.float32(symlog_c)
        mag = value if value >= np.float32(0.0) else -value
        scaled = np.float32(np.log10(np.float32(1.0) + mag / denom))
        if value > np.float32(0.0):
            return scaled
        if value < np.float32(0.0):
            return -scaled
        return np.float32(0.0)

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _scalar_window_lut_rgba(component, scale_code, symlog_c, low, span, lut):
        """scalar chain: scale -> window[low,low+span] -> LUT -> RGBA uint8.

        Bit-identical to ``window_intensity`` + ``_sample_lut_rgb`` + the
        ``~isfinite(scalar)`` alpha mask in ``cpu_display_rgba``.
        """
        height, width = component.shape
        entries = lut.shape[0]
        top = np.float32(entries - 1)
        out = np.empty((height, width, 4), np.uint8)
        for i in prange(height):
            for j in range(width):
                scalar = _apply_scale_scalar(component[i, j], scale_code, symlog_c)
                finite = np.isfinite(scalar)
                # window_intensity: (scalar-low)/span, clip[0,1], nan->0
                t = (scalar - low) / span
                if np.isnan(t) or t < np.float32(0.0):
                    t = np.float32(0.0)
                elif t > np.float32(1.0):
                    t = np.float32(1.0)
                # _sample_lut_rgb: linear interpolate with rint rounding
                scaled = t * top
                lower = int(np.floor(scaled))
                upper = lower + 1 if lower + 1 < entries else entries - 1
                weight = scaled - np.float32(lower)
                inv = np.float32(1.0) - weight
                for c in range(3):
                    color = lut[lower, c] * inv + lut[upper, c] * weight
                    color = np.float32(np.rint(color))
                    if color > np.float32(255.0):
                        color = np.float32(255.0)
                    elif color < np.float32(0.0):
                        color = np.float32(0.0)
                    out[i, j, c] = np.uint8(color)
                out[i, j, 3] = np.uint8(255) if finite else np.uint8(0)
        return out

    @njit(cache=True, nogil=True, parallel=True, fastmath=False)
    def _rgb_window_kernel(base, histogram, low, span):
        """base * clip(nan_to_num((hist-low)/span),0,1) -> uint8 RGB.

        Bit-identical to ``image_upload.rgb_display_for_levels`` (truncating
        uint8 cast; nan->0, +inf->1, -inf->0 before the clip).
        """
        height, width, _ = base.shape
        out = np.empty((height, width, 3), np.uint8)
        for i in prange(height):
            for j in range(width):
                t = (histogram[i, j] - low) / span
                if np.isnan(t):
                    t = np.float32(0.0)
                elif np.isinf(t):
                    t = np.float32(1.0) if t > np.float32(0.0) else np.float32(0.0)
                if t < np.float32(0.0):
                    t = np.float32(0.0)
                elif t > np.float32(1.0):
                    t = np.float32(1.0)
                for c in range(3):
                    value = base[i, j, c] * t
                    if value > np.float32(255.0):
                        value = np.float32(255.0)
                    out[i, j, c] = np.uint8(value)
        return out

    # Force compilation now (still off the visible path) so the first real
    # call is already fast.
    tiny_c = np.zeros((2, 2), dtype=np.float32)
    tiny_lut = np.zeros((2, 3), dtype=np.float32)
    for code in (_SCALE_LINEAR, _SCALE_LOG, _SCALE_SYMLOG):
        _scalar_window_lut_rgba(
            tiny_c, code, np.float32(0.0), np.float32(0.0), np.float32(1.0), tiny_lut
        )
    _rgb_window_kernel(
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.float32(0.0),
        np.float32(1.0),
    )
    _scalar_kernel = _scalar_window_lut_rgba
    _rgb_kernel = _rgb_window_kernel
    _READY.set()


def ensure_prewarming() -> None:
    """Import numba + compile kernels on a background daemon thread, once.

    Never blocks the caller.  Until compilation finishes, ``ready()`` stays
    False and callers use their NumPy reference.
    """

    global _WARM_STARTED
    if _READY.is_set() or _NUMBA_UNAVAILABLE:
        return
    with _WARM_LOCK:
        if _WARM_STARTED:
            return
        _WARM_STARTED = True
    threading.Thread(target=_compile, name="arrayscope-shader-kernel-warm", daemon=True).start()


def prewarm_blocking() -> bool:
    """Compile synchronously (for tests/benchmarks). Returns availability."""

    if not _READY.is_set() and not _NUMBA_UNAVAILABLE:
        with _WARM_LOCK:
            if not _READY.is_set():
                _compile()
    return _READY.is_set()


def ready() -> bool:
    return _READY.is_set()


def scalar_rgba(component, scale_code, symlog_constant, low, span, lut):
    """Run the fused scalar->RGBA kernel. ``lut`` is float32 (entries, 3)."""

    return _scalar_kernel(
        np.ascontiguousarray(component, dtype=np.float32),
        int(scale_code),
        np.float32(symlog_constant),
        np.float32(low),
        np.float32(span),
        lut,
    )


def rgb_window(base, histogram, low, span):
    """Run the fused RGB-window kernel. ``base`` is (H,W,3) float32 contiguous."""

    return _rgb_kernel(
        base,
        np.ascontiguousarray(histogram, dtype=np.float32),
        np.float32(low),
        np.float32(span),
    )
