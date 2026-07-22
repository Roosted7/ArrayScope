"""Equivalence tests for the optional numba LOD-reduction accelerator.

Every assertion pins the numba fast path to the pure-NumPy reference in
:mod:`arrayscope.display.pyramid`.  Real inputs reduce in an identical
two-pass float32 accumulation *order*, so they match bit-for-bit; complex
inputs match to float32 rounding.  The fallback (numba absent or not warmed)
must produce exactly the incumbent NumPy result.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display import _numba_pyramid
from arrayscope.display.pyramid import reduce_source_grid
from arrayscope.gpu.keys import (
    REDUCER_MEAN,
    REDUCER_MEAN_ABS,
    REDUCER_NATIVE,
    REDUCER_PHASE_VECTOR,
    REDUCER_POWER,
    REDUCER_RMS,
)


@pytest.fixture(scope="module", autouse=True)
def _warm_numba():
    """Compile the kernels once so the fast path is actually exercised."""
    _numba_pyramid.prewarm()
    if not _numba_pyramid.NUMBA_AVAILABLE:
        pytest.skip("numba not available in this environment")
    assert _numba_pyramid.is_ready()


def _numpy_two_pass(acc, y_starts, y_counts, x_starts, x_counts):
    """The exact reference the accelerator replaces (see _reduce_planned_bins)."""
    sums_y = np.add.reduceat(acc, y_starts, axis=0, dtype=acc.dtype)
    sums = np.add.reduceat(sums_y, x_starts, axis=1, dtype=acc.dtype)
    counts = y_counts[:, None] * x_counts[None, :]
    return sums / counts


def _contiguous_partition(length, rng):
    """Arbitrary contiguous non-empty integer partition of ``[0, length)``."""
    edges = [0]
    while edges[-1] < length:
        edges.append(min(length, edges[-1] + int(rng.integers(1, 4))))
    starts = np.array(edges[:-1], dtype=np.intp)
    counts = np.diff(np.array(edges, dtype=np.float32)).astype(np.float32)
    return starts, counts


def _uniform_partition(length, factor):
    starts = np.arange(0, length, factor, dtype=np.intp)
    stops = np.append(starts[1:], length)
    counts = (stops - starts).astype(np.float32)
    return starts, counts


_BIN_CASES = {
    "uniform_2x2": lambda h, w, rng: (
        _uniform_partition(h, 2),
        _uniform_partition(w, 2),
    ),
    "partial_last_bin": lambda h, w, rng: (
        _uniform_partition(h, 3),
        _uniform_partition(w, 3),
    ),
    "arbitrary": lambda h, w, rng: (
        _contiguous_partition(h, rng),
        _contiguous_partition(w, rng),
    ),
    "single_bin": lambda h, w, rng: (
        (np.array([0], dtype=np.intp), np.array([h], dtype=np.float32)),
        (np.array([0], dtype=np.intp), np.array([w], dtype=np.float32)),
    ),
    "single_row_source": lambda h, w, rng: (
        (np.array([0], dtype=np.intp), np.array([1], dtype=np.float32)),
        _uniform_partition(w, 2),
    ),
    "single_col_source": lambda h, w, rng: (
        _uniform_partition(h, 2),
        (np.array([0], dtype=np.intp), np.array([1], dtype=np.float32)),
    ),
}


@pytest.mark.parametrize("case", sorted(_BIN_CASES))
@pytest.mark.parametrize("complex_input", [False, True])
def test_kernel_matches_numpy_two_pass(case, complex_input):
    rng = np.random.default_rng(hash((case, complex_input)) & 0xFFFF)
    height = 1 if case == "single_row_source" else 17
    width = 1 if case == "single_col_source" else 23
    if complex_input:
        acc = (
            rng.standard_normal((height, width)) + 1j * rng.standard_normal((height, width))
        ).astype(np.complex64)
    else:
        acc = rng.standard_normal((height, width)).astype(np.float32)
    (y_starts, y_counts), (x_starts, x_counts) = _BIN_CASES[case](height, width, rng)

    reference = _numpy_two_pass(acc, y_starts, y_counts, x_starts, x_counts)
    accelerated = _numba_pyramid.reduce_accumulated_if_ready(
        acc, y_starts, y_counts, x_starts, x_counts
    )

    assert accelerated is not None
    assert accelerated.dtype == acc.dtype
    assert accelerated.shape == reference.shape
    if complex_input:
        np.testing.assert_allclose(accelerated, reference, rtol=1e-5, atol=1e-6)
    elif case == "uniform_2x2":
        # The dominant LOD case: 2-element sums divided by an exact power of
        # two reproduce the NumPy reduceat bit-for-bit.
        np.testing.assert_array_equal(accelerated, reference)
    else:
        # Variable-count bins carry at most ~1 float32 ULP because the JIT may
        # group the reduction differently than reduceat; still exact to 1e-6.
        np.testing.assert_allclose(accelerated, reference, rtol=1e-6, atol=1e-6)


def test_unaccelerated_inputs_return_none():
    starts = np.array([0], dtype=np.intp)
    counts = np.array([2.0], dtype=np.float32)
    # float64 is not a stored page dtype.
    assert (
        _numba_pyramid.reduce_accumulated_if_ready(
            np.zeros((2, 2), dtype=np.float64), starts, counts, starts, counts
        )
        is None
    )
    # 3D component planes stay on the NumPy path.
    assert (
        _numba_pyramid.reduce_accumulated_if_ready(
            np.zeros((2, 2, 3), dtype=np.float32), starts, counts, starts, counts
        )
        is None
    )


def _global_complex_plane(rect):
    y0, y1, x0, x1 = rect
    height, width = (y1 - y0, x1 - x0)
    # Random per-sample phase so reduced phase-vector resultants vary bin to
    # bin (a smooth phase field collapses every resultant to magnitude ~1 and
    # degenerates the downstream chunk histogram -- unrelated to this path).
    rng = np.random.default_rng(0xC0FFEE)
    magnitude = (1.0 + rng.random((height, width))).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=(height, width))
    return (magnitude * np.exp(1j * phase)).astype(np.complex64)


def _global_real_plane(rect):
    y0, y1, x0, x1 = rect
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (yy * 1000 + xx).astype(np.float32)


# valid_rect is deliberately NOT a multiple of the reduction factor on either
# axis, so every reducer exercises the partial-edge (variable-count) bins.
_REAL_RECT = (99, 110, 100, 117)
_COMPLEX_RECT = (40, 51, 60, 77)


def _run(reducer, *, complex_source):
    rect = _COMPLEX_RECT if complex_source else _REAL_RECT
    source = _global_complex_plane(rect) if complex_source else _global_real_plane(rect)
    result = reduce_source_grid(
        source,
        source_origin_yx=(rect[0], rect[2]),
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),  # factor 2 on each axis
        reducer=reducer,
    )
    return np.asarray(result.values)


@pytest.mark.parametrize(
    ("reducer", "complex_source"),
    [
        (REDUCER_MEAN, False),
        (REDUCER_MEAN, True),
        (REDUCER_MEAN_ABS, True),
        (REDUCER_POWER, True),
        (REDUCER_RMS, True),
        (REDUCER_PHASE_VECTOR, True),
        (REDUCER_MEAN_ABS, False),
        (REDUCER_POWER, False),
        (REDUCER_RMS, False),
    ],
)
def test_reduce_source_grid_numba_matches_numpy(reducer, complex_source, monkeypatch):
    """Every reducer's end-to-end page values match the NumPy fallback."""
    accelerated = _run(reducer, complex_source=complex_source)

    # Force the NumPy reference path and recompute identically.
    monkeypatch.setattr(_numba_pyramid, "reduce_accumulated_if_ready", lambda *a, **k: None)
    reference = _run(reducer, complex_source=complex_source)

    assert accelerated.dtype == reference.dtype
    if np.iscomplexobj(accelerated):
        np.testing.assert_allclose(accelerated, reference, rtol=1e-5, atol=1e-6)
    else:
        np.testing.assert_array_equal(accelerated, reference)


def test_native_path_bypasses_accelerator(monkeypatch):
    """NATIVE (count==1) returns before the reduction seam is ever reached."""
    calls = []
    real_fn = _numba_pyramid.reduce_accumulated_if_ready

    def _spy(*args, **kwargs):
        calls.append(args[0].shape)
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(_numba_pyramid, "reduce_accumulated_if_ready", _spy)

    rect = (0, 4, 0, 4)
    native = reduce_source_grid(
        _global_real_plane(rect),
        source_origin_yx=(0, 0),
        valid_source_rect_yx=rect,
        reduction_yx=(0, 0),
        reducer=REDUCER_NATIVE,
    )
    assert calls == []  # native returns before the reduction seam
    np.testing.assert_array_equal(np.asarray(native.values), _global_real_plane(rect))


def test_numba_unavailable_fallback(monkeypatch):
    """With numba reported absent, the seam declines and NumPy still reduces."""
    monkeypatch.setattr(_numba_pyramid, "NUMBA_AVAILABLE", False)
    assert not _numba_pyramid.is_ready()
    starts = np.array([0, 2], dtype=np.intp)
    counts = np.array([2.0, 2.0], dtype=np.float32)
    assert (
        _numba_pyramid.reduce_accumulated_if_ready(
            np.ones((4, 4), dtype=np.float32), starts, counts, starts, counts
        )
        is None
    )
    # And the public reducer keeps producing the exact oracle mean.
    rect = (0, 4, 0, 4)
    result = reduce_source_grid(
        _global_real_plane(rect),
        source_origin_yx=(0, 0),
        valid_source_rect_yx=rect,
        reduction_yx=(1, 1),
        reducer=REDUCER_MEAN,
    )
    expected = _global_real_plane(rect).reshape(2, 2, 2, 2).mean(axis=(1, 3), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(result.values), expected)


def test_prewarm_is_idempotent():
    _numba_pyramid.prewarm()
    _numba_pyramid.prewarm()
    assert _numba_pyramid.is_ready() == _numba_pyramid.NUMBA_AVAILABLE
