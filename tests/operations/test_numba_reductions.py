"""Tests for the optional numba-accelerated RSS reduction.

These assert the numba fast path is numerically equivalent to the numpy
reference across dtypes / shapes / axes (including non-contiguous inputs and
size-1 reduction axes), that integer inputs are widened before squaring, and
that the numpy fallback stays correct when numba is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations import _numba_reductions
from arrayscope.operations.pipeline import RootSumSquares, _root_sum_squares


def _numpy_reference(data, axis):
    array = np.asarray(data)
    if np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_:
        array = array.astype(np.float32)
    return np.sqrt(np.sum(np.abs(array) ** 2, axis=axis))


DTYPES = [np.float32, np.float64, np.complex64, np.complex128, np.int16, np.int32]
SHAPES = [(5, 7), (4, 5, 6), (1, 8), (8, 1), (2, 3, 4, 5)]


def _sample(dtype, shape, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.complexfloating):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
    if np.issubdtype(dt, np.integer):
        # Include magnitudes > 181 to exercise the int16-square overflow guard.
        return rng.integers(-800, 800, size=shape).astype(dt)
    return rng.standard_normal(shape).astype(dt)


@pytest.fixture(scope="module", autouse=True)
def _warm_numba():
    if _numba_reductions.NUMBA_AVAILABLE:
        _numba_reductions.prewarm()


@pytest.mark.skipif(not _numba_reductions.NUMBA_AVAILABLE, reason="numba not installed")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_numba_matches_numpy_reference_all_axes(dtype, shape):
    data = _sample(dtype, shape)
    for axis in range(len(shape)):
        got = _root_sum_squares(data, axis)
        expected = _numpy_reference(data, axis)
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
        assert got.dtype == expected.dtype


@pytest.mark.skipif(not _numba_reductions.NUMBA_AVAILABLE, reason="numba not installed")
def test_numba_path_is_actually_taken_for_contiguous_last_axis():
    data = _sample(np.complex64, (16, 16, 8))
    # Last axis of a C-contiguous array is the accelerated case.
    assert _numba_reductions.rss_if_ready(data, 2) is not None
    # Non-last axis is not accelerated (copy would eat the win).
    assert _numba_reductions.rss_if_ready(data, 0) is None


@pytest.mark.skipif(not _numba_reductions.NUMBA_AVAILABLE, reason="numba not installed")
def test_non_contiguous_view_falls_back_but_stays_correct():
    data = _sample(np.complex128, (6, 5, 4))
    view = np.swapaxes(data, 0, 2)  # non-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    # Reducing the (now non-contiguous) last axis is not accelerated...
    assert _numba_reductions.rss_if_ready(view, view.ndim - 1) is None
    # ...but the pipeline result is still correct via the numpy fallback.
    np.testing.assert_allclose(
        _root_sum_squares(view, view.ndim - 1),
        _numpy_reference(view, view.ndim - 1),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(not _numba_reductions.NUMBA_AVAILABLE, reason="numba not installed")
def test_size_one_reduction_axis():
    data = _sample(np.float32, (7, 1))
    got = _root_sum_squares(data, 1)
    expected = _numpy_reference(data, 1)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _numba_reductions.NUMBA_AVAILABLE, reason="numba not installed")
def test_int16_widening_avoids_silent_overflow():
    # |v| > 181 => v**2 overflows int16; the widening guard must prevent NaN.
    data = np.array([[200, 300], [181, 182]], dtype=np.int16)
    got = _root_sum_squares(data, 0)
    expected = np.sqrt(np.sum(np.abs(data.astype(np.float32)) ** 2, axis=0))
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-4)
    assert np.isfinite(got).all()
    assert got.dtype == np.float32


def test_fallback_matches_reference_when_numba_disabled(monkeypatch):
    monkeypatch.setattr(_numba_reductions, "NUMBA_AVAILABLE", False)
    # rss_if_ready must decline immediately, and _root_sum_squares uses numpy.
    for dtype in DTYPES:
        data = _sample(dtype, (4, 5, 6), seed=1)
        for axis in range(3):
            assert _numba_reductions.rss_if_ready(np.asarray(data), axis) is None
            got = _root_sum_squares(data, axis)
            expected = _numpy_reference(data, axis)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_rss_operation_apply_uses_helper():
    data = _sample(np.complex64, (8, 8, 4))
    op = RootSumSquares(axis=2)
    np.testing.assert_allclose(op.apply(data), _numpy_reference(data, 2), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _numba_reductions.NUMBA_AVAILABLE, reason="numba not installed")
def test_prewarm_is_idempotent_and_reports_ready():
    _numba_reductions.prewarm()
    _numba_reductions.prewarm()
    assert _numba_reductions.is_ready()
