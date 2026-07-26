"""The measured Numba paths agree with and never replace NumPy fallbacks."""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations import _numba_native_ops
from arrayscope.operations.pipeline import LogMagnitude, Normalize, SoftThreshold


@pytest.mark.skipif(not _numba_native_ops.NUMBA_AVAILABLE, reason="numba is not installed")
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_warm_numba_normalize_matches_numpy_reference(dtype):
    rng = np.random.default_rng(20260726)
    real = rng.standard_normal((12, 10, 8)).astype(np.float32)
    data = real if dtype is np.float32 else (real + 1j * np.flip(real, axis=1)).astype(dtype)
    _numba_native_ops.prewarm()

    normalize_result = _numba_native_ops.normalize_if_ready(data, -1)

    norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1, keepdims=True))
    normalize_reference = np.zeros_like(data)
    np.divide(data, norm, out=normalize_reference, where=norm != 0)

    np.testing.assert_allclose(normalize_result, normalize_reference, rtol=2e-5, atol=2e-6)


def test_pipeline_uses_numpy_fallback_when_numba_is_unavailable(monkeypatch):
    monkeypatch.setattr(_numba_native_ops, "NUMBA_AVAILABLE", False)
    data = np.array([[0.0, 1.0, -2.0], [3.0, -4.0, 5.0]], dtype=np.float32)

    np.testing.assert_allclose(
        LogMagnitude(epsilon=1e-3).apply(data),
        np.log(np.maximum(np.abs(data), np.float32(1e-3))),
    )
    magnitude = np.abs(data)
    scale = np.zeros_like(data)
    np.divide(np.maximum(magnitude - 0.5, 0), magnitude, out=scale, where=magnitude != 0)
    np.testing.assert_allclose(SoftThreshold(threshold=0.5).apply(data), data * scale)
    norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=1, keepdims=True))
    reference = np.zeros_like(data)
    np.divide(data, norm, out=reference, where=norm != 0)
    np.testing.assert_allclose(Normalize(axis=1).apply(data), reference)


@pytest.mark.skipif(not _numba_native_ops.NUMBA_AVAILABLE, reason="numba is not installed")
@pytest.mark.parametrize("length", [4096, 200_000])
def test_numba_normalize_stays_accurate_as_the_axis_grows(length):
    # The kernel sums sequentially where NumPy sums pairwise, so a float32
    # accumulator drifted with the axis length -- 2.6e-4 relative at 1e6
    # samples. That was worse than an accuracy bug: the kernel only engages
    # once the JIT is warm, so the same operation on the same data returned
    # different values depending on compilation timing.
    #
    # The accumulator is float64 now, which makes the kernel *more* accurate
    # than the float32 NumPy path -- so bit-equality is the wrong bar and would
    # only be met by reintroducing the drift. The invariant that matters is
    # that the error stays at float32 noise and does not grow with the axis,
    # so measure both paths against a float64 ground truth.
    rng = np.random.default_rng(20260726)
    data = (rng.standard_normal((3, length)) * 10).astype(np.float32)
    _numba_native_ops.prewarm()

    accelerated = _numba_native_ops.normalize_if_ready(data, -1)

    exact = data.astype(np.float64)
    exact = exact / np.sqrt(np.sum(np.abs(exact) ** 2, axis=-1, keepdims=True))
    numpy_norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=-1, keepdims=True))
    numpy_path = np.zeros_like(data)
    np.divide(data, numpy_norm, out=numpy_path, where=numpy_norm != 0)

    def max_relative_error(candidate):
        return float(np.max(np.abs(candidate - exact) / np.maximum(np.abs(exact), 1e-30)))

    accelerated_error = max_relative_error(accelerated)
    assert accelerated_error < 1e-6
    # Never worse than the fallback it replaces, at any axis length.
    assert accelerated_error <= max_relative_error(numpy_path) + 1e-9


@pytest.mark.skipif(not _numba_native_ops.NUMBA_AVAILABLE, reason="numba is not installed")
def test_normalize_accelerates_measured_axis_copy_but_refuses_strided_input():
    _numba_native_ops.prewarm()
    data = np.ones((4, 5, 6), dtype=np.complex64)

    assert _numba_native_ops.normalize_if_ready(data, 1) is not None
    assert _numba_native_ops.normalize_if_ready(data[:, :, ::2], 2) is None
