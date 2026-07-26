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
def test_normalize_accelerates_measured_axis_copy_but_refuses_strided_input():
    _numba_native_ops.prewarm()
    data = np.ones((4, 5, 6), dtype=np.complex64)

    assert _numba_native_ops.normalize_if_ready(data, 1) is not None
    assert _numba_native_ops.normalize_if_ready(data[:, :, ::2], 2) is None
