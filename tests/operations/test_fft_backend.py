import numpy as np
import pytest

from arrayscope.operations import dim_ops, fft_backend
from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT


def test_scipy_backend_matches_numpy_convention_for_centered_fft_and_ifft():
    data = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    scipy_backend = fft_backend.ScipyFFTBackend()
    numpy_backend = fft_backend.NumpyFFTBackend()

    np.testing.assert_allclose(scipy_backend.centered_fft(data, 1, workers=1), numpy_backend.centered_fft(data, 1))
    np.testing.assert_allclose(scipy_backend.centered_ifft(data, 1, workers=1), numpy_backend.centered_ifft(data, 1))


def test_centered_fft_ifft_round_trip_preserves_existing_viewer_convention():
    data = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)

    result = CenteredIFFT(axis=1).apply(CenteredFFT(axis=1).apply(data))

    np.testing.assert_allclose(result, data, atol=1e-6)


def test_resolve_fft_workers_auto_is_conservative():
    assert fft_backend.resolve_fft_workers("auto", cpu_count=16) == 4
    assert fft_backend.resolve_fft_workers("auto", cpu_count=2) == 1


def test_resolve_fft_workers_all_minus_one():
    assert fft_backend.resolve_fft_workers("all_minus_one", cpu_count=8) == 7


def test_numpy_backend_available_as_fallback():
    assert "numpy" in fft_backend.available_fft_backends()
    assert fft_backend.resolve_fft_backend("numpy").name == "numpy"


def test_pyfftw_backend_unavailable_does_not_require_dependency(monkeypatch):
    monkeypatch.setattr(fft_backend, "_module_available", lambda name: False)

    assert fft_backend.resolve_fft_backend("pyfftw").name == "numpy"


def test_pyfftw_backend_available_in_conda_environment():
    pytest.importorskip("pyfftw")
    assert "pyfftw" in fft_backend.available_fft_backends()
    data = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    backend = fft_backend.resolve_fft_backend("pyfftw")

    np.testing.assert_allclose(backend.centered_fft(data, 0, workers=1), fft_backend.NumpyFFTBackend().centered_fft(data, 0), atol=1e-6)


def test_dim_ops_uses_runtime_options():
    fft_backend.set_fft_runtime_options(backend="numpy", workers="1")
    data = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(dim_ops.centered_fft(data, 0), fft_backend.NumpyFFTBackend().centered_fft(data, 0))
