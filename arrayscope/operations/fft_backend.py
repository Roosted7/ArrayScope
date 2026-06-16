"""FFT backend selection and worker-count configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import os
from typing import Protocol

import numpy as np


class FFTBackendChoice(Enum):
    AUTO = "auto"
    SCIPY = "scipy"
    PYFFTW = "pyfftw"
    NUMPY = "numpy"


class FFTWorkersChoice(Enum):
    AUTO = "auto"
    ONE = "1"
    TWO = "2"
    FOUR = "4"
    ALL_MINUS_ONE = "all_minus_one"


class FFTBackend(Protocol):
    name: str

    def centered_fft(self, data, axis: int, *, workers: int = 1): ...

    def centered_ifft(self, data, axis: int, *, workers: int = 1): ...


@dataclass(frozen=True)
class ScipyFFTBackend:
    name: str = "scipy"

    def centered_fft(self, data, axis: int, *, workers: int = 1):
        from scipy import fft

        # ArrayScope's names are MRI/k-space oriented: "centered FFT" maps to
        # an inverse FFT internally, preserving the historical viewer behavior.
        return fft.ifftshift(fft.ifft(fft.fftshift(data, axes=axis), axis=axis, norm="ortho", workers=int(workers)), axes=axis)

    def centered_ifft(self, data, axis: int, *, workers: int = 1):
        from scipy import fft

        # Inverse in viewer vocabulary maps to a forward FFT internally.
        return fft.ifftshift(fft.fft(fft.fftshift(data, axes=axis), axis=axis, norm="ortho", workers=int(workers)), axes=axis)


@dataclass(frozen=True)
class NumpyFFTBackend:
    name: str = "numpy"

    def centered_fft(self, data, axis: int, *, workers: int = 1):
        del workers
        return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data, axes=axis), axis=axis, norm="ortho"), axes=axis)

    def centered_ifft(self, data, axis: int, *, workers: int = 1):
        del workers
        return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(data, axes=axis), axis=axis, norm="ortho"), axes=axis)


@dataclass(frozen=True)
class PyFFTWBackend:
    name: str = "pyfftw"

    def centered_fft(self, data, axis: int, *, workers: int = 1):
        from pyfftw.interfaces import numpy_fft

        return numpy_fft.ifftshift(
            numpy_fft.ifft(numpy_fft.fftshift(data, axes=axis), axis=axis, norm="ortho", threads=int(workers)),
            axes=axis,
        )

    def centered_ifft(self, data, axis: int, *, workers: int = 1):
        from pyfftw.interfaces import numpy_fft

        return numpy_fft.ifftshift(
            numpy_fft.fft(numpy_fft.fftshift(data, axes=axis), axis=axis, norm="ortho", threads=int(workers)),
            axes=axis,
        )


_runtime_backend_choice = FFTBackendChoice.AUTO
_runtime_workers_choice = FFTWorkersChoice.AUTO


def available_fft_backends() -> tuple[str, ...]:
    names = ["numpy"]
    if _module_available("scipy"):
        names.insert(0, "scipy")
    if _module_available("pyfftw"):
        names.append("pyfftw")
    return tuple(names)


def resolve_fft_backend(choice: FFTBackendChoice | str) -> FFTBackend:
    choice = _normalize_backend_choice(choice)
    if choice == FFTBackendChoice.NUMPY:
        return NumpyFFTBackend()
    if choice == FFTBackendChoice.PYFFTW:
        if _module_available("pyfftw"):
            return PyFFTWBackend()
        return _scipy_or_numpy_backend()
    if choice == FFTBackendChoice.SCIPY:
        return _scipy_or_numpy_backend()
    return _scipy_or_numpy_backend()


def resolve_fft_workers(choice: FFTWorkersChoice | str, *, cpu_count: int | None = None) -> int:
    choice = _normalize_workers_choice(choice)
    count = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    count = max(1, count)
    if choice == FFTWorkersChoice.ONE:
        return 1
    if choice == FFTWorkersChoice.TWO:
        return 2
    if choice == FFTWorkersChoice.FOUR:
        return 4
    if choice == FFTWorkersChoice.ALL_MINUS_ONE:
        return max(1, count - 1)
    return min(4, max(1, count // 2))


def set_fft_runtime_options(*, backend: FFTBackendChoice | str, workers: FFTWorkersChoice | str) -> None:
    global _runtime_backend_choice, _runtime_workers_choice
    _runtime_backend_choice = _normalize_backend_choice(backend)
    _runtime_workers_choice = _normalize_workers_choice(workers)


def get_fft_runtime_options() -> tuple[FFTBackendChoice, FFTWorkersChoice]:
    return _runtime_backend_choice, _runtime_workers_choice


def runtime_fft_backend() -> FFTBackend:
    return resolve_fft_backend(_runtime_backend_choice)


def runtime_fft_workers() -> int:
    return resolve_fft_workers(_runtime_workers_choice)


def _scipy_or_numpy_backend() -> FFTBackend:
    if _module_available("scipy"):
        return ScipyFFTBackend()
    return NumpyFFTBackend()


def _normalize_backend_choice(value) -> FFTBackendChoice:
    if isinstance(value, FFTBackendChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return FFTBackendChoice(str(value))
    except Exception:
        return FFTBackendChoice.AUTO


def _normalize_workers_choice(value) -> FFTWorkersChoice:
    if isinstance(value, FFTWorkersChoice):
        return value
    value = getattr(value, "value", value)
    try:
        return FFTWorkersChoice(str(value))
    except Exception:
        return FFTWorkersChoice.AUTO


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None
