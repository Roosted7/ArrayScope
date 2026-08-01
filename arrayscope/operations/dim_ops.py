"""Pure NumPy dimension operations for ArrayScope."""

from __future__ import annotations

import numpy as np

from arrayscope.core.axis_utils import normalize_axis_selection, validate_axis
from arrayscope.operations import fft_backend


def centered_fft(data, axis, *, backend=None, workers=None):
    """Apply the viewer's centered forward FFT transform along one axis."""
    axis = _validate_axis(data, axis)
    backend = fft_backend.runtime_fft_backend() if backend is None else backend
    workers = fft_backend.runtime_fft_workers() if workers is None else int(workers)
    return backend.centered_fft(data, axis, workers=workers)


def centered_ifft(data, axis, *, backend=None, workers=None):
    """Apply the viewer's centered inverse FFT transform along one axis."""
    axis = _validate_axis(data, axis)
    backend = fft_backend.runtime_fft_backend() if backend is None else backend
    workers = fft_backend.runtime_fft_workers() if workers is None else int(workers)
    return backend.centered_ifft(data, axis, workers=workers)


def centered_fftn(data, axes, *, backend=None, workers=None):
    """Centered forward FFT over an axis SET (2-D / 3-D k-space transforms).

    Deliberately a *composition* of the single-axis transform, one axis at a
    time, rather than one N-D call: it is then byte-identical to applying
    :class:`~arrayscope.operations.pipeline.CenteredFFT` over each axis in turn,
    which is what the viewer's established semantics already mean and what the
    oracle test pins. A single ``fftn`` over the axis set is the same transform
    mathematically, but its internal ordering and accumulation are a backend
    implementation detail -- scipy, pyfftw and numpy need not agree in the last
    bits, and this operation exists precisely so a user does not have to wonder
    whether the two-axis op and the two one-axis ops gave the same array.

    What the multi-axis op buys is not arithmetic but *scheduling*: one stage,
    one cache entry, one region translation, and a capability declaration that
    names the whole set as blocking so the complement stays chunkable.
    """

    axes = _validate_axes(data, axes)
    result = data
    for axis in axes:
        result = centered_fft(result, axis, backend=backend, workers=workers)
    return result


def centered_ifftn(data, axes, *, backend=None, workers=None):
    """Centered inverse FFT over an axis SET (see :func:`centered_fftn`)."""

    axes = _validate_axes(data, axes)
    result = data
    for axis in axes:
        result = centered_ifft(result, axis, backend=backend, workers=workers)
    return result


def apply_fftshift(data, axis):
    axis = _validate_axis(data, axis)
    return np.fft.fftshift(data, axes=axis)


def combine_real_imag_axis(data, axis):
    """Combine a size-2 real/imag axis into a singleton complex axis."""
    axis = _validate_axis(data, axis)
    if np.iscomplexobj(data):
        raise ValueError("data is already complex")
    if data.shape[axis] != 2:
        raise ValueError(f"axis {axis} must have size 2 to combine as complex")

    real_slice = [slice(None)] * data.ndim
    imag_slice = [slice(None)] * data.ndim
    real_slice[axis] = 0
    imag_slice[axis] = 1

    combined = data[tuple(real_slice)] + 1j * data[tuple(imag_slice)]
    return np.expand_dims(combined, axis=axis)


def split_complex_axis(data, axis):
    """Split a singleton complex axis into a size-2 real/imag axis."""
    axis = _validate_axis(data, axis)
    if not np.iscomplexobj(data):
        raise ValueError("data must be complex")
    if data.shape[axis] != 1:
        raise ValueError(f"axis {axis} must have size 1 to split to real/imag")

    squeezed = np.squeeze(data, axis=axis)
    return np.stack([np.real(squeezed), np.imag(squeezed)], axis=axis)


def _validate_axis(data, axis):
    return validate_axis(np.ndim(data), axis)


def _validate_axes(data, axes):
    return normalize_axis_selection(axes, ndim=np.ndim(data))
