"""Pure NumPy dimension operations for ArrayScope."""

from __future__ import annotations

import numpy as np

from arrayscope.core.axis_utils import validate_axis


def centered_fft(data, axis):
    """Apply the viewer's centered forward FFT transform along one axis."""
    axis = _validate_axis(data, axis)
    return np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data), axis=axis, norm="ortho"))


def centered_ifft(data, axis):
    """Apply the viewer's centered inverse FFT transform along one axis."""
    axis = _validate_axis(data, axis)
    return np.fft.ifftshift(np.fft.fft(np.fft.fftshift(data), axis=axis, norm="ortho"))


def apply_fftshift(data, axis):
    axis = _validate_axis(data, axis)
    return np.fft.fftshift(data, axes=axis)


def undo_fftshift(data, axis):
    axis = _validate_axis(data, axis)
    return np.fft.ifftshift(data, axes=axis)


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
