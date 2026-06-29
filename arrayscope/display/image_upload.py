"""Image upload and RGB windowing helpers."""

from __future__ import annotations

import numpy as np


def rgb_display_for_levels(base, histogram_data, levels) -> np.ndarray:
    low, high = levels
    span = max(float(high) - float(low), 1e-12)
    intensity = np.array(histogram_data, dtype=np.float32, copy=True)
    intensity -= np.float32(low)
    intensity /= np.float32(span)
    np.nan_to_num(intensity, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.clip(intensity, 0.0, 1.0, out=intensity)
    display = np.multiply(np.asarray(base, dtype=np.float32), intensity[..., np.newaxis])
    np.clip(display, 0.0, 255.0, out=display)
    return display.astype(np.uint8)


def ensure_imageitem_array(data):
    array = np.asarray(data)
    if array.flags.c_contiguous:
        return data
    return np.ascontiguousarray(array)
