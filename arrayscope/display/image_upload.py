"""Image upload and RGB windowing helpers."""

from __future__ import annotations

import numpy as np


def rgb_display_for_levels(base, histogram_data, levels) -> np.ndarray:
    low, high = levels
    span = max(float(high) - float(low), 1e-12)
    intensity = np.clip((np.asarray(histogram_data, dtype=np.float32) - float(low)) / span, 0.0, 1.0)
    intensity = np.nan_to_num(intensity, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(np.asarray(base, dtype=np.float32) * intensity[..., np.newaxis], 0, 255).astype(np.uint8)


def ensure_imageitem_array(data):
    array = np.asarray(data)
    if array.flags.c_contiguous:
        return data
    return np.ascontiguousarray(array)
