import numpy as np

from arrayscope.core.memory_budget import (
    estimate_array_bytes,
    estimate_display_image_bytes,
    format_bytes,
)


def test_estimate_array_bytes_uses_shape_dtype_and_channels():
    assert estimate_array_bytes((2, 3), np.float32) == 24
    assert estimate_array_bytes((2, 3), np.uint8, channels=4) == 24


def test_estimate_display_image_bytes_includes_rgb_and_histogram():
    assert estimate_display_image_bytes((10, 20), np.float32) == 800
    assert estimate_display_image_bytes((10, 20), np.float32, rgb=True, histogram=True) == 800 + 800


def test_format_bytes_uses_binary_units():
    assert format_bytes(1024) == "1.0 KiB"
