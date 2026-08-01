"""Small memory-budget helpers for render-time guardrails."""

from __future__ import annotations

import numpy as np

DEFAULT_VISIBLE_RENDER_BUDGET_BYTES = 512 * 1024 * 1024


def estimate_array_bytes(shape, dtype, *, channels=1) -> int:
    count = int(np.prod(tuple(int(size) for size in shape), dtype=np.int64))
    return count * np.dtype(dtype).itemsize * max(1, int(channels))


def estimate_display_image_bytes(shape, dtype, *, rgb=False, histogram=False) -> int:
    shape = tuple(int(size) for size in shape)
    base_shape = shape[:2]
    channels = 4 if rgb else 1
    total = estimate_array_bytes(base_shape, np.uint8 if rgb else dtype, channels=channels)
    if histogram:
        total += estimate_array_bytes(base_shape, np.float32)
    return int(total)


def format_bytes(nbytes: int) -> str:
    value = float(int(nbytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
