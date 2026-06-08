"""Small memory-budget helpers for render-time guardrails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


VISIBLE_RENDER_BUDGET_BYTES = 512 * 1024 * 1024
MONTAGE_BUDGET_BYTES = 1024 * 1024 * 1024
PREFETCH_BUDGET_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class MemoryEstimate:
    bytes: int
    label: str
    over_budget: bool


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


def estimate_montage_bytes(tile_shape, tile_count, dtype, *, rgb=False, histogram=True, gap=1, columns=None) -> int:
    tile_count = max(0, int(tile_count))
    if tile_count == 0:
        return 0
    tile_height, tile_width = (int(tile_shape[0]), int(tile_shape[1]))
    columns = int(columns or np.ceil(np.sqrt(tile_count)))
    columns = max(1, min(columns, tile_count))
    rows = int(np.ceil(tile_count / columns))
    gap = max(0, int(gap))
    height = rows * tile_height + gap * max(0, rows - 1)
    width = columns * tile_width + gap * max(0, columns - 1)
    return estimate_display_image_bytes((height, width), dtype, rgb=rgb, histogram=histogram)


def estimate(label, nbytes, budget_bytes) -> MemoryEstimate:
    nbytes = int(nbytes)
    return MemoryEstimate(bytes=nbytes, label=str(label), over_budget=nbytes > int(budget_bytes))


def format_bytes(nbytes: int) -> str:
    value = float(int(nbytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
