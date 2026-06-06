"""Pure NumPy montage display helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.display.slice_engine import DisplayImage


@dataclass(frozen=True)
class MontageLayout:
    tile_shape: tuple[int, int]
    count: int
    columns: int
    rows: int
    gap: int = 1


def make_montage_from_images(images, *, histogram_images=None, columns=None, gap=1):
    images = tuple(np.asarray(image) for image in images)
    if not images:
        return DisplayImage(np.zeros((1, 1), dtype=float))
    first = images[0]
    if first.ndim not in (2, 3):
        raise ValueError(f"montage images must be 2D scalar or RGB, got shape {first.shape}")
    count = len(images)
    columns = int(columns or np.ceil(np.sqrt(count)))
    columns = max(1, min(columns, count))
    rows = int(np.ceil(count / columns))
    height, width = first.shape[:2]
    gap = max(0, int(gap))
    fill_shape = (rows * height + gap * (rows - 1), columns * width + gap * (columns - 1)) + first.shape[2:]
    montage = np.zeros(fill_shape, dtype=first.dtype)
    hist_montage = None
    if histogram_images is not None:
        histogram_images = tuple(np.asarray(image) for image in histogram_images)
        hist_montage = np.full(fill_shape[:2], np.nan, dtype=float)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        y0 = row * (height + gap)
        x0 = column * (width + gap)
        montage[y0 : y0 + height, x0 : x0 + width, ...] = image
        if hist_montage is not None and index < len(histogram_images):
            hist_montage[y0 : y0 + height, x0 : x0 + width] = histogram_images[index]
    return DisplayImage(data=montage, histogram_data=hist_montage)


def optimal_montage_columns(count, tile_shape, viewport_shape, gap=1):
    count = max(1, int(count))
    tile_height, tile_width = int(tile_shape[0]), int(tile_shape[1])
    viewport_height, viewport_width = int(viewport_shape[0]), int(viewport_shape[1])
    gap = max(0, int(gap))
    best_columns = 1
    best_used_fraction = -1.0
    best_aspect_error = None
    best_scale = None
    viewport_area = max(viewport_width * viewport_height, 1)
    viewport_aspect = viewport_width / max(viewport_height, 1)
    for columns in range(1, count + 1):
        rows = int(np.ceil(count / columns))
        total_width = columns * tile_width + gap * (columns - 1)
        total_height = rows * tile_height + gap * (rows - 1)
        scale = min(viewport_width / max(total_width, 1), viewport_height / max(total_height, 1))
        used_area = (total_width * scale) * (total_height * scale)
        used_fraction = min(used_area / viewport_area, 1.0)
        layout_aspect = total_width / max(total_height, 1)
        aspect_error = abs(np.log(layout_aspect / viewport_aspect)) if layout_aspect > 0 and viewport_aspect > 0 else float("inf")
        if (
            used_fraction > best_used_fraction
            or (
                np.isclose(used_fraction, best_used_fraction)
                and (best_aspect_error is None or aspect_error < best_aspect_error)
            )
            or (
                np.isclose(used_fraction, best_used_fraction)
                and best_aspect_error is not None
                and np.isclose(aspect_error, best_aspect_error)
                and (best_scale is None or scale > best_scale)
            )
        ):
            best_columns = columns
            best_used_fraction = used_fraction
            best_aspect_error = aspect_error
            best_scale = scale
    return best_columns
