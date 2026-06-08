"""Pure NumPy montage display helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arrayscope.display.geometry import MontageGeometry
from arrayscope.display.slice_engine import DisplayImage


@dataclass(frozen=True)
class MontageLayout:
    tile_shape: tuple[int, int]
    count: int
    columns: int
    rows: int
    gap: int = 1


@dataclass(frozen=True)
class RenderedMontage:
    image: DisplayImage
    geometry: MontageGeometry


@dataclass(frozen=True)
class MontageTile:
    montage_index: int
    source_index: int
    row: int
    col: int
    x0: int
    y0: int
    width: int
    height: int
    view_state: object


@dataclass(frozen=True)
class MontagePlan:
    axis: int
    tile_shape: tuple[int, int]
    grid_shape: tuple[int, int]
    columns: int
    rows: int
    gap: int
    tiles: tuple[MontageTile, ...]

    @property
    def display_shape(self) -> tuple[int, int]:
        height, width = self.tile_shape
        rows, columns = self.grid_shape
        return (
            rows * height + self.gap * max(0, rows - 1),
            columns * width + self.gap * max(0, columns - 1),
        )

    @property
    def geometry(self) -> MontageGeometry:
        return MontageGeometry(
            indices=tuple(tile.source_index for tile in self.tiles),
            tile_shape=self.tile_shape,
            columns=self.columns,
            rows=self.rows,
            gap=self.gap,
        )

    def tiles_intersecting(self, view_range, *, margin_tiles=1) -> tuple[MontageTile, ...]:
        if view_range is None:
            return self.tiles[: min(len(self.tiles), max(1, self.columns * 2))]
        x_range, y_range = view_range
        x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
        y0, y1 = sorted((float(y_range[0]), float(y_range[1])))
        margin_x = (self.tile_shape[1] + self.gap) * max(0, int(margin_tiles))
        margin_y = (self.tile_shape[0] + self.gap) * max(0, int(margin_tiles))
        x0 -= margin_x
        x1 += margin_x
        y0 -= margin_y
        y1 += margin_y
        visible = []
        for tile in self.tiles:
            tile_x1 = tile.x0 + tile.width
            tile_y1 = tile.y0 + tile.height
            if tile_x1 >= x0 and tile.x0 <= x1 and tile_y1 >= y0 and tile.y0 <= y1:
                visible.append(tile)
        if not visible and self.tiles:
            return self.tiles[: min(len(self.tiles), max(1, self.columns))]
        return tuple(visible)


def make_montage_plan(view_state, *, axis, indices, tile_shape, columns=None, viewport_shape=None, gap=1):
    indices = tuple(int(index) for index in indices)
    count = len(indices)
    tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
    gap = max(0, int(gap))
    if count == 0:
        return MontagePlan(int(axis), tile_shape, (0, 1), 1, 0, gap, ())
    if columns is None:
        if viewport_shape is None:
            columns = int(np.ceil(np.sqrt(count)))
        else:
            columns = optimal_montage_columns(count, tile_shape, viewport_shape, gap=gap)
    columns = max(1, min(int(columns), count))
    rows = int(np.ceil(count / columns))
    tiles = []
    for montage_index, source_index in enumerate(indices):
        row = montage_index // columns
        col = montage_index % columns
        x0 = col * (tile_shape[1] + gap)
        y0 = row * (tile_shape[0] + gap)
        tile_state = view_state.with_slice(axis, source_index).with_montage_axis(None)
        tiles.append(
            MontageTile(
                montage_index=montage_index,
                source_index=source_index,
                row=row,
                col=col,
                x0=x0,
                y0=y0,
                width=tile_shape[1],
                height=tile_shape[0],
                view_state=tile_state,
            )
        )
    return MontagePlan(int(axis), tile_shape, (rows, columns), columns, rows, gap, tuple(tiles))


@dataclass(frozen=True)
class RenderedTile:
    tile: MontageTile
    image: np.ndarray
    histogram_data: np.ndarray | None
    eval_ms: float
    slab_shape: tuple[int, ...]
    slab_nbytes: int | None

    def nbytes(self) -> int:
        total = int(self.image.nbytes)
        if isinstance(self.histogram_data, np.ndarray):
            total += int(self.histogram_data.nbytes)
        return total


def make_montage(images, *, histogram_images=None, columns=None, gap=1, indices=None):
    images = tuple(np.asarray(image) for image in images)
    if not images:
        geometry = MontageGeometry(indices=(), tile_shape=(1, 1), columns=1, rows=0, gap=max(0, int(gap)))
        return RenderedMontage(DisplayImage(np.zeros((1, 1), dtype=float)), geometry)
    first = images[0]
    if first.ndim not in (2, 3):
        raise ValueError(f"montage images must be 2D scalar or RGB, got shape {first.shape}")
    count = len(images)
    indices = tuple(range(count)) if indices is None else tuple(int(index) for index in indices)
    if len(indices) != count:
        raise ValueError("indices length must match image count")
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
    geometry = MontageGeometry(indices=indices, tile_shape=(height, width), columns=columns, rows=rows, gap=gap)
    return RenderedMontage(DisplayImage(data=montage, histogram_data=hist_montage), geometry)


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
