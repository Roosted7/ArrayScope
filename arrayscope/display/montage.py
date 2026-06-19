"""Pure NumPy montage display helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Sequence
from enum import Enum

import numpy as np

from arrayscope.core.memory_budget import estimate_viewport_canvas_bytes, format_bytes
from arrayscope.display.geometry import MontageGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.shader_mapping import ShaderMapping, TexturePlaneKind
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


class _ValueEnum(Enum):
    def __eq__(self, other):
        if isinstance(other, Enum):
            return self.value == getattr(other, "value", object())
        return self.value == other

    def __hash__(self):
        return hash(self.value)


class MontageTileState(_ValueEnum):
    LOADED = "loaded"
    LOADING = "loading"
    SKIPPED = "skipped"
    UNLOADED = "unloaded"


@dataclass(frozen=True)
class MontageTileStatus:
    tile_number: int
    montage_index: int
    source_index: int
    state: MontageTileState


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

    def tile_at(self, x: int, y: int) -> MontageTile | None:
        x = int(x)
        y = int(y)
        if x < 0 or y < 0 or not self.tiles:
            return None
        tile_height, tile_width = self.tile_shape
        stride_x = tile_width + self.gap
        stride_y = tile_height + self.gap
        if stride_x <= 0 or stride_y <= 0:
            return None
        column = x // stride_x
        row = y // stride_y
        if column < 0 or row < 0 or column >= self.columns or row >= self.rows:
            return None
        local_x = x - column * stride_x
        local_y = y - row * stride_y
        if local_x < 0 or local_x >= tile_width or local_y < 0 or local_y >= tile_height:
            return None
        tile_number = row * self.columns + column
        if tile_number >= len(self.tiles):
            return None
        return self.tiles[tile_number]

    def display_rect_for_tiles(self, tiles: Sequence[MontageTile]) -> tuple[int, int, int, int] | None:
        tiles = tuple(tiles)
        if not tiles:
            return None
        x0 = min(int(tile.x0) for tile in tiles)
        y0 = min(int(tile.y0) for tile in tiles)
        x1 = max(int(tile.x0 + tile.width) for tile in tiles)
        y1 = max(int(tile.y0 + tile.height) for tile in tiles)
        return x0, y0, x1, y1

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
class RenderedTilePayload:
    image: np.ndarray
    histogram_data: np.ndarray | None
    eval_ms: float
    slab_shape: tuple[int, ...]
    slab_nbytes: int | None
    shader_mapping: ShaderMapping | None = None
    texture_kind: TexturePlaneKind | None = None
    semantic_data: np.ndarray | None = None
    lod: LodInfo | None = None

    def nbytes(self) -> int:
        total = int(self.image.nbytes)
        if isinstance(self.histogram_data, np.ndarray):
            total += int(self.histogram_data.nbytes)
        if isinstance(self.semantic_data, np.ndarray) and self.semantic_data is not self.image:
            total += int(self.semantic_data.nbytes)
        return total

    def bind(self, tile: MontageTile) -> "RenderedTile":
        return RenderedTile(
            tile=tile,
            image=self.image,
            histogram_data=self.histogram_data,
            eval_ms=self.eval_ms,
            slab_shape=self.slab_shape,
            slab_nbytes=self.slab_nbytes,
            shader_mapping=self.shader_mapping,
            texture_kind=self.texture_kind,
            semantic_data=self.semantic_data,
            lod=self.lod,
        )


@dataclass(frozen=True)
class RenderedTile:
    tile: MontageTile
    image: np.ndarray
    histogram_data: np.ndarray | None
    eval_ms: float
    slab_shape: tuple[int, ...]
    slab_nbytes: int | None
    shader_mapping: ShaderMapping | None = None
    texture_kind: TexturePlaneKind | None = None
    semantic_data: np.ndarray | None = None
    lod: LodInfo | None = None

    def nbytes(self) -> int:
        total = int(self.image.nbytes)
        if isinstance(self.histogram_data, np.ndarray):
            total += int(self.histogram_data.nbytes)
        if isinstance(self.semantic_data, np.ndarray) and self.semantic_data is not self.image:
            total += int(self.semantic_data.nbytes)
        return total

    def payload(self) -> RenderedTilePayload:
        return RenderedTilePayload(
            image=self.image,
            histogram_data=self.histogram_data,
            eval_ms=self.eval_ms,
            slab_shape=self.slab_shape,
            slab_nbytes=self.slab_nbytes,
            shader_mapping=self.shader_mapping,
            texture_kind=self.texture_kind,
            semantic_data=self.semantic_data,
            lod=self.lod,
        )


@dataclass(frozen=True)
class MontageViewportCanvas:
    data: np.ndarray
    histogram_data: np.ndarray | None
    origin_x: int
    origin_y: int
    full_plan: MontagePlan
    tile_states: tuple[MontageTileState, ...] = ()
    canvas_rect: tuple[int, int, int, int] = (0, 0, 1, 1)

    @property
    def display_shape(self) -> tuple[int, int]:
        return self.data.shape[:2]


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


def make_montage_viewport_canvas(
    plan: MontagePlan,
    rendered_tiles: Sequence[RenderedTile],
    *,
    view_range=None,
    viewport_shape: tuple[int, int] | None = None,
    budget_bytes: int,
    dtype=None,
    rgb: bool = False,
    include_histogram: bool = True,
    loading_tiles: Sequence[MontageTile] = (),
    skipped_tiles: Sequence[MontageTile] = (),
) -> MontageViewportCanvas:
    rendered_tiles = tuple(rendered_tiles)
    tile_by_index = {int(rendered.tile.montage_index): rendered for rendered in rendered_tiles}
    rect = montage_rect_for_viewport(plan, view_range=view_range, viewport_shape=viewport_shape)
    x0, y0, x1, y1 = rect
    width = max(1, int(x1) - int(x0))
    height = max(1, int(y1) - int(y0))
    rendered_images = tuple(np.asarray(rendered.image) for rendered in rendered_tiles)
    sample = rendered_images[0] if rendered_images else None
    if dtype is None:
        dtype = sample.dtype if sample is not None else float
    data_dtype = np.dtype(dtype)
    is_rgb = bool(rgb or any(image.ndim == 3 for image in rendered_images))
    trailing_shape = _rgb_trailing_shape(rendered_images) if is_rgb else ()
    canvas_shape = (height, width) + tuple(trailing_shape)
    estimated = estimate_viewport_canvas_bytes((height, width), dtype, rgb=is_rgb, histogram=include_histogram)
    if estimated > int(budget_bytes):
        raise MemoryError(
            f"Montage viewport canvas would allocate {format_bytes(estimated)} "
            f"over budget {format_bytes(int(budget_bytes))}."
        )
    if is_rgb:
        data = np.full(canvas_shape, 42, dtype=np.uint8)
    else:
        # Keep visual placeholders finite. Tile state, not display pixel value,
        # distinguishes loading/skipped/unloaded regions; histogram_data remains
        # NaN so ROI/statistics ignore non-loaded regions.
        data = np.zeros(canvas_shape, dtype=data_dtype)
    histogram_data = np.full((height, width), np.nan, dtype=np.float32) if include_histogram else None
    canvas_rect = (int(x0), int(y0), int(x0) + width, int(y0) + height)
    tile_states = _tile_states_for_plan(
        plan,
        loaded_indices=tile_by_index.keys(),
        loading_tiles=loading_tiles,
        skipped_tiles=skipped_tiles,
    )
    for rendered in sorted(tile_by_index.values(), key=lambda item: item.tile.montage_index):
        _copy_rendered_tile_into_canvas(rendered, data, histogram_data, canvas_rect)
    return MontageViewportCanvas(data, histogram_data, int(x0), int(y0), plan, tile_states, canvas_rect)


def montage_rect_for_viewport(plan: MontagePlan, *, view_range=None, viewport_shape=None) -> tuple[int, int, int, int]:
    full_height, full_width = plan.display_shape
    if full_height <= 0 or full_width <= 0:
        return (0, 0, 1, 1)
    if view_range is None:
        if viewport_shape is None:
            width = full_width
            height = full_height
        else:
            height = min(full_height, max(1, int(viewport_shape[0])))
            width = min(full_width, max(1, int(viewport_shape[1])))
        rect = (0, 0, max(1, int(width)), max(1, int(height)))
        return _expand_rect_to_tile_bounds(plan, rect)
    rect = _rect_for_view_range(view_range, plan, viewport_shape)
    rect = _intersect_rect(rect, (0, 0, full_width, full_height)) or (0, 0, min(1, full_width), min(1, full_height))
    return _expand_rect_to_tile_bounds(plan, rect)


def _expand_rect_to_tile_bounds(plan: MontagePlan, rect) -> tuple[int, int, int, int]:
    full_height, full_width = plan.display_shape
    selected = []
    for tile in plan.tiles:
        tile_rect = (int(tile.x0), int(tile.y0), int(tile.x0 + tile.width), int(tile.y0 + tile.height))
        if _intersect_rect(tile_rect, rect) is not None:
            selected.append(tile_rect)
    if not selected:
        return rect
    x0 = min(tile_rect[0] for tile_rect in selected)
    y0 = min(tile_rect[1] for tile_rect in selected)
    x1 = max(tile_rect[2] for tile_rect in selected)
    y1 = max(tile_rect[3] for tile_rect in selected)
    return _intersect_rect((x0, y0, x1, y1), (0, 0, full_width, full_height)) or rect


def tile_status_at_global_point(
    plan: MontagePlan,
    tile_states: Sequence[MontageTileState],
    x: int,
    y: int,
) -> MontageTileStatus | None:
    tile = plan.tile_at(int(x), int(y))
    if tile is None:
        return None
    state = MontageTileState.UNLOADED
    if int(tile.montage_index) < len(tile_states):
        state = MontageTileState(tile_states[int(tile.montage_index)])
    return MontageTileStatus(
        tile_number=int(tile.montage_index),
        montage_index=int(tile.montage_index),
        source_index=int(tile.source_index),
        state=state,
    )


def copy_rendered_tile_into_canvas(rendered: RenderedTile, canvas: MontageViewportCanvas) -> MontageViewportCanvas:
    data = np.array(canvas.data, copy=True)
    histogram_data = None if canvas.histogram_data is None else np.array(canvas.histogram_data, copy=True)
    _copy_rendered_tile_into_canvas(rendered, data, histogram_data, canvas.canvas_rect)
    states = list(canvas.tile_states or _tile_states_for_plan(canvas.full_plan))
    if int(rendered.tile.montage_index) < len(states):
        states[int(rendered.tile.montage_index)] = MontageTileState.LOADED
    return replace(canvas, data=data, histogram_data=histogram_data, tile_states=tuple(states))


def patch_rendered_tile_into_canvas(rendered: RenderedTile, canvas: MontageViewportCanvas) -> tuple[int, int, int, int] | None:
    """Patch one rendered tile into an existing viewport canvas.

    Returns the dirty rect in canvas-local coordinates, or None when the tile
    does not intersect the current viewport canvas.
    """
    dirty = _copy_rendered_tile_into_canvas(rendered, canvas.data, canvas.histogram_data, canvas.canvas_rect)
    if dirty is None:
        return None
    states = list(canvas.tile_states or _tile_states_for_plan(canvas.full_plan))
    if int(rendered.tile.montage_index) < len(states):
        states[int(rendered.tile.montage_index)] = MontageTileState.LOADED
        object.__setattr__(canvas, "tile_states", tuple(states))
    return dirty


def _tile_states_for_plan(
    plan: MontagePlan,
    *,
    loaded_indices=(),
    loading_tiles: Sequence[MontageTile] = (),
    skipped_tiles: Sequence[MontageTile] = (),
) -> tuple[MontageTileState, ...]:
    states = [MontageTileState.UNLOADED for _tile in plan.tiles]
    for tile in skipped_tiles:
        index = int(tile.montage_index)
        if 0 <= index < len(states):
            states[index] = MontageTileState.SKIPPED
    for tile in loading_tiles:
        index = int(tile.montage_index)
        if 0 <= index < len(states) and states[index] != MontageTileState.SKIPPED:
            states[index] = MontageTileState.LOADING
    for index in loaded_indices:
        index = int(index)
        if 0 <= index < len(states):
            states[index] = MontageTileState.LOADED
    return tuple(states)


def _rect_for_view_range(view_range, plan: MontagePlan, viewport_shape) -> tuple[int, int, int, int]:
    x_range, y_range = view_range
    x0, x1 = sorted((float(x_range[0]), float(x_range[1])))
    y0, y1 = sorted((float(y_range[0]), float(y_range[1])))
    stride_x = int(plan.tile_shape[1]) + int(plan.gap)
    stride_y = int(plan.tile_shape[0]) + int(plan.gap)
    margin_x = max(0, stride_x)
    margin_y = max(0, stride_y)
    if viewport_shape is not None:
        margin_y = max(margin_y, min(int(viewport_shape[0]), stride_y))
        margin_x = max(margin_x, min(int(viewport_shape[1]), stride_x))
    return (
        int(np.floor(x0)) - margin_x,
        int(np.floor(y0)) - margin_y,
        int(np.ceil(x1)) + margin_x,
        int(np.ceil(y1)) + margin_y,
    )


def _intersect_rect(a, b) -> tuple[int, int, int, int] | None:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(int(ax0), int(bx0))
    y0 = max(int(ay0), int(by0))
    x1 = min(int(ax1), int(bx1))
    y1 = min(int(ay1), int(by1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _copy_rendered_tile_into_canvas(rendered: RenderedTile, data, histogram_data, canvas_rect) -> tuple[int, int, int, int] | None:
    tile = rendered.tile
    tile_rect = (int(tile.x0), int(tile.y0), int(tile.x0 + tile.width), int(tile.y0 + tile.height))
    clipped = _intersect_rect(tile_rect, canvas_rect)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    source_x0 = x0 - tile_rect[0]
    source_y0 = y0 - tile_rect[1]
    dest_x0 = x0 - canvas_rect[0]
    dest_y0 = y0 - canvas_rect[1]
    height = y1 - y0
    width = x1 - x0
    source_image = np.asarray(rendered.image)[
        source_y0 : source_y0 + height,
        source_x0 : source_x0 + width,
        ...,
    ]
    source_image = _coerce_rendered_image_for_canvas(source_image, data)
    if source_image is None:
        return None
    data[dest_y0 : dest_y0 + height, dest_x0 : dest_x0 + width, ...] = source_image
    if histogram_data is None:
        return dest_x0, dest_y0, dest_x0 + width, dest_y0 + height
    source = rendered.histogram_data
    if source is None:
        source = rendered.image
        if np.asarray(source).ndim == 3:
            return dest_x0, dest_y0, dest_x0 + width, dest_y0 + height
    histogram_data[dest_y0 : dest_y0 + height, dest_x0 : dest_x0 + width] = np.asarray(source)[
        source_y0 : source_y0 + height,
        source_x0 : source_x0 + width,
    ]
    return dest_x0, dest_y0, dest_x0 + width, dest_y0 + height


def _rgb_trailing_shape(images: Sequence[np.ndarray]) -> tuple[int, ...]:
    for image in images:
        if np.asarray(image).ndim == 3:
            return tuple(int(value) for value in np.asarray(image).shape[2:])
    return (3,)


def _coerce_rendered_image_for_canvas(source, data) -> np.ndarray | None:
    source = np.asarray(source)
    if source.ndim == data.ndim:
        return source
    if source.ndim == 2 and data.ndim == 3:
        return np.broadcast_to(source[..., np.newaxis], source.shape + tuple(data.shape[2:]))
    if source.ndim == 3 and data.ndim == 2:
        return None
    return None


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
