"""Pure NumPy montage display helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np

from arrayscope.display.geometry import MontageGeometry
from arrayscope.display.lod import LodInfo
from arrayscope.display.shader_mapping import ShaderMapping, TexturePlaneKind


@dataclass(frozen=True)
class MontageLayout:
    tile_shape: tuple[int, int]
    count: int
    columns: int
    rows: int
    gap: int = 1


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
    axis: int | None
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

    def display_rect_for_tiles(
        self, tiles: Sequence[MontageTile]
    ) -> tuple[int, int, int, int] | None:
        tiles = tuple(tiles)
        if not tiles:
            return None
        x0 = min(int(tile.x0) for tile in tiles)
        y0 = min(int(tile.y0) for tile in tiles)
        x1 = max(int(tile.x0 + tile.width) for tile in tiles)
        y1 = max(int(tile.y0 + tile.height) for tile in tiles)
        return x0, y0, x1, y1

    def _axis_span(self, lo: float, hi: float, *, axis: int) -> tuple[int, int]:
        """Inclusive row/column range covering ``[lo, hi]`` on one axis.

        The montage is a regular grid — ``x0 = col * (width + gap)`` — so the
        covered range is arithmetic, not a search.  Solving the membership
        test ``tile_x1 >= lo and tile.x0 <= hi`` for the integer index gives
        ``ceil((lo - extent) / stride) <= index <= floor(hi / stride)``.
        Returns an empty range (``lo > hi``) when nothing is covered.
        """

        extent = self.tile_shape[1] if axis else self.tile_shape[0]
        limit = (self.columns if axis else self.rows) - 1
        stride = extent + self.gap
        if stride <= 0 or limit < 0:
            return 0, limit
        return (
            max(0, math.ceil((lo - extent) / stride)),
            min(limit, math.floor(hi / stride)),
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
        # View ranges and tile geometry are expressed on pixel-center
        # coordinates. A tile landing exactly on the range boundary can
        # therefore contribute the boundary pixel and is a render obligation,
        # not speculative coverage — hence the inclusive bounds above.
        col0, col1 = self._axis_span(x0, x1, axis=1)
        row0, row1 = self._axis_span(y0, y1, axis=0)
        if col0 > col1 or row0 > row1:
            return ()
        count = len(self.tiles)
        columns = self.columns
        visible: list[MontageTile] = []
        for row in range(row0, row1 + 1):
            start = row * columns + col0
            if start >= count:
                # A partially filled last row ends the grid; later rows cannot
                # contribute tiles that do not exist.
                break
            visible.extend(self.tiles[start : min(row * columns + col1 + 1, count)])
        return tuple(visible)


def make_montage_plan(
    view_state, *, axis, indices, tile_shape, columns=None, viewport_shape=None, gap=1
):
    """Build the montage layout, reusing the previous plan when unchanged.

    A pan or zoom leaves the layout — indices, tile shape, columns, gap —
    completely untouched, yet this is called on every viewport retarget.
    Rebuilding allocated one ``MontageTile`` per source index each time, and
    the fresh object also missed every downstream cache keyed on ``id(plan)``
    (near-tile numbers, per-tile source ids, plan-tile maps), so the O(N)
    cost landed several times over. The plan is a pure function of the
    normalized arguments, so returning the identical object lets those caches
    hit. ``MontagePlan`` is frozen; callers compare layouts by ``geometry``
    value, never by identity.
    """

    indices = tuple(int(index) for index in indices)
    tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
    gap = max(0, int(gap))
    axis = None if axis is None else int(axis)
    columns = None if columns is None else int(columns)
    viewport_shape = (
        None if viewport_shape is None else tuple(int(value) for value in viewport_shape)
    )
    try:
        return _make_montage_plan_cached(
            view_state, axis, indices, tile_shape, columns, viewport_shape, gap
        )
    except TypeError:
        # An unhashable stand-in view state (test doubles) still gets a plan;
        # it just does not get the reuse.
        return _build_montage_plan(
            view_state, axis, indices, tile_shape, columns, viewport_shape, gap
        )


@lru_cache(maxsize=4)
def _make_montage_plan_cached(
    view_state, axis, indices, tile_shape, columns, viewport_shape, gap
) -> MontagePlan:
    return _build_montage_plan(view_state, axis, indices, tile_shape, columns, viewport_shape, gap)


def _build_montage_plan(
    view_state, axis, indices, tile_shape, columns, viewport_shape, gap
) -> MontagePlan:
    count = len(indices)
    if count == 0:
        return MontagePlan(None if axis is None else int(axis), tile_shape, (0, 1), 1, 0, gap, ())
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
        tile_state = (
            view_state.with_montage_axis(None)
            if axis is None
            else view_state.tile_state_for_slice(axis, source_index)
        )
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
    return MontagePlan(
        None if axis is None else int(axis),
        tile_shape,
        (rows, columns),
        columns,
        rows,
        gap,
        tuple(tiles),
    )


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
    semantic_histogram_data: np.ndarray | None = None
    lod_source_data: np.ndarray | None = None
    lod: LodInfo | None = None
    level_data: np.ndarray | None = None
    level_stats: object | None = None
    quality: str = "exact"

    def nbytes(self) -> int:
        total = int(self.image.nbytes)
        if isinstance(self.histogram_data, np.ndarray):
            total += int(self.histogram_data.nbytes)
        if isinstance(self.semantic_data, np.ndarray) and self.semantic_data is not self.image:
            total += int(self.semantic_data.nbytes)
        if (
            isinstance(self.lod_source_data, np.ndarray)
            and self.lod_source_data is not self.image
            and self.lod_source_data is not self.semantic_data
        ):
            total += int(self.lod_source_data.nbytes)
        if (
            isinstance(self.semantic_histogram_data, np.ndarray)
            and self.semantic_histogram_data is not self.histogram_data
            and self.semantic_histogram_data is not self.semantic_data
        ):
            total += int(self.semantic_histogram_data.nbytes)
        if (
            isinstance(self.level_data, np.ndarray)
            and self.level_data is not self.image
            and self.level_data is not self.histogram_data
        ):
            total += int(self.level_data.nbytes)
        return total

    def bind(self, tile: MontageTile) -> RenderedTile:
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
            semantic_histogram_data=self.semantic_histogram_data,
            lod_source_data=self.lod_source_data,
            lod=self.lod,
            level_data=self.level_data,
            level_stats=self.level_stats,
            quality=self.quality,
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
    semantic_histogram_data: np.ndarray | None = None
    lod_source_data: np.ndarray | None = None
    lod: LodInfo | None = None
    level_data: np.ndarray | None = None
    level_stats: object | None = None
    quality: str = "exact"

    def nbytes(self) -> int:
        total = int(self.image.nbytes)
        if isinstance(self.histogram_data, np.ndarray):
            total += int(self.histogram_data.nbytes)
        if isinstance(self.semantic_data, np.ndarray) and self.semantic_data is not self.image:
            total += int(self.semantic_data.nbytes)
        if (
            isinstance(self.lod_source_data, np.ndarray)
            and self.lod_source_data is not self.image
            and self.lod_source_data is not self.semantic_data
        ):
            total += int(self.lod_source_data.nbytes)
        if (
            isinstance(self.semantic_histogram_data, np.ndarray)
            and self.semantic_histogram_data is not self.histogram_data
            and self.semantic_histogram_data is not self.semantic_data
        ):
            total += int(self.semantic_histogram_data.nbytes)
        if (
            isinstance(self.level_data, np.ndarray)
            and self.level_data is not self.image
            and self.level_data is not self.histogram_data
        ):
            total += int(self.level_data.nbytes)
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
            semantic_histogram_data=self.semantic_histogram_data,
            lod_source_data=self.lod_source_data,
            lod=self.lod,
            level_data=self.level_data,
            level_stats=self.level_stats,
            quality=self.quality,
        )


def montage_rect_for_viewport(
    plan: MontagePlan, *, view_range=None, viewport_shape=None
) -> tuple[int, int, int, int]:
    full_height, full_width = plan.display_shape
    if full_height <= 0 or full_width <= 0:
        return (0, 0, 1, 1)
    if view_range is None:
        # No measured camera yet (e.g. on load before the view is sized, when
        # `_montage_viewport_plan` falls through to current_range=None).  A
        # montage defaults to fit-to-window, so the WHOLE montage is visible.
        # The previous `min(full, viewport_shape)` clamp treated the window's
        # PIXEL height/width as WORLD extent and cropped the visible set to the
        # top-left corner — the "on load only ~6 of N tiles appear and the rest
        # park until you scroll" bug (viewport_shape is pixels; display_shape is
        # world).  Cover the full montage instead; a real view_range refines it.
        rect = (0, 0, max(1, int(full_width)), max(1, int(full_height)))
        return _expand_rect_to_tile_bounds(plan, rect)
    rect = _rect_for_view_range(view_range, plan, viewport_shape)
    rect = _intersect_rect(rect, (0, 0, full_width, full_height)) or (
        0,
        0,
        min(1, full_width),
        min(1, full_height),
    )
    return _expand_rect_to_tile_bounds(plan, rect)


def _expand_rect_to_tile_bounds(plan: MontagePlan, rect) -> tuple[int, int, int, int]:
    full_height, full_width = plan.display_shape
    tile_height, tile_width = plan.tile_shape
    stride_x = tile_width + plan.gap
    stride_y = tile_height + plan.gap
    count = len(plan.tiles)
    columns = plan.columns
    if not count or stride_x <= 0 or stride_y <= 0:
        return rect
    rx0, ry0, rx1, ry1 = (int(value) for value in rect)
    if rx1 <= rx0 or ry1 <= ry0:
        return rect
    # ``_intersect_rect`` demands POSITIVE overlap area (``x1 <= x0`` is a
    # miss), unlike ``tiles_intersecting``'s inclusive pixel-center bounds.
    # Solving ``tile_x1 > rx0 and tile_x0 < rx1`` for the integer grid index
    # gives the strict form below — do not unify these two conventions.
    col0 = max(0, math.floor((rx0 - tile_width) / stride_x) + 1)
    col1 = min(columns - 1, math.ceil(rx1 / stride_x) - 1)
    row0 = max(0, math.floor((ry0 - tile_height) / stride_y) + 1)
    row1 = min(plan.rows - 1, math.ceil(ry1 / stride_y) - 1)
    if col0 > col1 or row0 > row1:
        return rect
    x0 = y0 = x1 = y1 = None
    for row in range(row0, row1 + 1):
        base = row * columns
        if base + col0 >= count:
            # Partially filled last row: no further row holds real tiles.
            break
        # The last row can stop short of ``col1``, which narrows the bound.
        end_col = min(col1, count - 1 - base)
        if end_col < col0:
            continue
        if x0 is None:
            x0, y0 = col0 * stride_x, row * stride_y
        x1 = max(x1 or 0, end_col * stride_x + tile_width)
        y1 = row * stride_y + tile_height
    if x0 is None:
        return rect
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


def _rect_for_view_range(
    view_range, plan: MontagePlan, viewport_shape
) -> tuple[int, int, int, int]:
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


def optimal_montage_columns(count, tile_shape, viewport_shape, gap=1):
    count = max(1, int(count))
    tile_height, tile_width = int(tile_shape[0]), int(tile_shape[1])
    viewport_height, viewport_width = int(viewport_shape[0]), int(viewport_shape[1])
    gap = max(0, int(gap))
    viewport_aspect = viewport_width / max(viewport_height, 1)
    stride_width = max(tile_width + gap, 1)
    stride_height = max(tile_height + gap, 1)
    estimate = math.sqrt(max(1.0, count * viewport_aspect * stride_height / stride_width))
    candidates = {1, count}
    center = round(estimate)
    for value in range(center - 8, center + 9):
        if 1 <= value <= count:
            candidates.add(value)
    row_estimate = math.sqrt(max(1.0, count * stride_width / (viewport_aspect * stride_height)))
    row_center = round(row_estimate)
    for rows in range(row_center - 8, row_center + 9):
        if rows > 0:
            columns = math.ceil(count / rows)
            if 1 <= columns <= count:
                candidates.add(columns)

    best_columns = 1
    best_score = (-1.0, -float("inf"), -float("inf"))
    viewport_area = max(viewport_width * viewport_height, 1)
    for columns in candidates:
        rows = math.ceil(count / columns)
        total_width = columns * tile_width + gap * (columns - 1)
        total_height = rows * tile_height + gap * (rows - 1)
        scale = min(viewport_width / max(total_width, 1), viewport_height / max(total_height, 1))
        used_area = (total_width * scale) * (total_height * scale)
        used_fraction = min(used_area / viewport_area, 1.0)
        layout_aspect = total_width / max(total_height, 1)
        aspect_error = (
            abs(np.log(layout_aspect / viewport_aspect))
            if layout_aspect > 0 and viewport_aspect > 0
            else float("inf")
        )
        score = (round(float(used_fraction), 12), round(float(-aspect_error), 12), float(scale))
        if score > best_score:
            best_columns = columns
            best_score = score
    return best_columns
