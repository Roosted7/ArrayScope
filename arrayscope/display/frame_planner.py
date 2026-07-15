"""Unified semantic frame planning for tiled image presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.display.backend_contract import ImageViewBackendCapabilities
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.montage import MontageTileState, make_montage_plan
from arrayscope.display.scene import DisplayLayout


DEFAULT_INTERNAL_TILE_SHAPE = (1024, 1024)

# Chunk size for source-anchored plans (ADR 0055 G3). Finer than the classic
# internal tile so that windows a few chunks wide contain *interior* chunks —
# the ones whose content keys survive a window shift. A window smaller than
# ~3 chunks still re-materializes fully in G3b-1 (full-chunk materialization
# with camera-clipped windows lifts that in the next stage).
ANCHORED_CHUNK_SHAPE = (256, 256)


@dataclass(frozen=True)
class FrameRegion:
    """One semantic source/display region in a frame plan."""

    region_id: int
    source_index: int | None
    bounds: tuple[float, float, float, float]
    data_slices: tuple[slice, slice]
    view_state: object
    active: bool = True
    planned: bool = True
    near: bool = True
    materialization_key: object | None = None
    # ADR 0055 G3: (y0, y1, x0, x1) content coordinates of this region on a
    # source-anchored tile grid — invariant under display-window shifts on
    # anchored axes. None on the classic window-relative path.
    source_rect: tuple[int, int, int, int] | None = None

    @property
    def width(self) -> int:
        return max(0, int(self.bounds[2] - self.bounds[0] + 1.0))

    @property
    def height(self) -> int:
        return max(0, int(self.bounds[3] - self.bounds[1] + 1.0))


@dataclass(frozen=True)
class FramePlan:
    """Backend-independent frame layout and tile-region decision."""

    target: FrameTarget
    geometry: DisplayGeometry
    layout: DisplayLayout
    tile_shape: tuple[int, int]
    regions: tuple[FrameRegion, ...]
    semantic_key: object
    materialization_key: object
    # Shift-invariant content identity shared by this plan's source-anchored
    # regions (ADR 0055 G3); None when the plan is window-relative.
    source_content_key: object | None = None
    _active_region_ids: tuple[int, ...] = field(init=False, repr=False)
    _planned_region_ids: tuple[int, ...] = field(init=False, repr=False)
    _near_region_ids: tuple[int, ...] = field(init=False, repr=False)
    scene_region_signature: tuple[tuple[int, int | None, tuple[float, float, float, float]], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_active_region_ids", tuple(region.region_id for region in self.regions if region.active))
        object.__setattr__(self, "_planned_region_ids", tuple(region.region_id for region in self.regions if region.planned))
        object.__setattr__(self, "_near_region_ids", tuple(region.region_id for region in self.regions if region.near))
        object.__setattr__(
            self,
            "scene_region_signature",
            tuple(
                (
                    int(region.region_id),
                    None if region.source_index is None else int(region.source_index),
                    tuple(float(value) for value in region.bounds),
                )
                for region in self.regions
            ),
        )

    @property
    def active_region_ids(self) -> tuple[int, ...]:
        return self._active_region_ids

    @property
    def planned_region_ids(self) -> tuple[int, ...]:
        return self._planned_region_ids

    @property
    def near_region_ids(self) -> tuple[int, ...]:
        return self._near_region_ids


class FramePlanner:
    """Plan one semantic tiled surface for image frames."""

    def __init__(
        self,
        *,
        internal_tile_shape: tuple[int, int] = DEFAULT_INTERNAL_TILE_SHAPE,
    ) -> None:
        self.internal_tile_shape = _shape2(internal_tile_shape)

    def plan(
        self,
        *,
        target: FrameTarget,
        view_state,
        display_shape: tuple[int, int],
        backend_capabilities: ImageViewBackendCapabilities,
        viewport_shape: tuple[int, int] | None = None,
        view_range=None,
        memory_policy=None,
        montage_plan=None,
        source_anchoring=None,
    ) -> FramePlan:
        display_shape = _shape2(display_shape)
        if getattr(view_state, "montage_axis", None) is not None:
            return self._plan_montage(
                target=target,
                view_state=view_state,
                display_shape=display_shape,
                viewport_shape=viewport_shape,
                view_range=view_range,
                montage_plan=montage_plan,
            )
        return self._plan_single(
            target=target,
            view_state=view_state,
            display_shape=display_shape,
            view_range=view_range,
            memory_policy=memory_policy,
            source_anchoring=source_anchoring,
        )

    def _plan_single(
        self,
        *,
        target: FrameTarget,
        view_state,
        display_shape: tuple[int, int],
        view_range,
        memory_policy,
        source_anchoring=None,
    ) -> FramePlan:
        anchored_starts = (None, None)
        content_key = None
        if source_anchoring is not None:
            # A fully-unanchored anchoring (no windowable axis, e.g. FFT on
            # both display axes) still content-keys the plan: the grid stays
            # window-relative (starts 0) and every window is folded into the
            # content key, so reuse happens only when the identical
            # plane/window is revisited — never across a shift.
            anchored_starts = tuple(source_anchoring.anchored_starts)
            content_key = source_anchoring.content_key
        if content_key is not None:
            tile_shape = (
                min(int(display_shape[0]), ANCHORED_CHUNK_SHAPE[0]) if anchored_starts[0] is None else ANCHORED_CHUNK_SHAPE[0],
                min(int(display_shape[1]), ANCHORED_CHUNK_SHAPE[1]) if anchored_starts[1] is None else ANCHORED_CHUNK_SHAPE[1],
            )
        else:
            tile_shape = self._single_tile_shape(display_shape)
        row_origins = _axis_origins(display_shape[0], tile_shape[0], anchored_starts[0])
        column_origins = _axis_origins(display_shape[1], tile_shape[1], anchored_starts[1])
        regions = tuple(
            self._single_tile_regions(
                view_state,
                display_shape,
                tile_shape,
                view_range,
                row_origins=row_origins,
                column_origins=column_origins,
                anchored_starts=anchored_starts,
                anchored=content_key is not None,
            )
        )
        columns = max(1, len(column_origins))
        rows = max(1, len(row_origins))
        geometry = DisplayGeometry(
            view_state=view_state,
            display_shape=display_shape,
            montage=MontageGeometry(
                indices=tuple(int(region.region_id) for region in regions),
                tile_shape=tile_shape,
                columns=columns,
                rows=rows,
                gap=0,
            ),
            montage_tile_states=tuple(MontageTileState.LOADED for _region in regions),
        )
        materialization_key = (
            target.semantic_key,
            "single",
            display_shape,
            tile_shape,
        )
        return FramePlan(
            target=target,
            geometry=geometry,
            layout=DisplayLayout.SINGLE,
            tile_shape=tile_shape,
            regions=regions,
            semantic_key=target.semantic_key,
            materialization_key=materialization_key,
            source_content_key=content_key,
        )

    def _plan_montage(
        self,
        *,
        target: FrameTarget,
        view_state,
        display_shape: tuple[int, int],
        viewport_shape: tuple[int, int] | None,
        view_range,
        montage_plan=None,
    ) -> FramePlan:
        axis = int(view_state.montage_axis)
        if montage_plan is not None:
            # The applied montage layout is the single source of truth.  The
            # viewport planner may override requested columns (automatic
            # layout while the camera is auto-owned), so re-deriving columns
            # from the raw view state here would disagree with the applied
            # plan and commit a geometry the camera was never fitted to.
            indices = tuple(int(tile.source_index) for tile in montage_plan.tiles)
            tile_shape = tuple(int(value) for value in montage_plan.tile_shape)
        else:
            indices = tuple(view_state.montage_indices or tuple(range(int(view_state.shape[axis]))))
            image_axes = tuple(view_state.image_axes or ())
            tile_shape = (
                _display_axis_size(view_state, image_axes[0]),
                _display_axis_size(view_state, image_axes[1]),
            )
            montage_plan = make_montage_plan(
                view_state,
                axis=axis,
                indices=indices,
                tile_shape=tile_shape,
                columns=view_state.montage_columns,
                viewport_shape=viewport_shape,
            )
        geometry = DisplayGeometry(
            view_state=view_state,
            display_shape=display_shape,
            montage=montage_plan.geometry,
        )
        active = {
            int(tile.montage_index)
            for tile in montage_plan.tiles_intersecting(view_range, margin_tiles=0)
        }
        if view_range is None:
            active = {int(tile.montage_index) for tile in montage_plan.tiles}
        near = {
            int(tile.montage_index)
            for tile in montage_plan.tiles_intersecting(view_range, margin_tiles=1)
        }
        if view_range is None:
            near = active
        planned = {int(tile.montage_index) for tile in montage_plan.tiles}
        regions = tuple(
            FrameRegion(
                region_id=int(tile.montage_index),
                source_index=int(tile.source_index),
                bounds=(
                    float(tile.x0),
                    float(tile.y0),
                    float(tile.x0 + tile.width - 1),
                    float(tile.y0 + tile.height - 1),
                ),
                data_slices=(slice(0, int(tile.height)), slice(0, int(tile.width))),
                view_state=tile.view_state,
                active=int(tile.montage_index) in active,
                planned=int(tile.montage_index) in planned,
                near=int(tile.montage_index) in near,
                materialization_key=(target.semantic_key, "montage", axis, int(tile.source_index)),
            )
            for tile in montage_plan.tiles
        )
        return FramePlan(
            target=target,
            geometry=geometry,
            layout=DisplayLayout.MONTAGE,
            tile_shape=tile_shape,
            regions=regions,
            semantic_key=target.semantic_key,
            materialization_key=(target.semantic_key, "montage", axis, indices, tile_shape),
        )

    def _single_tile_shape(self, display_shape: tuple[int, int]) -> tuple[int, int]:
        return (
            min(int(display_shape[0]), int(self.internal_tile_shape[0])),
            min(int(display_shape[1]), int(self.internal_tile_shape[1])),
        )

    @staticmethod
    def _single_tile_regions(
        view_state,
        display_shape: tuple[int, int],
        tile_shape: tuple[int, int],
        view_range,
        *,
        row_origins: tuple[int, ...] | None = None,
        column_origins: tuple[int, ...] | None = None,
        anchored_starts: tuple[int | None, int | None] = (None, None),
        anchored: bool = False,
    ) -> Iterable[FrameRegion]:
        height, width = display_shape
        tile_h, tile_w = tile_shape
        if row_origins is None:
            row_origins = _axis_origins(height, tile_h, None)
        if column_origins is None:
            column_origins = _axis_origins(width, tile_w, None)
        y_start = int(anchored_starts[0] or 0)
        x_start = int(anchored_starts[1] or 0)
        active_rect = _view_rect_or_full(view_range, width=width, height=height)
        region_id = 0
        for y0 in row_origins:
            y1 = _axis_span_end(y0, height, tile_h, y_start)
            for x0 in column_origins:
                x1 = _axis_span_end(x0, width, tile_w, x_start)
                active = _rects_intersect((x0, y0, x1, y1), active_rect)
                # Content coordinates: source-anchored on anchored axes,
                # window-relative (start 0) otherwise — that axis's window
                # stays part of the plan's content key, so identity is still
                # correct across shifts on the anchored axis alone.
                source_rect = (y_start + y0, y_start + y1, x_start + x0, x_start + x1)
                yield FrameRegion(
                    region_id=region_id,
                    source_index=None,
                    bounds=(float(x0), float(y0), float(x1 - 1), float(y1 - 1)),
                    data_slices=(slice(y0, y1), slice(x0, x1)),
                    view_state=view_state,
                    active=active,
                    planned=True,
                    near=active,
                    materialization_key=(
                        ("single-src", source_rect)
                        if anchored
                        else ("single", region_id, (y0, y1, x0, x1))
                    ),
                    source_rect=source_rect if anchored else None,
                )
                region_id += 1


def _axis_origins(extent: int, tile: int, start: int | None) -> tuple[int, ...]:
    """Window-relative tile origins along one axis.

    Unanchored (``start is None``): multiples of ``tile`` from 0. Anchored:
    tile boundaries fall on *source-coordinate* multiples of ``tile``, so the
    first window tile is clipped when the window starts mid-chunk and the
    interior tiles stay identical under window shifts.
    """

    extent = int(extent)
    tile = int(tile)
    if extent <= 0:
        return ()
    if start is None or int(start) % tile == 0:
        return tuple(range(0, extent, tile))
    first_boundary = ((int(start) // tile) + 1) * tile
    origins = [0]
    origins.extend(range(first_boundary - int(start), extent, tile))
    return tuple(origins)


def _axis_span_end(origin: int, extent: int, tile: int, start: int) -> int:
    """Window-relative end of the tile at ``origin`` (next source boundary)."""

    to_boundary = tile - (int(start) + int(origin)) % tile
    return min(int(extent), int(origin) + to_boundary)


def _shape2(shape) -> tuple[int, int]:
    values = tuple(int(value) for value in tuple(shape)[:2])
    if len(values) != 2 or values[0] < 1 or values[1] < 1:
        raise ValueError("frame planner shapes must be positive (height, width)")
    return values


def _pixel_count(shape: tuple[int, int]) -> int:
    return int(shape[0]) * int(shape[1])


def _display_axis_size(view_state, axis: int) -> int:
    indices = view_state.axis_range_indices[int(axis)]
    if indices is not None:
        return len(indices)
    return int(view_state.shape[int(axis)])


def _view_rect_or_full(view_range, *, width: int, height: int) -> tuple[int, int, int, int]:
    if view_range is None:
        return (0, 0, int(width), int(height))
    x_range, y_range = view_range
    x0, x1 = sorted((int(float(x_range[0])), int(float(x_range[1]))))
    y0, y1 = sorted((int(float(y_range[0])), int(float(y_range[1]))))
    return (
        max(0, min(int(width), x0)),
        max(0, min(int(height), y0)),
        max(0, min(int(width), x1)),
        max(0, min(int(height), y1)),
    )


def _rects_intersect(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    left_x0, left_y0, left_x1, left_y1 = left
    right_x0, right_y0, right_x1, right_y1 = right
    return left_x1 > right_x0 and left_x0 < right_x1 and left_y1 > right_y0 and left_y0 < right_y1


__all__ = ["FramePlanner", "FramePlan", "FrameRegion"]
