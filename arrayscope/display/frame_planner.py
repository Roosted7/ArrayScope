"""Unified semantic frame planning for raster and tiled image presentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable

from arrayscope.core.scheduler import FrameTarget
from arrayscope.display.backend_contract import ImageViewBackendCapabilities
from arrayscope.display.geometry import DisplayGeometry
from arrayscope.display.montage import make_montage_plan
from arrayscope.display.scene import DisplayLayout, DisplayStorage


DEFAULT_INTERNAL_TILE_SHAPE = (1024, 1024)


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
    storage: DisplayStorage
    tile_shape: tuple[int, int]
    regions: tuple[FrameRegion, ...]
    semantic_key: object
    materialization_key: object
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
    """Plan one semantic tiled surface for normal images and montages."""

    def __init__(
        self,
        *,
        internal_tile_shape: tuple[int, int] = DEFAULT_INTERNAL_TILE_SHAPE,
        max_raster_pixels: int | None = None,
    ) -> None:
        self.internal_tile_shape = _shape2(internal_tile_shape)
        default_pixels = self.internal_tile_shape[0] * self.internal_tile_shape[1]
        self.max_raster_pixels = int(max_raster_pixels or default_pixels)

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
    ) -> FramePlan:
        display_shape = _shape2(display_shape)
        if getattr(view_state, "montage_axis", None) is not None:
            return self._plan_montage(
                target=target,
                view_state=view_state,
                display_shape=display_shape,
                backend_capabilities=backend_capabilities,
                viewport_shape=viewport_shape,
                view_range=view_range,
            )
        return self._plan_single(
            target=target,
            view_state=view_state,
            display_shape=display_shape,
            backend_capabilities=backend_capabilities,
            view_range=view_range,
            memory_policy=memory_policy,
        )

    def _plan_single(
        self,
        *,
        target: FrameTarget,
        view_state,
        display_shape: tuple[int, int],
        backend_capabilities: ImageViewBackendCapabilities,
        view_range,
        memory_policy,
    ) -> FramePlan:
        tile_shape = self._single_tile_shape(display_shape, memory_policy=memory_policy)
        force_tiled = _pixel_count(display_shape) > self.max_raster_pixels
        can_tile = bool(getattr(backend_capabilities, "direct_montage_tile_payloads", False))
        storage = DisplayStorage.TILED if force_tiled and can_tile else DisplayStorage.RASTER
        geometry = DisplayGeometry(view_state=view_state, display_shape=display_shape)
        regions = (
            self._single_region(view_state=view_state, display_shape=display_shape)
            if storage is DisplayStorage.RASTER
            else tuple(self._single_tile_regions(view_state, display_shape, tile_shape, view_range))
        )
        materialization_key = (
            target.semantic_key,
            "single",
            display_shape,
            tile_shape if storage is DisplayStorage.TILED else display_shape,
        )
        return FramePlan(
            target=target,
            geometry=geometry,
            layout=DisplayLayout.SINGLE,
            storage=storage,
            tile_shape=display_shape if storage is DisplayStorage.RASTER else tile_shape,
            regions=regions,
            semantic_key=target.semantic_key,
            materialization_key=materialization_key,
        )

    def _plan_montage(
        self,
        *,
        target: FrameTarget,
        view_state,
        display_shape: tuple[int, int],
        backend_capabilities: ImageViewBackendCapabilities,
        viewport_shape: tuple[int, int] | None,
        view_range,
    ) -> FramePlan:
        axis = int(view_state.montage_axis)
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
        storage = (
            DisplayStorage.TILED
            if getattr(backend_capabilities, "direct_montage_tile_payloads", False)
            else DisplayStorage.RASTER
        )
        return FramePlan(
            target=target,
            geometry=geometry,
            layout=DisplayLayout.MONTAGE,
            storage=storage,
            tile_shape=tile_shape,
            regions=regions,
            semantic_key=target.semantic_key,
            materialization_key=(target.semantic_key, "montage", axis, indices, tile_shape),
        )

    def _single_tile_shape(self, display_shape: tuple[int, int], *, memory_policy) -> tuple[int, int]:
        del memory_policy
        return (
            min(int(display_shape[0]), int(self.internal_tile_shape[0])),
            min(int(display_shape[1]), int(self.internal_tile_shape[1])),
        )

    @staticmethod
    def _single_region(*, view_state, display_shape: tuple[int, int]) -> tuple[FrameRegion, ...]:
        height, width = display_shape
        return (
            FrameRegion(
                region_id=0,
                source_index=None,
                bounds=(0.0, 0.0, float(width - 1), float(height - 1)),
                data_slices=(slice(0, height), slice(0, width)),
                view_state=view_state,
                materialization_key=("single", 0, display_shape),
            ),
        )

    @staticmethod
    def _single_tile_regions(
        view_state,
        display_shape: tuple[int, int],
        tile_shape: tuple[int, int],
        view_range,
    ) -> Iterable[FrameRegion]:
        height, width = display_shape
        tile_h, tile_w = tile_shape
        active_rect = _view_rect_or_full(view_range, width=width, height=height)
        region_id = 0
        for row in range(ceil(height / tile_h)):
            y0 = row * tile_h
            y1 = min(height, y0 + tile_h)
            for column in range(ceil(width / tile_w)):
                x0 = column * tile_w
                x1 = min(width, x0 + tile_w)
                active = _rects_intersect((x0, y0, x1, y1), active_rect)
                yield FrameRegion(
                    region_id=region_id,
                    source_index=None,
                    bounds=(float(x0), float(y0), float(x1 - 1), float(y1 - 1)),
                    data_slices=(slice(y0, y1), slice(x0, x1)),
                    view_state=view_state,
                    active=active,
                    planned=True,
                    near=active,
                    materialization_key=("single", region_id, (y0, y1, x0, x1)),
                )
                region_id += 1


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
