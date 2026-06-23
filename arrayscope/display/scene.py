"""Backend-neutral display scene and region model.

A normal image is a one-region scene. A montage is a multi-region scene.
Whether those regions are stored as one raster, individual PyQtGraph items, or
GPU atlas pages is a backend strategy and must not change coordinate, profile,
or inspection semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Mapping

from arrayscope.display.geometry import DisplayGeometry


class DisplayLayout(Enum):
    SINGLE = "single"
    MONTAGE = "montage"


class DisplayStorage(Enum):
    RASTER = "raster"
    TILED = "tiled"


@dataclass(frozen=True)
class DisplayRegion:
    """One semantic image region in world/display coordinates."""

    region_id: int
    source_index: int | None
    bounds: tuple[float, float, float, float]
    status: str = "loaded"
    active: bool = True
    planned: bool = True
    near: bool = True
    resident: bool = False

    @property
    def width(self) -> float:
        return max(0.0, float(self.bounds[2]) - float(self.bounds[0]) + 1.0)

    @property
    def height(self) -> float:
        return max(0.0, float(self.bounds[3]) - float(self.bounds[1]) + 1.0)


@dataclass(frozen=True)
class DisplayScene:
    """Semantic scene shared by all rendering backends."""

    geometry: DisplayGeometry
    layout: DisplayLayout
    storage: DisplayStorage
    regions: tuple[DisplayRegion, ...]
    bounds: tuple[float, float, float, float]

    @property
    def active_region_ids(self) -> tuple[int, ...]:
        return tuple(region.region_id for region in self.regions if region.active)

    @property
    def planned_region_ids(self) -> tuple[int, ...]:
        return tuple(region.region_id for region in self.regions if region.planned)

    @property
    def near_region_ids(self) -> tuple[int, ...]:
        return tuple(region.region_id for region in self.regions if region.near)

    @property
    def resident_region_ids(self) -> tuple[int, ...]:
        return tuple(region.region_id for region in self.regions if region.resident)

    def region(self, region_id: int) -> DisplayRegion | None:
        region_id = int(region_id)
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None


def display_scene_for_presentation(presentation) -> DisplayScene:
    """Build the semantic scene represented by a decided presentation."""

    tile_state = getattr(presentation, "tile_state", None)
    tile_delta = getattr(presentation, "tile_delta", None)
    storage = DisplayStorage.TILED if tile_state is not None and tile_delta is not None else DisplayStorage.RASTER
    payloads = getattr(tile_state, "payloads", {}) if tile_state is not None else {}
    return display_scene_for_geometry(
        presentation.geometry,
        storage=storage,
        payloads=payloads,
        tile_delta=tile_delta,
    )


def display_scene_for_geometry(
    geometry: DisplayGeometry,
    *,
    storage: DisplayStorage | str = DisplayStorage.RASTER,
    payloads: Mapping[int, object] | None = None,
    tile_delta=None,
) -> DisplayScene:
    """Build scene state without depending on a concrete presentation class."""

    storage = storage if isinstance(storage, DisplayStorage) else DisplayStorage(str(storage))
    montage = geometry.montage
    if montage is None:
        height, width = geometry.display_shape
        bounds = _shape_bounds(height, width)
        return DisplayScene(
            geometry=geometry,
            layout=DisplayLayout.SINGLE,
            storage=storage,
            regions=(
                DisplayRegion(
                    region_id=0,
                    source_index=None,
                    bounds=bounds,
                    resident=True,
                ),
            ),
            bounds=bounds,
        )

    if tile_delta is None:
        tile_count = len(montage.indices)
        active = tuple(range(tile_count))
        planned = active
        near = active
    else:
        active = _unique_int_tuple(getattr(tile_delta, "active_tiles", ()) or ())
        planned = _unique_int_tuple(getattr(tile_delta, "planned_tiles", ()) or ())
        near = _unique_int_tuple(getattr(tile_delta, "near_tiles", ()) or ())
    payload_keys = _unique_int_tuple(dict(payloads or {}))
    return _montage_scene_cached(
        geometry,
        storage,
        payload_keys,
        tuple(geometry.montage_tile_states or ()),
        active,
        planned,
        near,
    )


@lru_cache(maxsize=128)
def _montage_scene_cached(
    geometry: DisplayGeometry,
    storage: DisplayStorage,
    payload_keys: tuple[int, ...],
    states: tuple[object, ...],
    active: tuple[int, ...],
    planned: tuple[int, ...],
    near: tuple[int, ...],
) -> DisplayScene:
    montage = geometry.montage
    if montage is None:
        raise ValueError("cached montage scene requires montage geometry")
    active_set = set(active)
    planned_set = set(planned)
    near_set = set(near)
    payload_key_set = set(payload_keys)
    regions = []
    for tile_number, source_index in enumerate(montage.indices):
        row = tile_number // montage.columns
        column = tile_number % montage.columns
        x0 = column * (montage.tile_width + montage.gap)
        y0 = row * (montage.tile_height + montage.gap)
        bounds = (
            float(x0),
            float(y0),
            float(x0 + montage.tile_width - 1),
            float(y0 + montage.tile_height - 1),
        )
        status = "loaded"
        if states and tile_number < len(states):
            status = str(getattr(states[tile_number], "value", states[tile_number]))
        regions.append(
            DisplayRegion(
                region_id=int(tile_number),
                source_index=int(source_index),
                bounds=bounds,
                status=status,
                active=tile_number in active_set,
                planned=tile_number in planned_set,
                near=tile_number in near_set,
                resident=tile_number in payload_key_set or (storage is DisplayStorage.RASTER and status == "loaded"),
            )
        )

    full_width = montage.columns * montage.tile_width + max(0, montage.columns - 1) * montage.gap
    full_height = montage.rows * montage.tile_height + max(0, montage.rows - 1) * montage.gap
    return DisplayScene(
        geometry=geometry,
        layout=DisplayLayout.MONTAGE,
        storage=storage,
        regions=tuple(regions),
        bounds=_shape_bounds(full_height, full_width),
    )


def _unique_int_tuple(values) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in tuple(values or ())))


def _shape_bounds(height: int, width: int) -> tuple[float, float, float, float]:
    return (
        0.0,
        0.0,
        float(max(0, int(width) - 1)),
        float(max(0, int(height) - 1)),
    )


__all__ = [
    "DisplayLayout",
    "DisplayRegion",
    "DisplayScene",
    "DisplayStorage",
    "display_scene_for_geometry",
    "display_scene_for_presentation",
]
