"""Backend-neutral tiled display scene and region model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

from arrayscope.display.geometry import DisplayGeometry


class DisplayLayout(Enum):
    SINGLE = "single"
    MONTAGE = "montage"


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
    regions: tuple[DisplayRegion, ...]
    bounds: tuple[float, float, float, float]
    _region_by_id: dict[int, DisplayRegion] = field(init=False, repr=False, compare=False)
    _active_region_ids: tuple[int, ...] = field(init=False, repr=False)
    _planned_region_ids: tuple[int, ...] = field(init=False, repr=False)
    _near_region_ids: tuple[int, ...] = field(init=False, repr=False)
    _resident_region_ids: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        regions = tuple(self.regions)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self, "_region_by_id", {int(region.region_id): region for region in regions}
        )
        object.__setattr__(
            self,
            "_active_region_ids",
            tuple(region.region_id for region in regions if region.active),
        )
        object.__setattr__(
            self,
            "_planned_region_ids",
            tuple(region.region_id for region in regions if region.planned),
        )
        object.__setattr__(
            self, "_near_region_ids", tuple(region.region_id for region in regions if region.near)
        )
        object.__setattr__(
            self,
            "_resident_region_ids",
            tuple(region.region_id for region in regions if region.resident),
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

    @property
    def resident_region_ids(self) -> tuple[int, ...]:
        return self._resident_region_ids

    def region(self, region_id: int) -> DisplayRegion | None:
        return self._region_by_id.get(int(region_id))


def display_scene_for_presentation(presentation) -> DisplayScene:
    """Build the semantic scene represented by a decided presentation."""

    tile_state = getattr(presentation, "tile_state", None)
    tile_delta = getattr(presentation, "tile_delta", None)
    if tile_state is None or tile_delta is None:
        raise TypeError("display scenes require tiled presentation state")
    payloads = getattr(tile_state, "payloads", {}) if tile_state is not None else {}
    return display_scene_for_geometry(
        presentation.geometry,
        payloads=payloads,
        tile_delta=tile_delta,
        frame_plan=getattr(presentation, "frame_plan", None),
    )


def display_scene_for_geometry(
    geometry: DisplayGeometry,
    *,
    payloads: Mapping[int, object] | None = None,
    tile_delta=None,
    frame_plan=None,
) -> DisplayScene:
    """Build scene state without depending on a concrete presentation class."""

    if frame_plan is not None:
        return _scene_for_frame_plan(
            geometry,
            payloads=payloads,
            tile_delta=tile_delta,
            frame_plan=frame_plan,
        )
    montage = geometry.montage
    if montage is None:
        height, width = geometry.display_shape
        bounds = _shape_bounds(height, width)
        return DisplayScene(
            geometry=geometry,
            layout=DisplayLayout.SINGLE,
            regions=(
                DisplayRegion(
                    region_id=0,
                    source_index=None,
                    bounds=bounds,
                    resident=0 in {int(key) for key in dict(payloads or {})},
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
        payload_keys,
        tuple(geometry.montage_tile_states or ()),
        active,
        planned,
        near,
    )


@lru_cache(maxsize=128)
def _montage_scene_cached(
    geometry: DisplayGeometry,
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
                resident=tile_number in payload_key_set,
            )
        )

    full_width = montage.columns * montage.tile_width + max(0, montage.columns - 1) * montage.gap
    full_height = montage.rows * montage.tile_height + max(0, montage.rows - 1) * montage.gap
    return DisplayScene(
        geometry=geometry,
        layout=DisplayLayout.MONTAGE,
        regions=tuple(regions),
        bounds=_shape_bounds(full_height, full_width),
    )


def _scene_for_frame_plan(
    geometry: DisplayGeometry,
    *,
    payloads: Mapping[int, object] | None,
    tile_delta,
    frame_plan,
) -> DisplayScene:
    payload_key_set = {int(key) for key in dict(payloads or {})}
    if tile_delta is not None:
        active = _unique_int_tuple(getattr(tile_delta, "active_tiles", ()) or ())
        planned = _unique_int_tuple(getattr(tile_delta, "planned_tiles", ()) or ())
        near = _unique_int_tuple(getattr(tile_delta, "near_tiles", ()) or ())
    else:
        active = tuple(int(value) for value in getattr(frame_plan, "active_region_ids", ()) or ())
        planned = tuple(int(value) for value in getattr(frame_plan, "planned_region_ids", ()) or ())
        near = tuple(int(value) for value in getattr(frame_plan, "near_region_ids", ()) or ())
    regions = _frame_plan_regions_cached(
        id(frame_plan),
        tuple(getattr(frame_plan, "scene_region_signature", ()) or ()),
        active,
        planned,
        near,
        tuple(sorted(payload_key_set)),
    )
    if not regions:
        return display_scene_for_geometry(geometry, payloads=payloads, frame_plan=None)
    layout = getattr(frame_plan, "layout", None)
    layout = (
        layout
        if isinstance(layout, DisplayLayout)
        else DisplayLayout(str(getattr(layout, "value", layout or "single")))
    )
    return DisplayScene(
        geometry=geometry,
        layout=layout,
        regions=regions,
        bounds=_shape_bounds(*geometry.display_shape),
    )


@lru_cache(maxsize=256)
def _frame_plan_regions_cached(
    _frame_plan_id: int,
    frame_regions: tuple[tuple[int, int | None, tuple[float, float, float, float]], ...],
    active: tuple[int, ...],
    planned: tuple[int, ...],
    near: tuple[int, ...],
    resident: tuple[int, ...],
) -> tuple[DisplayRegion, ...]:
    active_ids = {int(value) for value in active}
    planned_ids = {int(value) for value in planned}
    near_ids = {int(value) for value in near}
    resident_ids = {int(value) for value in resident}
    return tuple(
        DisplayRegion(
            region_id=int(region_id),
            source_index=None if source_index is None else int(source_index),
            bounds=tuple(float(value) for value in bounds),
            status="loaded",
            active=int(region_id) in active_ids,
            planned=int(region_id) in planned_ids,
            near=int(region_id) in near_ids,
            resident=int(region_id) in resident_ids,
        )
        for region_id, source_index, bounds in frame_regions
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
    "display_scene_for_geometry",
    "display_scene_for_presentation",
]
