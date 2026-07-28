"""Tile-region layout shared by tiled display backends."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class TileLayoutRegion:
    tile_number: int
    source_index: int | None
    x: int
    y: int
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.x),
            float(self.y),
            float(self.x + self.width - 1),
            float(self.y + self.height - 1),
        )


@dataclass(frozen=True)
class _TileLayout:
    """Every layout view one commit needs, derived once per geometry."""

    regions: tuple[TileLayoutRegion, ...]
    region_map: dict[int, TileLayoutRegion]
    shape: tuple[int, int]
    planned_count: int


# Layout is a pure function of the montage geometry (or frame plan), but every
# commit asks for it up to three times, and building one ``TileLayoutRegion``
# per tile made that an O(montage) term on a commit that touched one tile.
# Both key objects are frozen values, so identity is a sound cache key; the
# cache holds them alive, which is what makes ``id()`` reuse impossible.
_LAYOUT_CACHE_LIMIT = 4
_LAYOUT_CACHE: dict[int, tuple[object, object, tuple[int, ...], _TileLayout]] = {}


def _resolve_tile_layout(geometry, frame_plan) -> _TileLayout:
    montage = getattr(geometry, "montage", None)
    owner = frame_plan if frame_plan is not None else montage
    fallback_shape = tuple(int(value) for value in getattr(geometry, "display_shape", (1, 1))[:2])
    if owner is None:
        return _TileLayout((), {}, fallback_shape or (1, 1), 1)
    key = id(owner)
    cached = _LAYOUT_CACHE.get(key)
    if (
        cached is not None
        and cached[0] is owner
        and cached[1] is montage
        and cached[2] == fallback_shape
    ):
        return cached[3]
    regions = _build_tile_layout_regions(geometry, frame_plan)
    region_map = {int(region.tile_number): region for region in regions}
    if regions:
        shape = (
            max(1, *(int(region.y + region.height) for region in regions)),
            max(1, *(int(region.x + region.width) for region in regions)),
        )
        planned = len(regions)
    else:
        shape = fallback_shape or (1, 1)
        planned = len(tuple(getattr(montage, "indices", ()) or ())) if montage is not None else 0
    layout = _TileLayout(regions, region_map, shape, planned)
    if len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_LIMIT:
        _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = (owner, montage, fallback_shape, layout)
    return layout


def _build_tile_layout_regions(geometry, frame_plan) -> tuple[TileLayoutRegion, ...]:
    if frame_plan is not None:
        regions = []
        for region in tuple(getattr(frame_plan, "regions", ()) or ()):
            x0, y0, x1, y1 = tuple(float(value) for value in getattr(region, "bounds", ()))
            width = max(0, round(x1 - x0 + 1.0))
            height = max(0, round(y1 - y0 + 1.0))
            if width < 1 or height < 1:
                continue
            regions.append(
                TileLayoutRegion(
                    tile_number=int(region.region_id),
                    source_index=(
                        None
                        if getattr(region, "source_index", None) is None
                        else int(region.source_index)
                    ),
                    x=round(x0),
                    y=round(y0),
                    width=width,
                    height=height,
                )
            )
        return tuple(regions)

    montage = getattr(geometry, "montage", None)
    if montage is None:
        return ()
    tile_w = int(montage.tile_width)
    tile_h = int(montage.tile_height)
    stride_x = tile_w + int(montage.gap)
    stride_y = tile_h + int(montage.gap)
    regions = []
    for tile_number, source_index in enumerate(tuple(montage.indices)):
        row = int(tile_number) // int(montage.columns)
        column = int(tile_number) % int(montage.columns)
        regions.append(
            TileLayoutRegion(
                tile_number=int(tile_number),
                source_index=int(source_index),
                x=int(column * stride_x),
                y=int(row * stride_y),
                width=tile_w,
                height=tile_h,
            )
        )
    return tuple(regions)


def tile_layout_regions(geometry, *, frame_plan=None) -> tuple[TileLayoutRegion, ...]:
    """Return drawable tile placement for either a frame plan or montage geometry."""

    return _resolve_tile_layout(geometry, frame_plan).regions


def tile_layout_map(geometry, *, frame_plan=None) -> dict[int, TileLayoutRegion]:
    """Tile-keyed placement.

    The returned mapping is shared with the layout cache and must be treated
    as read-only; callers that need to mutate placement take their own copy.
    """

    return _resolve_tile_layout(geometry, frame_plan).region_map


def tile_layout_shape(geometry, *, frame_plan=None) -> tuple[int, int]:
    return _resolve_tile_layout(geometry, frame_plan).shape


def planned_tile_count(geometry, *, frame_plan=None, minimum: int = 1) -> int:
    return max(int(minimum), _resolve_tile_layout(geometry, frame_plan).planned_count)


def filter_layout_ids(layout: Iterable[TileLayoutRegion], ids: Iterable[int]) -> tuple[int, ...]:
    valid = {int(region.tile_number) for region in layout}
    return tuple(int(tile) for tile in tuple(ids or ()) if int(tile) in valid)


__all__ = [
    "TileLayoutRegion",
    "filter_layout_ids",
    "planned_tile_count",
    "tile_layout_map",
    "tile_layout_regions",
    "tile_layout_shape",
]
