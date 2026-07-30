"""Tile-region layout shared by tiled display backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


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


class _TileLayout:
    """The layout views one commit needs, each derived at most once.

    Derivation is **lazy** on purpose.  Every caller wants exactly one of
    these four views, and a cache miss must never cost more than computing
    that one view did before this cache existed — otherwise the miss path
    pays for the hit path, and the miss path is the common one whenever the
    geometry is being rebuilt (a crop or slice sweep rebuilds it every step).
    """

    __slots__ = (
        "_fallback_shape",
        "_montage",
        "_plan_regions",
        "_region_map",
        "_regions",
        "_shape",
    )

    def __init__(self, montage, plan_regions, fallback_shape):
        self._montage = montage
        # The plan's regions, not the plan: retaining a FramePlan would retain
        # its geometry, target and layout for as long as this entry is cached.
        self._plan_regions = plan_regions
        self._fallback_shape = fallback_shape
        self._regions = None
        self._region_map = None
        self._shape = None

    @property
    def regions(self) -> tuple[TileLayoutRegion, ...]:
        if self._regions is None:
            self._regions = _build_tile_layout_regions(self._montage, self._plan_regions)
        return self._regions

    @property
    def region_map(self) -> Mapping[int, TileLayoutRegion]:
        if self._region_map is None:
            # Read-only by construction, not by documentation. This mapping is
            # shared by every caller that resolves an equal geometry, so a
            # single stray write would corrupt placement for all of them until
            # the entry is evicted — a failure that would surface as misplaced
            # tiles far from the code that caused it.
            self._region_map = MappingProxyType(
                {int(region.tile_number): region for region in self.regions}
            )
        return self._region_map

    @property
    def shape(self) -> tuple[int, int]:
        if self._shape is None:
            regions = self.regions
            if regions:
                self._shape = (
                    max(1, *(int(region.y + region.height) for region in regions)),
                    max(1, *(int(region.x + region.width) for region in regions)),
                )
            else:
                self._shape = self._fallback_shape or (1, 1)
        return self._shape

    @property
    def planned_count(self) -> int:
        if self._plan_regions is None and self._montage is not None:
            # One region per montage index by construction, so the count needs
            # no region objects at all.
            return len(tuple(getattr(self._montage, "indices", ()) or ()))
        return len(self.regions)


# Layout is a pure function of the montage geometry (or the frame plan's region
# signature), but every commit asks for it up to three times and each ask built
# one TileLayoutRegion per tile — an O(montage) term on a commit that touched
# one tile.
#
# The key is by VALUE, not by object identity. Identity keying measured a 9%
# hit rate in the real workflow (40 hits, 412 misses): the geometry is
# reconstructed constantly, and an equal-but-new MontageGeometry missed every
# time. MontageGeometry is a frozen dataclass and hashes by value; FramePlan is
# not hashable (its regions carry slices), but the region signature it already
# precomputes is exactly this function's input.
_LAYOUT_CACHE_LIMIT = 8
_LAYOUT_CACHE: dict[object, _TileLayout] = {}


def _resolve_tile_layout(geometry, frame_plan) -> _TileLayout:
    montage = getattr(geometry, "montage", None)
    fallback_shape = tuple(int(value) for value in getattr(geometry, "display_shape", (1, 1))[:2])
    if frame_plan is not None:
        # The plan's region signature is not merely a good key — it is the
        # COMPLETE input, the same (region id, source index, bounds) triples
        # the builder reads. Building from it rather than from the plan's
        # regions makes "what the cache keys on" and "what the layout is
        # derived from" the same value, so no input can change without
        # changing the key. It also keeps a cached layout from retaining
        # FrameRegion objects and the view states they reference.
        signature = getattr(frame_plan, "scene_region_signature", None)
        if signature is None:
            return _TileLayout(montage, (), fallback_shape)
        plan_regions = tuple(signature)
        key = ("plan", plan_regions, fallback_shape)
    elif montage is not None:
        # MontageGeometry hashes and compares by value over exactly the fields
        # the builder reads: indices (the active set), tile shape, columns,
        # rows and gap. Placement is world space and does not depend on the
        # viewport, so a pan or zoom correctly does not invalidate it.
        plan_regions = None
        key = ("montage", montage, fallback_shape)
    else:
        return _TileLayout(None, None, fallback_shape or (1, 1))
    cached = _LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    layout = _TileLayout(montage, plan_regions, fallback_shape)
    if len(_LAYOUT_CACHE) >= _LAYOUT_CACHE_LIMIT:
        _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = layout
    return layout


def _build_tile_layout_regions(montage, plan_regions) -> tuple[TileLayoutRegion, ...]:
    if plan_regions is not None:
        regions = []
        for region_id, source_index, bounds in plan_regions:
            x0, y0, x1, y1 = (float(value) for value in bounds)
            width = max(0, round(x1 - x0 + 1.0))
            height = max(0, round(y1 - y0 + 1.0))
            if width < 1 or height < 1:
                continue
            regions.append(
                TileLayoutRegion(
                    tile_number=int(region_id),
                    source_index=None if source_index is None else int(source_index),
                    x=round(x0),
                    y=round(y0),
                    width=width,
                    height=height,
                )
            )
        return tuple(regions)

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


def tile_layout_map(geometry, *, frame_plan=None) -> Mapping[int, TileLayoutRegion]:
    """Tile-keyed placement.

    The returned mapping is shared with the layout cache and is read-only;
    callers that need to mutate placement take their own copy (``dict(...)``).
    """

    return _resolve_tile_layout(geometry, frame_plan).region_map


def tile_layout_shape(geometry, *, frame_plan=None) -> tuple[int, int]:
    return _resolve_tile_layout(geometry, frame_plan).shape


def planned_tile_count(geometry, *, frame_plan=None, minimum: int = 1) -> int:
    return max(int(minimum), _resolve_tile_layout(geometry, frame_plan).planned_count)


__all__ = [
    "TileLayoutRegion",
    "planned_tile_count",
    "tile_layout_map",
    "tile_layout_shape",
]
