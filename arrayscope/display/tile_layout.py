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


def tile_layout_regions(geometry, *, frame_plan=None) -> tuple[TileLayoutRegion, ...]:
    """Return drawable tile placement for either a frame plan or montage geometry."""

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


def tile_layout_map(geometry, *, frame_plan=None) -> dict[int, TileLayoutRegion]:
    return {
        int(region.tile_number): region
        for region in tile_layout_regions(geometry, frame_plan=frame_plan)
    }


def tile_layout_shape(geometry, *, frame_plan=None) -> tuple[int, int]:
    regions = tile_layout_regions(geometry, frame_plan=frame_plan)
    if not regions:
        return tuple(int(value) for value in getattr(geometry, "display_shape", (1, 1))[:2])
    width = max(int(region.x + region.width) for region in regions)
    height = max(int(region.y + region.height) for region in regions)
    return max(1, height), max(1, width)


def planned_tile_count(geometry, *, frame_plan=None, minimum: int = 1) -> int:
    regions = tile_layout_regions(geometry, frame_plan=frame_plan)
    if regions:
        return max(int(minimum), len(regions))
    montage = getattr(geometry, "montage", None)
    if montage is not None:
        return max(int(minimum), len(tuple(getattr(montage, "indices", ()) or ())))
    return max(1, int(minimum))


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
