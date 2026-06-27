"""Region-first payload sources for tiled display materialization."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from arrayscope.display.model.frame import DisplayTilePayload


class DisplayRegionSource(Protocol):
    def read_region(self, region, *, quality: str, deadline_ns: int | None = None) -> DisplayTilePayload: ...


class EagerDisplayRegionSource:
    """Adapt an already materialized DisplayImage to the region-first contract."""

    def __init__(self, display_image, *, source_key) -> None:
        self.display_image = display_image
        self.source_key = source_key
        self.data = np.asarray(display_image.data)
        self.histogram_data = None if display_image.histogram_data is None else np.asarray(display_image.histogram_data)
        self.semantic_data = None if getattr(display_image, "semantic_data", None) is None else np.asarray(display_image.semantic_data)
        self.texture_data = None if getattr(display_image, "texture_data", None) is None else np.asarray(display_image.texture_data)

    def read_region(self, region, *, quality: str, deadline_ns: int | None = None) -> DisplayTilePayload:
        tile_number = int(region.region_id)
        y_slice, x_slice = region.data_slices
        tile_data = self.data[y_slice, x_slice, ...]
        tile_hist = None if self.histogram_data is None else self.histogram_data[y_slice, x_slice]
        tile_semantic = None if self.semantic_data is None else self.semantic_data[y_slice, x_slice, ...]
        tile_texture = None if self.texture_data is None else self.texture_data[y_slice, x_slice, ...]
        return DisplayTilePayload(
            tile_number=tile_number,
            source_index=tile_number,
            image=tile_data,
            histogram_data=tile_hist,
            source_id=(
                self.source_key,
                getattr(region, "materialization_key", None),
                quality,
                deadline_ns is not None,
                tuple(int(value) for value in tile_data.shape),
                str(tile_data.dtype),
                id(tile_data.base if getattr(tile_data, "base", None) is not None else tile_data),
            ),
            texture_data=tile_texture,
            texture_kind=getattr(self.display_image, "texture_kind", None),
            semantic_data=tile_semantic,
            semantic_histogram_data=tile_hist,
            source_shape=tile_data.shape[:2],
            lod=getattr(self.display_image, "lod", None),
            shader_mapping=getattr(self.display_image, "shader_mapping", None),
        )


__all__ = ["DisplayRegionSource", "EagerDisplayRegionSource"]
