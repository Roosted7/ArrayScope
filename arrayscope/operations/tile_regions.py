"""Pure tile-region demand request models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TileRegionPurpose = Literal["visible", "roi", "profile", "prefetch"]
TileRegionSource = Literal["committed_canvas", "tile_cache", "region_cache", "stage_cache", "computed"]


@dataclass(frozen=True)
class TileRegionRequest:
    document_key: object
    view_state: object
    montage_axis: int | None
    source_index: int | None
    tile_number: int | None
    tile_local_region: tuple[slice, slice] | None
    purpose: TileRegionPurpose


@dataclass(frozen=True)
class TileRegionResult:
    request: TileRegionRequest
    image: np.ndarray
    histogram_data: np.ndarray | None
    source: TileRegionSource
