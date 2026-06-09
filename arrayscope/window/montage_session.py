"""Qt-free state for progressive montage rendering."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from arrayscope.display.montage import MontagePlan, MontageTile, MontageViewportCanvas, RenderedTile


@dataclass
class MontageRenderSession:
    session_id: int
    key: object
    plan: MontagePlan
    view_state: object
    document: object
    montage_axis: int
    colormap_lut: object | None
    viewport_shape: tuple[int, int]
    view_range: object
    output_dtype: np.dtype
    rgb: bool
    window_mode: object
    previous_levels: object
    previous_bounds: object
    force_auto: bool
    visible_tiles: tuple[MontageTile, ...]
    rendered_tiles: dict[int, RenderedTile]
    loading_tiles: set[int]
    skipped_tiles: set[int]
    pending_tiles: list[MontageTile]
    canvas: MontageViewportCanvas | None = None
    last_commit_monotonic: float = 0.0
    final_commit_pending: bool = False
    show_loading_overlays: bool = False

    def is_tile_loaded(self, tile) -> bool:
        return int(tile.montage_index) in self.rendered_tiles

    def mark_loaded(self, rendered: RenderedTile) -> None:
        index = int(rendered.tile.montage_index)
        self.rendered_tiles[index] = rendered
        self.loading_tiles.discard(index)
        self.skipped_tiles.discard(index)
        self.pending_tiles = [tile for tile in self.pending_tiles if int(tile.montage_index) != index]

    def mark_loading(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles and index not in self.skipped_tiles:
            self.loading_tiles.add(index)

    def mark_skipped(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles:
            self.loading_tiles.discard(index)
            self.skipped_tiles.add(index)
            self.pending_tiles = [pending for pending in self.pending_tiles if int(pending.montage_index) != index]

    def next_tile(self) -> MontageTile | None:
        while self.pending_tiles:
            tile = self.pending_tiles.pop(0)
            index = int(tile.montage_index)
            if index not in self.rendered_tiles and index not in self.skipped_tiles:
                self.mark_loading(tile)
                return tile
        return None

    def rendered_tuple(self) -> tuple[RenderedTile, ...]:
        return tuple(sorted(self.rendered_tiles.values(), key=lambda rendered: rendered.tile.montage_index))

    def loading_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(self.plan.tiles[index] for index in sorted(self.loading_tiles) if 0 <= index < len(self.plan.tiles))

    def skipped_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(self.plan.tiles[index] for index in sorted(self.skipped_tiles) if 0 <= index < len(self.plan.tiles))

    def note_committed(self) -> None:
        self.last_commit_monotonic = monotonic()
        self.final_commit_pending = False
