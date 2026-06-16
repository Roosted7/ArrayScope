"""Qt-free state for progressive montage rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

import numpy as np

from arrayscope.display.montage import (
    MontagePlan,
    MontageTile,
    MontageTileState,
    MontageViewportCanvas,
    RenderedTile,
    patch_rendered_tile_into_canvas,
)


@dataclass
class MontageRenderSession:
    session_id: int
    key: object
    render_generation: int
    level_key: object
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
    force_auto: bool
    visible_tiles: tuple[MontageTile, ...]
    rendered_tiles: dict[int, RenderedTile]
    loading_tiles: set[int]
    skipped_tiles: set[int]
    pending_tiles: list[MontageTile]
    active_tile_requests: set[int] = field(default_factory=set)
    tile_stage_keys: dict[int, object] = field(default_factory=dict)
    stage_waiting_tiles: dict[object, list[MontageTile]] = field(default_factory=dict)
    active_stage_requests: set[object] = field(default_factory=set)
    attached_stage_requests: set[object] = field(default_factory=set)
    stage_values: dict[object, object] = field(default_factory=dict)
    canvas: MontageViewportCanvas | None = None
    canvas_data: np.ndarray | None = None
    canvas_histogram_data: np.ndarray | None = None
    canvas_rect: tuple[int, int, int, int] | None = None
    tile_states: list[MontageTileState] = field(default_factory=list)
    dirty_rects: list[tuple[int, int, int, int]] = field(default_factory=list)
    flush_pending: bool = False
    last_commit_monotonic: float = 0.0
    final_commit_pending: bool = False
    show_loading_overlays: bool = False
    defer_side_panels: bool = False
    display_committed: bool = False
    applied_level_source: object | None = None
    pending_level_tiles: list[RenderedTile] = field(default_factory=list)

    def is_tile_loaded(self, tile) -> bool:
        return int(tile.montage_index) in self.rendered_tiles

    def mark_loaded(self, rendered: RenderedTile) -> None:
        index = int(rendered.tile.montage_index)
        self.rendered_tiles[index] = rendered
        self.active_tile_requests.discard(index)
        self.loading_tiles.discard(index)
        self.skipped_tiles.discard(index)
        self.pending_tiles = [tile for tile in self.pending_tiles if int(tile.montage_index) != index]
        self.mark_tile_state(rendered.tile, MontageTileState.LOADED)

    def mark_loading(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles and index not in self.skipped_tiles:
            self.loading_tiles.add(index)
            self.mark_tile_state(tile, MontageTileState.LOADING)

    def mark_skipped(self, tile: MontageTile) -> None:
        index = int(tile.montage_index)
        if index not in self.rendered_tiles:
            self.active_tile_requests.discard(index)
            self.loading_tiles.discard(index)
            self.skipped_tiles.add(index)
            self.pending_tiles = [pending for pending in self.pending_tiles if int(pending.montage_index) != index]
            self.mark_tile_state(tile, MontageTileState.SKIPPED)

    def next_tile(self) -> MontageTile | None:
        while self.pending_tiles:
            tile = self.pending_tiles.pop(0)
            index = int(tile.montage_index)
            if index not in self.rendered_tiles and index not in self.skipped_tiles:
                self.mark_loading(tile)
                self.active_tile_requests.add(index)
                return tile
        return None

    def rendered_tuple(self) -> tuple[RenderedTile, ...]:
        return tuple(sorted(self.rendered_tiles.values(), key=lambda rendered: rendered.tile.montage_index))

    def loading_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(self.plan.tiles[index] for index in sorted(self.loading_tiles) if 0 <= index < len(self.plan.tiles))

    def skipped_tile_tuple(self) -> tuple[MontageTile, ...]:
        return tuple(self.plan.tiles[index] for index in sorted(self.skipped_tiles) if 0 <= index < len(self.plan.tiles))

    def initialize_canvas(self, canvas: MontageViewportCanvas) -> None:
        self.canvas = canvas
        self.canvas_data = canvas.data
        self.canvas_histogram_data = canvas.histogram_data
        self.canvas_rect = tuple(int(value) for value in canvas.canvas_rect)
        self.tile_states = list(canvas.tile_states)
        self.dirty_rects.clear()

    def has_canvas(self) -> bool:
        return self.canvas is not None

    def current_canvas(self) -> MontageViewportCanvas:
        if self.canvas is None:
            raise RuntimeError("montage session has no canvas")
        if self.tile_states:
            object.__setattr__(self.canvas, "tile_states", tuple(self.tile_states))
        return self.canvas

    def mark_tile_state(self, tile: MontageTile, state: MontageTileState) -> None:
        index = int(tile.montage_index)
        if not self.tile_states:
            self.tile_states = [MontageTileState.UNLOADED for _tile in self.plan.tiles]
        if 0 <= index < len(self.tile_states):
            self.tile_states[index] = MontageTileState(state)
            if self.canvas is not None:
                object.__setattr__(self.canvas, "tile_states", tuple(self.tile_states))

    def patch_rendered_tile(self, rendered: RenderedTile) -> bool:
        if self.canvas is None:
            return False
        dirty = patch_rendered_tile_into_canvas(rendered, self.canvas)
        self.mark_tile_state(rendered.tile, MontageTileState.LOADED)
        if dirty is None:
            return False
        self.dirty_rects.append(tuple(int(value) for value in dirty))
        return True

    def consume_dirty_rects(self) -> tuple[tuple[int, int, int, int], ...]:
        rects = tuple(self.dirty_rects)
        self.dirty_rects.clear()
        return rects

    def note_committed(self) -> None:
        self.last_commit_monotonic = monotonic()
        self.final_commit_pending = False
        self.flush_pending = False
