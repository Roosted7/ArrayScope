"""Qt-free state for progressive montage rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import monotonic

import numpy as np

from arrayscope.display.lod import LodInfo, select_lod_factor
from arrayscope.display.montage import (
    MontagePlan,
    MontageTile,
    MontageTileState,
    MontageViewportCanvas,
    RenderedTile,
    patch_rendered_tile_into_canvas,
)
from arrayscope.display.shader_mapping import TexturePlaneKind
from arrayscope.window.display_frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState


def _shader_mapping_key(mapping):
    return None if mapping is None else getattr(mapping, "identity_key", mapping)


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
    dirty_tiles: list[int] = field(default_factory=list)
    flush_pending: bool = False
    last_commit_monotonic: float = 0.0
    final_commit_pending: bool = False
    show_loading_overlays: bool = False
    defer_side_panels: bool = False
    display_committed: bool = False
    applied_level_source: object | None = None
    pending_level_tiles: list[RenderedTile] = field(default_factory=list)
    pending_completed_tiles: list[tuple[MontageTile, object]] = field(default_factory=list)
    tile_compute_cache_hits: int = 0
    tile_compute_stage_backed: int = 0
    tile_compute_direct: int = 0
    tile_compute_waiting_for_stage: int = 0
    lead_direct_tiles: int = 0
    stage_backed_tiles_pending: int = 0
    retained_stage_index: int | None = None
    retained_stage_decision: str = ""
    repeated_expensive_stage_per_tile: bool = False
    tile_source_ids: dict[int, object] = field(default_factory=dict)
    display_tile_payloads: dict[int, DisplayTilePayload] = field(default_factory=dict)
    tile_presentation_state: TilePresentationState = field(default_factory=TilePresentationState)
    structure_revision: int = 0
    payload_revision: int = 0
    visibility_revision: int = 0
    level_revision: int = 0
    histogram_revision: int = 0
    viewport_revision: int = 0
    tile_lod_factor: int = 1
    desired_tile_lod_factor: int = 1
    _last_active_tiles: tuple[int, ...] = ()
    _last_planned_tiles: tuple[int, ...] = ()
    _last_near_tiles: tuple[int, ...] = ()

    def is_tile_loaded(self, tile) -> bool:
        return int(tile.montage_index) in self.rendered_tiles

    def mark_loaded(self, rendered: RenderedTile) -> None:
        index = int(rendered.tile.montage_index)
        self.rendered_tiles[index] = rendered
        # A replacement tile must be assigned a fresh semantic source identity
        # and presentation payload on the next commit.  Other loaded tiles keep
        # their cached wrappers across progressive batches.
        self.tile_source_ids.pop(index, None)
        self.display_tile_payloads.pop(index, None)
        self.active_tile_requests.discard(index)
        self.loading_tiles.discard(index)
        self.skipped_tiles.discard(index)
        self.pending_tiles = [tile for tile in self.pending_tiles if int(tile.montage_index) != index]
        self.mark_tile_state(rendered.tile, MontageTileState.LOADED)

    def snapshot_display_tile_payloads(self, source_ids: dict[int, object]) -> dict[int, DisplayTilePayload]:
        """Return immutable-by-convention payload wrappers for loaded tiles.

        Arrays are not copied.  Stable wrappers are retained across progressive
        commits, avoiding repeated dataclass construction and dtype/shape
        normalization for every tile already on screen.
        """

        loaded = {int(index) for index in self.rendered_tiles}
        for stale in tuple(self.display_tile_payloads):
            if int(stale) not in loaded:
                self.display_tile_payloads.pop(int(stale), None)
        lod_factor = self._selected_lod_factor()
        for tile_number, rendered in self.rendered_tiles.items():
            tile_number = int(tile_number)
            base_source_id = source_ids.get(tile_number, ("rendered_tile", tile_number, id(rendered.image)))
            mapping = getattr(rendered, "shader_mapping", None)
            texture_kind = getattr(rendered, "texture_kind", None)
            exact_image = np.asarray(rendered.image)
            exact_histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
            semantic = getattr(rendered, "semantic_data", None)
            semantic = exact_image if semantic is None else np.asarray(semantic)
            lod = self._planned_lod_info(rendered, factor=lod_factor)
            source_id = self._payload_source_id(base_source_id, texture_kind=texture_kind, mapping=mapping, lod=lod)
            previous = self.display_tile_payloads.get(tile_number)
            if (
                previous is not None
                and previous.source_id == source_id
                and previous.image is exact_image
                and previous.histogram_data is exact_histogram
                and _shader_mapping_key(previous.shader_mapping) == _shader_mapping_key(mapping)
            ):
                continue
            texture_data, texture_histogram, lod = self._texture_for_rendered_tile(rendered, factor=lod_factor)
            del texture_histogram
            self.display_tile_payloads[tile_number] = DisplayTilePayload(
                tile_number=tile_number,
                source_index=int(rendered.tile.source_index),
                image=exact_image,
                histogram_data=exact_histogram,
                source_id=source_id,
                texture_data=texture_data,
                texture_kind=texture_kind,
                semantic_data=semantic,
                semantic_histogram_data=exact_histogram,
                source_shape=tuple(int(value) for value in exact_image.shape[:2]),
                lod=lod,
                shader_mapping=mapping,
            )
        return dict(self.display_tile_payloads)

    def seed_display_tile_payloads(self, previous_payloads: dict[int, DisplayTilePayload], source_ids: dict[int, object]) -> None:
        """Reuse materialized payload wrappers from a previous compatible tiled frame."""

        if not previous_payloads or not self.rendered_tiles:
            return
        lod_factor = self._selected_lod_factor()
        by_source = {payload.source_id: payload for payload in dict(previous_payloads).values()}
        for tile_number, rendered in self.rendered_tiles.items():
            tile_number = int(tile_number)
            if tile_number in self.display_tile_payloads:
                continue
            base_source_id = source_ids.get(tile_number, ("rendered_tile", tile_number, id(rendered.image)))
            lod = self._planned_lod_info(rendered, factor=lod_factor)
            source_id = self._payload_source_id(
                base_source_id,
                texture_kind=getattr(rendered, "texture_kind", None),
                mapping=getattr(rendered, "shader_mapping", None),
                lod=lod,
            )
            previous = by_source.get(source_id)
            if previous is None:
                continue
            if int(previous.tile_number) == tile_number and int(previous.source_index) == int(rendered.tile.source_index):
                self.display_tile_payloads[tile_number] = previous
            else:
                self.display_tile_payloads[tile_number] = replace(
                    previous,
                    tile_number=tile_number,
                    source_index=int(rendered.tile.source_index),
                )

    def _payload_source_id(self, base_source_id, *, texture_kind, mapping, lod: LodInfo) -> tuple[object, ...]:
        del mapping
        return (
            base_source_id,
            "texture_kind",
            None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
            # Shader uniforms do not change texture content.  Keeping this
            # compatibility marker avoids invalidating existing source-key
            # parsing while preventing level/LUT changes from re-uploading.
            "shader",
            None,
            "lod",
            int(lod.factor),
            int(lod.level),
            int(lod.gutter),
        )

    def _selected_lod_factor(self) -> int:
        desired = select_lod_factor(
            self.view_range,
            self.viewport_shape,
            self.plan.tile_shape,
            previous_factor=self.desired_tile_lod_factor,
        )
        self.desired_tile_lod_factor = int(desired)
        # CPU pyramid construction used to happen synchronously from
        # snapshot_display_tile_payloads(), which is a UI commit path.  Until a
        # worker/GPU LOD cache can retain adjacent levels, keep the exact texture
        # resident and let hardware filtering handle zoomed-out sampling.
        self.tile_lod_factor = 1
        return 1

    def _planned_lod_info(self, rendered: RenderedTile, *, factor: int) -> LodInfo:
        del factor
        texture_kind = getattr(rendered, "texture_kind", None)
        if texture_kind is not None and not isinstance(texture_kind, TexturePlaneKind):
            texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
        if texture_kind == TexturePlaneKind.COMPLEX_RG32F and getattr(rendered, "semantic_data", None) is not None:
            source = np.asarray(rendered.semantic_data)
        else:
            source = np.asarray(rendered.image)
        source_shape = tuple(int(value) for value in source.shape[:2])
        return LodInfo(level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0)

    def _texture_for_rendered_tile(self, rendered: RenderedTile, *, factor: int | None = None) -> tuple[np.ndarray, np.ndarray | None, LodInfo]:
        del factor
        texture_kind = getattr(rendered, "texture_kind", None)
        if texture_kind is not None and not isinstance(texture_kind, TexturePlaneKind):
            texture_kind = TexturePlaneKind(getattr(texture_kind, "value", texture_kind))
        if texture_kind == TexturePlaneKind.COMPLEX_RG32F and getattr(rendered, "semantic_data", None) is not None:
            source = np.asarray(rendered.semantic_data)
        else:
            source = np.asarray(rendered.image)
        histogram = None if rendered.histogram_data is None else np.asarray(rendered.histogram_data)
        source_shape = tuple(int(value) for value in source.shape[:2])
        lod = LodInfo(level=0, factor=1, source_shape=source_shape, texture_shape=source_shape, gutter=0)
        return source, histogram, lod

    def build_tile_presentation(
        self,
        source_ids: dict[int, object] | None,
        *,
        source_ids_trusted: bool = True,
    ) -> tuple[TilePresentationState, TilePresentationDelta]:
        source_ids = dict(source_ids or {})
        previous_payloads = dict(self.tile_presentation_state.payloads)
        current_payloads = self.snapshot_display_tile_payloads(source_ids)
        current_loaded = set(current_payloads)
        removals = tuple(sorted(int(tile) for tile in previous_payloads if int(tile) not in current_loaded))
        upserts: dict[int, DisplayTilePayload] = {}
        for tile_number, payload in sorted(current_payloads.items()):
            previous = previous_payloads.get(int(tile_number))
            if previous is payload:
                continue
            if (
                previous is not None
                and previous.source_id == payload.source_id
                and previous.image is payload.image
                and previous.histogram_data is payload.histogram_data
            ):
                continue
            upserts[int(tile_number)] = payload

        planned = tuple(
            int(tile.montage_index)
            for tile in tuple(self.visible_tiles)
            if int(tile.montage_index) not in self.skipped_tiles
        )
        active = tuple(int(tile) for tile in planned if int(tile) in current_loaded)
        near_candidates = self.plan.tiles_intersecting(self.view_range, margin_tiles=2)
        near = tuple(
            int(tile.montage_index)
            for tile in near_candidates
            if int(tile.montage_index) not in self.skipped_tiles
        )
        near_source_ids = {int(tile): source_ids[int(tile)] for tile in near if int(tile) in source_ids}

        if upserts or removals:
            self.payload_revision += 1
        if active != self._last_active_tiles or planned != self._last_planned_tiles:
            self.visibility_revision += 1
        if near != self._last_near_tiles:
            self.viewport_revision += 1
        if planned != self._last_planned_tiles:
            self.structure_revision += 1
        self._last_active_tiles = active
        self._last_planned_tiles = planned
        self._last_near_tiles = near

        delta = TilePresentationDelta(
            structure_revision=self.structure_revision,
            payload_revision=self.payload_revision,
            visibility_revision=self.visibility_revision,
            level_revision=self.level_revision,
            histogram_revision=self.histogram_revision,
            viewport_revision=self.viewport_revision,
            upserts=upserts,
            removals=removals,
            active_tiles=active,
            planned_tiles=planned,
            near_tiles=near,
            near_tile_source_ids=near_source_ids,
            force_refresh=not bool(source_ids_trusted),
        )
        state = self.tile_presentation_state.apply_delta(delta)
        self.tile_presentation_state = state
        return state, delta

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

    def ensure_tile_states(self) -> tuple[MontageTileState, ...]:
        states = [MontageTileState.UNLOADED for _tile in self.plan.tiles]
        for index in tuple(self.skipped_tiles):
            index = int(index)
            if 0 <= index < len(states):
                states[index] = MontageTileState.SKIPPED
        for index in tuple(self.loading_tiles):
            index = int(index)
            if 0 <= index < len(states) and states[index] != MontageTileState.SKIPPED:
                states[index] = MontageTileState.LOADING
        for index in tuple(self.rendered_tiles):
            index = int(index)
            if 0 <= index < len(states):
                states[index] = MontageTileState.LOADED
        self.tile_states = states
        if self.canvas is not None:
            object.__setattr__(self.canvas, "tile_states", tuple(self.tile_states))
        return tuple(self.tile_states)

    def is_complete(self) -> bool:
        return not (
            self.pending_tiles
            or self.loading_tiles
            or self.pending_completed_tiles
            or self.active_tile_requests
            or self.active_stage_requests
            or self.attached_stage_requests
            or self.stage_waiting_tiles
            or self.final_commit_pending
            or self.flush_pending
        )

    def initialize_canvas(self, canvas: MontageViewportCanvas) -> None:
        self.canvas = canvas
        self.canvas_data = canvas.data
        self.canvas_histogram_data = canvas.histogram_data
        self.canvas_rect = tuple(int(value) for value in canvas.canvas_rect)
        self.tile_states = list(canvas.tile_states)
        self.dirty_rects.clear()
        self.dirty_tiles.clear()

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
        self.dirty_tiles.append(int(rendered.tile.montage_index))
        return True

    def consume_dirty_rects(self) -> tuple[tuple[int, int, int, int], ...]:
        rects = tuple(self.dirty_rects)
        self.dirty_rects.clear()
        return rects

    def consume_dirty_tiles(self) -> tuple[int, ...]:
        tiles = tuple(dict.fromkeys(int(tile) for tile in self.dirty_tiles))
        self.dirty_tiles.clear()
        return tiles

    def note_committed(self) -> None:
        self.last_commit_monotonic = monotonic()
        self.final_commit_pending = False
        self.flush_pending = False
