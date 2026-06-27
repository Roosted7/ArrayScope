"""Stateful per-tile montage display items."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from pyqtgraph.graphicsItems.ImageItem import ImageItem

from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.tile_layout import tile_layout_map, tile_layout_regions

from arrayscope.display.image_upload import rgb_display_for_levels


RGB_SOURCE_CACHE_BUDGET_BYTES = 128 * 1024 * 1024


@dataclass
class TileLayerItemState:
    tile_number: int
    source_index: int
    item: ImageItem
    local_rect: tuple[int, int, int, int]
    world_rect: tuple[int, int, int, int]
    source_array_id: object
    histogram_array_id: object | None
    levels: tuple[float, float]
    rgb_already_windowed: bool
    visible: bool
    rgb_base: np.ndarray | None = None
    hist_source: np.ndarray | None = None
    display_cache: np.ndarray | None = None
    source_cache_serial: int = 0


@dataclass(frozen=True)
class TileLayerUpdateStats:
    visible_items: int = 0
    presented_tiles: tuple[int, ...] | None = None
    committed_upserts: tuple[int, ...] | None = None
    updated_tiles: tuple[int, ...] = ()
    items_created: int = 0
    items_updated: int = 0
    items_skipped: int = 0
    rgb_window_tiles: int = 0
    image_replacements: int = 0
    existing_items_shown: int = 0
    relocated_tiles: int = 0
    # Backend-neutral diagnostics.  CPU tile layers leave these at zero;
    # GPU-backed implementations fill them so the diagnostics UI can expose
    # residency and upload behaviour instead of treating every tile layer as
    # equivalent.
    resident_items: int = 0
    storage_capacity: int = 0
    storage_rebuilds: int = 0
    storage_evictions: int = 0
    texture_uploads: int = 0
    texture_upload_bytes: int = 0
    texture_prepare_ms: float = 0.0
    texture_submit_ms: float = 0.0
    vertex_uploads: int = 0
    level_updates: int = 0
    estimated_gpu_bytes: int = 0
    cpu_shadow_bytes: int = 0
    page_count: int = 0
    active_pages: int = 0
    device_max_texture_size: int = 0
    budget_bytes: int = 0
    near_resident_items: int = 0
    warm_resident_items: int = 0
    evicted_near_items: int = 0
    capacity_warning: str = ""
    lod_level: int = 0
    lod_factor: int = 1
    source_texels_per_pixel: float = 0.0
    gutter_pixels: int = 0
    mipmap_updates: int = 0
    mipmap_available: bool = False
    complex_texture_uploads: int = 0
    shader_uniform_updates: int = 0
    upload_ms: float = 0.0
    level_update_processed_items: int = 0
    level_update_pending_items: int = 0


class MontageTileLayer:
    def __init__(
        self,
        layer_owner,
        *,
        set_image_item_data: Callable,
        record_upload_timing: Callable[[str, float], None],
        histogram_levels_for_display: Callable,
        is_rgb_image: Callable[[object], bool],
    ):
        self.layer_owner = layer_owner
        self._set_image_item_data = set_image_item_data
        self._record_upload_timing = record_upload_timing
        self._histogram_levels_for_display = histogram_levels_for_display
        self._is_rgb_image = is_rgb_image
        self._states: dict[int, TileLayerItemState] = {}
        self._states_by_source_key: dict[object, TileLayerItemState] = {}
        self._direct_reuse_pool: list[TileLayerItemState] = []
        self._source_cache_serial = 0
        self._rgb_source_cache_budget_bytes = RGB_SOURCE_CACHE_BUDGET_BYTES

    @property
    def states(self) -> dict[int, TileLayerItemState]:
        return self._states

    def set_lookup_table(self, lut) -> None:
        """Apply the frame colormap to every resident scalar tile item."""

        for state in self._states.values():
            image = getattr(state.item, "image", None)
            if image is not None and np.asarray(image).ndim == 2:
                state.item.setLookupTable(lut)

    def clear(self) -> None:
        for state in tuple(self._states.values()):
            self.layer_owner.remove_tile_item(state.tile_number)
        self._states.clear()
        self._states_by_source_key.clear()
        self._direct_reuse_pool.clear()

    def update_presentation(
        self,
        img,
        *,
        histogram_data,
        geometry,
        levels: tuple[float, float],
        rgb_already_windowed: bool,
        dirty_tiles: tuple[int, ...] | None,
        tile_source_ids: dict[int, object] | None = None,
        tile_payloads: dict[int, DisplayTilePayload] | None = None,
        tile_delta=None,
        frame_plan=None,
    ) -> TileLayerUpdateStats:
        if tile_payloads is not None:
            return self._update_direct_payload_presentation(
                tile_payloads,
                geometry=geometry,
                levels=levels,
                rgb_already_windowed=rgb_already_windowed,
                dirty_tiles=dirty_tiles,
                tile_source_ids=tile_source_ids,
                tile_delta=tile_delta,
                frame_plan=frame_plan,
            )
        raise ValueError("PyQtGraph tiled presentation requires typed tile payloads")

    def _update_direct_payload_presentation(
        self,
        tile_payloads: dict[int, DisplayTilePayload],
        *,
        geometry,
        levels: tuple[float, float],
        rgb_already_windowed: bool,
        dirty_tiles: tuple[int, ...] | None,
        tile_source_ids: dict[int, object] | None = None,
        tile_delta=None,
        frame_plan=None,
    ) -> TileLayerUpdateStats:
        layout = tile_layout_map(geometry, frame_plan=frame_plan)
        if not layout:
            return TileLayerUpdateStats()
        requested_active = {
            int(tile)
            for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ())
            if int(tile) in tile_payloads
        } if tile_delta is not None else set()
        active = {
            int(tile)
            for tile in requested_active
            if int(tile) in self._states and bool(getattr(self._states[int(tile)], "visible", False))
        }
        states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
        dirty_set = None if dirty_tiles is None else {int(tile) for tile in dirty_tiles}
        cold_deadline_ms = None if tile_delta is None else getattr(tile_delta, "cold_deadline_ms", None)
        cold_start = perf_counter()
        cold_tiles_committed = 0
        update_start = perf_counter()
        levels = (float(levels[0]), float(levels[1]))
        visible_items = len(active)
        items_created = 0
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0
        image_replacements = 0
        existing_items_shown = 0
        relocated_tiles = 0
        level_updates = 0
        updated_tiles: list[int] = []
        requested_upserts = (
            set(int(tile) for tile in tile_payloads)
            if tile_delta is None
            else set(int(tile) for tile in dict(getattr(tile_delta, "upserts", {}) or {}))
        )
        committed_upserts: set[int] = set()
        self._discard_direct_reuse_pool()

        tile_order = _direct_tile_order(
            layout,
            tile_payloads,
            tile_delta,
            self._states,
            tile_states=states,
            tile_source_ids=tile_source_ids,
            rgb_already_windowed=bool(rgb_already_windowed),
        )
        level_update_pending_items = sum(
            1
            for tile in requested_active
            if int(tile) in self._states and tuple(self._states[int(tile)].levels) != levels
        )
        level_update_tiles = tuple(
            int(tile)
            for tile in requested_upserts
            if int(tile) in self._states and tuple(self._states[int(tile)].levels) != levels
        )
        if level_update_tiles:
            tile_order = tuple(dict.fromkeys(tuple(tile_order) + tuple(sorted(level_update_tiles))))
        # Match the VisPy atlas path's ordering: resolve active payloads to
        # resident identities, bind tile placement to resident storage, then
        # decide whether any data upload is needed.  For PyQtGraph the
        # resident storage is the ImageItem state itself.
        preclaim_specs = _direct_preclaim_specs(
            layout,
            tile_order,
            tile_payloads,
            states=states,
            tile_source_ids=tile_source_ids,
        )
        cold_holes = _direct_cold_hole_count(
            preclaim_specs,
            self._states_by_source_key,
            rgb_already_windowed=bool(rgb_already_windowed),
        )
        # Moving an ImageItem is destructive for its old slot.  Unlike VisPy's
        # coherent atlas remap, a backend-local cold deadline could stop before
        # the displaced slot is replaced.  The unconstrained range-shift path
        # still preclaims all resident items; deadline-capped callbacks keep
        # old pixels visible unless the move can be completed safely.
        allow_resident_reuse = cold_deadline_ms is None or cold_holes <= 1
        if allow_resident_reuse:
            for tile_number, spec in preclaim_specs.items():
                item_state = self._states.get(int(tile_number))
                if _direct_state_matches(
                    item_state,
                    source_id=spec[0],
                    histogram_id=spec[1],
                    local_rect=spec[2],
                    rgb_already_windowed=bool(rgb_already_windowed),
                ):
                    continue
                self._take_resident_direct_state(
                    int(tile_number),
                    source_id=spec[0],
                    histogram_id=spec[1],
                    local_rect=spec[2],
                    rgb_already_windowed=bool(rgb_already_windowed),
                )
        for tile_number in tile_order:
            region = layout.get(int(tile_number))
            if region is None:
                continue
            source_index = int(region.source_index) if region.source_index is not None else int(tile_number)
            state_value = "loaded"
            if states and tile_number < len(states):
                state_value = str(getattr(states[tile_number], "value", states[tile_number]))
            payload = None if state_value == "skipped" else tile_payloads.get(int(tile_number))
            if payload is None:
                self._hide_tile(tile_number)
                continue

            if not isinstance(payload, DisplayTilePayload):
                raise TypeError("typed tile-layer payloads must be DisplayTilePayload instances")
            tile_img = payload.image
            if tile_img is None:
                self._hide_tile(tile_number)
                continue
            tile_data = np.asarray(tile_img)
            if tile_data.ndim < 2:
                self._hide_tile(tile_number)
                continue
            width = min(int(region.width), int(tile_data.shape[1]))
            height = min(int(region.height), int(tile_data.shape[0]))
            if width <= 0 or height <= 0:
                self._hide_tile(tile_number)
                continue
            if width != int(tile_data.shape[1]) or height != int(tile_data.shape[0]):
                tile_data = tile_data[:height, :width, ...]
            tile_hist = None if payload.histogram_data is None else np.asarray(payload.histogram_data)[:height, :width]

            world_x = int(region.x)
            world_y = int(region.y)
            world_rect = (int(world_x), int(world_y), int(width), int(height))
            base_source_id = (
                tile_source_ids.get(int(tile_number), payload.source_id)
                if tile_source_ids is not None
                else payload.source_id
            )
            source_id = _direct_payload_source_id(base_source_id, payload)
            hist_id = ("tile-source", source_id) if tile_hist is not None else None
            local_rect = (0, 0, int(width), int(height))
            item_state = self._states.get(tile_number)
            reused_source = False
            resident_current = _direct_state_matches(
                item_state,
                source_id=source_id,
                histogram_id=hist_id,
                local_rect=local_rect,
                rgb_already_windowed=bool(rgb_already_windowed),
            )
            if allow_resident_reuse and (item_state is None or not resident_current):
                reused = self._take_resident_direct_state(
                    tile_number,
                    source_id=source_id,
                    histogram_id=hist_id,
                    local_rect=local_rect,
                    rgb_already_windowed=bool(rgb_already_windowed),
                )
                if reused is not None:
                    item_state = reused
                    reused_source = True
                    resident_current = True
            existing_item = item_state is not None
            geometry_changed = (
                item_state is None
                or tuple(item_state.local_rect) != local_rect
                or tuple(getattr(item_state, "world_rect", (-1, -1, -1, -1))) != world_rect
            )
            source_changed = (
                item_state is None
                or item_state.source_array_id != source_id
                or item_state.histogram_array_id != hist_id
                or tuple(item_state.local_rect) != local_rect
                or bool(item_state.rgb_already_windowed) != bool(rgb_already_windowed)
            )
            dirty = dirty_set is None or int(tile_number) in dirty_set
            if resident_current:
                dirty = False
            levels_changed = item_state is None or tuple(item_state.levels) != levels
            is_rgb_tile = self._is_rgb_image(tile_data)
            missing_display = item_state is not None and getattr(item_state.item, "image", None) is None and is_rgb_tile
            needs_source_rewindow = (
                item_state is not None
                and levels_changed
                and is_rgb_tile
                and not bool(rgb_already_windowed)
                and tile_hist is not None
                and (item_state.rgb_base is None or item_state.hist_source is None)
            )
            should_upload = bool(
                item_state is None
                or source_changed
                or dirty
                or (not item_state.visible and not resident_current)
                or missing_display
                or needs_source_rewindow
            )
            cold_candidate = bool(
                item_state is None
                or item_state.source_array_id == 0
                or source_changed
                or dirty
                or (not item_state.visible and not resident_current)
                or missing_display
                or needs_source_rewindow
            )
            if (
                cold_deadline_ms is not None
                and cold_candidate
                and cold_tiles_committed > 0
                and (perf_counter() - cold_start) * 1000.0 >= float(cold_deadline_ms)
            ):
                if item_state is not None and item_state.visible:
                    active.add(int(tile_number))
                continue

            created_item = False
            if item_state is None:
                item_state = self._direct_reuse_pool.pop() if self._direct_reuse_pool else None
                if item_state is None:
                    item = ImageItem(axisOrder="row-major")
                    item_state = TileLayerItemState(
                        tile_number=int(tile_number),
                        source_index=int(source_index),
                        item=item,
                        local_rect=(-1, -1, -1, -1),
                        world_rect=(-1, -1, -1, -1),
                        source_array_id=0,
                        histogram_array_id=None,
                        levels=levels,
                        rgb_already_windowed=bool(rgb_already_windowed),
                        visible=False,
                    )
                    created_item = True
                    items_created += 1
                item_state.tile_number = int(tile_number)
                self.layer_owner.add_tile_item(tile_number, item_state.item)
                self._states[int(tile_number)] = item_state

            item_state.item.setVisible(True)
            item_state.item.setPos(float(world_x), float(world_y))
            if int(tile_number) not in active:
                visible_items += 1
            active.add(int(tile_number))

            if should_upload:
                updated, windowed = self._set_tile_data(
                    item_state,
                    tile_data,
                    tile_hist,
                    levels,
                    source_index=int(source_index),
                    source_array_id=source_id,
                    histogram_array_id=hist_id,
                    local_rect=local_rect,
                    rgb_already_windowed=bool(rgb_already_windowed),
                )
                item_state.world_rect = world_rect
                items_updated += int(updated)
                if updated:
                    updated_tiles.append(int(tile_number))
                rgb_window_tiles += int(windowed)
                image_replacements += int(updated and not created_item)
                cold_tiles_committed += int(cold_candidate)
                if updated and int(tile_number) in requested_upserts:
                    committed_upserts.add(int(tile_number))
            elif levels_changed:
                updated, windowed = self._update_tile_levels(item_state, levels)
                item_state.world_rect = world_rect
                level_updates += int(existing_item)
                items_updated += int(updated)
                if updated:
                    updated_tiles.append(int(tile_number))
                rgb_window_tiles += int(windowed)
                if not updated:
                    items_skipped += 1
                if int(tile_number) in requested_upserts:
                    committed_upserts.add(int(tile_number))
            else:
                items_skipped += 1
                item_state.levels = levels
                item_state.visible = True
                item_state.source_index = int(source_index)
                item_state.world_rect = world_rect
                existing_items_shown += 1
                relocated_tiles += int(geometry_changed or reused_source)
                if int(tile_number) in requested_upserts:
                    committed_upserts.add(int(tile_number))

        for tile_number in tuple(self._states):
            if int(tile_number) not in active:
                self._hide_tile(tile_number)
        self._discard_direct_reuse_pool()
        self._prune_rgb_source_cache()

        return TileLayerUpdateStats(
            visible_items=int(visible_items),
            presented_tiles=tuple(int(tile) for tile in sorted(active)),
            committed_upserts=tuple(int(tile) for tile in sorted(committed_upserts)),
            updated_tiles=tuple(int(tile) for tile in updated_tiles),
            items_created=int(items_created),
            items_updated=int(items_updated),
            items_skipped=int(items_skipped),
            rgb_window_tiles=int(rgb_window_tiles),
            image_replacements=int(image_replacements),
            existing_items_shown=int(existing_items_shown),
            relocated_tiles=int(relocated_tiles),
            level_updates=int(level_updates),
            level_update_processed_items=int(level_updates),
            upload_ms=(perf_counter() - update_start) * 1000.0,
            level_update_pending_items=max(0, int(level_update_pending_items) - int(level_updates)),
        )

    def update_levels(
        self,
        levels,
        *,
        image=None,
        histogram_data=None,
    ) -> TileLayerUpdateStats:
        levels = (float(levels[0]), float(levels[1]))
        image_array = None if image is None else np.asarray(image)
        hist_array = None if histogram_data is None else np.asarray(histogram_data)
        visible_items = 0
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0
        update_start = perf_counter()
        processed = 0
        for state in tuple(self._states.values()):
            if not state.visible:
                continue
            visible_items += 1
            updated, windowed = self._update_tile_levels(state, levels, image=image_array, histogram_data=hist_array)
            processed += 1
            items_updated += int(updated)
            rgb_window_tiles += int(windowed)
            if not updated:
                items_skipped += 1
        self._prune_rgb_source_cache()
        return TileLayerUpdateStats(
            visible_items=visible_items,
            presented_tiles=tuple(sorted(int(state.tile_number) for state in self._states.values() if state.visible)),
            items_updated=items_updated,
            items_skipped=items_skipped,
            rgb_window_tiles=rgb_window_tiles,
            level_updates=processed,
            level_update_processed_items=processed,
            upload_ms=(perf_counter() - update_start) * 1000.0,
        )

    def _take_resident_direct_state(
        self,
        tile_number: int,
        *,
        source_id: object,
        histogram_id: object | None,
        local_rect: tuple[int, int, int, int],
        rgb_already_windowed: bool,
    ) -> TileLayerItemState | None:
        tile_number = int(tile_number)
        key = _direct_state_key(
            source_id=source_id,
            histogram_id=histogram_id,
            local_rect=local_rect,
            rgb_already_windowed=rgb_already_windowed,
        )
        state = self._states_by_source_key.get(key)
        if state is None or int(state.tile_number) == tile_number:
            return None
        self._remove_from_direct_reuse_pool(state)
        old_tile = int(state.tile_number)
        was_assigned = self._states.get(old_tile) is state
        existing = self._states.get(tile_number)
        if existing is not None and existing is not state:
            self._states.pop(tile_number, None)
            self.layer_owner.remove_tile_item(tile_number)
            existing.visible = False
            self._direct_reuse_pool.append(existing)
        if was_assigned:
            self._states.pop(old_tile, None)
        self._states[tile_number] = state
        move_item = getattr(self.layer_owner, "move_tile_item", None)
        if was_assigned and callable(move_item):
            move_item(old_tile, tile_number, state.item)
        else:
            self.layer_owner.add_tile_item(tile_number, state.item)
        state.tile_number = tile_number
        return state

    def _hide_tile(self, tile_number: int) -> None:
        state = self._states.get(int(tile_number))
        if state is None:
            return
        state.item.setVisible(False)
        state.visible = False
        state.rgb_base = None
        state.hist_source = None
        state.display_cache = None
        self._remove_tile(tile_number)

    def _remove_tile(self, tile_number: int) -> None:
        state = self._states.pop(int(tile_number), None)
        if state is None:
            return
        self._unregister_source_state(state)
        try:
            self.layer_owner.remove_tile_item(int(tile_number))
        except Exception:
            pass

    def _set_tile_data(
        self,
        state: TileLayerItemState,
        tile_data,
        tile_hist,
        levels: tuple[float, float],
        *,
        source_index: int,
        source_array_id: object,
        histogram_array_id: object | None,
        local_rect: tuple[int, int, int, int],
        rgb_already_windowed: bool,
    ) -> tuple[bool, bool]:
        self._unregister_source_state(state)
        is_rgb = self._is_rgb_image(tile_data)
        windowed = False
        if is_rgb:
            if rgb_already_windowed or tile_hist is None:
                state.rgb_base = None
                state.hist_source = None
                display = np.asarray(tile_data)[..., :3]
            else:
                rgb_start = perf_counter()
                base = np.asarray(tile_data)[..., :3].astype(np.float32, copy=False)
                hist = np.asarray(tile_hist, dtype=np.float32)
                display = rgb_display_for_levels(base, hist, levels)
                rgb_ms = (perf_counter() - rgb_start) * 1000.0
                self._record_upload_timing("rgb_window_ms", rgb_ms)
                self._record_upload_timing("tile_layer_rgb_window_ms", rgb_ms)
                state.rgb_base = base
                state.hist_source = hist
                self._touch_rgb_source_cache(state)
                windowed = True
            state.display_cache = display
            upload_start = perf_counter()
            self._set_image_item_data(state.item, display, (0, 255), role="visible", emit_histogram_change=False)
            self._record_upload_timing("tile_layer_upload_ms", (perf_counter() - upload_start) * 1000.0)
        else:
            state.rgb_base = None
            state.hist_source = None
            state.display_cache = None
            upload_start = perf_counter()
            self._set_image_item_data(
                state.item,
                tile_data,
                self._histogram_levels_for_display(levels),
                role="visible",
                emit_histogram_change=False,
            )
            self._record_upload_timing("tile_layer_upload_ms", (perf_counter() - upload_start) * 1000.0)
        state.source_index = int(source_index)
        state.source_array_id = source_array_id
        state.histogram_array_id = histogram_array_id
        state.local_rect = tuple(int(value) for value in local_rect)
        state.levels = levels
        state.rgb_already_windowed = bool(rgb_already_windowed)
        state.visible = True
        self._register_source_state(state)
        return True, windowed

    def _update_tile_levels(
        self,
        state: TileLayerItemState,
        levels: tuple[float, float],
        *,
        image=None,
        histogram_data=None,
    ) -> tuple[bool, bool]:
        if state.rgb_base is None and state.hist_source is None and state.display_cache is not None:
            rebuilt = self._set_tile_data_from_current_source(state, levels, image=image, histogram_data=histogram_data)
            if rebuilt is not None:
                return rebuilt
        if state.rgb_base is not None and state.hist_source is not None:
            rgb_start = perf_counter()
            display = rgb_display_for_levels(state.rgb_base, state.hist_source, levels)
            rgb_ms = (perf_counter() - rgb_start) * 1000.0
            self._record_upload_timing("rgb_window_ms", rgb_ms)
            self._record_upload_timing("tile_layer_rgb_window_ms", rgb_ms)
            state.display_cache = display
            self._touch_rgb_source_cache(state)
            state.levels = levels
            upload_start = perf_counter()
            self._set_image_item_data(state.item, display, (0, 255), role="visible", emit_histogram_change=False)
            self._record_upload_timing("tile_layer_upload_ms", (perf_counter() - upload_start) * 1000.0)
            return True, True
        if state.display_cache is not None:
            if state.rgb_already_windowed:
                state.item.setLevels((0, 255))
                state.levels = levels
            return False, False
        state.item.setLevels(self._histogram_levels_for_display(levels))
        state.levels = levels
        return False, False

    def _set_tile_data_from_current_source(
        self,
        state: TileLayerItemState,
        levels: tuple[float, float],
        *,
        image,
        histogram_data,
    ) -> tuple[bool, bool] | None:
        if state.rgb_already_windowed or image is None or histogram_data is None:
            return None
        x, y, width, height = tuple(int(value) for value in state.local_rect)
        if width <= 0 or height <= 0:
            return None
        image_array = np.asarray(image)
        hist_array = np.asarray(histogram_data)
        if y < 0 or x < 0 or y + height > image_array.shape[0] or x + width > image_array.shape[1]:
            return None
        if y + height > hist_array.shape[0] or x + width > hist_array.shape[1]:
            return None
        tile_data = image_array[y : y + height, x : x + width, ...]
        if not self._is_rgb_image(tile_data):
            return None
        tile_hist = hist_array[y : y + height, x : x + width]
        return self._set_tile_data(
            state,
            tile_data,
            tile_hist,
            levels,
            source_index=state.source_index,
            source_array_id=state.source_array_id,
            histogram_array_id=state.histogram_array_id,
            local_rect=state.local_rect,
            rgb_already_windowed=False,
        )

    def _touch_rgb_source_cache(self, state: TileLayerItemState) -> None:
        if state.rgb_base is None and state.hist_source is None:
            state.source_cache_serial = 0
            return
        self._source_cache_serial += 1
        state.source_cache_serial = int(self._source_cache_serial)

    def _prune_rgb_source_cache(self) -> None:
        budget = int(self._rgb_source_cache_budget_bytes)
        states = [
            state
            for state in self._states.values()
            if not bool(state.visible) and (state.rgb_base is not None or state.hist_source is not None)
        ]
        if budget <= 0:
            for state in states:
                state.rgb_base = None
                state.hist_source = None
                state.source_cache_serial = 0
            return
        total = sum(_source_cache_nbytes(state) for state in states)
        if total <= budget:
            return
        for state in sorted(states, key=lambda item: int(item.source_cache_serial)):
            if total <= budget:
                break
            total -= _source_cache_nbytes(state)
            state.rgb_base = None
            state.hist_source = None
            state.source_cache_serial = 0

    def _register_source_state(self, state: TileLayerItemState) -> None:
        key = _source_key_for_state(state)
        if key is not None:
            self._states_by_source_key[key] = state

    def _unregister_source_state(self, state: TileLayerItemState) -> None:
        key = _source_key_for_state(state)
        if key is not None and self._states_by_source_key.get(key) is state:
            self._states_by_source_key.pop(key, None)

    def _remove_from_direct_reuse_pool(self, state: TileLayerItemState) -> None:
        try:
            self._direct_reuse_pool.remove(state)
        except ValueError:
            pass

    def _discard_direct_reuse_pool(self) -> None:
        for state in tuple(self._direct_reuse_pool):
            self._unregister_source_state(state)
            state.visible = False
            state.rgb_base = None
            state.hist_source = None
            state.display_cache = None
        self._direct_reuse_pool.clear()


def _source_cache_nbytes(state: TileLayerItemState) -> int:
    total = 0
    if state.rgb_base is not None:
        total += int(getattr(state.rgb_base, "nbytes", 0) or 0)
    if state.hist_source is not None:
        total += int(getattr(state.hist_source, "nbytes", 0) or 0)
    return total


def _direct_state_matches(
    state: TileLayerItemState | None,
    *,
    source_id: object,
    histogram_id: object | None,
    local_rect: tuple[int, int, int, int],
    rgb_already_windowed: bool,
) -> bool:
    if state is None:
        return False
    return (
        state.source_array_id == source_id
        and state.histogram_array_id == histogram_id
        and tuple(state.local_rect) == tuple(int(value) for value in local_rect)
        and bool(state.rgb_already_windowed) == bool(rgb_already_windowed)
    )


def _source_key_for_state(state: TileLayerItemState) -> object | None:
    if state.source_array_id == 0:
        return None
    return _direct_state_key(
        source_id=state.source_array_id,
        histogram_id=state.histogram_array_id,
        local_rect=state.local_rect,
        rgb_already_windowed=state.rgb_already_windowed,
    )


def _direct_state_key(
    *,
    source_id: object,
    histogram_id: object | None,
    local_rect: tuple[int, int, int, int],
    rgb_already_windowed: bool,
) -> tuple[object, object | None, tuple[int, int, int, int], bool]:
    return (
        source_id,
        histogram_id,
        tuple(int(value) for value in local_rect),
        bool(rgb_already_windowed),
    )


def _direct_payload_source_id(base_source_id: object, payload: DisplayTilePayload) -> tuple[object, ...]:
    image = np.asarray(payload.image)
    histogram = None if payload.histogram_data is None else np.asarray(payload.histogram_data)
    texture_kind = getattr(payload, "texture_kind", None)
    return (
        base_source_id,
        "pyqtgraph_display",
        tuple(int(value) for value in image.shape),
        str(image.dtype),
        id(image),
        None if histogram is None else tuple(int(value) for value in histogram.shape),
        None if histogram is None else str(histogram.dtype),
        None if histogram is None else id(histogram),
        "texture_kind",
        None if texture_kind is None else getattr(texture_kind, "value", texture_kind),
    )


def _direct_tile_order(
    layout: dict[int, object],
    tile_payloads: dict[int, DisplayTilePayload],
    tile_delta,
    state_map: dict[int, TileLayerItemState] | None = None,
    *,
    tile_states: tuple[object, ...] = (),
    tile_source_ids: dict[int, object] | None = None,
    rgb_already_windowed: bool = False,
) -> tuple[int, ...]:
    if tile_delta is None:
        return tuple(sorted(int(tile) for tile in layout))
    candidates: list[int] = []
    candidates.extend(int(tile) for tile in tuple(getattr(tile_delta, "removals", ()) or ()))
    candidates.extend(int(tile) for tile in tuple(getattr(tile_delta, "upserts", ()) or ()))
    active_tiles = tuple(int(tile) for tile in tuple(getattr(tile_delta, "active_tiles", ()) or ()))
    if bool(getattr(tile_delta, "force_refresh", False)):
        candidates.extend(active_tiles)
    else:
        state_map = state_map or {}
        candidates.extend(
            int(tile)
            for tile in active_tiles
            if int(tile) in tile_payloads
            and (
                int(tile) not in state_map
                or not bool(getattr(state_map[int(tile)], "visible", False))
                or _direct_tile_geometry_changed(state_map[int(tile)], layout, int(tile), tile_payloads[int(tile)])
                or _direct_tile_binding_stale(
                    state_map.get(int(tile)),
                    layout,
                    int(tile),
                    tile_payloads[int(tile)],
                    tile_states=tile_states,
                    tile_source_ids=tile_source_ids,
                    rgb_already_windowed=bool(rgb_already_windowed),
                )
            )
        )
    seen: set[int] = set()
    ordered: list[int] = []
    for tile in candidates:
        if int(tile) < 0 or int(tile) in seen:
            continue
        seen.add(int(tile))
        ordered.append(int(tile))
    return tuple(ordered)


def _direct_tile_binding_stale(
    state: TileLayerItemState | None,
    layout: dict[int, object],
    tile_number: int,
    payload: DisplayTilePayload,
    *,
    tile_states: tuple[object, ...],
    tile_source_ids: dict[int, object] | None,
    rgb_already_windowed: bool,
) -> bool:
    tile_number = int(tile_number)
    state_value = "loaded"
    if tile_states and tile_number < len(tile_states):
        state_value = str(getattr(tile_states[tile_number], "value", tile_states[tile_number]))
    if state_value == "skipped":
        return False
    region = layout.get(tile_number)
    if region is None:
        return False
    tile_data = np.asarray(payload.image)
    if tile_data.ndim < 2:
        return False
    width = min(int(region.width), int(tile_data.shape[1]))
    height = min(int(region.height), int(tile_data.shape[0]))
    if width <= 0 or height <= 0:
        return False
    base_source_id = (
        tile_source_ids.get(tile_number, payload.source_id)
        if tile_source_ids is not None
        else payload.source_id
    )
    source_id = _direct_payload_source_id(base_source_id, payload)
    histogram_id = ("tile-source", source_id) if payload.histogram_data is not None else None
    return not _direct_state_matches(
        state,
        source_id=source_id,
        histogram_id=histogram_id,
        local_rect=(0, 0, int(width), int(height)),
        rgb_already_windowed=bool(rgb_already_windowed),
    )


def _direct_preclaim_specs(
    layout: dict[int, object],
    tile_order: tuple[int, ...],
    tile_payloads: dict[int, DisplayTilePayload],
    *,
    states: tuple[object, ...],
    tile_source_ids: dict[int, object] | None,
) -> dict[int, tuple[object, object | None, tuple[int, int, int, int]]]:
    specs: dict[int, tuple[object, object | None, tuple[int, int, int, int]]] = {}
    for tile_number in tile_order:
        tile_number = int(tile_number)
        region = layout.get(tile_number)
        if region is None:
            continue
        state_value = "loaded"
        if states and tile_number < len(states):
            state_value = str(getattr(states[tile_number], "value", states[tile_number]))
        payload = None if state_value == "skipped" else tile_payloads.get(tile_number)
        if not isinstance(payload, DisplayTilePayload):
            continue
        tile_data = np.asarray(payload.image)
        if tile_data.ndim < 2:
            continue
        width = min(int(region.width), int(tile_data.shape[1]))
        height = min(int(region.height), int(tile_data.shape[0]))
        if width <= 0 or height <= 0:
            continue
        base_source_id = (
            tile_source_ids.get(tile_number, payload.source_id)
            if tile_source_ids is not None
            else payload.source_id
        )
        source_id = _direct_payload_source_id(base_source_id, payload)
        histogram_id = ("tile-source", source_id) if payload.histogram_data is not None else None
        specs[tile_number] = (
            source_id,
            histogram_id,
            (0, 0, int(width), int(height)),
        )
    return specs


def _direct_cold_hole_count(
    specs: dict[int, tuple[object, object | None, tuple[int, int, int, int]]],
    states_by_source_key: dict[object, TileLayerItemState],
    *,
    rgb_already_windowed: bool,
) -> int:
    cold = 0
    for source_id, histogram_id, local_rect in specs.values():
        key = _direct_state_key(
            source_id=source_id,
            histogram_id=histogram_id,
            local_rect=local_rect,
            rgb_already_windowed=bool(rgb_already_windowed),
        )
        if key not in states_by_source_key:
            cold += 1
    return cold


def _direct_tile_geometry_changed(state: TileLayerItemState, layout: dict[int, object], tile_number: int, payload: DisplayTilePayload) -> bool:
    data = np.asarray(payload.image)
    if data.ndim < 2:
        return False
    region = layout.get(int(tile_number))
    if region is None:
        return False
    width = min(int(region.width), int(data.shape[1]))
    height = min(int(region.height), int(data.shape[0]))
    if width <= 0 or height <= 0:
        return False
    expected_world = (int(region.x), int(region.y), int(width), int(height))
    expected_local = (0, 0, int(width), int(height))
    return (
        tuple(getattr(state, "local_rect", (-1, -1, -1, -1))) != expected_local
        or tuple(getattr(state, "world_rect", (-1, -1, -1, -1))) != expected_world
    )
