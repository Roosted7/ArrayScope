"""Stateful per-tile montage display items."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from pyqtgraph.graphicsItems.ImageItem import ImageItem

from arrayscope.display.image_upload import rgb_display_for_levels


RGB_SOURCE_CACHE_BUDGET_BYTES = 128 * 1024 * 1024


@dataclass
class TileLayerItemState:
    tile_number: int
    source_index: int
    item: ImageItem
    local_rect: tuple[int, int, int, int]
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
    items_updated: int = 0
    items_skipped: int = 0
    rgb_window_tiles: int = 0


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
        self._source_cache_serial = 0
        self._rgb_source_cache_budget_bytes = RGB_SOURCE_CACHE_BUDGET_BYTES

    @property
    def states(self) -> dict[int, TileLayerItemState]:
        return self._states

    def clear(self) -> None:
        for state in tuple(self._states.values()):
            self.layer_owner.remove_tile_item(state.tile_number)
        self._states.clear()

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
    ) -> TileLayerUpdateStats:
        montage = geometry.montage
        tile_h = int(montage.tile_height)
        tile_w = int(montage.tile_width)
        stride_x = tile_w + int(montage.gap)
        stride_y = tile_h + int(montage.gap)
        active = set()
        states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
        dirty_set = None if dirty_tiles is None else {int(tile) for tile in dirty_tiles}
        image = np.asarray(img)
        hist = None if histogram_data is None else np.asarray(histogram_data)
        levels = (float(levels[0]), float(levels[1]))
        visible_items = 0
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0

        for tile_number, source_index in enumerate(tuple(montage.indices)):
            state = states[tile_number] if tile_number < len(states) else "unloaded"
            state_value = str(getattr(state, "value", state))
            if state_value != "loaded":
                self._hide_tile(tile_number)
                continue

            row = tile_number // int(montage.columns)
            col = tile_number % int(montage.columns)
            global_x = col * stride_x
            global_y = row * stride_y
            local_x = global_x - int(geometry.montage_origin_x)
            local_y = global_y - int(geometry.montage_origin_y)
            source_x0 = max(0, -local_x)
            source_y0 = max(0, -local_y)
            dest_x0 = max(0, local_x)
            dest_y0 = max(0, local_y)
            width = min(tile_w - source_x0, int(image.shape[1]) - dest_x0)
            height = min(tile_h - source_y0, int(image.shape[0]) - dest_y0)
            if width <= 0 or height <= 0:
                self._hide_tile(tile_number)
                continue

            item_state = self._states.get(tile_number)
            if item_state is None:
                item = ImageItem(axisOrder="row-major")
                self.layer_owner.add_tile_item(tile_number, item)
                item_state = TileLayerItemState(
                    tile_number=int(tile_number),
                    source_index=int(source_index),
                    item=item,
                    local_rect=(-1, -1, -1, -1),
                    source_array_id=0,
                    histogram_array_id=None,
                    levels=levels,
                    rgb_already_windowed=bool(rgb_already_windowed),
                    visible=False,
                )
                self._states[int(tile_number)] = item_state

            item_state.item.setVisible(True)
            world_x = global_x + source_x0
            world_y = global_y + source_y0
            item_state.item.setPos(float(world_x), float(world_y))
            visible_items += 1
            active.add(int(tile_number))

            local_rect = (int(dest_x0), int(dest_y0), int(width), int(height))
            source_id = (
                tile_source_ids.get(int(tile_number), id(img))
                if tile_source_ids is not None
                else id(img)
            )
            hist_id = (
                ("tile-source", source_id)
                if tile_source_ids is not None
                else (None if histogram_data is None else id(histogram_data))
            )
            source_changed = (
                item_state.source_array_id != source_id
                or item_state.histogram_array_id != hist_id
                or tuple(item_state.local_rect) != local_rect
                or int(item_state.source_index) != int(source_index)
                or bool(item_state.rgb_already_windowed) != bool(rgb_already_windowed)
            )
            dirty = dirty_set is None or int(tile_number) in dirty_set
            levels_changed = tuple(item_state.levels) != levels
            tile_view = image[dest_y0 : dest_y0 + height, dest_x0 : dest_x0 + width, ...]
            is_rgb_tile = self._is_rgb_image(tile_view)
            missing_display = getattr(item_state.item, "image", None) is None and is_rgb_tile
            needs_source_rewindow = (
                levels_changed
                and is_rgb_tile
                and not bool(rgb_already_windowed)
                and hist is not None
                and (item_state.rgb_base is None or item_state.hist_source is None)
            )
            should_upload = bool(source_changed or dirty or not item_state.visible or missing_display or needs_source_rewindow)

            if should_upload:
                tile_data = tile_view
                tile_hist = None if hist is None else hist[dest_y0 : dest_y0 + height, dest_x0 : dest_x0 + width]
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
                items_updated += int(updated)
                rgb_window_tiles += int(windowed)
            elif levels_changed:
                updated, windowed = self._update_tile_levels(item_state, levels)
                items_updated += int(updated)
                rgb_window_tiles += int(windowed)
                if not updated:
                    items_skipped += 1
            else:
                items_skipped += 1
                item_state.levels = levels
                item_state.visible = True

        for tile_number in tuple(self._states):
            if int(tile_number) not in active:
                self._hide_tile(tile_number)
        self._prune_rgb_source_cache()

        return TileLayerUpdateStats(
            visible_items=int(visible_items),
            items_updated=int(items_updated),
            items_skipped=int(items_skipped),
            rgb_window_tiles=int(rgb_window_tiles),
        )

    def update_levels(self, levels, *, image=None, histogram_data=None) -> TileLayerUpdateStats:
        levels = (float(levels[0]), float(levels[1]))
        image_array = None if image is None else np.asarray(image)
        hist_array = None if histogram_data is None else np.asarray(histogram_data)
        visible_items = 0
        items_updated = 0
        items_skipped = 0
        rgb_window_tiles = 0
        for state in tuple(self._states.values()):
            if not state.visible:
                continue
            visible_items += 1
            updated, windowed = self._update_tile_levels(state, levels, image=image_array, histogram_data=hist_array)
            items_updated += int(updated)
            rgb_window_tiles += int(windowed)
            if not updated:
                items_skipped += 1
            self._prune_rgb_source_cache()
        return TileLayerUpdateStats(visible_items, items_updated, items_skipped, rgb_window_tiles)

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
        states = [state for state in self._states.values() if state.rgb_base is not None or state.hist_source is not None]
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


def _source_cache_nbytes(state: TileLayerItemState) -> int:
    total = 0
    if state.rgb_base is not None:
        total += int(getattr(state.rgb_base, "nbytes", 0) or 0)
    if state.hist_source is not None:
        total += int(getattr(state.hist_source, "nbytes", 0) or 0)
    return total
