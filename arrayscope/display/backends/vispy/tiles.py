"""Batched VisPy montage tile rendering.

The renderer keeps tile identity, GPU residency, and draw visibility separate.
Presentation commits may change the set of drawn tiles without invalidating
unchanged resident texture slots.  Level-only changes update uniforms only.
"""

from __future__ import annotations

import os

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.display.lod import inner_uv_for_gutter
from arrayscope.display.shader_mapping import (
    ShaderDisplayMode,
    ShaderScale,
    TexturePlaneKind,
    normalize_lut_rgb,
    shader_component_uniform,
)
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.model.tile_stats import TileLayerUpdateStats
from arrayscope.display.tile_layout import planned_tile_count, tile_layout_map

try:
    from vispy.visuals import Visual
except Exception:  # pragma: no cover - optional dependency import path
    Visual = object


@dataclass(frozen=True)
class GpuDeviceLimits:
    max_texture_size: int = 4096
    max_texture_image_units: int = 0
    vendor: str = ""
    renderer: str = ""
    version: str = ""
    source: str = "fallback"
    warnings: tuple[str, ...] = ()


class AtlasCapacityError(RuntimeError):
    """Raised when one atlas page cannot contain the requested tile set."""


_ATLAS_GROWTH_TARGET_BYTES = 32 * 1024 * 1024
_UNSET = object()

# Raw-GL constants for atlas mipmaps (gloo has no mipmap support; the visual
# generates them itself with the context current, see _refresh_mipmaps).
#
# DISABLED BY DEFAULT (field regression 2026-07-04): whole-atlas regen cannot
# keep coarse mips coherent while tiles stream — between regens, minified
# tiles sample stale mip content, including the PREVIOUS occupant of a reused
# atlas slot (wrong slice / wrong window on screen), and any regen throttling
# trades that staleness window against per-frame full-atlas GPU regens.  A
# correct implementation needs per-slot mip invalidation (render-to-mip of
# the dirty region only) — revisit with the backend/presentation rework.
_GL_TEXTURE_MAX_LEVEL = 0x813D
_GL_NEAREST_MIPMAP_LINEAR = 0x2702
_ATLAS_MIPMAP_LEVEL_CAP = 5
_ATLAS_MIPMAPS_ENABLED = os.environ.get("ARRAYSCOPE_ATLAS_MIPMAPS", "") == "1"


def _atlas_mipmap_levels(tile_shape: tuple[int, int]) -> int:
    """Bleed-free mipmap depth for an atlas of contiguous tiles.

    Mip level *k* averages 2^k x 2^k native blocks; a block never spans two
    tiles as long as both tile edges divide by 2^k, so the usable depth is
    the smaller power-of-two factor of the tile edges (capped: deeper mips
    add nothing at montage scales).  Zero for odd tile edges = no mipmaps.
    """

    def _trailing(value: int) -> int:
        value = int(value)
        if value <= 0:
            return 0
        return (value & -value).bit_length() - 1

    return min(_trailing(tile_shape[0]), _trailing(tile_shape[1]), _ATLAS_MIPMAP_LEVEL_CAP)


class TextureAtlasPage:
    def __init__(self, gloo, *, tile_shape: tuple[int, int], capacity: int, storage_mode: str, max_texture_size: int):
        self._gloo = gloo
        self.tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        self.capacity = max(1, int(capacity))
        self.storage_mode = _normalize_storage_mode(storage_mode)
        self.columns, self.rows = _atlas_grid(
            tile_shape=self.tile_shape,
            capacity=self.capacity,
            max_texture_size=max_texture_size,
        )
        tile_h, tile_w = self.tile_shape
        self.atlas_shape = (int(self.rows * tile_h), int(self.columns * tile_w))
        self.scalar_is_atlas = self.storage_mode in {"scalar", "scalar_color", "complex"}
        self.complex_is_atlas = self.storage_mode == "complex"
        self.color_is_atlas = self.storage_mode in {"color", "scalar_color"}
        if self.scalar_is_atlas:
            channels = 2 if self.complex_is_atlas else 1
            self.scalar_texture = self._gloo.Texture2D(
                shape=self.atlas_shape + (channels,),
                format="rg" if self.complex_is_atlas else "red",
                internalformat="rg32f" if self.complex_is_atlas else "r32f",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        else:
            self.scalar_texture = self._gloo.Texture2D(
                np.ones((1, 1), dtype=np.float32),
                format="red",
                internalformat="r32f",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        if self.color_is_atlas:
            self.color_texture = self._gloo.Texture2D(
                shape=self.atlas_shape + (3,),
                format="rgb",
                internalformat="rgb8",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        else:
            self.color_texture = self._gloo.Texture2D(
                np.zeros((1, 1, 3), dtype=np.uint8),
                format="rgb",
                internalformat="rgb8",
                interpolation="nearest",
                wrapping="clamp_to_edge",
            )
        self.slot_owners: list[object | None] = [None] * self.capacity
        # Allocate slots from an explicit stack instead of scanning the whole
        # owner list for every cold tile.  Large atlas pages may contain
        # hundreds or thousands of slots, and visible commits can allocate many
        # tiles in one UI callback.
        self._free_slots: list[int] = list(range(self.capacity - 1, -1, -1))
        # Atlas mipmaps (ADR 0050): regenerated GPU-side by the visual after
        # uploads, sampled NEAREST within a mip (no cross-tile bleed) and
        # LINEAR between mips (smooth minification between CPU levels).
        self.mipmap_levels = (
            _atlas_mipmap_levels(self.tile_shape)
            if _ATLAS_MIPMAPS_ENABLED and (self.scalar_is_atlas or self.color_is_atlas)
            else 0
        )
        self.mipmap_dirty = bool(self.mipmap_levels)
        self.mipmap_ready = False
        self.mipmap_updates = 0

    def take_free_slot(self, owner: object) -> int | None:
        while self._free_slots:
            slot = int(self._free_slots.pop())
            if self.slot_owners[slot] is None:
                self.slot_owners[slot] = owner
                return slot
        return None

    @property
    def estimated_gpu_bytes(self) -> int:
        pixels = int(self.atlas_shape[0]) * int(self.atlas_shape[1])
        scalar_bytes = 8 if self.complex_is_atlas else 4
        return pixels * (scalar_bytes if self.scalar_is_atlas else 0) + pixels * (3 if self.color_is_atlas else 0)

    def uv_for_slot(self, slot: int) -> tuple[float, float, float, float]:
        tile_h, tile_w = self.tile_shape
        atlas_h, atlas_w = self.atlas_shape
        row = int(slot) // int(self.columns)
        col = int(slot) % int(self.columns)
        y0 = row * tile_h
        x0 = col * tile_w
        y1 = y0 + tile_h
        x1 = x0 + tile_w
        uv = (x0 / atlas_w, y0 / atlas_h, x1 / atlas_w, y1 / atlas_h)
        gutter = 0
        return uv

    def uv_for_slot_with_gutter(self, slot: int, gutter: int = 0) -> tuple[float, float, float, float]:
        u0, v0, u1, v1 = self.uv_for_slot(slot)
        gutter = max(0, int(gutter))
        if gutter == 0:
            return u0, v0, u1, v1
        tile_h, tile_w = self.tile_shape
        local = inner_uv_for_gutter((tile_h, tile_w), gutter=gutter)
        du = u1 - u0
        dv = v1 - v0
        return (
            u0 + local[0] * du,
            v0 + local[1] * dv,
            u0 + local[2] * du,
            v0 + local[3] * dv,
        )

    def offset_for_slot(self, slot: int) -> tuple[int, int]:
        tile_h, tile_w = self.tile_shape
        row = int(slot) // int(self.columns)
        col = int(slot) % int(self.columns)
        return int(row * tile_h), int(col * tile_w)


class TextureAtlasPool:
    """Multi-page texture atlas with stable, LRU-managed tile residency.

    ``source_id`` is treated as an immutable content identity.  A clean commit
    can therefore reuse a resident slot even when it is the first commit of a
    new viewport session.  Callers must change the source id or mark a tile
    dirty when its pixels change.
    """

    def __init__(self, gloo, *, limits: GpuDeviceLimits | None = None, max_texture_size: int | None = None, budget_bytes: int = 0):
        self._gloo = gloo
        self.device_limits = limits or GpuDeviceLimits(max_texture_size=max_texture_size or 4096)
        self.max_texture_size = max(1, int(max_texture_size or self.device_limits.max_texture_size or 4096))
        self.budget_bytes = max(0, int(budget_bytes))
        self.tile_shape: tuple[int, int] | None = None
        self.storage_mode: str | None = None
        self.pages: list[TextureAtlasPage] = []
        self.resident_slots: dict[object, tuple[int, int]] = {}
        self.tile_slots: dict[int, tuple[int, int]] = {}
        self.tile_resident_keys: dict[int, object] = {}
        self.resident_tiles: dict[object, set[int]] = {}
        self.tile_uvs: dict[int, tuple[float, float, float, float]] = {}
        self.source_ids: dict[object, object] = {}
        self.last_used: dict[object, int] = {}
        self.active_resident_keys: set[object] = set()
        # Keys whose tile(s) now present a different residency class (ADR
        # 0050): the acknowledged replacement makes these slots reclaimable.
        self.superseded_keys: set[object] = set()
        self.serial = 0
        self.rebuild_count = 0
        self.eviction_count = 0
        self.evicted_near_count = 0
        self.superseded_reclaimed_count = 0
        self.pages_dropped_count = 0
        # ADR 0050 zero-upload zoom cycles: a level flip between two
        # already-resident classes must be an identity swap.  These counters
        # make that observable in diagnostics and the profile JSONL.
        self.lod_level_swaps_zero_upload = 0
        self.lod_level_swaps_with_upload = 0
        # Atlas mipmap regenerations already reported through update stats.
        self._mipmap_updates_reported = 0
        # Base (LOD-invariant) source identities of the active payload set:
        # a superseded key whose base is active is the retained adjacent
        # level of a visible tile and is reclaimed only as a last resort.
        self.active_base_source_ids: set[object] = set()
        self._clock = 0

    @property
    def resident_count(self) -> int:
        return len(self.source_ids)

    @property
    def capacity(self) -> int:
        return sum(int(page.capacity) for page in self.pages)

    @property
    def slots(self) -> dict[int, int]:
        return {tile: page_index * 1_000_000 + slot for tile, (page_index, slot) in self.tile_slots.items()}

    @property
    def slot_owners(self) -> list[object | None]:
        owners: list[object | None] = []
        for page in self.pages:
            owners.extend(page.slot_owners)
        return owners

    @property
    def atlas_shape(self) -> tuple[int, int]:
        return self.pages[0].atlas_shape if self.pages else (1, 1)

    @property
    def scalar_texture(self):
        return self.pages[0].scalar_texture if self.pages else None

    @property
    def color_texture(self):
        return self.pages[0].color_texture if self.pages else None

    @property
    def scalar_is_atlas(self) -> bool:
        return bool(self.pages and self.pages[0].scalar_is_atlas)

    @property
    def color_is_atlas(self) -> bool:
        return bool(self.pages and self.pages[0].color_is_atlas)

    @property
    def cpu_shadow_bytes(self) -> int:
        # Atlas storage is allocated by shape on the GPU.  Only per-tile
        # staging arrays exist during a sub-upload.
        return 0

    @property
    def estimated_gpu_bytes(self) -> int:
        return sum(int(page.estimated_gpu_bytes) for page in self.pages)

    def ensure_layout(
        self,
        *,
        tile_shape: tuple[int, int],
        count: int,
        storage_mode: str = "scalar_color",
        budget_bytes: int | None = None,
    ) -> bool:
        tile_h, tile_w = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        requested = max(1, int(count))
        if budget_bytes is not None:
            self.budget_bytes = max(0, int(budget_bytes))
        max_columns = self.max_texture_size // tile_w
        max_rows = self.max_texture_size // tile_h
        if max_columns < 1 or max_rows < 1:
            raise AtlasCapacityError(
                f"tile shape {(tile_h, tile_w)} exceeds atlas texture limit {self.max_texture_size}"
            )
        max_slots_per_page = int(max_columns * max_rows)
        bytes_per_slot = _storage_mode_bytes_per_pixel(storage_mode) * tile_h * tile_w
        budget_slots = requested if self.budget_bytes <= 0 else self.budget_bytes // max(1, bytes_per_slot)
        if requested > budget_slots:
            raise AtlasCapacityError(
                f"{requested} active tiles of shape {(tile_h, tile_w)} exceed tile residency budget "
                f"{self.budget_bytes} bytes (capacity {budget_slots})"
            )

        storage_mode = _normalize_storage_mode(storage_mode)
        shape_changed = self.tile_shape != (tile_h, tile_w)
        mode_changed = self.storage_mode != storage_mode
        if not shape_changed and not mode_changed and requested <= self._class_capacity((tile_h, tile_w)):
            return False

        self.tile_shape = (tile_h, tile_w)
        if shape_changed or mode_changed:
            self.pages.clear()
            self.resident_slots.clear()
            self.tile_slots.clear()
            self.tile_resident_keys.clear()
            self.resident_tiles.clear()
            self.tile_uvs.clear()
            self.source_ids.clear()
            self.last_used.clear()
            self.active_resident_keys.clear()
            self.superseded_keys.clear()
        self.storage_mode = storage_mode
        while self._class_capacity((tile_h, tile_w)) < requested:
            remaining = requested - self._class_capacity((tile_h, tile_w))
            page_capacity = min(max_slots_per_page, remaining)
            self.pages.append(
                TextureAtlasPage(
                    self._gloo,
                    tile_shape=(tile_h, tile_w),
                    capacity=page_capacity,
                    storage_mode=storage_mode,
                    max_texture_size=self.max_texture_size,
                )
            )
        self.serial += 1
        self.rebuild_count += 1
        return True

    def _class_capacity(self, tile_shape: tuple[int, int]) -> int:
        """Slot count of the pages whose slot shape matches ``tile_shape``.

        Pages are classed by texture shape (ADR 0050): a reduced-level tile
        must never occupy a native-shaped slot, so capacity questions are
        answered per shape class.
        """

        shape = (int(tile_shape[0]), int(tile_shape[1]))
        return sum(int(page.capacity) for page in self.pages if page.tile_shape == shape)

    def _ensure_class_capacity(self, tile_shape: tuple[int, int], requested: int) -> int:
        """Append pages for a non-base shape class within the byte budget.

        Returns the class capacity after growth.  The base class is owned by
        ``ensure_layout``; this only serves additional coexisting LOD levels,
        so running out of budget degrades to fewer slots instead of raising.
        """

        shape = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        requested = max(1, int(requested))
        capacity = self._class_capacity(shape)
        if capacity >= requested or self.storage_mode is None:
            return capacity
        max_columns = max(1, self.max_texture_size // shape[1])
        max_rows = max(1, self.max_texture_size // shape[0])
        max_slots_per_page = max(1, int(max_columns * max_rows))
        bytes_per_slot = max(1, _storage_mode_bytes_per_pixel(self.storage_mode) * shape[0] * shape[1])
        while capacity < requested:
            remaining_slots = requested - capacity
            page_capacity = min(
                max_slots_per_page,
                max(
                    remaining_slots,
                    min(max_slots_per_page, int(_ATLAS_GROWTH_TARGET_BYTES // bytes_per_slot) or 1),
                ),
            )
            if self.budget_bytes > 0:
                budget_left = self.budget_bytes - self.estimated_gpu_bytes
                page_capacity = min(page_capacity, budget_left // bytes_per_slot)
                if page_capacity < 1:
                    # Before budget-limiting the class, reclaim slots whose
                    # tiles now present a different class and drop pages that
                    # become empty; retry only when bytes were recovered.
                    if self._release_superseded_capacity(protect_shape=shape):
                        continue
                    break
            self.pages.append(
                TextureAtlasPage(
                    self._gloo,
                    tile_shape=shape,
                    capacity=int(page_capacity),
                    storage_mode=self.storage_mode,
                    max_texture_size=self.max_texture_size,
                )
            )
            self.serial += 1
            capacity += int(page_capacity)
        return capacity

    def _release_superseded_capacity(self, *, protect_shape: tuple[int, int] | None = None) -> bool:
        """Free superseded slots and drop emptied pages to recover budget bytes.

        A slot is superseded when its tile presents an acknowledged payload of
        a different residency class (e.g. the native slot of a tile whose
        reduced level is now on screen).  A slot that is the currently
        presented payload for any tile is never freed (ADR 0041 gate 5).
        Only pages that this call itself emptied are dropped, so capacity
        guaranteed to the current commit by ``ensure_layout`` or class growth
        is never torn down.  Returns True when at least one page was dropped,
        i.e. GPU budget bytes were actually recovered.
        """

        protect = None if protect_shape is None else (int(protect_shape[0]), int(protect_shape[1]))
        active_bases = self.active_base_source_ids
        # Reclaim in LRU order among superseded slots, keeping the retained
        # adjacent level of active tiles for a second pass: losing it costs a
        # re-upload on the next level flip, so it goes only when reclaiming
        # everything else recovered no bytes (ADR 0050).
        ordered = sorted(
            tuple(self.superseded_keys),
            key=lambda key: (
                _lod_invariant_source_id(self.source_ids.get(key)) in active_bases,
                self.last_used.get(key, -1),
            ),
        )
        for adjacent_pass in (False, True):
            touched_pages: set[int] = set()
            for key in ordered:
                adjacent = _lod_invariant_source_id(self.source_ids.get(key)) in active_bases
                if adjacent != adjacent_pass:
                    continue
                if key in self.active_resident_keys or self.resident_tiles.get(key):
                    continue
                slot_ref = self.resident_slots.get(key)
                if slot_ref is None:
                    self.superseded_keys.discard(key)
                    continue
                page_index, slot = (int(slot_ref[0]), int(slot_ref[1]))
                page = self.pages[page_index]
                if protect is not None and page.tile_shape == protect:
                    # Same-class slots are useful as-is: ordinary eviction
                    # reuses them without any byte recovery, so keep them.
                    continue
                page.slot_owners[slot] = None
                page._free_slots.append(slot)
                self.resident_slots.pop(key, None)
                self.source_ids.pop(key, None)
                self.last_used.pop(key, None)
                self.superseded_keys.discard(key)
                self.eviction_count += 1
                self.superseded_reclaimed_count += 1
                touched_pages.add(page_index)
            empties = [
                index
                for index in sorted(touched_pages)
                if all(owner is None for owner in self.pages[index].slot_owners)
            ]
            if empties:
                self._drop_pages(empties)
                return True
        return False

    def _drop_pages(self, page_indices) -> None:
        dropped = {int(index) for index in page_indices}
        remap: dict[int, int] = {}
        kept: list[TextureAtlasPage] = []
        for old_index, page in enumerate(self.pages):
            if old_index in dropped:
                continue
            remap[old_index] = len(kept)
            kept.append(page)
        self.pages = kept
        self.resident_slots = {
            key: (remap[int(page_index)], int(slot))
            for key, (page_index, slot) in self.resident_slots.items()
        }
        self.tile_slots = {
            tile: (remap[int(page_index)], int(slot))
            for tile, (page_index, slot) in self.tile_slots.items()
        }
        self.pages_dropped_count += len(dropped)
        self.serial += 1

    def requested_capacity(
        self,
        *,
        active_count: int,
        reserve_count: int,
        storage_mode: str,
        tile_shape: tuple[int, int],
        budget_bytes: int | None = None,
    ) -> int:
        """Return a bounded, chunked atlas capacity target.

        A multi-page atlas does not need to allocate the complete montage on
        the first progressive commit.  Doing so can synchronously reserve
        hundreds of MiB while only one or two tiles are useful.  Grow in
        byte-sized page chunks instead; existing pages and their residency are
        retained when another page is appended.
        """

        active_count = max(1, int(active_count))
        reserve_count = max(active_count, int(reserve_count or active_count))
        tile_h, tile_w = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        bytes_per_slot = _storage_mode_bytes_per_pixel(storage_mode) * tile_h * tile_w
        max_columns = max(1, self.max_texture_size // tile_w)
        max_rows = max(1, self.max_texture_size // tile_h)
        max_slots_per_page = max(1, int(max_columns * max_rows))
        growth_slots = max(
            1,
            min(
                max_slots_per_page,
                int(_ATLAS_GROWTH_TARGET_BYTES // max(1, bytes_per_slot)),
            ),
        )
        chunked = int(np.ceil(active_count / growth_slots)) * growth_slots
        requested = max(active_count, min(reserve_count, chunked))
        effective_budget = self.budget_bytes if budget_bytes is None else max(0, int(budget_bytes))
        if effective_budget > 0:
            budget_slots = max(0, effective_budget // max(1, bytes_per_slot))
            # The active set is mandatory.  Returning active_count lets
            # ensure_layout raise a precise capacity error when even it does
            # not fit, while speculative reserve headroom is simply clamped.
            requested = max(active_count, min(requested, budget_slots))
        return max(1, int(requested))

    def update_payloads(
        self,
        payloads: dict[int, DisplayTilePayload],
        *,
        tile_shape: tuple[int, int],
        dirty_tiles: tuple[int, ...] | None,
        rgb_already_windowed: bool,
        reserve_count: int | None = None,
        near_tiles: tuple[int, ...] = (),
        near_tile_source_ids: dict[int, object] | None = None,
        budget_bytes: int | None = None,
        tile_delta=None,
    ) -> tuple[dict[int, tuple[float, float, float, float]], TileLayerUpdateStats]:
        # Residency is a data-keyed cache; visibility is a presentation choice.
        # A viewport commit may hide or reveal tile mappings, but it must not
        # make resident sources cold again.  Only incompatible atlas storage,
        # explicit reset/context loss, budget eviction, or a new source identity
        # can require another texture upload.
        start = perf_counter()
        tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
        payload_map = {int(key): value for key, value in dict(payloads or {}).items()}
        explicit_upserts = {
            int(key): value
            for key, value in dict(getattr(tile_delta, "upserts", {}) or {}).items()
        }
        raw_active_tiles = None if tile_delta is None else getattr(tile_delta, "active_tiles", None)
        active_tiles = (
            tuple(int(tile) for tile in tuple(raw_active_tiles))
            if raw_active_tiles is not None
            else tuple(sorted(payload_map))
        )
        active_set = set(active_tiles)
        raw_payload_items = tuple(
            (int(tile), payload_map[int(tile)])
            for tile in active_tiles
            if int(tile) in payload_map
        )
        storage_mode = (
            _atlas_storage_mode(raw_payload_items, rgb_already_windowed=rgb_already_windowed)
            if raw_payload_items
            else (self.storage_mode or "scalar")
        )
        payload_items = tuple(
            (tile, payload)
            for tile, payload in raw_payload_items
            if _payload_supported_by_storage_mode(payload, storage_mode, rgb_already_windowed=rgb_already_windowed)
        )
        unsupported_items = len(raw_payload_items) - len(payload_items)
        base_class_items = sum(
            1
            for _tile_number, item_payload in payload_items
            if _payload_class_shape(item_payload) == (tile_h, tile_w)
        )
        if base_class_items == len(payload_items):
            base_active_count = max(1, len(payload_items))
            base_reserve_count = max(len(active_tiles), int(reserve_count or 0))
        else:
            # Reduced levels occupy their own shape classes (ADR 0050).  The
            # base class is sized by the payloads that actually need native
            # slots, so an ingest-reduced cold fill does not allocate a full
            # native atlas it never uploads to.
            base_active_count = max(1, base_class_items)
            base_reserve_count = base_active_count
        requested_capacity = self.requested_capacity(
            active_count=base_active_count,
            reserve_count=base_reserve_count,
            storage_mode=storage_mode,
            tile_shape=tile_shape,
            budget_bytes=budget_bytes,
        )
        layout_invalidates_residency = (
            self.tile_shape != (tile_h, tile_w)
            or self.storage_mode != _normalize_storage_mode(storage_mode)
        )
        rebuilt = self.ensure_layout(
            tile_shape=tile_shape,
            count=requested_capacity,
            storage_mode=storage_mode,
            budget_bytes=budget_bytes,
        )
        active = {int(tile_number) for tile_number, _payload in payload_items}
        active_keys = {_resident_key(payload) for _tile_number, payload in payload_items}
        self.active_resident_keys = set(active_keys)
        self.active_base_source_ids = {
            _lod_invariant_source_id(payload.source_id) for _tile_number, payload in payload_items
        }
        for tile_number in tuple(getattr(tile_delta, "removals", ()) or ()):
            self._clear_tile_mapping(int(tile_number))
        near = {int(tile) for tile in tuple(near_tiles or ())}
        near_keys = self._near_resident_keys(near_tile_source_ids)
        near_keys.update(_resident_key(payload) for tile_number, payload in payload_items if int(tile_number) in near)
        uvs: dict[int, tuple[float, float, float, float]] = {}
        active_tile_slots: dict[int, tuple[int, int]] = {}
        active_tile_keys: dict[int, object] = {}
        active_tile_uvs: dict[int, tuple[float, float, float, float]] = {}
        uploads = 0
        upload_bytes = 0
        complex_uploads = 0
        texture_prepare_ms = 0.0
        texture_submit_ms = 0.0
        updated = 0
        skipped = int(unsupported_items)
        evictions_before = self.eviction_count
        evicted_near_before = self.evicted_near_count
        superseded_reclaimed_before = self.superseded_reclaimed_count
        uploaded_keys: set[object] = set()
        tile_h, tile_w = self.tile_shape or tile_shape
        base_shape = (int(tile_h), int(tile_w))
        class_counts: dict[tuple[int, int], int] = {}
        for _tile_number, payload in payload_items:
            class_shape = _payload_class_shape(payload)
            if class_shape != base_shape:
                class_counts[class_shape] = class_counts.get(class_shape, 0) + 1
        for class_shape, class_count in class_counts.items():
            self._ensure_class_capacity(class_shape, class_count)
        capacity_skipped_tiles: set[int] = set()

        for tile_number, payload in payload_items:
            resident_key = _resident_key(payload)
            class_shape = _payload_class_shape(payload)
            try:
                page_index, slot, newly_assigned = self._slot_for(
                    resident_key,
                    active_keys=active_keys,
                    near_keys=near_keys,
                    tile_shape=class_shape,
                )
            except AtlasCapacityError:
                if class_shape == base_shape:
                    raise
                # Reduced-level slots are additive capacity.  When their class
                # cannot grow within the budget, retain whatever level the
                # tile currently presents instead of tearing the mapping down.
                capacity_skipped_tiles.add(int(tile_number))
                skipped += 1
                continue
            active_tile_slots[int(tile_number)] = (int(page_index), int(slot))
            active_tile_keys[int(tile_number)] = resident_key
            page = self.pages[int(page_index)]
            self._touch(resident_key)
            source_changed = self.source_ids.get(resident_key) != payload.source_id
            missing_uploaded_source = resident_key not in self.source_ids
            should_upload = bool(
                layout_invalidates_residency
                or newly_assigned
                or source_changed
                or missing_uploaded_source
            )
            uvs[tile_number] = page.uv_for_slot_with_gutter(slot, gutter=_payload_gutter(payload))
            active_tile_uvs[int(tile_number)] = uvs[tile_number]
            if not should_upload:
                skipped += 1
                continue
            uploaded_keys.add(resident_key)

            scalar, color, prepare_ms = _prepare_payload_texture_data(
                payload,
                tile_shape=page.tile_shape,
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=self.scalar_is_atlas,
                need_color=self.color_is_atlas,
            )
            texture_prepare_ms += prepare_ms
            y0, x0 = page.offset_for_slot(slot)
            if scalar is not None:
                texture_submit_ms += _upload_texture_plane(
                    page.scalar_texture,
                    scalar,
                    offset=(int(y0), int(x0)),
                    copy=_upload_copy_required(scalar, payload, force=page.complex_is_atlas),
                )
                uploads += 1
                upload_bytes += int(scalar.nbytes)
                if page.complex_is_atlas:
                    complex_uploads += 1
            if color is not None:
                texture_submit_ms += _upload_texture_plane(
                    page.color_texture,
                    color,
                    offset=(int(y0), int(x0)),
                    copy=_upload_copy_required(color, payload),
                )
                uploads += 1
                upload_bytes += int(color.nbytes)
            if (scalar is not None or color is not None) and page.mipmap_levels:
                page.mipmap_dirty = True
            self.source_ids[resident_key] = payload.source_id
            updated += 1

        presented_tiles = tuple(
            int(tile_number)
            for tile_number, payload in payload_items
            if int(tile_number) in active_tile_slots
            and self.source_ids.get(_resident_key(payload)) == payload.source_id
        )
        presented_set = set(presented_tiles)
        for tile in active_set:
            if int(tile) not in presented_set and int(tile) not in capacity_skipped_tiles:
                self._clear_tile_mapping(int(tile))
        level_swaps_zero_upload = 0
        level_swaps_with_upload = 0
        for tile in presented_tiles:
            new_key = active_tile_keys[int(tile)]
            previous_key = self.tile_resident_keys.get(int(tile))
            if (
                previous_key is not None
                and previous_key != new_key
                and _resident_key_lod(previous_key) != _resident_key_lod(new_key)
            ):
                if new_key in uploaded_keys:
                    level_swaps_with_upload += 1
                else:
                    level_swaps_zero_upload += 1
            self._set_tile_mapping(
                int(tile),
                new_key,
                active_tile_slots[int(tile)][0],
                active_tile_slots[int(tile)][1],
                active_tile_uvs[int(tile)],
            )
        self.lod_level_swaps_zero_upload += level_swaps_zero_upload
        self.lod_level_swaps_with_upload += level_swaps_with_upload
        uvs = self.tile_uvs
        # Mipmap regens run at draw time (visual-side); stats report the
        # regens completed since the previous update, one commit behind.
        mipmap_updates_total = sum(int(getattr(page, "mipmap_updates", 0) or 0) for page in self.pages)
        mipmap_updates_delta = max(0, mipmap_updates_total - self._mipmap_updates_reported)
        self._mipmap_updates_reported = mipmap_updates_total
        mipmap_available = any(bool(getattr(page, "mipmap_ready", False)) for page in self.pages)
        elapsed = (perf_counter() - start) * 1000.0 if updated or rebuilt else 0.0
        return uvs, TileLayerUpdateStats(
            visible_items=len(presented_tiles),
            presented_tiles=presented_tiles,
            committed_upserts=tuple(
                int(tile)
                for tile in sorted(explicit_upserts)
                if int(tile) in presented_set
            ),
            resident_items=self.resident_count,
            storage_capacity=self.capacity,
            storage_rebuilds=int(rebuilt),
            storage_evictions=self.eviction_count - evictions_before,
            texture_uploads=uploads,
            texture_upload_bytes=upload_bytes,
            items_updated=updated,
            items_skipped=skipped,
            estimated_gpu_bytes=self.estimated_gpu_bytes,
            cpu_shadow_bytes=self.cpu_shadow_bytes,
            upload_ms=elapsed,
            texture_prepare_ms=texture_prepare_ms,
            texture_submit_ms=texture_submit_ms,
            page_count=len(self.pages),
            active_pages=len({self.tile_slots[int(tile)][0] for tile in active if int(tile) in self.tile_slots}),
            device_max_texture_size=self.max_texture_size,
            budget_bytes=self.budget_bytes,
            near_resident_items=len(near_keys.intersection(self.source_ids)),
            warm_resident_items=max(0, self.resident_count - len(active)),
            evicted_near_items=self.evicted_near_count - evicted_near_before,
            lod_level=_max_payload_lod_level(payload_map),
            lod_factor=_max_payload_lod_factor(payload_map),
            source_texels_per_pixel=float(_max_payload_lod_factor(payload_map)),
            gutter_pixels=_max_payload_gutter(payload_map),
            mipmap_updates=mipmap_updates_delta,
            mipmap_available=mipmap_available,
            complex_texture_uploads=complex_uploads,
            lod_level_swaps_zero_upload=level_swaps_zero_upload,
            lod_level_swaps_with_upload=level_swaps_with_upload,
            superseded_reclaimed_under_pressure=(
                self.superseded_reclaimed_count - superseded_reclaimed_before
            ),
        )

    def warm_payloads(
        self,
        payloads: dict[int, DisplayTilePayload],
        *,
        tile_shape: tuple[int, int],
        rgb_already_windowed: bool,
        near_tile_source_ids: dict[int, object] | None = None,
        budget_bytes: int | None = None,
    ) -> TileLayerUpdateStats:
        start = perf_counter()
        if not payloads:
            return TileLayerUpdateStats(
                resident_items=self.resident_count,
                storage_capacity=self.capacity,
                estimated_gpu_bytes=self.estimated_gpu_bytes,
                cpu_shadow_bytes=self.cpu_shadow_bytes,
                page_count=len(self.pages),
                device_max_texture_size=self.max_texture_size,
                budget_bytes=self.budget_bytes,
                warm_resident_items=max(0, self.resident_count - len(self.active_resident_keys)),
            )
        payload_items = tuple((int(key), value) for key, value in payloads.items())
        requested_mode = _atlas_storage_mode(payload_items, rgb_already_windowed=rgb_already_windowed)
        if self.storage_mode is not None and self.storage_mode != requested_mode:
            # Warm work must never replace the active atlas layout.  A later
            # visible commit can deliberately switch storage modes.
            return TileLayerUpdateStats(
                resident_items=self.resident_count,
                storage_capacity=self.capacity,
                items_skipped=len(payload_items),
                estimated_gpu_bytes=self.estimated_gpu_bytes,
                cpu_shadow_bytes=self.cpu_shadow_bytes,
                page_count=len(self.pages),
                device_max_texture_size=self.max_texture_size,
                budget_bytes=self.budget_bytes,
                warm_resident_items=max(0, self.resident_count - len(self.active_resident_keys)),
                capacity_warning="warm payload storage mode differs from the active atlas",
            )
        target_count = len(set(self.active_resident_keys).union(_resident_key(payload) for _tile, payload in payload_items))
        tile_h, tile_w = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        bytes_per_slot = _storage_mode_bytes_per_pixel(requested_mode) * tile_h * tile_w
        effective_budget = self.budget_bytes if budget_bytes is None else max(0, int(budget_bytes))
        if effective_budget > 0:
            target_count = min(target_count, effective_budget // max(1, bytes_per_slot))
        mandatory_count = max(1, len(self.active_resident_keys))
        # Once a chunk is full, requesting one slot beyond current capacity
        # advances to the next chunk.  The mandatory active set remains
        # separate so speculative warm coverage is clamped by the budget.
        growth_trigger = max(mandatory_count, min(target_count, max(1, self.capacity + 1)))
        requested_capacity = self.requested_capacity(
            active_count=growth_trigger,
            reserve_count=max(1, target_count),
            storage_mode=requested_mode,
            tile_shape=tile_shape,
            budget_bytes=budget_bytes,
        )
        self.ensure_layout(
            tile_shape=tile_shape,
            count=requested_capacity,
            storage_mode=requested_mode,
            budget_bytes=budget_bytes,
        )

        if budget_bytes is not None:
            self.budget_bytes = max(0, int(budget_bytes))
        near_keys = self._near_resident_keys(near_tile_source_ids)
        near_keys.update(_resident_key(payload) for payload in dict(payloads).values())
        uploads = 0
        upload_bytes = 0
        complex_uploads = 0
        texture_prepare_ms = 0.0
        texture_submit_ms = 0.0
        updated = 0
        skipped = 0
        evictions_before = self.eviction_count
        evicted_near_before = self.evicted_near_count
        tile_h, tile_w = self.tile_shape or tile_shape

        # At most capacity-active distinct warm keys can coexist.  Limiting the
        # batch prevents upload/evict/upload thrash when the near ring is larger
        # than the configured GPU budget.
        available_warm_keys = max(0, self.capacity - len(self.active_resident_keys))
        new_warm_budget = available_warm_keys
        skipped_budget = 0
        for _tile_number, payload in payload_items:
            if not _payload_supported_by_storage_mode(payload, self.storage_mode, rgb_already_windowed=rgb_already_windowed):
                skipped += 1
                continue
            resident_key = _resident_key(payload)
            if self.source_ids.get(resident_key) == payload.source_id:
                self._touch(resident_key)
                skipped += 1
                continue
            if resident_key not in self.resident_slots and new_warm_budget <= 0:
                skipped += 1
                skipped_budget += 1
                continue
            if resident_key not in self.resident_slots:
                new_warm_budget -= 1
            class_shape = _payload_class_shape(payload)
            if class_shape != (tile_h, tile_w):
                self._ensure_class_capacity(class_shape, 1)
            try:
                page_index, slot, _newly_assigned = self._slot_for(
                    resident_key,
                    active_keys=set(self.active_resident_keys),
                    near_keys=near_keys,
                    tile_shape=class_shape,
                )
            except AtlasCapacityError:
                # Warm work is speculative; never let a full shape class stop
                # the batch or evict active residency of another class.
                skipped += 1
                continue
            page = self.pages[int(page_index)]
            scalar, color, prepare_ms = _prepare_payload_texture_data(
                payload,
                tile_shape=page.tile_shape,
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=page.scalar_is_atlas,
                need_color=page.color_is_atlas,
            )
            texture_prepare_ms += prepare_ms
            y0, x0 = page.offset_for_slot(slot)
            if scalar is not None:
                texture_submit_ms += _upload_texture_plane(
                    page.scalar_texture,
                    scalar,
                    offset=(int(y0), int(x0)),
                    copy=_upload_copy_required(scalar, payload, force=page.complex_is_atlas),
                )
                uploads += 1
                upload_bytes += int(scalar.nbytes)
                if page.complex_is_atlas:
                    complex_uploads += 1
            if color is not None:
                texture_submit_ms += _upload_texture_plane(
                    page.color_texture,
                    color,
                    offset=(int(y0), int(x0)),
                    copy=_upload_copy_required(color, payload),
                )
                uploads += 1
                upload_bytes += int(color.nbytes)
            if (scalar is not None or color is not None) and page.mipmap_levels:
                page.mipmap_dirty = True
            self.source_ids[resident_key] = payload.source_id
            self._touch(resident_key)
            updated += 1

        elapsed = (perf_counter() - start) * 1000.0 if updated else 0.0
        return TileLayerUpdateStats(
            resident_items=self.resident_count,
            storage_capacity=self.capacity,
            storage_evictions=self.eviction_count - evictions_before,
            texture_uploads=uploads,
            texture_upload_bytes=upload_bytes,
            items_updated=updated,
            items_skipped=skipped,
            estimated_gpu_bytes=self.estimated_gpu_bytes,
            cpu_shadow_bytes=self.cpu_shadow_bytes,
            upload_ms=elapsed,
            texture_prepare_ms=texture_prepare_ms,
            texture_submit_ms=texture_submit_ms,
            page_count=len(self.pages),
            active_pages=len({page_index for page_index, _slot in self.tile_slots.values()}),
            device_max_texture_size=self.max_texture_size,
            budget_bytes=self.budget_bytes,
            near_resident_items=len(near_keys.intersection(self.source_ids)),
            warm_resident_items=max(0, self.resident_count - len(self.active_resident_keys)),
            evicted_near_items=self.evicted_near_count - evicted_near_before,
            lod_level=_max_payload_lod_level(payloads),
            lod_factor=_max_payload_lod_factor(payloads),
            source_texels_per_pixel=float(_max_payload_lod_factor(payloads)),
            gutter_pixels=_max_payload_gutter(payloads),
            mipmap_updates=0,
            mipmap_available=False,
            complex_texture_uploads=complex_uploads,
            capacity_warning=(
                f"skipped {skipped_budget} warm tiles because the residency budget is full"
                if skipped_budget
                else ""
            ),
        )

    def _slot_for(
        self,
        resident_key: object,
        *,
        active_keys: set[object],
        near_keys: set[object],
        tile_shape: tuple[int, int] | None = None,
    ) -> tuple[int, int, bool]:
        current = self.resident_slots.get(resident_key)
        if current is not None:
            return int(current[0]), int(current[1]), False

        shape = self.tile_shape if tile_shape is None else (int(tile_shape[0]), int(tile_shape[1]))
        class_pages = tuple(
            (page_index, page)
            for page_index, page in enumerate(self.pages)
            if shape is None or page.tile_shape == shape
        )
        for page_index, page in class_pages:
            slot = page.take_free_slot(resident_key)
            if slot is None:
                continue
            self.resident_slots[resident_key] = (int(page_index), int(slot))
            return int(page_index), int(slot), True

        candidates = []
        active_bases = self.active_base_source_ids
        for page_index, page in class_pages:
            for slot, owner in enumerate(page.slot_owners):
                if owner is not None and owner not in active_keys:
                    # Eviction preference under pressure (ADR 0050): first
                    # superseded classes of tiles that are neither active nor
                    # near, then presented classes of non-active tiles, and
                    # only as a last resort the superseded (retained
                    # adjacent-level) classes of active tiles, whose loss
                    # forces a re-upload on the next level flip.  A presented
                    # class of an active tile is never a candidate at all.
                    superseded = owner in self.superseded_keys and not self.resident_tiles.get(owner)
                    adjacent = superseded and _lod_invariant_source_id(self.source_ids.get(owner)) in active_bases
                    if superseded and not adjacent and owner not in near_keys:
                        rank = 0
                    elif superseded and not adjacent:
                        rank = 1
                    elif not superseded and owner not in near_keys:
                        rank = 2
                    elif not superseded:
                        rank = 3
                    else:
                        rank = 4
                    candidates.append((rank, self.last_used.get(owner, -1), owner, int(page_index), int(slot)))
        if not candidates:
            raise AtlasCapacityError(
                f"atlas has {self._class_capacity(shape) if shape else self.capacity} slots of shape {shape} "
                f"but {len(active_keys)} active tiles require residency"
            )
        _priority, _last, victim, page_index, slot = min(candidates, key=lambda item: (item[0], item[1], repr(item[2])))
        self._discard_tile_mappings_for_resident_key(victim)
        self.resident_slots.pop(victim, None)
        self.source_ids.pop(victim, None)
        self.last_used.pop(victim, None)
        if victim in self.superseded_keys:
            self.superseded_reclaimed_count += 1
        self.superseded_keys.discard(victim)
        self.eviction_count += 1
        if victim in near_keys:
            self.evicted_near_count += 1
        page = self.pages[int(page_index)]
        page.slot_owners[int(slot)] = resident_key
        self.resident_slots[resident_key] = (int(page_index), int(slot))
        return int(page_index), int(slot), True

    def _clear_tile_mapping(self, tile_number: int) -> None:
        tile_number = int(tile_number)
        old_key = self.tile_resident_keys.pop(tile_number, None)
        if old_key is not None:
            tiles = self.resident_tiles.get(old_key)
            if tiles is not None:
                tiles.discard(tile_number)
                if not tiles:
                    self.resident_tiles.pop(old_key, None)
        self.tile_slots.pop(tile_number, None)
        self.tile_uvs.pop(tile_number, None)

    def _set_tile_mapping(
        self,
        tile_number: int,
        resident_key: object,
        page_index: int,
        slot: int,
        uv: tuple[float, float, float, float],
    ) -> None:
        tile_number = int(tile_number)
        old_key = self.tile_resident_keys.get(tile_number)
        if old_key is not None and old_key != resident_key:
            tiles = self.resident_tiles.get(old_key)
            if tiles is not None:
                tiles.discard(tile_number)
                if not tiles:
                    self.resident_tiles.pop(old_key, None)
            # The replacement is backend-acknowledged and presented for this
            # tile: the displaced key's slot becomes reclaimable once no tile
            # presents it anymore (ADR 0041 gate 5 holds until here).
            if not self.resident_tiles.get(old_key) and old_key in self.resident_slots:
                self.superseded_keys.add(old_key)
        self.superseded_keys.discard(resident_key)
        self.tile_slots[tile_number] = (int(page_index), int(slot))
        self.tile_resident_keys[tile_number] = resident_key
        self.resident_tiles.setdefault(resident_key, set()).add(tile_number)
        self.tile_uvs[tile_number] = uv

    def _discard_tile_mappings_for_resident_key(self, resident_key: object) -> None:
        for tile_number in tuple(self.resident_tiles.pop(resident_key, set())):
            tile_number = int(tile_number)
            if self.tile_resident_keys.get(tile_number) == resident_key:
                self.tile_resident_keys.pop(tile_number, None)
                self.tile_slots.pop(tile_number, None)
                self.tile_uvs.pop(tile_number, None)

    def _touch(self, resident_key: object) -> None:
        self._clock += 1
        self.last_used[resident_key] = int(self._clock)

    def _near_resident_keys(self, near_tile_source_ids) -> set[object]:
        """Resolve base or complete source identities to current residents."""

        requested = tuple(dict(near_tile_source_ids or {}).values())
        exact = {_source_resident_key(source_id) for source_id in requested}
        bases = {_base_texture_source_id(source_id) for source_id in requested}
        for resident_key, source_id in self.source_ids.items():
            if resident_key in exact or _base_texture_source_id(source_id) in bases:
                exact.add(resident_key)
        return exact


class GpuMontageLayer:
    def __init__(self, *, scene, visuals, gloo, transforms, parent, limits: GpuDeviceLimits | None = None):
        self._scene = scene
        self._visuals = visuals
        self._gloo = gloo
        self._transforms = transforms
        self._parent = parent
        self._device_limits = limits or query_gpu_device_limits(gloo)
        self._pool = TextureAtlasPool(gloo, limits=self._device_limits)
        self._visuals_by_page: list[object] = []
        self._geometry_keys: dict[int, tuple[object, ...]] = {}
        self._page_payloads_by_index: list[dict[int, DisplayTilePayload]] = []
        self._montage_geometry_key: tuple[int, int, int, int, int] | None = None
        self._atlas_serial: int = -1
        self._levels: tuple[float, float] = (0.0, 1.0)
        self._shader_mapping = None
        self._shader_mapping_key = None
        self._visible_items = 0
        self._last_stats = TileLayerUpdateStats()
        self._ensure_visual_count(1)

    @property
    def visual(self):
        self._ensure_visual_count(1)
        return self._visuals_by_page[0]

    @property
    def last_stats(self) -> TileLayerUpdateStats:
        return self._last_stats

    def reset_residency(self) -> None:
        for visual in self._visuals_by_page:
            visual.visible = False
        self._pool = TextureAtlasPool(self._gloo, limits=self._device_limits)
        self._geometry_keys.clear()
        self._page_payloads_by_index.clear()
        self._montage_geometry_key = None
        self._atlas_serial = -1
        self._visible_items = 0
        self._last_stats = TileLayerUpdateStats()

    def clear(self) -> None:
        # Hiding a layer must not discard useful GPU residency.  A later
        # viewport/session can reuse the same source identities.
        for visual in self._visuals_by_page:
            visual.visible = False
        self._geometry_keys.clear()
        self._page_payloads_by_index.clear()
        self._montage_geometry_key = None
        self._atlas_serial = -1
        self._visible_items = 0

    def set_levels(self, levels) -> TileLayerUpdateStats:
        return self.set_presentation_uniforms(levels=levels)

    def set_shader_mapping(self, mapping) -> TileLayerUpdateStats:
        return self.set_presentation_uniforms(shader_mapping=mapping)

    def set_presentation_uniforms(self, *, levels=_UNSET, shader_mapping=_UNSET) -> TileLayerUpdateStats:
        level_updates = 0
        mapping_updates = 0
        if levels is not _UNSET:
            normalized = _normalize_levels(levels, self._levels)
            if normalized != self._levels:
                self._levels = normalized
                for visual in self._visuals_by_page:
                    level_updates += int(bool(visual.set_levels(normalized)))
        if shader_mapping is not _UNSET:
            mapping_key = _mapping_identity_key(shader_mapping)
            if mapping_key != self._shader_mapping_key:
                self._shader_mapping = shader_mapping
                self._shader_mapping_key = mapping_key
                for visual in self._visuals_by_page:
                    mapping_updates += int(bool(visual.set_shader_mapping(shader_mapping)))
        previous = self._last_stats
        self._last_stats = TileLayerUpdateStats(
            visible_items=self._visible_items,
            presented_tiles=tuple(int(tile) for tile in sorted(self._pool.tile_slots)),
            resident_items=self._pool.resident_count,
            storage_capacity=self._pool.capacity,
            level_updates=int(bool(level_updates)),
            shader_uniform_updates=level_updates + mapping_updates,
            items_skipped=self._visible_items,
            estimated_gpu_bytes=self._pool.estimated_gpu_bytes,
            cpu_shadow_bytes=self._pool.cpu_shadow_bytes,
            page_count=len(self._pool.pages),
            active_pages=sum(1 for visual in self._visuals_by_page if bool(getattr(visual, "visible", False))),
            device_max_texture_size=self._pool.max_texture_size,
            budget_bytes=self._pool.budget_bytes,
            near_resident_items=int(getattr(previous, "near_resident_items", 0) or 0),
            warm_resident_items=max(0, self._pool.resident_count - len(self._pool.active_resident_keys)),
            capacity_warning=str(getattr(previous, "capacity_warning", "") or ""),
            lod_level=int(getattr(previous, "lod_level", 0) or 0),
            lod_factor=int(getattr(previous, "lod_factor", 1) or 1),
            source_texels_per_pixel=float(getattr(previous, "source_texels_per_pixel", 0.0) or 0.0),
            gutter_pixels=int(getattr(previous, "gutter_pixels", 0) or 0),
            mipmap_available=bool(getattr(previous, "mipmap_available", False)),
        )
        return self._last_stats

    def _ensure_visual_count(self, count: int) -> None:
        while len(self._visuals_by_page) < max(1, int(count)):
            visual = self._scene.visuals.create_visual_node(GpuWindowedTileVisual)(parent=self._parent)
            visual.order = 10
            visual.visible = False
            visual.set_levels(self._levels)
            visual.set_shader_mapping(self._shader_mapping)
            self._visuals_by_page.append(visual)

    def update(
        self,
        *,
        payloads: dict[int, DisplayTilePayload],
        geometry,
        levels,
        dirty_tiles: tuple[int, ...] | None,
        rgb_already_windowed: bool,
        shader_mapping=None,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
        frame_plan=None,
    ) -> TileLayerUpdateStats:
        layout = tile_layout_map(geometry, frame_plan=frame_plan)
        if not layout:
            self.clear()
            return TileLayerUpdateStats()
        payloads = {int(key): value for key, value in dict(payloads or {}).items()}
        reserve_count = max(
            _atlas_reserve_count(geometry, minimum=len(payloads), frame_plan=frame_plan),
            len(tuple(getattr(tile_delta, "planned_tiles", ()) or ())),
        )
        near_tiles = tuple(getattr(tile_delta, "near_tiles", ()) or ())
        near_tile_source_ids = dict(getattr(tile_delta, "near_tile_source_ids", {}) or {})
        uvs, texture_stats = self._pool.update_payloads(
            payloads,
            tile_shape=_atlas_base_tile_shape_for_payloads(
                payloads,
                fallback=_layout_tile_shape(layout),
            ),
            dirty_tiles=dirty_tiles,
            rgb_already_windowed=rgb_already_windowed,
            reserve_count=reserve_count,
            near_tiles=near_tiles,
            near_tile_source_ids=near_tile_source_ids,
            budget_bytes=tile_residency_budget_bytes,
            tile_delta=tile_delta,
        )
        self._ensure_visual_count(len(self._pool.pages))
        vertex_uploads = 0
        active_pages = set()
        page_payloads_by_index, dirty_pages = self._sync_page_payloads(
            payloads,
            presented_tiles=tuple(texture_stats.presented_tiles or ()),
            layout=layout,
            rgb_already_windowed=rgb_already_windowed,
        )
        for page_index, page in enumerate(self._pool.pages):
            page_payloads = page_payloads_by_index[page_index]
            visual = self._visuals_by_page[page_index]
            if page_payloads:
                active_pages.add(page_index)
            if page_index in dirty_pages or page_index not in self._geometry_keys:
                geometry_key = _page_geometry_key(
                    page_payloads,
                    self._pool,
                    page,
                    layout,
                    rgb_already_windowed=rgb_already_windowed,
                )
            else:
                geometry_key = self._geometry_keys.get(page_index)
            if geometry_key != self._geometry_keys.get(page_index):
                vertices, texcoords, modes = _quad_buffers(
                    layout,
                    page_payloads,
                    uvs,
                    rgb_already_windowed=rgb_already_windowed,
                )
                visual.set_geometry(vertices, texcoords, modes)
                self._geometry_keys[page_index] = geometry_key
                vertex_uploads += 1
        levels = _normalize_levels(levels, self._levels)
        level_updates = 0
        if levels != self._levels:
            self._levels = levels
            for visual in self._visuals_by_page:
                level_updates += int(bool(visual.set_levels(levels)))
        mapping_key = _mapping_identity_key(shader_mapping)
        mapping_changed = mapping_key != self._shader_mapping_key
        if mapping_changed:
            self._shader_mapping = shader_mapping
            self._shader_mapping_key = mapping_key
        mapping_updates = 0
        for page_index, page in enumerate(self._pool.pages):
            visual = self._visuals_by_page[page_index]
            visual.set_textures(page.scalar_texture, page.color_texture)
            set_mipmap_page = getattr(visual, "set_mipmap_page", None)
            if set_mipmap_page is not None:
                set_mipmap_page(page)
            if mapping_changed:
                mapping_updates += int(bool(visual.set_shader_mapping(shader_mapping)))
            visual.visible = page_index in active_pages
        for visual in self._visuals_by_page[len(self._pool.pages):]:
            visual.visible = False
        effective_presented_tiles = tuple(
            int(tile)
            for tile in tuple(texture_stats.presented_tiles or ())
        )
        self._visible_items = len(effective_presented_tiles)
        self._last_stats = TileLayerUpdateStats(
            visible_items=len(effective_presented_tiles),
            presented_tiles=effective_presented_tiles,
            committed_upserts=tuple(texture_stats.committed_upserts or ()),
            resident_items=texture_stats.resident_items,
            storage_capacity=texture_stats.storage_capacity,
            storage_rebuilds=texture_stats.storage_rebuilds,
            storage_evictions=texture_stats.storage_evictions,
            texture_uploads=texture_stats.texture_uploads,
            texture_upload_bytes=texture_stats.texture_upload_bytes,
            vertex_uploads=vertex_uploads,
            items_updated=texture_stats.items_updated,
            items_skipped=texture_stats.items_skipped,
            level_updates=int(bool(level_updates)),
            estimated_gpu_bytes=texture_stats.estimated_gpu_bytes,
            cpu_shadow_bytes=texture_stats.cpu_shadow_bytes,
            upload_ms=texture_stats.upload_ms,
            texture_prepare_ms=texture_stats.texture_prepare_ms,
            texture_submit_ms=texture_stats.texture_submit_ms,
            page_count=texture_stats.page_count,
            active_pages=len(active_pages),
            device_max_texture_size=texture_stats.device_max_texture_size,
            budget_bytes=texture_stats.budget_bytes,
            near_resident_items=texture_stats.near_resident_items,
            warm_resident_items=texture_stats.warm_resident_items,
            evicted_near_items=texture_stats.evicted_near_items,
            capacity_warning=texture_stats.capacity_warning,
            lod_level=texture_stats.lod_level,
            lod_factor=texture_stats.lod_factor,
            source_texels_per_pixel=texture_stats.source_texels_per_pixel,
            gutter_pixels=texture_stats.gutter_pixels,
            mipmap_updates=texture_stats.mipmap_updates,
            mipmap_available=texture_stats.mipmap_available,
            complex_texture_uploads=texture_stats.complex_texture_uploads,
            shader_uniform_updates=level_updates + mapping_updates,
            lod_level_swaps_zero_upload=texture_stats.lod_level_swaps_zero_upload,
            lod_level_swaps_with_upload=texture_stats.lod_level_swaps_with_upload,
            superseded_reclaimed_under_pressure=texture_stats.superseded_reclaimed_under_pressure,
        )
        return self._last_stats

    def _sync_page_payloads(self, payloads, *, presented_tiles, layout, rgb_already_windowed: bool):
        page_count = len(self._pool.pages)
        layout_key = (
            tuple(
                (int(region.tile_number), int(region.x), int(region.y), int(region.width), int(region.height))
                for region in sorted(layout.values(), key=lambda item: int(item.tile_number))
            ),
        )
        page_payloads_by_index: list[dict[int, DisplayTilePayload]] = [{} for _page in self._pool.pages]
        active = {int(tile) for tile in tuple(presented_tiles or ())}
        payload_map = {int(tile): payload for tile, payload in dict(payloads or {}).items()}
        for tile in sorted(active):
            payload = payload_map.get(int(tile))
            if payload is None:
                continue
            page_index, _slot = self._pool.tile_slots.get(int(tile), (-1, -1))
            if 0 <= int(page_index) < page_count:
                page_payloads_by_index[int(page_index)][int(tile)] = payload
        dirty_pages: set[int] = set()
        full_refresh = (
            len(self._page_payloads_by_index) != page_count
            or self._montage_geometry_key != layout_key
            or int(self._atlas_serial) != int(self._pool.serial)
        )
        for page_index in range(page_count):
            if full_refresh or _page_geometry_key(
                page_payloads_by_index[int(page_index)],
                self._pool,
                self._pool.pages[int(page_index)],
                layout,
                rgb_already_windowed=rgb_already_windowed,
            ) != self._geometry_keys.get(int(page_index)):
                dirty_pages.add(int(page_index))
        self._page_payloads_by_index = page_payloads_by_index
        self._montage_geometry_key = layout_key
        self._atlas_serial = int(self._pool.serial)
        return self._page_payloads_by_index, dirty_pages

    def warm_residency(
        self,
        *,
        payloads: dict[int, DisplayTilePayload],
        geometry,
        rgb_already_windowed: bool,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
    ) -> TileLayerUpdateStats:
        montage = getattr(geometry, "montage", None)
        if montage is None:
            return TileLayerUpdateStats()
        try:
            return self._pool.warm_payloads(
                {int(key): value for key, value in dict(payloads or {}).items()},
                tile_shape=_atlas_base_tile_shape_for_payloads(payloads, fallback=(int(montage.tile_height), int(montage.tile_width))),
                rgb_already_windowed=rgb_already_windowed,
                near_tile_source_ids=dict(getattr(tile_delta, "near_tile_source_ids", {}) or {}),
                budget_bytes=tile_residency_budget_bytes,
            )
        except AtlasCapacityError as exc:
            previous = self._last_stats
            return TileLayerUpdateStats(
                visible_items=int(previous.visible_items),
                presented_tiles=previous.presented_tiles,
                resident_items=self._pool.resident_count,
                storage_capacity=self._pool.capacity,
                estimated_gpu_bytes=self._pool.estimated_gpu_bytes,
                cpu_shadow_bytes=self._pool.cpu_shadow_bytes,
                page_count=len(self._pool.pages),
                active_pages=int(previous.active_pages),
                device_max_texture_size=self._pool.max_texture_size,
                budget_bytes=self._pool.budget_bytes,
                near_resident_items=int(previous.near_resident_items),
                warm_resident_items=max(0, self._pool.resident_count - len(self._pool.active_resident_keys)),
                capacity_warning=str(exc),
            )


class GpuWindowedTileVisual(Visual):
    _vertex_shader = """
    attribute vec2 a_position;
    attribute vec2 a_texcoord;
    attribute float a_mode;
    varying vec2 v_texcoord;
    varying float v_mode;

    void main() {
        v_texcoord = a_texcoord;
        v_mode = a_mode;
        gl_Position = $transform(vec4(a_position, 0.0, 1.0));
    }
    """

    _fragment_shader = """
    uniform sampler2D u_scalar_texture;
    uniform sampler2D u_color_texture;
    uniform sampler2D u_lut_texture;
    uniform vec2 u_levels;
    uniform float u_scale_mode;
    uniform float u_symlog_constant;
    uniform float u_component_mode;
    varying vec2 v_texcoord;
    varying float v_mode;

    float complex_component(vec2 z) {
        if (u_component_mode > 2.5) {
            return atan(z.y, z.x);
        }
        if (u_component_mode > 1.5) {
            return length(z);
        }
        if (u_component_mode > 0.5) {
            return z.y;
        }
        return z.x;
    }

    float map_scale(float value) {
        if (u_scale_mode > 1.5) {
            return sign(value) * log(1.0 + abs(value) / pow(10.0, u_symlog_constant)) / log(10.0);
        }
        if (u_scale_mode > 0.5) {
            return log(max(value, 0.0)) / log(10.0);
        }
        return value;
    }

    void main() {
        if (v_mode < 0.5) {
            float scalar = texture2D(u_scalar_texture, v_texcoord).r;
            scalar = map_scale(scalar);
            float span = max(u_levels.y - u_levels.x, 1e-12);
            float intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
            if (scalar != scalar) {
                discard;
            }
            vec3 color = texture2D(u_lut_texture, vec2(intensity, 0.5)).rgb;
            gl_FragColor = vec4(color, 1.0);
        } else if (v_mode < 1.5) {
            float scalar = texture2D(u_scalar_texture, v_texcoord).r;
            vec3 color = texture2D(u_color_texture, v_texcoord).rgb;
            scalar = map_scale(scalar);
            float span = max(u_levels.y - u_levels.x, 1e-12);
            float intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
            if (scalar != scalar) {
                discard;
            }
            gl_FragColor = vec4(color * intensity, 1.0);
        } else if (v_mode < 2.5) {
            vec3 color = texture2D(u_color_texture, v_texcoord).rgb;
            gl_FragColor = vec4(color, 1.0);
        } else if (v_mode < 3.5) {
            vec2 z = texture2D(u_scalar_texture, v_texcoord).rg;
            float scalar = complex_component(z);
            scalar = map_scale(scalar);
            float span = max(u_levels.y - u_levels.x, 1e-12);
            float intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
            if (scalar != scalar) {
                discard;
            }
            vec3 color = texture2D(u_lut_texture, vec2(intensity, 0.5)).rgb;
            gl_FragColor = vec4(color, 1.0);
        } else {
            vec2 z = texture2D(u_scalar_texture, v_texcoord).rg;
            float component_scalar = complex_component(z);
            float scalar = component_scalar;
            float span = max(u_levels.y - u_levels.x, 1e-12);
            float phase_index;
            float intensity = 1.0;
            if (u_component_mode > 2.5) {
                scalar = map_scale(component_scalar);
                phase_index = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
            } else {
                scalar = map_scale(length(z));
                intensity = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
                float phase = atan(z.y, z.x);
                phase_index = clamp((phase + 3.141592653589793) / 6.283185307179586, 0.0, 1.0);
            }
            vec3 color = texture2D(u_lut_texture, vec2(phase_index, 0.5)).rgb;
            if (scalar != scalar) {
                discard;
            }
            gl_FragColor = vec4(color * intensity, 1.0);
        }
    }
    """

    def __init__(self, **kwargs):
        from vispy import gloo
        super().__init__(vcode=self._vertex_shader, fcode=self._fragment_shader, **kwargs)
        self._vertices = gloo.VertexBuffer(np.zeros((0, 2), dtype=np.float32))
        self._texcoords = gloo.VertexBuffer(np.zeros((0, 2), dtype=np.float32))
        self._modes = gloo.VertexBuffer(np.zeros((0,), dtype=np.float32))
        self.vertex_data = np.zeros((0, 2), dtype=np.float32)
        self.texcoord_data = np.zeros((0, 2), dtype=np.float32)
        self.mode_data = np.zeros((0,), dtype=np.float32)
        self._bounds_xy = (0.0, 0.0, 0.0, 0.0)
        self._scalar_texture = None
        self._color_texture = None
        self._lut_texture = None
        self._lut_key = None
        self._lut_default_phase = False
        self._shader_mapping_key = None
        self._levels = (0.0, 1.0)
        self._scale_mode = 0.0
        self._symlog_constant = 0.0
        self._component_mode = 0.0
        self._mipmap_page = None
        self.set_gl_state(depth_test=False, cull_face=False, blend=False)
        self._draw_mode = "triangles"
        self.freeze()

    def set_geometry(self, vertices, texcoords, modes) -> None:
        vertices = np.asarray(vertices, dtype=np.float32).reshape((-1, 2))
        texcoords = np.asarray(texcoords, dtype=np.float32).reshape((-1, 2))
        modes = np.asarray(modes, dtype=np.float32).reshape((-1,))
        self.vertex_data = vertices
        self.texcoord_data = texcoords
        self.mode_data = modes
        self._vertices.set_data(vertices)
        self._texcoords.set_data(texcoords)
        self._modes.set_data(modes)
        if len(vertices):
            self._bounds_xy = (
                float(np.nanmin(vertices[:, 0])),
                float(np.nanmax(vertices[:, 0])),
                float(np.nanmin(vertices[:, 1])),
                float(np.nanmax(vertices[:, 1])),
            )
        else:
            self._bounds_xy = (0.0, 0.0, 0.0, 0.0)
        self.update()

    def set_textures(self, scalar_texture, color_texture) -> None:
        if scalar_texture is self._scalar_texture and color_texture is self._color_texture:
            return
        self._scalar_texture = scalar_texture
        self._color_texture = color_texture
        self.update()

    def set_mipmap_page(self, page) -> None:
        """Atlas page whose textures this visual keeps mipmapped at draw time."""

        self._mipmap_page = page

    def set_levels(self, levels) -> bool:
        levels = _normalize_levels(levels, self._levels)
        if levels == self._levels:
            return False
        self._levels = levels
        self.update()
        return True

    def set_shader_mapping(self, mapping) -> bool:
        mapping_key = _visual_shader_mapping_key(mapping)
        if mapping_key == self._shader_mapping_key:
            return False
        scale_mode, symlog_constant, component_mode, phase_default, lut_key = mapping_key
        lut = _normalized_lut(getattr(mapping, "lut_data", None), phase_default=bool(phase_default))
        self._shader_mapping_key = mapping_key
        self._scale_mode = scale_mode
        self._symlog_constant = symlog_constant
        self._component_mode = component_mode
        self._lut_default_phase = phase_default
        self._set_lut_texture(lut, key=lut_key)
        self.update()
        return True

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._scalar_texture is None or self._color_texture is None:
            return False
        if self._lut_texture is None:
            self._set_lut_texture(None, phase_default=self._lut_default_phase)
        program = view.view_program
        program["a_position"] = self._vertices
        program["a_texcoord"] = self._texcoords
        program["a_mode"] = self._modes
        program["u_scalar_texture"] = self._scalar_texture
        program["u_color_texture"] = self._color_texture
        program["u_lut_texture"] = self._lut_texture
        program["u_levels"] = tuple(float(value) for value in self._levels)
        program["u_scale_mode"] = float(self._scale_mode)
        program["u_symlog_constant"] = float(self._symlog_constant)
        program["u_component_mode"] = float(self._component_mode)
        self._refresh_mipmaps()
        return True

    def _refresh_mipmaps(self) -> None:
        """Regenerate atlas mipmaps GPU-side when uploads dirtied the page.

        gloo has no mipmap support, so this runs raw GL with the context
        current (ADR 0050 atlas mipmaps): flush pending GLIR commands so the
        regen sees this commit's uploads, generate the clamped mip chain, and
        switch minification to NEAREST_MIPMAP_LINEAR — nearest within a mip
        (atlas neighbors never bleed), linear between mips (level transitions
        stay smooth).  Magnification stays nearest.  Zero CPU: the driver
        reduces on the GPU.  Any GL error disables mipmaps for the page
        rather than risking the draw.
        """

        page = self._mipmap_page
        if page is None or int(getattr(page, "mipmap_levels", 0) or 0) <= 0:
            return
        if not getattr(page, "mipmap_dirty", False) and getattr(page, "mipmap_ready", False):
            return
        try:
            from vispy.gloo import gl
            from vispy.gloo.context import get_current_canvas

            canvas = get_current_canvas()
            if canvas is None:
                return
            context = canvas.context
            # Make this commit's queued texture uploads visible to the regen.
            context.flush_commands()
            parser = context.shared.parser
            updates = 0
            for texture, is_atlas in (
                (self._scalar_texture, bool(page.scalar_is_atlas)),
                (self._color_texture, bool(page.color_is_atlas)),
            ):
                if not is_atlas or texture is None:
                    continue
                try:
                    handle = int(parser.get_object(texture.id).handle)
                except Exception:
                    handle = 0
                if handle <= 0:
                    # Texture not realized yet (first frame): stays dirty and
                    # is retried on the next draw.
                    return
                gl.glBindTexture(gl.GL_TEXTURE_2D, handle)
                gl.glGetError()  # clear stale error state
                gl.glTexParameteri(gl.GL_TEXTURE_2D, _GL_TEXTURE_MAX_LEVEL, int(page.mipmap_levels))
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                if gl.glGetError():
                    # Incomplete mip chain with a mipmap filter would draw
                    # black: keep plain nearest and disable for this page.
                    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                    page.mipmap_levels = 0
                    page.mipmap_ready = False
                    page.mipmap_dirty = False
                    return
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, _GL_NEAREST_MIPMAP_LINEAR)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                updates += 1
        except Exception:
            # Mipmaps are a display polish; a failure must never break the
            # draw.  Disable for this page and carry on.
            page.mipmap_levels = 0
            page.mipmap_ready = False
            page.mipmap_dirty = False
            return
        if updates:
            page.mipmap_dirty = False
            page.mipmap_ready = True
            page.mipmap_updates = int(getattr(page, "mipmap_updates", 0) or 0) + updates

    def _set_lut_texture(self, lut_data, *, key=None, phase_default: bool | None = None) -> bool:
        if phase_default is None:
            phase_default = bool(getattr(self, "_lut_default_phase", False))
        lut = _normalized_lut(lut_data, phase_default=phase_default)
        key = _array_content_key(lut) if key is None else key
        if key == self._lut_key and self._lut_texture is not None:
            return False
        lut_texture_data = np.ascontiguousarray(lut.reshape((1, lut.shape[0], 3)))
        from vispy import gloo

        if self._lut_texture is None or tuple(getattr(self._lut_texture, "shape", ())) != tuple(lut_texture_data.shape):
            self._lut_texture = gloo.Texture2D(
                lut_texture_data,
                format="rgb",
                internalformat="rgb8",
                interpolation="linear",
                wrapping="clamp_to_edge",
            )
        else:
            self._lut_texture.set_data(lut_texture_data, copy=False)
        self._lut_key = key
        return True

    def _bounds(self, axis, view):
        self._last_bounds_view = view
        if axis == 0:
            return self._bounds_xy[0], self._bounds_xy[1]
        if axis == 1:
            return self._bounds_xy[2], self._bounds_xy[3]
        return (0.0, 0.0)


def create_gpu_montage_layer(*, scene, visuals, gloo, transforms, parent, limits: GpuDeviceLimits | None = None) -> GpuMontageLayer:
    return GpuMontageLayer(scene=scene, visuals=visuals, gloo=gloo, transforms=transforms, parent=parent, limits=limits)


def query_gpu_device_limits(gloo) -> GpuDeviceLimits:
    try:
        gl = getattr(gloo, "gl", None)
        if gl is None:
            from vispy import gloo as vispy_gloo

            gl = getattr(vispy_gloo, "gl", None)
        if gl is None:
            raise RuntimeError("VisPy GL module unavailable")

        def get_integer(name: str, fallback: int = 0) -> int:
            value = gl.glGetParameter(getattr(gl, name))
            try:
                return int(value)
            except Exception:
                return int(np.asarray(value).ravel()[0])

        def get_string(name: str) -> str:
            # VisPy's gloo.gl exposes string queries through glGetParameter;
            # plain glGetString only exists on some older GL wrappers.
            getter = getattr(gl, "glGetString", None) or gl.glGetParameter
            value = getter(getattr(gl, name))
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value or "")

        max_texture = get_integer("GL_MAX_TEXTURE_SIZE", 4096)
        if int(max_texture) <= 0:
            # GL queries answer 0 without a current context; that is not a
            # real device limit, so report the conservative fallback instead.
            raise RuntimeError("OpenGL reported GL_MAX_TEXTURE_SIZE == 0 (no current context)")
        max_units = get_integer("GL_MAX_TEXTURE_IMAGE_UNITS", 0)
        return GpuDeviceLimits(
            max_texture_size=max(1, int(max_texture)),
            max_texture_image_units=max(0, int(max_units)),
            vendor=get_string("GL_VENDOR"),
            renderer=get_string("GL_RENDERER"),
            version=get_string("GL_VERSION"),
            source="opengl",
        )
    except Exception as exc:
        return GpuDeviceLimits(
            max_texture_size=4096,
            source="fallback",
            warnings=(f"OpenGL device limit query failed: {exc}",),
        )


def _payload_textures(payload: DisplayTilePayload, *, tile_shape: tuple[int, int], rgb_already_windowed: bool):
    """Compatibility helper returning both texture planes for tests/tools."""

    scalar, color = _payload_texture_data(
        payload,
        tile_shape=tile_shape,
        rgb_already_windowed=rgb_already_windowed,
        need_scalar=True,
        need_color=True,
    )
    return scalar, color


def _payload_texture_data(
    payload: DisplayTilePayload,
    *,
    tile_shape: tuple[int, int],
    rgb_already_windowed: bool,
    need_scalar: bool,
    need_color: bool,
):
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    texture = np.asarray(payload.texture_data if payload.texture_data is not None else payload.image)
    texture_kind = _payload_texture_kind(payload)
    if texture_kind == TexturePlaneKind.COMPLEX_RG32F:
        scalar = _fit_complex_rg(_complex_rg_texture(texture), (tile_h, tile_w)) if need_scalar else None
        color = np.zeros((tile_h, tile_w, 3), dtype=np.uint8) if need_color else None
        return scalar, color
    image = texture
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        color = _fit_color(image[..., :3], (tile_h, tile_w)) if need_color else None
        scalar = None
        if need_scalar:
            if rgb_already_windowed:
                scalar = np.ones((tile_h, tile_w), dtype=np.float32)
            else:
                source = payload.histogram_data
                if source is None:
                    source = _luminance(image[..., :3])
                scalar = _fit_scalar(source, (tile_h, tile_w))
        return scalar, color
    scalar = _fit_scalar(image, (tile_h, tile_w)) if need_scalar else None
    color = np.zeros((tile_h, tile_w, 3), dtype=np.uint8) if need_color else None
    return scalar, color


def _prepare_payload_texture_data(
    payload: DisplayTilePayload,
    *,
    tile_shape: tuple[int, int],
    rgb_already_windowed: bool,
    need_scalar: bool,
    need_color: bool,
):
    start = perf_counter()
    scalar, color = _payload_texture_data(
        payload,
        tile_shape=tile_shape,
        rgb_already_windowed=rgb_already_windowed,
        need_scalar=need_scalar,
        need_color=need_color,
    )
    return scalar, color, (perf_counter() - start) * 1000.0


def _upload_texture_plane(texture, data: np.ndarray, *, offset: tuple[int, int], copy: bool) -> float:
    start = perf_counter()
    texture.set_data(data, offset=offset, copy=copy)
    return (perf_counter() - start) * 1000.0


def _fit_scalar(data, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if tuple(arr.shape[:2]) == tuple(shape) and arr.ndim == 2 and arr.flags.c_contiguous:
        # Gloo may retain the array in its command queue.  Reusing the
        # immutable payload plane avoids another full tile copy before upload.
        return arr
    out = np.zeros(shape, dtype=np.float32)
    height = min(shape[0], int(arr.shape[0]))
    width = min(shape[1], int(arr.shape[1]))
    if height > 0 and width > 0:
        out[:height, :width] = arr[:height, :width]
    return out


def _fit_color(data, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(data)
    if (
        arr.dtype == np.uint8
        and arr.ndim == 3
        and arr.shape[-1] == 3
        and tuple(arr.shape[:2]) == tuple(shape)
        and arr.flags.c_contiguous
    ):
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0.0, 255.0)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    if (
        arr.ndim == 3
        and arr.shape[-1] == 3
        and tuple(arr.shape[:2]) == tuple(shape)
        and arr.flags.c_contiguous
    ):
        return arr
    out = np.zeros(shape + (3,), dtype=np.uint8)
    height = min(shape[0], int(arr.shape[0]))
    width = min(shape[1], int(arr.shape[1]))
    if height > 0 and width > 0:
        out[:height, :width, :] = arr[:height, :width, :3]
    return out


def _fit_complex_rg(data, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if (
        arr.ndim == 3
        and arr.shape[-1] == 2
        and tuple(arr.shape[:2]) == tuple(shape)
        and arr.flags.c_contiguous
    ):
        return arr
    out = np.zeros(shape + (2,), dtype=np.float32)
    height = min(shape[0], int(arr.shape[0]))
    width = min(shape[1], int(arr.shape[1]))
    if height > 0 and width > 0:
        out[:height, :width, :] = arr[:height, :width, :2]
    return out


def _complex_rg_texture(data) -> np.ndarray:
    arr = np.asarray(data)
    if np.iscomplexobj(arr):
        if arr.dtype != np.complex64 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr.astype(np.complex64, copy=False))
        return arr.view(np.float32).reshape(arr.shape + (2,))
    packed = np.asarray(arr, dtype=np.float32)
    if packed.ndim < 3 or packed.shape[-1] != 2:
        raise ValueError("complex RG32F texture data must be complex or have trailing size 2")
    return np.ascontiguousarray(packed)


def _upload_copy_required(staging: np.ndarray, payload: DisplayTilePayload, *, force: bool = False) -> bool:
    """Ask VisPy to retain temporary staging planes before queued GL upload."""

    if force:
        return True
    try:
        staging = np.asarray(staging)
        for source in (
            getattr(payload, "texture_data", None),
            getattr(payload, "semantic_data", None),
            getattr(payload, "image", None),
            getattr(payload, "histogram_data", None),
            getattr(payload, "semantic_histogram_data", None),
        ):
            if source is not None and np.shares_memory(staging, np.asarray(source)):
                return False
    except Exception:
        return True
    return True


def _quad_buffers(layout, payloads, uvs, *, rgb_already_windowed: bool):
    vertices = []
    texcoords = []
    modes = []
    for tile_number, payload in sorted((int(key), value) for key, value in dict(payloads).items()):
        region = layout.get(int(tile_number))
        if region is None:
            continue
        uv = uvs.get(int(tile_number))
        if uv is None:
            continue
        x0 = float(region.x)
        y0 = float(region.y)
        x1 = x0 + float(region.width)
        y1 = y0 + float(region.height)
        u0, v0, u1, v1 = uv
        vertices.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)))
        texcoords.extend(((u0, v0), (u1, v0), (u1, v1), (u0, v0), (u1, v1), (u0, v1)))
        mode = float(_payload_mode(payload, rgb_already_windowed=rgb_already_windowed))
        modes.extend((mode,) * 6)
    return (
        np.asarray(vertices, dtype=np.float32).reshape((-1, 2)),
        np.asarray(texcoords, dtype=np.float32).reshape((-1, 2)),
        np.asarray(modes, dtype=np.float32).reshape((-1,)),
    )


def _page_geometry_key(payloads, pool, page, layout, *, rgb_already_windowed: bool) -> tuple[object, ...]:
    return (
        tuple(
            (
                int(key),
                int(pool.tile_slots[int(key)][1]),
                int(layout[int(key)].x),
                int(layout[int(key)].y),
                int(layout[int(key)].width),
                int(layout[int(key)].height),
                _payload_mode(payload, rgb_already_windowed=rgb_already_windowed),
                _payload_gutter(payload),
            )
            for key, payload in sorted(dict(payloads or {}).items())
            if int(key) in pool.tile_slots and int(key) in layout
        ),
        id(page),
        tuple(int(value) for value in page.atlas_shape),
    )


class PayloadBatchQueue:
    """Ordered speculative-upload queue with bounded batch removal.

    Repeatedly splitting ``dict`` objects for warm residency makes the queue
    O(n²) over a large near-tile set.  This queue materializes the upload order
    once and removes accepted work from the left in O(batch_size).
    """

    def __init__(self, payloads=()) -> None:
        items = () if payloads is None else payloads.items()
        self._pending = deque((int(tile), payload) for tile, payload in items)

    def __bool__(self) -> bool:
        return bool(self._pending)

    def __len__(self) -> int:
        return len(self._pending)

    def take(
        self,
        *,
        max_items: int = 4,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> dict[int, DisplayTilePayload]:
        max_items = max(1, int(max_items))
        max_bytes = max(1, int(max_bytes))
        batch: dict[int, DisplayTilePayload] = {}
        batch_bytes = 0
        while self._pending:
            tile, payload = self._pending[0]
            payload_bytes = _payload_upload_nbytes(payload)
            fits_items = len(batch) < max_items
            fits_bytes = not batch or batch_bytes + payload_bytes <= max_bytes
            if not (fits_items and fits_bytes):
                break
            self._pending.popleft()
            batch[int(tile)] = payload
            batch_bytes += payload_bytes
        return batch

    def remaining_payloads(self) -> dict[int, DisplayTilePayload]:
        return {int(tile): payload for tile, payload in self._pending}


def take_payload_batch(
    payloads,
    *,
    max_items: int = 4,
    max_bytes: int = 8 * 1024 * 1024,
) -> tuple[dict[int, DisplayTilePayload], dict[int, DisplayTilePayload]]:
    """Split speculative upload work into a bounded UI-event-loop batch.

    Compatibility wrapper for callers that still expect ``(batch, remaining)``.
    Production warm-residency code should keep a ``PayloadBatchQueue`` and avoid
    rebuilding the remaining mapping on every timer tick.
    """

    queue = PayloadBatchQueue(payloads)
    batch = queue.take(max_items=max_items, max_bytes=max_bytes)
    return batch, queue.remaining_payloads()


def _payload_upload_nbytes(payload: DisplayTilePayload) -> int:
    texture = payload.texture_data if payload.texture_data is not None else payload.image
    total = int(np.asarray(texture).nbytes)
    histogram = getattr(payload, "histogram_data", None)
    if histogram is not None and histogram is not texture:
        total += int(np.asarray(histogram).nbytes)
    return max(1, total)


def _normalized_lut(lut_data, *, phase_default: bool = False) -> np.ndarray:
    return normalize_lut_rgb(lut_data, phase_default=phase_default)


def _array_content_key(array: np.ndarray) -> tuple[object, ...]:
    array = np.asarray(array)
    return (tuple(int(value) for value in array.shape), array.dtype.str, array.tobytes())


def _luminance(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


def _normalize_levels(levels, fallback):
    if levels is None:
        return tuple(float(value) for value in fallback)
    low, high = levels
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return tuple(float(value) for value in fallback)
    return low, high


def _atlas_storage_mode(payload_items, *, rgb_already_windowed: bool) -> str:
    needs_scalar = False
    needs_color = False
    needs_complex = False
    for _tile_number, payload in payload_items:
        kind = _payload_texture_kind(payload)
        if kind == TexturePlaneKind.COMPLEX_RG32F:
            needs_complex = True
            continue
        image = np.asarray(payload.texture_data if payload.texture_data is not None else payload.image)
        is_color = image.ndim == 3 and image.shape[-1] in (3, 4)
        if is_color:
            needs_color = True
            needs_scalar = needs_scalar or not bool(rgb_already_windowed)
        else:
            needs_scalar = True
    if needs_complex:
        return "complex"
    if needs_scalar and needs_color:
        return "scalar_color"
    if needs_color:
        return "color"
    return "scalar"


def _normalize_storage_mode(mode: str) -> str:
    mode = str(mode or "scalar_color")
    if mode not in {"scalar", "color", "scalar_color", "complex"}:
        raise ValueError(f"unknown atlas storage mode: {mode}")
    return mode


def _storage_mode_bytes_per_pixel(mode: str) -> int:
    mode = _normalize_storage_mode(mode)
    if mode == "scalar":
        return 4
    if mode == "color":
        return 3
    if mode == "complex":
        return 8
    return 7


def _atlas_grid(*, tile_shape: tuple[int, int], capacity: int, max_texture_size: int) -> tuple[int, int]:
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    capacity = max(1, int(capacity))
    max_columns = max_texture_size // tile_w
    max_rows = max_texture_size // tile_h
    # Aim for an approximately square atlas in pixels, not in slot count.
    ideal_columns = int(np.ceil(np.sqrt(capacity * tile_h / tile_w)))
    columns = max(1, min(max_columns, ideal_columns))
    rows = int(np.ceil(capacity / columns))
    if rows > max_rows:
        columns = int(np.ceil(capacity / max_rows))
        rows = int(np.ceil(capacity / columns))
    if columns > max_columns or rows > max_rows:
        raise AtlasCapacityError(
            f"atlas grid {columns}x{rows} exceeds texture limit {max_texture_size} for tile shape {tile_shape}"
        )
    return columns, rows


def _atlas_reserve_count(geometry, *, minimum: int, frame_plan=None) -> int:
    """Reserve slots for the complete non-skipped visible montage set.

    Progressive sessions begin with most tiles in ``unloaded`` state.  Counting
    only loaded/loading tiles makes the atlas grow repeatedly as scheduling
    advances, which discards otherwise reusable GPU residency.  Geometry is the
    authoritative visibility plan, so reserve every planned slot except tiles
    explicitly marked skipped.
    """

    indices = (
        tuple(range(planned_tile_count(geometry, frame_plan=frame_plan, minimum=0)))
        if frame_plan is not None
        else tuple(getattr(getattr(geometry, "montage", None), "indices", ()) or ())
    )
    states = tuple(getattr(geometry, "montage_tile_states", ()) or ())
    if not indices and not states:
        return max(1, int(minimum))

    planned = max(len(indices), len(states))
    skipped = 0
    for state in states:
        value = str(getattr(state, "value", state))
        if value == "skipped":
            skipped += 1
    expected = max(0, planned - skipped)
    return max(1, int(minimum), int(expected))


def _layout_tile_shape(layout) -> tuple[int, int]:
    if not layout:
        return (1, 1)
    heights = {int(region.height) for region in layout.values()}
    widths = {int(region.width) for region in layout.values()}
    return (max(1, max(heights)), max(1, max(widths)))


def _payload_mode(payload: DisplayTilePayload, *, rgb_already_windowed: bool) -> int:
    if _payload_texture_kind(payload) == TexturePlaneKind.COMPLEX_RG32F:
        mapping = getattr(payload, "shader_mapping", None)
        display_mode = getattr(getattr(mapping, "display_mode", None), "value", getattr(mapping, "display_mode", None))
        return 4 if display_mode == ShaderDisplayMode.PHASE_COLOR.value else 3
    image = np.asarray(payload.texture_data if payload.texture_data is not None else payload.image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return 2 if rgb_already_windowed else 1
    return 0


def _mapping_identity_key(mapping):
    return None if mapping is None else getattr(mapping, "identity_key", mapping)


def _visual_shader_mapping_key(mapping) -> tuple[object, ...]:
    scale_mode = _shader_scale_uniform(getattr(mapping, "scale", None))
    symlog_constant = float(getattr(mapping, "symlog_constant", 0.0) or 0.0)
    component_mode = shader_component_uniform(getattr(mapping, "component", None))
    display_mode = getattr(getattr(mapping, "display_mode", None), "value", getattr(mapping, "display_mode", None))
    phase_default = bool(display_mode == ShaderDisplayMode.PHASE_COLOR.value)
    lut = _normalized_lut(getattr(mapping, "lut_data", None), phase_default=phase_default)
    lut_key = _array_content_key(lut)
    return (float(scale_mode), float(symlog_constant), float(component_mode), phase_default, lut_key)


def _shader_scale_uniform(scale) -> float:
    if scale is None:
        return 0.0
    value = scale.value if isinstance(scale, ShaderScale) else getattr(scale, "value", scale)
    if value == ShaderScale.LOG.value:
        return 1.0
    if value == ShaderScale.SYMLOG.value:
        return 2.0
    return 0.0


def _payload_supported_by_storage_mode(payload: DisplayTilePayload, storage_mode: str | None, *, rgb_already_windowed: bool) -> bool:
    if not _payload_texture_matches_kind(payload):
        return False
    mode = _normalize_storage_mode(storage_mode or "scalar")
    if _payload_texture_kind(payload) == TexturePlaneKind.COMPLEX_RG32F:
        return mode == "complex"
    image = np.asarray(payload.texture_data if payload.texture_data is not None else payload.image)
    is_color = image.ndim == 3 and image.shape[-1] in (3, 4)
    if not is_color:
        return mode in {"scalar", "scalar_color"}
    if rgb_already_windowed:
        return mode in {"color", "scalar_color"}
    return mode == "scalar_color"


def _payload_texture_kind(payload: DisplayTilePayload) -> TexturePlaneKind:
    kind = getattr(payload, "texture_kind", None)
    if kind is not None:
        return kind if isinstance(kind, TexturePlaneKind) else TexturePlaneKind(getattr(kind, "value", kind))
    texture = np.asarray(getattr(payload, "texture_data", None) if getattr(payload, "texture_data", None) is not None else payload.image)
    if np.iscomplexobj(texture) or (texture.ndim == 3 and texture.shape[-1] == 2):
        return TexturePlaneKind.COMPLEX_RG32F
    if texture.ndim == 3 and texture.shape[-1] in (3, 4):
        return TexturePlaneKind.RGB8
    return TexturePlaneKind.SCALAR_R32F


def _payload_texture_matches_kind(payload: DisplayTilePayload) -> bool:
    kind = _payload_texture_kind(payload)
    texture = getattr(payload, "texture_data", None)
    if texture is None:
        texture = getattr(payload, "semantic_data", None)
    if texture is None:
        texture = getattr(payload, "image", None)
    if texture is None:
        return False
    image = np.asarray(texture)
    if image.ndim < 2:
        return False
    if kind == TexturePlaneKind.COMPLEX_RG32F:
        return np.iscomplexobj(image) or (image.ndim == 3 and image.shape[-1] == 2)
    if kind == TexturePlaneKind.RGB8:
        return image.ndim == 3 and image.shape[-1] in (3, 4)
    return image.ndim == 2 and not np.iscomplexobj(image)


def _payload_gutter(payload: DisplayTilePayload) -> int:
    lod = getattr(payload, "lod", None)
    return 0 if lod is None else int(getattr(lod, "gutter", 0) or 0)


def _max_payload_lod_level(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((int(getattr(getattr(payload, "lod", None), "level", 0) or 0) for payload in dict(payloads or {}).values()), default=0)


def _max_payload_lod_factor(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((int(getattr(getattr(payload, "lod", None), "factor", 1) or 1) for payload in dict(payloads or {}).values()), default=1)


def _max_payload_gutter(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((_payload_gutter(payload) for payload in dict(payloads or {}).values()), default=0)


def _payload_class_shape(payload: DisplayTilePayload) -> tuple[int, int]:
    """Slot shape class for a payload: its actual texture shape (ADR 0050)."""

    texture = payload.texture_data if payload.texture_data is not None else payload.image
    shape = np.shape(texture)
    return (int(shape[0]), int(shape[1]))


def _atlas_base_tile_shape_for_payloads(payloads: dict[int, DisplayTilePayload], *, fallback: tuple[int, int]) -> tuple[int, int]:
    """Native (level 0) slot shape for the atlas base class.

    Reduced-level payloads report their native source shape, so a commit whose
    active set is entirely reduced does not shrink the base class and thereby
    discard resident native slots.
    """

    shapes = []
    for payload in dict(payloads or {}).values():
        lod = getattr(payload, "lod", None)
        if lod is not None and int(getattr(lod, "level", 0) or 0) > 0:
            shapes.append(tuple(int(value) for value in lod.source_shape))
        else:
            shapes.append(_payload_class_shape(payload))
    if not shapes:
        return (max(1, int(fallback[0])), max(1, int(fallback[1])))
    return (max(1, max(shape[0] for shape in shapes)), max(1, max(shape[1] for shape in shapes)))


def _atlas_tile_shape_for_payloads(payloads: dict[int, DisplayTilePayload], *, fallback: tuple[int, int]) -> tuple[int, int]:
    shapes = []
    for payload in dict(payloads or {}).values():
        texture = np.asarray(payload.texture_data if payload.texture_data is not None else payload.image)
        shapes.append(tuple(int(value) for value in texture.shape[:2]))
    if not shapes:
        return (max(1, int(fallback[0])), max(1, int(fallback[1])))
    return (max(1, max(shape[0] for shape in shapes)), max(1, max(shape[1] for shape in shapes)))


def _resident_key(payload: DisplayTilePayload) -> object:
    """Return the GPU residency identity for a semantic tile payload.

    Tile numbers describe where content is drawn in the current montage plan.
    They are not content identities: index scrolling can draw the same source
    through a different tile number.  Residency is therefore keyed by
    ``source_id`` whenever possible, with an object-id fallback for rare
    unhashable identities.

    The key is a pure function of immutable payload identity fields, and a
    commit visits every active payload several times, so it is memoized on the
    payload instance.
    """

    cached = payload.__dict__.get("_vispy_resident_key")
    if cached is not None:
        return cached
    texture = np.asarray(payload.texture_data if payload.texture_data is not None else payload.image)
    lod = getattr(payload, "lod", None)
    key = (
        _source_resident_key(payload.source_id),
        "texture_kind",
        None if payload.texture_kind is None else getattr(payload.texture_kind, "value", payload.texture_kind),
        "texture_shape",
        tuple(int(value) for value in texture.shape),
        "texture_dtype",
        str(texture.dtype),
        "lod",
        None if lod is None else (int(lod.factor), int(lod.level), int(lod.gutter)),
    )
    try:
        object.__setattr__(payload, "_vispy_resident_key", key)
    except Exception:
        pass
    return key


def _resident_key_lod(resident_key: object) -> object:
    """Return the LOD component of a residency key (None for foreign keys)."""

    if isinstance(resident_key, tuple) and resident_key and resident_key[-2] == "lod":
        return resident_key[-1]
    return None


def _lod_invariant_source_id(source_id: object) -> object:
    """Strip texture-kind/LOD/content decorations from a payload source id.

    Levels of the same tile content share this identity, so it links a
    superseded residency class back to the active tile that retains it as an
    adjacent level.
    """

    base = _base_texture_source_id(source_id)
    if isinstance(base, tuple) and "lod" in base:
        return base[: base.index("lod")]
    return base


def _source_resident_key(source_id: object) -> object:
    normalized = _display_tile_texture_source_id(source_id)
    if normalized is not None:
        return normalized
    try:
        hash(source_id)
    except Exception:
        return ("source-object", id(source_id))
    return ("source", source_id)


def _base_texture_source_id(source_id: object) -> object:
    normalized = _display_tile_texture_source_id(source_id)
    if normalized is not None:
        return normalized
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    if isinstance(source_id, tuple) and "texture_kind" in source_id:
        return source_id[: source_id.index("texture_kind")]
    return source_id


def _display_tile_texture_source_id(source_id: object) -> object | None:
    display_key = None
    if isinstance(source_id, tuple) and len(source_id) >= 8 and source_id[0] == "display_tile":
        display_key = source_id
    elif (
        isinstance(source_id, tuple)
        and source_id
        and isinstance(source_id[0], tuple)
        and len(source_id[0]) >= 8
        and source_id[0][0] == "display_tile"
    ):
        display_key = source_id[0]
    if display_key is None:
        return None
    return (
        "display_tile_texture",
        display_key[1],
        display_key[5],
        display_key[6],
        display_key[7],
    )
