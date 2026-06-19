"""Batched VisPy montage tile rendering.

The renderer keeps tile identity, GPU residency, and draw visibility separate.
Presentation commits may change the set of drawn tiles without invalidating
unchanged resident texture slots.  Level-only changes update uniforms only.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.display.lod import inner_uv_for_gutter
from arrayscope.display.shader_mapping import (
    ShaderDisplayMode,
    ShaderScale,
    TexturePlaneKind,
    default_phase_lut,
    pack_texture_data,
    shader_component_uniform,
)
from arrayscope.display.model.frame import DisplayTilePayload

try:
    from vispy.visuals import Visual
except Exception:  # pragma: no cover - optional dependency import path
    Visual = object


@dataclass(frozen=True)
class GpuMontageLayerStats:
    visible_items: int = 0
    resident_items: int = 0
    atlas_capacity: int = 0
    atlas_rebuilds: int = 0
    atlas_evictions: int = 0
    texture_uploads: int = 0
    texture_upload_bytes: int = 0
    vertex_uploads: int = 0
    items_updated: int = 0
    items_skipped: int = 0
    level_updates: int = 0
    estimated_gpu_bytes: int = 0
    cpu_shadow_bytes: int = 0
    upload_ms: float = 0.0
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
        self.source_ids: dict[object, object] = {}
        self.last_used: dict[object, int] = {}
        self.active_resident_keys: set[object] = set()
        self.serial = 0
        self.rebuild_count = 0
        self.eviction_count = 0
        self.evicted_near_count = 0
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
        if not shape_changed and not mode_changed and requested <= self.capacity:
            return False

        self.tile_shape = (tile_h, tile_w)
        if shape_changed or mode_changed:
            self.pages.clear()
            self.resident_slots.clear()
            self.tile_slots.clear()
            self.source_ids.clear()
            self.last_used.clear()
            self.active_resident_keys.clear()
        self.storage_mode = storage_mode
        while self.capacity < requested:
            remaining = requested - self.capacity
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
    ) -> tuple[dict[int, tuple[float, float, float, float]], GpuMontageLayerStats]:
        start = perf_counter()
        payload_items = tuple(sorted((int(key), value) for key, value in payloads.items()))
        storage_mode = (
            _atlas_storage_mode(payload_items, rgb_already_windowed=rgb_already_windowed)
            if payload_items
            else (self.storage_mode or "scalar")
        )
        tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
        requested_capacity = self.requested_capacity(
            active_count=len(payload_items),
            reserve_count=max(len(payload_items), int(reserve_count or 0)),
            storage_mode=storage_mode,
            tile_shape=tile_shape,
            budget_bytes=budget_bytes,
        )
        rebuilt = self.ensure_layout(
            tile_shape=tile_shape,
            count=requested_capacity,
            storage_mode=storage_mode,
            budget_bytes=budget_bytes,
        )
        dirty_all = dirty_tiles is None
        dirty = set() if dirty_all else {int(tile) for tile in dirty_tiles}
        active = {int(tile_number) for tile_number, _payload in payload_items}
        active_keys = {_resident_key(payload) for _tile_number, payload in payload_items}
        self.active_resident_keys = set(active_keys)
        near = {int(tile) for tile in tuple(near_tiles or ())}
        near_keys = self._near_resident_keys(near_tile_source_ids)
        near_keys.update(_resident_key(payload) for tile_number, payload in payload_items if int(tile_number) in near)
        uvs: dict[int, tuple[float, float, float, float]] = {}
        active_tile_slots: dict[int, tuple[int, int]] = {}
        uploads = 0
        upload_bytes = 0
        complex_uploads = 0
        updated = 0
        skipped = 0
        evictions_before = self.eviction_count
        evicted_near_before = self.evicted_near_count
        tile_h, tile_w = self.tile_shape or tile_shape

        for tile_number, payload in payload_items:
            resident_key = _resident_key(payload)
            page_index, slot, newly_assigned = self._slot_for(resident_key, active_keys=active_keys, near_keys=near_keys)
            active_tile_slots[int(tile_number)] = (int(page_index), int(slot))
            page = self.pages[int(page_index)]
            self._touch(resident_key)
            source_changed = self.source_ids.get(resident_key) != payload.source_id
            should_upload = bool(dirty_all or newly_assigned or source_changed or tile_number in dirty)
            uvs[tile_number] = page.uv_for_slot_with_gutter(slot, gutter=_payload_gutter(payload))
            if not should_upload:
                skipped += 1
                continue

            scalar, color = _payload_texture_data(
                payload,
                tile_shape=(tile_h, tile_w),
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=self.scalar_is_atlas,
                need_color=self.color_is_atlas,
            )
            y0, x0 = page.offset_for_slot(slot)
            if scalar is not None:
                page.scalar_texture.set_data(scalar, offset=(int(y0), int(x0)), copy=False)
                uploads += 1
                upload_bytes += int(scalar.nbytes)
                if page.complex_is_atlas:
                    complex_uploads += 1
            if color is not None:
                page.color_texture.set_data(color, offset=(int(y0), int(x0)), copy=False)
                uploads += 1
                upload_bytes += int(color.nbytes)
            self.source_ids[resident_key] = payload.source_id
            updated += 1

        self.tile_slots = active_tile_slots
        elapsed = (perf_counter() - start) * 1000.0 if updated or rebuilt else 0.0
        return uvs, GpuMontageLayerStats(
            visible_items=len(payload_items),
            resident_items=self.resident_count,
            atlas_capacity=self.capacity,
            atlas_rebuilds=int(rebuilt),
            atlas_evictions=self.eviction_count - evictions_before,
            texture_uploads=uploads,
            texture_upload_bytes=upload_bytes,
            items_updated=updated,
            items_skipped=skipped,
            estimated_gpu_bytes=self.estimated_gpu_bytes,
            cpu_shadow_bytes=self.cpu_shadow_bytes,
            upload_ms=elapsed,
            page_count=len(self.pages),
            active_pages=len({self.tile_slots[int(tile)][0] for tile in active if int(tile) in self.tile_slots}),
            device_max_texture_size=self.max_texture_size,
            budget_bytes=self.budget_bytes,
            near_resident_items=len(near_keys.intersection(self.source_ids)),
            warm_resident_items=max(0, self.resident_count - len(active)),
            evicted_near_items=self.evicted_near_count - evicted_near_before,
            lod_level=_max_payload_lod_level(payloads),
            lod_factor=_max_payload_lod_factor(payloads),
            source_texels_per_pixel=float(_max_payload_lod_factor(payloads)),
            gutter_pixels=_max_payload_gutter(payloads),
            mipmap_updates=0,
            mipmap_available=False,
            complex_texture_uploads=complex_uploads,
        )

    def warm_payloads(
        self,
        payloads: dict[int, DisplayTilePayload],
        *,
        tile_shape: tuple[int, int],
        rgb_already_windowed: bool,
        near_tile_source_ids: dict[int, object] | None = None,
        budget_bytes: int | None = None,
    ) -> GpuMontageLayerStats:
        start = perf_counter()
        if not payloads:
            return GpuMontageLayerStats(
                resident_items=self.resident_count,
                atlas_capacity=self.capacity,
                estimated_gpu_bytes=self.estimated_gpu_bytes,
                cpu_shadow_bytes=self.cpu_shadow_bytes,
                page_count=len(self.pages),
                device_max_texture_size=self.max_texture_size,
                budget_bytes=self.budget_bytes,
                warm_resident_items=max(0, self.resident_count - len(self.active_resident_keys)),
            )
        payload_items = tuple(sorted((int(key), value) for key, value in payloads.items()))
        requested_mode = _atlas_storage_mode(payload_items, rgb_already_windowed=rgb_already_windowed)
        if self.storage_mode is not None and self.storage_mode != requested_mode:
            # Warm work must never replace the active atlas layout.  A later
            # visible commit can deliberately switch storage modes.
            return GpuMontageLayerStats(
                resident_items=self.resident_count,
                atlas_capacity=self.capacity,
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
        deferred = 0
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
                deferred += 1
                continue
            if resident_key not in self.resident_slots:
                new_warm_budget -= 1
            page_index, slot, _newly_assigned = self._slot_for(
                resident_key,
                active_keys=set(self.active_resident_keys),
                near_keys=near_keys,
            )
            page = self.pages[int(page_index)]
            scalar, color = _payload_texture_data(
                payload,
                tile_shape=(tile_h, tile_w),
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=page.scalar_is_atlas,
                need_color=page.color_is_atlas,
            )
            y0, x0 = page.offset_for_slot(slot)
            if scalar is not None:
                page.scalar_texture.set_data(scalar, offset=(int(y0), int(x0)), copy=False)
                uploads += 1
                upload_bytes += int(scalar.nbytes)
                if page.complex_is_atlas:
                    complex_uploads += 1
            if color is not None:
                page.color_texture.set_data(color, offset=(int(y0), int(x0)), copy=False)
                uploads += 1
                upload_bytes += int(color.nbytes)
            self.source_ids[resident_key] = payload.source_id
            self._touch(resident_key)
            updated += 1

        elapsed = (perf_counter() - start) * 1000.0 if updated else 0.0
        return GpuMontageLayerStats(
            resident_items=self.resident_count,
            atlas_capacity=self.capacity,
            atlas_evictions=self.eviction_count - evictions_before,
            texture_uploads=uploads,
            texture_upload_bytes=upload_bytes,
            items_updated=updated,
            items_skipped=skipped,
            estimated_gpu_bytes=self.estimated_gpu_bytes,
            cpu_shadow_bytes=self.cpu_shadow_bytes,
            upload_ms=elapsed,
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
                f"deferred {deferred} warm tiles because the residency budget is full"
                if deferred
                else ""
            ),
        )

    def _slot_for(self, resident_key: object, *, active_keys: set[object], near_keys: set[object]) -> tuple[int, int, bool]:
        current = self.resident_slots.get(resident_key)
        if current is not None:
            return int(current[0]), int(current[1]), False

        for page_index, page in enumerate(self.pages):
            try:
                slot = page.slot_owners.index(None)
            except ValueError:
                continue
            page.slot_owners[int(slot)] = resident_key
            self.resident_slots[resident_key] = (int(page_index), int(slot))
            return int(page_index), int(slot), True

        candidates = []
        for page_index, page in enumerate(self.pages):
            for slot, owner in enumerate(page.slot_owners):
                if owner is not None and owner not in active_keys:
                    candidates.append((0 if owner not in near_keys else 1, self.last_used.get(owner, -1), owner, int(page_index), int(slot)))
        if not candidates:
            raise AtlasCapacityError(
                f"atlas has {self.capacity} slots but {len(active_keys)} active tiles require residency"
            )
        priority, _last, victim, page_index, slot = min(candidates, key=lambda item: (item[0], item[1], repr(item[2])))
        del priority
        self.resident_slots.pop(victim, None)
        self.source_ids.pop(victim, None)
        self.last_used.pop(victim, None)
        self.eviction_count += 1
        if victim in near_keys:
            self.evicted_near_count += 1
        page = self.pages[int(page_index)]
        page.slot_owners[int(slot)] = resident_key
        self.resident_slots[resident_key] = (int(page_index), int(slot))
        return int(page_index), int(slot), True

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
        self._levels: tuple[float, float] = (0.0, 1.0)
        self._shader_mapping = None
        self._shader_mapping_key = None
        self._visible_items = 0
        self._last_stats = GpuMontageLayerStats()
        self._ensure_visual_count(1)

    @property
    def visual(self):
        self._ensure_visual_count(1)
        return self._visuals_by_page[0]

    @property
    def last_stats(self) -> GpuMontageLayerStats:
        return self._last_stats

    def clear(self) -> None:
        # Hiding a layer must not discard useful GPU residency.  A later
        # viewport/session can reuse the same source identities.
        for visual in self._visuals_by_page:
            visual.visible = False
        self._geometry_keys.clear()
        self._visible_items = 0

    def set_levels(self, levels) -> GpuMontageLayerStats:
        return self.set_presentation_uniforms(levels=levels)

    def set_shader_mapping(self, mapping) -> GpuMontageLayerStats:
        return self.set_presentation_uniforms(shader_mapping=mapping)

    def set_presentation_uniforms(self, *, levels=_UNSET, shader_mapping=_UNSET) -> GpuMontageLayerStats:
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
        self._last_stats = GpuMontageLayerStats(
            visible_items=self._visible_items,
            resident_items=self._pool.resident_count,
            atlas_capacity=self._pool.capacity,
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
    ) -> GpuMontageLayerStats:
        montage = getattr(geometry, "montage", None)
        if montage is None:
            self.clear()
            return GpuMontageLayerStats()
        payloads = {int(key): value for key, value in dict(payloads or {}).items()}
        reserve_count = max(
            _atlas_reserve_count(geometry, minimum=len(payloads)),
            len(tuple(getattr(tile_delta, "planned_tiles", ()) or ())),
        )
        near_tiles = tuple(getattr(tile_delta, "near_tiles", ()) or ())
        near_tile_source_ids = dict(getattr(tile_delta, "near_tile_source_ids", {}) or {})
        uvs, texture_stats = self._pool.update_payloads(
            payloads,
            tile_shape=_atlas_tile_shape_for_payloads(payloads, fallback=(int(montage.tile_height), int(montage.tile_width))),
            dirty_tiles=dirty_tiles,
            rgb_already_windowed=rgb_already_windowed,
            reserve_count=reserve_count,
            near_tiles=near_tiles,
            near_tile_source_ids=near_tile_source_ids,
            budget_bytes=tile_residency_budget_bytes,
        )
        self._ensure_visual_count(len(self._pool.pages))
        vertex_uploads = 0
        active_pages = set()
        page_payloads_by_index: list[dict[int, DisplayTilePayload]] = [
            {} for _page in self._pool.pages
        ]
        for tile, payload in payloads.items():
            page_index, _slot = self._pool.tile_slots.get(int(tile), (-1, -1))
            if 0 <= int(page_index) < len(page_payloads_by_index):
                page_payloads_by_index[int(page_index)][int(tile)] = payload
        for page_index, page in enumerate(self._pool.pages):
            page_payloads = page_payloads_by_index[page_index]
            visual = self._visuals_by_page[page_index]
            geometry_key = (
                tuple(
                    (
                        int(key),
                        int(self._pool.tile_slots[int(key)][1]),
                        _payload_mode(payload, rgb_already_windowed=rgb_already_windowed),
                        _payload_gutter(payload),
                    )
                    for key, payload in sorted(page_payloads.items())
                ),
                int(montage.tile_height),
                int(montage.tile_width),
                int(montage.columns),
                int(montage.rows),
                int(montage.gap),
                id(page),
                tuple(int(value) for value in page.atlas_shape),
            )
            if page_payloads:
                active_pages.add(page_index)
            if geometry_key != self._geometry_keys.get(page_index):
                vertices, texcoords, modes = _quad_buffers(
                    montage,
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
            if mapping_changed:
                mapping_updates += int(bool(visual.set_shader_mapping(shader_mapping)))
            visual.visible = page_index in active_pages
        for visual in self._visuals_by_page[len(self._pool.pages):]:
            visual.visible = False
        self._visible_items = len(payloads)
        self._last_stats = GpuMontageLayerStats(
            visible_items=texture_stats.visible_items,
            resident_items=texture_stats.resident_items,
            atlas_capacity=texture_stats.atlas_capacity,
            atlas_rebuilds=texture_stats.atlas_rebuilds,
            atlas_evictions=texture_stats.atlas_evictions,
            texture_uploads=texture_stats.texture_uploads,
            texture_upload_bytes=texture_stats.texture_upload_bytes,
            vertex_uploads=vertex_uploads,
            items_updated=texture_stats.items_updated,
            items_skipped=texture_stats.items_skipped,
            level_updates=int(bool(level_updates)),
            estimated_gpu_bytes=texture_stats.estimated_gpu_bytes,
            cpu_shadow_bytes=texture_stats.cpu_shadow_bytes,
            upload_ms=texture_stats.upload_ms,
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
        )
        return self._last_stats

    def warm_residency(
        self,
        *,
        payloads: dict[int, DisplayTilePayload],
        geometry,
        rgb_already_windowed: bool,
        tile_delta=None,
        tile_residency_budget_bytes: int = 0,
    ) -> GpuMontageLayerStats:
        montage = getattr(geometry, "montage", None)
        if montage is None:
            return GpuMontageLayerStats()
        try:
            return self._pool.warm_payloads(
                {int(key): value for key, value in dict(payloads or {}).items()},
                tile_shape=_atlas_tile_shape_for_payloads(payloads, fallback=(int(montage.tile_height), int(montage.tile_width))),
                rgb_already_windowed=rgb_already_windowed,
                near_tile_source_ids=dict(getattr(tile_delta, "near_tile_source_ids", {}) or {}),
                budget_bytes=tile_residency_budget_bytes,
            )
        except AtlasCapacityError as exc:
            previous = self._last_stats
            return GpuMontageLayerStats(
                visible_items=int(previous.visible_items),
                resident_items=self._pool.resident_count,
                atlas_capacity=self._pool.capacity,
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
            gl_FragColor = vec4(vec3(intensity), 1.0);
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
            gl_FragColor = vec4(vec3(intensity), 1.0);
        } else {
            vec2 z = texture2D(u_scalar_texture, v_texcoord).rg;
            float scalar = complex_component(z);
            scalar = map_scale(scalar);
            float span = max(u_levels.y - u_levels.x, 1e-12);
            float phase_index;
            float intensity = 1.0;
            if (u_component_mode > 2.5) {
                phase_index = clamp((scalar - u_levels.x) / span, 0.0, 1.0);
            } else {
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
        self._shader_mapping_key = None
        self._levels = (0.0, 1.0)
        self._scale_mode = 0.0
        self._symlog_constant = 0.0
        self._component_mode = 0.0
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

    def set_levels(self, levels) -> bool:
        levels = _normalize_levels(levels, self._levels)
        if levels == self._levels:
            return False
        self._levels = levels
        self.update()
        return True

    def set_shader_mapping(self, mapping) -> bool:
        scale_mode = _shader_scale_uniform(getattr(mapping, "scale", None))
        symlog_constant = float(getattr(mapping, "symlog_constant", 0.0) or 0.0)
        component_mode = shader_component_uniform(getattr(mapping, "component", None))
        lut = _normalized_lut(getattr(mapping, "lut_data", None))
        lut_key = _array_content_key(lut)
        mapping_key = (float(scale_mode), float(symlog_constant), float(component_mode), lut_key)
        if mapping_key == self._shader_mapping_key:
            return False
        self._shader_mapping_key = mapping_key
        self._scale_mode = scale_mode
        self._symlog_constant = symlog_constant
        self._component_mode = component_mode
        self._set_lut_texture(lut, key=lut_key)
        self.update()
        return True

    def _prepare_transforms(self, view) -> None:
        view.view_program.vert["transform"] = view.transforms.get_transform()

    def _prepare_draw(self, view):
        if self._scalar_texture is None or self._color_texture is None:
            return False
        if self._lut_texture is None:
            self._set_lut_texture(None)
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
        return True

    def _set_lut_texture(self, lut_data, *, key=None) -> bool:
        lut = _normalized_lut(lut_data)
        key = _array_content_key(lut) if key is None else key
        if key == self._lut_key and self._lut_texture is not None:
            return False
        from vispy import gloo

        if self._lut_texture is None or tuple(getattr(self._lut_texture, "shape", ())) != tuple(lut.shape):
            self._lut_texture = gloo.Texture2D(
                lut,
                format="rgb",
                internalformat="rgb8",
                interpolation="linear",
                wrapping="clamp_to_edge",
            )
        else:
            self._lut_texture.set_data(lut, copy=False)
        self._lut_key = key
        return True

    def _bounds(self, axis, view):
        del view
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
            value = gl.glGetString(getattr(gl, name))
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value or "")

        max_texture = get_integer("GL_MAX_TEXTURE_SIZE", 4096)
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
        scalar = _fit_complex_rg(pack_texture_data(texture, TexturePlaneKind.COMPLEX_RG32F), (tile_h, tile_w)) if need_scalar else None
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


def _fit_scalar(data, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if tuple(arr.shape[:2]) == tuple(shape) and arr.ndim == 2 and arr.flags.c_contiguous:
        # Gloo retains the array in its deferred command queue.  Reusing the
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


def _quad_buffers(montage, payloads, uvs, *, rgb_already_windowed: bool):
    vertices = []
    texcoords = []
    modes = []
    tile_w = int(montage.tile_width)
    tile_h = int(montage.tile_height)
    stride_x = tile_w + int(montage.gap)
    stride_y = tile_h + int(montage.gap)
    tile_count = len(montage.indices)
    for tile_number, payload in sorted((int(key), value) for key, value in dict(payloads).items()):
        if tile_number < 0 or tile_number >= tile_count:
            continue
        uv = uvs.get(int(tile_number))
        if uv is None:
            continue
        row = int(tile_number) // int(montage.columns)
        col = int(tile_number) % int(montage.columns)
        x0 = float(col * stride_x)
        y0 = float(row * stride_y)
        x1 = x0 + tile_w
        y1 = y0 + tile_h
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


def take_payload_batch(
    payloads,
    *,
    max_items: int = 4,
    max_bytes: int = 8 * 1024 * 1024,
) -> tuple[dict[int, DisplayTilePayload], dict[int, DisplayTilePayload]]:
    """Split speculative upload work into a bounded UI-event-loop batch."""

    max_items = max(1, int(max_items))
    max_bytes = max(1, int(max_bytes))
    batch: dict[int, DisplayTilePayload] = {}
    remaining: dict[int, DisplayTilePayload] = {}
    batch_bytes = 0
    for tile, payload in dict(payloads or {}).items():
        payload_bytes = _payload_upload_nbytes(payload)
        fits_items = len(batch) < max_items
        fits_bytes = not batch or batch_bytes + payload_bytes <= max_bytes
        if fits_items and fits_bytes:
            batch[int(tile)] = payload
            batch_bytes += payload_bytes
        else:
            remaining[int(tile)] = payload
    return batch, remaining


def _payload_upload_nbytes(payload: DisplayTilePayload) -> int:
    texture = payload.texture_data if payload.texture_data is not None else payload.image
    total = int(np.asarray(texture).nbytes)
    histogram = getattr(payload, "histogram_data", None)
    if histogram is not None and histogram is not texture:
        total += int(np.asarray(histogram).nbytes)
    return max(1, total)


def _normalized_lut(lut_data) -> np.ndarray:
    lut = default_phase_lut() if lut_data is None else np.asarray(lut_data)
    if lut.ndim != 2 or lut.shape[0] < 1 or lut.shape[1] < 3:
        lut = default_phase_lut()
    lut = lut[:, :3]
    if lut.dtype != np.uint8:
        if np.issubdtype(lut.dtype, np.floating) and lut.size and float(np.nanmax(lut)) <= 1.0:
            lut = lut * 255.0
        lut = np.clip(lut, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(lut.reshape((1, lut.shape[0], 3)))


def _array_content_key(array: np.ndarray) -> tuple[object, ...]:
    array = np.asarray(array)
    return (tuple(int(value) for value in array.shape), array.dtype.str, hash(array.tobytes()))


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
    if needs_complex and not needs_scalar and not needs_color:
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


def _atlas_reserve_count(geometry, *, minimum: int) -> int:
    """Reserve slots for the complete non-skipped visible montage set.

    Progressive sessions begin with most tiles in ``unloaded`` state.  Counting
    only loaded/loading tiles makes the atlas grow repeatedly as scheduling
    advances, which discards otherwise reusable GPU residency.  Geometry is the
    authoritative visibility plan, so reserve every planned slot except tiles
    explicitly marked skipped.
    """

    montage = getattr(geometry, "montage", None)
    indices = tuple(getattr(montage, "indices", ()) or ())
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


def _payload_gutter(payload: DisplayTilePayload) -> int:
    lod = getattr(payload, "lod", None)
    return 0 if lod is None else int(getattr(lod, "gutter", 0) or 0)


def _max_payload_lod_level(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((int(getattr(getattr(payload, "lod", None), "level", 0) or 0) for payload in dict(payloads or {}).values()), default=0)


def _max_payload_lod_factor(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((int(getattr(getattr(payload, "lod", None), "factor", 1) or 1) for payload in dict(payloads or {}).values()), default=1)


def _max_payload_gutter(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((_payload_gutter(payload) for payload in dict(payloads or {}).values()), default=0)


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
    """

    return _source_resident_key(payload.source_id)


def _source_resident_key(source_id: object) -> object:
    try:
        hash(source_id)
    except Exception:
        return ("source-object", id(source_id))
    return ("source", source_id)


def _base_texture_source_id(source_id: object) -> object:
    if isinstance(source_id, tuple) and len(source_id) >= 3 and source_id[1] == "texture_kind":
        return source_id[0]
    return source_id
