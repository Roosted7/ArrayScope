"""Batched VisPy montage tile rendering.

The renderer keeps tile identity, GPU residency, and draw visibility separate.
Presentation commits may change the set of drawn tiles without invalidating
unchanged resident texture slots.  Level-only changes update uniforms only.
"""

from __future__ import annotations

import contextlib
import os
from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from arrayscope.display.frame_planner import ANCHORED_CHUNK_SHAPE
from arrayscope.display.lod import inner_uv_for_gutter
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.model.tile_identity import (
    acknowledged_identity_satisfies_target,
    array_plane_identities,
    plane_identity_record,
    tile_ack_identity,
)
from arrayscope.display.model.tile_stats import TileLayerUpdateStats
from arrayscope.display.pyramid import LodPagePlan
from arrayscope.display.shader_mapping import (
    ShaderDisplayMode,
    ShaderScale,
    TexturePlaneKind,
    normalize_lut_rgb,
    shader_component_uniform,
)
from arrayscope.display.tile_layout import planned_tile_count, tile_layout_map
from arrayscope.gpu.keys import (
    COMPLEX_RG32F,
    REDUCER_NATIVE,
    REDUCER_PHASE_VECTOR,
    RGB8,
    ChunkLod,
    DataChunkKey,
)
from arrayscope.gpu.page_table import (
    PageResolution,
    PageSlot,
    PageTable,
)

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


@dataclass(frozen=True)
class TileDrawPart:
    """One quad of a view tile: a world sub-rect sampling a cropped UV rect.

    ADR 0055 G3: a view tile no longer necessarily owns one full slot. It may
    draw as several parts, each sampling a UV sub-window of a resident slot
    (window shifts become texcoord updates plus boundary uploads). A tile
    with no parts registered draws the classic single full-slot quad.

    Seam rules (see docs/proposals/gpu-engine-plan.md G3): world-space part
    edges must fall on integer texel boundaries (atlas filtering is NEAREST),
    and cropped UV rects must stay inside the gutter-protected inner region
    of their slot while whole-atlas mipmaps are enabled.
    """

    world_rect: tuple[float, float, float, float]  # x0, y0, x1, y1
    uv_rect: tuple[float, float, float, float]  # u0, v0, u1, v1
    page_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_rect", tuple(float(v) for v in self.world_rect))
        object.__setattr__(self, "uv_rect", tuple(float(v) for v in self.uv_rect))
        if self.page_index is not None:
            object.__setattr__(self, "page_index", int(self.page_index))


@dataclass(frozen=True)
class _PayloadChunk:
    """One origin-anchored chunk of an anchored payload plane (ADR 0055 G3b-2).

    ``key`` is the canonical source-grid residency identity: equal keys mean
    equal texels regardless of the display window that produced the payload.
    Boundary chunks carry their *clipped* native footprint, so they remain
    distinct from aligned interiors and legitimately re-upload on a shift.
    """

    key: DataChunkKey
    rect: tuple[int, int, int, int]  # native source rect (y0, y1, x0, x1)
    plane_rect: tuple[int, int, int, int]  # same rect in payload plane pixels


@dataclass(frozen=True)
class _ChunkCommit:
    """Result of committing one payload through the chunked-residency path."""

    page_index: int
    slot: int
    uv: tuple[float, float, float, float]
    uploads: int
    upload_bytes: int
    complex_uploads: int
    prepare_ms: float
    submit_ms: float
    uploaded_any: bool


@dataclass(frozen=True)
class _PageUpload:
    """Residency-only result for checked canonical page values."""

    uploads: int
    upload_bytes: int
    complex_uploads: int
    prepare_ms: float
    submit_ms: float
    uploaded_keys: tuple[DataChunkKey, ...]


def _payload_chunked_eligible(payload: DisplayTilePayload) -> bool:
    """Whether a native payload may take the legacy chunk-residency path.

    ADR 0056 G5 permits this backend-local partition only for native samples.
    Reduced values must carry canonical ``page_backing`` planned by the
    source-grid route; VisPy consumes those keys and must never reconstruct
    reduced-page identity from a whole-plane payload or ``LodInfo``.
    """

    if getattr(payload, "page_backing", None) is not None:
        return False
    anchor = getattr(payload, "source_anchor", None)
    if anchor is None:
        return False
    if str(getattr(payload, "quality", "exact") or "exact") != "exact":
        return False
    lod = getattr(payload, "lod", None)
    factor = 1 if lod is None else int(getattr(lod, "factor", 1) or 1)
    level = 0 if lod is None else int(getattr(lod, "level", 0) or 0)
    if factor != 1 or level != 0:
        return False
    if lod is not None and int(getattr(lod, "gutter", 0) or 0) != 0:
        return False
    try:
        y0, y1, x0, x1 = (int(value) for value in anchor.source_rect)
    except Exception:
        return False
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        return False
    plane_h, plane_w = _payload_class_shape(payload)
    native_h, native_w = (y1 - y0, x1 - x0)
    if (plane_h, plane_w) != (native_h, native_w):
        return False
    if lod is not None:
        if tuple(int(value) for value in lod.source_shape) != (native_h, native_w):
            return False
        if tuple(int(value) for value in lod.texture_shape) != (plane_h, plane_w):
            return False
    chunk_h, chunk_w = ANCHORED_CHUNK_SHAPE
    return not (plane_h <= int(chunk_h) and plane_w <= int(chunk_w))


def _payload_chunk_plan(payload: DisplayTilePayload) -> tuple[_PayloadChunk, ...]:
    """Origin-anchored chunk partition of an eligible native payload plane.

    Chunk boundaries fall at integer multiples of the
    chunk shape in NATIVE source coordinates; interior chunks keep
    full-chunk rects (shift-invariant identity), chunks clipped by the
    window edge carry the clipped rect.

    Reduced payloads are rejected: their exact partition and keys come from
    ``PageBackedPresentation.requested_plans``.
    """

    _require_native_chunk_payload(payload)
    anchor = payload.source_anchor
    y0, y1, x0, x1 = (int(value) for value in anchor.source_rect)
    chunk_h, chunk_w = (int(ANCHORED_CHUNK_SHAPE[0]), int(ANCHORED_CHUNK_SHAPE[1]))
    kind = _payload_texture_kind(payload).value
    texture = np.asarray(
        payload.texture_data if payload.texture_data is not None else payload.image
    )
    dtype = str(texture.dtype)
    chunks: list[_PayloadChunk] = []
    for cy in range((y0 // chunk_h) * chunk_h, y1, chunk_h):
        ry0, ry1 = (max(cy, y0), min(cy + chunk_h, y1))
        for cx in range((x0 // chunk_w) * chunk_w, x1, chunk_w):
            rx0, rx1 = (max(cx, x0), min(cx + chunk_w, x1))
            rect = (
                ry0,
                ry1,
                rx0,
                rx1,
            )
            chunks.append(
                _PayloadChunk(
                    key=_data_chunk_key(
                        content_key=anchor.content_key,
                        rect=rect,
                        representation=kind,
                        dtype=dtype,
                    ),
                    rect=rect,
                    plane_rect=(ry0 - y0, ry1 - y0, rx0 - x0, rx1 - x0),
                )
            )
    return tuple(chunks)


def _payload_lod_is_reduced(payload: DisplayTilePayload) -> bool:
    """Whether payload metadata describes anything other than native L0."""

    lod = getattr(payload, "lod", None)
    if lod is None:
        return False
    return int(getattr(lod, "factor", 1) or 1) != 1 or int(getattr(lod, "level", 0) or 0) != 0


def _require_canonical_reduced_payload(
    payload: DisplayTilePayload,
    *,
    action: str,
) -> None:
    """Reject the deleted window-local reduced-payload compatibility path."""

    if _payload_lod_is_reduced(payload) and getattr(payload, "page_backing", None) is None:
        raise ValueError(
            f"VisPy cannot {action} a reduced payload without canonical page_backing; "
            "factor>1 values must be planned and keyed by the source-grid route"
        )


def _require_native_chunk_payload(payload: DisplayTilePayload) -> None:
    """Guard the sole remaining backend-local key construction (native L0)."""

    if getattr(payload, "page_backing", None) is not None:
        raise ValueError(
            "page-backed payloads must consume their canonical requested plans; "
            "backend chunk planning is forbidden"
        )
    _require_canonical_reduced_payload(payload, action="partition")
    if not _payload_chunked_eligible(payload):
        raise ValueError("payload is not eligible for native chunk planning")


def _rect_intersection_yx(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    y0 = max(int(left[0]), int(right[0]))
    y1 = min(int(left[1]), int(right[1]))
    x0 = max(int(left[2]), int(right[2]))
    x1 = min(int(left[3]), int(right[3]))
    if y1 <= y0 or x1 <= x0:
        return None
    return (y0, y1, x0, x1)


def _stored_edge_for_source(
    source_edge: int,
    *,
    source_start: int,
    source_stop: int,
    stored_start: int,
    stored_stop: int,
) -> float:
    """Map an exact native edge through one uniform canonical draw block."""

    source_extent = int(source_stop) - int(source_start)
    if source_extent <= 0:
        raise ValueError("source draw-block extent must be positive")
    return float(stored_start) + (
        (int(source_edge) - int(source_start))
        * (int(stored_stop) - int(stored_start))
        / float(source_extent)
    )


def _data_chunk_key(
    *,
    content_key: object,
    rect: tuple[int, int, int, int],
    representation: str,
    dtype: str,
) -> DataChunkKey:
    """Canonical native-page identity for one anchored atlas chunk.

    ``SourceAnchoring`` already separates the immutable document/evaluation
    identity from the window-free request.  Test/foreign anchors that do not
    carry that tagged shape remain one opaque document generation rather than
    being guessed apart. Reduced keys are constructed only by the source-grid
    planner and arrive through ``page_backing``.
    """

    if (
        isinstance(content_key, tuple)
        and len(content_key) == 3
        and content_key[0] == "src-anchored"
    ):
        document_generation, operation_key = content_key[1], content_key[2]
    else:
        document_generation, operation_key = content_key, None
    y0, y1, x0, x1 = (int(value) for value in rect)
    return DataChunkKey(
        document_generation=document_generation,
        operation_key=operation_key,
        lod=ChunkLod(
            level=0,
            factor=1,
            gutter=0,
            reduction=(0, 0),
            reducer=REDUCER_NATIVE,
        ),
        chunk_origin=(y0, x0),
        chunk_shape=(y1 - y0, x1 - x0),
        dtype=dtype,
        representation=representation,
    )


_ATLAS_GROWTH_TARGET_BYTES = 32 * 1024 * 1024
_UNSET = object()


def _page_target_pin_owner(tile_number: int, scope: object = "legacy") -> tuple[object, ...]:
    return ("vispy-page-target", scope, int(tile_number))


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
    def __init__(
        self,
        gloo,
        *,
        tile_shape: tuple[int, int],
        capacity: int,
        storage_mode: str,
        max_texture_size: int,
    ):
        self._gloo = gloo
        self.tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
        self.capacity = max(1, int(capacity))
        self.storage_mode = _normalize_storage_mode(storage_mode)
        self.max_texture_size = max(1, int(max_texture_size))
        self.columns, self.rows = _atlas_grid(
            tile_shape=self.tile_shape,
            capacity=self.capacity,
            max_texture_size=self.max_texture_size,
        )
        tile_h, tile_w = self.tile_shape
        self.atlas_shape = (int(self.rows * tile_h), int(self.columns * tile_w))
        self.scalar_is_atlas = self.storage_mode in {"scalar", "scalar_color", "complex"}
        self.complex_is_atlas = self.storage_mode == "complex"
        self.color_is_atlas = self.storage_mode in {"color", "scalar_color"}
        if self.scalar_is_atlas:
            channels = 2 if self.complex_is_atlas else 1
            self.scalar_texture = self._gloo.Texture2D(
                shape=(*self.atlas_shape, channels),
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
                shape=(*self.atlas_shape, 3),
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
        return _atlas_allocation_bytes(
            tile_shape=self.tile_shape,
            capacity=self.capacity,
            storage_mode=self.storage_mode,
            max_texture_size=self.max_texture_size,
        )

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
        return uv

    def uv_for_slot_with_gutter(
        self, slot: int, gutter: int = 0
    ) -> tuple[float, float, float, float]:
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

    def __init__(
        self,
        gloo,
        *,
        limits: GpuDeviceLimits | None = None,
        max_texture_size: int | None = None,
        budget_bytes: int = 0,
    ):
        self._gloo = gloo
        self.device_limits = limits or GpuDeviceLimits(max_texture_size=max_texture_size or 4096)
        self.max_texture_size = max(
            1, int(max_texture_size or self.device_limits.max_texture_size or 4096)
        )
        self.budget_bytes = max(0, int(budget_bytes))
        self.tile_shape: tuple[int, int] | None = None
        self.storage_mode: str | None = None
        self.pages: list[TextureAtlasPage] = []
        # Chunk residency (which key lives in which page slot, LRU stamps)
        # is the ADR 0055 page table; the atlas keeps only texture mechanics
        # and the view-tile → resident-key presentation maps below.
        self._page_table = PageTable()
        self.page_target_resolutions: dict[int, PageResolution] = {}
        self._page_target_pin_tiles: set[int] = set()
        self.tile_page_target_resolutions: dict[int, tuple[PageResolution, ...]] = {}
        self.tile_page_candidate_missing: dict[int, tuple[DataChunkKey, ...]] = {}
        self._tile_page_pin_owners: dict[int, object] = {}
        self.tile_slots: dict[int, tuple[int, int]] = {}
        # ADR 0055 G3: optional per-tile quad list (UV-cropped sub-window
        # sampling). Tiles absent from this map draw one full-slot quad.
        self.tile_draw_parts: dict[int, tuple[TileDrawPart, ...]] = {}
        # ADR 0055 G3b-2 chunked residency: tiles whose anchored payload is
        # resident as N origin-anchored chunk slots instead of one whole-tile
        # slot. Pure residency bookkeeping — tile-level identity records stay
        # keyed by _resident_key(payload) exactly as on the classic path.
        self.tile_chunk_residency: dict[int, tuple[object, ...]] = {}
        self.chunk_resident_tiles: dict[object, set[int]] = {}
        # Chunk keys of the current active tile set: protected from eviction
        # alongside active tile-level keys.
        self.active_chunk_keys: set[object] = set()
        # Tile-level resident keys whose residency is a chunk set (no slot of
        # their own in the page table).
        self._chunked_tile_keys: set[object] = set()
        self.chunk_upload_count = 0
        self.chunk_reuse_count = 0
        self.tile_resident_keys: dict[int, object] = {}
        self.resident_tiles: dict[object, set[int]] = {}
        self.tile_uvs: dict[int, tuple[float, float, float, float]] = {}
        self.source_ids: dict[object, object] = {}
        self.acknowledged_identities: dict[object, object] = {}
        self.physical_upload_records: dict[object, dict[str, object]] = {}
        # Checked canonical geometry travels with each resident logical page.
        # PageTable selects the actual key/slot; this plan maps clipped native
        # bins into that slot without reconstructing route geometry here.
        self.page_plans: dict[DataChunkKey, LodPagePlan] = {}
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

    @property
    def resident_slots(self) -> dict[object, tuple[int, int]]:
        """Resident key → (page, slot) view of the page table (diagnostics/tests)."""

        return {
            key: (slot.page_index, slot.slot_index) for key, slot in self._page_table.slot_items()
        }

    def presented_identities(self) -> dict[int, object]:
        """Ground truth: the payload identity each drawn tile slot holds NOW.

        ADR 0051 rule 1: session bookkeeping converges against this map —
        the 2026-07-05 field defects all came from the session believing its
        own acknowledgement records over what the pool actually presents.
        """

        return {
            int(tile): self.acknowledged_identities.get(key, self.source_ids.get(key))
            for tile, key in self.tile_resident_keys.items()
            if key in self.source_ids
        }

    def resolve_page_targets(
        self,
        targets: dict[int, DataChunkKey],
    ) -> dict[int, PageResolution | None]:
        """Bind logical view targets to their best resident physical pages.

        Resolution is one bounded CPU lookup per target.  It changes only
        mappings and owner-scoped pins: not-resident stays explicit and this
        method never uploads, evaluates, or schedules work.
        """

        requested = {int(tile): key for tile, key in dict(targets).items()}
        for tile in self._page_target_pin_tiles.difference(requested):
            self._page_table.replace_pin_set(_page_target_pin_owner(tile), ())
            self.page_target_resolutions.pop(int(tile), None)
            self._clear_tile_mapping(int(tile))
        results: dict[int, PageResolution | None] = {}
        for tile, target in requested.items():
            resolution = self._page_table.resolve(target)
            results[tile] = resolution
            owner = _page_target_pin_owner(tile)
            if resolution is None:
                self._page_table.replace_pin_set(owner, ())
                self.page_target_resolutions.pop(tile, None)
                self._clear_tile_mapping(tile)
                continue
            self._page_table.replace_pin_set(owner, (resolution.actual_key,))
            self._page_table.touch(resolution.actual_key)
            page_index = int(resolution.slot.page_index)
            slot = int(resolution.slot.slot_index)
            page = self.pages[page_index]
            self._set_tile_mapping(
                tile,
                resolution.actual_key,
                page_index,
                slot,
                page.uv_for_slot(slot),
                page_backed=True,
            )
            self.page_target_resolutions[tile] = resolution
        self._page_target_pin_tiles = set(requested)
        return results

    def resolve_tile_page_targets(
        self,
        targets: dict[int, tuple[DataChunkKey, ...]],
        *,
        owner_scope: object,
    ) -> dict[int, tuple[PageResolution, ...] | None]:
        """Atomically replace each tile's complete multi-page resolution set.

        ``targets`` is a partial per-commit map, not the complete frame.  An
        omitted tile retains its mapping and owner pins; presentation removal
        flows through :meth:`_clear_tile_mapping` at the frame boundary.
        Missing candidate coverage never clears the previous complete pinned
        set. Resolution is pure CPU page-table work: no upload or scheduling.
        """

        requested = {int(tile): tuple(keys) for tile, keys in dict(targets).items()}
        for keys in requested.values():
            if not keys or len(set(keys)) != len(keys):
                raise ValueError("tile page targets must be non-empty and unique")
            if any(not isinstance(key, DataChunkKey) for key in keys):
                raise TypeError("tile page targets must be DataChunkKey values")
        results: dict[int, tuple[PageResolution, ...] | None] = {}
        for tile, keys in requested.items():
            candidate = tuple(self._page_table.resolve(key) for key in keys)
            missing = tuple(
                key for key, resolution in zip(keys, candidate, strict=True) if resolution is None
            )
            if missing:
                self.tile_page_candidate_missing[tile] = missing
                results[tile] = None
                continue
            resolved = tuple(candidate)
            owner = _page_target_pin_owner(tile, owner_scope)
            previous_owner = self._tile_page_pin_owners.get(tile)
            if previous_owner is not None and previous_owner != owner:
                self._page_table.replace_pin_set(previous_owner, ())
            actual_keys = tuple(dict.fromkeys(resolution.actual_key for resolution in resolved))
            self._page_table.replace_pin_set(owner, actual_keys)
            for key in actual_keys:
                self._page_table.touch(key)
            self._tile_page_pin_owners[tile] = owner
            self.tile_page_target_resolutions[tile] = resolved
            self.tile_page_candidate_missing.pop(tile, None)
            results[tile] = resolved
        return results

    def tile_truth_physical_rows(self) -> dict[int, dict[str, object]]:
        rows: dict[int, dict[str, object]] = {}
        for tile, resident_key in self.tile_resident_keys.items():
            slot_ref = self.tile_slots.get(int(tile))
            record = self.physical_upload_records.get(resident_key)
            if slot_ref is None or record is None:
                continue
            page_index, slot = (int(slot_ref[0]), int(slot_ref[1]))
            rows[int(tile)] = {
                **record,
                "physical_page": page_index,
                "physical_slot": slot,
                "physical_acknowledged_identity": self.acknowledged_identities.get(
                    resident_key,
                    self.source_ids.get(resident_key),
                ),
            }
            resolution = self.page_target_resolutions.get(int(tile))
            if resolution is not None:
                rows[int(tile)].update(
                    {
                        "physical_page_target_key": resolution.target_key,
                        "physical_page_actual_key": resolution.actual_key,
                        "physical_page_lod": resolution.actual_key.lod,
                        "physical_page_quality": (
                            "exact"
                            if resolution.actual_key == resolution.target_key
                            else "fallback"
                        ),
                        "physical_page_binding_generation": int(resolution.binding_generation),
                    }
                )
            multi_resolution = self.tile_page_target_resolutions.get(int(tile))
            if multi_resolution is not None:
                rows[int(tile)].update(
                    {
                        "physical_page_bindings": tuple(
                            {
                                "target_key": item.target_key,
                                "requested_lod": item.target_key.lod,
                                "actual_key": item.actual_key,
                                "actual_lod": item.actual_key.lod,
                                "scale": tuple(float(value) for value in item.scale),
                                "offset": tuple(float(value) for value in item.offset),
                                "quality": (
                                    "exact" if item.actual_key == item.target_key else "fallback"
                                ),
                                "slot": item.slot,
                                "binding_generation": int(item.binding_generation),
                            }
                            for item in multi_resolution
                        ),
                        "physical_page_candidate_missing": self.tile_page_candidate_missing.get(
                            int(tile), ()
                        ),
                    }
                )
        return rows

    @property
    def resident_count(self) -> int:
        return len(self.source_ids)

    def payload_resident(self, payload: DisplayTilePayload) -> bool:
        """Return exact physical payload residency without touching LRU state."""

        _require_canonical_reduced_payload(payload, action="resolve residency for")
        backing = getattr(payload, "page_backing", None)
        if backing is not None:
            # Admission asks whether committing this payload performs physical
            # work.  A requested key may resolve through a coarser ancestor,
            # but a payload that supplies finer pages will upload those exact
            # pages when admitted.  Only the supplied materialized page set is
            # therefore allowed to bypass the item/upload budget.
            return all(
                self._page_table.lookup(page.key) is not None for page in backing.materialized_pages
            )
        if _payload_chunked_eligible(payload):
            # Native source-anchored planes are physically partitioned into
            # DataChunkKey pages. Hidden warming intentionally creates no
            # tile-level mapping/acknowledgement, so only this exact chunk set
            # can describe the state consumed by the zero-upload commit.
            return all(
                self._page_table.lookup(chunk.key) is not None
                for chunk in _payload_chunk_plan(payload)
            )
        resident_key = _resident_key(payload)
        return bool(
            self.source_ids.get(resident_key) == payload.source_id
            and self.acknowledged_identities.get(resident_key)
            == (getattr(payload, "tile_identity", None) or payload.source_id)
            and self._page_table.lookup(resident_key) is not None
        )

    @property
    def capacity(self) -> int:
        return sum(int(page.capacity) for page in self.pages)

    @property
    def slots(self) -> dict[int, int]:
        return {
            tile: page_index * 1_000_000 + slot
            for tile, (page_index, slot) in self.tile_slots.items()
        }

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
    def cpu_shadow_bytes(self) -> int:
        # Atlas storage is allocated by shape on the GPU.  Only per-tile
        # staging arrays exist during a sub-upload.
        return 0

    @property
    def estimated_gpu_bytes(self) -> int:
        return sum(int(page.estimated_gpu_bytes) for page in self.pages)

    def diagnostics_snapshot(self) -> dict[str, object]:
        """Derived atlas/page-table state for live physical-truth traces.

        Keep this as a read-only projection of the pool's canonical maps.  In
        particular, candidate-missing counts come from the resolver result and
        residency comes from the page table; no presentation/session cache is
        allowed to infer either value independently.
        """

        classes: dict[tuple[tuple[int, int], str], dict[str, object]] = {}
        for page_index, page in enumerate(self.pages):
            key = (tuple(int(value) for value in page.tile_shape), str(page.storage_mode))
            row = classes.setdefault(
                key,
                {
                    "shape": key[0],
                    "storage_mode": key[1],
                    "page_count": 0,
                    "page_indices": [],
                    "capacity": 0,
                    "occupied": 0,
                    "bytes": 0,
                },
            )
            row["page_count"] = int(row["page_count"]) + 1
            row["page_indices"].append(int(page_index))
            row["capacity"] = int(row["capacity"]) + int(page.capacity)
            row["occupied"] = int(row["occupied"]) + sum(
                owner is not None for owner in page.slot_owners
            )
            row["bytes"] = int(row["bytes"]) + int(page.estimated_gpu_bytes)
        page_classes = tuple(
            {
                **row,
                "page_indices": tuple(int(index) for index in row["page_indices"]),
            }
            for _key, row in sorted(classes.items(), key=lambda item: item[0])
        )
        missing = tuple(self.tile_page_candidate_missing.values())
        return {
            "page_candidate_missing_tile_count": len(missing),
            "page_candidate_missing_key_count": sum(len(keys) for keys in missing),
            "page_table_resident_count": len(self._page_table),
            "atlas_page_classes": page_classes,
            "atlas_estimated_gpu_bytes": int(self.estimated_gpu_bytes),
            "atlas_budget_bytes": int(self.budget_bytes),
        }

    def ensure_layout(
        self,
        *,
        tile_shape: tuple[int, int],
        count: int,
        storage_mode: str = "scalar_color",
        budget_bytes: int | None = None,
    ) -> bool:
        tile_h, tile_w = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        requested = max(0, int(count))
        if budget_bytes is not None:
            self.budget_bytes = max(0, int(budget_bytes))
        max_columns = self.max_texture_size // tile_w
        max_rows = self.max_texture_size // tile_h
        if max_columns < 1 or max_rows < 1:
            raise AtlasCapacityError(
                f"tile shape {(tile_h, tile_w)} exceeds atlas texture limit {self.max_texture_size}"
            )
        max_slots_per_page = int(max_columns * max_rows)
        storage_mode = _normalize_storage_mode(storage_mode)
        required_bytes = _atlas_class_allocation_bytes(
            tile_shape=(tile_h, tile_w),
            count=requested,
            storage_mode=storage_mode,
            max_texture_size=self.max_texture_size,
        )
        if self.budget_bytes > 0 and required_bytes > self.budget_bytes:
            raise AtlasCapacityError(
                f"{requested} active tiles of shape {(tile_h, tile_w)} exceed tile residency budget "
                f"{self.budget_bytes} bytes (physical allocation {required_bytes} bytes)"
            )

        shape_changed = self.tile_shape != (tile_h, tile_w)
        if not shape_changed and requested <= self._class_capacity(
            (tile_h, tile_w), storage_mode=storage_mode
        ):
            self.storage_mode = storage_mode
            return False

        self.tile_shape = (tile_h, tile_w)
        if shape_changed:
            self.pages.clear()
            self._page_table = PageTable()
            self.page_target_resolutions.clear()
            self._page_target_pin_tiles.clear()
            self.tile_page_target_resolutions.clear()
            self.tile_page_candidate_missing.clear()
            self._tile_page_pin_owners.clear()
            self.tile_slots.clear()
            self.tile_draw_parts.clear()
            self.tile_chunk_residency.clear()
            self.chunk_resident_tiles.clear()
            self.active_chunk_keys.clear()
            self._chunked_tile_keys.clear()
            self.tile_resident_keys.clear()
            self.resident_tiles.clear()
            self.tile_uvs.clear()
            self.source_ids.clear()
            self.acknowledged_identities.clear()
            self.physical_upload_records.clear()
            self.page_plans.clear()
            self.active_resident_keys.clear()
            self.superseded_keys.clear()
        self.storage_mode = storage_mode
        if shape_changed:
            # The shape reset above emptied the pool, so this initial class
            # cannot exceed the total budget after the per-class check.
            while self._class_capacity((tile_h, tile_w), storage_mode=storage_mode) < requested:
                remaining = requested - self._class_capacity(
                    (tile_h, tile_w), storage_mode=storage_mode
                )
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
        else:
            capacity = self._ensure_class_capacity(
                (tile_h, tile_w),
                requested,
                storage_mode=storage_mode,
            )
            if capacity < requested:
                raise AtlasCapacityError(
                    f"atlas cannot allocate {requested} active slots of shape "
                    f"{(tile_h, tile_w)} and storage {storage_mode} within "
                    f"{self.budget_bytes} bytes"
                )
        self.serial += 1
        self.rebuild_count += 1
        return True

    def _class_capacity(
        self,
        tile_shape: tuple[int, int],
        *,
        storage_mode: str | None = None,
    ) -> int:
        """Slot count of the pages whose slot shape matches ``tile_shape``.

        Pages are classed by texture shape (ADR 0050): a reduced-level tile
        must never occupy a native-shaped slot, so capacity questions are
        answered per shape class.
        """

        shape = (int(tile_shape[0]), int(tile_shape[1]))
        mode = None if storage_mode is None else _normalize_storage_mode(storage_mode)
        return sum(
            int(page.capacity)
            for page in self.pages
            if page.tile_shape == shape and (mode is None or page.storage_mode == mode)
        )

    def _ensure_class_capacity(
        self,
        tile_shape: tuple[int, int],
        requested: int,
        *,
        storage_mode: str | None = None,
    ) -> int:
        """Append pages for a non-base shape class within the byte budget.

        Returns the class capacity after growth.  The base class is owned by
        ``ensure_layout``; this only serves additional coexisting LOD levels,
        so running out of budget degrades to fewer slots instead of raising.
        """

        shape = (max(1, int(tile_shape[0])), max(1, int(tile_shape[1])))
        requested = max(1, int(requested))
        mode = _normalize_storage_mode(storage_mode or self.storage_mode or "scalar")
        capacity = self._class_capacity(shape, storage_mode=mode)
        if capacity >= requested:
            return capacity
        max_columns = max(1, self.max_texture_size // shape[1])
        max_rows = max(1, self.max_texture_size // shape[0])
        max_slots_per_page = max(1, int(max_columns * max_rows))
        bytes_per_slot = max(1, _storage_mode_bytes_per_pixel(mode) * shape[0] * shape[1])
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
                # Atlas textures allocate the complete rectangular grid, not
                # merely ``capacity`` logical slots.  Fit against that rounded
                # physical allocation so a nominally in-budget page cannot
                # push the pool over its byte limit.
                low, high = 0, max(0, int(page_capacity))
                while low < high:
                    candidate = (low + high + 1) // 2
                    candidate_bytes = _atlas_allocation_bytes(
                        tile_shape=shape,
                        capacity=candidate,
                        storage_mode=mode,
                        max_texture_size=self.max_texture_size,
                    )
                    if candidate_bytes <= budget_left:
                        low = candidate
                    else:
                        high = candidate - 1
                page_capacity = int(low)
                if page_capacity < 1:
                    # Before budget-limiting the class, reclaim slots whose
                    # tiles now present a different class and drop pages that
                    # become empty; retry only when bytes were recovered.
                    if self._release_superseded_capacity(protect_class=(shape, mode)):
                        continue
                    if self._release_inactive_page_capacity(protect_class=(shape, mode)):
                        continue
                    break
            self.pages.append(
                TextureAtlasPage(
                    self._gloo,
                    tile_shape=shape,
                    capacity=int(page_capacity),
                    storage_mode=mode,
                    max_texture_size=self.max_texture_size,
                )
            )
            self.serial += 1
            capacity += int(page_capacity)
        return capacity

    def _release_superseded_capacity(
        self,
        *,
        protect_class: tuple[tuple[int, int], str] | None = None,
    ) -> bool:
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

        protect = (
            None
            if protect_class is None
            else (
                (int(protect_class[0][0]), int(protect_class[0][1])),
                _normalize_storage_mode(protect_class[1]),
            )
        )
        active_bases = self.active_base_source_ids
        # Reclaim in LRU order among superseded slots, keeping the retained
        # adjacent level of active tiles for a second pass: losing it costs a
        # re-upload on the next level flip, so it goes only when reclaiming
        # everything else recovered no bytes (ADR 0050).
        ordered = sorted(
            self.superseded_keys,
            key=lambda key: (
                _lod_invariant_source_id(self.source_ids.get(key)) in active_bases,
                self._page_table.last_use(key),
            ),
        )
        for adjacent_pass in (False, True):
            touched_pages: set[int] = set()
            for key in ordered:
                adjacent = _lod_invariant_source_id(self.source_ids.get(key)) in active_bases
                if adjacent != adjacent_pass:
                    continue
                if (
                    key in self.active_resident_keys
                    or self.resident_tiles.get(key)
                    or self._page_table.is_pinned(key)
                ):
                    continue
                slot_ref = self._page_table.lookup(key)
                if slot_ref is None:
                    self.superseded_keys.discard(key)
                    continue
                page_index, slot = (slot_ref.page_index, slot_ref.slot_index)
                page = self.pages[page_index]
                if protect is not None and (page.tile_shape, page.storage_mode) == protect:
                    # Same-class slots are useful as-is: ordinary eviction
                    # reuses them without any byte recovery, so keep them.
                    continue
                page.slot_owners[slot] = None
                page._free_slots.append(slot)
                self._page_table.unbind(key)
                self.source_ids.pop(key, None)
                self.acknowledged_identities.pop(key, None)
                self.physical_upload_records.pop(key, None)
                self.page_plans.pop(key, None)
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

    def _release_inactive_page_capacity(
        self,
        *,
        protect_class: tuple[tuple[int, int], str],
    ) -> bool:
        """Drop one wholly inactive physical page to fund another class.

        Storage mode is a page property.  A hidden complex page cannot lend a
        slot to a scalar value, so under a hard byte budget the only truthful
        operation transition is to release an entire inactive page and create
        the requested class.  Pages with a drawn tile, an active target, a
        chunk owner, or any owner pin are ineligible.  With budget headroom no
        page is touched, preserving fast operation reverts.
        """

        protected_shape = (
            int(protect_class[0][0]),
            int(protect_class[0][1]),
        )
        protected_mode = _normalize_storage_mode(protect_class[1])
        candidates: list[tuple[int, int]] = []
        for page_index, page in enumerate(self.pages):
            if (page.tile_shape, page.storage_mode) == (
                protected_shape,
                protected_mode,
            ):
                continue
            owners = tuple(owner for owner in page.slot_owners if owner is not None)
            if any(
                owner in self.active_resident_keys
                or owner in self.active_chunk_keys
                or bool(self.resident_tiles.get(owner))
                or bool(self.chunk_resident_tiles.get(owner))
                or self._page_table.is_pinned(owner)
                for owner in owners
            ):
                continue
            oldest_use = max(
                (int(self._page_table.last_use(owner)) for owner in owners),
                default=-1,
            )
            candidates.append((oldest_use, int(page_index)))
        if not candidates:
            return False
        _last_use, page_index = min(candidates)
        page = self.pages[int(page_index)]
        for slot, owner in enumerate(tuple(page.slot_owners)):
            if owner is None:
                continue
            page.slot_owners[int(slot)] = None
            self._release_victim(owner, near_keys=set())
        self._drop_pages((int(page_index),))
        return True

    def _drop_pages(self, page_indices) -> None:
        dropped = {int(index) for index in page_indices}
        if not dropped:
            return
        invalid = tuple(sorted(index for index in dropped if index < 0 or index >= len(self.pages)))
        if invalid:
            raise IndexError(f"cannot drop atlas page indices {invalid!r}")

        remap: dict[int, int] = {}
        kept: list[TextureAtlasPage] = []
        for old_index, page in enumerate(self.pages):
            if old_index in dropped:
                continue
            remap[old_index] = len(kept)
            kept.append(page)

        def remapped_page_index(old_index: int, *, owner: object) -> int:
            old_index = int(old_index)
            if old_index not in remap:
                raise RuntimeError(
                    f"cannot compact atlas: {owner!r} still references dropped page {old_index}"
                )
            return int(remap[old_index])

        # Build every presentation-side remap before changing the canonical
        # page table.  A page is droppable only after all of its bindings and
        # draw owners have been released; finding one here is an invariant
        # violation, not a reason to publish a partially compacted atlas.
        for key, slot_ref in self._page_table.slot_items():
            remapped_page_index(slot_ref.page_index, owner=key)
        remapped_tile_slots = {
            int(tile): (
                remapped_page_index(page_index, owner=("tile-slot", int(tile))),
                int(slot),
            )
            for tile, (page_index, slot) in self.tile_slots.items()
        }
        remapped_draw_parts = {
            int(tile): tuple(
                part
                if part.page_index is None
                else replace(
                    part,
                    page_index=remapped_page_index(
                        part.page_index,
                        owner=("tile-draw-part", int(tile)),
                    ),
                )
                for part in parts
            )
            for tile, parts in self.tile_draw_parts.items()
        }

        cached_resolutions = tuple(self.page_target_resolutions.values()) + tuple(
            resolution
            for resolutions in self.tile_page_target_resolutions.values()
            for resolution in resolutions
        )
        for resolution in cached_resolutions:
            # Preserve the physically presented actual page until the normal
            # resolver deliberately replaces it.  A newly warmed finer page
            # may now win a fresh target lookup, but compaction alone must not
            # silently change presentation quality or geometry.
            binding = self._page_table.resolve(resolution.actual_key)
            if (
                binding is None
                or binding.actual_key != resolution.actual_key
                or binding.slot != resolution.slot
            ):
                raise RuntimeError(
                    "cannot compact atlas with a stale cached page resolution: "
                    f"target={resolution.target_key!r}, "
                    f"actual={resolution.actual_key!r}"
                )

        self._page_table.remap_slots(
            lambda slot: PageSlot(slot.pool_id, remap[slot.page_index], slot.slot_index)
        )

        def refreshed_resolution(resolution: PageResolution) -> PageResolution:
            binding = self._page_table.resolve(resolution.actual_key)
            if binding is None or binding.actual_key != resolution.actual_key:
                raise RuntimeError(
                    f"atlas compaction lost a resident page binding: {resolution.actual_key!r}"
                )
            return replace(
                resolution,
                slot=binding.slot,
                binding_generation=int(binding.binding_generation),
            )

        remapped_page_target_resolutions = {
            int(tile): refreshed_resolution(resolution)
            for tile, resolution in self.page_target_resolutions.items()
        }
        remapped_tile_page_target_resolutions = {
            int(tile): tuple(refreshed_resolution(resolution) for resolution in resolutions)
            for tile, resolutions in self.tile_page_target_resolutions.items()
        }

        # Publish the compacted physical pages and every derived presentation
        # map together, before the serial tells the layer to rebuild visuals.
        self.pages = kept
        self.tile_slots = remapped_tile_slots
        self.tile_draw_parts = remapped_draw_parts
        self.page_target_resolutions = remapped_page_target_resolutions
        self.tile_page_target_resolutions = remapped_tile_page_target_resolutions
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
            budget_slots = _max_atlas_class_capacity_within_bytes(
                tile_shape=(tile_h, tile_w),
                max_capacity=requested,
                storage_mode=storage_mode,
                max_texture_size=self.max_texture_size,
                budget_bytes=effective_budget,
            )
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
        tile_world_regions: dict[int, tuple[int, int, int, int]] | None = None,
    ) -> tuple[dict[int, tuple[float, float, float, float]], TileLayerUpdateStats]:
        # Residency is a data-keyed cache; visibility is a presentation choice.
        # A viewport commit may hide or reveal tile mappings, but it must not
        # make resident sources cold again.  Only incompatible atlas storage,
        # explicit reset/context loss, budget eviction, or a new source identity
        # can require another texture upload.
        start = perf_counter()
        tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
        payload_map = {int(key): value for key, value in dict(payloads or {}).items()}
        for payload in payload_map.values():
            _require_canonical_reduced_payload(payload, action="present")
        explicit_upserts = {
            int(key): value for key, value in dict(getattr(tile_delta, "upserts", {}) or {}).items()
        }
        raw_active_tiles = None if tile_delta is None else getattr(tile_delta, "active_tiles", None)
        if raw_active_tiles is None:
            declared_active = tuple(sorted(payload_map))
            active_tiles = declared_active
        else:
            declared_active = tuple(int(tile) for tile in tuple(raw_active_tiles))
            active_set = set(declared_active)
            # ``upserts`` is an ordered presentation command produced by the
            # canonical admission queue.  Active scope is membership only;
            # iterating it first silently replaced center-out priority with
            # montage-index order immediately before physical texture work.
            active_tiles = tuple(
                dict.fromkeys(
                    (
                        *(tile for tile in explicit_upserts if tile in active_set),
                        *declared_active,
                    )
                )
            )
        active_set = set(active_tiles)
        target_identities = dict(getattr(tile_delta, "target_identities", {}) or {})
        removed_tiles = {int(tile) for tile in tuple(getattr(tile_delta, "removals", ()) or ())}
        expected_source_ids = {
            int(tile): source_id
            for tile, source_id in dict(
                getattr(tile_delta, "near_tile_source_ids", {}) or {}
            ).items()
        }
        retained_active_keys: dict[int, object] = {}
        raw_payload_item_list: list[tuple[int, DisplayTilePayload]] = []
        identity_rejected_tiles: list[int] = []
        for tile in active_tiles:
            tile = int(tile)
            if tile not in payload_map:
                continue
            item_payload = payload_map[tile]
            if acknowledged_identity_satisfies_target(
                getattr(item_payload, "tile_identity", None) or item_payload.source_id,
                target_identities.get(tile),
            ):
                raw_payload_item_list.append((tile, item_payload))
            else:
                # Not presentable for this delta's typed target; the payload
                # is dropped from this commit.  Report it loudly — a payload
                # that can NEVER satisfy its target re-appears here on every
                # flush, and silence turned that into a starvation stall
                # (2026-07-16, session 148).
                identity_rejected_tiles.append(tile)
        raw_payload_items = tuple(raw_payload_item_list)
        storage_mode = (
            _atlas_storage_mode(raw_payload_items, rgb_already_windowed=rgb_already_windowed)
            if raw_payload_items
            else (self.storage_mode or "scalar")
        )
        payload_items = tuple(
            (tile, payload)
            for tile, payload in raw_payload_items
            if _payload_supported_by_storage_mode(
                payload, storage_mode, rgb_already_windowed=rgb_already_windowed
            )
        )
        unsupported_items = len(raw_payload_items) - len(payload_items)
        supported_payload_tiles = {int(tile) for tile, _payload in payload_items}
        retained_active_keys.update(
            {
                int(tile): self.tile_resident_keys[int(tile)]
                for tile in active_set
                if int(tile) not in removed_tiles
                and int(tile) not in supported_payload_tiles
                and int(tile) in self.tile_resident_keys
                and self.tile_resident_keys[int(tile)] in self.source_ids
                and int(tile) in self.tile_slots
                and _resident_source_matches_expected(
                    self.source_ids[self.tile_resident_keys[int(tile)]],
                    expected_source_ids.get(int(tile)),
                )
                and acknowledged_identity_satisfies_target(
                    self.acknowledged_identities.get(
                        self.tile_resident_keys[int(tile)],
                        self.source_ids[self.tile_resident_keys[int(tile)]],
                    ),
                    target_identities.get(int(tile)),
                )
            }
        )
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
            base_active_count = int(base_class_items)
            base_reserve_count = base_active_count
        requested_capacity = (
            0
            if base_active_count <= 0
            else self.requested_capacity(
                active_count=base_active_count,
                reserve_count=base_reserve_count,
                storage_mode=storage_mode,
                tile_shape=tile_shape,
                budget_bytes=budget_bytes,
            )
        )
        layout_invalidates_residency = self.tile_shape != (tile_h, tile_w)
        rebuilt = self.ensure_layout(
            tile_shape=tile_shape,
            count=requested_capacity,
            storage_mode=storage_mode,
            budget_bytes=budget_bytes,
        )
        active_keys = {_resident_key(payload) for _tile_number, payload in payload_items}
        active_keys.update(retained_active_keys.values())
        self.active_resident_keys = set(active_keys)
        # ADR 0055 G3b-2: anchored payloads with a matching world region take
        # the chunked-residency path (backend-private; identity bookkeeping
        # stays tile-level). Everything else is the classic whole-tile path.
        world_regions = {
            int(tile): tuple(int(value) for value in rect)
            for tile, rect in dict(tile_world_regions or {}).items()
        }
        chunk_plans: dict[int, tuple[_PayloadChunk, ...]] = {}
        page_backed_tiles: set[int] = set()
        for tile_number, payload in payload_items:
            region = world_regions.get(int(tile_number))
            if getattr(payload, "page_backing", None) is not None:
                if region is None:
                    raise ValueError(
                        "page-backed payload requires an exact tile world region; "
                        "legacy whole-plane fallback is forbidden"
                    )
                page_backed_tiles.add(int(tile_number))
                continue
            if region is None or not _payload_chunked_eligible(payload):
                continue
            ay0, ay1, ax0, ax1 = (int(value) for value in payload.source_anchor.source_rect)
            if (int(region[3]), int(region[2])) != (ay1 - ay0, ax1 - ax0):
                # The layout region must span the anchor's NATIVE extent
                # (world == native units; a reduced plane pixel stretches by
                # the LOD factor).  At factor 1 this is the classic 1:1
                # plane-pixel-to-world-unit requirement; any mismatch falls
                # back to the classic single-quad path.
                continue
            chunk_plans[int(tile_number)] = _payload_chunk_plan(payload)
        active_chunk_keys: set[object] = set()
        for chunks in chunk_plans.values():
            active_chunk_keys.update(chunk.key for chunk in chunks)
        for tile_number, payload in payload_items:
            if int(tile_number) in page_backed_tiles:
                active_chunk_keys.update(payload.page_backing.requested_keys)
                for target_key in payload.page_backing.requested_keys:
                    resolution = self._page_table.resolve(target_key)
                    if resolution is not None:
                        active_chunk_keys.add(resolution.actual_key)
        for tile_number in active_set:
            tile_number = int(tile_number)
            if (
                tile_number in chunk_plans
                or tile_number in page_backed_tiles
                or tile_number in removed_tiles
            ):
                continue
            active_chunk_keys.update(self.tile_chunk_residency.get(tile_number, ()))
        self.active_chunk_keys = active_chunk_keys
        protected_keys = set(active_keys) | active_chunk_keys
        self.active_base_source_ids = {
            _lod_invariant_source_id(payload.source_id) for _tile_number, payload in payload_items
        }
        self.active_base_source_ids.update(
            _lod_invariant_source_id(self.source_ids[key])
            for key in retained_active_keys.values()
            if key in self.source_ids
        )
        for tile_number in tuple(removed_tiles):
            self._clear_tile_mapping(int(tile_number))
        near = {int(tile) for tile in tuple(near_tiles or ())}
        near_keys = self._near_resident_keys(near_tile_source_ids)
        near_keys.update(
            _resident_key(payload)
            for tile_number, payload in payload_items
            if int(tile_number) in near
        )
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
        class_counts: dict[tuple[tuple[int, int], str], int] = {}
        for tile_number, payload in payload_items:
            if int(tile_number) in chunk_plans or int(tile_number) in page_backed_tiles:
                # Chunk slots live in their own shape class; the chunk
                # allocator grows it same-page as needed.
                continue
            class_shape = _payload_class_shape(payload)
            if class_shape != base_shape:
                class_mode = _atlas_storage_mode(
                    ((int(tile_number), payload),),
                    rgb_already_windowed=rgb_already_windowed,
                )
                class_key = (class_shape, class_mode)
                class_counts[class_key] = class_counts.get(class_key, 0) + 1
        for (class_shape, class_mode), class_count in class_counts.items():
            self._ensure_class_capacity(
                class_shape,
                class_count,
                storage_mode=class_mode,
            )
        capacity_skipped_tiles: set[int] = set()

        for tile_number, payload in payload_items:
            resident_key = _resident_key(payload)
            if int(tile_number) in page_backed_tiles:
                committed = self._commit_page_backed_payload(
                    int(tile_number),
                    payload,
                    world_region=world_regions[int(tile_number)],
                    protected_keys=protected_keys,
                    near_keys=near_keys,
                    rgb_already_windowed=rgb_already_windowed,
                )
                if committed is None:
                    capacity_skipped_tiles.add(int(tile_number))
                    skipped += 1
                    continue
                active_tile_slots[int(tile_number)] = (
                    int(committed.page_index),
                    int(committed.slot),
                )
                active_tile_keys[int(tile_number)] = resident_key
                uvs[tile_number] = committed.uv
                active_tile_uvs[int(tile_number)] = committed.uv
                uploads += committed.uploads
                upload_bytes += committed.upload_bytes
                complex_uploads += committed.complex_uploads
                texture_prepare_ms += committed.prepare_ms
                texture_submit_ms += committed.submit_ms
                if committed.uploaded_any:
                    uploaded_keys.add(resident_key)
                    updated += 1
                else:
                    skipped += 1
                continue
            chunks = chunk_plans.get(int(tile_number))
            if chunks is not None:
                committed = self._commit_chunked_payload(
                    int(tile_number),
                    payload,
                    chunks,
                    world_region=world_regions[int(tile_number)],
                    protected_keys=protected_keys,
                    near_keys=near_keys,
                    rgb_already_windowed=rgb_already_windowed,
                )
                if committed is None:
                    # Chunk set cannot be placed (same-page capacity): fall
                    # back cleanly to the classic whole-tile path below.
                    chunk_plans.pop(int(tile_number), None)
                else:
                    active_tile_slots[int(tile_number)] = (
                        int(committed.page_index),
                        int(committed.slot),
                    )
                    active_tile_keys[int(tile_number)] = resident_key
                    uvs[tile_number] = committed.uv
                    active_tile_uvs[int(tile_number)] = committed.uv
                    uploads += committed.uploads
                    upload_bytes += committed.upload_bytes
                    complex_uploads += committed.complex_uploads
                    texture_prepare_ms += committed.prepare_ms
                    texture_submit_ms += committed.submit_ms
                    if committed.uploaded_any:
                        uploaded_keys.add(resident_key)
                        updated += 1
                    else:
                        skipped += 1
                    continue
            class_shape = _payload_class_shape(payload)
            class_mode = _atlas_storage_mode(
                ((int(tile_number), payload),),
                rgb_already_windowed=rgb_already_windowed,
            )
            try:
                page_index, slot, newly_assigned = self._slot_for(
                    resident_key,
                    active_keys=protected_keys,
                    near_keys=near_keys,
                    tile_shape=class_shape,
                    storage_mode=class_mode,
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
            acknowledged_changed = self.acknowledged_identities.get(
                resident_key
            ) != tile_ack_identity(payload)
            missing_uploaded_source = resident_key not in self.source_ids
            # A drawn tile must always have a physical upload record (tile
            # truth).  Records travel with source_ids on every reclamation
            # path, so this is normally redundant — it self-heals any legacy
            # residency that predates record-keeping instead of presenting a
            # truth-less tile.
            missing_physical_record = resident_key not in self.physical_upload_records
            should_upload = bool(
                layout_invalidates_residency
                or newly_assigned
                or source_changed
                or acknowledged_changed
                or missing_uploaded_source
                or missing_physical_record
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
            self.acknowledged_identities[resident_key] = (
                getattr(payload, "tile_identity", None) or payload.source_id
            )
            upload_plane = scalar if scalar is not None else color
            real_plane, imag_plane = array_plane_identities(upload_plane)
            self.physical_upload_records[resident_key] = {
                "physical_texture_kind": _payload_texture_kind(payload).value,
                "physical_storage_mode": str(page.storage_mode),
                "physical_texture_dtype": str(np.asarray(upload_plane).dtype),
                "physical_texture_shape": tuple(
                    int(value) for value in np.asarray(upload_plane).shape
                ),
                "physical_real_plane_identity": plane_identity_record(real_plane),
                "physical_imag_plane_identity": plane_identity_record(imag_plane),
            }
            updated += 1

        payload_presented_tiles = tuple(
            int(tile_number)
            for tile_number, payload in payload_items
            if int(tile_number) in active_tile_slots
            and self.source_ids.get(_resident_key(payload)) == payload.source_id
            and self.acknowledged_identities.get(_resident_key(payload))
            == tile_ack_identity(payload)
        )
        presented_tiles = tuple(
            dict.fromkeys(
                (
                    *payload_presented_tiles,
                    *(
                        int(tile)
                        for tile, key in sorted(retained_active_keys.items())
                        if int(tile) not in removed_tiles
                        and key in self.source_ids
                        and int(tile) in self.tile_slots
                    ),
                    *(
                        int(tile)
                        for tile in sorted(capacity_skipped_tiles)
                        if int(tile) not in removed_tiles
                        and int(tile) in self.tile_resident_keys
                        and self.tile_resident_keys[int(tile)] in self.source_ids
                        and int(tile) in self.tile_slots
                    ),
                )
            )
        )
        presented_set = set(presented_tiles)
        for tile in active_set - presented_set:
            resident_key = self.tile_resident_keys.get(int(tile))
            acknowledged = (
                None
                if resident_key is None
                else self.acknowledged_identities.get(
                    resident_key,
                    self.source_ids.get(resident_key),
                )
            )
            if not acknowledged_identity_satisfies_target(
                acknowledged,
                target_identities.get(int(tile)),
            ):
                self._clear_tile_mapping(int(tile))
        # A bounded commit may include only the first few replacement
        # payloads while many active tiles still have a valid older LOD
        # mapping.  Absence from this commit's payload map is not a removal:
        # keep the old mapping until an explicit removal or an acknowledged
        # replacement arrives.  Otherwise zoom/retarget drains replace
        # correct lower-quality pixels with black holes.
        level_swaps_zero_upload = 0
        level_swaps_with_upload = 0
        for tile in payload_presented_tiles:
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
                chunked=int(tile) in chunk_plans or int(tile) in page_backed_tiles,
                page_backed=int(tile) in page_backed_tiles,
            )
        # Reclaim identity records of chunked keys that no tile presents
        # anymore, now that every mapping of this commit has settled (the
        # per-call forget skips active keys so a retarget cannot destroy a
        # key's records between its displacing and re-presenting tiles).
        self._sweep_unreferenced_chunked_keys()
        self.lod_level_swaps_zero_upload += level_swaps_zero_upload
        self.lod_level_swaps_with_upload += level_swaps_with_upload
        uvs = self.tile_uvs
        reported_presented_tiles = tuple(
            tile for tile in declared_active if int(tile) in presented_set
        )
        # Mipmap regens run at draw time (visual-side); stats report the
        # regens completed since the previous update, one commit behind.
        mipmap_updates_total = sum(
            int(getattr(page, "mipmap_updates", 0) or 0) for page in self.pages
        )
        mipmap_updates_delta = max(0, mipmap_updates_total - self._mipmap_updates_reported)
        self._mipmap_updates_reported = mipmap_updates_total
        mipmap_available = any(bool(getattr(page, "mipmap_ready", False)) for page in self.pages)
        elapsed = (perf_counter() - start) * 1000.0 if updated or rebuilt else 0.0
        return uvs, TileLayerUpdateStats(
            visible_items=len(reported_presented_tiles),
            presented_tiles=reported_presented_tiles,
            presented_identities=self.presented_identities(),
            committed_upserts=tuple(
                int(tile) for tile in explicit_upserts if int(tile) in presented_set
            ),
            identity_rejected_items=len(identity_rejected_tiles),
            identity_rejected_tiles=tuple(identity_rejected_tiles),
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
            active_pages=len(
                {
                    self.tile_slots[int(tile)][0]
                    for tile in presented_set
                    if int(tile) in self.tile_slots
                }
            ),
            device_max_texture_size=self.max_texture_size,
            budget_bytes=self.budget_bytes,
            near_resident_items=len(near_keys.intersection(self.source_ids)),
            warm_resident_items=max(0, self.resident_count - len(self.active_resident_keys)),
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
        for _tile_number, payload in payload_items:
            _require_canonical_reduced_payload(payload, action="warm")
        requested_mode = _atlas_storage_mode(
            payload_items, rgb_already_windowed=rgb_already_windowed
        )
        page_warm = None
        if any(
            getattr(payload, "page_backing", None) is not None for _tile, payload in payload_items
        ):
            page_items = tuple(
                (tile_number, payload)
                for tile_number, payload in payload_items
                if getattr(payload, "page_backing", None) is not None
            )
            payload_items = tuple(
                (tile_number, payload)
                for tile_number, payload in payload_items
                if getattr(payload, "page_backing", None) is None
            )
            page_warm = self._warm_page_backed_items(
                page_items,
                rgb_already_windowed=rgb_already_windowed,
            )
        # ADR 0055 G4c: anchored payloads warm as pure chunk residency — the
        # later visible commit finds their chunks via _chunk_slots_for and
        # uploads nothing. Classic whole-tile warming would be useless for
        # them (the chunked visible path never consults whole-tile slots).
        chunk_warm = None
        if any(_payload_chunked_eligible(payload) for _tile, payload in payload_items):
            chunk_items = tuple(
                (tile_number, payload)
                for tile_number, payload in payload_items
                if _payload_chunked_eligible(payload)
            )
            payload_items = tuple(
                (tile_number, payload)
                for tile_number, payload in payload_items
                if not _payload_chunked_eligible(payload)
            )
            chunk_warm = self._warm_anchored_chunk_items(
                chunk_items,
                rgb_already_windowed=rgb_already_windowed,
            )
        residency_warm = _merge_warm_counters(page_warm, chunk_warm)
        if not payload_items:
            return TileLayerUpdateStats(
                resident_items=self.resident_count,
                storage_capacity=self.capacity,
                texture_uploads=residency_warm["uploads"],
                texture_upload_bytes=residency_warm["upload_bytes"],
                items_updated=residency_warm["updated"],
                items_skipped=residency_warm["skipped"],
                estimated_gpu_bytes=self.estimated_gpu_bytes,
                cpu_shadow_bytes=self.cpu_shadow_bytes,
                upload_ms=(perf_counter() - start) * 1000.0 if residency_warm["updated"] else 0.0,
                texture_prepare_ms=residency_warm["prepare_ms"],
                texture_submit_ms=residency_warm["submit_ms"],
                page_count=len(self.pages),
                device_max_texture_size=self.max_texture_size,
                budget_bytes=self.budget_bytes,
                warm_resident_items=max(0, self.resident_count - len(self.active_resident_keys)),
                complex_texture_uploads=residency_warm["complex_uploads"],
                capacity_warning=_warm_capacity_warning(page_warm, chunk_warm),
            )
        target_count = len(
            set(self.active_resident_keys).union(
                _resident_key(payload) for _tile, payload in payload_items
            )
        )
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
        near_keys.update(_resident_key(payload) for _tile, payload in payload_items)
        # Mixed batches fold the (already completed) chunk-warm counters in.
        uploads = int(residency_warm["uploads"])
        upload_bytes = int(residency_warm["upload_bytes"])
        complex_uploads = int(residency_warm["complex_uploads"])
        texture_prepare_ms = float(residency_warm["prepare_ms"])
        texture_submit_ms = float(residency_warm["submit_ms"])
        updated = int(residency_warm["updated"])
        skipped = int(residency_warm["skipped"])
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
            payload_mode = _atlas_storage_mode(
                ((_tile_number, payload),),
                rgb_already_windowed=rgb_already_windowed,
            )
            if not _payload_supported_by_storage_mode(
                payload,
                payload_mode,
                rgb_already_windowed=rgb_already_windowed,
            ):
                skipped += 1
                continue
            resident_key = _resident_key(payload)
            if self.source_ids.get(
                resident_key
            ) == payload.source_id and self.acknowledged_identities.get(
                resident_key
            ) == tile_ack_identity(payload):
                self._touch(resident_key)
                skipped += 1
                continue
            if resident_key not in self._page_table and new_warm_budget <= 0:
                skipped += 1
                skipped_budget += 1
                continue
            if resident_key not in self._page_table:
                new_warm_budget -= 1
            class_shape = _payload_class_shape(payload)
            if class_shape != (tile_h, tile_w):
                self._ensure_class_capacity(
                    class_shape,
                    1,
                    storage_mode=payload_mode,
                )
            try:
                page_index, slot, _newly_assigned = self._slot_for(
                    resident_key,
                    active_keys=set(self.active_resident_keys) | set(self.active_chunk_keys),
                    near_keys=near_keys,
                    tile_shape=class_shape,
                    storage_mode=payload_mode,
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
            self.acknowledged_identities[resident_key] = (
                getattr(payload, "tile_identity", None) or payload.source_id
            )
            # Warm uploads are real texels: record them like the visible
            # path does.  Without this, a warm→visible promotion presents
            # through the acknowledged-identity skip (no re-upload) and the
            # drawn tile has no physical truth row (field defect 2026-07-15:
            # ``phys None/None`` on prefetch-warmed montage tiles).
            upload_plane = scalar if scalar is not None else color
            if upload_plane is not None:
                real_plane, imag_plane = array_plane_identities(upload_plane)
                self.physical_upload_records[resident_key] = {
                    "physical_texture_kind": _payload_texture_kind(payload).value,
                    "physical_storage_mode": str(page.storage_mode),
                    "physical_texture_dtype": str(np.asarray(upload_plane).dtype),
                    "physical_texture_shape": tuple(
                        int(value) for value in np.asarray(upload_plane).shape
                    ),
                    "physical_real_plane_identity": plane_identity_record(real_plane),
                    "physical_imag_plane_identity": plane_identity_record(imag_plane),
                }
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

    def _warm_page_backed_items(
        self,
        payload_items: tuple[tuple[int, DisplayTilePayload], ...],
        *,
        rgb_already_windowed: bool,
    ) -> dict[str, object]:
        """Upload canonical pages without changing tile mappings or owner pins."""

        counters = _empty_warm_counters()
        protected = set(self.active_resident_keys) | set(self.active_chunk_keys)
        for _tile_number, payload in payload_items:
            if not self.pages or self.storage_mode is None:
                counters["skipped"] += 1
                counters["skipped_denied"] += 1
                continue
            uploaded = self._upload_page_backed_values(
                payload,
                protected_keys=protected,
                near_keys=set(),
                rgb_already_windowed=rgb_already_windowed,
                allow_eviction=False,
            )
            if uploaded is None:
                counters["skipped"] += 1
                counters["skipped_denied"] += 1
                continue
            counters["uploads"] += uploaded.uploads
            counters["upload_bytes"] += uploaded.upload_bytes
            counters["complex_uploads"] += uploaded.complex_uploads
            counters["prepare_ms"] += uploaded.prepare_ms
            counters["submit_ms"] += uploaded.submit_ms
            if uploaded.uploaded_keys:
                counters["updated"] += 1
            else:
                counters["skipped"] += 1
        return counters

    def _warm_anchored_chunk_items(
        self,
        payload_items: tuple[tuple[int, DisplayTilePayload], ...],
        *,
        rgb_already_windowed: bool,
    ) -> dict[str, object]:
        """Warm anchored payloads as pure chunk residency (ADR 0055 G4c).

        Chunks become page-table residents that a later visible commit finds
        via ``_chunk_slots_for`` and reuses upload-free.  Nothing is
        presented: no draw parts, no tile mapping, no tile-level identity or
        acknowledgement records.  The branch is strictly speculative — it
        never evicts anything (free slots plus budgeted chunk-class growth
        only; ``_ensure_class_capacity`` enforces ``budget_bytes``) and never
        establishes or rebuilds the atlas layout.  Denied placements are
        skipped silently; the pool budget itself is owned by visible commits
        and is deliberately not reconfigured here.
        """

        counters = _empty_warm_counters()
        protected = set(self.active_resident_keys) | set(self.active_chunk_keys)
        for _tile_number, payload in payload_items:
            if not self.pages or self.storage_mode is None:
                # Warm work never creates the atlas layout: without a prior
                # visible commit there is nothing to be adjacent to.
                counters["skipped"] += 1
                counters["skipped_denied"] += 1
                continue
            payload_mode = _atlas_storage_mode(
                ((_tile_number, payload),),
                rgb_already_windowed=rgb_already_windowed,
            )
            if not _payload_supported_by_storage_mode(
                payload, payload_mode, rgb_already_windowed=rgb_already_windowed
            ):
                counters["skipped"] += 1
                continue
            chunks = _payload_chunk_plan(payload)
            if all(self._page_table.lookup(chunk.key) is not None for chunk in chunks):
                for chunk in chunks:
                    self._page_table.touch(chunk.key)
                counters["skipped"] += 1
                continue
            try:
                slots = self._chunk_slots_for(
                    tuple(chunk.key for chunk in chunks),
                    protected_keys=protected,
                    near_keys=set(),
                    allow_eviction=False,
                    storage_mode=payload_mode,
                )
            except AtlasCapacityError:
                counters["skipped"] += 1
                counters["skipped_denied"] += 1
                continue
            need_upload = tuple(chunk for chunk in chunks if slots[chunk.key][2])
            if not need_upload:
                counters["skipped"] += 1
                continue
            page = self.pages[int(slots[chunks[0].key][0])]
            scalar, color, prepare_ms = _prepare_payload_texture_data(
                payload,
                tile_shape=_payload_class_shape(payload),
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=page.scalar_is_atlas,
                need_color=page.color_is_atlas,
            )
            counters["prepare_ms"] += prepare_ms
            for chunk in need_upload:
                py0, py1, px0, px1 = chunk.plane_rect
                _page_index, slot, _newly = slots[chunk.key]
                y_off, x_off = page.offset_for_slot(int(slot))
                if scalar is not None:
                    sub = np.ascontiguousarray(scalar[py0:py1, px0:px1])
                    counters["submit_ms"] += _upload_texture_plane(
                        page.scalar_texture,
                        sub,
                        offset=(int(y_off), int(x_off)),
                        copy=_upload_copy_required(sub, payload, force=page.complex_is_atlas),
                    )
                    counters["uploads"] += 1
                    counters["upload_bytes"] += int(sub.nbytes)
                    if page.complex_is_atlas:
                        counters["complex_uploads"] += 1
                if color is not None:
                    sub = np.ascontiguousarray(color[py0:py1, px0:px1])
                    counters["submit_ms"] += _upload_texture_plane(
                        page.color_texture,
                        sub,
                        offset=(int(y_off), int(x_off)),
                        copy=_upload_copy_required(sub, payload),
                    )
                    counters["uploads"] += 1
                    counters["upload_bytes"] += int(sub.nbytes)
            if page.mipmap_levels:
                page.mipmap_dirty = True
            self.chunk_upload_count += len(need_upload)
            self.chunk_reuse_count += len(chunks) - len(need_upload)
            counters["updated"] += 1
        return counters

    def _slot_for(
        self,
        resident_key: object,
        *,
        active_keys: set[object],
        near_keys: set[object],
        tile_shape: tuple[int, int] | None = None,
        storage_mode: str | None = None,
        allow_eviction: bool = True,
    ) -> tuple[int, int, bool]:
        current = self._page_table.lookup(resident_key)
        if current is not None:
            expected_mode = _normalize_storage_mode(storage_mode or self.storage_mode or "scalar")
            actual_mode = self.pages[int(current.page_index)].storage_mode
            if actual_mode != expected_mode:
                raise ValueError(
                    "resident key aliases incompatible atlas storage modes: "
                    f"{resident_key!r} is {actual_mode}, requested {expected_mode}"
                )
            return current.page_index, current.slot_index, False

        shape = self.tile_shape if tile_shape is None else (int(tile_shape[0]), int(tile_shape[1]))
        mode = _normalize_storage_mode(storage_mode or self.storage_mode or "scalar")
        class_pages = tuple(
            (page_index, page)
            for page_index, page in enumerate(self.pages)
            if shape is None or page.tile_shape == shape
            if page.storage_mode == mode
        )
        for page_index, page in class_pages:
            slot = page.take_free_slot(resident_key)
            if slot is None:
                continue
            self._bind_resident_slot(resident_key, page_index, slot, page)
            return int(page_index), int(slot), True

        if self.budget_bytes > 0 and shape is not None:
            # Budgeted pools grow before evicting warm residency: reclaiming
            # a warm slot while byte headroom remains would silently destroy
            # cross-plane reuse (index scroll-back re-uploads).
            self._ensure_class_capacity(
                shape,
                self._class_capacity(shape, storage_mode=mode) + 1,
                storage_mode=mode,
            )
            for page_index, page in enumerate(self.pages):
                if page.tile_shape != shape or page.storage_mode != mode:
                    continue
                slot = page.take_free_slot(resident_key)
                if slot is None:
                    continue
                self._bind_resident_slot(resident_key, page_index, slot, page)
                return int(page_index), int(slot), True

        if not allow_eviction:
            raise AtlasCapacityError(f"atlas has no free speculative slot of shape {shape}")

        candidates = []
        active_bases = self.active_base_source_ids
        for page_index, page in class_pages:
            for slot, owner in enumerate(page.slot_owners):
                if (
                    owner is not None
                    and owner not in active_keys
                    and not self._page_table.is_pinned(owner)
                ):
                    # Eviction preference under pressure (ADR 0050): first
                    # superseded classes of tiles that are neither active nor
                    # near, then presented classes of non-active tiles, and
                    # only as a last resort the superseded (retained
                    # adjacent-level) classes of active tiles, whose loss
                    # forces a re-upload on the next level flip.  A presented
                    # class of an active tile is never a candidate at all.
                    superseded = owner in self.superseded_keys and not self.resident_tiles.get(
                        owner
                    )
                    adjacent = (
                        superseded
                        and _lod_invariant_source_id(self.source_ids.get(owner)) in active_bases
                    )
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
                    candidates.append(
                        (rank, self._page_table.last_use(owner), owner, int(page_index), int(slot))
                    )
        if not candidates:
            raise AtlasCapacityError(
                f"atlas has {self._class_capacity(shape, storage_mode=mode) if shape else self.capacity} "
                f"slots of shape {shape} and storage {mode} "
                f"but {len(active_keys)} active tiles require residency"
            )
        _priority, _last, victim, page_index, slot = min(
            candidates, key=lambda item: (item[0], item[1], repr(item[2]))
        )
        self._release_victim(victim, near_keys=near_keys)
        page = self.pages[int(page_index)]
        page.slot_owners[int(slot)] = resident_key
        self._bind_resident_slot(resident_key, page_index, slot, page)
        return int(page_index), int(slot), True

    def _release_victim(self, victim: object, *, near_keys: set[object]) -> None:
        """Evict one resident key: drop its slot binding and identity records.

        Works for classic tile-level keys and for anchored chunk keys — the
        tile-mapping discard below invalidates the owning tile(s) of a chunk
        victim so the next commit re-uploads that chunk.
        """

        self._discard_tile_mappings_for_resident_key(victim)
        self._page_table.unbind(victim)
        self.source_ids.pop(victim, None)
        self.acknowledged_identities.pop(victim, None)
        self.physical_upload_records.pop(victim, None)
        self.page_plans.pop(victim, None)
        if victim in self.superseded_keys:
            self.superseded_reclaimed_count += 1
        self.superseded_keys.discard(victim)
        self.eviction_count += 1
        if victim in near_keys:
            self.evicted_near_count += 1

    def _bind_resident_slot(
        self, resident_key: object, page_index: int, slot: int, page: TextureAtlasPage
    ) -> None:
        nbytes = (
            _storage_mode_bytes_per_pixel(page.storage_mode)
            * page.tile_shape[0]
            * page.tile_shape[1]
        )
        self._page_table.bind(
            resident_key,
            PageSlot("vispy-atlas", int(page_index), int(slot)),
            nbytes=nbytes,
        )

    def _chunk_slots_for(
        self,
        chunk_keys: tuple[object, ...],
        *,
        protected_keys: set[object],
        near_keys: set[object],
        allow_eviction: bool = True,
        slot_shape: tuple[int, int] | None = None,
        storage_mode: str | None = None,
    ) -> dict[object, tuple[int, int, bool]]:
        """Place one tile's chunk set, all on the SAME page.

        The montage layer buckets a tile to one visual via its (page, slot),
        so a tile's chunks may never straddle pages. Prefers the page already
        holding the most of this set; grows the chunk shape class when no
        page has room. Raises :class:`AtlasCapacityError` when the set cannot
        be placed on any single page (caller falls back to the classic path).

        ``allow_eviction=False`` restricts placement to genuinely free slots
        plus budgeted class growth (ADR 0055 G4c): speculative warm work must
        never destroy resident content, so denial raises instead of evicting.
        """

        shape = tuple(int(value) for value in (slot_shape or ANCHORED_CHUNK_SHAPE))
        mode = _normalize_storage_mode(storage_mode or self.storage_mode or "scalar")
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"chunk slot shape must be positive 2D, got {shape}")
        keys = tuple(chunk_keys)
        key_set = set(keys)
        lookups = {key: self._page_table.lookup(key) for key in keys}
        incompatible = tuple(
            key
            for key, slot_ref in lookups.items()
            if slot_ref is not None and self.pages[int(slot_ref.page_index)].storage_mode != mode
        )
        if incompatible:
            raise ValueError(
                "chunk keys alias incompatible atlas storage modes: "
                f"{incompatible!r} requested {mode}"
            )
        votes: dict[int, int] = {}
        for slot_ref in lookups.values():
            if slot_ref is not None:
                votes[slot_ref.page_index] = votes.get(slot_ref.page_index, 0) + 1

        def class_page_indices() -> tuple[int, ...]:
            return tuple(
                index
                for index, page in enumerate(self.pages)
                if page.tile_shape == shape and page.storage_mode == mode
            )

        def page_headroom(page_index: int, *, allow_eviction: bool) -> bool:
            page = self.pages[page_index]
            resident_here = sum(
                1
                for slot_ref in lookups.values()
                if slot_ref is not None and slot_ref.page_index == page_index
            )
            needed = len(keys) - resident_here
            free = sum(1 for owner in page.slot_owners if owner is None)
            if free >= needed:
                return True
            if not allow_eviction:
                return False
            evictable = sum(
                1
                for owner in page.slot_owners
                if (
                    owner is not None
                    and owner not in protected_keys
                    and owner not in key_set
                    and not self._page_table.is_pinned(owner)
                )
            )
            return free + evictable >= needed

        def scan(*, allow_eviction: bool) -> int | None:
            ordered = sorted(votes, key=lambda index: -votes[index])
            ordered.extend(index for index in class_page_indices() if index not in votes)
            return next(
                (index for index in ordered if page_headroom(index, allow_eviction=allow_eviction)),
                None,
            )

        def grow_and_rescan(*, allow_eviction: bool) -> int | None:
            self._ensure_class_capacity(
                shape,
                self._class_capacity(shape, storage_mode=mode) + len(keys),
                storage_mode=mode,
            )
            return scan(allow_eviction=allow_eviction)

        # Placement preference: a page (ideally already holding most of this
        # set) with genuinely free room first. With an explicit byte budget,
        # grow within it BEFORE evicting warm (non-protected) chunks —
        # evicting warm residency while budget headroom remains silently
        # destroys cross-plane reuse (index scroll-back). Budget-less pools
        # keep the legacy fixed-capacity order (evict, then grow as a last
        # resort).
        chosen = scan(allow_eviction=False)
        if chosen is None and self.budget_bytes > 0:
            chosen = grow_and_rescan(allow_eviction=False)
        if chosen is None and allow_eviction:
            chosen = scan(allow_eviction=True)
        if chosen is None and allow_eviction:
            chosen = grow_and_rescan(allow_eviction=True)
        if chosen is None:
            raise AtlasCapacityError(
                f"no atlas page of shape {shape} can hold {len(keys)} chunks of one tile"
            )
        page = self.pages[int(chosen)]
        if not allow_eviction:
            # Eviction-free (speculative warm) placements must never disturb
            # existing residency: relocating a foreign-page resident releases
            # its slot and invalidates any tile drawn from it, which violates
            # "speculative work never changes visible presentation outcomes".
            # Deny the set instead; a visible commit has relocation rights.
            for key in keys:
                slot_ref = lookups[key]
                if slot_ref is not None and slot_ref.page_index != int(chosen):
                    raise AtlasCapacityError(
                        "eviction-free chunk placement would relocate a resident chunk "
                        f"from page {slot_ref.page_index} to {int(chosen)}"
                    )
        results: dict[object, tuple[int, int, bool]] = {}
        for key in keys:
            slot_ref = lookups[key]
            if slot_ref is not None and slot_ref.page_index == int(chosen):
                self._page_table.touch(key)
                results[key] = (int(chosen), int(slot_ref.slot_index), False)
                continue
            if slot_ref is not None:
                # Resident on another page: same-page bucketing wins. Release
                # the foreign slot (invalidating any tile drawn from it) and
                # re-upload into this tile's page.
                self._discard_tile_mappings_for_resident_key(key)
                foreign = self.pages[int(slot_ref.page_index)]
                foreign.slot_owners[int(slot_ref.slot_index)] = None
                foreign._free_slots.append(int(slot_ref.slot_index))
                self._page_table.unbind(key)
            slot = page.take_free_slot(key)
            if slot is None:
                if not allow_eviction:
                    # Unreachable given the headroom precondition; if it ever
                    # fires, roll back this set's fresh binds so no key stays
                    # "resident" without its texels ever being uploaded.
                    for bound_key, (bound_page_index, bound_slot, newly) in results.items():
                        if not newly:
                            continue
                        bound_page = self.pages[int(bound_page_index)]
                        bound_page.slot_owners[int(bound_slot)] = None
                        bound_page._free_slots.append(int(bound_slot))
                        self._page_table.unbind(bound_key)
                    raise AtlasCapacityError(
                        f"atlas page {chosen} ran out of free chunk slots during an eviction-free placement"
                    )
                slot = self._evict_page_victim(
                    int(chosen),
                    protected_keys=protected_keys | key_set,
                    near_keys=near_keys,
                )
                page.slot_owners[int(slot)] = key
            self._bind_resident_slot(key, int(chosen), int(slot), page)
            results[key] = (int(chosen), int(slot), True)
        return results

    def _evict_page_victim(
        self,
        page_index: int,
        *,
        protected_keys: set[object],
        near_keys: set[object],
    ) -> int:
        """Evict the best victim on ONE page and return its freed slot index."""

        page = self.pages[int(page_index)]
        candidates = []
        active_bases = self.active_base_source_ids
        for slot, owner in enumerate(page.slot_owners):
            if owner is None or owner in protected_keys or self._page_table.is_pinned(owner):
                continue
            superseded = owner in self.superseded_keys and not self.resident_tiles.get(owner)
            adjacent = (
                superseded and _lod_invariant_source_id(self.source_ids.get(owner)) in active_bases
            )
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
            candidates.append((rank, self._page_table.last_use(owner), owner, int(slot)))
        if not candidates:
            raise AtlasCapacityError(
                f"atlas page {page_index} has no evictable slot for a chunk allocation"
            )
        _rank, _last, victim, slot = min(
            candidates, key=lambda item: (item[0], item[1], repr(item[2]))
        )
        self._release_victim(victim, near_keys=near_keys)
        return int(slot)

    def _upload_page_backed_values(
        self,
        payload: DisplayTilePayload,
        *,
        protected_keys: set[object],
        near_keys: set[object],
        rgb_already_windowed: bool,
        allow_eviction: bool,
    ) -> _PageUpload | None:
        """Upload checked canonical pages without changing presentation state."""

        backing = payload.page_backing
        if backing is None:
            raise ValueError("page-backed upload requires page_backing")
        plans = tuple(backing.requested_plans)
        slot_shape = tuple(int(value) for value in plans[0].stored_page_shape)
        storage_mode = _atlas_storage_mode(
            ((int(payload.tile_number), payload),),
            rgb_already_windowed=rgb_already_windowed,
        )
        if any(tuple(plan.stored_page_shape) != slot_shape for plan in plans):
            raise ValueError("one page-backed payload cannot mix physical page shapes")

        uploads = 0
        upload_bytes = 0
        complex_uploads = 0
        prepare_ms = 0.0
        submit_ms = 0.0
        uploaded_keys: list[DataChunkKey] = []
        missing_keys = {
            page.key
            for page in backing.materialized_pages
            if self._page_table.lookup(page.key) is None
        }
        if missing_keys:
            free_slots = sum(
                owner is None
                for page in self.pages
                if page.tile_shape == slot_shape and page.storage_mode == storage_mode
                for owner in page.slot_owners
            )
            shortage = max(0, len(missing_keys) - int(free_slots))
        else:
            shortage = 0
        if shortage:
            try:
                self._ensure_class_capacity(
                    slot_shape,
                    self._class_capacity(slot_shape, storage_mode=storage_mode) + int(shortage),
                    storage_mode=storage_mode,
                )
            except AtlasCapacityError:
                return None
        for materialized in backing.materialized_pages:
            key = materialized.key
            existing_plan = self.page_plans.get(key)
            if existing_plan is not None and existing_plan != materialized.plan:
                raise RuntimeError(
                    "resident page key was supplied with different canonical geometry"
                )
            slot_ref = self._page_table.lookup(key)
            newly_assigned = slot_ref is None
            try:
                if slot_ref is None:
                    page_index, slot, newly_assigned = self._slot_for(
                        key,
                        active_keys=protected_keys | set(backing.requested_keys),
                        near_keys=near_keys,
                        tile_shape=slot_shape,
                        storage_mode=storage_mode,
                        allow_eviction=allow_eviction,
                    )
                else:
                    page_index, slot = int(slot_ref.page_index), int(slot_ref.slot_index)
                    self._page_table.touch(key)
            except AtlasCapacityError:
                return None
            if not newly_assigned:
                self.page_plans[key] = materialized.plan
                continue
            page = self.pages[int(page_index)]
            texture_kind = (
                TexturePlaneKind.COMPLEX_RG32F
                if key.representation == COMPLEX_RG32F
                else (
                    TexturePlaneKind.RGB8
                    if key.representation == RGB8
                    else TexturePlaneKind.SCALAR_R32F
                )
            )
            page_payload = replace(
                payload,
                image=materialized.values,
                texture_data=materialized.values,
                texture_kind=texture_kind,
                semantic_data=None,
                semantic_histogram_data=None,
                page_backing=None,
                source_id=key,
            )
            scalar, color, elapsed = _prepare_payload_texture_data(
                page_payload,
                tile_shape=slot_shape,
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=page.scalar_is_atlas,
                need_color=page.color_is_atlas,
            )
            prepare_ms += elapsed
            y_off, x_off = page.offset_for_slot(int(slot))
            if scalar is not None:
                submit_ms += _upload_texture_plane(
                    page.scalar_texture,
                    scalar,
                    offset=(int(y_off), int(x_off)),
                    copy=_upload_copy_required(scalar, page_payload, force=page.complex_is_atlas),
                )
                uploads += 1
                upload_bytes += int(scalar.nbytes)
                complex_uploads += int(page.complex_is_atlas)
            if color is not None:
                submit_ms += _upload_texture_plane(
                    page.color_texture,
                    color,
                    offset=(int(y_off), int(x_off)),
                    copy=_upload_copy_required(color, page_payload),
                )
                uploads += 1
                upload_bytes += int(color.nbytes)
            if (scalar is not None or color is not None) and page.mipmap_levels:
                page.mipmap_dirty = True
            record_plane = scalar if scalar is not None else color
            self.source_ids[key] = key
            self.acknowledged_identities[key] = key
            if record_plane is not None:
                real_plane, imag_plane = array_plane_identities(record_plane)
                self.physical_upload_records[key] = {
                    "physical_texture_kind": texture_kind.value,
                    "physical_storage_mode": str(page.storage_mode),
                    "physical_texture_dtype": str(np.asarray(record_plane).dtype),
                    "physical_texture_shape": tuple(np.asarray(record_plane).shape),
                    "physical_real_plane_identity": plane_identity_record(real_plane),
                    "physical_imag_plane_identity": plane_identity_record(imag_plane),
                }
            self.page_plans[key] = materialized.plan
            uploaded_keys.append(key)

        self.chunk_upload_count += len(uploaded_keys)
        self.chunk_reuse_count += max(0, len(backing.materialized_pages) - len(uploaded_keys))
        return _PageUpload(
            uploads=uploads,
            upload_bytes=upload_bytes,
            complex_uploads=complex_uploads,
            prepare_ms=prepare_ms,
            submit_ms=submit_ms,
            uploaded_keys=tuple(uploaded_keys),
        )

    def _commit_page_backed_payload(
        self,
        tile_number: int,
        payload: DisplayTilePayload,
        *,
        world_region: tuple[int, int, int, int],
        protected_keys: set[object],
        near_keys: set[object],
        rgb_already_windowed: bool,
    ) -> _ChunkCommit | None:
        """Admit checked pages, then atomically resolve and bind targets."""

        backing = payload.page_backing
        if backing is None:
            raise ValueError("page-backed commit requires page_backing")
        plans = tuple(backing.requested_plans)
        uploaded = self._upload_page_backed_values(
            payload,
            protected_keys=protected_keys,
            near_keys=near_keys,
            rgb_already_windowed=rgb_already_windowed,
            allow_eviction=True,
        )
        if uploaded is None:
            return None

        owner_scope = (
            "presentation",
            getattr(payload, "presentation_identity", None),
            getattr(payload, "tile_identity", None),
        )
        resolved = self.resolve_tile_page_targets(
            {int(tile_number): backing.requested_keys},
            owner_scope=owner_scope,
        )[int(tile_number)]
        if resolved is None:
            # Candidate fine coverage is incomplete. The resolver retained
            # the previous complete pinned set and draw mapping.
            return None

        plan_by_key = {plan.key: plan for plan in plans}
        coverage_y0, coverage_y1, coverage_x0, coverage_x1 = backing.source_coverage_yx
        region_x, region_y, region_w, region_h = (int(value) for value in world_region)
        coverage_h, coverage_w = coverage_y1 - coverage_y0, coverage_x1 - coverage_x0
        parts: list[TileDrawPart] = []
        for resolution in resolved:
            plan = plan_by_key[resolution.target_key]
            actual_plan = self.page_plans.get(resolution.actual_key)
            if actual_plan is None:
                raise RuntimeError(
                    "resolved page has no checked canonical geometry; "
                    "page-less reduced residency is forbidden"
                )
            page_index = int(resolution.slot.page_index)
            page = self.pages[page_index]
            u0, v0, u1, v1 = page.uv_for_slot(int(resolution.slot.slot_index))
            slot_h, slot_w = page.tile_shape
            mapped_area = 0
            target_area = sum(
                (block.source_rect_yx[1] - block.source_rect_yx[0])
                * (block.source_rect_yx[3] - block.source_rect_yx[2])
                for block in plan.draw_blocks
            )
            for block in plan.draw_blocks:
                for actual_block in actual_plan.draw_blocks:
                    intersection = _rect_intersection_yx(
                        block.source_rect_yx,
                        actual_block.source_rect_yx,
                    )
                    if intersection is None:
                        continue
                    by0, by1, bx0, bx1 = intersection
                    asy0, asy1, asx0, asx1 = actual_block.stored_rect_yx
                    aby0, aby1, abx0, abx1 = actual_block.source_rect_yx
                    actual_y0 = _stored_edge_for_source(
                        by0,
                        source_start=aby0,
                        source_stop=aby1,
                        stored_start=asy0,
                        stored_stop=asy1,
                    )
                    actual_y1 = _stored_edge_for_source(
                        by1,
                        source_start=aby0,
                        source_stop=aby1,
                        stored_start=asy0,
                        stored_stop=asy1,
                    )
                    actual_x0 = _stored_edge_for_source(
                        bx0,
                        source_start=abx0,
                        source_stop=abx1,
                        stored_start=asx0,
                        stored_stop=asx1,
                    )
                    actual_x1 = _stored_edge_for_source(
                        bx1,
                        source_start=abx0,
                        source_stop=abx1,
                        stored_start=asx0,
                        stored_stop=asx1,
                    )
                    mapped_area += (by1 - by0) * (bx1 - bx0)
                    parts.append(
                        TileDrawPart(
                            world_rect=(
                                region_x + ((bx0 - coverage_x0) * region_w) / float(coverage_w),
                                region_y + ((by0 - coverage_y0) * region_h) / float(coverage_h),
                                region_x + ((bx1 - coverage_x0) * region_w) / float(coverage_w),
                                region_y + ((by1 - coverage_y0) * region_h) / float(coverage_h),
                            ),
                            uv_rect=(
                                u0 + (u1 - u0) * (actual_x0 / float(slot_w)),
                                v0 + (v1 - v0) * (actual_y0 / float(slot_h)),
                                u0 + (u1 - u0) * (actual_x1 / float(slot_w)),
                                v0 + (v1 - v0) * (actual_y1 / float(slot_h)),
                            ),
                            page_index=page_index,
                        )
                    )
            if mapped_area != target_area:
                raise RuntimeError(
                    "resolved canonical page geometry does not completely cover target"
                )

        actual_keys = tuple(dict.fromkeys(item.actual_key for item in resolved))
        previous_keys = set(self.tile_chunk_residency.get(int(tile_number), ()))
        for key in previous_keys.difference(actual_keys):
            owners = self.chunk_resident_tiles.get(key)
            if owners is not None:
                owners.discard(int(tile_number))
                if not owners:
                    self.chunk_resident_tiles.pop(key, None)
        self.tile_chunk_residency[int(tile_number)] = actual_keys
        for key in actual_keys:
            self.chunk_resident_tiles.setdefault(key, set()).add(int(tile_number))
        self.tile_draw_parts[int(tile_number)] = tuple(parts)

        resident_key = _resident_key(payload)
        self.source_ids[resident_key] = payload.source_id
        self.acknowledged_identities[resident_key] = (
            getattr(payload, "tile_identity", None) or payload.source_id
        )
        self._chunked_tile_keys.add(resident_key)
        first_record = next(
            (
                self.physical_upload_records.get(key)
                for key in actual_keys
                if key in self.physical_upload_records
            ),
            None,
        )
        if first_record is not None:
            self.physical_upload_records[resident_key] = dict(first_record)
        first = resolved[0]
        first_page = self.pages[int(first.slot.page_index)]
        return _ChunkCommit(
            page_index=int(first.slot.page_index),
            slot=int(first.slot.slot_index),
            uv=first_page.uv_for_slot(int(first.slot.slot_index)),
            uploads=uploaded.uploads,
            upload_bytes=uploaded.upload_bytes,
            complex_uploads=uploaded.complex_uploads,
            prepare_ms=uploaded.prepare_ms,
            submit_ms=uploaded.submit_ms,
            uploaded_any=bool(uploaded.uploaded_keys),
        )

    def _commit_chunked_payload(
        self,
        tile_number: int,
        payload: DisplayTilePayload,
        chunks: tuple[_PayloadChunk, ...],
        *,
        world_region: tuple[int, int, int, int],
        protected_keys: set[object],
        near_keys: set[object],
        rgb_already_windowed: bool,
    ) -> _ChunkCommit | None:
        """Commit one anchored payload as chunk residency plus draw parts.

        Uploads only chunks whose keys are not resident; already-resident
        (interior) chunks are reused byte-identically across window shifts.
        Returns None when the chunk set cannot be placed on one page — the
        caller then falls back to the classic whole-tile path.
        """

        tile_number = int(tile_number)
        try:
            storage_mode = _atlas_storage_mode(
                ((tile_number, payload),),
                rgb_already_windowed=rgb_already_windowed,
            )
            slots = self._chunk_slots_for(
                tuple(chunk.key for chunk in chunks),
                protected_keys=protected_keys,
                near_keys=near_keys,
                storage_mode=storage_mode,
            )
        except AtlasCapacityError:
            return None
        page_index = slots[chunks[0].key][0]
        page = self.pages[int(page_index)]
        resident_key = _resident_key(payload)
        need_upload = tuple(chunk for chunk in chunks if slots[chunk.key][2])
        need_records = resident_key not in self.physical_upload_records
        scalar = color = None
        prepare_ms = 0.0
        if need_upload or need_records:
            scalar, color, prepare_ms = _prepare_payload_texture_data(
                payload,
                tile_shape=_payload_class_shape(payload),
                rgb_already_windowed=rgb_already_windowed,
                need_scalar=page.scalar_is_atlas,
                need_color=page.color_is_atlas,
            )
        submit_ms = 0.0
        uploads = 0
        upload_bytes = 0
        complex_uploads = 0
        for chunk in need_upload:
            py0, py1, px0, px1 = chunk.plane_rect
            _page_idx, slot, _newly = slots[chunk.key]
            y_off, x_off = page.offset_for_slot(int(slot))
            if scalar is not None:
                sub = np.ascontiguousarray(scalar[py0:py1, px0:px1])
                submit_ms += _upload_texture_plane(
                    page.scalar_texture,
                    sub,
                    offset=(int(y_off), int(x_off)),
                    copy=_upload_copy_required(sub, payload, force=page.complex_is_atlas),
                )
                uploads += 1
                upload_bytes += int(sub.nbytes)
                if page.complex_is_atlas:
                    complex_uploads += 1
            if color is not None:
                sub = np.ascontiguousarray(color[py0:py1, px0:px1])
                submit_ms += _upload_texture_plane(
                    page.color_texture,
                    sub,
                    offset=(int(y_off), int(x_off)),
                    copy=_upload_copy_required(sub, payload),
                )
                uploads += 1
                upload_bytes += int(sub.nbytes)
        if need_upload and page.mipmap_levels:
            page.mipmap_dirty = True

        # Draw parts: one UV-cropped quad per chunk. World rects tile the
        # layout region exactly — plane pixel ``p`` maps to world edge
        # ``region + p * region_extent / plane_extent``, the SAME uniform
        # stretch the classic single reduced quad applies, so visual
        # placement is identical to the whole-tile presentation.  At factor 1
        # the stretch is exactly 1 (region == plane); at factor > 1 with a
        # factor-divisible extent it is exactly the LOD factor.  Adjacent
        # chunks share the identical edge expression (bitwise-equal floats:
        # the integer product ``p * extent`` is exact, so the shared division
        # is too) — no gaps, no overlaps, and the last edge lands exactly on
        # the region end.  Clipped chunks sample only the valid sub-region of
        # their slot, never beyond it.
        region_x, region_y = (int(world_region[0]), int(world_region[1]))
        region_w, region_h = (int(world_region[2]), int(world_region[3]))
        plane_h, plane_w = _payload_class_shape(payload)
        chunk_h, chunk_w = page.tile_shape
        parts = []
        for chunk in chunks:
            py0, py1, px0, px1 = chunk.plane_rect
            _page_idx, slot, _newly = slots[chunk.key]
            u0, v0, u1, v1 = page.uv_for_slot(int(slot))
            width = px1 - px0
            height = py1 - py0
            parts.append(
                TileDrawPart(
                    world_rect=(
                        region_x + (px0 * region_w) / float(plane_w),
                        region_y + (py0 * region_h) / float(plane_h),
                        region_x + (px1 * region_w) / float(plane_w),
                        region_y + (py1 * region_h) / float(plane_h),
                    ),
                    uv_rect=(
                        u0,
                        v0,
                        u0 + (u1 - u0) * (width / float(chunk_w)),
                        v0 + (v1 - v0) * (height / float(chunk_h)),
                    ),
                )
            )

        new_keys = tuple(chunk.key for chunk in chunks)
        new_set = set(new_keys)
        for key in self.tile_chunk_residency.get(tile_number, ()):
            if key in new_set:
                continue
            owners = self.chunk_resident_tiles.get(key)
            if owners is not None:
                owners.discard(tile_number)
                if not owners:
                    self.chunk_resident_tiles.pop(key, None)
        self.tile_chunk_residency[tile_number] = new_keys
        for key in new_keys:
            self.chunk_resident_tiles.setdefault(key, set()).add(tile_number)
        self.tile_draw_parts[tile_number] = tuple(parts)

        # Tile-level identity bookkeeping stays exactly as on the classic
        # path: acknowledgement, presented identities, and tile truth are
        # keyed by _resident_key(payload). The chunk layer is pure residency.
        self.source_ids[resident_key] = payload.source_id
        self.acknowledged_identities[resident_key] = (
            getattr(payload, "tile_identity", None) or payload.source_id
        )
        self._chunked_tile_keys.add(resident_key)
        record_plane = scalar if scalar is not None else color
        if record_plane is not None:
            real_plane, imag_plane = array_plane_identities(record_plane)
            self.physical_upload_records[resident_key] = {
                "physical_texture_kind": _payload_texture_kind(payload).value,
                "physical_storage_mode": str(page.storage_mode),
                "physical_texture_dtype": str(np.asarray(record_plane).dtype),
                "physical_texture_shape": tuple(
                    int(value) for value in np.asarray(record_plane).shape
                ),
                "physical_real_plane_identity": plane_identity_record(real_plane),
                "physical_imag_plane_identity": plane_identity_record(imag_plane),
            }
        self.chunk_upload_count += len(need_upload)
        self.chunk_reuse_count += len(chunks) - len(need_upload)
        first_slot = slots[chunks[0].key]
        return _ChunkCommit(
            page_index=int(first_slot[0]),
            slot=int(first_slot[1]),
            uv=page.uv_for_slot(int(first_slot[1])),
            uploads=uploads,
            upload_bytes=upload_bytes,
            complex_uploads=complex_uploads,
            prepare_ms=prepare_ms,
            submit_ms=submit_ms,
            uploaded_any=bool(need_upload),
        )

    def _release_tile_chunks(self, tile_number: int) -> bool:
        """Unlink a tile from its chunk residency (chunks stay LRU-evictable)."""

        tile_number = int(tile_number)
        keys = self.tile_chunk_residency.pop(tile_number, ())
        for key in keys:
            owners = self.chunk_resident_tiles.get(key)
            if owners is not None:
                owners.discard(tile_number)
                if not owners:
                    self.chunk_resident_tiles.pop(key, None)
        if keys:
            self.tile_draw_parts.pop(tile_number, None)
        return bool(keys)

    def _forget_unreferenced_chunked_key(self, resident_key: object) -> None:
        """Drop identity records of a chunked tile-level key no tile presents.

        Chunked tile-level keys own no slot of their own; without this their
        records would outlive the draw mapping (classic keys instead become
        superseded and are reclaimed together with their slot).

        Active keys are exempt: within one commit the mapping loop may move a
        key between tile numbers (index-window retarget), transiently leaving
        it unreferenced between the displacing and the re-presenting mapping
        call.  Forgetting at that moment destroyed the re-presented tile's
        just-committed records and chunk registration (field defect
        2026-07-15: zoomed tiles + ``phys None/None`` truth rows).  The
        commit-end sweep in ``update_payloads`` reclaims records of chunked
        keys that genuinely stopped being presented.
        """

        if (
            resident_key in self._chunked_tile_keys
            and not self.resident_tiles.get(resident_key)
            and resident_key not in self.active_resident_keys
        ):
            self._chunked_tile_keys.discard(resident_key)
            self.source_ids.pop(resident_key, None)
            self.acknowledged_identities.pop(resident_key, None)
            self.physical_upload_records.pop(resident_key, None)

    def _sweep_unreferenced_chunked_keys(self) -> None:
        """Commit-end reclamation of chunked tile-level identity records.

        Runs after the mapping loop has settled every tile of the commit, so
        ``resident_tiles`` is final: any chunked key without a presenting
        tile now (active or not) has genuinely stopped being drawn and its
        records may go.  A later re-present rebuilds them from the still-warm
        chunks (``_commit_chunked_payload`` ``need_records``)."""

        for key in tuple(self._chunked_tile_keys):
            if not self.resident_tiles.get(key):
                self._chunked_tile_keys.discard(key)
                self.source_ids.pop(key, None)
                self.acknowledged_identities.pop(key, None)
                self.physical_upload_records.pop(key, None)

    def _clear_tile_mapping(self, tile_number: int) -> None:
        tile_number = int(tile_number)
        self._clear_tile_page_target_binding(tile_number)
        old_key = self.tile_resident_keys.pop(tile_number, None)
        if old_key is not None:
            tiles = self.resident_tiles.get(old_key)
            if tiles is not None:
                tiles.discard(tile_number)
                if not tiles:
                    self.resident_tiles.pop(old_key, None)
            self._forget_unreferenced_chunked_key(old_key)
        self.tile_slots.pop(tile_number, None)
        self.tile_draw_parts.pop(tile_number, None)
        self.tile_uvs.pop(tile_number, None)
        self._release_tile_chunks(tile_number)

    def _clear_tile_page_target_binding(self, tile_number: int) -> None:
        """Release every canonical page-resolution cache/pin for one tile."""

        tile_number = int(tile_number)
        if tile_number in self._page_target_pin_tiles:
            self._page_table.replace_pin_set(
                _page_target_pin_owner(tile_number),
                (),
            )
            self._page_target_pin_tiles.discard(tile_number)
        self.page_target_resolutions.pop(tile_number, None)
        page_owner = self._tile_page_pin_owners.pop(tile_number, None)
        if page_owner is not None:
            self._page_table.replace_pin_set(page_owner, ())
        self.tile_page_target_resolutions.pop(tile_number, None)
        self.tile_page_candidate_missing.pop(tile_number, None)

    def clear_presentation(self) -> None:
        """Drop every visible mapping and owner pin, preserving resident bytes.

        A semantic-source invalidation is a visibility boundary, not a cache
        flush. Keeping ``tile_resident_keys`` after visuals are hidden lets a
        later compatible rebirth accidentally resurrect stale pages under new
        uniforms. Clear the presentation maps at the same boundary while the
        page table and atlas slots remain available for a genuine revert.
        """

        mapped_tiles = set(self.tile_resident_keys)
        mapped_tiles.update(self.tile_page_target_resolutions)
        mapped_tiles.update(self.tile_chunk_residency)
        for tile_number in tuple(mapped_tiles):
            self._clear_tile_mapping(int(tile_number))
        for tile_number in tuple(self._page_target_pin_tiles):
            self._page_table.replace_pin_set(_page_target_pin_owner(tile_number), ())
        self._page_target_pin_tiles.clear()
        self.page_target_resolutions.clear()
        self.active_resident_keys.clear()
        self.active_chunk_keys.clear()
        self.active_base_source_ids.clear()

    def _set_tile_mapping(
        self,
        tile_number: int,
        resident_key: object,
        page_index: int,
        slot: int,
        uv: tuple[float, float, float, float],
        *,
        chunked: bool = False,
        page_backed: bool = False,
    ) -> None:
        tile_number = int(tile_number)
        if not page_backed:
            self._clear_tile_page_target_binding(tile_number)
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
            if not self.resident_tiles.get(old_key) and old_key in self._page_table:
                self.superseded_keys.add(old_key)
            self._forget_unreferenced_chunked_key(old_key)
        self.superseded_keys.discard(resident_key)
        if not chunked:
            # The tile now presents whole-tile residency: any chunk links and
            # per-chunk draw parts from a previous presentation are stale (the
            # chunks themselves stay warm and LRU-evictable).  This must be
            # decided by HOW this commit presented the tile, not by whether
            # the key is registered chunked — the same resident key can be
            # committed chunked once and classically later (the anchor is not
            # part of the key), and the registry can be transiently stale
            # within a retarget commit (field defect 2026-07-15).
            self._release_tile_chunks(tile_number)
            self.tile_draw_parts.pop(tile_number, None)
        self.tile_slots[tile_number] = (int(page_index), int(slot))
        self.tile_resident_keys[tile_number] = resident_key
        self.resident_tiles.setdefault(resident_key, set()).add(tile_number)
        self.tile_uvs[tile_number] = uv
        if not chunked and resident_key in self._chunked_tile_keys:  # noqa: SIM102
            # A classic presentation of a previously chunked key: the key now
            # owns a whole-tile slot, so its identity records live and die
            # with that slot like any classic key.  Keep the chunked
            # registration only while another tile still draws this key
            # through chunk parts.
            if not any(
                self.tile_chunk_residency.get(int(other))
                for other in self.resident_tiles.get(resident_key, ())
                if int(other) != tile_number
            ):
                self._chunked_tile_keys.discard(resident_key)

    def _discard_tile_mappings_for_resident_key(self, resident_key: object) -> None:
        for tile_number in tuple(self.resident_tiles.pop(resident_key, set())):
            tile_number = int(tile_number)
            if self.tile_resident_keys.get(tile_number) == resident_key:
                self.tile_resident_keys.pop(tile_number, None)
                self.tile_slots.pop(tile_number, None)
                self.tile_draw_parts.pop(tile_number, None)
                self.tile_uvs.pop(tile_number, None)
                self._release_tile_chunks(tile_number)
        # A chunk victim invalidates its owning tile(s) entirely: one missing
        # chunk means the tile no longer presents its full content, so the
        # next commit must re-chunk (and re-upload only what was lost).
        for tile_number in tuple(self.chunk_resident_tiles.get(resident_key, ())):
            self._clear_tile_mapping(int(tile_number))
        self.chunk_resident_tiles.pop(resident_key, None)
        self._chunked_tile_keys.discard(resident_key)

    def _touch(self, resident_key: object) -> None:
        self._page_table.touch(resident_key)

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
    def __init__(
        self, *, scene, visuals, gloo, transforms, parent, limits: GpuDeviceLimits | None = None
    ):
        self._scene = scene
        self._visuals = visuals
        self._gloo = gloo
        self._transforms = transforms
        self._parent = parent
        self._device_limits = limits or query_gpu_device_limits(gloo)
        self._pool = TextureAtlasPool(gloo, limits=self._device_limits)
        self._visuals_by_page: list[object] = []
        self._geometry_keys: dict[int, tuple[object, ...]] = {}
        self._page_texture_keys: list[tuple[object, object]] = []
        self._page_mipmap_pages: list[object | None] = []
        self._page_visibility: list[bool] = []
        self._page_payloads_by_index: list[dict[int, DisplayTilePayload]] = []
        self._montage_geometry_key: tuple[object, ...] | None = None
        self._active_mapping_key: tuple[object, ...] | None = None
        self._atlas_serial: int = -1
        self._levels: tuple[float, float] = (0.0, 1.0)
        self._shader_mapping = None
        self._shader_mapping_key = None
        # Inputs the physical-truth audit needs between commits: the last
        # committed layout/rgb flag reproduce the exact per-page ``a_mode``
        # buffer expectation, and the visual-granularity mapping key is
        # cached per layer mapping identity so the LUT content is not
        # re-hashed on every clean commit.
        self._last_layout: dict = {}
        self._rgb_already_windowed = False
        self._visual_mapping_key_cache: tuple[object, ...] | None = None
        self._visible_items = 0
        self._changed_pages: tuple[int, ...] = ()
        self._last_stats = TileLayerUpdateStats()
        self._ensure_visual_count(1)

    @property
    def visual(self):
        self._ensure_visual_count(1)
        return self._visuals_by_page[0]

    @property
    def last_stats(self) -> TileLayerUpdateStats:
        return self._last_stats

    def diagnostics_snapshot(self) -> dict[str, object]:
        """Physical drawability plus the pool's canonical residency snapshot."""

        visible_tiles: set[int] = set()
        visible_pages = 0
        orphan_visible_pages = 0
        for page_index, visual in enumerate(self._visuals_by_page):
            if not bool(getattr(visual, "visible", False)):
                continue
            vertices = np.asarray(
                getattr(visual, "vertex_data", getattr(visual, "vertices", ()))
            ).reshape((-1, 2))
            texcoords = np.asarray(
                getattr(visual, "texcoord_data", getattr(visual, "texcoords", ()))
            ).reshape((-1, 2))
            modes = np.asarray(getattr(visual, "mode_data", getattr(visual, "modes", ()))).reshape(
                (-1,)
            )
            drawable_vertices = min(len(vertices), len(texcoords), len(modes))
            if drawable_vertices < 3 or not _visual_textures_ready(visual):
                continue
            visible_pages += 1
            page_tiles = {
                int(tile)
                for tile, offset, count, _mode in self._page_mode_spans_for(page_index)
                if int(count) > 0
                and int(offset) >= 0
                and int(offset) + int(count) <= int(drawable_vertices)
            }
            if page_tiles:
                visible_tiles.update(page_tiles)
            else:
                # A visible non-empty buffer without a current tile span is
                # itself important physical evidence: stale/orphan geometry
                # still contributes pixels even though it cannot be named as
                # a current tile. Count the page conservatively so callers do
                # not mistake it for a black surface.
                orphan_visible_pages += 1
        physically_visible_tile_count = len(visible_tiles) + orphan_visible_pages
        return {
            **self._pool.diagnostics_snapshot(),
            "presented_tiles": tuple(sorted(visible_tiles)),
            "presented_tile_count": physically_visible_tile_count,
            "physically_visible_tile_count": physically_visible_tile_count,
            "tile_visual_visible_pages": visible_pages,
            "physical_visible_page_count": visible_pages,
            "orphan_visible_page_count": orphan_visible_pages,
        }

    def tile_truth_physical_rows(self) -> dict[int, dict[str, object]]:
        rows = self._pool.tile_truth_physical_rows()
        spans_by_page: dict[int, dict[int, tuple[int, int, float]]] = {}
        for tile_number, row in rows.items():
            slot_ref = self._pool.tile_slots.get(int(tile_number))
            if slot_ref is None:
                continue
            page_index = int(slot_ref[0])
            if page_index >= len(self._visuals_by_page):
                continue
            visual = self._visuals_by_page[page_index]
            if page_index not in spans_by_page:
                spans_by_page[page_index] = {
                    tile: (offset, count, mode)
                    for tile, offset, count, mode in self._page_mode_spans_for(page_index)
                }
            span = spans_by_page[page_index].get(int(tile_number))
            mode_data = np.asarray(getattr(visual, "mode_data", ()), dtype=np.float32)
            physical_mode = (
                None
                if span is None or span[1] <= 0 or span[0] >= len(mode_data)
                else float(mode_data[span[0]])
            )
            quads = _tile_quad_rects(
                int(tile_number),
                self._last_layout,
                self._pool.tile_uvs,
                self._pool.tile_draw_parts,
            )
            draw_rects = tuple(tuple(float(value) for value in world) for world, _uv in quads)
            draw_uv_rects = tuple(tuple(float(value) for value in uv) for _world, uv in quads)
            draw_bounds = (
                None
                if not draw_rects
                else (
                    min(rect[0] for rect in draw_rects),
                    min(rect[1] for rect in draw_rects),
                    max(rect[2] for rect in draw_rects),
                    max(rect[3] for rect in draw_rects),
                )
            )
            region = self._last_layout.get(int(tile_number))
            expected_rect = (
                None
                if region is None
                else (
                    float(region.x),
                    float(region.y),
                    float(region.x + region.width),
                    float(region.y + region.height),
                )
            )
            row.update(
                {
                    "physical_mapping_mode": physical_mode,
                    "physical_component_mode": float(getattr(visual, "_component_mode", 0.0)),
                    "physical_levels": tuple(
                        float(value) for value in getattr(visual, "_levels", ())
                    ),
                    "physical_shader_mapping_key": repr(
                        getattr(visual, "_shader_mapping_key", None)
                    ),
                    "physical_draw_world_rects": draw_rects,
                    "physical_draw_uv_rects": draw_uv_rects,
                    "physical_draw_world_bounds": draw_bounds,
                    "physical_expected_world_rect": expected_rect,
                    "physical_draw_bounds_match_layout": bool(
                        draw_bounds is not None
                        and expected_rect is not None
                        and np.allclose(draw_bounds, expected_rect, rtol=0.0, atol=1e-6)
                    ),
                }
            )
        return rows

    def changed_page_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self._changed_pages)

    def _desired_visual_mapping_key(self) -> tuple[object, ...]:
        """Visual-granularity key for the layer's desired shader mapping.

        Cached per layer mapping identity (``_mapping_identity_key``): the
        visual key hashes the normalized LUT content, which must not be
        recomputed on every clean commit.
        """

        cache = self._visual_mapping_key_cache
        if cache is None or cache[0] != self._shader_mapping_key:
            cache = (
                self._shader_mapping_key,
                _visual_shader_mapping_key(self._shader_mapping),
            )
            self._visual_mapping_key_cache = cache
        return cache[1]

    def _page_mode_spans_for(self, page_index: int):
        payloads = (
            self._page_payloads_by_index[int(page_index)]
            if 0 <= int(page_index) < len(self._page_payloads_by_index)
            else {}
        )
        return _page_mode_spans(
            self._last_layout,
            payloads,
            self._pool.tile_uvs,
            self._pool.tile_draw_parts,
            page_index=int(page_index),
            rgb_already_windowed=self._rgb_already_windowed,
        )

    def physical_page_divergences(self) -> dict[int, tuple[str, ...]]:
        """Desired-vs-physical audit of every ACTIVE page visual.

        The layer-level caches (``_shader_mapping_key``/``_levels``/
        ``_geometry_keys``) are desired-state hints, not physical truth: a
        hidden page can re-enter holding an older uniform set, and nothing
        pins the per-quad ``a_mode`` vertex buffer (field defect 2026-07-15:
        stale mode 3 / stale ``u_component_mode`` rendered zero-magnitude
        complex texels as the PAL-relaxed LUT[0] orange, invisible to the
        identity layer by construction).  This audit compares each visible
        page visual's own physical state against the layer's desired state.

        Bounded cost per commit: per active page, three cached-key/uniform
        comparisons plus one float32 equality sweep over that page's
        ``a_mode`` buffer (6 floats per drawn quad — a few hundred entries
        for a full montage page).  The desired visual mapping key is cached
        per mapping identity, so no LUT hashing happens here.  Visuals that
        do not expose physical state (CPU/back-compat fakes) contribute no
        evidence and are never flagged.
        """

        divergences: dict[int, tuple[str, ...]] = {}
        desired_mapping = self._desired_visual_mapping_key()
        for page_index, visual in enumerate(self._visuals_by_page):
            if page_index >= len(self._page_visibility) or not self._page_visibility[page_index]:
                continue
            page_payloads = (
                self._page_payloads_by_index[page_index]
                if page_index < len(self._page_payloads_by_index)
                else {}
            )
            if not page_payloads:
                continue
            kinds: list[str] = []
            physical_mapping = getattr(visual, "_shader_mapping_key", _UNSET)
            if physical_mapping is not _UNSET:
                stale_key = physical_mapping != desired_mapping
                # A stale uniform can hide behind a fresh-looking key (the
                # injected-corruption class): audit the derived uniforms too.
                stale_uniforms = (
                    float(getattr(visual, "_scale_mode", desired_mapping[0]))
                    != float(desired_mapping[0])
                    or float(getattr(visual, "_symlog_constant", desired_mapping[1]))
                    != float(desired_mapping[1])
                    or float(getattr(visual, "_component_mode", desired_mapping[2]))
                    != float(desired_mapping[2])
                )
                if stale_key or stale_uniforms:
                    kinds.append("mapping")
            physical_levels = getattr(visual, "_levels", _UNSET)
            if physical_levels is not _UNSET and tuple(
                float(value) for value in physical_levels
            ) != tuple(float(value) for value in self._levels):
                kinds.append("levels")
            mode_data = getattr(visual, "mode_data", _UNSET)
            if (
                mode_data is not _UNSET
                and self._last_layout
                and self._mode_buffer_diverged(page_index, mode_data)
            ):
                kinds.append("modes")
            if kinds:
                divergences[int(page_index)] = tuple(kinds)
        return divergences

    def _mode_buffer_diverged(self, page_index: int, mode_data) -> bool:
        spans = self._page_mode_spans_for(page_index)
        mode_data = np.asarray(mode_data, dtype=np.float32).reshape((-1,))
        total = 0 if not spans else int(spans[-1][1] + spans[-1][2])
        if len(mode_data) != total:
            return True
        for _tile, offset, count, mode in spans:
            if count and not bool(np.all(mode_data[offset : offset + count] == np.float32(mode))):
                return True
        return False

    def _repair_physical_divergences(
        self, divergences: dict[int, tuple[str, ...]]
    ) -> tuple[int, int, int]:
        """Re-apply desired state onto divergent page visuals.

        Returns ``(mapping_updates, level_updates, vertex_uploads)`` so the
        caller can charge the repairs to the commit stats — a repaired
        presentation must never be acknowledged as a physical no-op.
        """

        mapping_updates = 0
        level_updates = 0
        vertex_uploads = 0
        for page_index, kinds in sorted(dict(divergences).items()):
            visual = self._visuals_by_page[int(page_index)]
            if "mapping" in kinds:
                # Clear the visual's own no-op cache first: a stale uniform
                # behind a fresh-looking key would otherwise no-op the setter.
                visual._shader_mapping_key = None
                mapping_updates += int(bool(visual.set_shader_mapping(self._shader_mapping)))
            if "levels" in kinds:
                level_updates += int(bool(visual.set_levels(self._levels)))
            if "modes" in kinds:
                page_payloads = (
                    self._page_payloads_by_index[int(page_index)]
                    if int(page_index) < len(self._page_payloads_by_index)
                    else {}
                )
                vertices, texcoords, modes = _quad_buffers(
                    self._last_layout,
                    page_payloads,
                    self._pool.tile_uvs,
                    rgb_already_windowed=self._rgb_already_windowed,
                    draw_parts=self._pool.tile_draw_parts,
                    page_index=int(page_index),
                )
                visual.set_geometry(vertices, texcoords, modes)
                vertex_uploads += 1
        return mapping_updates, level_updates, vertex_uploads

    def reset_residency(self) -> None:
        for visual in self._visuals_by_page:
            visual.visible = False
        self._pool = TextureAtlasPool(self._gloo, limits=self._device_limits)
        self._geometry_keys.clear()
        self._page_texture_keys = [(None, None) for _visual in self._visuals_by_page]
        self._page_mipmap_pages = [None for _visual in self._visuals_by_page]
        self._page_visibility = [False for _visual in self._visuals_by_page]
        self._page_payloads_by_index.clear()
        self._montage_geometry_key = None
        self._active_mapping_key = None
        self._atlas_serial = -1
        self._last_layout = {}
        self._visible_items = 0
        self._changed_pages = ()
        self._last_stats = TileLayerUpdateStats()

    def clear(self) -> None:
        # Hiding a layer must not discard useful GPU residency.  A later
        # viewport/session can reuse the same source identities.
        self._pool.clear_presentation()
        for visual in self._visuals_by_page:
            visual.visible = False
        self._geometry_keys.clear()
        self._page_texture_keys = [(None, None) for _visual in self._visuals_by_page]
        self._page_mipmap_pages = [None for _visual in self._visuals_by_page]
        self._page_visibility = [False for _visual in self._visuals_by_page]
        self._page_payloads_by_index.clear()
        self._montage_geometry_key = None
        self._active_mapping_key = None
        self._atlas_serial = -1
        self._last_layout = {}
        self._visible_items = 0
        self._changed_pages = ()

    def set_levels(self, levels) -> TileLayerUpdateStats:
        return self.set_presentation_uniforms(levels=levels)

    def set_shader_mapping(self, mapping) -> TileLayerUpdateStats:
        return self.set_presentation_uniforms(shader_mapping=mapping)

    def set_presentation_uniforms(
        self, *, levels=_UNSET, shader_mapping=_UNSET
    ) -> TileLayerUpdateStats:
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
        # Physical presentation truth: this uniforms-only path may present
        # as a no-op ONLY when every active page visual physically matches
        # the desired mapping/levels/mode-buffer state.  Divergence is
        # repaired here and charged to the returned stats.
        divergences = self.physical_page_divergences()
        repaired_vertex_uploads = 0
        physical_repairs = 0
        if divergences:
            repaired_mappings, repaired_levels, repaired_vertex_uploads = (
                self._repair_physical_divergences(divergences)
            )
            mapping_updates += repaired_mappings
            level_updates += repaired_levels
            physical_repairs = sum(len(kinds) for kinds in divergences.values())
        previous = self._last_stats
        changed_pages = (
            tuple(
                int(index)
                for index, visual in enumerate(self._visuals_by_page)
                if bool(getattr(visual, "visible", False))
            )
            if level_updates or mapping_updates
            else ()
        )
        changed_pages = tuple(sorted({*changed_pages, *divergences}))
        self._changed_pages = changed_pages
        self._last_stats = TileLayerUpdateStats(
            visible_items=self._visible_items,
            presented_tiles=tuple(int(tile) for tile in sorted(self._pool.tile_slots)),
            presented_identities=self._pool.presented_identities(),
            # Rule 1 (ADR 0051): this path can never apply payload upserts, so
            # it must say so explicitly.  ``None`` would make the commit
            # report fall back to acknowledging upserts by tile-number
            # intersection with the pool slots — falsely accepting payload
            # identities that were never uploaded (field defect 2026-07-05).
            committed_upserts=(),
            resident_items=self._pool.resident_count,
            storage_capacity=self._pool.capacity,
            level_updates=int(bool(level_updates)),
            shader_uniform_updates=level_updates + mapping_updates,
            vertex_uploads=repaired_vertex_uploads,
            physical_repairs=physical_repairs,
            items_skipped=self._visible_items,
            estimated_gpu_bytes=self._pool.estimated_gpu_bytes,
            cpu_shadow_bytes=self._pool.cpu_shadow_bytes,
            page_count=len(self._pool.pages),
            active_pages=sum(
                1 for visual in self._visuals_by_page if bool(getattr(visual, "visible", False))
            ),
            device_max_texture_size=self._pool.max_texture_size,
            budget_bytes=self._pool.budget_bytes,
            near_resident_items=int(getattr(previous, "near_resident_items", 0) or 0),
            warm_resident_items=max(
                0, self._pool.resident_count - len(self._pool.active_resident_keys)
            ),
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
            visual = self._scene.visuals.create_visual_node(GpuWindowedTileVisual)(
                parent=self._parent
            )
            visual.order = 10
            visual.visible = False
            visual.set_levels(self._levels)
            visual.set_shader_mapping(self._shader_mapping)
            self._visuals_by_page.append(visual)
            self._page_texture_keys.append((None, None))
            self._page_mipmap_pages.append(None)
            self._page_visibility.append(False)
        while len(self._page_texture_keys) < len(self._visuals_by_page):
            self._page_texture_keys.append((None, None))
        while len(self._page_mipmap_pages) < len(self._visuals_by_page):
            self._page_mipmap_pages.append(None)
        while len(self._page_visibility) < len(self._visuals_by_page):
            self._page_visibility.append(False)

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
        levels = _normalize_levels(levels, self._levels)
        mapping_key = _mapping_identity_key(shader_mapping)
        self._last_layout = layout
        self._rgb_already_windowed = bool(rgb_already_windowed)
        layout_key = _layout_geometry_key(layout)
        # Mapping order is the admission order.  Backends own storage, never
        # presentation priority, so this must not be sorted by tile number.
        explicit_upserts = tuple(
            int(tile) for tile in dict(getattr(tile_delta, "upserts", {}) or {})
        )
        removals = tuple(int(tile) for tile in tuple(getattr(tile_delta, "removals", ()) or ()))
        raw_active_tiles = (
            getattr(tile_delta, "active_tiles", None) if tile_delta is not None else None
        )
        active_tiles = (
            tuple(int(tile) for tile in tuple(raw_active_tiles))
            if raw_active_tiles is not None
            else tuple(sorted(payloads))
        )
        if bool(getattr(tile_delta, "atomic_handoff", False)):
            candidate_missing = _atomic_page_candidate_missing(
                self._pool,
                dict(getattr(tile_delta, "upserts", {}) or {}),
            )
            if candidate_missing:
                # Page resolution may truthfully retain a predecessor for one
                # incomplete tile.  Levels and shader mapping are shared by
                # every page visual, though, so advancing them would reinterpret
                # those predecessor texels as the successor (raw pages under a
                # complex shader was the field failure).  Reject the whole
                # immutable handoff before touching pool mappings or uniforms.
                self._pool.tile_page_candidate_missing.update(candidate_missing)
                self._changed_pages = ()
                self._last_stats = _clean_layer_stats(
                    self._last_stats,
                    visible_items=self._visible_items,
                    presented_tiles=self._last_stats.presented_tiles,
                    committed_upserts=(),
                    pool=self._pool,
                    page_visibility=self._page_visibility,
                )
                return self._last_stats
        active_mapping_key = _active_mapping_key(
            payloads,
            active_tiles=active_tiles,
            pool=self._pool,
            rgb_already_windowed=rgb_already_windowed,
        )
        if (
            dirty_tiles is not None
            and not tuple(dirty_tiles)
            and not removals
            and not bool(getattr(tile_delta, "force_refresh", False))
            and self._pool.pages
            and int(self._atlas_serial) == int(self._pool.serial)
            and layout_key == self._montage_geometry_key
            and active_mapping_key is not None
            and active_mapping_key == self._active_mapping_key
            and levels == self._levels
            and mapping_key == self._shader_mapping_key
        ):
            # The desired-state caches above prove the COMMIT is clean, not
            # that the page visuals physically hold that state (1e36084b
            # proved levels can diverge; the mapping key and mode buffer can
            # too).  Re-presenting without touching visuals is allowed only
            # when the physical audit passes; divergence is repaired and the
            # repair is charged to the acknowledged stats.
            divergences = self.physical_page_divergences()
            self._last_stats = _clean_layer_stats(
                self._last_stats,
                visible_items=self._visible_items,
                presented_tiles=tuple(
                    tile for tile in active_tiles if int(tile) in self._pool.tile_slots
                ),
                committed_upserts=explicit_upserts,
                pool=self._pool,
                page_visibility=self._page_visibility,
            )
            if divergences:
                repaired_mappings, repaired_levels, repaired_vertex_uploads = (
                    self._repair_physical_divergences(divergences)
                )
                self._last_stats = replace(
                    self._last_stats,
                    vertex_uploads=repaired_vertex_uploads,
                    level_updates=int(bool(repaired_levels)),
                    shader_uniform_updates=repaired_mappings + repaired_levels,
                    physical_repairs=sum(len(kinds) for kinds in divergences.values()),
                )
                self._changed_pages = tuple(sorted(divergences))
            else:
                self._changed_pages = ()
            return self._last_stats
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
            tile_world_regions={
                int(tile): (
                    int(region.x),
                    int(region.y),
                    int(region.width),
                    int(region.height),
                )
                for tile, region in layout.items()
            },
        )
        active_mapping_key = _active_mapping_key(
            payloads,
            active_tiles=tuple(texture_stats.presented_tiles or active_tiles),
            pool=self._pool,
            rgb_already_windowed=rgb_already_windowed,
        )
        self._ensure_visual_count(len(self._pool.pages))
        vertex_uploads = 0
        atlas_serial_before_sync = int(self._atlas_serial)
        previous_active_pages = {
            int(page_index)
            for page_index, page_payloads in enumerate(tuple(self._page_payloads_by_index or ()))
            if page_payloads
        }
        page_payloads_by_index, dirty_pages, active_pages = self._sync_page_payloads(
            payloads,
            presented_tiles=tuple(texture_stats.presented_tiles or ()),
            layout=layout,
            layout_key=layout_key,
            active_mapping_key=active_mapping_key,
            rgb_already_windowed=rgb_already_windowed,
        )
        page_indices = set(dirty_pages) | set(active_pages) | previous_active_pages
        if atlas_serial_before_sync != int(self._pool.serial):
            page_indices = set(range(len(self._pool.pages)))
        changed_pages: set[int] = set()
        for tile_number in explicit_upserts:
            slot_ref = self._pool.tile_slots.get(int(tile_number))
            if slot_ref is not None:
                changed_pages.add(int(slot_ref[0]))
        for page_index in sorted(
            index for index in page_indices if 0 <= int(index) < len(self._pool.pages)
        ):
            page = self._pool.pages[page_index]
            page_payloads = page_payloads_by_index[page_index]
            visual = self._visuals_by_page[page_index]
            if page_index in dirty_pages or page_index not in self._geometry_keys:
                geometry_key = _page_geometry_key(
                    page_payloads,
                    self._pool,
                    page,
                    layout,
                    page_index=int(page_index),
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
                    draw_parts=self._pool.tile_draw_parts,
                    page_index=int(page_index),
                )
                visual.set_geometry(vertices, texcoords, modes)
                self._geometry_keys[page_index] = geometry_key
                vertex_uploads += 1
                changed_pages.add(int(page_index))
        level_updates = 0
        if levels != self._levels:
            self._levels = levels
            for visual in self._visuals_by_page:
                level_updates += int(bool(visual.set_levels(levels)))
        mapping_changed = mapping_key != self._shader_mapping_key
        if mapping_changed:
            self._shader_mapping = shader_mapping
            self._shader_mapping_key = mapping_key
        mapping_updates = 0
        visual_indices = set(range(len(self._pool.pages))) if mapping_changed else set(page_indices)
        for page_index in sorted(
            index for index in visual_indices if 0 <= int(index) < len(self._pool.pages)
        ):
            page = self._pool.pages[page_index]
            visual = self._visuals_by_page[page_index]
            texture_key = (page.scalar_texture, page.color_texture)
            if texture_key != self._page_texture_keys[page_index]:
                visual.set_textures(page.scalar_texture, page.color_texture)
                self._page_texture_keys[page_index] = texture_key
                changed_pages.add(int(page_index))
            set_mipmap_page = getattr(visual, "set_mipmap_page", None)
            if set_mipmap_page is not None and page is not self._page_mipmap_pages[page_index]:
                set_mipmap_page(page)
                self._page_mipmap_pages[page_index] = page
                changed_pages.add(int(page_index))
            # The layer-level key is only a desired-state cache; each atlas
            # page owns a separate visual/uniform set. A page can retain an
            # older mapping while hidden and later re-enter through a payload
            # or geometry update even though the desired key itself did not
            # change. Synchronize every touched page and let the page-local
            # setter keep the common case a no-op.
            if bool(visual.set_shader_mapping(shader_mapping)):
                mapping_updates += 1
                changed_pages.add(int(page_index))
            visible = page_index in active_pages
            if (
                visible != self._page_visibility[page_index]
                or bool(getattr(visual, "visible", False)) != visible
            ):
                visual.visible = visible
                self._page_visibility[page_index] = visible
                changed_pages.add(int(page_index))
        for page_index, visual in enumerate(
            self._visuals_by_page[len(self._pool.pages) :], start=len(self._pool.pages)
        ):
            if page_index >= len(self._page_visibility) or not self._page_visibility[page_index]:
                continue
            visual.visible = False
            self._page_visibility[page_index] = False
            changed_pages.add(int(page_index))
        effective_presented_tiles = tuple(
            int(tile) for tile in tuple(texture_stats.presented_tiles or ())
        )
        self._visible_items = len(effective_presented_tiles)
        if level_updates:
            changed_pages.update(
                int(index)
                for index, visual in enumerate(self._visuals_by_page)
                if bool(getattr(visual, "visible", False))
            )
        # Physical presentation truth: even a full update rebuilds geometry
        # and re-applies uniforms only where the desired-state caches say so;
        # a physically clobbered page that those caches call fresh would be
        # acknowledged stale.  Audit and repair before reporting.
        physical_repairs = 0
        divergences = self.physical_page_divergences()
        if divergences:
            repaired_mappings, repaired_levels, repaired_vertex_uploads = (
                self._repair_physical_divergences(divergences)
            )
            mapping_updates += repaired_mappings
            level_updates += repaired_levels
            vertex_uploads += repaired_vertex_uploads
            physical_repairs = sum(len(kinds) for kinds in divergences.values())
            changed_pages.update(int(index) for index in divergences)
        self._changed_pages = tuple(sorted(changed_pages))
        self._last_stats = TileLayerUpdateStats(
            visible_items=len(effective_presented_tiles),
            presented_tiles=effective_presented_tiles,
            presented_identities=texture_stats.presented_identities,
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
            physical_repairs=physical_repairs,
            lod_level_swaps_zero_upload=texture_stats.lod_level_swaps_zero_upload,
            lod_level_swaps_with_upload=texture_stats.lod_level_swaps_with_upload,
            superseded_reclaimed_under_pressure=texture_stats.superseded_reclaimed_under_pressure,
        )
        return self._last_stats

    def _sync_page_payloads(
        self,
        payloads,
        *,
        presented_tiles,
        layout,
        layout_key,
        active_mapping_key,
        rgb_already_windowed: bool,
    ):
        page_count = len(self._pool.pages)
        page_payloads_by_index: list[dict[int, DisplayTilePayload]] = [
            {} for _page in self._pool.pages
        ]
        active = {int(tile) for tile in tuple(presented_tiles or ())}
        payload_map = {int(tile): payload for tile, payload in dict(payloads or {}).items()}
        previous_payloads = {
            int(tile): payload
            for page_payloads in tuple(self._page_payloads_by_index or ())
            for tile, payload in dict(page_payloads or {}).items()
        }
        for tile in sorted(active):
            payload = payload_map.get(int(tile))
            if payload is None:
                payload = previous_payloads.get(int(tile))
            if payload is None:
                continue
            part_pages = {
                int(part.page_index)
                for part in self._pool.tile_draw_parts.get(int(tile), ())
                if part.page_index is not None
            }
            if not part_pages:
                page_index, _slot = self._pool.tile_slots.get(int(tile), (-1, -1))
                part_pages = {int(page_index)}
            for page_index in part_pages:
                if 0 <= int(page_index) < page_count:
                    page_payloads_by_index[int(page_index)][int(tile)] = payload
        dirty_pages: set[int] = set()
        active_pages = {
            int(index)
            for index, page_payloads in enumerate(page_payloads_by_index)
            if page_payloads
        }
        previous_active_pages = {
            int(index)
            for index, page_payloads in enumerate(tuple(self._page_payloads_by_index or ()))
            if page_payloads
        }
        full_refresh = (
            len(self._page_payloads_by_index) != page_count
            or self._montage_geometry_key != layout_key
            or int(self._atlas_serial) != int(self._pool.serial)
        )
        candidates = (
            set(range(page_count)) if full_refresh else active_pages | previous_active_pages
        )
        for page_index in candidates:
            if full_refresh or _page_geometry_key(
                page_payloads_by_index[int(page_index)],
                self._pool,
                self._pool.pages[int(page_index)],
                layout,
                page_index=int(page_index),
                rgb_already_windowed=rgb_already_windowed,
            ) != self._geometry_keys.get(int(page_index)):
                dirty_pages.add(int(page_index))
        self._page_payloads_by_index = page_payloads_by_index
        self._montage_geometry_key = layout_key
        self._active_mapping_key = active_mapping_key
        self._atlas_serial = int(self._pool.serial)
        return self._page_payloads_by_index, dirty_pages, active_pages

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
        payload_map = {int(key): value for key, value in dict(payloads or {}).items()}
        for payload in payload_map.values():
            _require_canonical_reduced_payload(payload, action="warm")
        if montage is None:
            # ADR 0055 G4c: non-montage geometries warm only source-anchored
            # payloads, and only as chunk residency (the pool's chunk-warm
            # branch). Anything else has no montage layout to warm against.
            payload_map = {
                tile: payload
                for tile, payload in payload_map.items()
                if _payload_chunked_eligible(payload)
            }
            if not payload_map:
                return TileLayerUpdateStats()
            fallback = _payload_class_shape(next(iter(payload_map.values())))
        else:
            fallback = (int(montage.tile_height), int(montage.tile_width))
        try:
            return self._pool.warm_payloads(
                payload_map,
                tile_shape=_atlas_base_tile_shape_for_payloads(payload_map, fallback=fallback),
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
                warm_resident_items=max(
                    0, self._pool.resident_count - len(self._pool.active_resident_keys)
                ),
                capacity_warning=str(exc),
            )

    def payload_resident(self, payload: DisplayTilePayload) -> bool:
        """Return physical residency truth without changing bindings or LRU."""

        return self._pool.payload_resident(payload)


def _visual_textures_ready(visual) -> bool:
    """Whether a real or test visual has the textures required to draw."""

    if hasattr(visual, "_scalar_texture") or hasattr(visual, "_color_texture"):
        return bool(
            getattr(visual, "_scalar_texture", None) is not None
            and getattr(visual, "_color_texture", None) is not None
        )
    textures = getattr(visual, "textures", None)
    return bool(
        isinstance(textures, tuple)
        and len(textures) == 2
        and textures[0] is not None
        and textures[1] is not None
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
            if (v_mode > 4.5) {
                // phase_vector pages contain a circular unit-vector mean.
                // Their canonical resultant magnitude is already [0, 1];
                // applying the native complex-amplitude levels makes every
                // reduced page black when native FFT values span thousands.
                intensity = clamp(length(z), 0.0, 1.0);
                float phase = atan(z.y, z.x);
                phase_index = clamp((phase + 3.141592653589793) / 6.283185307179586, 0.0, 1.0);
                scalar = intensity;
            } else if (u_component_mode > 2.5) {
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
                gl.glTexParameteri(
                    gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, _GL_NEAREST_MIPMAP_LINEAR
                )
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

        if self._lut_texture is None or tuple(getattr(self._lut_texture, "shape", ())) != tuple(
            lut_texture_data.shape
        ):
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


def create_gpu_montage_layer(
    *, scene, visuals, gloo, transforms, parent, limits: GpuDeviceLimits | None = None
) -> GpuMontageLayer:
    return GpuMontageLayer(
        scene=scene, visuals=visuals, gloo=gloo, transforms=transforms, parent=parent, limits=limits
    )


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


def _payload_textures(
    payload: DisplayTilePayload, *, tile_shape: tuple[int, int], rgb_already_windowed: bool
):
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
    texture = np.asarray(
        payload.texture_data if payload.texture_data is not None else payload.image
    )
    texture_kind = _payload_texture_kind(payload)
    if texture_kind == TexturePlaneKind.COMPLEX_RG32F:
        scalar = (
            _fit_complex_rg(_complex_rg_texture(texture), (tile_h, tile_w)) if need_scalar else None
        )
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


def _upload_texture_plane(
    texture, data: np.ndarray, *, offset: tuple[int, int], copy: bool
) -> float:
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
    out = np.zeros((*shape, 3), dtype=np.uint8)
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
    out = np.zeros((*shape, 2), dtype=np.float32)
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
        return arr.view(np.float32).reshape((*arr.shape, 2))
    packed = np.asarray(arr, dtype=np.float32)
    if packed.ndim < 3 or packed.shape[-1] != 2:
        raise ValueError("complex RG32F texture data must be complex or have trailing size 2")
    return np.ascontiguousarray(packed)


def _upload_copy_required(
    staging: np.ndarray, payload: DisplayTilePayload, *, force: bool = False
) -> bool:
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


def _tile_quad_rects(tile_number, layout, uvs, draw_parts, *, page_index=None):
    """(world_rect, uv_rect) quads for one tile — registered parts, else the
    classic single full-slot quad; empty when the tile has no layout/UV."""

    parts = None if draw_parts is None else draw_parts.get(int(tile_number))
    if parts:
        selected = tuple(
            part
            for part in parts
            if page_index is None
            or part.page_index is None
            or int(part.page_index) == int(page_index)
        )
        return tuple((part.world_rect, part.uv_rect) for part in selected)
    region = layout.get(int(tile_number))
    if region is None:
        return ()
    uv = uvs.get(int(tile_number))
    if uv is None:
        return ()
    x0 = float(region.x)
    y0 = float(region.y)
    return (((x0, y0, x0 + float(region.width), y0 + float(region.height)), tuple(uv)),)


def _quad_buffers(
    layout,
    payloads,
    uvs,
    *,
    rgb_already_windowed: bool,
    draw_parts=None,
    page_index=None,
):
    vertices = []
    texcoords = []
    modes = []
    for tile_number, payload in sorted((int(key), value) for key, value in dict(payloads).items()):
        quads = _tile_quad_rects(tile_number, layout, uvs, draw_parts, page_index=page_index)
        if not quads:
            continue
        mode = float(_payload_mode(payload, rgb_already_windowed=rgb_already_windowed))
        for (x0, y0, x1, y1), (u0, v0, u1, v1) in quads:
            vertices.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)))
            texcoords.extend(((u0, v0), (u1, v0), (u1, v1), (u0, v0), (u1, v1), (u0, v1)))
            modes.extend((mode,) * 6)
    return (
        np.asarray(vertices, dtype=np.float32).reshape((-1, 2)),
        np.asarray(texcoords, dtype=np.float32).reshape((-1, 2)),
        np.asarray(modes, dtype=np.float32).reshape((-1,)),
    )


def _page_mode_spans(
    layout,
    payloads,
    uvs,
    draw_parts,
    *,
    page_index=None,
    rgb_already_windowed: bool,
):
    """Ordered ``(tile, offset, count, mode)`` spans of a page's ``a_mode`` buffer.

    Mirrors ``_quad_buffers`` exactly — same sorted-tile order, same
    draw-parts quad emission, same skip rule for tiles without a layout
    region or UV — so ``offset:offset + count`` addresses a tile's vertices
    in the visual's physical mode buffer and ``mode`` is the desired
    ``_payload_mode`` for those vertices.  Shared by the tile-truth
    diagnostics rows and the physical-divergence audit.  Cost: O(tiles on
    page).
    """

    spans = []
    offset = 0
    for tile_number, payload in sorted(
        (int(key), value) for key, value in dict(payloads or {}).items()
    ):
        quads = _tile_quad_rects(tile_number, layout, uvs, draw_parts, page_index=page_index)
        count = 6 * len(quads)
        spans.append(
            (
                tile_number,
                offset,
                count,
                float(_payload_mode(payload, rgb_already_windowed=rgb_already_windowed)),
            )
        )
        offset += count
    return spans


def _page_geometry_key(
    payloads,
    pool,
    page,
    layout,
    *,
    page_index=None,
    rgb_already_windowed: bool,
) -> tuple[object, ...]:
    # Registered draw parts are keyed explicitly: their UV crops are geometry
    # inputs that slot/region/gutter alone no longer determine (a window
    # shift changes only the crop rects, and must rebuild the buffers).
    draw_parts = pool.tile_draw_parts
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
                tuple(
                    (part.world_rect, part.uv_rect, part.page_index)
                    for part in draw_parts.get(int(key), ())
                    if page_index is None
                    or part.page_index is None
                    or int(part.page_index) == int(page_index)
                ),
            )
            for key, payload in sorted(dict(payloads or {}).items())
            if int(key) in pool.tile_slots and int(key) in layout
        ),
        id(page),
        tuple(int(value) for value in page.atlas_shape),
    )


def _layout_geometry_key(layout) -> tuple[object, ...]:
    return (
        tuple(
            (
                int(region.tile_number),
                int(region.x),
                int(region.y),
                int(region.width),
                int(region.height),
            )
            for region in sorted(
                dict(layout or {}).values(), key=lambda item: int(item.tile_number)
            )
        ),
    )


def _active_mapping_key(
    payloads, *, active_tiles, pool, rgb_already_windowed: bool
) -> tuple[object, ...] | None:
    payload_map = {int(tile): payload for tile, payload in dict(payloads or {}).items()}
    items = []
    for tile in tuple(active_tiles or ()):
        tile = int(tile)
        payload = payload_map.get(tile)
        if payload is None:
            return None
        resident_key = _resident_key(payload)
        if pool.tile_resident_keys.get(tile) != resident_key:
            return None
        if pool.source_ids.get(resident_key) != payload.source_id:
            return None
        if pool.acknowledged_identities.get(resident_key) != tile_ack_identity(payload):
            return None
        slot = pool.tile_slots.get(tile)
        if slot is None:
            return None
        items.append(
            (
                tile,
                resident_key,
                int(slot[0]),
                int(slot[1]),
                payload.source_id,
                _payload_mode(payload, rgb_already_windowed=rgb_already_windowed),
                _payload_gutter(payload),
            )
        )
    return tuple(items)


def _atomic_page_candidate_missing(pool, payloads) -> dict[int, tuple[DataChunkKey, ...]]:
    """Preflight an atomic page handoff without changing physical state."""

    missing_by_tile: dict[int, tuple[DataChunkKey, ...]] = {}
    for tile, payload in dict(payloads or {}).items():
        backing = getattr(payload, "page_backing", None)
        if backing is None:
            continue
        missing = tuple(
            target for target in backing.requested_keys if pool._page_table.resolve(target) is None
        )
        if missing:
            missing_by_tile[int(tile)] = missing
    return missing_by_tile


def _empty_warm_counters() -> dict[str, object]:
    return {
        "uploads": 0,
        "upload_bytes": 0,
        "complex_uploads": 0,
        "prepare_ms": 0.0,
        "submit_ms": 0.0,
        "updated": 0,
        "skipped": 0,
        "skipped_denied": 0,
    }


def _merge_warm_counters(*parts) -> dict[str, object]:
    merged = _empty_warm_counters()
    for part in parts:
        if part is None:
            continue
        for key in merged:
            merged[key] += part[key]
    return merged


def _warm_capacity_warning(page_warm, chunk_warm) -> str:
    page_denied = 0 if page_warm is None else int(page_warm["skipped_denied"])
    chunk_denied = 0 if chunk_warm is None else int(chunk_warm["skipped_denied"])
    if page_denied and chunk_denied:
        return (
            f"skipped {page_denied + chunk_denied} warm page/chunk payloads because "
            "atlas capacity/budget is full"
        )
    if page_denied:
        return (
            f"skipped {page_denied} warm page-backed payloads because page capacity/budget is full"
        )
    if chunk_denied:
        return (
            f"skipped {chunk_denied} warm anchored payloads because chunk capacity/budget is full"
        )
    return ""


def _clean_layer_stats(
    previous: TileLayerUpdateStats,
    *,
    visible_items: int,
    presented_tiles,
    committed_upserts,
    pool,
    page_visibility,
) -> TileLayerUpdateStats:
    return replace(
        previous,
        visible_items=int(visible_items),
        presented_tiles=tuple(int(tile) for tile in tuple(presented_tiles or ())),
        committed_upserts=tuple(int(tile) for tile in tuple(committed_upserts or ())),
        presented_identities=pool.presented_identities(),
        items_updated=0,
        items_skipped=int(visible_items),
        texture_uploads=0,
        texture_upload_bytes=0,
        texture_prepare_ms=0.0,
        texture_submit_ms=0.0,
        vertex_uploads=0,
        level_updates=0,
        shader_uniform_updates=0,
        physical_repairs=0,
        upload_ms=0.0,
        resident_items=pool.resident_count,
        storage_capacity=pool.capacity,
        estimated_gpu_bytes=pool.estimated_gpu_bytes,
        cpu_shadow_bytes=pool.cpu_shadow_bytes,
        page_count=len(pool.pages),
        active_pages=sum(1 for visible in tuple(page_visibility or ()) if bool(visible)),
        device_max_texture_size=pool.max_texture_size,
        budget_bytes=pool.budget_bytes,
        warm_resident_items=max(0, pool.resident_count - len(pool.active_resident_keys)),
        mipmap_updates=0,
        complex_texture_uploads=0,
        lod_level_swaps_zero_upload=0,
        lod_level_swaps_with_upload=0,
        superseded_reclaimed_under_pressure=0,
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
        image = np.asarray(
            payload.texture_data if payload.texture_data is not None else payload.image
        )
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


def _atlas_grid(
    *, tile_shape: tuple[int, int], capacity: int, max_texture_size: int
) -> tuple[int, int]:
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


def _atlas_allocation_bytes(
    *,
    tile_shape: tuple[int, int],
    capacity: int,
    storage_mode: str,
    max_texture_size: int,
) -> int:
    """Physical bytes of the rectangular texture grid for one atlas page."""

    columns, rows = _atlas_grid(
        tile_shape=tile_shape,
        capacity=capacity,
        max_texture_size=max_texture_size,
    )
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    pixels = int(columns) * tile_w * int(rows) * tile_h
    return pixels * _storage_mode_bytes_per_pixel(storage_mode)


def _atlas_class_allocation_bytes(
    *,
    tile_shape: tuple[int, int],
    count: int,
    storage_mode: str,
    max_texture_size: int,
) -> int:
    """Physical bytes for a class split across bounded texture pages."""

    remaining = max(0, int(count))
    if remaining == 0:
        return 0
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    max_slots_per_page = max(1, int(max_texture_size // tile_w)) * max(
        1,
        int(max_texture_size // tile_h),
    )
    total = 0
    while remaining:
        page_capacity = min(remaining, max_slots_per_page)
        total += _atlas_allocation_bytes(
            tile_shape=(tile_h, tile_w),
            capacity=page_capacity,
            storage_mode=storage_mode,
            max_texture_size=max_texture_size,
        )
        remaining -= page_capacity
    return int(total)


def _max_atlas_class_capacity_within_bytes(
    *,
    tile_shape: tuple[int, int],
    max_capacity: int,
    storage_mode: str,
    max_texture_size: int,
    budget_bytes: int,
) -> int:
    """Largest logical class capacity whose rounded pages fit the budget."""

    low, high = 0, max(0, int(max_capacity))
    budget = max(0, int(budget_bytes))
    while low < high:
        candidate = (low + high + 1) // 2
        required = _atlas_class_allocation_bytes(
            tile_shape=tile_shape,
            count=candidate,
            storage_mode=storage_mode,
            max_texture_size=max_texture_size,
        )
        if required <= budget:
            low = candidate
        else:
            high = candidate - 1
    return int(low)


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
    return (max(1, *heights), max(1, *widths))


def _payload_mode(payload: DisplayTilePayload, *, rgb_already_windowed: bool) -> int:
    if _payload_texture_kind(payload) == TexturePlaneKind.COMPLEX_RG32F:
        mapping = getattr(payload, "shader_mapping", None)
        display_mode = getattr(
            getattr(mapping, "display_mode", None), "value", getattr(mapping, "display_mode", None)
        )
        if display_mode == ShaderDisplayMode.PHASE_COLOR.value:
            backing = getattr(payload, "page_backing", None)
            plans = tuple(getattr(backing, "requested_plans", ()) or ())
            if plans and all(plan.reducer == REDUCER_PHASE_VECTOR for plan in plans):
                return 5
            return 4
        return 3
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
    display_mode = getattr(
        getattr(mapping, "display_mode", None), "value", getattr(mapping, "display_mode", None)
    )
    phase_default = bool(display_mode == ShaderDisplayMode.PHASE_COLOR.value)
    lut = _normalized_lut(getattr(mapping, "lut_data", None), phase_default=phase_default)
    lut_key = _array_content_key(lut)
    return (
        float(scale_mode),
        float(symlog_constant),
        float(component_mode),
        phase_default,
        lut_key,
    )


def _shader_scale_uniform(scale) -> float:
    if scale is None:
        return 0.0
    value = scale.value if isinstance(scale, ShaderScale) else getattr(scale, "value", scale)
    if value == ShaderScale.LOG.value:
        return 1.0
    if value == ShaderScale.SYMLOG.value:
        return 2.0
    return 0.0


def _payload_supported_by_storage_mode(
    payload: DisplayTilePayload, storage_mode: str | None, *, rgb_already_windowed: bool
) -> bool:
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
        return (
            kind
            if isinstance(kind, TexturePlaneKind)
            else TexturePlaneKind(getattr(kind, "value", kind))
        )
    texture = np.asarray(
        getattr(payload, "texture_data", None)
        if getattr(payload, "texture_data", None) is not None
        else payload.image
    )
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
    return max(
        (
            int(getattr(getattr(payload, "lod", None), "level", 0) or 0)
            for payload in dict(payloads or {}).values()
        ),
        default=0,
    )


def _max_payload_lod_factor(payloads: dict[int, DisplayTilePayload]) -> int:
    return max(
        (
            int(getattr(getattr(payload, "lod", None), "factor", 1) or 1)
            for payload in dict(payloads or {}).values()
        ),
        default=1,
    )


def _max_payload_gutter(payloads: dict[int, DisplayTilePayload]) -> int:
    return max((_payload_gutter(payload) for payload in dict(payloads or {}).values()), default=0)


def _payload_class_shape(payload: DisplayTilePayload) -> tuple[int, int]:
    """Slot shape class for a payload: its actual texture shape (ADR 0050)."""

    texture = payload.texture_data if payload.texture_data is not None else payload.image
    shape = np.shape(texture)
    return (int(shape[0]), int(shape[1]))


def _atlas_base_tile_shape_for_payloads(
    payloads: dict[int, DisplayTilePayload], *, fallback: tuple[int, int]
) -> tuple[int, int]:
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
    return (max(1, *(shape[0] for shape in shapes)), max(1, *(shape[1] for shape in shapes)))


def _atlas_tile_shape_for_payloads(
    payloads: dict[int, DisplayTilePayload], *, fallback: tuple[int, int]
) -> tuple[int, int]:
    shapes = []
    for payload in dict(payloads or {}).values():
        texture = np.asarray(
            payload.texture_data if payload.texture_data is not None else payload.image
        )
        shapes.append(tuple(int(value) for value in texture.shape[:2]))
    if not shapes:
        return (max(1, int(fallback[0])), max(1, int(fallback[1])))
    return (max(1, *(shape[0] for shape in shapes)), max(1, *(shape[1] for shape in shapes)))


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
    if isinstance(payload.source_id, DataChunkKey):
        # A canonical logical page is already complete residency identity.
        # Wrapping it in the legacy tile texture tuple would make the page
        # invisible to best-resident-ancestor resolution.
        return payload.source_id
    texture = np.asarray(
        payload.texture_data if payload.texture_data is not None else payload.image
    )
    lod = getattr(payload, "lod", None)
    key = (
        _source_resident_key(payload.source_id),
        "texture_kind",
        None
        if payload.texture_kind is None
        else getattr(payload.texture_kind, "value", payload.texture_kind),
        "texture_shape",
        tuple(int(value) for value in texture.shape),
        "texture_dtype",
        str(texture.dtype),
        "lod",
        None if lod is None else (int(lod.factor), int(lod.level), int(lod.gutter)),
    )
    with contextlib.suppress(Exception):
        object.__setattr__(payload, "_vispy_resident_key", key)
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


def _resident_source_matches_expected(source_id: object, expected_source_id: object | None) -> bool:
    if expected_source_id is None:
        return True
    return _lod_invariant_source_id(source_id) == _lod_invariant_source_id(expected_source_id)


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
