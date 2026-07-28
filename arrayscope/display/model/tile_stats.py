"""Tile-layer commit/diagnostics stats contract shared by all backends (roadmap Y2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TileLayerUpdateStats:
    visible_items: int = 0
    presented_tiles: tuple[int, ...] | None = None
    committed_upserts: tuple[int, ...] | None = None
    # Ground truth for ADR 0051 rule 1: the payload identity each drawn tile
    # slot ACTUALLY holds after this update (tile -> source_id).  Session
    # bookkeeping converges against this map, never the other way around.
    # None = backend does not report identities (legacy/CPU layers).
    presented_identities: Mapping[int, object] | None = field(default=None)
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
    # Structural, non-steady-state time spent growing backend storage during
    # this commit. It remains part of upload_ms/full callback diagnostics but
    # is split out so a pacing model does not learn it as per-cohort cost.
    pool_growth_ms: float = 0.0
    executor_initialization_ms: float = 0.0
    storage_evictions: int = 0
    texture_uploads: int = 0
    texture_upload_bytes: int = 0
    texture_prepare_ms: float = 0.0
    texture_submit_ms: float = 0.0
    # The subset of preparation that is pure array work over immutable
    # payloads -- packing and page extraction. This is what a worker can own;
    # the rest of preparation walks live presentation state.
    texture_pack_ms: float = 0.0
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
    # ADR 0050 zero-upload zoom cycles: level flips between already-resident
    # classes must be identity swaps.  Per-commit counts; GPU backends fill
    # them, CPU tile layers leave them at zero.
    lod_level_swaps_zero_upload: int = 0
    lod_level_swaps_with_upload: int = 0
    superseded_reclaimed_under_pressure: int = 0
    shader_uniform_updates: int = 0
    # WGPU mapping-only publication accounting. Per-commit, mutually
    # exclusive: the backend either reused the complete physical tile
    # binding or rebuilt/rebound it. CPU backends leave both at zero.
    binding_fast_path_commits: int = 0
    binding_incremental_commits: int = 0
    binding_full_republications: int = 0
    # Physical presentation truth (P9): count of desired-vs-physical page
    # divergences (stale mapping key/uniform, stale levels, stale per-quad
    # mode buffer) detected AND repaired during this update.  Zero on a
    # healthy commit; any non-zero value means an acknowledgement would have
    # been produced from a physically divergent layer state without the
    # repair.  GPU backends fill it; CPU tile layers leave it at zero.
    physical_repairs: int = 0
    # Payload upserts whose typed identity cannot satisfy the delta's target
    # identity for that tile.  They are excluded from presentation without
    # any other side effect, so a non-zero value on consecutive commits of
    # the same delta means the presenter is re-emitting payloads the backend
    # can never acknowledge (field stall 2026-07-16: the silent form of this
    # rejection starved 91 required tiles of any producer).  Backends that
    # enforce typed targets fill it; CPU tile layers leave it at zero.
    identity_rejected_items: int = 0
    identity_rejected_tiles: tuple[int, ...] = ()
    upload_ms: float = 0.0
    level_update_processed_items: int = 0
    level_update_pending_items: int = 0
