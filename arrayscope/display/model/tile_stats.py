"""Tile-layer commit/diagnostics stats contract shared by all backends (roadmap Y2)."""

from __future__ import annotations

from dataclasses import dataclass


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
