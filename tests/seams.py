"""Observation and drive seams that exist only for the tests.

Each function here used to be a method on a production class with no production
caller: an accessor a test read, a convenience the tests drove state through, or
an oracle only an assertion ever asked for. The dead-code gate
(:mod:`tests.dead_code`) refuses that shape, and the honest home for it is the
test suite -- so it lives here, next to the assertions that want it, expressed
over the same internals the production code uses.

Reaching into a private attribute is deliberate: these are test scaffolding, and
writing the access out in the open is what keeps the production class free of a
public surface nobody ships.

What earns a place here is *driving or observing state production owns*, from an
angle no shipped caller needs — ``_update_tile_levels`` per tile rather than
inside the batched commit, a supersession advanced without a successor to submit,
a process-lifetime cache dropped for a second answer. A one-line read of a
private attribute does not qualify; that is written at the call site, because an
indirection nobody needs is the same rot one directory over.

Moving a method here is **not** a way to keep it. The gate scans this file too
(:func:`tests.dead_code.unreferenced_test_helpers`), so a seam whose last caller
goes fails at the hook exactly as a production definition would.
"""

from __future__ import annotations

import numpy as np


# --- kernel -----------------------------------------------------------------
def supersede(kernel, family, value) -> None:
    """Advance a supersession family without submitting a replacement.

    Production always supersedes by submitting the successor; this drives the
    same transition without a second task in flight.
    """

    from arrayscope.kernel.task import Supersession

    with kernel._lock:
        kernel._advance_supersession_locked(Supersession(family, value))
        kernel._cond.notify_all()
    kernel._flush_release_wake()


# --- LOD page cache ---------------------------------------------------------
def begin_claim(cache, key, owner) -> bool:
    """Own the single producer claim for one page key, if it is missing."""

    from arrayscope.gpu.keys import DataChunkKey

    if not isinstance(key, DataChunkKey):
        raise TypeError("LOD page cache claims require DataChunkKey keys")
    with cache._lock:
        if cache._cache.peek(key) is not None or key in cache._claims:
            return False
        cache._claims[key] = owner
        return True


# --- montage level tracker --------------------------------------------------
def update_from_tile(
    tracker, key, source_index, histogram_data, image, *, refined=False, aggregate=True
):
    """Sample one tile and feed it in -- what a commit does in two steps."""

    from arrayscope.display.model.montage_levels import _sample_tile_stats

    source = histogram_data if histogram_data is not None else image
    tile_stats = _sample_tile_stats(source, int(source_index), refined=bool(refined))
    if tile_stats is None:
        return tracker.stats_for(key) if aggregate else None
    return tracker.update_from_stats(key, tile_stats, aggregate=aggregate)


# --- frame session ----------------------------------------------------------
def snapshot_display_tile_payloads(session, source_ids):
    """Build the typed payload wrappers for every loaded tile.

    Production builds these one tile at a time as results land; this drives the
    same builder over the whole rendered set so a test can inspect the result.
    """

    loaded = {int(index) for index in session.rendered_tiles}
    for stale in tuple(session.display_tile_payloads):
        if int(stale) not in loaded:
            session.display_tile_payloads.pop(int(stale), None)
    lod_factor = session._selected_lod_factor()
    for tile_number, rendered in session.rendered_tiles.items():
        session._ensure_display_tile_payload(
            int(tile_number), rendered, source_ids, lod_factor=lod_factor
        )
    return dict(session.display_tile_payloads)


def mark_preview_refinements_dirty(session, tile_numbers) -> None:
    changed = False
    for tile_number in tuple(tile_numbers or ()):
        index = int(tile_number)
        if index in session.rendered_tiles:
            session.dirty_payloads[index] = None
            changed = True
    if changed:
        session.flush_pending = True
        session.final_commit_pending = True


# --- tile lifecycle ---------------------------------------------------------
def level_declined(lifecycle, tile_number, level_key):
    """Decline a claimed level that never became resident."""

    from arrayscope.presentation.tile_lifecycle import LevelPhase, ReleaseClaim

    rec = lifecycle.record(tile_number)
    entry = rec.levels.get(level_key)
    if entry is None or entry.phase in (LevelPhase.RESIDENT, LevelPhase.RELEASED):
        return ()
    entry.phase = LevelPhase.RELEASED
    lifecycle._level_revision += 1
    return (ReleaseClaim(rec.tile_number, level_key, entry.owner),)


# --- pyqtgraph montage tile layer -------------------------------------------
def presented_index_scan(layer) -> set[int]:
    """The full visibility scan the maintained presented index replaces."""

    from arrayscope.display.backends.pyqtgraph.tiles import _state_is_physically_visible

    return {
        int(state.tile_number)
        for tile, state in layer._states.items()
        if int(tile) == int(state.tile_number) and _state_is_physically_visible(state)
    }


def assert_presented_index_matches_scan(layer) -> None:
    """Raise unless the maintained index equals a fresh full scan."""

    scanned = presented_index_scan(layer)
    maintained = layer.presented_tiles
    if maintained != scanned:
        raise AssertionError(
            "presented index drifted from the full scan: "
            f"index-only={sorted(maintained - scanned)} "
            f"scan-only={sorted(scanned - maintained)}"
        )


def update_levels(layer, levels, *, image=None, histogram_data=None):
    """Rewindow every visible tile to new levels, as a levels-only commit does.

    Production reaches ``_update_tile_levels`` one tile at a time from inside
    the batched commit; this drives the same per-tile call over the visible set
    and reports it in the layer's own stats shape.
    """

    from dataclasses import replace
    from time import perf_counter

    from arrayscope.display.backends.pyqtgraph.tiles import TileLayerUpdateStats

    levels = (float(levels[0]), float(levels[1]))
    atlas = layer._preview_atlas_item
    if atlas is not None:
        atlas.set_mapping(replace(atlas.mapping, levels=levels))
    image_array = None if image is None else np.asarray(image)
    hist_array = None if histogram_data is None else np.asarray(histogram_data)
    items_updated = 0
    items_skipped = 0
    rgb_window_tiles = 0
    processed = 0
    update_start = perf_counter()
    for state in tuple(layer._states.values()):
        if not state.visible:
            continue
        processed += 1
        updated, windowed = layer._update_tile_levels(
            state, levels, image=image_array, histogram_data=hist_array
        )
        items_updated += int(updated)
        rgb_window_tiles += int(windowed)
        if not updated:
            items_skipped += 1
    layer._prune_rgb_source_cache()
    resident_items = layer._resident_count()
    physically_presented = layer._physically_presented_tiles()
    atlas_tiles = len(layer.preview_atlas_active_tiles)
    return TileLayerUpdateStats(
        visible_items=len(physically_presented),
        presented_tiles=physically_presented,
        presented_identities=layer._presented_identities(),
        items_updated=items_updated,
        items_skipped=items_skipped,
        rgb_window_tiles=rgb_window_tiles,
        level_updates=processed,
        resident_items=int(resident_items) + atlas_tiles,
        storage_capacity=int(resident_items) + atlas_tiles,
        cpu_shadow_bytes=int(layer._resident_bytes) + layer._preview_atlas_nbytes(),
        warm_resident_items=max(0, int(resident_items) + atlas_tiles - len(physically_presented)),
        level_update_processed_items=processed,
        upload_ms=(perf_counter() - update_start) * 1000.0,
    )


# --- process-lifetime caches ------------------------------------------------
#
# Production resolves each of these once per process and never drops it; only a
# test wants a second answer, so the drop lives here rather than as a public
# reset nothing ships a caller for.
def reset_topology_cache() -> None:
    """Drop the detected GPU topology (adapter/device lifecycle churn)."""

    from arrayscope.gpu import device_topology

    device_topology._CACHED = None


def reset_plugin_cache() -> None:
    """Clear the loaded-spec cache so entry points are discovered again."""

    from arrayscope.operations import plugin_conformance, plugins

    plugins._SPEC_CACHE.clear()
    plugin_conformance._CHARACTERIZATION_CACHE.clear()
    for name in plugin_conformance._CHARACTERIZATION_STATS:
        plugin_conformance._CHARACTERIZATION_STATS[name] = 0


def reset_operation_packs() -> None:
    """Clear pack registration so packs re-load with a changed backend."""

    from arrayscope.operations import registry

    registry._PACK_SPECS.clear()
    registry._PACKS_LOADED = False


# --- retained tiled payloads -------------------------------------------------
def clear_retained_store(store, reason: str) -> None:
    """Drop every retained payload wholesale.

    Production never clears the store in one shot -- it drops superseded source
    ids as they are superseded -- so this is the cold-start a test wants and no
    shipped path takes.
    """

    store._payloads.clear()
    store.last_clear_reason = str(reason)


# --- view state --------------------------------------------------------------
def with_axis_fftshifted(state, axis: int, fftshifted: bool):
    """Set one axis's fftshift flag.

    The flag is live data -- ``core/view_recipe`` writes it on restore and
    ``ViewState.for_shape`` migrates it -- but nothing in the product sets it
    one axis at a time, so the per-axis setter belongs here rather than beside
    ``with_axis_flipped``.
    """

    from dataclasses import replace

    flags = list(state.axis_fftshifted)
    flags[int(axis)] = bool(fftshifted)
    return replace(state, axis_fftshifted=tuple(flags))
