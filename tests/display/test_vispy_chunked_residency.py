"""ADR 0055 G3b-2: backend-private chunked residency in the VisPy atlas.

The live session flow ships ONE window-sized exact payload per non-montage
commit, stamped with a window-invariant ``PayloadSourceAnchor``. The pool
splits that plane into origin-anchored 256x256 chunks keyed by
``(content_key, clipped native rect, texture kind, dtype, lod)`` so a 1-px
display-window shift re-uploads only the boundary strips and re-registers
draw parts over the surviving interior chunks.
"""

from dataclasses import replace

import numpy as np

from arrayscope.display.backends.vispy.tiles import (
    ANCHORED_CHUNK_SHAPE,
    AtlasCapacityError,
    TextureAtlasPool,
    _payload_chunk_plan,
    _payload_chunked_eligible,
)
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor
from arrayscope.display.pyramid import reduce_box_mean

CHUNK = int(ANCHORED_CHUNK_SHAPE[0])
HEIGHT = 2 * CHUNK
DATA_WIDTH = 8 * CHUNK
EXTENT = 4 * CHUNK


class RecordingTexture2D:
    def __init__(self, data=None, *, shape=None, **kwargs):
        self.shape = tuple(shape) if shape is not None else tuple(np.shape(data))
        self.uploads: list[tuple[tuple[int, int], np.ndarray]] = []

    def set_data(self, data, *, offset=None, copy=True):
        self.uploads.append((tuple(int(v) for v in (offset or (0, 0))), np.array(data, copy=True)))


class FakeGloo:
    Texture2D = RecordingTexture2D


def _data():
    return np.random.default_rng(3).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)


CONTENT_KEY = ("src-anchored", "doc-rev-0", "windowless-view")


def anchored_payload(data, start, *, extent=EXTENT, content_key=CONTENT_KEY, tile_number=0, quality="exact"):
    plane = np.ascontiguousarray(data[:, start : start + extent])
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=0,
        image=plane,
        histogram_data=None,
        # Live source ids embed the window: they RENAME on a shift. Chunk
        # keys must carry reuse regardless.
        source_id=("window", start, extent, tile_number),
        quality=quality,
        source_anchor=PayloadSourceAnchor(
            content_key=content_key,
            source_rect=(0, plane.shape[0], start, start + extent),
        ),
    )


def commit(pool, payloads, *, extent=EXTENT):
    regions = {
        int(tile): (0, 0, int(payload.image.shape[1]), int(payload.image.shape[0]))
        for tile, payload in payloads.items()
    }
    _uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=(HEIGHT, extent),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(payloads),
        tile_world_regions=regions,
    )
    return stats


def chunk_pages_and_slots(pool, tile=0):
    slots = pool.resident_slots
    return {key: slots[key] for key in pool.tile_chunk_residency[tile]}


def scalar_uploads(pool):
    uploads = []
    for page in pool.pages:
        uploads.extend(page.scalar_texture.uploads)
    return uploads


def test_chunk_plan_keys_and_rects():
    data = _data()
    payload = anchored_payload(data, 100)
    assert _payload_chunked_eligible(payload)
    plan = _payload_chunk_plan(payload)
    # 2 chunk rows x 5 chunk columns intersect the window [100, 1124).
    assert len(plan) == 10
    rects = {chunk.rect for chunk in plan}
    assert (0, 256, 100, 256) in rects  # left boundary: clipped rect
    assert (0, 256, 256, 512) in rects  # interior: full chunk rect
    assert (256, 512, 1024, 1124) in rects  # right boundary: clipped rect
    # Keys fold the full content identity.
    key = plan[0].key
    assert key[0] == "anchored-chunk"
    assert key[1] == CONTENT_KEY
    assert key[3] == "scalar_r32f" or isinstance(key[3], str)
    # The plane tiles exactly: chunk plane rects cover every pixel once.
    covered = np.zeros((HEIGHT, EXTENT), dtype=np.int32)
    for chunk in plan:
        py0, py1, px0, px1 = chunk.plane_rect
        covered[py0:py1, px0:px1] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_one_pixel_shift_uploads_only_boundary_chunks():
    data = _data()
    pool = TextureAtlasPool(FakeGloo())

    cold = commit(pool, {0: anchored_payload(data, 100)})
    assert cold.items_updated == 1
    assert 0 in pool.tile_chunk_residency
    chunks_a = set(pool.tile_chunk_residency[0])
    assert len(chunks_a) == 10
    assert pool.chunk_upload_count == 10
    # Same-page constraint: all chunks of one tile on one page.
    placements = chunk_pages_and_slots(pool)
    assert len({page for page, _slot in placements.values()}) == 1
    # Uploaded chunk content is byte-identical to the plane sub-arrays.
    plane_a = data[:, 100 : 100 + EXTENT]
    page_index = next(iter(placements.values()))[0]
    page = pool.pages[page_index]
    by_offset = {offset: content for offset, content in page.scalar_texture.uploads}
    plan = _payload_chunk_plan(anchored_payload(data, 100))
    for chunk in plan:
        page_i, slot = placements[chunk.key]
        offset = page.offset_for_slot(slot)
        py0, py1, px0, px1 = chunk.plane_rect
        assert np.array_equal(by_offset[offset], plane_a[py0:py1, px0:px1])

    uploads_before = pool.chunk_upload_count
    warm = commit(pool, {0: anchored_payload(data, 101)})
    chunks_b = set(pool.tile_chunk_residency[0])
    assert len(chunks_b) == 10
    # Interior chunks survive the shift byte-identically; only the two
    # boundary strips per chunk row change identity.
    assert len(chunks_a & chunks_b) == 6
    assert pool.chunk_upload_count - uploads_before == 4
    assert warm.items_updated == 1  # one tile touched, boundary-only uploads
    # Draw parts tile the region exactly: no gaps, no overlaps.
    parts = pool.tile_draw_parts[0]
    assert len(parts) == 10
    area = sum(
        (part.world_rect[2] - part.world_rect[0]) * (part.world_rect[3] - part.world_rect[1])
        for part in parts
    )
    assert area == HEIGHT * EXTENT
    xs = sorted({part.world_rect[0] for part in parts} | {part.world_rect[2] for part in parts})
    ys = sorted({part.world_rect[1] for part in parts} | {part.world_rect[3] for part in parts})
    assert xs[0] == 0.0 and xs[-1] == float(EXTENT)
    assert ys[0] == 0.0 and ys[-1] == float(HEIGHT)
    # Boundary edges fall where native chunk boundaries map into the window.
    assert xs == [0.0, 155.0, 411.0, 667.0, 923.0, float(EXTENT)]

    # Scrolling back reuses the still-resident original boundary chunks.
    uploads_before = pool.chunk_upload_count
    back = commit(pool, {0: anchored_payload(data, 100)})
    assert pool.chunk_upload_count == uploads_before
    assert back.items_updated == 0
    assert back.items_skipped >= 1


def test_presented_identity_and_tile_truth_follow_chunked_tiles():
    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    payload = anchored_payload(data, 100)
    stats = commit(pool, {0: payload})
    assert stats.presented_tiles == (0,)
    assert pool.presented_identities()[0] == payload.source_id
    rows = pool.tile_truth_physical_rows()
    assert 0 in rows
    assert rows[0]["physical_texture_dtype"] == "float32"


def test_chunk_eviction_invalidates_only_owning_tile():
    data = _data()
    other = np.random.default_rng(9).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
    pool = TextureAtlasPool(FakeGloo())
    commit(
        pool,
        {
            0: anchored_payload(data, 100, tile_number=0),
            1: anchored_payload(other, 100, content_key=("src-anchored", "doc-B", "view"), tile_number=1),
        },
    )
    assert set(pool.tile_chunk_residency) == {0, 1}
    victim = pool.tile_chunk_residency[0][3]
    survivor_chunks = tuple(pool.tile_chunk_residency[1])

    # Evict one chunk of tile 0 the way slot pressure does.
    slot_ref = pool._page_table.lookup(victim)
    pool._release_victim(victim, near_keys=set())
    page = pool.pages[slot_ref.page_index]
    page.slot_owners[slot_ref.slot_index] = None
    page._free_slots.append(slot_ref.slot_index)

    # Only the owning tile lost its mapping.
    assert 0 not in pool.tile_chunk_residency
    assert 0 not in pool.tile_draw_parts
    assert 0 not in pool.tile_slots
    assert pool.tile_chunk_residency[1] == survivor_chunks
    assert 1 in pool.tile_draw_parts

    # The next commit restores tile 0 by re-uploading ONLY the lost chunk.
    uploads_before = pool.chunk_upload_count
    commit(
        pool,
        {
            0: anchored_payload(data, 100, tile_number=0),
            1: anchored_payload(other, 100, content_key=("src-anchored", "doc-B", "view"), tile_number=1),
        },
    )
    assert pool.chunk_upload_count - uploads_before == 1
    assert len(pool.tile_chunk_residency[0]) == 10


def test_ineligible_payloads_take_classic_path():
    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    # No anchor (montage payloads): classic.
    plain = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=np.ascontiguousarray(data[:, :EXTENT]),
        histogram_data=None,
        source_id=("montage-tile", 0),
    )
    assert not _payload_chunked_eligible(plain)
    commit(pool, {0: plain})
    assert 0 not in pool.tile_chunk_residency
    assert 0 not in pool.tile_draw_parts
    assert 0 in pool.tile_slots

    # Preview quality: classic even when anchored.
    preview = anchored_payload(data, 100, quality="preview", tile_number=1)
    assert not _payload_chunked_eligible(preview)
    # Plane not larger than one chunk: classic.
    small = anchored_payload(data, 100, extent=CHUNK)
    small_one = DisplayTilePayload(
        tile_number=2,
        source_index=0,
        image=np.ascontiguousarray(data[:CHUNK, 100 : 100 + CHUNK]),
        histogram_data=None,
        source_id=("window", 100, CHUNK, 2),
        source_anchor=PayloadSourceAnchor(
            content_key=CONTENT_KEY, source_rect=(0, CHUNK, 100, 100 + CHUNK)
        ),
    )
    assert not _payload_chunked_eligible(small_one)
    del small


def test_chunked_to_classic_transition_releases_chunk_links():
    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    commit(pool, {0: anchored_payload(data, 100)})
    assert 0 in pool.tile_chunk_residency and 0 in pool.tile_draw_parts

    classic = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=np.ascontiguousarray(data[:, :EXTENT]),
        histogram_data=None,
        source_id=("classic-replacement", 0),
    )
    stats = commit(pool, {0: classic})
    assert stats.presented_tiles == (0,)
    assert 0 not in pool.tile_chunk_residency
    # Stale per-chunk draw parts must not survive a classic presentation.
    assert 0 not in pool.tile_draw_parts
    # The chunked tile-level key's records were dropped with the mapping.
    assert ("window", 100, EXTENT, 0) not in {
        source_id for source_id in pool.source_ids.values()
    }
    # The chunks themselves stay warm and reusable: re-presenting the
    # anchored payload re-links them without any texture upload.
    uploads_before = pool.chunk_upload_count
    commit(pool, {0: anchored_payload(data, 100)})
    assert pool.chunk_upload_count == uploads_before
    assert len(pool.tile_chunk_residency[0]) == 10


def test_warm_payloads_chunks_are_pure_residency_and_make_commit_upload_free():
    """ADR 0055 G4c: background warming of an adjacent plane's chunks.

    Warming uploads chunk residency only — no draw parts, no tile mapping,
    no tile-level identity — and the later visible commit of the same plane
    finds every chunk resident and uploads nothing.
    """

    data = _data()
    other = np.random.default_rng(11).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
    plane_b_key = ("src-anchored", "doc-rev-0", "plane-1")
    pool = TextureAtlasPool(FakeGloo())

    payload_a = anchored_payload(data, 100)
    commit(pool, {0: payload_a})
    chunks_a = set(pool.tile_chunk_residency[0])
    parts_a = pool.tile_draw_parts[0]

    warm_payload = replace(
        anchored_payload(other, 100, content_key=plane_b_key, tile_number=0),
        source_id=("prefetch-warm", "plane-1"),
    )
    plan_b = _payload_chunk_plan(warm_payload)
    stats = pool.warm_payloads(
        {0: warm_payload},
        tile_shape=(HEIGHT, EXTENT),
        rgb_already_windowed=False,
    )
    assert stats.items_updated == 1
    assert stats.texture_uploads == len(plan_b) == 10
    assert stats.storage_evictions == 0

    # Residency: every warm chunk is in the page table, byte-identical to
    # the plane sub-arrays it was cut from.
    slots = pool.resident_slots
    plane_b = other[:, 100 : 100 + EXTENT]
    for chunk in plan_b:
        assert chunk.key in slots, f"warm chunk not resident: {chunk.key!r}"
        page_index, slot = slots[chunk.key]
        page = pool.pages[page_index]
        by_offset = {offset: content for offset, content in page.scalar_texture.uploads}
        py0, py1, px0, px1 = chunk.plane_rect
        assert np.array_equal(by_offset[page.offset_for_slot(slot)], plane_b[py0:py1, px0:px1])

    # Presentation is untouched: tile 0 still presents plane A.
    assert set(pool.tile_chunk_residency[0]) == chunks_a
    assert pool.tile_draw_parts[0] == parts_a
    assert pool.presented_identities()[0] == payload_a.source_id
    # No tile-level identity records for the warm-only payload.
    assert warm_payload.source_id not in set(pool.source_ids.values())
    assert warm_payload.source_id not in set(pool.acknowledged_identities.values())

    # Re-warming the same plane is a residency touch, not more uploads.
    uploads_before = pool.chunk_upload_count
    again = pool.warm_payloads(
        {0: warm_payload},
        tile_shape=(HEIGHT, EXTENT),
        rgb_already_windowed=False,
    )
    assert pool.chunk_upload_count == uploads_before
    assert again.items_updated == 0
    assert again.items_skipped == 1

    # The visible commit of the warmed plane uploads ZERO chunks and
    # registers the presentation (draw parts + identity).
    uploads_before = pool.chunk_upload_count
    scalar_upload_count_before = len(scalar_uploads(pool))
    visible = commit(pool, {0: warm_payload})
    assert pool.chunk_upload_count == uploads_before
    assert len(scalar_uploads(pool)) == scalar_upload_count_before
    assert visible.texture_uploads == 0
    assert visible.presented_tiles == (0,)
    assert len(pool.tile_draw_parts[0]) == 10
    assert set(pool.tile_chunk_residency[0]) == {chunk.key for chunk in plan_b}
    assert pool.presented_identities()[0] == warm_payload.source_id
    # Plane A's chunks survived as warm residency (no eviction happened).
    assert all(key in pool.resident_slots for key in chunks_a)


def test_warm_payloads_denied_by_budget_skips_without_evicting():
    """G4c hard invariant: warm work never evicts to make room for itself."""

    base_bytes = HEIGHT * EXTENT * 4
    chunk_bytes = CHUNK * CHUNK * 4
    budget = base_bytes + 10 * chunk_bytes  # exactly plane A's residency
    data = _data()
    other = np.random.default_rng(13).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
    pool = TextureAtlasPool(FakeGloo(), budget_bytes=budget)

    commit(pool, {0: anchored_payload(data, 100)})
    chunks_a = set(pool.tile_chunk_residency[0])
    assert len(chunks_a) == 10

    warm_payload = anchored_payload(
        other, 100, content_key=("src-anchored", "doc-rev-0", "plane-1"), tile_number=0
    )
    uploads_before = pool.chunk_upload_count
    stats = pool.warm_payloads(
        {0: warm_payload},
        tile_shape=(HEIGHT, EXTENT),
        rgb_already_windowed=False,
    )
    assert stats.items_updated == 0
    assert stats.items_skipped == 1
    assert stats.texture_uploads == 0
    assert stats.storage_evictions == 0
    assert "warm anchored payloads" in stats.capacity_warning
    assert pool.chunk_upload_count == uploads_before
    # Nothing warm was destroyed: plane A stays fully resident and presented.
    assert all(key in pool.resident_slots for key in chunks_a)
    assert set(pool.tile_chunk_residency[0]) == chunks_a
    warm_keys = {chunk.key for chunk in _payload_chunk_plan(warm_payload)}
    assert not any(key in pool.resident_slots for key in warm_keys)


def test_layout_reset_clears_chunk_maps():
    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    commit(pool, {0: anchored_payload(data, 100)})
    assert pool.tile_chunk_residency
    # A storage-mode change rebuilds the atlas and drops all residency.
    pool.ensure_layout(tile_shape=(HEIGHT, EXTENT), count=1, storage_mode="complex")
    assert not pool.tile_chunk_residency
    assert not pool.chunk_resident_tiles
    assert not pool._chunked_tile_keys


# ---------------------------------------------------------------------------
# ADR 0056 G5 slice 1: uniform plane-pixel pages across LODs (factor > 1)
# ---------------------------------------------------------------------------
#
# A 256x256 chunk slot holds 256x256 STORED SAMPLES at any LOD (covering
# 256*factor native samples per axis), so reduced planes share the native
# planes' shape class instead of occupying one whole-plane slot per plane
# size.  HONEST LIMIT: reduction bins are anchored to the window origin, so
# a +-1 NATIVE-pixel window shift resamples the reduced plane — texels
# genuinely differ and chunks must NOT survive such shifts (their native key
# rects all move).  Only an identical revisit at the same LOD reuses chunks.

R_FACTOR = 4
R_LEVEL = 2
R_HEIGHT = 6 * CHUNK  # native window height; divisible by R_FACTOR
R_EXTENT = 8 * CHUNK  # native window width; divisible by R_FACTOR
R_DATA_WIDTH = 9 * CHUNK
R_PLANE_H = R_HEIGHT // R_FACTOR  # 384: one full + one clipped chunk row
R_PLANE_W = R_EXTENT // R_FACTOR  # 512: two full chunk columns
R_CONTENT_KEY = ("src-anchored", "doc-rev-0", "windowless-view-reduced")


def _reduced_data():
    return np.random.default_rng(7).random((R_HEIGHT, R_DATA_WIDTH), dtype=np.float32)


def reduced_anchored_payload(
    data,
    start,
    *,
    factor=R_FACTOR,
    level=R_LEVEL,
    extent=R_EXTENT,
    height=None,
    content_key=R_CONTENT_KEY,
    tile_number=0,
    quality="exact",
):
    height = int(data.shape[0]) if height is None else int(height)
    native = np.ascontiguousarray(data[:height, start : start + extent])
    plane = np.asarray(reduce_box_mean(native, (factor, factor)))
    lod = LodInfo(
        level=level,
        factor=factor,
        source_shape=native.shape[:2],
        texture_shape=plane.shape[:2],
        gutter=0,
    )
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=0,
        image=plane,
        histogram_data=None,
        source_id=("window", start, extent, tile_number, "lod", factor),
        quality=quality,
        lod=lod,
        source_anchor=PayloadSourceAnchor(
            content_key=content_key,
            source_rect=(0, height, start, start + extent),
        ),
    )


def commit_anchored(pool, payloads, *, tile_shape):
    """Commit anchored payloads with world regions spanning the NATIVE anchor
    extent (world == native units; reduced plane pixels stretch by the LOD
    factor), matching the live layout-region contract."""

    regions = {}
    for tile, payload in payloads.items():
        anchor = payload.source_anchor
        if anchor is None:
            shape = payload.image.shape
            regions[int(tile)] = (0, 0, int(shape[1]), int(shape[0]))
        else:
            y0, y1, x0, x1 = (int(value) for value in anchor.source_rect)
            regions[int(tile)] = (0, 0, int(x1 - x0), int(y1 - y0))
    _uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=tile_shape,
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(payloads),
        tile_world_regions=regions,
    )
    return stats


def test_reduced_plan_uniform_plane_pixel_chunks_with_native_scaled_keys():
    data = _reduced_data()
    payload = reduced_anchored_payload(data, 100)
    assert _payload_chunked_eligible(payload)
    plan = _payload_chunk_plan(payload)
    # Plane 384x512: chunk rows [0,256),[256,384); columns [0,256),[256,512).
    assert len(plan) == 4
    plane_rects = {chunk.plane_rect for chunk in plan}
    assert plane_rects == {
        (0, 256, 0, 256),
        (0, 256, 256, 512),
        (256, 384, 0, 256),
        (256, 384, 256, 512),
    }
    # Key rects are NATIVE source coordinates: anchor origin + plane rect
    # scaled by the factor (y0 + py*factor, x0 + px*factor), clipped to the
    # anchor rect.
    rects = {chunk.rect for chunk in plan}
    assert rects == {
        (0, 1024, 100, 1124),
        (0, 1024, 1124, 2148),
        (1024, 1536, 100, 1124),
        (1024, 1536, 1124, 2148),
    }
    # Keys fold content identity plus the LOD triple.
    for chunk in plan:
        assert chunk.key[0] == "anchored-chunk"
        assert chunk.key[1] == R_CONTENT_KEY
        assert chunk.key[5] == (R_FACTOR, R_LEVEL, 0)
    # The plane tiles exactly: chunk plane rects cover every sample once.
    covered = np.zeros((R_PLANE_H, R_PLANE_W), dtype=np.int32)
    for chunk in plan:
        py0, py1, px0, px1 = chunk.plane_rect
        covered[py0:py1, px0:px1] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_reduced_commit_revisit_reuse_and_honest_one_native_px_shift_limit():
    data = _reduced_data()
    pool = TextureAtlasPool(FakeGloo())

    payload_a = reduced_anchored_payload(data, 100)
    cold = commit_anchored(pool, {0: payload_a}, tile_shape=(R_HEIGHT, R_EXTENT))
    assert cold.items_updated == 1
    assert cold.presented_tiles == (0,)
    chunks_a = set(pool.tile_chunk_residency[0])
    assert len(chunks_a) == 4
    assert pool.chunk_upload_count == 4
    # Uniform plane-pixel slots: every chunk lives in the 256x256 shape
    # class, all on one page.
    placements = chunk_pages_and_slots(pool)
    pages = {page for page, _slot in placements.values()}
    assert len(pages) == 1
    page = pool.pages[next(iter(pages))]
    assert page.tile_shape == tuple(ANCHORED_CHUNK_SHAPE)
    # Uploaded chunk content is byte-identical to the reduced plane.
    plane_a = np.asarray(payload_a.image)
    by_offset = {offset: content for offset, content in page.scalar_texture.uploads}
    for chunk in _payload_chunk_plan(payload_a):
        _page_i, slot = placements[chunk.key]
        py0, py1, px0, px1 = chunk.plane_rect
        assert np.array_equal(by_offset[page.offset_for_slot(slot)], plane_a[py0:py1, px0:px1])

    # Identical revisit at the same LOD: identical keys, ZERO uploads.
    revisit = commit_anchored(
        pool,
        {0: reduced_anchored_payload(data, 100)},
        tile_shape=(R_HEIGHT, R_EXTENT),
    )
    assert pool.chunk_upload_count == 4
    assert set(pool.tile_chunk_residency[0]) == chunks_a
    assert revisit.items_updated == 0
    assert revisit.items_skipped >= 1

    # HONEST LIMIT: a 1-NATIVE-pixel shift resamples the reduced plane
    # (reduction bins move with the window origin), so every chunk key
    # changes and the full plane re-uploads — reuse here would show wrong
    # texels.
    payload_b = reduced_anchored_payload(data, 101)
    assert not np.array_equal(np.asarray(payload_b.image), plane_a)
    shifted = commit_anchored(pool, {0: payload_b}, tile_shape=(R_HEIGHT, R_EXTENT))
    chunks_b = set(pool.tile_chunk_residency[0])
    assert len(chunks_b) == 4
    assert chunks_a & chunks_b == set(), (
        "reduced-LOD chunks must not be reused across a native-pixel shift: "
        "the reduction bins moved, so equal keys would alias different texels"
    )
    assert pool.chunk_upload_count == 8
    assert shifted.items_updated == 1

    # Scroll-back to the original window: the original chunks are still
    # resident and byte-valid (same bins, same texels) — zero uploads.
    back = commit_anchored(
        pool,
        {0: reduced_anchored_payload(data, 100)},
        tile_shape=(R_HEIGHT, R_EXTENT),
    )
    assert pool.chunk_upload_count == 8
    assert set(pool.tile_chunk_residency[0]) == chunks_a
    assert back.items_updated == 0


def test_mixed_factor_planes_share_chunk_shape_class_without_cross_eviction():
    native_data = _data()
    reduced_data = _reduced_data()
    # Budgeted pool with headroom: mixed LODs must coexist, not thrash.
    pool = TextureAtlasPool(FakeGloo(), budget_bytes=256 * 1024 * 1024)

    native_payload = anchored_payload(native_data, 100, tile_number=0)
    reduced_payload = reduced_anchored_payload(reduced_data, 100, tile_number=1)
    stats = commit_anchored(
        pool,
        {0: native_payload, 1: reduced_payload},
        tile_shape=(R_HEIGHT, R_EXTENT),
    )
    assert set(stats.presented_tiles) == {0, 1}
    assert set(pool.tile_chunk_residency) == {0, 1}
    assert len(pool.tile_chunk_residency[0]) == 10
    assert len(pool.tile_chunk_residency[1]) == 4
    # Both LODs' chunks live in the SAME 256x256 shape class.
    slots = pool.resident_slots
    chunk_pages = {
        slots[key][0]
        for keys in pool.tile_chunk_residency.values()
        for key in keys
    }
    assert {pool.pages[index].tile_shape for index in chunk_pages} == {
        tuple(ANCHORED_CHUNK_SHAPE)
    }
    assert pool.eviction_count == 0

    # Re-presenting both again evicts nothing and uploads nothing.
    uploads_before = pool.chunk_upload_count
    commit_anchored(
        pool,
        {
            0: anchored_payload(native_data, 100, tile_number=0),
            1: reduced_anchored_payload(reduced_data, 100, tile_number=1),
        },
        tile_shape=(R_HEIGHT, R_EXTENT),
    )
    assert pool.chunk_upload_count == uploads_before
    assert pool.eviction_count == 0


def test_reduced_draw_parts_tile_layout_region_exactly():
    data = _reduced_data()
    pool = TextureAtlasPool(FakeGloo())
    commit_anchored(
        pool,
        {0: reduced_anchored_payload(data, 100)},
        tile_shape=(R_HEIGHT, R_EXTENT),
    )
    parts = pool.tile_draw_parts[0]
    assert len(parts) == 4
    # Divisible extents: each plane pixel spans exactly R_FACTOR world units,
    # so world rects are the plane rects scaled by the factor.
    plan = _payload_chunk_plan(reduced_anchored_payload(data, 100))
    world_rects = sorted(part.world_rect for part in parts)
    expected = sorted(
        (
            float(px0 * R_FACTOR),
            float(py0 * R_FACTOR),
            float(px1 * R_FACTOR),
            float(py1 * R_FACTOR),
        )
        for py0, py1, px0, px1 in (chunk.plane_rect for chunk in plan)
    )
    assert world_rects == expected
    # No gaps, no overlaps: areas sum to the region, edges land on the
    # region boundary, and edge sets tile it.
    area = sum(
        (part.world_rect[2] - part.world_rect[0]) * (part.world_rect[3] - part.world_rect[1])
        for part in parts
    )
    assert area == R_HEIGHT * R_EXTENT
    xs = sorted({part.world_rect[0] for part in parts} | {part.world_rect[2] for part in parts})
    ys = sorted({part.world_rect[1] for part in parts} | {part.world_rect[3] for part in parts})
    assert xs == [0.0, 1024.0, float(R_EXTENT)]
    assert ys == [0.0, 1024.0, float(R_HEIGHT)]
    # Clipped chunks sample only the valid sub-window of their slot: the
    # 128-row bottom chunks crop half the slot's v span.
    page = pool.pages[next(iter({slot[0] for slot in chunk_pages_and_slots(pool).values()}))]
    for part, chunk in zip(parts, plan):
        py0, py1, px0, px1 = chunk.plane_rect
        u0, v0, u1, v1 = page.uv_for_slot(
            chunk_pages_and_slots(pool)[chunk.key][1]
        )
        assert part.uv_rect[0] == u0 and part.uv_rect[1] == v0
        assert part.uv_rect[2] <= u1 and part.uv_rect[3] <= v1
        assert part.uv_rect[2] - part.uv_rect[0] == (u1 - u0) * ((px1 - px0) / CHUNK)
        assert part.uv_rect[3] - part.uv_rect[1] == (v1 - v0) * ((py1 - py0) / CHUNK)


def test_reduced_non_divisible_extent_matches_single_quad_stretch():
    """ceil-shaped planes (extent not divisible by the factor) stay eligible.

    ``reduce_box_mean`` produces ``ceil(extent / factor)`` samples per axis
    (the trailing partial box is averaged), and the classic single reduced
    quad stretches that plane uniformly over the native region.  Chunk world
    rects apply the SAME uniform stretch — identical placement, shared
    edges, exact region coverage at the extremes.
    """

    data = _reduced_data()
    height = R_HEIGHT - 2  # 1534 -> plane_h ceil(1534/4) = 384
    extent = R_EXTENT - 2  # 2046 -> plane_w ceil(2046/4) = 512
    payload = reduced_anchored_payload(data, 100, height=height, extent=extent)
    assert np.asarray(payload.image).shape[:2] == (384, 512)
    assert _payload_chunked_eligible(payload)
    pool = TextureAtlasPool(FakeGloo())
    stats = commit_anchored(pool, {0: payload}, tile_shape=(height, extent))
    assert stats.presented_tiles == (0,)
    assert len(pool.tile_chunk_residency[0]) == 4
    parts = pool.tile_draw_parts[0]
    xs = sorted({part.world_rect[0] for part in parts} | {part.world_rect[2] for part in parts})
    ys = sorted({part.world_rect[1] for part in parts} | {part.world_rect[3] for part in parts})
    # Uniform stretch: plane pixel p maps to p * extent / plane_extent, the
    # same mapping the single quad applies to its texels.
    assert xs == [0.0, (256 * extent) / 512.0, float(extent)]
    assert ys == [0.0, (256 * height) / 384.0, float(height)]
    # Adjacent chunks share their edge values bitwise: 2x2 parts produce
    # exactly 3 distinct edges per axis, and the extremes land exactly on
    # the region boundary.
    assert len(xs) == 3 and len(ys) == 3
    assert xs[-1] == float(extent) and ys[-1] == float(height)


def test_reduced_ineligible_payloads_fall_back_to_classic():
    data = _reduced_data()

    # Anisotropic reduction (x by 4, y by 1): the scalar-factor ceil relation
    # cannot hold on the unreduced axis -> classic path, never guessed.
    native = np.ascontiguousarray(data[:, 100 : 100 + R_EXTENT])
    aniso_plane = np.asarray(reduce_box_mean(native, (R_FACTOR, 1)))
    aniso = DisplayTilePayload(
        tile_number=0,
        source_index=0,
        image=aniso_plane,
        histogram_data=None,
        source_id=("window", 100, R_EXTENT, 0, "lod-aniso", R_FACTOR),
        lod=LodInfo(
            level=R_LEVEL,
            factor=R_FACTOR,
            source_shape=native.shape[:2],
            texture_shape=aniso_plane.shape[:2],
            gutter=0,
        ),
        source_anchor=PayloadSourceAnchor(
            content_key=R_CONTENT_KEY, source_rect=(0, R_HEIGHT, 100, 100 + R_EXTENT)
        ),
    )
    assert not _payload_chunked_eligible(aniso)

    # LodInfo that disagrees with the anchor rect (source_shape lies about
    # the native extent) -> classic path.
    good = reduced_anchored_payload(data, 100)
    lying_lod = LodInfo(
        level=R_LEVEL,
        factor=R_FACTOR,
        source_shape=(R_HEIGHT - R_FACTOR, R_EXTENT),
        texture_shape=np.asarray(good.image).shape[:2],
        gutter=0,
    )
    lying = replace(good, lod=lying_lod)
    assert not _payload_chunked_eligible(lying)

    # Gutter payloads keep the classic path at any factor.
    guttered = replace(
        good,
        lod=LodInfo(
            level=R_LEVEL,
            factor=R_FACTOR,
            source_shape=(R_HEIGHT, R_EXTENT),
            texture_shape=np.asarray(good.image).shape[:2],
            gutter=1,
        ),
    )
    assert not _payload_chunked_eligible(guttered)

    # A floor-shaped plane (one row short of the ceil shape) is not the
    # reducer's output for this rect -> classic path.
    floor_plane = np.ascontiguousarray(np.asarray(good.image)[:-1, :])
    floored = replace(
        good,
        image=floor_plane,
        texture_data=floor_plane,
        lod=LodInfo(
            level=R_LEVEL,
            factor=R_FACTOR,
            source_shape=(R_HEIGHT, R_EXTENT),
            texture_shape=floor_plane.shape[:2],
            gutter=0,
        ),
    )
    assert not _payload_chunked_eligible(floored)
    pool = TextureAtlasPool(FakeGloo())
    stats = commit_anchored(pool, {0: floored}, tile_shape=(R_HEIGHT, R_EXTENT))
    assert stats.presented_tiles == (0,)
    assert 0 in pool.tile_slots
    assert 0 not in pool.tile_chunk_residency
    assert 0 not in pool.tile_draw_parts


def test_reduced_warm_payloads_make_visible_commit_upload_free():
    data = _reduced_data()
    other = np.random.default_rng(17).random((R_HEIGHT, R_DATA_WIDTH), dtype=np.float32)
    plane_b_key = ("src-anchored", "doc-rev-0", "reduced-plane-1")
    pool = TextureAtlasPool(FakeGloo())

    payload_a = reduced_anchored_payload(data, 100)
    commit_anchored(pool, {0: payload_a}, tile_shape=(R_HEIGHT, R_EXTENT))
    chunks_a = set(pool.tile_chunk_residency[0])

    warm_payload = replace(
        reduced_anchored_payload(other, 100, content_key=plane_b_key),
        source_id=("prefetch-warm", "reduced-plane-1"),
    )
    stats = pool.warm_payloads(
        {0: warm_payload},
        tile_shape=(R_HEIGHT, R_EXTENT),
        rgb_already_windowed=False,
    )
    assert stats.items_updated == 1
    assert stats.texture_uploads == 4
    assert stats.storage_evictions == 0
    # Warm chunks are pure residency: tile 0 still presents plane A.
    assert set(pool.tile_chunk_residency[0]) == chunks_a

    uploads_before = pool.chunk_upload_count
    visible = commit_anchored(pool, {0: warm_payload}, tile_shape=(R_HEIGHT, R_EXTENT))
    assert pool.chunk_upload_count == uploads_before
    assert visible.texture_uploads == 0
    assert visible.presented_tiles == (0,)
    assert len(pool.tile_draw_parts[0]) == 4
    # Plane A's chunks survived as warm residency.
    assert all(key in pool.resident_slots for key in chunks_a)


def test_eviction_free_placement_never_relocates_foreign_page_residents():
    """Speculative warm must never disturb existing residency.

    The same-page bucketing branch releases a foreign-page resident and
    re-uploads it on the chosen page — legitimate for visible commits, a
    presentation-outcome change when done speculatively. Under
    allow_eviction=False a set straddling pages is a denial, not a move.
    """

    import pytest as _pytest

    from arrayscope.gpu.page_table import PageSlot

    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    payload = anchored_payload(data, 100)
    commit(pool, {0: payload})
    plan = _payload_chunk_plan(payload)
    keys = [chunk.key for chunk in plan]

    # Manufacture a straddle: move ONE chunk's residency to a fresh page of
    # the same class, keeping page bookkeeping coherent.
    pool._ensure_class_capacity((256, 256), pool._class_capacity((256, 256)) + len(keys))
    moved = keys[0]
    old_page_index, old_slot = pool.resident_slots[moved]
    target_index = next(
        index
        for index, page in enumerate(pool.pages)
        if page.tile_shape == (256, 256) and index != old_page_index
    )
    old_page = pool.pages[old_page_index]
    old_page.slot_owners[old_slot] = None
    old_page._free_slots.append(old_slot)
    pool._page_table.unbind(moved)
    target_page = pool.pages[target_index]
    new_slot = target_page.take_free_slot(moved)
    pool._page_table.bind(
        moved, PageSlot("vispy-atlas", target_index, new_slot), nbytes=0
    )

    resident_before = dict(pool.resident_slots)

    with _pytest.raises(AtlasCapacityError):
        pool._chunk_slots_for(
            tuple(keys),
            protected_keys=set(),
            near_keys=set(),
            allow_eviction=False,
        )

    # Nothing moved: the straddling resident stayed exactly where it was.
    assert pool.resident_slots == resident_before


# ---------------------------------------------------------------------------
# Field defect 2026-07-15: stale draw parts / missing physical truth rows
# across chunked->classic transitions and index-window retargets.
# ---------------------------------------------------------------------------


def classic_payload(tile_number, data, *, source_id):
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=np.ascontiguousarray(data[:, :EXTENT]),
        histogram_data=None,
        source_id=source_id,
    )


def commit_classic(pool, payloads):
    """Commit without world regions: every payload takes the classic path."""

    _uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=(HEIGHT, EXTENT),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(payloads),
    )
    return stats


def anchored_content_payload(data, *, content_key, source_id, tile_number=0):
    """Anchored payload whose source id does NOT embed the tile number, so a
    retarget can present the identical resident key under a new tile."""

    plane = np.ascontiguousarray(data[:, :EXTENT])
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=0,
        image=plane,
        histogram_data=None,
        source_id=source_id,
        quality="exact",
        source_anchor=PayloadSourceAnchor(
            content_key=content_key,
            source_rect=(0, plane.shape[0], 0, EXTENT),
        ),
    )


def test_classic_re_present_of_same_key_clears_stale_draw_parts():
    """The residency key does not include the anchor: the same payload
    identity committed chunked once (world region supplied) and classically
    later (no region) must not keep the chunked presentation's UV-cropped
    draw parts — they sample a sub-window of the plane stretched over the
    whole tile (the field's "zoomed tile")."""

    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    commit(pool, {0: anchored_payload(data, 100)})
    assert 0 in pool.tile_draw_parts and 0 in pool.tile_chunk_residency

    stats = commit_classic(pool, {0: anchored_payload(data, 100)})
    assert stats.presented_tiles == (0,)
    assert 0 not in pool.tile_draw_parts
    assert 0 not in pool.tile_chunk_residency
    # The key now owns whole-tile residency: it must no longer be registered
    # chunked, or a later displacement would drop its identity records while
    # its classic slot stays resident.
    assert not pool._chunked_tile_keys
    rows = pool.tile_truth_physical_rows()
    assert 0 in rows and rows[0]["physical_texture_dtype"] == "float32"


def test_index_retarget_remap_keeps_chunked_records_and_draw_parts():
    """One commit moves content A from tile 0 to tile 1 while tile 0 gets new
    content C.  The mapping loop processes tile 0 first, transiently leaving
    key A unreferenced; forgetting it at that moment destroyed tile 1's
    just-committed chunk links, draw parts, AND physical upload records
    (field defect 2026-07-15: zoomed tiles + ``phys None/None`` rows)."""

    data_a = _data()
    data_b = np.random.default_rng(9).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
    data_c = np.random.default_rng(11).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
    pool = TextureAtlasPool(FakeGloo())
    commit(
        pool,
        {
            0: anchored_content_payload(data_a, content_key=("doc", "A"), source_id=("plane", "A")),
            1: anchored_content_payload(data_b, content_key=("doc", "B"), source_id=("plane", "B"), tile_number=1),
        },
    )
    assert set(pool.tile_chunk_residency) == {0, 1}
    chunk_count = len(pool.tile_chunk_residency[0])

    uploads_before = pool.chunk_upload_count
    stats = commit(
        pool,
        {
            0: anchored_content_payload(data_c, content_key=("doc", "C"), source_id=("plane", "C")),
            1: anchored_content_payload(data_a, content_key=("doc", "A"), source_id=("plane", "A"), tile_number=1),
        },
    )
    assert stats.presented_tiles == (0, 1)
    # The remapped tile keeps its full chunked presentation.
    assert len(pool.tile_chunk_residency.get(1, ())) == chunk_count
    assert len(pool.tile_draw_parts.get(1, ())) == chunk_count
    assert pool.presented_identities()[1] == ("plane", "A")
    # Content A moved tiles without re-uploading a single chunk.
    assert pool.chunk_upload_count - uploads_before == chunk_count  # only C's chunks
    # Every drawn tile has a physical truth row (the field showed phys
    # None/None on exactly the remapped tiles).
    rows = pool.tile_truth_physical_rows()
    assert 0 in rows and 1 in rows


def test_commit_end_sweep_reclaims_records_of_unpresented_chunked_keys():
    """Deferred forgetting must not leak: once a chunked key genuinely stops
    being presented, the commit-end sweep drops its identity records (the
    chunks themselves stay warm for a later re-present)."""

    data_a = _data()
    data_c = np.random.default_rng(11).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
    pool = TextureAtlasPool(FakeGloo())
    payload_a = anchored_content_payload(data_a, content_key=("doc", "A"), source_id=("plane", "A"))
    commit(pool, {0: payload_a})
    resident_key_a = next(iter(pool._chunked_tile_keys))

    commit(pool, {0: anchored_content_payload(data_c, content_key=("doc", "C"), source_id=("plane", "C"))})
    assert resident_key_a not in pool._chunked_tile_keys
    assert resident_key_a not in pool.source_ids
    assert resident_key_a not in pool.physical_upload_records
    # Re-presenting A re-links the warm chunks and rebuilds its records.
    uploads_before = pool.chunk_upload_count
    commit(pool, {0: anchored_content_payload(data_a, content_key=("doc", "A"), source_id=("plane", "A"))})
    assert pool.chunk_upload_count == uploads_before
    assert 0 in pool.tile_truth_physical_rows()


def test_warm_promoted_classic_tile_has_physical_truth_row():
    """Classic warm uploads are real texels: the later visible commit
    presents them through the acknowledged-identity skip path (zero
    uploads), so the warm path itself must leave the physical upload record
    (field defect 2026-07-15: ``phys None/None`` on prefetch-warmed montage
    tiles scrolled into view)."""

    data = _data()
    pool = TextureAtlasPool(FakeGloo())
    seed = classic_payload(1, data, source_id=("montage-tile", 8))
    commit_classic(pool, {1: seed})

    warm = classic_payload(0, data, source_id=("montage-tile", 7))
    pool.warm_payloads({0: warm}, tile_shape=(HEIGHT, EXTENT), rgb_already_windowed=False)

    stats = commit_classic(pool, {0: warm, 1: seed})
    assert 0 in stats.presented_tiles
    assert stats.texture_uploads == 0  # promotion reuses the warm upload
    rows = pool.tile_truth_physical_rows()
    assert 0 in rows and 1 in rows
    assert rows[0]["physical_texture_dtype"] == "float32"


def test_window_session_to_classic_montage_same_tile_numbers_no_stale_state():
    """The task's field sequence: a chunked (anchored) window session for
    tiles 0..N, then a classic montage presentation reusing the SAME tile
    numbers without an intervening reset, then a remap-style re-present.
    No montage tile may draw with leftover chunk draw parts, and every drawn
    tile must have a physical truth row."""

    pool = TextureAtlasPool(FakeGloo())
    window = {
        t: anchored_content_payload(
            np.random.default_rng(20 + t).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32),
            content_key=("doc", f"W{t}"),
            source_id=("plane", f"W{t}"),
            tile_number=t,
        )
        for t in range(3)
    }
    commit(pool, window)
    assert all(t in pool.tile_draw_parts for t in range(3))

    montage_data = {
        t: np.random.default_rng(30 + t).standard_normal((HEIGHT, DATA_WIDTH)).astype(np.float32)
        for t in range(4)
    }
    montage = {
        t: classic_payload(t, montage_data[t], source_id=("montage-tile", t)) for t in range(3)
    }
    commit_classic(pool, montage)
    assert all(t not in pool.tile_draw_parts for t in range(3))
    assert all(t not in pool.tile_chunk_residency for t in range(3))
    rows = pool.tile_truth_physical_rows()
    assert all(t in rows for t in range(3))

    # Remap-style re-present: content shifts one tile number; the shifted
    # keys are already resident, so the pool presents them via the skip path.
    remap = {
        t: classic_payload(t, montage_data[t + 1], source_id=("montage-tile", t + 1))
        for t in range(3)
    }
    stats = commit_classic(pool, remap)
    assert stats.presented_tiles == (0, 1, 2)
    rows = pool.tile_truth_physical_rows()
    assert all(t in rows for t in range(3))
    assert all(t not in pool.tile_draw_parts for t in range(3))
    assert pool.presented_identities() == {
        0: ("montage-tile", 1),
        1: ("montage-tile", 2),
        2: ("montage-tile", 3),
    }
