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
    TextureAtlasPool,
    _payload_chunk_plan,
    _payload_chunked_eligible,
)
from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor

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
