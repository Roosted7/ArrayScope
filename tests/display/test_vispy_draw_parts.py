"""TileDrawPart geometry: a view tile drawing as N UV-cropped quads (ADR 0055 G3)."""

import numpy as np

from arrayscope.display.backends.vispy.tiles import (
    TextureAtlasPool,
    TileDrawPart,
    _quad_buffers,
    _tile_quad_rects,
)
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.tile_layout import TileLayoutRegion
from arrayscope.gpu import ChunkLod, DataChunkKey, PageSlot
from tests.display.vispy_test_utils import FakeGloo


def payload(tile_number: int, value: float = 1.0) -> DisplayTilePayload:
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=np.full((2, 2), value, dtype=np.float32),
        histogram_data=None,
        source_id=("tile", tile_number, value),
    )


def region(
    tile_number: int, x: int = 0, y: int = 0, width: int = 2, height: int = 2
) -> TileLayoutRegion:
    return TileLayoutRegion(
        tile_number=tile_number, source_index=tile_number, x=x, y=y, width=width, height=height
    )


FULL_UV = (0.0, 0.0, 1.0, 1.0)


def test_default_single_quad_unchanged_without_parts():
    layout = {0: region(0), 1: region(1, x=2)}
    payloads = {0: payload(0), 1: payload(1)}
    uvs = {0: FULL_UV, 1: (0.5, 0.0, 1.0, 0.5)}
    baseline = _quad_buffers(layout, payloads, uvs, rgb_already_windowed=False)
    with_empty = _quad_buffers(layout, payloads, uvs, rgb_already_windowed=False, draw_parts={})
    for a, b in zip(baseline, with_empty, strict=False):
        assert np.array_equal(a, b)
    vertices, texcoords, modes = baseline
    assert vertices.shape == (12, 2)
    assert texcoords.shape == (12, 2)
    assert modes.shape == (12,)


def test_registered_parts_replace_the_single_quad():
    layout = {0: region(0)}
    payloads = {0: payload(0)}
    parts = (
        TileDrawPart(world_rect=(0.0, 0.0, 1.0, 2.0), uv_rect=(0.5, 0.0, 1.0, 1.0)),
        TileDrawPart(world_rect=(1.0, 0.0, 2.0, 2.0), uv_rect=(0.0, 0.0, 0.25, 1.0)),
    )
    vertices, texcoords, modes = _quad_buffers(
        layout, payloads, {0: FULL_UV}, rgb_already_windowed=False, draw_parts={0: parts}
    )
    assert vertices.shape == (12, 2)
    # First quad spans world x in [0, 1], second x in [1, 2].
    assert vertices[:6, 0].min() == 0.0
    assert vertices[:6, 0].max() == 1.0
    assert vertices[6:, 0].min() == 1.0
    assert vertices[6:, 0].max() == 2.0
    # Each quad samples its own cropped UV rect, not the slot's full rect.
    assert texcoords[:6, 0].min() == 0.5
    assert texcoords[:6, 0].max() == 1.0
    assert texcoords[6:, 0].min() == 0.0
    assert texcoords[6:, 0].max() == 0.25
    assert np.all(modes == modes[0])


def test_parts_and_default_tiles_mix_with_correct_offsets():
    layout = {0: region(0), 1: region(1, x=2), 2: region(2, x=4)}
    payloads = {n: payload(n) for n in (0, 1, 2)}
    uvs = dict.fromkeys((0, 1, 2), FULL_UV)
    parts = {
        1: (
            TileDrawPart(world_rect=(2.0, 0.0, 3.0, 2.0), uv_rect=(0.0, 0.0, 0.5, 1.0)),
            TileDrawPart(world_rect=(3.0, 0.0, 4.0, 2.0), uv_rect=(0.5, 0.0, 1.0, 1.0)),
            TileDrawPart(world_rect=(2.0, 0.0, 4.0, 1.0), uv_rect=(0.0, 0.0, 1.0, 0.5)),
        )
    }
    vertices, _texcoords, modes = _quad_buffers(
        layout, payloads, uvs, rgb_already_windowed=False, draw_parts=parts
    )
    # tile 0: 1 quad, tile 1: 3 quads, tile 2: 1 quad -> 5 quads = 30 verts.
    assert vertices.shape == (30, 2)
    assert modes.shape == (30,)
    # Tile 2's quad sits after tile 1's three quads (sorted tile order).
    assert vertices[24:, 0].min() == 4.0


def test_tile_quad_rects_reports_parts_or_default():
    layout = {0: region(0, x=1, y=2, width=3, height=4)}
    uvs = {0: (0.1, 0.2, 0.3, 0.4)}
    default = _tile_quad_rects(0, layout, uvs, None)
    assert default == (((1.0, 2.0, 4.0, 6.0), (0.1, 0.2, 0.3, 0.4)),)
    part = TileDrawPart(world_rect=(0, 0, 1, 1), uv_rect=(0, 0, 0.5, 0.5))
    assert _tile_quad_rects(0, layout, uvs, {0: (part,)}) == (
        ((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.5, 0.5)),
    )
    # No layout/UV -> nothing to draw.
    assert _tile_quad_rects(9, layout, uvs, None) == ()


def test_page_compaction_atomically_remaps_bindings_and_draw_parts():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=2)
    pool.ensure_layout(tile_shape=(2, 2), count=2, storage_mode="scalar")
    assert len(pool.pages) == 2

    target = DataChunkKey(
        document_generation=("doc", 1),
        operation_key=("op", "identity"),
        lod=ChunkLod(reduction=(0, 0), reducer="mean"),
        chunk_origin=(2, 2),
        chunk_shape=(2, 2),
        dtype="float32",
    )
    coarse = DataChunkKey(
        document_generation=target.document_generation,
        operation_key=target.operation_key,
        lod=ChunkLod(reduction=(2, 2), reducer="mean"),
        chunk_origin=(0, 0),
        chunk_shape=(8, 8),
        dtype="float32",
    )
    survivor = pool.pages[1]
    slot = survivor.take_free_slot(coarse)
    assert slot == 0
    pool._bind_resident_slot(coarse, 1, slot, survivor)
    pool.source_ids[coarse] = coarse
    pool.acknowledged_identities[coarse] = coarse

    single = pool.resolve_page_targets({7: target})[7]
    multi = pool.resolve_tile_page_targets(
        {7: (target,)},
        owner_scope=("session", 1),
    )[7]
    assert single is not None
    assert multi is not None
    assert single.actual_key == coarse
    assert single.slot == PageSlot("vispy-atlas", 1, 0)
    pool.tile_draw_parts[7] = (
        TileDrawPart(
            world_rect=(3.0, 5.0, 5.0, 7.0),
            uv_rect=(0.0, 0.0, 1.0, 1.0),
            page_index=1,
        ),
    )

    layout = {7: region(7, x=3, y=5)}
    payloads = {7: payload(7)}
    before_geometry = _quad_buffers(
        layout,
        payloads,
        pool.tile_uvs,
        rgb_already_windowed=False,
        draw_parts=pool.tile_draw_parts,
    )
    generation_before = single.binding_generation

    # Page zero is empty.  Dropping it moves the surviving physical page from
    # index one to zero without changing its slot, pixels, or draw geometry.
    pool._drop_pages((0,))

    authoritative = pool._page_table.resolve(target)
    assert authoritative is not None
    assert pool.pages == [survivor]
    assert authoritative.actual_key == coarse
    assert authoritative.slot == PageSlot("vispy-atlas", 0, 0)
    assert authoritative.binding_generation > generation_before
    assert pool.page_target_resolutions[7] == authoritative
    assert pool.tile_page_target_resolutions[7] == (authoritative,)
    assert pool.tile_slots[7] == (0, 0)
    assert pool.tile_draw_parts[7][0].page_index == 0

    after_geometry = _quad_buffers(
        layout,
        payloads,
        pool.tile_uvs,
        rgb_already_windowed=False,
        draw_parts=pool.tile_draw_parts,
    )
    for before, after in zip(before_geometry, after_geometry, strict=True):
        np.testing.assert_array_equal(after, before)

    # Returning this tile to a non-page-backed/native mapping must release the
    # canonical resolution cache and its owner pin. Otherwise physical truth
    # continues to claim a coarse binding that no draw part consumes.
    assert pool._page_table.is_pinned(coarse)
    pool._set_tile_mapping(
        7,
        coarse,
        0,
        0,
        survivor.uv_for_slot(0),
        chunked=True,
    )
    assert 7 not in pool.page_target_resolutions
    assert 7 not in pool.tile_page_target_resolutions
    assert 7 not in pool._tile_page_pin_owners
    assert not pool._page_table.is_pinned(coarse)
