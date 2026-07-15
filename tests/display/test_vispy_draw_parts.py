"""TileDrawPart geometry: a view tile drawing as N UV-cropped quads (ADR 0055 G3)."""

import numpy as np

from arrayscope.display.backends.vispy.tiles import (
    TileDrawPart,
    _quad_buffers,
    _tile_quad_rects,
)
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.tile_layout import TileLayoutRegion


def payload(tile_number: int, value: float = 1.0) -> DisplayTilePayload:
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=np.full((2, 2), value, dtype=np.float32),
        histogram_data=None,
        source_id=("tile", tile_number, value),
    )


def region(tile_number: int, x: int = 0, y: int = 0, width: int = 2, height: int = 2) -> TileLayoutRegion:
    return TileLayoutRegion(tile_number=tile_number, source_index=tile_number, x=x, y=y, width=width, height=height)


FULL_UV = (0.0, 0.0, 1.0, 1.0)


def test_default_single_quad_unchanged_without_parts():
    layout = {0: region(0), 1: region(1, x=2)}
    payloads = {0: payload(0), 1: payload(1)}
    uvs = {0: FULL_UV, 1: (0.5, 0.0, 1.0, 0.5)}
    baseline = _quad_buffers(layout, payloads, uvs, rgb_already_windowed=False)
    with_empty = _quad_buffers(layout, payloads, uvs, rgb_already_windowed=False, draw_parts={})
    for a, b in zip(baseline, with_empty):
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
    assert vertices[:6, 0].min() == 0.0 and vertices[:6, 0].max() == 1.0
    assert vertices[6:, 0].min() == 1.0 and vertices[6:, 0].max() == 2.0
    # Each quad samples its own cropped UV rect, not the slot's full rect.
    assert texcoords[:6, 0].min() == 0.5 and texcoords[:6, 0].max() == 1.0
    assert texcoords[6:, 0].min() == 0.0 and texcoords[6:, 0].max() == 0.25
    assert np.all(modes == modes[0])


def test_parts_and_default_tiles_mix_with_correct_offsets():
    layout = {0: region(0), 1: region(1, x=2), 2: region(2, x=4)}
    payloads = {n: payload(n) for n in (0, 1, 2)}
    uvs = {n: FULL_UV for n in (0, 1, 2)}
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
