"""Glyph atlas bake/cache/growth semantics (queue row 3 text gap).

Offscreen ring: the atlas is pure CPU + Qt rasterization, so every oracle
here is exact.  The GPU half (glyph quads sampling this atlas) is pinned by
``tests/gpu/test_wgpu_command_protocol.py`` and the wgpu view tests.
"""

import numpy as np
import pytest

from arrayscope.core.trace import TRACE, close_trace, configure_trace
from arrayscope.display.glyph_atlas import GlyphAtlas


def _coverage(atlas: GlyphAtlas, entry) -> int:
    cell = atlas.image[entry.y : entry.y + entry.height, entry.x : entry.x + entry.width]
    return int(np.count_nonzero(cell))


def test_glyph_bakes_once_and_cache_hits_leave_the_version_alone(qt_app):
    atlas = GlyphAtlas()
    version_empty = atlas.version
    entry = atlas.glyph("monospace", 9, "A")
    assert _coverage(atlas, entry) > 0, "baked glyph must have ink"
    assert atlas.version > version_empty

    version_baked = atlas.version
    again = atlas.glyph("monospace", 9, "A")
    assert again == entry
    assert atlas.version == version_baked, "cache hit must not dirty the atlas"


def test_dpr_scaled_raster_differs_from_dpr_one(qt_app):
    # Crispness proxy: DPR 2 bakes at twice the pixel size, so the raster is
    # a genuinely different (finer) image, not a scaled copy of the DPR 1
    # cell.  Distinct cache keys guarantee both live in the atlas at once.
    atlas = GlyphAtlas()
    dpr1 = atlas.glyph("monospace", 9, "g")
    dpr2 = atlas.glyph("monospace", 18, "g")
    assert (dpr2.width, dpr2.height) != (dpr1.width, dpr1.height)
    assert dpr2.height > dpr1.height
    assert _coverage(atlas, dpr2) > _coverage(atlas, dpr1)


def test_layout_text_places_lines_and_skips_spaces(qt_app):
    atlas = GlyphAtlas()
    layout = atlas.layout_text("ab cd\nx", "monospace", 9)
    assert len(layout.placements) == 5, "spaces advance but never place glyphs"
    line_step = atlas.line_height("monospace", 9)
    first_line = [p for p in layout.placements if p.y == 0.0]
    second_line = [p for p in layout.placements if p.y == pytest.approx(line_step)]
    assert len(first_line) == 4 and len(second_line) == 1
    xs = [p.x for p in first_line]
    assert xs == sorted(xs) and len(set(xs)) == 4
    space_gap = xs[2] - xs[1]
    glyph_gap = xs[1] - xs[0]
    assert space_gap > glyph_gap, "the space must advance the pen"
    assert layout.width >= xs[-1]
    assert layout.height >= 2 * line_step


def test_atlas_growth_is_bounded_and_preserves_baked_cells(qt_app):
    atlas = GlyphAtlas(initial_size=32, max_size=128)
    first = atlas.glyph("monospace", 9, "0")
    first_cell = atlas.image[
        first.y : first.y + first.height, first.x : first.x + first.width
    ].copy()
    glyphs = "123456789ABCDEFGHIJKLMNOPQRSTUV"
    for char in glyphs:
        atlas.glyph("monospace", 9, char)
    assert atlas.size > 32, "the working set must have forced growth"
    assert atlas.size <= 128
    assert atlas.evictions == 0
    np.testing.assert_array_equal(
        atlas.image[first.y : first.y + first.height, first.x : first.x + first.width],
        first_cell,
    )


def test_atlas_overflow_evicts_loudly_and_rebakes(qt_app, tmp_path):
    configure_trace(tmp_path / "trace.jsonl")
    try:
        atlas = GlyphAtlas(initial_size=16, max_size=16)
        # 16x16 with padded ~7x12 cells holds very few glyphs; digits overflow.
        for char in "0123456789":
            atlas.glyph("monospace", 9, char)
        assert atlas.evictions >= 1
        events = [
            event
            for event in TRACE.snapshot()
            if event.get("kind") == "wgpu_glyph_atlas_evicted"
        ]
        assert events, "eviction must be loud on the trace bus"
        assert atlas.size == 16, "eviction must never exceed the bound"
        # Post-eviction the atlas still serves the live working set.
        entry = atlas.glyph("monospace", 9, "9")
        assert _coverage(atlas, entry) > 0
    finally:
        close_trace()


def test_oversized_glyph_cell_is_rejected_loudly(qt_app):
    atlas = GlyphAtlas(initial_size=16, max_size=16)
    with pytest.raises(ValueError, match="exceeds the atlas bound"):
        atlas.glyph("monospace", 64, "W")
