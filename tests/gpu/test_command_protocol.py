"""Protocol-shape tests (no GPU, no Qt): the seam every renderer implements."""

import pytest

from arrayscope.gpu.command_protocol import (
    DispatchHistogram,
    DisplayMapping,
    FrameSubmission,
    PresentGeneration,
    TileInstance,
    UpdateTileInstances,
)


def test_display_mapping_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mapping mode"):
        DisplayMapping(mode="sqrt")


def test_display_mapping_rejects_empty_levels_window():
    with pytest.raises(ValueError, match="non-empty"):
        DisplayMapping(mode="magnitude", level_lo=1.0, level_hi=1.0)


def test_tile_instance_normalizes_and_clamps():
    tile = TileInstance(dst_rect=(0, 0, 1, 1), src_origin=(10, 20), src_size=(256, 128), lod_level=-3)
    assert tile.dst_rect == (0.0, 0.0, 1.0, 1.0)
    assert tile.src_origin == (10.0, 20.0)
    assert tile.lod_level == 0


def test_tile_instance_rejects_malformed_geometry():
    with pytest.raises((ValueError, TypeError)):
        TileInstance(dst_rect=(0, 0, 1), src_origin=(0, 0), src_size=(1, 1))


def test_histogram_command_rejects_nonpositive_bins():
    with pytest.raises(ValueError, match="positive"):
        DispatchHistogram(keys=(), bins=0)


def test_frame_submission_freezes_command_order():
    tiles = UpdateTileInstances(tiles=(TileInstance((0, 0, 1, 1), (0, 0), (1, 1)),))
    sub = FrameSubmission(generation=7, commands=[tiles, PresentGeneration(7)])
    assert sub.generation == 7
    assert isinstance(sub.commands, tuple)
    assert sub.commands[-1] == PresentGeneration(7)
