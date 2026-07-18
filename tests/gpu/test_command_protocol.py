"""Protocol-shape tests (no GPU, no Qt): the seam every renderer implements."""

import pytest

from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
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


def test_display_mapping_rejects_unknown_scale():
    with pytest.raises(ValueError, match="unknown mapping scale"):
        DisplayMapping(scale="sqrt")


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


def test_histogram_command_dynamic_bounds_and_mapping_are_validated():
    command = DispatchHistogram(
        keys=(), lo=None, hi=None, mode="real", scale="symlog", symlog_constant=2.0
    )
    assert command.lo is None and command.hi is None
    with pytest.raises(ValueError, match="both be set or both be omitted"):
        DispatchHistogram(keys=(), lo=None, hi=1.0)
    with pytest.raises(ValueError, match="mapping mode"):
        DispatchHistogram(keys=(), mode="power")


def test_content_plane_validates_shape_and_representation():
    plane = ContentPlane("doc", "op", (10.0, 20.0), max_lod=-1, representation="rgb8")
    assert plane.plane_shape == (10, 20)
    assert plane.max_lod == 0
    with pytest.raises(ValueError, match="positive"):
        ContentPlane("doc", "op", (0, 8))
    with pytest.raises(ValueError, match="unknown plane representation"):
        ContentPlane("doc", "op", (8, 8), representation="bgr")


def test_bind_content_planes_requires_content_planes():
    bind = BindContentPlanes([ContentPlane("doc", "op", (8, 8))])
    assert isinstance(bind.planes, tuple)
    with pytest.raises(TypeError, match="ContentPlane"):
        BindContentPlanes(("not-a-plane",))


def test_tile_instance_plane_index_validates():
    assert TileInstance((0, 0, 1, 1), (0, 0), (1, 1), plane_index=3).plane_index == 3
    with pytest.raises(ValueError, match="plane_index"):
        TileInstance((0, 0, 1, 1), (0, 0), (1, 1), plane_index=-1)


def test_frame_submission_freezes_command_order():
    tiles = UpdateTileInstances(tiles=(TileInstance((0, 0, 1, 1), (0, 0), (1, 1)),))
    sub = FrameSubmission(generation=7, commands=[tiles, PresentGeneration(7)])
    assert sub.generation == 7
    assert isinstance(sub.commands, tuple)
    assert sub.commands[-1] == PresentGeneration(7)
