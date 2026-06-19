from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display.vispy_tiled_renderer import (
    AtlasCapacityError,
    TextureAtlasPool,
    _fit_color,
    _fit_scalar,
    _payload_textures,
)
from arrayscope.window.display_frame import DisplayTilePayload


class FakeTexture2D:
    def __init__(self, data=None, *, shape=None, **kwargs):
        if data is not None and shape is not None:
            raise ValueError("data and shape are mutually exclusive")
        self.initial_data = data
        self.shape = tuple(shape) if shape is not None else tuple(np.shape(data))
        self.kwargs = dict(kwargs)
        self.updates: list[tuple[np.ndarray, tuple[int, int] | None, bool]] = []

    def set_data(self, data, *, offset=None, copy=True):
        self.updates.append((np.array(data, copy=True), offset, bool(copy)))


class FakeGloo:
    Texture2D = FakeTexture2D


def payload(tile_number: int, value: float, *, source_id=None) -> DisplayTilePayload:
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=np.full((2, 2), value, dtype=np.float32),
        histogram_data=None,
        source_id=("tile", tile_number, value) if source_id is None else source_id,
    )


def color_payload(tile_number: int, value: int, *, window_scalar=True) -> DisplayTilePayload:
    image = np.full((2, 2, 3), int(value), dtype=np.uint8)
    scalar = np.full((2, 2), float(value), dtype=np.float32) if window_scalar else None
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=image,
        histogram_data=scalar,
        source_id=("color", tile_number, value, bool(window_scalar)),
    )


def test_atlas_keeps_stable_slots_when_active_set_changes():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4)
    first_uvs, first = pool.update_payloads(
        {0: payload(0, 10.0), 1: payload(1, 20.0)},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
    )
    slot_one = pool.slots[1]

    second_uvs, second = pool.update_payloads(
        {1: payload(1, 20.0), 2: payload(2, 30.0)},
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=2,
    )

    assert first.items_updated == 2
    assert pool.slots[1] == slot_one
    assert second_uvs[1] == first_uvs[1]
    assert second.items_updated == 1
    assert second.items_skipped == 1
    assert second.atlas_evictions == 1
    assert 0 not in pool.source_ids
    assert pool.source_ids[2] == ("tile", 2, 30.0)


def test_atlas_reserve_avoids_progressive_reallocation():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    pool.update_payloads(
        {0: payload(0, 0.0)},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=6,
    )
    serial = pool.serial
    textures = (pool.scalar_texture, pool.color_texture)

    for count in range(2, 7):
        pool.update_payloads(
            {index: payload(index, float(index)) for index in range(count)},
            tile_shape=(2, 2),
            dirty_tiles=tuple(range(count - 1, count)),
            rgb_already_windowed=False,
            reserve_count=6,
        )

    assert pool.capacity == 6
    assert pool.serial == serial
    assert (pool.scalar_texture, pool.color_texture) == textures
    assert pool.rebuild_count == 1


def test_atlas_retains_offscreen_payload_for_later_clean_reuse():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    values = {0: payload(0, 1.0), 1: payload(1, 2.0)}
    pool.update_payloads(
        values,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )
    slot_zero = pool.slots[0]

    pool.update_payloads(
        {1: values[1]},
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=4,
    )
    _uvs, reused = pool.update_payloads(
        {0: values[0]},
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=4,
    )

    assert pool.slots[0] == slot_zero
    assert reused.items_updated == 0
    assert reused.items_skipped == 1
    assert reused.resident_items == 2


def test_atlas_uses_shape_only_gpu_allocation_and_subuploads():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    pool.update_payloads(
        {0: payload(0, 4096.0)},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
    )

    assert pool.cpu_shadow_bytes == 0
    assert pool.storage_mode == "scalar"
    assert pool.scalar_texture.initial_data is None
    assert pool.color_texture.initial_data is not None
    assert pool.scalar_texture.shape[-1] == 1
    assert pool.color_texture.shape == (1, 1, 3)
    assert len(pool.scalar_texture.updates) == 1
    assert not pool.color_texture.updates
    assert all(offset is not None for _data, offset, _copy in pool.scalar_texture.updates)
    assert pool.estimated_gpu_bytes == int(np.prod(pool.atlas_shape)) * 4


def test_atlas_allocates_only_color_plane_for_display_ready_rgb():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    _uvs, stats = pool.update_payloads(
        {0: color_payload(0, 128, window_scalar=False)},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=True,
        reserve_count=2,
    )

    assert pool.storage_mode == "color"
    assert pool.scalar_texture.shape == (1, 1)
    assert pool.color_texture.initial_data is None
    assert not pool.scalar_texture.updates
    assert len(pool.color_texture.updates) == 1
    assert stats.texture_uploads == 1
    assert pool.estimated_gpu_bytes == int(np.prod(pool.atlas_shape)) * 3


def test_atlas_allocates_scalar_and_color_planes_for_windowable_rgb():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    _uvs, stats = pool.update_payloads(
        {0: color_payload(0, 128, window_scalar=True)},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
    )

    assert pool.storage_mode == "scalar_color"
    assert pool.scalar_texture.initial_data is None
    assert pool.color_texture.initial_data is None
    assert len(pool.scalar_texture.updates) == 1
    assert len(pool.color_texture.updates) == 1
    assert stats.texture_uploads == 2
    assert pool.estimated_gpu_bytes == int(np.prod(pool.atlas_shape)) * 7


def test_atlas_rejects_tile_set_that_requires_multiple_pages():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4)
    with pytest.raises(AtlasCapacityError, match="more than one"):
        pool.ensure_layout(tile_shape=(2, 2), count=5)


def test_payload_texture_conversion_preserves_scalar_dynamic_range():
    tile = payload(0, 4096.0)
    scalar, color = _payload_textures(tile, tile_shape=(2, 2), rgb_already_windowed=False)
    np.testing.assert_allclose(scalar, 4096.0)
    assert scalar.dtype == np.float32
    assert color.dtype == np.uint8
    assert not np.any(color)


def test_exact_payload_planes_are_reused_without_staging_copy():
    scalar = np.arange(12, dtype=np.float32).reshape(3, 4)
    color = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)

    assert _fit_scalar(scalar, (3, 4)) is scalar
    assert _fit_color(color, (3, 4)) is color


def test_noncontiguous_or_padded_payload_planes_get_safe_staging_arrays():
    scalar = np.arange(24, dtype=np.float32).reshape(3, 8)[:, ::2]
    color = np.arange(72, dtype=np.uint8).reshape(3, 8, 3)[:, ::2, :]

    fitted_scalar = _fit_scalar(scalar, (3, 4))
    fitted_color = _fit_color(color, (3, 4))

    assert fitted_scalar.flags.c_contiguous
    assert fitted_color.flags.c_contiguous
    assert fitted_scalar is not scalar
    assert fitted_color is not color
    np.testing.assert_array_equal(fitted_scalar, scalar)
    np.testing.assert_array_equal(fitted_color, color)
