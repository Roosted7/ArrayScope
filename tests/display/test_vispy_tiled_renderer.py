from __future__ import annotations

import numpy as np
import pytest

from arrayscope.display.vispy_tiled_renderer import AtlasCapacityError, TextureAtlasPool, _payload_textures
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
    assert pool.scalar_texture.initial_data is None
    assert pool.color_texture.initial_data is None
    assert pool.scalar_texture.shape[-1] == 1
    assert pool.color_texture.shape[-1] == 3
    assert all(offset is not None for _data, offset, _copy in pool.scalar_texture.updates)
    assert all(offset is not None for _data, offset, _copy in pool.color_texture.updates)


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
