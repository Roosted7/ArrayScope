from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.display.backends.vispy.tiles import (
    AtlasCapacityError,
    GpuDeviceLimits,
    GpuMontageLayer,
    GpuWindowedTileVisual,
    TextureAtlasPage,
    TextureAtlasPool,
    _atlas_reserve_count,
    _fit_color,
    _fit_scalar,
    _payload_mode,
    _payload_textures,
    _quad_buffers,
    _resident_key,
    query_gpu_device_limits,
    take_payload_batch,
)
from arrayscope.display.shader_mapping import ShaderComponent, ShaderDisplayMode, ShaderMapping, TexturePlaneKind
from arrayscope.display.model.frame import DisplayTilePayload


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


class FakeVisual:
    def __init__(self):
        self.visible = False
        self.levels = []
        self.geometry_calls = 0
        self.mapping_calls = 0
        self.mappings = []
        self.textures = None

    def set_levels(self, levels):
        levels = tuple(float(value) for value in levels)
        changed = not self.levels or self.levels[-1] != levels
        if changed:
            self.levels.append(levels)
        return changed

    def set_geometry(self, vertices, texcoords, modes):
        self.geometry_calls += 1

    def set_textures(self, scalar, color):
        changed = self.textures != (scalar, color)
        self.textures = (scalar, color)
        return changed

    def set_shader_mapping(self, mapping):
        self.mapping_calls += 1
        key = None if mapping is None else mapping.identity_key
        previous = None if not self.mappings else self.mappings[-1][0]
        changed = not self.mappings or previous != key
        if changed:
            self.mappings.append((key, mapping))
        return changed


class FakeSceneVisuals:
    @staticmethod
    def create_visual_node(_visual_type):
        return lambda parent=None: FakeVisual()


class FakeScene:
    visuals = FakeSceneVisuals()


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


def complex_payload(tile_number: int) -> DisplayTilePayload:
    data = np.array([[1 + 0j, 1j], [-1 + 0j, -1j]], dtype=np.complex64)
    histogram = np.abs(data).astype(np.float32)
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=data,
        histogram_data=histogram,
        source_id=("complex", tile_number),
        texture_data=data,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=data,
        semantic_histogram_data=histogram,
        shader_mapping=ShaderMapping(
            component=ShaderComponent.ABS,
            display_mode=ShaderDisplayMode.PHASE_COLOR,
        ),
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
    assert ("tile", 0, 10.0) not in pool.source_ids.values()
    assert ("tile", 2, 30.0) in pool.source_ids.values()


def test_atlas_reserve_includes_pending_visible_tiles():
    geometry = SimpleNamespace(
        montage=SimpleNamespace(indices=tuple(range(6))),
        montage_tile_states=(
            "loaded",
            "loading",
            "unloaded",
            "unloaded",
            "skipped",
            "unloaded",
        ),
    )

    assert _atlas_reserve_count(geometry, minimum=1) == 5


def test_complex_payload_quad_buffers_use_phase_color_shader_mode():
    montage = SimpleNamespace(indices=(0,), tile_width=2, tile_height=2, columns=1, rows=1, gap=0)
    payload = complex_payload(0)

    _vertices, _texcoords, modes = _quad_buffers(
        montage,
        {0: payload},
        {0: (0.0, 0.0, 1.0, 1.0)},
        rgb_already_windowed=False,
    )

    assert _payload_mode(payload, rgb_already_windowed=False) == 4
    np.testing.assert_array_equal(modes, np.full((6,), 4.0, dtype=np.float32))


def test_gpu_windowed_tile_shader_supports_complex_components():
    shader = GpuWindowedTileVisual._fragment_shader

    assert "uniform float u_component_mode" in shader
    assert "float complex_component" in shader
    assert "if (u_component_mode > 2.5)" in shader
    assert "float intensity = 1.0;" in shader


def test_gpu_windowed_tile_mapping_tracks_component_uniform_without_texture_identity():
    visual = object.__new__(GpuWindowedTileVisual)
    visual._shader_mapping_key = None
    visual._scale_mode = 0.0
    visual._symlog_constant = 0.0
    visual._component_mode = 0.0
    visual._lut_key = None
    visual._lut_texture = object()
    updates = []
    visual.update = lambda: updates.append("update")
    visual._set_lut_texture = lambda lut, key=None: False

    assert visual.set_shader_mapping(ShaderMapping(component=ShaderComponent.REAL)) is True
    assert visual._component_mode == 0.0
    assert visual.set_shader_mapping(ShaderMapping(component=ShaderComponent.REAL)) is False
    assert visual.set_shader_mapping(ShaderMapping(component=ShaderComponent.IMAG)) is True
    assert visual._component_mode == 1.0
    assert len(updates) == 2


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


def test_atlas_capacity_hint_grows_in_bounded_byte_chunks():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4096)

    requested = pool.requested_capacity(
        active_count=1,
        reserve_count=1000,
        storage_mode="complex",
        tile_shape=(512, 512),
    )

    assert requested == 16
    assert requested < 1000


def test_new_atlas_page_inherits_current_levels_without_reuploading_old_geometry():
    layer = GpuMontageLayer(
        scene=FakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=4),
    )
    montage = SimpleNamespace(
        indices=tuple(range(5)),
        tile_width=2,
        tile_height=2,
        columns=5,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",) * 5)
    delta = SimpleNamespace(planned_tiles=tuple(range(5)), near_tiles=(), near_tile_source_ids={})

    layer.update(
        payloads={index: payload(index, float(index)) for index in range(4)},
        geometry=geometry,
        levels=(10.0, 20.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=delta,
    )
    first_visual = layer._visuals_by_page[0]
    assert first_visual.levels[-1] == (10.0, 20.0)
    assert first_visual.geometry_calls == 1

    layer.update(
        payloads={index: payload(index, float(index)) for index in range(5)},
        geometry=geometry,
        levels=(10.0, 20.0),
        dirty_tiles=(4,),
        rgb_already_windowed=False,
        tile_delta=delta,
    )

    assert len(layer._visuals_by_page) == 2
    assert layer._visuals_by_page[1].levels[-1] == (10.0, 20.0)
    assert first_visual.geometry_calls == 1
    assert layer._visuals_by_page[1].geometry_calls == 1


def test_mapping_only_update_is_uniform_across_pages_without_texture_or_vertex_uploads():
    layer = GpuMontageLayer(
        scene=FakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=4),
    )
    montage = SimpleNamespace(
        indices=tuple(range(5)),
        tile_width=2,
        tile_height=2,
        columns=5,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",) * 5)
    delta = SimpleNamespace(planned_tiles=tuple(range(5)), near_tiles=(), near_tile_source_ids={})
    first_mapping = ShaderMapping(component=ShaderComponent.REAL)
    second_mapping = ShaderMapping(component=ShaderComponent.IMAG)

    first = layer.update(
        payloads={index: payload(index, float(index)) for index in range(5)},
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        shader_mapping=first_mapping,
        tile_delta=delta,
    )
    geometry_calls = [visual.geometry_calls for visual in layer._visuals_by_page]
    texture_updates = sum(
        len(texture.updates)
        for page in layer._pool.pages
        for texture in (page.scalar_texture, page.color_texture)
    )

    second = layer.set_presentation_uniforms(
        levels=(0.0, 4.0),
        shader_mapping=second_mapping,
    )

    assert first.page_count == 2
    assert second.texture_uploads == 0
    assert second.vertex_uploads == 0
    assert second.shader_uniform_updates == 2
    assert [visual.geometry_calls for visual in layer._visuals_by_page] == geometry_calls
    assert sum(
        len(texture.updates)
        for page in layer._pool.pages
        for texture in (page.scalar_texture, page.color_texture)
    ) == texture_updates
    assert all(visual.mappings[-1][1] is second_mapping for visual in layer._visuals_by_page)


def test_none_dirty_set_forces_refresh_even_when_sources_match():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    values = {0: payload(0, 1.0), 1: payload(1, 2.0)}
    pool.update_payloads(
        values,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
    )

    _uvs, refreshed = pool.update_payloads(
        values,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
    )

    assert refreshed.items_updated == 2
    assert refreshed.items_skipped == 0
    assert refreshed.texture_uploads == 2


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


def test_atlas_reuses_resident_source_when_tile_number_changes():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    sources = {index: ("source", index) for index in range(4)}
    pool.update_payloads(
        {index: payload(index, float(index), source_id=sources[index]) for index in range(4)},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )

    _uvs, reused = pool.update_payloads(
        {
            0: payload(0, 2.0, source_id=sources[2]),
            1: payload(1, 3.0, source_id=sources[3]),
        },
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=4,
    )

    assert reused.items_updated == 0
    assert reused.items_skipped == 2
    assert reused.texture_uploads == 0
    assert reused.resident_items == 4
    page_index, slot = pool.resident_slots[_resident_key(payload(0, 2.0, source_id=sources[2]))]
    assert pool.slots[0] == page_index * 1_000_000 + slot


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
    rebuilt = pool.ensure_layout(tile_shape=(2, 2), count=5, storage_mode="scalar")

    assert rebuilt is True
    assert len(pool.pages) == 2
    assert pool.capacity == 5


def test_atlas_rejects_active_set_that_exceeds_byte_budget():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16, budget_bytes=2 * 2 * 4)
    with pytest.raises(AtlasCapacityError, match="budget"):
        pool.update_payloads(
            {0: payload(0, 0.0), 1: payload(1, 1.0)},
            tile_shape=(2, 2),
            dirty_tiles=None,
            rgb_already_windowed=False,
            reserve_count=2,
        )


def test_atlas_eviction_prefers_far_inactive_tiles_before_near_tiles():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4)
    values = {index: payload(index, float(index)) for index in range(4)}
    pool.update_payloads(values, tile_shape=(2, 2), dirty_tiles=None, rgb_already_windowed=False, reserve_count=4)
    pool.update_payloads(
        {2: values[2], 3: values[3]},
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=4,
        near_tiles=(0,),
        near_tile_source_ids={0: values[0].source_id},
    )
    pool.update_payloads(
        {2: values[2], 3: values[3], 4: payload(4, 4.0)},
        tile_shape=(2, 2),
        dirty_tiles=(4,),
        rgb_already_windowed=False,
        reserve_count=4,
        near_tiles=(0,),
        near_tile_source_ids={0: values[0].source_id},
    )

    assert ("tile", 0, 0.0) in pool.source_ids.values()
    assert ("tile", 1, 1.0) not in pool.source_ids.values()


def test_near_base_source_identity_protects_wrapped_resident_payload():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4)
    base_zero = ("base", 0)
    wrapped_zero = (base_zero, "texture_kind", "scalar_r32f", "shader", None, "lod", 1, 0, 0)
    values = {
        0: payload(0, 0.0, source_id=wrapped_zero),
        1: payload(1, 1.0),
        2: payload(2, 2.0),
    }
    pool.update_payloads(
        values,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=3,
    )

    pool.update_payloads(
        {2: values[2], 3: payload(3, 3.0)},
        tile_shape=(2, 2),
        dirty_tiles=(3,),
        rgb_already_windowed=False,
        reserve_count=3,
        near_tiles=(0,),
        near_tile_source_ids={0: base_zero},
    )

    assert wrapped_zero in pool.source_ids.values()
    assert ("tile", 1, 1.0) not in pool.source_ids.values()


def test_atlas_warms_loaded_near_payload_without_changing_active_slots():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    values = {index: payload(index, float(index)) for index in range(3)}
    pool.update_payloads(
        {0: values[0]},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=3,
    )
    active_slots = dict(pool.slots)

    warmed = pool.warm_payloads(
        {1: values[1], 2: values[2]},
        tile_shape=(2, 2),
        rgb_already_windowed=False,
        near_tile_source_ids={index: values[index].source_id for index in range(3)},
    )

    assert warmed.items_updated == 2
    assert warmed.texture_uploads == 2
    assert warmed.resident_items == 3
    assert pool.slots == active_slots


def test_payload_texture_conversion_preserves_scalar_dynamic_range():
    tile = payload(0, 4096.0)
    scalar, color = _payload_textures(tile, tile_shape=(2, 2), rgb_already_windowed=False)
    np.testing.assert_allclose(scalar, 4096.0)
    assert scalar.dtype == np.float32
    assert color.dtype == np.uint8
    assert not np.any(color)


def test_complex_atlas_samples_raw_values_without_linear_filtering():
    page = TextureAtlasPage(
        FakeGloo,
        tile_shape=(2, 2),
        capacity=2,
        storage_mode="complex",
        max_texture_size=16,
    )

    assert page.scalar_texture.kwargs["interpolation"] == "nearest"


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


def test_device_limit_query_falls_back_when_gl_is_unavailable():
    limits = query_gpu_device_limits(SimpleNamespace())

    assert limits.max_texture_size == 4096
    assert limits.source == "fallback"
    assert limits.warnings


def test_speculative_payload_batches_are_bounded_by_items_and_bytes():
    values = {index: payload(index, float(index)) for index in range(6)}

    batch, remaining = take_payload_batch(values, max_items=3, max_bytes=33)

    assert tuple(batch) == (0, 1)
    assert tuple(remaining) == (2, 3, 4, 5)


def test_quad_generation_iterates_active_payloads_not_the_complete_plan():
    class Indices:
        def __len__(self):
            return 100_000

        def __iter__(self):
            raise AssertionError("complete montage population must not be scanned")

    montage = SimpleNamespace(indices=Indices(), tile_width=2, tile_height=2, columns=100, rows=1000, gap=0)
    vertices, _texcoords, _modes = _quad_buffers(
        montage,
        {12_345: payload(12_345, 1.0)},
        {12_345: (0.0, 0.0, 1.0, 1.0)},
        rgb_already_windowed=False,
    )

    assert vertices.shape == (6, 2)
