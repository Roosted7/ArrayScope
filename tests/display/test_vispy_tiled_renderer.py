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
    PayloadBatchQueue,
    _atlas_reserve_count,
    _complex_rg_texture,
    _fit_color,
    _fit_scalar,
    _payload_mode,
    _payload_textures,
    _quad_buffers,
    _resident_key,
    query_gpu_device_limits,
    take_payload_batch,
)
from arrayscope.display.tile_layout import TileLayoutRegion
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
        self.vertices = None
        self.texcoords = None
        self.modes = None
        self.mapping_calls = 0
        self.mappings = []
        self.textures = None
        self.update_calls = 0

    def set_levels(self, levels):
        levels = tuple(float(value) for value in levels)
        changed = not self.levels or self.levels[-1] != levels
        if changed:
            self.levels.append(levels)
        return changed

    def set_geometry(self, vertices, texcoords, modes):
        self.geometry_calls += 1
        self.vertices = np.asarray(vertices)
        self.texcoords = np.asarray(texcoords)
        self.modes = np.asarray(modes)

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

    def update(self):
        self.update_calls += 1


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
    assert second.storage_evictions == 1
    assert ("tile", 0, 10.0) not in pool.source_ids.values()
    assert ("tile", 2, 30.0) in pool.source_ids.values()


def test_atlas_page_uses_free_slot_stack_without_owner_scan(monkeypatch):
    from arrayscope.display.backends.vispy.tiles import TextureAtlasPage

    class OwnerList(list):
        def index(self, _value, *_args):
            raise AssertionError("free-slot allocation must not scan slot owners")

    page = TextureAtlasPage(FakeGloo, tile_shape=(2, 2), capacity=2, storage_mode="scalar", max_texture_size=8)
    page.slot_owners = OwnerList(page.slot_owners)

    assert page.take_free_slot(("owner", 1)) == 0
    assert page.take_free_slot(("owner", 2)) == 1
    assert page.take_free_slot(("owner", 3)) is None


def test_dirty_resident_payload_reuses_uploaded_source_when_source_id_matches():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    payloads = {0: payload(0, 1.0)}

    pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
    )
    clean_texture_updates = len(pool.scalar_texture.updates)
    _uvs, clean = pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=1,
    )

    assert clean.items_updated == 0
    assert len(pool.scalar_texture.updates) == clean_texture_updates

    _uvs, dirty = pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        reserve_count=1,
    )

    assert dirty.items_updated == 0
    assert dirty.items_skipped == 1
    assert dirty.texture_uploads == 0
    assert len(pool.scalar_texture.updates) == clean_texture_updates


def test_allocated_slot_without_uploaded_source_is_not_treated_clean():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    payloads = {0: payload(0, 1.0)}
    pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
    )
    resident_key = _resident_key(payloads[0])
    pool.source_ids.pop(resident_key)
    upload_count = len(pool.scalar_texture.updates)

    _uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=(),
        rgb_already_windowed=False,
        reserve_count=1,
    )

    assert stats.items_updated == 1
    assert len(pool.scalar_texture.updates) == upload_count + 1
    assert pool.source_ids[resident_key] == payloads[0].source_id


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
    layout = {0: TileLayoutRegion(tile_number=0, source_index=0, x=0, y=0, width=2, height=2)}
    payload = complex_payload(0)

    _vertices, _texcoords, modes = _quad_buffers(
        layout,
        {0: payload},
        {0: (0.0, 0.0, 1.0, 1.0)},
        rgb_already_windowed=False,
    )

    assert _payload_mode(payload, rgb_already_windowed=False) == 4
    np.testing.assert_array_equal(modes, np.full((6,), 4.0, dtype=np.float32))


def test_complex_payload_upload_uses_defensive_copy_into_atlas():
    payloads = {0: complex_payload(0)}
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4)

    pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
    )

    uploaded, _offset, copy = pool.scalar_texture.updates[0]
    assert copy
    np.testing.assert_allclose(uploaded, _complex_rg_texture(payloads[0].texture_data))


def test_invalid_complex_payload_is_not_made_visible_from_slot_zero():
    bad_rgb = np.full((2, 2, 3), 255, dtype=np.uint8)
    bad = DisplayTilePayload(
        0,
        0,
        bad_rgb,
        np.ones((2, 2), dtype=np.float32),
        ("bad", 0),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=bad_rgb,
    )
    good = complex_payload(1)
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=4)

    uvs, stats = pool.update_payloads(
        {0: bad, 1: good},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
    )

    assert 0 not in uvs
    assert 1 in uvs
    assert stats.visible_items == 1
    assert stats.presented_tiles == (1,)
    assert stats.items_skipped == 1


def test_gpu_windowed_tile_shader_supports_complex_components():
    shader = GpuWindowedTileVisual._fragment_shader

    assert "uniform float u_component_mode" in shader
    assert "float complex_component" in shader
    assert "if (u_component_mode > 2.5)" in shader
    assert "float intensity = 1.0;" in shader
    assert "scalar = map_scale(length(z));" in shader


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


def test_gpu_windowed_tile_mapping_tracks_lut_without_texture_identity():
    visual = object.__new__(GpuWindowedTileVisual)
    visual._shader_mapping_key = None
    visual._scale_mode = 0.0
    visual._symlog_constant = 0.0
    visual._component_mode = 0.0
    visual._lut_key = None
    visual._lut_texture = object()
    visual._lut_default_phase = False
    updates = []
    uploaded_luts = []
    visual.update = lambda: updates.append("update")
    visual._set_lut_texture = lambda lut, key=None: uploaded_luts.append(np.array(lut, copy=True)) or True

    first = ShaderMapping(lut_data=np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8))
    second = ShaderMapping(lut_data=np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8))

    assert visual.set_shader_mapping(first) is True
    assert visual.set_shader_mapping(first) is False
    assert visual.set_shader_mapping(second) is True
    assert len(updates) == 2
    assert len(uploaded_luts) == 2
    np.testing.assert_array_equal(uploaded_luts[-1], second.lut_data)


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


def test_atlas_reuses_matching_source_identity_even_when_dirty_is_unknown():
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

    assert refreshed.items_updated == 0
    assert refreshed.items_skipped == 2
    assert refreshed.texture_uploads == 0

    changed = {0: payload(0, 10.0, source_id=("changed", 0)), 1: values[1]}
    _uvs, uploaded = pool.update_payloads(
        changed,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
    )

    assert uploaded.items_updated == 1
    assert uploaded.items_skipped == 1
    assert uploaded.texture_uploads == 1


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


def test_gpu_layer_empty_active_set_hides_visuals_but_keeps_residency():
    layer = GpuMontageLayer(
        scene=FakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=8),
    )
    montage = SimpleNamespace(
        indices=(0, 1),
        tile_width=2,
        tile_height=2,
        columns=2,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded", "loaded"))
    delta = SimpleNamespace(planned_tiles=(0, 1), near_tiles=(0, 1), near_tile_source_ids={})

    first = layer.update(
        payloads={0: payload(0, 1.0), 1: payload(1, 2.0)},
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=delta,
    )
    empty = layer.update(
        payloads={},
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=(),
        rgb_already_windowed=False,
        tile_delta=delta,
    )

    assert first.active_pages == 1
    assert empty.visible_items == 0
    assert empty.active_pages == 0
    assert layer._pool.resident_count == 2
    assert not any(visual.visible for visual in layer._visuals_by_page)


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


def test_gpu_layer_updates_geometry_for_clean_resident_complex_retarget():
    layer = GpuMontageLayer(
        scene=FakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=8),
    )
    montage = SimpleNamespace(
        indices=(0, 1, 2, 3),
        tile_width=2,
        tile_height=2,
        columns=2,
        rows=2,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",) * 4)
    initial_payloads = {index: complex_payload(index) for index in range(4)}
    first_delta = SimpleNamespace(
        upserts=initial_payloads,
        removals=(),
        active_tiles=(0, 1, 2, 3),
        planned_tiles=(0, 1, 2, 3),
        near_tiles=(0, 1, 2, 3),
        near_tile_source_ids={index: value.source_id for index, value in initial_payloads.items()},
        force_refresh=True,
    )

    first = layer.update(
        payloads=initial_payloads,
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=first_delta,
    )
    visual = layer._visuals_by_page[0]
    first_geometry_calls = int(visual.geometry_calls)
    source_two_slot = layer._pool.tile_slots[2]
    source_two = initial_payloads[2]
    retargeted = DisplayTilePayload(
        0,
        source_two.source_index,
        source_two.image,
        source_two.histogram_data,
        source_two.source_id,
        texture_data=source_two.texture_data,
        texture_kind=source_two.texture_kind,
        semantic_data=source_two.semantic_data,
        semantic_histogram_data=source_two.semantic_histogram_data,
        source_shape=source_two.source_shape,
        lod=source_two.lod,
        shader_mapping=source_two.shader_mapping,
    )
    clean_delta = SimpleNamespace(
        upserts={},
        removals=(),
        active_tiles=(0,),
        planned_tiles=(0, 1, 2, 3),
        near_tiles=(0, 1, 2, 3),
        near_tile_source_ids={0: retargeted.source_id},
        force_refresh=False,
    )

    shifted = layer.update(
        payloads={0: retargeted},
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=(),
        rgb_already_windowed=False,
        tile_delta=clean_delta,
    )

    assert first.texture_uploads == 4
    assert shifted.texture_uploads == 0
    assert shifted.items_updated == 0
    assert shifted.vertex_uploads == 1
    assert shifted.presented_tiles == (0,)
    assert layer._pool.tile_slots[0] == source_two_slot
    assert layer._pool.tile_resident_keys[0] == _resident_key(retargeted)
    assert visual.geometry_calls == first_geometry_calls + 1


def test_clean_complex_active_tile_reuploads_when_uploaded_source_proof_is_missing():
    layer = GpuMontageLayer(
        scene=FakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=8),
    )
    montage = SimpleNamespace(
        indices=(0,),
        tile_width=2,
        tile_height=2,
        columns=1,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",))
    payloads = {0: complex_payload(0)}
    first_delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
        near_tile_source_ids={0: payloads[0].source_id},
        force_refresh=True,
    )
    layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=first_delta,
    )
    resident_key = _resident_key(payloads[0])
    layer._pool.source_ids.pop(resident_key)
    clean_delta = SimpleNamespace(
        upserts={},
        removals=(),
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
        near_tile_source_ids={0: payloads[0].source_id},
        force_refresh=False,
    )

    recovered = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 2.0),
        dirty_tiles=(),
        rgb_already_windowed=False,
        tile_delta=clean_delta,
    )

    assert recovered.presented_tiles == (0,)
    assert recovered.items_updated == 1
    assert recovered.texture_uploads == 1
    assert layer._pool.source_ids[resident_key] == payloads[0].source_id


def test_delta_uploads_only_admitted_upserts_when_dirty_list_is_broad():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    old_zero = payload(0, 0.0)
    new_one = payload(1, 1.0)
    waiting_two = payload(2, 2.0)
    pool.update_payloads(
        {0: old_zero},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )
    initial_updates = len(pool.pages[0].scalar_texture.updates)
    delta = SimpleNamespace(
        upserts={1: new_one},
        removals=(),
        active_tiles=(0, 1, 2),
        planned_tiles=(0, 1, 2),
        near_tiles=(0, 1, 2),
        near_tile_source_ids={0: old_zero.source_id, 1: new_one.source_id},
        force_refresh=False,
    )

    _uvs, stats = pool.update_payloads(
        {0: old_zero, 1: new_one, 2: waiting_two},
        tile_shape=(2, 2),
        dirty_tiles=(0, 1, 2),
        rgb_already_windowed=False,
        reserve_count=4,
        tile_delta=delta,
    )

    assert stats.items_updated == 2
    assert stats.texture_uploads == 2
    assert stats.presented_tiles == (0, 1, 2)
    assert len(pool.pages[0].scalar_texture.updates) == initial_updates + 2
    assert 2 in pool.tile_slots


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
    # A gloo with an empty ``gl`` keeps the query away from the process-global
    # VisPy GL module, which can hold a real live context in a full test run.
    limits = query_gpu_device_limits(SimpleNamespace(gl=SimpleNamespace()))

    assert limits.max_texture_size == 4096
    assert limits.source == "fallback"
    assert limits.warnings


def test_speculative_payload_batches_are_bounded_by_items_and_bytes():
    values = {index: payload(index, float(index)) for index in range(6)}

    batch, remaining = take_payload_batch(values, max_items=3, max_bytes=33)

    assert tuple(batch) == (0, 1)
    assert tuple(remaining) == (2, 3, 4, 5)


def test_speculative_payload_queue_removes_batches_without_rebuilding_remaining_mapping():
    values = {index: payload(index, float(index)) for index in range(6)}
    queue = PayloadBatchQueue(values)

    first = queue.take(max_items=3, max_bytes=33)
    second = queue.take(max_items=3, max_bytes=33)

    assert tuple(first) == (0, 1)
    assert tuple(second) == (2, 3)
    assert tuple(queue.remaining_payloads()) == (4, 5)


def test_quad_generation_iterates_active_payloads_not_the_complete_plan():
    layout = {12_345: TileLayoutRegion(tile_number=12_345, source_index=12_345, x=90, y=246, width=2, height=2)}
    vertices, _texcoords, _modes = _quad_buffers(
        layout,
        {12_345: payload(12_345, 1.0)},
        {12_345: (0.0, 0.0, 1.0, 1.0)},
        rgb_already_windowed=False,
    )

    assert vertices.shape == (6, 2)


def _lod_payload(tile_number: int, value: float, *, level: int, source_shape=(4, 4)) -> DisplayTilePayload:
    from arrayscope.display.lod import LodInfo

    factor = 2 ** int(level)
    texture_shape = (
        max(1, int(source_shape[0]) // factor),
        max(1, int(source_shape[1]) // factor),
    )
    image = np.full(tuple(source_shape), value, dtype=np.float32)
    texture = image if level == 0 else np.full(texture_shape, value, dtype=np.float32)
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=image,
        histogram_data=None,
        source_id=("tile", tile_number, value, "lod", int(level)),
        texture_data=texture,
        lod=LodInfo(
            level=int(level),
            factor=factor,
            source_shape=tuple(source_shape),
            texture_shape=texture.shape[:2],
        ),
    )


def test_atlas_classes_pages_by_texture_shape_for_mixed_levels():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)

    payloads = {
        0: _lod_payload(0, 1.0, level=0),
        1: _lod_payload(1, 2.0, level=0),
        2: _lod_payload(2, 3.0, level=1),
        3: _lod_payload(3, 4.0, level=1),
    }
    uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )

    assert stats.items_updated == 4
    assert set(stats.presented_tiles) == {0, 1, 2, 3}
    page_shapes = {page.tile_shape for page in pool.pages}
    assert (4, 4) in page_shapes and (2, 2) in page_shapes
    for tile_number, payload in payloads.items():
        page_index, _slot = pool.tile_slots[int(tile_number)]
        texture = np.asarray(payload.texture_data)
        assert pool.pages[page_index].tile_shape == tuple(texture.shape[:2]), (
            "a payload must only occupy a slot of its own texture shape class"
        )


def test_reduced_payload_never_lands_in_native_shaped_slot():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    native = {0: _lod_payload(0, 1.0, level=0)}
    pool.update_payloads(
        native,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=8,
    )
    assert all(page.tile_shape == (4, 4) for page in pool.pages)

    reduced = {0: _lod_payload(0, 1.0, level=1)}
    _uvs, stats = pool.update_payloads(
        reduced,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=8,
    )

    assert stats.presented_tiles == (0,)
    page_index, _slot = pool.tile_slots[0]
    assert pool.pages[page_index].tile_shape == (2, 2)
    # The native level for the same tile remains resident in its own class.
    assert ("tile", 0, 1.0, "lod", 0) in pool.source_ids.values()
    assert ("tile", 0, 1.0, "lod", 1) in pool.source_ids.values()


def test_level_flip_back_to_native_does_not_reupload_source_pixels():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    native = {0: _lod_payload(0, 1.0, level=0)}
    reduced = {0: _lod_payload(0, 1.0, level=1)}

    _uvs, first = pool.update_payloads(
        native, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False, reserve_count=4
    )
    _uvs, second = pool.update_payloads(
        reduced, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False, reserve_count=4
    )
    _uvs, third = pool.update_payloads(
        native, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False, reserve_count=4
    )

    assert first.texture_uploads == 1
    assert second.texture_uploads == 1
    assert third.texture_uploads == 0, "an already-resident native level must not re-upload"
    assert third.items_skipped >= 1


def test_mixed_level_commit_keeps_base_class_when_actives_are_all_reduced():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    pool.update_payloads(
        {0: _lod_payload(0, 1.0, level=0)},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )
    base_pages = sum(1 for page in pool.pages if page.tile_shape == (4, 4))
    assert base_pages >= 1

    # The layer derives the base shape from lod.source_shape, so an
    # all-reduced active set must not rebuild/clear the native class.
    from arrayscope.display.backends.vispy.tiles import _atlas_base_tile_shape_for_payloads

    reduced_only = {0: _lod_payload(0, 1.0, level=1)}
    base_shape = _atlas_base_tile_shape_for_payloads(reduced_only, fallback=(2, 2))
    assert base_shape == (4, 4)

    _uvs, stats = pool.update_payloads(
        reduced_only,
        tile_shape=base_shape,
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )

    assert stats.storage_rebuilds == 0
    assert ("tile", 0, 1.0, "lod", 0) in pool.source_ids.values()


def test_reduced_class_budget_exhaustion_retains_previous_mapping():
    tile_bytes = 4 * 4 * 4  # scalar float32 slots
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8, budget_bytes=tile_bytes)
    native = {0: _lod_payload(0, 1.0, level=0)}
    pool.update_payloads(
        native,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
        budget_bytes=tile_bytes,
    )
    native_mapping = dict(pool.tile_slots)
    assert native_mapping

    reduced = {0: _lod_payload(0, 1.0, level=1)}
    _uvs, stats = pool.update_payloads(
        reduced,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
        budget_bytes=tile_bytes,
    )

    # No budget headroom for a second shape class: the reduced payload is
    # skipped and the native mapping stays presented rather than clearing.
    assert stats.presented_tiles == ()
    assert dict(pool.tile_slots) == native_mapping
    assert ("tile", 0, 1.0, "lod", 0) in pool.source_ids.values()


def test_cold_fill_at_reduced_level_performs_zero_native_uploads():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    reduced = {index: _lod_payload(index, float(index + 1), level=1) for index in range(3)}

    _uvs, stats = pool.update_payloads(
        reduced,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=8,
    )

    assert set(stats.presented_tiles) == {0, 1, 2}
    assert stats.texture_uploads == 3
    assert stats.texture_upload_bytes == 3 * (2 * 2 * 4)
    # The native class is never uploaded to, and it is not pre-allocated for
    # the whole montage when the active set needs no native slots.
    native_pages = [page for page in pool.pages if page.tile_shape == (4, 4)]
    assert all(owner is None for page in native_pages for owner in page.slot_owners)
    assert all(not page.scalar_texture.updates for page in native_pages)
    assert sum(page.capacity for page in native_pages) <= 1


def test_superseded_native_slots_are_reclaimed_under_reduced_class_pressure():
    native_slot = 4 * 4 * 4
    reduced_slot = 2 * 2 * 4
    budget = native_slot + reduced_slot
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16, budget_bytes=budget)

    pool.update_payloads(
        {0: _lod_payload(0, 1.0, level=0)},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
        budget_bytes=budget,
    )
    _uvs, flipped = pool.update_payloads(
        {0: _lod_payload(0, 1.0, level=1)},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
        budget_bytes=budget,
    )
    assert flipped.presented_tiles == (0,)
    # The reduced payload is acknowledged and presented: the native slot for
    # the same tile is now superseded but still allocated.
    assert ("tile", 0, 1.0, "lod", 0) in pool.source_ids.values()
    assert pool.superseded_keys

    _uvs, stats = pool.update_payloads(
        {0: _lod_payload(0, 1.0, level=1), 1: _lod_payload(1, 2.0, level=1)},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
        budget_bytes=budget,
    )

    # Instead of budget-limiting the reduced class, the superseded native
    # slot was freed and its emptied page dropped to recover the bytes.
    assert set(stats.presented_tiles) == {0, 1}
    assert pool.superseded_reclaimed_count == 1
    assert pool.pages_dropped_count == 1
    assert ("tile", 0, 1.0, "lod", 0) not in pool.source_ids.values()
    assert pool.tile_slots[0] != pool.tile_slots[1]


def test_presented_native_slot_is_never_reclaimed_for_reduced_capacity():
    native_slot = 4 * 4 * 4
    reduced_slot = 2 * 2 * 4
    budget = 2 * native_slot + reduced_slot
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16, budget_bytes=budget)

    shared_native = _lod_payload(0, 1.0, level=0)
    other_native = DisplayTilePayload(
        tile_number=1,
        source_index=1,
        image=np.asarray(shared_native.image),
        histogram_data=None,
        source_id=shared_native.source_id,
        texture_data=shared_native.texture_data,
        lod=shared_native.lod,
    )
    pool.update_payloads(
        {0: shared_native, 1: other_native},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
        budget_bytes=budget,
    )

    # Tile 0 flips to its reduced level while tile 1 keeps presenting the
    # shared native slot.
    _uvs, stats = pool.update_payloads(
        {0: _lod_payload(0, 1.0, level=1), 1: other_native},
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
        budget_bytes=budget,
    )
    assert set(stats.presented_tiles) == {0, 1}
    native_key = pool.tile_resident_keys[1]
    assert pool.resident_tiles[native_key] == {1}

    # Reduced-class pressure must not free the slot tile 1 still presents
    # (ADR 0041 gate 5: presented stays usable until its replacement lands).
    assert pool._ensure_class_capacity((2, 2), 5) == 1
    assert pool.superseded_reclaimed_count == 0
    assert native_key in pool.resident_slots
    assert pool.tile_resident_keys[1] == native_key
    assert pool.source_ids[native_key] == other_native.source_id


def test_eviction_prefers_superseded_slots_over_lru_and_near_keys():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    pool.ensure_layout(tile_shape=(4, 4), count=2, storage_mode="scalar")

    assert pool._slot_for("superseded", active_keys=set(), near_keys=set())[2]
    assert pool._slot_for("plain-old", active_keys=set(), near_keys=set())[2]
    pool.source_ids["superseded"] = ("source", "superseded")
    pool.source_ids["plain-old"] = ("source", "plain-old")
    # LRU alone would evict "plain-old"; the superseded key must go first
    # even though it was touched more recently and is protected as near.
    pool._touch("plain-old")
    pool._touch("superseded")
    pool.superseded_keys.add("superseded")

    _page, _slot, newly = pool._slot_for(
        "incoming", active_keys={"incoming"}, near_keys={"superseded", "plain-old"}
    )

    assert newly
    assert "superseded" not in pool.resident_slots
    assert "plain-old" in pool.resident_slots
    assert "superseded" not in pool.superseded_keys
    assert pool.eviction_count == 1


def _cycle_update(pool, payloads, budget=None):
    return pool.update_payloads(
        payloads,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=2,
        budget_bytes=budget,
    )


def test_zoom_cycle_over_resident_classes_is_zero_upload_after_first_materialization():
    """ADR 0050 gate 6: level flips between resident classes are identity swaps."""

    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    native = {0: _lod_payload(0, 1.0, level=0), 1: _lod_payload(1, 2.0, level=0)}
    reduced = {0: _lod_payload(0, 1.0, level=1), 1: _lod_payload(1, 2.0, level=1)}

    _uvs, first = _cycle_update(pool, native)
    assert first.texture_uploads == 2

    # Zoom out: reduced class materializes once, counted as swaps-with-upload.
    _uvs, out = _cycle_update(pool, reduced)
    assert out.texture_uploads == 2
    assert out.lod_level_swaps_with_upload == 2
    assert out.lod_level_swaps_zero_upload == 0

    # Zoom back in and back out: both classes resident, zero uploads, pure
    # identity swaps, and no superseded slot was reclaimed merely because a
    # swap happened.
    _uvs, back_in = _cycle_update(pool, native)
    assert back_in.texture_uploads == 0
    assert back_in.lod_level_swaps_zero_upload == 2
    assert back_in.lod_level_swaps_with_upload == 0
    assert back_in.superseded_reclaimed_under_pressure == 0

    _uvs, back_out = _cycle_update(pool, reduced)
    assert back_out.texture_uploads == 0
    assert back_out.lod_level_swaps_zero_upload == 2
    assert back_out.lod_level_swaps_with_upload == 0
    assert back_out.superseded_reclaimed_under_pressure == 0

    assert pool.lod_level_swaps_zero_upload == 4
    assert pool.lod_level_swaps_with_upload == 2
    assert pool.superseded_reclaimed_count == 0


def test_eviction_reclaims_active_tiles_adjacent_level_only_as_last_resort():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    pool.ensure_layout(tile_shape=(4, 4), count=3, storage_mode="scalar")

    # Slot 1: the superseded native level of an ACTIVE tile (it presents its
    # reduced level right now).  Slot 2: a warm presented class of a
    # non-active tile.  Slot 3: a superseded class of a non-active tile.
    assert pool._slot_for("active-adjacent", active_keys=set(), near_keys=set())[2]
    assert pool._slot_for("warm-presented", active_keys=set(), near_keys=set())[2]
    assert pool._slot_for("stale-superseded", active_keys=set(), near_keys=set())[2]
    pool.source_ids["active-adjacent"] = ("tile", 0, 1.0, "lod", 0)
    pool.source_ids["warm-presented"] = ("tile", 7, 7.0, "lod", 0)
    pool.source_ids["stale-superseded"] = ("tile", 9, 9.0, "lod", 0)
    pool.superseded_keys.update({"active-adjacent", "stale-superseded"})
    pool.active_base_source_ids = {("tile", 0, 1.0)}
    # LRU says the adjacent level is oldest; preference order must still
    # protect it behind the other candidates.
    pool._touch("stale-superseded")
    pool._touch("warm-presented")

    pool._slot_for("incoming-1", active_keys={"incoming-1"}, near_keys=set())
    assert "stale-superseded" not in pool.resident_slots
    assert "active-adjacent" in pool.resident_slots
    assert "warm-presented" in pool.resident_slots

    pool._slot_for("incoming-2", active_keys={"incoming-1", "incoming-2"}, near_keys=set())
    assert "warm-presented" not in pool.resident_slots
    assert "active-adjacent" in pool.resident_slots, (
        "the retained adjacent level of an active tile goes last"
    )

    pool._slot_for("incoming-3", active_keys={"incoming-1", "incoming-2", "incoming-3"}, near_keys=set())
    assert "active-adjacent" not in pool.resident_slots
    # Both superseded victims counted as pressure reclaims.
    assert pool.superseded_reclaimed_count == 2


def test_active_tiles_only_resident_presented_class_is_never_evicted():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    pool.ensure_layout(tile_shape=(4, 4), count=1, storage_mode="scalar")
    payload = _lod_payload(0, 1.0, level=0)
    key = _resident_key(payload)
    assert pool._slot_for(key, active_keys={key}, near_keys=set())[2]
    pool._set_tile_mapping(0, key, 0, 0, (0.0, 0.0, 1.0, 1.0))

    with pytest.raises(AtlasCapacityError):
        pool._slot_for("incoming", active_keys={key, "incoming"}, near_keys=set())
    assert pool.tile_resident_keys[0] == key
