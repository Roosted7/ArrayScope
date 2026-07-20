from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.display.backends.vispy.tiles import (
    AtlasCapacityError,
    GpuDeviceLimits,
    GpuMontageLayer,
    GpuWindowedTileVisual,
    PayloadBatchQueue,
    TextureAtlasPage,
    TextureAtlasPool,
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
from arrayscope.display.lod import LodInfo
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    PageBackedPresentation,
    TilePresentationDelta,
)
from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    TexturePlaneKind,
)
from arrayscope.display.tile_layout import TileLayoutRegion
from tests.display.vispy_test_utils import (
    FakeGloo,
    FakeScene,
    color_payload,
    complex_payload,
    payload,
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

    page = TextureAtlasPage(
        FakeGloo, tile_shape=(2, 2), capacity=2, storage_mode="scalar", max_texture_size=8
    )
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


def test_active_vispy_tile_owns_successor_commit_slot_at_full_capacity():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=2)
    predecessor = payload(0, 1.0)
    successor = payload(0, 2.0)

    pool.update_payloads(
        {0: predecessor},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
    )

    assert pool.payload_resident(successor) is False
    assert pool.payload_commit_slot_owned(successor) is True


def test_atlas_consumes_ordered_upserts_before_active_grid_order():
    """Backend storage mechanics must not replace presentation priority."""

    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    initial = {index: payload(index, float(index)) for index in range(4)}
    pool.update_payloads(
        initial,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
    )
    update_start = len(pool.scalar_texture.updates)
    next_two = {
        3: payload(3, 40.0),
        2: payload(2, 30.0),
    }
    current = {**initial, 2: next_two[2], 3: next_two[3]}
    delta = SimpleNamespace(
        upserts=next_two,
        active_tiles=(0, 1, 2, 3),
        removals=(),
        target_identities={},
        near_tile_source_ids={},
    )

    _uvs, stats = pool.update_payloads(
        current,
        tile_shape=(2, 2),
        dirty_tiles=(3, 2),
        rgb_already_windowed=False,
        reserve_count=4,
        tile_delta=delta,
    )

    uploaded_values = [
        float(data[0, 0]) for data, _offset, _copy in pool.scalar_texture.updates[update_start:]
    ]
    assert uploaded_values == [40.0, 30.0]
    assert stats.committed_upserts == (3, 2)


def test_bounded_active_commit_retains_existing_presented_mappings():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    payloads = {index: payload(index, float(index)) for index in range(4)}
    delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=(0, 1, 2, 3),
        planned_tiles=(0, 1, 2, 3),
        near_tiles=(0, 1, 2, 3),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
    )
    pool.update_payloads(
        payloads,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
        tile_delta=delta,
    )
    retained_keys = dict(pool.tile_resident_keys)

    _uvs, stats = pool.update_payloads(
        {0: payloads[0]},
        tile_shape=(2, 2),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        reserve_count=4,
        tile_delta=SimpleNamespace(
            upserts={0: payloads[0]},
            removals=(),
            active_tiles=(0, 1, 2, 3),
            planned_tiles=(0, 1, 2, 3),
            near_tiles=(0, 1, 2, 3),
            near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        ),
    )

    assert set(stats.presented_tiles) == {0, 1, 2, 3}
    assert stats.committed_upserts == (0,)
    assert {tile: pool.tile_resident_keys[tile] for tile in (1, 2, 3)} == {
        tile: retained_keys[tile] for tile in (1, 2, 3)
    }


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


def test_phase_vector_page_quad_uses_resultant_range_shader_mode():
    values = np.asarray([[1 + 0j, 1j], [-1 + 0j, -1j]], dtype=np.complex64)
    plan = plan_source_grid_pages(
        content_key=("phase",),
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(1, 1),
        stored_page_shape=(2, 2),
        dtype="complex64",
        representation="complex_rg32f",
        reducer="phase_vector",
    )[0]
    page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan)
    lod = LodInfo(level=1, factor=2, source_shape=(2, 2), texture_shape=(1, 1), gutter=0)
    native = complex_payload(0)
    payload = replace(
        native,
        image=page.values,
        texture_data=page.values,
        histogram_data=None,
        lod=lod,
        page_backing=PageBackedPresentation((plan,), (page,), (0, 2, 0, 2), lod),
    )

    assert _payload_mode(payload, rgb_already_windowed=False) == 5


def test_mean_complex_page_keeps_level_controlled_phase_color_shader_mode():
    values = np.asarray([[1 + 0j, 10j], [-100 + 0j, -10j]], dtype=np.complex64)
    plan = plan_source_grid_pages(
        content_key=("complex",),
        valid_source_rect_yx=(0, 2, 0, 2),
        reduction_yx=(1, 1),
        stored_page_shape=(2, 2),
        dtype="complex64",
        representation="complex_rg32f",
        reducer="mean",
    )[0]
    page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan)
    lod = LodInfo(level=1, factor=2, source_shape=(2, 2), texture_shape=(1, 1), gutter=0)
    native = complex_payload(0)
    payload = replace(
        native,
        image=page.values,
        texture_data=page.values,
        histogram_data=None,
        lod=lod,
        page_backing=PageBackedPresentation((plan,), (page,), (0, 2, 0, 2), lod),
    )

    # Mode 4 applies u_levels to magnitude and phase to hue. Mode 5 is only
    # valid for explicit phase_vector pages and would make level drags inert.
    assert _payload_mode(payload, rgb_already_windowed=False) == 4


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
    visual._set_lut_texture = lambda lut, key=None: (
        uploaded_luts.append(np.array(lut, copy=True)) or True
    )

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


def test_bounded_gpu_layer_commit_keeps_retained_tile_geometry():
    layer = GpuMontageLayer(
        scene=FakeScene(),
        visuals=None,
        gloo=FakeGloo(),
        transforms=None,
        parent=None,
        limits=GpuDeviceLimits(max_texture_size=8),
    )
    montage = SimpleNamespace(
        indices=tuple(range(4)),
        tile_width=2,
        tile_height=2,
        columns=4,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",) * 4)
    payloads = {index: payload(index, float(index)) for index in range(4)}
    first_delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=(0, 1, 2, 3),
        planned_tiles=(0, 1, 2, 3),
        near_tiles=(0, 1, 2, 3),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
    )
    layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=first_delta,
    )

    bounded = layer.update(
        payloads={0: payloads[0]},
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        tile_delta=SimpleNamespace(
            upserts={0: payloads[0]},
            removals=(),
            active_tiles=(0, 1, 2, 3),
            planned_tiles=(0, 1, 2, 3),
            near_tiles=(0, 1, 2, 3),
            near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        ),
    )

    assert set(bounded.presented_tiles) == {0, 1, 2, 3}
    assert len(layer._page_payloads_by_index[0]) == 4
    assert layer._visuals_by_page[0].visible


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
    assert (
        sum(
            len(texture.updates)
            for page in layer._pool.pages
            for texture in (page.scalar_texture, page.color_texture)
        )
        == texture_updates
    )
    assert all(visual.mappings[-1][1] is second_mapping for visual in layer._visuals_by_page)


def test_storage_classes_coexist_and_visibility_clear_preserves_page_residency():
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
        tile_width=4,
        tile_height=4,
        columns=1,
        rows=1,
        gap=0,
    )
    geometry = SimpleNamespace(montage=montage, montage_tile_states=("loaded",))
    lod = LodInfo(level=1, factor=2, source_shape=(4, 4), texture_shape=(2, 2), gutter=0)

    def page_payload(content_key, *, values, supplied, mapping, representation, dtype):
        plans = plan_source_grid_pages(
            content_key=content_key,
            valid_source_rect_yx=(0, 4, 0, 4),
            reduction_yx=(1, 1),
            stored_page_shape=(2, 2),
            dtype=dtype,
            representation=representation,
            reducer="mean",
        )
        pages = (
            tuple(
                materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan) for plan in plans
            )
            if supplied
            else ()
        )
        image = pages[0].values if pages else np.zeros((2, 2), dtype=dtype)
        return DisplayTilePayload(
            tile_number=0,
            source_index=0,
            image=image,
            histogram_data=None,
            source_id=("page-frame", content_key),
            lod=lod,
            shader_mapping=mapping,
            page_backing=PageBackedPresentation(plans, pages, (0, 4, 0, 4), lod),
        )

    complex_mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    scalar_mapping = ShaderMapping(component=ShaderComponent.REAL)
    predecessor = page_payload(
        ("complex", 1),
        values=np.arange(16, dtype=np.float32).reshape(4, 4).astype(np.complex64),
        supplied=True,
        mapping=complex_mapping,
        representation="complex_rg32f",
        dtype="complex64",
    )
    first_delta = TilePresentationDelta(
        structure_revision=0,
        payload_revision=1,
        visibility_revision=0,
        level_revision=0,
        histogram_revision=0,
        viewport_revision=0,
        upserts={0: predecessor},
        active_tiles=(0,),
        planned_tiles=(0,),
    )
    layer.update(
        payloads={0: predecessor},
        geometry=geometry,
        levels=(-1.0, 1.0),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        shader_mapping=complex_mapping,
        tile_delta=first_delta,
    )
    predecessor_resident_key = layer._pool.tile_resident_keys[0]

    successor = page_payload(
        ("scalar", 2),
        values=np.zeros((4, 4), dtype=np.float32),
        supplied=False,
        mapping=scalar_mapping,
        representation="scalar_r32f",
        dtype="float32",
    )
    atomic_delta = TilePresentationDelta(
        structure_revision=0,
        payload_revision=2,
        visibility_revision=0,
        level_revision=1,
        histogram_revision=0,
        viewport_revision=0,
        base_revision=1,
        upserts={0: successor},
        active_tiles=(0,),
        planned_tiles=(0,),
        atomic_handoff=True,
    )
    rejected = layer.update(
        payloads={0: successor},
        geometry=geometry,
        levels=(0.0, 15.0),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        shader_mapping=scalar_mapping,
        tile_delta=atomic_delta,
    )

    assert rejected.committed_upserts == ()
    assert layer._pool.tile_resident_keys[0] == predecessor_resident_key
    assert layer._levels == (-1.0, 1.0)
    assert layer._shader_mapping is complex_mapping
    assert layer._visuals_by_page[0].mappings[-1][1] is complex_mapping
    assert layer._pool.tile_page_candidate_missing[0] == successor.page_backing.requested_keys

    supplied_successor = page_payload(
        ("scalar", 2),
        values=np.ones((4, 4), dtype=np.float32),
        supplied=True,
        mapping=scalar_mapping,
        representation="scalar_r32f",
        dtype="float32",
    )
    predecessor_draw_parts = layer._pool.tile_draw_parts[0]
    warmed = layer.warm_residency(
        payloads={0: supplied_successor},
        geometry=geometry,
        rgb_already_windowed=False,
        tile_delta=atomic_delta,
    )

    assert warmed.texture_uploads > 0
    assert layer._pool.tile_resident_keys[0] == predecessor_resident_key
    assert layer._pool.tile_draw_parts[0] == predecessor_draw_parts
    assert layer._levels == (-1.0, 1.0)
    assert layer._shader_mapping is complex_mapping
    assert all(
        layer._pool._page_table.resolve(key) is not None
        for key in supplied_successor.page_backing.requested_keys
    )
    assert layer.payload_resident(supplied_successor) is True

    admitted_delta = replace(atomic_delta, upserts={0: supplied_successor})
    admitted = layer.update(
        payloads={0: supplied_successor},
        geometry=geometry,
        levels=(0.0, 15.0),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        shader_mapping=scalar_mapping,
        tile_delta=admitted_delta,
    )

    assert admitted.committed_upserts == (0,)
    assert admitted.texture_uploads == 0
    assert layer._levels == (0.0, 15.0)
    assert layer._shader_mapping is scalar_mapping

    resident_page_keys = {
        *predecessor.page_backing.requested_keys,
        *supplied_successor.page_backing.requested_keys,
    }
    assert {page.storage_mode for page in layer._pool.pages} >= {"complex", "scalar"}

    layer.clear()

    assert not layer._pool.tile_resident_keys
    assert not layer._pool.tile_draw_parts
    assert not any(visual.visible for visual in layer._visuals_by_page)
    assert all(key in layer._pool._page_table for key in resident_page_keys)


def test_touched_atlas_page_repairs_stale_local_mapping_when_global_key_is_unchanged():
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
    mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    payloads = {index: payload(index, float(index)) for index in range(5)}
    delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=tuple(range(5)),
        planned_tiles=tuple(range(5)),
        near_tiles=(),
        near_tile_source_ids={},
        force_refresh=False,
    )
    first = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        shader_mapping=mapping,
        tile_delta=delta,
    )
    assert first.page_count == 2
    stale_visual = layer._visuals_by_page[1]
    stale_visual.mappings = [
        (
            ShaderMapping().identity_key,
            ShaderMapping(),
        )
    ]

    replacement = payload(4, 40.0)
    payloads[4] = replacement
    repaired = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=(4,),
        rgb_already_windowed=False,
        shader_mapping=mapping,
        tile_delta=SimpleNamespace(**{**vars(delta), "upserts": {4: replacement}}),
    )

    assert repaired.shader_uniform_updates == 1
    assert stale_visual.mappings[-1][0] == mapping.identity_key
    assert stale_visual.mappings[-1][1] is mapping


def test_clean_gpu_layer_update_reuses_page_bindings():
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
    payloads = {index: payload(index, float(index)) for index in range(5)}
    delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=tuple(range(5)),
        planned_tiles=tuple(range(5)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        force_refresh=False,
    )

    first = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=delta,
    )
    texture_calls = [visual.texture_calls for visual in layer._visuals_by_page]
    mipmap_calls = [visual.mipmap_calls for visual in layer._visuals_by_page]
    geometry_calls = [visual.geometry_calls for visual in layer._visuals_by_page]

    second = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=(),
        rgb_already_windowed=False,
        tile_delta=delta,
    )

    assert first.page_count == 2
    assert second.texture_uploads == 0
    assert second.vertex_uploads == 0
    assert [visual.texture_calls for visual in layer._visuals_by_page] == texture_calls
    assert [visual.mipmap_calls for visual in layer._visuals_by_page] == mipmap_calls
    assert [visual.geometry_calls for visual in layer._visuals_by_page] == geometry_calls


def test_clean_gpu_layer_update_skips_atlas_and_visual_walk(monkeypatch):
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
    payloads = {index: payload(index, float(index)) for index in range(5)}
    initial_delta = SimpleNamespace(
        upserts=payloads,
        removals=(),
        active_tiles=tuple(range(5)),
        planned_tiles=tuple(range(5)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        force_refresh=False,
    )

    layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=initial_delta,
    )
    texture_calls = [visual.texture_calls for visual in layer._visuals_by_page]
    mipmap_calls = [visual.mipmap_calls for visual in layer._visuals_by_page]
    geometry_calls = [visual.geometry_calls for visual in layer._visuals_by_page]
    mapping_calls = [visual.mapping_calls for visual in layer._visuals_by_page]

    def fail_update_payloads(*_args, **_kwargs):
        raise AssertionError("clean repeat commit must not enter atlas residency update")

    monkeypatch.setattr(layer._pool, "update_payloads", fail_update_payloads)
    clean_delta = SimpleNamespace(
        upserts={},
        removals=(),
        active_tiles=tuple(range(5)),
        planned_tiles=tuple(range(5)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
        force_refresh=False,
    )

    clean = layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=(),
        rgb_already_windowed=False,
        tile_delta=clean_delta,
    )

    assert clean.texture_uploads == 0
    assert clean.vertex_uploads == 0
    assert clean.items_skipped == 5
    assert layer.changed_page_indices() == ()
    assert [visual.texture_calls for visual in layer._visuals_by_page] == texture_calls
    assert [visual.mipmap_calls for visual in layer._visuals_by_page] == mipmap_calls
    assert [visual.geometry_calls for visual in layer._visuals_by_page] == geometry_calls
    assert [visual.mapping_calls for visual in layer._visuals_by_page] == mapping_calls


def test_gpu_layer_reports_only_changed_pages_for_small_upsert():
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
    payloads = {index: payload(index, float(index)) for index in range(5)}
    layer.update(
        payloads=payloads,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=None,
        rgb_already_windowed=False,
        tile_delta=SimpleNamespace(
            upserts=payloads,
            removals=(),
            active_tiles=tuple(range(5)),
            planned_tiles=tuple(range(5)),
            near_tiles=(),
            near_tile_source_ids={index: value.source_id for index, value in payloads.items()},
            force_refresh=False,
        ),
    )

    changed = dict(payloads)
    changed[4] = payload(4, 99.0, source_id=("changed", 4))
    update_delta = SimpleNamespace(
        upserts={4: changed[4]},
        removals=(),
        active_tiles=tuple(range(5)),
        planned_tiles=tuple(range(5)),
        near_tiles=(),
        near_tile_source_ids={index: value.source_id for index, value in changed.items()},
        force_refresh=False,
    )

    update = layer.update(
        payloads=changed,
        geometry=geometry,
        levels=(0.0, 4.0),
        dirty_tiles=(4,),
        rgb_already_windowed=False,
        tile_delta=update_delta,
    )

    assert update.texture_uploads == 1
    assert layer.changed_page_indices() == (1,)


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
    pool.update_payloads(
        values, tile_shape=(2, 2), dirty_tiles=None, rgb_already_windowed=False, reserve_count=4
    )
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
    layout = {
        12_345: TileLayoutRegion(
            tile_number=12_345, source_index=12_345, x=90, y=246, width=2, height=2
        )
    }
    vertices, _texcoords, _modes = _quad_buffers(
        layout,
        {12_345: payload(12_345, 1.0)},
        {12_345: (0.0, 0.0, 1.0, 1.0)},
        rgb_already_windowed=False,
    )

    assert vertices.shape == (6, 2)


def _lod_payload(
    tile_number: int, value: float, *, level: int, source_shape=(4, 4)
) -> DisplayTilePayload:
    from arrayscope.display.lod import LodInfo

    if int(level) != 0:
        raise ValueError("reduced renderer fixtures require canonical page_backing")
    image = np.full(tuple(source_shape), value, dtype=np.float32)
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=tile_number,
        image=image,
        histogram_data=None,
        source_id=("tile", tile_number, value, "lod", int(level)),
        texture_data=image,
        lod=LodInfo(
            level=0,
            factor=1,
            source_shape=tuple(source_shape),
            texture_shape=image.shape[:2],
        ),
    )


def test_unsupported_replacement_retains_previous_mapping():
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=8)
    original = {0: payload(0, 1.0)}
    pool.update_payloads(
        original,
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
    )
    original_mapping = dict(pool.tile_slots)
    incompatible = replace(payload(0, 2.0), texture_kind=TexturePlaneKind.RGB8)

    _uvs, stats = pool.update_payloads(
        {0: incompatible},
        tile_shape=(2, 2),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=1,
    )

    assert stats.items_skipped == 1
    assert stats.presented_tiles == (0,)
    assert dict(pool.tile_slots) == original_mapping
    assert pool.presented_identities()[0] == original[0].source_id


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

    pool._slot_for(
        "incoming-3", active_keys={"incoming-1", "incoming-2", "incoming-3"}, near_keys=set()
    )
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


def test_atlas_mipmaps_default_off_after_stale_mip_field_regression():
    # 2026-07-04: whole-atlas regen showed stale mip content (previous atlas
    # slot occupants) on minified tiles between regens.  Off until a
    # per-slot mip invalidation exists.
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    payloads = {0: _lod_payload(0, 1.0, level=0)}
    pool.update_payloads(payloads, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False)
    assert all(page.mipmap_levels == 0 for page in pool.pages)


def test_atlas_mipmap_levels_are_the_bleed_free_depth():
    from arrayscope.display.backends.vispy.tiles import _atlas_mipmap_levels

    # Mip k averages 2^k x 2^k blocks; blocks stay inside one tile only
    # while both tile edges divide by 2^k.
    assert _atlas_mipmap_levels((336, 336)) == 4
    assert _atlas_mipmap_levels((64, 64)) == 5  # capped
    assert _atlas_mipmap_levels((63, 64)) == 0  # odd edge: no safe mips
    assert _atlas_mipmap_levels((84, 84)) == 2


def test_uploads_dirty_the_page_for_draw_time_mipmap_regen(monkeypatch):
    import arrayscope.display.backends.vispy.tiles as tiles_module

    monkeypatch.setattr(tiles_module, "_ATLAS_MIPMAPS_ENABLED", True)
    pool = TextureAtlasPool(FakeGloo(), max_texture_size=16)
    payloads = {index: _lod_payload(index, float(index + 1), level=0) for index in range(2)}
    _uvs, stats = pool.update_payloads(
        payloads, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False
    )
    page = pool.pages[0]
    assert page.mipmap_levels == 2
    assert page.mipmap_dirty is True
    # Regeneration is draw-time work; the upload commit reports none yet.
    assert stats.mipmap_available is False
    assert stats.mipmap_updates == 0

    # Simulate the visual's draw-time regen: the next update reports the
    # regens exactly once (delta reporting), and availability follows the
    # page state.
    page.mipmap_dirty = False
    page.mipmap_ready = True
    page.mipmap_updates = 3
    _uvs, stats = pool.update_payloads(
        payloads, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False
    )
    assert stats.mipmap_available is True
    assert stats.mipmap_updates == 3
    _uvs, stats = pool.update_payloads(
        payloads, tile_shape=(4, 4), dirty_tiles=None, rgb_already_windowed=False
    )
    assert stats.mipmap_updates == 0
