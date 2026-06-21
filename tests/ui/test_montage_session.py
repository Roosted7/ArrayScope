from collections import deque

import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import RenderedTile, make_montage_plan
from arrayscope.display.model.frame import TileCommitReport
from arrayscope.display.shader_mapping import ShaderComponent, ShaderDisplayMode, ShaderMapping, ShaderScale, TexturePlaneKind
from arrayscope.window.montage_session import MontageRenderSession


def _session():
    state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(0, 1, 2, 3), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4)
    return MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0, 1, 2, 3),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=list(plan.tiles[:3]),
    )


def test_montage_render_session_returns_pending_tiles_in_order():
    session = _session()

    assert isinstance(session.pending_tiles, deque)
    assert isinstance(session.pending_level_tiles, deque)
    assert isinstance(session.pending_completed_tiles, deque)
    assert session.next_tile().source_index == 0
    assert session.next_tile().source_index == 1


def test_montage_render_session_materialized_tile_stays_loading_until_presented():
    session = _session()
    tile = session.next_tile()
    rendered = RenderedTile(tile, np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16)

    session.mark_loaded(rendered)

    assert session.is_tile_loaded(tile)
    assert int(tile.montage_index) in session.loading_tiles
    assert tile not in session.pending_tiles

    session.mark_presented((tile.montage_index,))

    assert int(tile.montage_index) not in session.loading_tiles
    assert int(tile.montage_index) in session.presented_tiles


def test_montage_render_session_keeps_skipped_separate_from_pending():
    session = _session()
    session.mark_skipped(session.plan.tiles[1])

    assert 1 in session.skipped_tiles
    assert session.plan.tiles[1] not in session.pending_tiles
    assert session.next_tile().source_index == 0
    assert session.next_tile().source_index == 2


def test_montage_render_session_patches_existing_canvas_in_place():
    from arrayscope.display.montage import MontageTileState, make_montage_viewport_canvas

    session = _session()
    tile0 = session.plan.tiles[0]
    tile1 = session.plan.tiles[1]
    rendered0 = RenderedTile(tile0, np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16)
    canvas = make_montage_viewport_canvas(
        session.plan,
        (rendered0,),
        viewport_shape=(10, 10),
        budget_bytes=1024 * 1024,
        dtype=np.float32,
        loading_tiles=(tile1,),
    )
    session.initialize_canvas(canvas)
    data_id = id(session.current_canvas().data)
    rendered1 = RenderedTile(tile1, np.full((2, 2), 5, dtype=np.float32), np.full((2, 2), 5, dtype=np.float32), 0.0, (2, 2), 16)

    assert session.patch_rendered_tile(rendered1)

    patched = session.current_canvas()
    assert id(patched.data) == data_id
    assert patched.tile_states[1] == MontageTileState.LOADED
    assert session.consume_dirty_rects() == ((3, 0, 5, 2),)
    np.testing.assert_array_equal(patched.data[0:2, 3:5], np.full((2, 2), 5, dtype=np.float32))


def test_montage_render_session_reuses_typed_payload_wrappers_until_tile_changes():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    histogram = image * 2.0
    first_rendered = RenderedTile(tile, image, histogram, 0.0, image.shape, image.nbytes)
    session.mark_loaded(first_rendered)

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})
    state, delta = session.build_tile_presentation({0: ("tile", 0)})
    session.acknowledge_tile_presentation(delta, TileCommitReport(presented_tiles=(0,)))
    clean_state, clean_delta = session.build_tile_presentation({0: ("tile", 0)})

    assert second[0] is first[0]
    assert state.payloads[0] is first[0]
    assert delta.upserts == {0: first[0]}
    assert clean_state.payloads[0] is first[0]
    assert clean_delta.upserts == {}
    replacement = np.full((2, 2), 3.0, dtype=np.float32)
    session.mark_loaded(RenderedTile(tile, replacement, replacement, 0.0, replacement.shape, replacement.nbytes))
    third_state, third_delta = session.build_tile_presentation({0: ("tile", 0, "replacement")})
    assert third_state.payloads[0] is not first[0]
    assert third_state.payloads[0].image is replacement
    assert third_delta.upserts[0] is third_state.payloads[0]


def test_montage_render_session_retries_deferred_payload_until_backend_acknowledges():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    proposed, delta = session.build_tile_presentation({0: ("tile", 0)})

    assert 0 in proposed.payloads
    assert session.tile_presentation_state.payloads == {}

    session.acknowledge_tile_presentation(delta, TileCommitReport(deferred_tiles=(0,)))
    retry_state, retry_delta = session.build_tile_presentation({0: ("tile", 0)})

    assert session.tile_presentation_state.payloads == {}
    assert retry_delta.upserts == {0: proposed.payloads[0]}
    assert retry_state.payloads[0] is proposed.payloads[0]

    session.acknowledge_tile_presentation(retry_delta, TileCommitReport(presented_tiles=(0,)))

    assert session.tile_presentation_state.payloads[0] is proposed.payloads[0]
    assert 0 not in session.dirty_payloads
    assert session.deferred_display_tiles == ()


def test_montage_render_session_dirty_payloads_keep_session_incomplete_until_acknowledged():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.pending_tiles.clear()
    session.loading_tiles.clear()

    session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    assert not session.is_complete()

    state, delta = session.build_tile_presentation({0: ("tile", 0)})
    session.acknowledge_tile_presentation(delta, TileCommitReport(presented_tiles=state.active_payloads(delta)))
    session.mark_presented(state.active_payloads(delta))

    assert session.is_complete()


def test_montage_render_session_delta_carries_near_sources_without_payloads():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    source_ids = {index: ("tile-source", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(source_ids)

    assert delta.near_tiles == (0, 1, 2, 3)
    assert delta.near_tile_source_ids[0] == delta.upserts[0].source_id
    assert {key: delta.near_tile_source_ids[key] for key in (1, 2, 3)} == {
        key: source_ids[key] for key in (1, 2, 3)
    }
    assert set(delta.upserts) == {0}


def test_lod_payload_does_not_reduce_display_ready_rgb_phase_tiles():
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(np.uint8),
        rgb=True,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    rgb = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    histogram = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        plan.tiles[0],
        rgb,
        histogram,
        0.0,
        rgb.shape,
        rgb.nbytes,
        texture_kind=TexturePlaneKind.RGB8,
        semantic_data=rgb,
    )
    session.mark_loaded(rendered)

    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert payload.lod.factor == 1
    assert payload.image.shape[:2] == (8, 8)
    assert payload.histogram_data.shape[:2] == (8, 8)
    assert payload.texture_data.shape[:2] == (8, 8)
    assert payload.semantic_data.shape[:2] == (8, 8)
    assert payload.semantic_histogram_data.shape[:2] == (8, 8)
    assert payload.source_shape == (8, 8)


def _zoomed_out_session(*, dtype=np.float32, rgb=False):
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    return MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(dtype),
        rgb=rgb,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )


def test_zoomed_out_scalar_payload_keeps_exact_texture_on_ui_commit_path():
    session = _zoomed_out_session()
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    histogram = image + 100.0
    rendered = RenderedTile(
        session.plan.tiles[0],
        image,
        histogram,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    session.mark_loaded(rendered)

    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert session.desired_tile_lod_factor > 1
    assert session.tile_lod_factor == 1
    assert payload.lod.factor == 1
    assert payload.texture_data is image
    assert payload.texture_data.shape[:2] == payload.image.shape[:2]
    assert payload.semantic_data.shape[:2] == (8, 8)
    assert payload.semantic_histogram_data.shape[:2] == (8, 8)


def test_zoomed_out_payload_reuses_clean_wrapper_without_cpu_lod_work():
    session = _zoomed_out_session()
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        session.plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    session.mark_loaded(rendered)

    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert second is first


def test_exact_payload_seeds_from_previous_session_without_materialization():
    first_session = _zoomed_out_session()
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        first_session.plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    first_session.mark_loaded(rendered)
    source_ids = {0: ("tile", 0)}
    first = first_session.snapshot_display_tile_payloads(source_ids)[0]

    second_session = _zoomed_out_session()
    second_session.mark_loaded(rendered)
    second_session.seed_display_tile_payloads({0: first}, source_ids)
    second = second_session.snapshot_display_tile_payloads(source_ids)[0]

    assert second is first


def test_zoomed_out_complex_payload_keeps_exact_semantics_and_texture():
    session = _zoomed_out_session(dtype=np.complex64, rgb=True)
    real = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    source = (real + 1j * (real + 100.0)).astype(np.complex64)
    histogram = np.log10(np.abs(source)).astype(np.float32)
    rendered = RenderedTile(
        session.plan.tiles[0],
        source,
        histogram,
        0.0,
        source.shape,
        source.nbytes,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=source,
        shader_mapping=ShaderMapping(
            component=ShaderComponent.ABS,
            scale=ShaderScale.LOG,
            display_mode=ShaderDisplayMode.PHASE_COLOR,
        ),
    )
    session.mark_loaded(rendered)

    payload = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert session.desired_tile_lod_factor > 1
    assert payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F
    assert payload.lod.factor == 1
    assert np.iscomplexobj(payload.texture_data)
    assert payload.texture_data.shape[:2] == payload.image.shape[:2]
    np.testing.assert_array_equal(payload.semantic_data, source)
    np.testing.assert_array_equal(payload.semantic_histogram_data, histogram)
    assert payload.source_shape == (8, 8)


def test_shader_mapping_change_reuses_texture_content_identity():
    session = _zoomed_out_session(dtype=np.complex64, rgb=True)
    source = np.ones((8, 8), dtype=np.complex64)
    histogram = np.ones((8, 8), dtype=np.float32)
    first_mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LINEAR,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    second_mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        scale=ShaderScale.LOG,
        display_mode=ShaderDisplayMode.PHASE_COLOR,
    )
    first_rendered = RenderedTile(
        session.plan.tiles[0], source, histogram, 0.0, source.shape, source.nbytes,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=source,
        shader_mapping=first_mapping,
    )
    session.mark_loaded(first_rendered)
    first = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    second_rendered = RenderedTile(
        session.plan.tiles[0], source, histogram, 0.0, source.shape, source.nbytes,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=source,
        shader_mapping=second_mapping,
    )
    session.mark_loaded(second_rendered)
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert second is not first
    assert second.source_id == first.source_id
    assert second.texture_data is first.texture_data
    assert second.shader_mapping == second_mapping


def test_retarget_viewport_separates_draw_set_from_loaded_residency():
    state = ViewState.from_shape((2, 2, 8)).with_montage_axis(2, indices=tuple(range(8)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(8)), tile_shape=(2, 2), columns=8)
    session = MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=tuple(range(8)),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(1, 1),
        view_range=((0.0, 1.0), (0.0, 1.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles[:2],
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    for tile in plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    additions, changed = session.retarget_viewport(
        view_range=((8.0, 9.0), (0.0, 1.0)),
        viewport_shape=(1, 1),
        coverage_margin_tiles=1,
        near_margin_tiles=2,
    )

    assert changed
    assert tuple(tile.montage_index for tile in session.visible_tiles) == (2, 3)
    assert set(session.rendered_tiles) == {0, 1}
    assert tuple(tile.montage_index for tile in additions) == (2, 3, 4)

    state, delta = session.build_tile_presentation({index: ("tile-source", index) for index in range(8)})
    assert delta.active_tiles == ()
    assert state.active_payloads(delta) == {}


def test_temporary_materialization_gap_does_not_remove_committed_payloads():
    session = _session()
    source_ids = {}
    for tile in session.plan.tiles[:2]:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))
    first_state, first_delta = session.build_tile_presentation(source_ids)
    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
    )
    session.mark_presented(first_state.active_payloads(first_delta))

    session.rendered_tiles.clear()
    session.display_tile_payloads.clear()
    next_state, next_delta = session.build_tile_presentation(source_ids)

    assert next_delta.removals == ()
    assert tuple(next_state.payloads) == (0, 1)
    assert session.ensure_tile_states()[0].value == "loaded"


def test_retarget_viewport_does_not_requeue_known_guard_band_tiles():
    session = _session()
    session.visible_tiles = session.plan.tiles[:2]
    session.loading_tiles = {2}
    session.pending_tiles = deque((session.plan.tiles[2],))

    additions, _changed = session.retarget_viewport(
        view_range=((3.0, 4.0), (0.0, 1.0)),
        viewport_shape=(1, 1),
        coverage_margin_tiles=1,
    )

    assert 2 not in {tile.montage_index for tile in additions}


def test_montage_render_session_passes_cold_deadline_without_slicing_upserts():
    session = _session()
    session.pending_tiles.clear()
    source_ids = {}
    for tile in session.plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    first_state, first_delta = session.build_tile_presentation(source_ids, cold_deadline_ms=3.5)

    assert tuple(first_delta.upserts) == (0, 1, 2, 3)
    assert tuple(first_state.payloads) == (0, 1, 2, 3)
    assert first_delta.cold_deadline_ms == 3.5
    assert session.deferred_display_tiles == ()
    assert not session.is_complete()

    session.acknowledge_tile_presentation(
        first_delta,
        TileCommitReport(presented_tiles=first_state.active_payloads(first_delta)),
    )
    session.mark_presented(first_state.active_payloads(first_delta))
    assert session.is_complete()


def test_montage_render_session_tile_states_keep_materialized_tiles_loading_until_presented():
    session = _session()
    session.pending_tiles.clear()
    session.visible_tiles = (session.plan.tiles[3], session.plan.tiles[1])
    source_ids = {}
    for tile in session.plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    state, delta = session.build_tile_presentation(source_ids)
    tile_states = session.ensure_tile_states()

    assert tuple(delta.upserts) == (0, 1, 2, 3)
    assert tuple(state.active_payloads(delta)) == (3, 1)
    assert {str(tile_states[index].value) for index in range(4)} == {"loading"}

    session.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    session.mark_presented(state.active_payloads(delta))
    tile_states = session.ensure_tile_states()

    assert {index for index in range(4) if tile_states[index].value == "loaded"} == {1, 3}
    assert {index for index in range(4) if tile_states[index].value == "loading"} == {0, 2}


def test_montage_render_session_does_not_force_clear_for_untrusted_source_ids():
    session = _session()
    session.pending_tiles.clear()
    source_ids = {}
    for tile in session.plan.tiles:
        image = np.full((2, 2), tile.source_index, dtype=np.float32)
        session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
        source_ids[int(tile.montage_index)] = ("tile-source", int(tile.montage_index))

    state, delta = session.build_tile_presentation(
        source_ids,
        source_ids_trusted=False,
        max_upserts=1,
    )

    assert not delta.force_refresh
    assert delta.clear_reason == ""
    assert tuple(delta.upserts) == (0, 1, 2, 3)
    assert tuple(state.payloads) == (0, 1, 2, 3)
    assert session.deferred_display_tiles == ()



def test_tile_presentation_state_rejects_stale_delta():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))
    state, delta = session.build_tile_presentation({0: ("tile-source", 0)})

    stale = type(delta)(
        structure_revision=delta.structure_revision,
        payload_revision=delta.payload_revision + 1,
        visibility_revision=delta.visibility_revision,
        level_revision=delta.level_revision,
        histogram_revision=delta.histogram_revision,
        viewport_revision=delta.viewport_revision,
        base_revision=delta.base_revision,
        target_revision=delta.target_revision + 1,
        upserts=delta.upserts,
    )

    assert state.revision == delta.target_revision
    assert state.apply_delta(stale) is state


def test_seeded_payloads_retain_committed_state_across_retarget():
    original = _session()
    tile = original.plan.tiles[2]
    image = np.full((2, 2), 2, dtype=np.float32)
    rendered = RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes)
    original.mark_loaded(rendered)
    source_ids = {2: ("tile-source", 2)}
    state, delta = original.build_tile_presentation(source_ids)
    original.acknowledge_tile_presentation(
        delta,
        TileCommitReport(presented_tiles=state.active_payloads(delta)),
    )
    original.mark_presented(state.active_payloads(delta))

    shifted_state = ViewState.from_shape((2, 2, 4)).with_montage_axis(2, indices=(2, 3), text="2:4")
    shifted_plan = make_montage_plan(shifted_state, axis=2, indices=(2, 3), tile_shape=(2, 2), columns=2)
    shifted_rendered = RenderedTile(
        shifted_plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
    )
    shifted = MontageRenderSession(
        session_id=2,
        key="key",
        render_generation=1,
        level_key="levels",
        level_expected_indices=(2, 3),
        plan=shifted_plan,
        view_state=shifted_state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(10, 10),
        view_range=None,
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=shifted_plan.tiles,
        rendered_tiles={0: shifted_rendered},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )

    shifted.seed_display_tile_payloads(state.payloads, {0: ("tile-source", 2)})
    next_state, next_delta = shifted.build_tile_presentation({0: ("tile-source", 2), 1: ("tile-source", 3)})

    assert 0 in next_state.payloads
    assert next_state.payloads[0].source_index == 2
    assert next_delta.removals == ()
    assert next_delta.upserts == {}
    assert 0 in shifted.presented_tiles
