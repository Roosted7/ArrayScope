import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import RenderedTile, make_montage_plan
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

    assert session.next_tile().source_index == 0
    assert session.next_tile().source_index == 1


def test_montage_render_session_mark_loaded_moves_tile_from_loading_to_loaded():
    session = _session()
    tile = session.next_tile()
    rendered = RenderedTile(tile, np.ones((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0.0, (2, 2), 16)

    session.mark_loaded(rendered)

    assert session.is_tile_loaded(tile)
    assert int(tile.montage_index) not in session.loading_tiles
    assert tile not in session.pending_tiles


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


def test_montage_render_session_delta_carries_near_sources_without_payloads():
    session = _session()
    tile = session.plan.tiles[0]
    image = np.ones((2, 2), dtype=np.float32)
    session.mark_loaded(RenderedTile(tile, image, image, 0.0, image.shape, image.nbytes))

    source_ids = {index: ("tile-source", index) for index in range(4)}
    _state, delta = session.build_tile_presentation(source_ids)

    assert delta.near_tiles == (0, 1, 2, 3)
    assert delta.near_tile_source_ids == source_ids
    assert set(delta.upserts) == {0}


def test_lod_payload_does_not_reduce_display_ready_rgb_phase_tiles():
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
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


def test_lod_payload_reduces_scalar_texture_and_histogram_together():
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    histogram = image + 100.0
    rendered = RenderedTile(
        plan.tiles[0],
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

    assert payload.lod.factor > 1
    assert payload.image.shape[:2] == (8, 8)
    assert payload.histogram_data.shape[:2] == (8, 8)
    assert payload.texture_data.shape[:2] != payload.image.shape[:2]
    assert payload.semantic_data.shape[:2] == (8, 8)
    assert payload.semantic_histogram_data.shape[:2] == (8, 8)


def test_lod_payload_reuses_clean_wrapper_without_rebuilding_pyramid(monkeypatch):
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(np.float32),
        rgb=False,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        plan.tiles[0],
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

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("clean LOD payload rebuilt pyramid")

    monkeypatch.setattr("arrayscope.window.montage_session.build_tile_lod_pyramid", fail_rebuild)
    second = session.snapshot_display_tile_payloads({0: ("tile", 0)})[0]

    assert second is first


def test_lod_payload_seeds_from_previous_session_without_rebuilding_pyramid(monkeypatch):
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)

    def new_session():
        return MontageRenderSession(
            session_id=1,
            key="key",
            render_generation=1,
            level_key="levels",
            plan=plan,
            view_state=state,
            document=None,
            montage_axis=2,
            colormap_lut=None,
            viewport_shape=(2, 2),
            view_range=((0.0, 64.0), (0.0, 64.0)),
            output_dtype=np.dtype(np.float32),
            rgb=False,
            window_mode=None,
            force_auto=False,
            visible_tiles=plan.tiles,
            rendered_tiles={},
            loading_tiles=set(),
            skipped_tiles=set(),
            pending_tiles=[],
        )

    image = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    rendered = RenderedTile(
        plan.tiles[0],
        image,
        image,
        0.0,
        image.shape,
        image.nbytes,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_data=image,
    )
    first_session = new_session()
    first_session.mark_loaded(rendered)
    source_ids = {0: ("tile", 0)}
    first = first_session.snapshot_display_tile_payloads(source_ids)[0]

    second_session = new_session()
    second_session.mark_loaded(rendered)
    second_session.seed_display_tile_payloads({0: first}, source_ids)

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("seeded LOD payload rebuilt pyramid")

    monkeypatch.setattr("arrayscope.window.montage_session.build_tile_lod_pyramid", fail_rebuild)
    second = second_session.snapshot_display_tile_payloads(source_ids)[0]

    assert second is first


def test_lod_payload_reduces_raw_complex_texture_and_preserves_exact_semantics():
    state = ViewState.from_shape((8, 8, 1)).with_montage_axis(2, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(8, 8), columns=1)
    session = MontageRenderSession(
        session_id=1,
        key="key",
        render_generation=1,
        level_key="levels",
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(2, 2),
        view_range=((0.0, 64.0), (0.0, 64.0)),
        output_dtype=np.dtype(np.complex64),
        rgb=True,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
    )
    real = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    source = (real + 1j * (real + 100.0)).astype(np.complex64)
    histogram = np.log10(np.abs(source)).astype(np.float32)
    rendered = RenderedTile(
        plan.tiles[0],
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

    assert payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F
    assert payload.lod.factor > 1
    assert np.iscomplexobj(payload.texture_data)
    assert payload.texture_data.shape[:2] != payload.image.shape[:2]
    np.testing.assert_array_equal(payload.semantic_data, source)
    np.testing.assert_array_equal(payload.semantic_histogram_data, histogram)
    assert payload.source_shape == (8, 8)
