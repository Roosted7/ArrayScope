from __future__ import annotations

from dataclasses import replace

import numpy as np

from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.display.backends.vispy.tiles import TextureAtlasPool
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
from arrayscope.display.model.tile_identity import (
    TileIdentity,
    TileLodIdentity,
    TilePresentationIdentity,
    array_plane_identities,
    complex_mapping_identity,
    tile_truth_record,
)
from arrayscope.display.montage import MontageTileState, make_montage_plan
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    TexturePlaneKind,
)
from arrayscope.display.slice_engine import complex_to_rgb
from arrayscope.render.lod import texture_source_for_rendered
from arrayscope.window.frame_session import FrameSession


class _Texture2D:
    def __init__(self, data=None, *, shape=None, **_kwargs):
        self.shape = tuple(shape) if shape is not None else tuple(np.shape(data))
        self.updates = []

    def set_data(self, data, *, offset=None, copy=True):
        self.updates.append((np.array(data, copy=True), offset, bool(copy)))


class _Gloo:
    Texture2D = _Texture2D


_MAPPING = ShaderMapping(
    component=ShaderComponent.ABS,
    display_mode=ShaderDisplayMode.PHASE_COLOR,
)


def _complex_plane(source_index: int) -> np.ndarray:
    yy, xx = np.mgrid[:4, :4]
    magnitude = np.float32(source_index + 1) + xx.astype(np.float32) / 4.0
    phase = np.float32(source_index) * (np.pi / 4.0) + yy.astype(np.float32) * (np.pi / 8.0)
    return np.asarray(magnitude * np.exp(1j * phase), dtype=np.complex64)


def _target(source_index: int) -> TileIdentity:
    return TileIdentity(
        document_generation=("synthetic-complex", 1),
        operation_key=("identity",),
        source_index=source_index,
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel=ChannelMode.COMPLEX,
        complex_mapping=complex_mapping_identity(_MAPPING),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_generation=("fixture", source_index),
        lod=TileLodIdentity(level=0, factor=1),
    )


def _payload(tile_number: int, source_index: int) -> DisplayTilePayload:
    plane = _complex_plane(source_index)
    rgb, magnitude = complex_to_rgb(plane)
    real_plane, imag_plane = array_plane_identities(plane)
    identity = replace(_target(source_index), real_plane=real_plane, imag_plane=imag_plane)
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=source_index,
        image=rgb,
        histogram_data=magnitude,
        source_id=("synthetic-complex", source_index),
        texture_data=plane,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=plane,
        semantic_histogram_data=magnitude,
        shader_mapping=_MAPPING,
        tile_identity=identity,
        presentation_identity=TilePresentationIdentity(
            levels_generation=9,
            levels=(0.0, 8.0),
            scale="linear",
            lut_identity="phase",
        ),
    )


def _geometry(source_indices: tuple[int, ...]) -> DisplayGeometry:
    state = ViewState.from_shape((4, 4, 7)).with_channel(ChannelMode.COMPLEX)
    state = state.with_montage_axis(2, columns=4, indices=source_indices, text=":")
    return DisplayGeometry(
        view_state=state,
        display_shape=(4, 19),
        montage=MontageGeometry(
            indices=source_indices,
            tile_shape=(4, 4),
            columns=4,
            rows=1,
            gap=1,
        ),
        montage_tile_states=(MontageTileState.LOADED,) * 4,
    )


def _delta(*, upserts, targets, base_revision: int) -> TilePresentationDelta:
    return TilePresentationDelta(
        structure_revision=base_revision + 1,
        payload_revision=base_revision + 1,
        visibility_revision=base_revision + 1,
        level_revision=9,
        histogram_revision=1,
        viewport_revision=base_revision + 1,
        base_revision=base_revision,
        target_revision=base_revision + 1,
        upserts=upserts,
        active_tiles=(0, 1, 2, 3),
        planned_tiles=(0, 1, 2, 3),
        target_identities=targets,
    )


def _transition_fixture():
    first = {tile: _payload(tile, tile) for tile in range(4)}
    successors = {tile: _payload(tile, tile + 3) for tile in range(4)}
    targets = {tile: _target(tile + 3) for tile in range(4)}
    mixed = dict(first)
    mixed[0] = successors[0]
    return first, successors, targets, mixed


def _session_for_dtype(dtype, *, channel: ChannelMode, shader_display: bool) -> FrameSession:
    state = ViewState.from_shape((4, 4, 1)).with_channel(channel)
    state = state.with_montage_axis(2, columns=1, indices=(0,), text=":")
    plan = make_montage_plan(state, axis=2, indices=(0,), tile_shape=(4, 4), columns=1)
    session = FrameSession(
        session_id=1,
        key=("complex-target-kind",),
        render_generation=1,
        level_key=None,
        level_expected_indices=(0,),
        plan=plan,
        view_state=state,
        document=None,
        montage_axis=2,
        colormap_lut=None,
        viewport_shape=(4, 4),
        view_range=((0.0, 4.0), (0.0, 4.0)),
        output_dtype=np.dtype(dtype),
        rgb=channel == ChannelMode.COMPLEX,
        window_mode=None,
        force_auto=False,
        visible_tiles=plan.tiles,
        rendered_tiles={},
        loading_tiles=set(),
        skipped_tiles=set(),
        pending_tiles=[],
        shader_display=bool(shader_display),
    )
    return session


def test_real_view_of_complex_source_targets_complex_shader_texture():
    session = _session_for_dtype(np.complex64, channel=ChannelMode.REAL, shader_display=True)
    tile = session.plan.tiles[0]
    target = session.tile_target_identity(tile, lod_level=0)
    mapping = ShaderMapping(
        component=ShaderComponent.REAL,
        display_mode=ShaderDisplayMode.SCALAR,
    )
    payload_identity = session.tile_payload_identity(
        tile,
        texture_data=_complex_plane(0),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        shader_mapping=mapping,
        lod=TileLodIdentity(),
        quality="exact",
    )

    assert target.texture_kind == TexturePlaneKind.COMPLEX_RG32F
    assert target.complex_mapping == ("scalar", "real", "mapped")
    assert payload_identity.satisfies_target(target)


def test_cpu_complex_view_targets_the_rgb_texture_it_physically_draws():
    session = _session_for_dtype(np.complex64, channel=ChannelMode.COMPLEX, shader_display=False)
    target = session.tile_target_identity(session.plan.tiles[0], lod_level=0)

    assert target.texture_kind == TexturePlaneKind.RGB8
    assert target.complex_mapping == ("phase_color", "abs", "mapped")


def test_cpu_complex_rendered_tile_reports_rgb_physical_texture_kind():
    rendered = type(
        "Rendered",
        (),
        {
            "image": np.zeros((4, 4, 3), dtype=np.uint8),
            "semantic_data": _complex_plane(0),
            "histogram_data": np.ones((4, 4), dtype=np.float32),
            "texture_kind": TexturePlaneKind.COMPLEX_RG32F,
        },
    )()

    source, _histogram, kind = texture_source_for_rendered(rendered, shader_display=False)

    assert source.shape == (4, 4, 3)
    assert kind == TexturePlaneKind.RGB8


def _assert_truth_record(targets, acknowledged, payloads):
    rows = tuple(
        tile_truth_record(
            tile_number=tile,
            target=targets[tile],
            acknowledged=acknowledged.get(tile),
            payload=payloads[tile],
        )
        for tile in range(4)
    )
    assert rows[0]["drawable"] is True
    assert rows[0]["texture_kind"] == "complex_rg32f"
    assert rows[0]["real_plane_identity"] is not None
    assert rows[0]["imag_plane_identity"] is not None
    assert rows[0]["complex_mapping"] == ("phase_color", "abs", "mapped")
    assert rows[0]["lod"] == {"level": 0, "factor": 1, "gutter": 0}
    assert rows[0]["levels_generation"] == 9
    assert all(row["target_identity"] is not None for row in rows)
    assert all(row["placeholder"] is True for row in rows[1:])


def test_pyqtgraph_complex_semantic_transition_hides_unacknowledged_tiles(qt_app):
    first, successors, targets, mixed = _transition_fixture()
    view = ImageView2D()
    try:
        view.setTiledPresentation(
            geometry=_geometry((0, 1, 2, 3)),
            tile_state=TilePresentationState(first, revision=1),
            tile_delta=_delta(
                upserts=first,
                targets={tile: _target(tile) for tile in range(4)},
                base_revision=0,
            ),
            histogramPlotData=None,
            levels=(0.0, 8.0),
            histogramRange=(0.0, 8.0),
        )

        report = view.setTiledPresentation(
            geometry=_geometry((3, 4, 5, 6)),
            tile_state=TilePresentationState(mixed, revision=2),
            tile_delta=_delta(upserts={0: successors[0]}, targets=targets, base_revision=1),
            histogramPlotData=None,
            levels=(0.0, 8.0),
            histogramRange=(0.0, 8.0),
        )

        assert report.presented_tiles == frozenset({0})
        assert report.presented_identities == {0: successors[0].tile_identity}
        assert tuple(tile for tile, state in view._montage_tile_layer.states.items() if state.visible) == (0,)
        _assert_truth_record(targets, report.presented_identities, mixed)
    finally:
        view.close()


def test_vispy_complex_semantic_transition_hides_unacknowledged_tiles():
    first, successors, targets, mixed = _transition_fixture()
    pool = TextureAtlasPool(_Gloo(), max_texture_size=32)
    pool.update_payloads(
        first,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=4,
        tile_delta=_delta(
            upserts=first,
            targets={tile: _target(tile) for tile in range(4)},
            base_revision=0,
        ),
    )

    _uvs, stats = pool.update_payloads(
        mixed,
        tile_shape=(4, 4),
        dirty_tiles=(0,),
        rgb_already_windowed=False,
        reserve_count=4,
        tile_delta=_delta(upserts={0: successors[0]}, targets=targets, base_revision=1),
    )

    assert stats.presented_tiles == (0,)
    assert stats.presented_identities == {0: successors[0].tile_identity}
    assert set(pool.tile_resident_keys) == {0}
    _assert_truth_record(targets, stats.presented_identities, mixed)
