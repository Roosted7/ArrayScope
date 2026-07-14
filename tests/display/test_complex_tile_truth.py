from __future__ import annotations

from dataclasses import replace

import numpy as np

from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.display.backends.vispy.tiles import TextureAtlasPool
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
    TiledValueSource,
)
from arrayscope.display.model.tile_identity import (
    TileIdentity,
    TileLodIdentity,
    TilePresentationIdentity,
    array_plane_identities,
    complex_mapping_identity,
    tile_truth_record,
)
from arrayscope.display.montage import MontageTileState, make_montage_plan
from arrayscope.display.tile_truth_overlay import tile_truth_overlay_text
from arrayscope.display.shader_mapping import (
    ShaderComponent,
    ShaderDisplayMode,
    ShaderMapping,
    TexturePlaneKind,
    cpu_display_rgba,
    pack_texture_data,
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


def _adversarial_complex_planes() -> tuple[tuple[str, np.ndarray], ...]:
    yy, xx = np.mgrid[:4, :4]
    phase_ramp = np.linspace(-np.pi, np.pi, 16, dtype=np.float32).reshape(4, 4)
    magnitude_ramp = np.linspace(0.25, 4.0, 16, dtype=np.float32).reshape(4, 4)
    signature = (600.0 + yy * 10.0 + xx) + 1j * (-600.0 - yy * 10.0 - xx)
    return (
        ("constant-magnitude-phase-ramp", np.asarray(2.0 * np.exp(1j * phase_ramp), dtype=np.complex64)),
        (
            "constant-phase-magnitude-ramp",
            np.asarray(magnitude_ramp * np.exp(1j * np.float32(np.pi / 4.0)), dtype=np.complex64),
        ),
        ("real-only", np.asarray(30.0 + yy * 4.0 + xx, dtype=np.complex64)),
        ("imaginary-only", np.asarray(1j * (40.0 + yy * 4.0 + xx), dtype=np.complex64)),
        ("zeros", np.zeros((4, 4), dtype=np.complex64)),
        ("source-signature", np.asarray(signature, dtype=np.complex64)),
    )


def _adversarial_payloads(*, shader_display: bool) -> dict[int, DisplayTilePayload]:
    payloads = {}
    for tile_number, (_name, plane) in enumerate(_adversarial_complex_planes()):
        rgb, magnitude = complex_to_rgb(plane)
        kind = TexturePlaneKind.COMPLEX_RG32F if shader_display else TexturePlaneKind.RGB8
        physical = plane if shader_display else rgb
        identity = replace(
            _target(tile_number),
            texture_kind=kind,
            real_plane=array_plane_identities(physical)[0],
            imag_plane=array_plane_identities(physical)[1],
        )
        payloads[tile_number] = DisplayTilePayload(
            tile_number=tile_number,
            source_index=tile_number,
            image=rgb,
            histogram_data=magnitude,
            source_id=("adversarial-complex", tile_number),
            texture_data=physical,
            texture_kind=kind,
            semantic_data=plane,
            semantic_histogram_data=magnitude,
            shader_mapping=_MAPPING,
            tile_identity=identity,
            presentation_identity=TilePresentationIdentity(
                levels_generation=17,
                levels=(0.0, 900.0),
                scale="linear",
                lut_identity="phase",
            ),
        )
    return payloads


def _adversarial_geometry() -> DisplayGeometry:
    source_indices = tuple(range(len(_adversarial_complex_planes())))
    state = ViewState.from_shape((4, 4, len(source_indices))).with_channel(ChannelMode.COMPLEX)
    state = state.with_montage_axis(2, columns=3, indices=source_indices, text=":")
    return DisplayGeometry(
        view_state=state,
        display_shape=(9, 14),
        montage=MontageGeometry(
            indices=source_indices,
            tile_shape=(4, 4),
            columns=3,
            rows=2,
            gap=1,
        ),
        montage_tile_states=(MontageTileState.LOADED,) * len(source_indices),
    )


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


def _payload(
    tile_number: int,
    source_index: int,
    *,
    shader_display: bool = True,
) -> DisplayTilePayload:
    plane = _complex_plane(source_index)
    rgb, magnitude = complex_to_rgb(plane)
    kind = TexturePlaneKind.COMPLEX_RG32F if shader_display else TexturePlaneKind.RGB8
    physical = plane if shader_display else rgb
    real_plane, imag_plane = array_plane_identities(physical)
    identity = replace(
        _target(source_index),
        texture_kind=kind,
        real_plane=real_plane,
        imag_plane=imag_plane,
    )
    return DisplayTilePayload(
        tile_number=tile_number,
        source_index=source_index,
        image=rgb,
        histogram_data=magnitude,
        source_id=("synthetic-complex", source_index),
        texture_data=physical,
        texture_kind=kind,
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


def _adversarial_delta(payloads: dict[int, DisplayTilePayload]) -> TilePresentationDelta:
    tiles = tuple(sorted(payloads))
    return TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=17,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        target_revision=1,
        upserts=payloads,
        active_tiles=tiles,
        planned_tiles=tiles,
        target_identities={tile: payloads[tile].tile_identity for tile in tiles},
    )


def _transition_fixture(*, shader_display: bool):
    kind = TexturePlaneKind.COMPLEX_RG32F if shader_display else TexturePlaneKind.RGB8
    first = {
        tile: _payload(tile, tile, shader_display=shader_display)
        for tile in range(4)
    }
    successors = {
        tile: _payload(tile, tile + 3, shader_display=shader_display)
        for tile in range(4)
    }
    targets = {
        tile: replace(_target(tile + 3), texture_kind=kind)
        for tile in range(4)
    }
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


def test_adversarial_complex_fixture_preserves_native_values_and_source_signatures():
    patterns = dict(_adversarial_complex_planes())

    np.testing.assert_allclose(np.abs(patterns["constant-magnitude-phase-ramp"]), 2.0, rtol=1e-6)
    assert np.unique(np.round(np.angle(patterns["constant-magnitude-phase-ramp"]), 5)).size >= 15
    np.testing.assert_allclose(
        np.angle(patterns["constant-phase-magnitude-ramp"]),
        np.pi / 4.0,
        rtol=1e-6,
    )
    assert np.unique(np.abs(patterns["constant-phase-magnitude-ramp"])).size == 16
    np.testing.assert_array_equal(np.imag(patterns["real-only"]), 0.0)
    np.testing.assert_array_equal(np.real(patterns["imaginary-only"]), 0.0)
    np.testing.assert_array_equal(patterns["zeros"], 0.0)
    assert patterns["source-signature"][2, 3] == np.complex64(623.0 - 623.0j)

    payloads = _adversarial_payloads(shader_display=True)
    values = TiledValueSource(payloads)
    mapping = type("Mapping", (), {"tile_number": 5, "local_y": 2, "local_x": 3})()
    assert values.value_at(mapping) == np.abs(np.complex64(623.0 - 623.0j))
    semantic, magnitude, source = values.tile_region(
        type("Tile", (), {"montage_index": 5})(),
        (slice(2, 3), slice(3, 4)),
    )
    np.testing.assert_array_equal(semantic, np.asarray([[623.0 - 623.0j]], dtype=np.complex64))
    np.testing.assert_allclose(magnitude, np.abs(semantic), rtol=1e-6)
    assert source == "committed_tile_payload"


def test_pyqtgraph_adversarial_complex_fixture_draws_cpu_reference(qt_app):
    payloads = _adversarial_payloads(shader_display=False)
    levels = (0.0, 900.0)
    view = ImageView2D()
    try:
        report = view.setTiledPresentation(
            geometry=_adversarial_geometry(),
            tile_state=TilePresentationState(payloads, revision=1),
            tile_delta=_adversarial_delta(payloads),
            histogramPlotData=None,
            levels=levels,
            histogramRange=levels,
        )

        assert report.presented_tiles == frozenset(payloads)
        for tile_number, payload in payloads.items():
            state = view._montage_tile_layer.states[tile_number]
            assert state.acknowledged_identity == payload.tile_identity
            assert payload.texture_kind == TexturePlaneKind.RGB8
            expected = rgb_display_for_levels(payload.image, payload.histogram_data, levels)
            np.testing.assert_array_equal(np.asarray(state.item.image), expected)
            if tile_number == 4:
                assert not np.any(np.asarray(state.item.image))
    finally:
        view.close()


def test_vispy_adversarial_complex_fixture_uploads_cpu_reference_planes():
    payloads = _adversarial_payloads(shader_display=True)
    pool = TextureAtlasPool(_Gloo(), max_texture_size=64)

    _uvs, stats = pool.update_payloads(
        payloads,
        tile_shape=(4, 4),
        dirty_tiles=None,
        rgb_already_windowed=False,
        reserve_count=len(payloads),
        tile_delta=_adversarial_delta(payloads),
    )

    assert stats.presented_tiles == tuple(payloads)
    assert stats.presented_identities == {
        tile: payload.tile_identity for tile, payload in payloads.items()
    }
    assert len(pool.scalar_texture.updates) == len(payloads)
    for (uploaded, _offset, copy), payload in zip(pool.scalar_texture.updates, payloads.values(), strict=True):
        assert copy
        assert payload.texture_kind == TexturePlaneKind.COMPLEX_RG32F
        np.testing.assert_array_equal(uploaded, pack_texture_data(payload.semantic_data, payload.texture_kind))
        reference = cpu_display_rgba(
            payload.semantic_data,
            replace(payload.shader_mapping, levels=(0.0, 900.0)),
        )
        assert reference.shape == (4, 4, 4)
        assert np.all(reference[..., 3] == 255)
        if payload.source_index == 4:
            assert not np.any(uploaded)
            assert not np.any(reference[..., :3])


def _assert_truth_record(
    targets,
    acknowledged,
    payloads,
    *,
    expected_texture_kind: TexturePlaneKind,
):
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
    assert rows[0]["texture_kind"] == expected_texture_kind.value
    assert rows[0]["real_plane_identity"] is not None
    assert rows[0]["real_plane_identity"]["pointer"] > 0
    if expected_texture_kind == TexturePlaneKind.COMPLEX_RG32F:
        assert rows[0]["imag_plane_identity"] is not None
        assert rows[0]["imag_plane_identity"]["pointer"] > 0
    else:
        assert rows[0]["imag_plane_identity"] is None
    assert rows[0]["complex_mapping"] == ("phase_color", "abs", "mapped")
    assert rows[0]["lod"] == {"level": 0, "factor": 1, "gutter": 0}
    assert rows[0]["levels_generation"] == 9
    assert rows[0]["target_source"] == 3
    assert rows[0]["acknowledged_source"] == 3
    assert rows[0]["target_texture_kind"] == expected_texture_kind.value
    assert rows[0]["acknowledged_texture_kind"] == expected_texture_kind.value
    assert rows[0]["target_channel"] == "complex"
    assert rows[0]["acknowledged_channel"] == "complex"
    assert rows[0]["target_lod"] == {"level": 0, "factor": 1, "gutter": 0}
    assert rows[0]["acknowledged_lod"] == {"level": 0, "factor": 1, "gutter": 0}
    assert all(row["target_identity"] is not None for row in rows)
    assert all(row["placeholder"] is True for row in rows[1:])
    overlay = tile_truth_overlay_text(rows)
    assert "slot 0  DRAW" in overlay
    assert "src 3 -> 3" in overlay
    assert (
        f"tex {expected_texture_kind.value} -> {expected_texture_kind.value}"
        in overlay
    )
    assert "planes r 0x" in overlay
    assert "complex  phase_color/abs/mapped" in overlay
    assert "lod 0 -> 0" in overlay
    assert "levels 9" in overlay


def test_pyqtgraph_complex_semantic_transition_hides_unacknowledged_tiles(qt_app):
    first, successors, targets, mixed = _transition_fixture(shader_display=False)
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
        physical = view.tileTruthPhysicalRows()
        assert physical[0]["physical_texture_kind"] == "rgb8"
        assert physical[0]["physical_mapping_mode"] == "cpu_rgb"
        assert physical[0]["physical_acknowledged_identity"] == successors[0].tile_identity
        assert (
            physical[0]["physical_acknowledged_identity"].texture_kind.value
            == physical[0]["physical_texture_kind"]
        )
        _assert_truth_record(
            targets,
            report.presented_identities,
            mixed,
            expected_texture_kind=TexturePlaneKind.RGB8,
        )
    finally:
        view.close()


def test_vispy_complex_semantic_transition_hides_unacknowledged_tiles():
    first, successors, targets, mixed = _transition_fixture(shader_display=True)
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
    physical = pool.tile_truth_physical_rows()
    assert physical[0]["physical_texture_kind"] == "complex_rg32f"
    assert physical[0]["physical_storage_mode"] == "complex"
    assert physical[0]["physical_texture_shape"] == (4, 4, 2)
    assert physical[0]["physical_acknowledged_identity"] == successors[0].tile_identity
    _assert_truth_record(
        targets,
        stats.presented_identities,
        mixed,
        expected_texture_kind=TexturePlaneKind.COMPLEX_RG32F,
    )
