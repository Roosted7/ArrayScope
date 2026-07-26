"""Targeted wgpu live-view tests: montage acks + complex/RGB commit modes.

Offscreen, adapter-skip pattern (mirrors ``test_imagesurface_contract``):
these pin the queue row 3(b) montage/complex slice — physical-truth per-tile
acknowledgement, content-keyed zero-upload behavior across montage scrolls
and mode/levels switches, the phase LUT, RGB display-ready bytes, and the
loud rejections at the honest scope boundary.
"""

import contextlib
import os
from dataclasses import replace

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.display.test_imageview2d import _present_tiled, _view_class


def _wgpu_adapter_available() -> bool:
    try:
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        with contextlib.suppress(RuntimeError):  # instance already exists
            # Vulkan-only instance BEFORE the first adapter request: letting
            # the probe create an all-backends instance makes GL adapter
            # enumeration can re-init EGL in workers that hold other live GPU
            # state (gate-B Tier 0; full-suite crash 2026-07-18).
            set_instance_extras(backends=["Vulkan"])
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _wgpu_adapter_available(), reason="no wgpu adapter on this machine"
)


def test_wgpu_camera_draw_publishes_presentation_ack(qt_app, monkeypatch):
    view = _view_class("wgpu")(present_method="screen")
    requests = []
    acknowledgements = []
    try:
        monkeypatch.setattr(
            view._wgpu_canvas,
            "request_draw",
            lambda *args, **kwargs: requests.append((args, kwargs)),
        )
        view.presentationDrawn.connect(lambda: acknowledgements.append("drawn"))

        view._request_wgpu_canvas_draw()

        assert requests == [((), {})]
        assert view.presentationDrawPending() is True
        view._wgpu_canvas_update_pending = False
        view._publish_wgpu_draw_ack(0)
        assert acknowledgements == ["drawn"]
        assert view.presentationDrawPending() is False
    finally:
        view.close()


def test_wgpu_pool_headroom_clamps_to_device_limit_but_active_pages_do_not():
    from arrayscope.display.wgpu_imageview2d import _wgpu_pool_layer_budget

    assert (
        _wgpu_pool_layer_budget(
            previous=0,
            needed=272,
            preferred=2084,
            max_layers=2048,
        )
        == 2048
    )
    with pytest.raises(RuntimeError, match=r"needed=2049, max_layers=2048"):
        _wgpu_pool_layer_budget(previous=0, needed=2049, max_layers=2048)


def test_wgpu_pool_retention_capacity_caps_headroom_instead_of_filling_policy():
    from arrayscope.display.wgpu_imageview2d import _wgpu_pool_layer_budget
    from arrayscope.gpu.wgpu_executor import PAGE

    budget = 256 * 1024 * 1024
    assert (
        _wgpu_pool_layer_budget(
            previous=0,
            needed=200,
            preferred=300,
            max_layers=2048,
            budget_bytes=budget,
            bytes_per_layer=PAGE * PAGE * 4,
        )
        == 408
    )
    assert (
        _wgpu_pool_layer_budget(
            previous=0,
            needed=200,
            preferred=300,
            max_layers=2048,
            budget_bytes=budget,
            bytes_per_layer=PAGE * PAGE * 8,
        )
        == 408
    )


def test_wgpu_pool_byte_policy_is_shared_across_representations():
    from arrayscope.display.wgpu_imageview2d import (
        _WGPU_POOL_TEXEL_BYTES,
        _wgpu_pool_layer_budget,
        _wgpu_representation_byte_budgets,
    )
    from arrayscope.gpu.wgpu_executor import (
        COMPLEX_RG32F,
        PAGE,
        RGB8,
        RGB_WINDOWED_RGBA32F,
        SCALAR_R32F,
    )

    representations = (SCALAR_R32F, COMPLEX_RG32F, RGB8, RGB_WINDOWED_RGBA32F)
    budget = 256 * 1024 * 1024
    shares = _wgpu_representation_byte_budgets(
        required_pages=dict.fromkeys(representations, 100),
        preferred_pages={},
        previous_pages={},
        budget_bytes=budget,
    )
    layers = {
        representation: _wgpu_pool_layer_budget(
            previous=0,
            needed=100,
            max_layers=2048,
            budget_bytes=shares[representation],
            bytes_per_layer=PAGE * PAGE * _WGPU_POOL_TEXEL_BYTES[representation],
        )
        for representation in representations
    }

    assert sum(shares.values()) == budget
    assert (
        sum(
            layers[representation] * PAGE * PAGE * _WGPU_POOL_TEXEL_BYTES[representation]
            for representation in representations
        )
        <= budget
    )


def test_wgpu_pool_capacity_counts_reusable_native_pages_not_preview_textures():
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor
    from arrayscope.display.wgpu_imageview2d import (
        _wgpu_payload_capacity_page_count,
        _wgpu_pool_layer_budget,
    )
    from arrayscope.gpu.wgpu_executor import PAGE, SCALAR_R32F

    source = np.arange(336 * 336, dtype=np.float32).reshape(336, 336)
    preview = source.reshape(21, 16, 21, 16).mean(axis=(1, 3))
    payload = DisplayTilePayload(
        0,
        0,
        preview,
        None,
        ("pool-capacity", 0),
        lod=LodInfo(
            level=4,
            factor=16,
            source_shape=source.shape,
            texture_shape=preview.shape,
        ),
        source_anchor=PayloadSourceAnchor(
            ("pool-capacity", 0),
            (0, 336, 0, 336),
            plane_shape=source.shape,
        ),
        native_residency_data=source,
    )

    pages_per_tile = _wgpu_payload_capacity_page_count(
        payload,
        texture_shape=preview.shape,
        representation=SCALAR_R32F,
        selected_lod=4,
    )
    assert pages_per_tile == 4
    assert (
        _wgpu_pool_layer_budget(
            previous=0,
            needed=272 * pages_per_tile,
            preferred=272,
            max_layers=2048,
            budget_bytes=256 * 1024 * 1024,
            bytes_per_layer=PAGE * PAGE * 4,
        )
        == 1088
    )


def test_wgpu_progressive_pre_reservation_is_bounded_by_plan_and_device():
    from arrayscope.display.wgpu_imageview2d import _wgpu_pre_reservation_page_count

    sparse_capacity = {0: 4, 1: 2}
    planned_count = 272
    per_tile_max = max(sparse_capacity.values())

    reserved = _wgpu_pre_reservation_page_count(
        sparse_capacity,
        planned_count=planned_count,
        max_layers=2048,
    )
    assert reserved == planned_count * per_tile_max
    assert reserved <= planned_count * per_tile_max
    assert reserved <= 2048

    device_capped = _wgpu_pre_reservation_page_count(
        sparse_capacity,
        planned_count=1024,
        max_layers=2048,
    )
    assert device_capped == 2048
    assert device_capped <= 1024 * per_tile_max

    with pytest.raises(RuntimeError, match=r"needed=2049, max_layers=2048"):
        _wgpu_pre_reservation_page_count(
            {0: 2049},
            planned_count=1,
            max_layers=2048,
        )


def _montage_geometry(tile_shape, columns, rows, *, loaded, gap=0):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    tile_h, tile_w = tile_shape
    height = rows * tile_h + max(0, rows - 1) * gap
    width = columns * tile_w + max(0, columns - 1) * gap
    return DisplayGeometry(
        view_state=ViewState.from_shape((height, width)).with_image_axes(0, 1),
        display_shape=(height, width),
        montage=MontageGeometry(
            indices=tuple(range(loaded)),
            tile_shape=(tile_h, tile_w),
            columns=columns,
            rows=rows,
            gap=gap,
        ),
        montage_tile_states=tuple([MontageTileState.LOADED] * loaded),
    )


def _payload(
    tile_number,
    image,
    *,
    source_id,
    shader_mapping=None,
    texture_kind=None,
    histogram_data=None,
    level_stats=None,
):
    from arrayscope.display.model.frame import DisplayTilePayload

    return DisplayTilePayload(
        tile_number,
        tile_number,
        image,
        histogram_data,
        source_id,
        shader_mapping=shader_mapping,
        texture_kind=texture_kind,
        level_stats=level_stats,
    )


def _typed_scalar_payload(
    tile_number,
    image,
    *,
    semantic_generation=("typed", 0),
    lod_level=0,
    shader_mapping=None,
    quality="exact",
    source_anchor=None,
    presentation_generation=1,
):
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.tile_identity import (
        TileIdentity,
        TileLodIdentity,
        TilePresentationIdentity,
        array_plane_identities,
        complex_mapping_identity,
    )
    from arrayscope.display.shader_mapping import TexturePlaneKind

    image = np.asarray(image)
    factor = 1 << int(lod_level)
    real_plane, imag_plane = array_plane_identities(image)
    identity = TileIdentity(
        document_generation=("typed-doc", 1),
        operation_key=("typed-op",),
        source_index=tile_number,
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel="real",
        complex_mapping=complex_mapping_identity(shader_mapping),
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_generation=semantic_generation,
        lod=TileLodIdentity(level=lod_level, factor=factor),
        quality=quality,
        real_plane=real_plane,
        imag_plane=imag_plane,
    )
    return DisplayTilePayload(
        tile_number,
        tile_number,
        image,
        None,
        ("typed", tile_number, semantic_generation, lod_level),
        lod=LodInfo(
            level=lod_level,
            factor=factor,
            source_shape=tuple(int(value) for value in image.shape[:2]),
            texture_shape=tuple(int(value) for value in image.shape[:2]),
        ),
        shader_mapping=shader_mapping,
        quality=quality,
        tile_identity=identity,
        presentation_identity=TilePresentationIdentity(
            levels_generation=presentation_generation,
            levels=(0.0, 1.0),
        ),
        source_anchor=source_anchor,
    )


def _lod_payload(
    tile_number,
    image,
    *,
    base_source_id,
    level,
    source_shape,
    payload_source_shape=None,
    factor=None,
):
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    factor = 1 << int(level) if factor is None else int(factor)
    lod = LodInfo(
        level=level,
        factor=factor,
        source_shape=source_shape,
        texture_shape=image.shape[:2],
    )
    identity = TileIdentity(
        document_generation="lod-doc",
        operation_key="lod-op",
        source_index=tile_number,
        image_axes=(0, 1),
        axis_flips=(False, False),
        channel="real",
        complex_mapping=None,
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        semantic_generation="lod-semantic",
        lod=TileLodIdentity(level=level, factor=factor),
    )
    return DisplayTilePayload(
        tile_number,
        tile_number,
        image,
        None,
        (
            base_source_id,
            "texture_kind",
            TexturePlaneKind.SCALAR_R32F.value,
            "lod",
            factor,
            level,
            0,
        ),
        source_shape=payload_source_shape or source_shape,
        lod=lod,
        tile_identity=identity,
    )


def _shown_view(qt_app, *, texture_codec="auto"):
    view = _view_class("wgpu")(texture_codec=texture_codec)
    view.resize(320, 260)
    view.show()
    return view


def _commit(view, geometry, payloads, *, levels, rgb_already_windowed=False):
    canvas = np.zeros(geometry.display_shape, dtype=np.float32)
    return _present_tiled(
        view,
        canvas,
        geometry=geometry,
        levels=levels,
        histogramRange=levels,
        montage_tile_payloads=payloads,
        rgb_already_windowed=rgb_already_windowed,
    )


def _rerender_internal(view):
    """Re-present the committed tiles to the executor's offscreen target."""

    from arrayscope.gpu.command_protocol import UpdateTileInstances

    camera = view._wgpu_camera_command()
    view._submit_wgpu((camera, UpdateTileInstances(view._wgpu_tile_instances())))


def _center_pixel(view):
    target = view._wgpu_executor.read_target()
    h, w = target.shape[:2]
    return target[h // 2, w // 2]


def test_mapping_only_commit_reuses_typed_binding_and_updates_levels(qt_app):
    """Fresh level-bearing wrappers keep one physical TileIdentity binding."""

    geometry = _montage_geometry((32, 32), 1, 1, loaded=1)
    image = np.full((32, 32), 0.2, dtype=np.float32)
    first = _typed_scalar_payload(0, image, presentation_generation=1)
    second = replace(
        first,
        presentation_identity=replace(
            first.presentation_identity,
            levels_generation=2,
            levels=(0.0, 0.25),
        ),
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        full = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: first},
        )
        fast = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 0.25),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=(),
            montage_tile_payloads={0: second},
        )
        _rerender_internal(view)

        assert full.binding_full_republications == 1
        assert fast.binding_fast_path_commits == 1
        assert fast.binding_full_republications == 0
        assert fast.resident_rebinds == 0
        assert fast.texture_uploads == 0
        assert np.allclose(_center_pixel(view), (204, 204, 204, 255), atol=2)
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_binding_fast_path_commits"] == 1
        assert diagnostics["wgpu_binding_full_republications"] == 1
    finally:
        view.close()


def test_mapping_only_commit_applies_lut_without_rebinding(qt_app):
    """Colormap changes are uniform/LUT state, not a tile-binding change."""

    from arrayscope.display.shader_mapping import ShaderMapping

    red = np.zeros((256, 3), dtype=np.uint8)
    red[:, 0] = 255
    green = np.zeros((256, 3), dtype=np.uint8)
    green[:, 1] = 255
    geometry = _montage_geometry((32, 32), 1, 1, loaded=1)
    image = np.full((32, 32), 0.5, dtype=np.float32)
    first = _typed_scalar_payload(0, image, shader_mapping=ShaderMapping(lut_data=red))
    second = replace(
        first,
        shader_mapping=ShaderMapping(lut_data=green),
        presentation_identity=replace(first.presentation_identity, levels_generation=2),
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: first},
        )
        report = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=(),
            montage_tile_payloads={0: second},
        )
        _rerender_internal(view)

        assert report.binding_fast_path_commits == 1
        assert report.texture_uploads == 0
        assert np.allclose(_center_pixel(view), (0, 255, 0, 255), atol=2)
    finally:
        view.close()


def test_unchanged_upsert_does_not_cross_mapping_only_boundary(qt_app):
    """Target acknowledgement stays on the full path until the ADR ladder owns it."""

    geometry = _montage_geometry((32, 32), 1, 1, loaded=1)
    payload = _typed_scalar_payload(0, np.full((32, 32), 0.2, dtype=np.float32))
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: payload},
        )
        report = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: payload},
        )

        assert report.binding_fast_path_commits == 0
        assert report.binding_full_republications == 1
    finally:
        view.close()


def test_forced_mapping_only_bypass_turns_framebuffer_oracle_red(qt_app, monkeypatch):
    """Fault injection: changed plane bytes must not pass as mapping-only."""

    geometry = _montage_geometry((32, 32), 1, 1, loaded=1)
    predecessor = _typed_scalar_payload(
        0,
        np.full((32, 32), 0.2, dtype=np.float32),
    )
    successor = _typed_scalar_payload(
        0,
        np.full((32, 32), 0.8, dtype=np.float32),
        presentation_generation=2,
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: predecessor},
        )
        monkeypatch.setattr(view, "_wgpu_tiled_binding_reusable", lambda **_kwargs: True)
        report = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: successor},
        )
        _rerender_internal(view)

        assert report.binding_fast_path_commits == 1
        actual = _center_pixel(view)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(actual, (204, 204, 204, 255), atol=2)
        assert np.allclose(actual, (51, 51, 51, 255), atol=2)
    finally:
        view.close()


def test_physical_tile_identity_covers_every_binding_dimension():
    from arrayscope.display.model.tile_identity import TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind
    from arrayscope.display.wgpu_imageview2d import _wgpu_physical_payload_identities

    image = np.full((8, 8), 0.2, dtype=np.float32)
    payload = _typed_scalar_payload(0, image)
    identity = payload.tile_identity
    baseline = _wgpu_physical_payload_identities({0: payload})

    changes = (
        replace(identity, lod=TileLodIdentity(level=1, factor=2)),
        replace(identity, texture_kind=TexturePlaneKind.COMPLEX_RG32F),
        replace(identity, complex_mapping=("phase_color", "abs", "mapped")),
        replace(identity, semantic_generation=("crop-window", 1)),
        replace(identity, document_generation=("atomic-successor", 2)),
        _typed_scalar_payload(0, image.copy()).tile_identity,
    )
    for changed in changes:
        assert (
            _wgpu_physical_payload_identities({0: replace(payload, tile_identity=changed)})
            != baseline
        )


def test_layout_identity_covers_shift_and_transpose():
    from arrayscope.display.wgpu_imageview2d import (
        _display_axes_transposed,
        _wgpu_layout_identity,
    )

    baseline = _montage_geometry((8, 10), 2, 1, loaded=2, gap=1)
    shifted = _montage_geometry((8, 10), 1, 2, loaded=2, gap=1)
    transposed = replace(
        baseline,
        view_state=baseline.view_state.with_image_axes(1, 0),
    )

    assert _wgpu_layout_identity(baseline) != _wgpu_layout_identity(shifted)
    assert _display_axes_transposed(baseline) is False
    assert _display_axes_transposed(transposed) is True


def test_page_eviction_readmission_and_remap_force_full_republication(qt_app):
    geometry = _montage_geometry((32, 32), 1, 1, loaded=1)
    payload = _typed_scalar_payload(0, np.full((32, 32), 0.2, dtype=np.float32))
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_tile_payloads={0: payload},
        )
        table = view._wgpu_executor.page_table
        key = table.resident_keys()[0]
        entry = table._entries[key]
        generation = table.generation
        slot = table.unbind(key)
        table.bind(key, slot, nbytes=entry.nbytes, pinned=entry.pinned)
        assert table.generation == generation + 2

        report = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=(),
            montage_tile_payloads={0: payload},
        )
        assert report.binding_fast_path_commits == 0
        assert report.binding_full_republications == 1

        generation = table.generation
        table.remap_slots(lambda current: current)
        assert table.generation == generation + 1
        report = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=(),
            montage_tile_payloads={0: payload},
        )
        assert report.binding_fast_path_commits == 0
        assert report.binding_full_republications == 1
    finally:
        view.close()


def test_wgpu_physical_rows_report_resident_draw_geometry(qt_app):
    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((8, 10), 2, 1, loaded=2, gap=1)
        payloads = {
            0: _payload(0, np.ones((8, 10), dtype=np.float32), source_id=("tile", 0)),
            1: _payload(1, np.ones((8, 10), dtype=np.float32), source_id=("tile", 1)),
        }

        _commit(view, geometry, payloads, levels=(0.0, 1.0))

        rows = view.tileTruthPhysicalRows()
        assert set(rows) == {0, 1}
        assert rows[0]["physical_draw_world_bounds"] == (0.0, 0.0, 10.0, 8.0)
        assert rows[1]["physical_draw_world_bounds"] == (11.0, 0.0, 21.0, 8.0)
        assert rows[0]["physical_draw_bounds_match_layout"] is True
        assert rows[0]["physical_storage_mode"] == "wgpu_page_table"
        assert rows[0]["physical_acknowledged_identity"] is not None
        bindings = rows[0]["physical_page_bindings"]
        expected_quality = (
            "lossy_compressed"
            if any(view._wgpu_executor.page_is_compressed(row["actual_key"]) for row in bindings)
            else "exact"
        )
        assert rows[0]["physical_quality"] == expected_quality
        assert all(
            row["quality"] != "exact"
            for row in bindings
            if view._wgpu_executor.page_is_compressed(row["actual_key"])
        )
        assert view.physicalVisibleTileCount() == 2
        assert view.wgpuPresentationDiagnostics()["physically_visible_tile_count"] == 2
    finally:
        view.close()


def _green_overlay_mask(target):
    pixels = np.asarray(target, dtype=np.int16)
    return (
        (pixels[..., 1] > 150)
        & (pixels[..., 1] > pixels[..., 0] + 45)
        & (pixels[..., 1] > pixels[..., 2] + 45)
    )


def _orange_overlay_mask(target):
    pixels = np.asarray(target, dtype=np.int16)
    return (pixels[..., 0] > 150) & (pixels[..., 0] > pixels[..., 1] + 45) & (pixels[..., 2] < 120)


def _cyan_sampling_mask(target):
    pixels = np.asarray(target, dtype=np.int16)
    return (
        (pixels[..., 0] > 45)
        & (pixels[..., 1] > 160)
        & (pixels[..., 2] > 200)
        & (pixels[..., 2] > pixels[..., 0] + 100)
    )


def _mask_center(mask):
    rows, columns = np.nonzero(mask)
    assert len(rows), "expected physical pixels for this mask"
    return (float(columns.mean()), float(rows.mean()))


def test_roi_and_profile_marker_are_executor_pixels_and_clear(qt_app, qtbot):
    """Thomas's 2026-07-18 dogfood report: both overlays were invisible.

    The oracle reads the executor target, not the QWidget backing store, so a
    QGraphics mirror cannot satisfy it (nor can a bookkeeping-only hook).
    """

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import InteractionTarget

    view = _shown_view(qt_app)
    try:
        image = np.full((64, 64), 0.02, dtype=np.float32)
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="overlay-pixel-oracle")},
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        roi = view.createRoi(
            RoiKind.RECTANGLE,
            rect=(10.0, 12.0, 20.0, 18.0),
            color=(40, 220, 80),
        )
        view.setProfileMarker(46.0, 42.0, visible=True)
        draws_before = int(view._wgpu_draw_count)
        view._wgpu_canvas_update_pending = False
        view._request_wgpu_canvas_draw()
        qtbot.waitUntil(
            lambda: int(view._wgpu_draw_count) > draws_before,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        assert view._wgpu_last_draw_error == ""
        _rerender_internal(view)

        target = view._wgpu_executor.read_target()
        base_green = np.count_nonzero(_green_overlay_mask(target))
        base_orange = np.count_nonzero(_orange_overlay_mask(target))
        assert base_green > 100
        assert base_orange > 100

        state = view.interaction_controller.set_hover(
            InteractionTarget(
                "roi",
                object_id=roi.id,
                part="handle",
                geometry_kind="rectangle",
                handle_index=0,
            ),
            point=(10.0, 12.0),
        )
        view.sync_interaction_state(state)
        _rerender_internal(view)
        assert np.count_nonzero(_green_overlay_mask(view._wgpu_executor.read_target())) > base_green
        view.highlightRoi(roi.id)
        _rerender_internal(view)

        state = view.interaction_controller.set_hover(
            InteractionTarget("profile", part="center"),
            point=(46.0, 42.0),
        )
        view.sync_interaction_state(state)
        _rerender_internal(view)
        assert (
            np.count_nonzero(_orange_overlay_mask(view._wgpu_executor.read_target())) > base_orange
        )

        overlay_writes = view._wgpu_executor.overlay_buffer_writes_total
        view.getView().setRange(xRange=(64, 128), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        assert not np.any(_orange_overlay_mask(view._wgpu_executor.read_target()))
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        assert np.count_nonzero(_orange_overlay_mask(view._wgpu_executor.read_target())) > 100
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes

        assert view.removeRoi(roi.id)
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        assert not np.any(_green_overlay_mask(target))
        assert np.count_nonzero(_orange_overlay_mask(target)) > 100

        view.hideProfileMarker()
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        assert not np.any(_orange_overlay_mask(target))
    finally:
        view.close()


def test_middle_sample_marker_is_fixed_screen_executor_pixels(qt_app):
    """wgpu screen mode composites the shared Qt marker into native frame pixels."""

    view = _view_class("wgpu")()
    view.resize(320, 260)
    view.show()
    try:
        image = np.full((64, 64), 0.02, dtype=np.float32)
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="sampling-marker-pixel-oracle")},
            levels=(0.0, 1.0),
        )
        if view.wgpuPresentMethod() != "screen":
            # Offscreen Qt cannot create the Wayland surface. Keep this test's
            # executor-pixel oracle useful there by exercising the same
            # compositor after the bitmap commit established rendering state.
            view._wgpu_canvas_update_pending = True
            view._wgpu_present_method = "screen"
            view._wgpu_chip_compositor.register(view._sampling_marker)
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        viewport = view.graphicsView.viewport()
        sample_scene = view.graphicsView.mapToScene(viewport.width() // 2, viewport.height() // 2)
        view._set_middle_sampling_marker(sample_scene)
        view._sync_wgpu_overlay_geometry()
        _rerender_internal(view)

        target = view._wgpu_executor.read_target()
        marker = _cyan_sampling_mask(target)
        assert np.count_nonzero(marker) >= 12
        marker_center = _mask_center(marker)
        canvas_w, canvas_h = view._wgpu_canvas.get_physical_size()
        assert marker_center == pytest.approx((canvas_w * 0.5, canvas_h * 0.5), abs=3.0)

        view._set_middle_sampling_marker(None)
        view._sync_wgpu_overlay_geometry()
        _rerender_internal(view)
        assert not np.any(_cyan_sampling_mask(view._wgpu_executor.read_target()))
    finally:
        view.close()


def test_world_overlay_and_tile_move_together_without_overlay_reupload(qt_app):
    """A camera-only frame must rigidly move tiles and world overlays."""

    from arrayscope.core.roi import RoiKind

    view = _shown_view(qt_app)
    try:
        image = np.zeros((64, 64), dtype=np.float32)
        image[22:34, 24:38] = 1.0
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="overlay-camera-oracle")},
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.createRoi(
            RoiKind.RECTANGLE,
            rect=(24.0, 22.0, 14.0, 12.0),
            color=(40, 220, 80),
        )
        _rerender_internal(view)
        before = view._wgpu_executor.read_target()
        tile_before = _mask_center(np.all(before[..., :3] > 180, axis=-1))
        overlay_before = _mask_center(_green_overlay_mask(before))
        overlay_writes = view._wgpu_executor.overlay_buffer_writes_total

        view.getView().setRange(xRange=(4, 68), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        after = view._wgpu_executor.read_target()
        tile_after = _mask_center(np.all(after[..., :3] > 180, axis=-1))
        overlay_after = _mask_center(_green_overlay_mask(after))

        tile_shift = (tile_after[0] - tile_before[0], tile_after[1] - tile_before[1])
        overlay_shift = (
            overlay_after[0] - overlay_before[0],
            overlay_after[1] - overlay_before[1],
        )
        assert tile_shift[0] < -20.0
        assert tile_shift == pytest.approx(overlay_shift, abs=2.0)
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes

        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.getView().invertX(True)
        view.getView().invertY(False)
        _rerender_internal(view)
        mirrored = view._wgpu_executor.read_target()
        tile_mirrored = _mask_center(np.all(mirrored[..., :3] > 180, axis=-1))
        overlay_mirrored = _mask_center(_green_overlay_mask(mirrored))
        target_h, target_w = mirrored.shape[:2]
        assert tile_mirrored == pytest.approx(
            (target_w - 1 - tile_before[0], target_h - 1 - tile_before[1]),
            abs=2.0,
        )
        assert overlay_mirrored == pytest.approx(
            (target_w - 1 - overlay_before[0], target_h - 1 - overlay_before[1]),
            abs=2.0,
        )
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes
    finally:
        view.close()


def test_loading_and_skipped_tile_geometry_is_in_executor_target(qt_app):
    from arrayscope.display.overlays import MontageTileOverlay

    view = _shown_view(qt_app)
    try:
        image = np.full((32, 64), 0.02, dtype=np.float32)
        geometry = _montage_geometry((32, 32), 2, 1, loaded=2)
        _commit(
            view,
            geometry,
            {
                0: _payload(0, image[:, :32], source_id="loading-tile"),
                1: _payload(1, image[:, 32:], source_id="skipped-tile"),
            },
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 32), padding=0)
        view.setMontageTileOverlays(
            (
                MontageTileOverlay(0, 0, 32, 32, "loading", "not rendered"),
                MontageTileOverlay(32, 0, 32, 32, "skipped", "skipped"),
            )
        )
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        bright = np.all(target[..., :3] > 150, axis=-1)
        midpoint = target.shape[1] // 2
        assert np.count_nonzero(bright[:, :midpoint]) > 20
        assert np.count_nonzero(bright[:, midpoint:]) > 20

        view.clearMontageTileOverlays()
        _rerender_internal(view)
        assert not np.any(np.all(view._wgpu_executor.read_target()[..., :3] > 150, axis=-1))
    finally:
        view.close()


def _truth_text_mask(target):
    # Truth-label glyph ink is #a5f3fc (165, 243, 252); the label border
    # (#22d3ee, red 34) and tile pixels fail the red-channel bound.
    pixels = np.asarray(target, dtype=np.int16)
    return (pixels[..., 0] > 120) & (pixels[..., 1] > 180) & (pixels[..., 2] > 200)


def _truth_border_mask(target):
    # The DRAW-state label border is #22d3ee (34, 211, 238) at full alpha.
    pixels = np.asarray(target, dtype=np.int16)
    return (np.abs(pixels[..., 0] - 34) < 30) & (pixels[..., 1] > 180) & (pixels[..., 2] > 200)


def _label_anchor_px(view, target_shape, world_point):
    """Expected on-target pixel of a world-space label anchor."""

    camera = view._wgpu_camera_command()
    x0, y0, x1, y1 = camera.world_rect
    height, width = target_shape[:2]
    wx, wy = world_point
    px = (x1 - wx if camera.x_inverted else wx - x0) / (x1 - x0) * width
    py = (y1 - wy if not camera.y_inverted else wy - y0) / (y1 - y0) * height
    return px, py


def test_tile_truth_labels_are_native_glyph_pixels_pan_with_camera_and_clear(qt_app):
    """Queue row 3 text gap: truth labels are executor pixels, not QLabels.

    Offscreen GPU ring.  Red-first oracles: glyph pixels render at the
    tile's on-screen corner (derived from the shared camera command, the
    same transform that places tiles) and vanish on removal; a camera-only
    pan moves the text WITH the image with zero atlas uploads and zero
    overlay buffer rewrites; zooming out far enough hides unreadable labels.
    """

    view = _shown_view(qt_app)
    try:
        image = np.zeros((64, 64), dtype=np.float32)
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="truth-label-oracle")},
            levels=(0.0, 1.0),
        )
        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.setTileTruthOverlayRows(
            ({"tile": 0, "drawable": True, "tile_rect": (0.0, 0.0, 64.0, 64.0)},)
        )
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        assert np.count_nonzero(_truth_text_mask(target)) > 80, "a truth label is many glyph pixels"
        border_rows, border_columns = np.nonzero(_truth_border_mask(target))
        assert len(border_rows), "the label draws its state border"
        anchor_x, anchor_y = _label_anchor_px(view, target.shape, (0.0, 0.0))
        # Label box top-left sits at the anchor plus the 2 px inset.
        assert float(border_columns.min()) == pytest.approx(anchor_x + 2.0, abs=2.0)
        assert float(border_rows.min()) == pytest.approx(anchor_y + 2.0, abs=2.0)
        assert view.tileTruthOverlayText().startswith("slot 0  DRAW")

        atlas_uploads = view._wgpu_executor.glyph_atlas_uploads_total
        overlay_writes = view._wgpu_executor.overlay_buffer_writes_total

        view.getView().setRange(xRange=(-16, 48), yRange=(0, 64), padding=0)
        _rerender_internal(view)
        after = view._wgpu_executor.read_target()
        assert np.count_nonzero(_truth_text_mask(after)) > 80
        after_rows, after_columns = np.nonzero(_truth_border_mask(after))
        anchor_x, anchor_y = _label_anchor_px(view, after.shape, (0.0, 0.0))
        assert float(after_columns.min()) == pytest.approx(anchor_x + 2.0, abs=2.0)
        assert float(after_rows.min()) == pytest.approx(anchor_y + 2.0, abs=2.0)
        assert view._wgpu_executor.glyph_atlas_uploads_total == atlas_uploads, (
            "camera-only frames must never re-upload cached glyphs"
        )
        assert view._wgpu_executor.overlay_buffer_writes_total == overlay_writes, (
            "world-anchored text must ride the camera uniform, not a rewrite"
        )

        # Unreadably small tiles hide their labels (QLabel-layer parity).
        view.getView().setRange(xRange=(0, 6400), yRange=(0, 6400), padding=0)
        _rerender_internal(view)
        assert not np.any(_truth_text_mask(view._wgpu_executor.read_target()))
        assert view.tileTruthOverlayText() == ""

        view.getView().setRange(xRange=(0, 64), yRange=(0, 64), padding=0)
        view.setTileTruthOverlayRows(())
        _rerender_internal(view)
        assert not np.any(_truth_text_mask(view._wgpu_executor.read_target()))
        assert view.tileTruthOverlayText() == ""
    finally:
        view.close()


def test_montage_commit_acks_per_tile_and_scrolls_zero_upload(qt_app):
    view = _shown_view(qt_app)
    try:
        rng = np.random.default_rng(11)
        images = {name: rng.random((20, 30), dtype=np.float32) for name in ("p0", "p1", "p2")}
        geometry = _montage_geometry((20, 30), 2, 1, loaded=2)

        payloads = {
            0: _payload(0, images["p0"], source_id=("wgpu-montage", "p0")),
            1: _payload(1, images["p1"], source_id=("wgpu-montage", "p1")),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.presented_identities == {
            0: ("wgpu-montage", "p0"),
            1: ("wgpu-montage", "p1"),
        }
        assert report.texture_uploads == 2  # one 256^2 page per tile
        # Identical re-commit: content-keyed residency makes it physical no-op.
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert set(report.committed_upserts or ()) == {0, 1}
        assert report.texture_uploads == 0
        initial_pixels = view._wgpu_executor.read_target()

        # Montage scroll: p1 moves to tile 0, new content p2 enters tile 1.
        # Only the genuinely new plane uploads; p1 stays warm across rebind.
        scrolled = {
            0: _payload(0, images["p1"], source_id=("wgpu-montage", "p1")),
            1: _payload(1, images["p2"], source_id=("wgpu-montage", "p2")),
        }
        report = _commit(view, geometry, scrolled, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.texture_uploads == 1
        scrolled_pixels = view._wgpu_executor.read_target()
        assert not np.array_equal(scrolled_pixels, initial_pixels)

        # Scroll back: every plane was seen before — zero upload.
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert set(report.committed_upserts or ()) == {0, 1}
        assert report.texture_uploads == 0
        assert np.array_equal(view._wgpu_executor.read_target(), initial_pixels)
    finally:
        view.close()


def test_phase1_exposes_fenced_resident_page_histogram(qt_app):
    view = _shown_view(qt_app)
    try:
        image = np.linspace(-2.0, 5.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("g6a", 0))}
        view.setResidentHistogramEvidenceRequired(True)

        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))

        assert report.presented_tiles == frozenset({0})
        (evidence,) = view.residentHistogramEvidence(payloads)
        evidence.wait_completed()
        counts, bounds = evidence.readback.resolve()
        assert bounds == pytest.approx((-2.0, 5.0))
        assert int(counts.sum()) == image.size
        assert evidence.frontier_keys == tuple(view._wgpu_committed["tiles"][0]["page_keys"])

        view.acceptResidentHistogramEvidence((evidence.evidence_key,))
        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 0

        view.setResidentHistogramEvidenceRequired(False)
        view.setResidentHistogramEvidenceRequired(True, ("next-coverage", 2))
        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))
        assert report.presented_tiles == frozenset({0})
        assert report.texture_uploads == 0
        (next_evidence,) = view.residentHistogramEvidence(payloads)
        assert next_evidence.evidence_key != evidence.evidence_key
    finally:
        view.close()


def test_phase1_prepared_stats_skip_redundant_resident_histogram(qt_app):
    from arrayscope.display.model.montage_levels import TileLevelStats

    view = _shown_view(qt_app)
    try:
        image = np.linspace(-2.0, 5.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("prepared-histogram", 0),
                level_stats=TileLevelStats(
                    source_index=0,
                    bounds=(-2.0, 5.0),
                    sample=image[::4, ::4].reshape(-1),
                ),
            )
        }
        view.setResidentHistogramEvidenceRequired(True)
        dispatches_before = (
            0 if view._wgpu_executor is None else view._wgpu_executor.histogram_dispatches_total
        )

        report = _commit(view, geometry, payloads, levels=(-2.0, 5.0))

        assert report.presented_tiles == frozenset({0})
        assert view.residentHistogramEvidence(payloads) == ()
        assert view._wgpu_executor.histogram_dispatches_total == dispatches_before
    finally:
        view.close()


def test_tiled_commit_publishes_histogram_to_shared_widget(qt_app):
    view = _shown_view(qt_app)
    try:
        image = np.linspace(0.0, 8.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        histogram = image[::2, ::2]
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("histogram-widget", 0))}

        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 8.0),
            histogramRange=(0.0, 8.0),
            histogramPlotData=histogram,
            montage_tile_payloads=payloads,
        )

        assert view._histogram_adapter.bound_item is view.histogramImageItem
        assert np.array_equal(view.histogramImageItem.image, histogram)
        assert view.histogram.getLevels() == pytest.approx((0.0, 8.0))

        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 8.0),
            histogramRange=(0.0, 8.0),
            histogramPlotData=histogram,
            montage_tile_payloads=payloads,
        )
        assert view.lastImageUploadTiming().histogram_bytes == 0

        # ``None`` means this presentation carries no histogram update. A
        # later tile/level acknowledgement must not erase the most recently
        # published semantic histogram while its source remains current.
        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 8.0),
            histogramRange=(0.0, 8.0),
            histogramPlotData=None,
            montage_tile_payloads=payloads,
        )
        assert view.histogramPlotSource is histogram
        assert np.array_equal(view.histogramImageItem.image, histogram)
    finally:
        view.close()


def test_histogram_frontier_evicted_in_same_submission_never_aborts_commit(qt_app, monkeypatch):
    """Dogfood crash 2026-07-19: pool pressure inside one submission evicted a
    snapshotted histogram frontier page; the executor's loud KeyError then
    killed the whole commit mid-batch (ensures applied, present never ran).
    The commit must instead complete, drop that evidence spec with a loud
    bail trace, and let the normal re-queue machinery retry the evidence."""

    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One physical complex layer for two planes with evidence required:
        # tile 1's upload evicts tile 0's page after tile 0's frontier was
        # snapshotted, inside the same submission.
        small = WgpuPlaneExecutor(pool_layers={"complex_rg32f": 1}, device=_shared_wgpu_device())
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small
        view.setResidentHistogramEvidenceRequired(True)

        bail_events = []
        monkeypatch.setattr(
            "arrayscope.display.wgpu_imageview2d.emit_trace",
            lambda kind, **fields: bail_events.append((kind, fields)),
        )

        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        payloads = {
            0: _payload(
                0,
                np.full((16, 24), 3.0 + 4.0j, np.complex64),
                source_id=("hist-race", 0),
            ),
            1: _payload(
                1,
                np.full((16, 24), 6.0 + 8.0j, np.complex64),
                source_id=("hist-race", 1),
            ),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))

        # Blast radius: the commit completes with partial residency and only
        # the evicted frontier's evidence spec is dropped — loudly.
        assert set(report.presented_tiles) == {1}
        assert [
            (kind, fields.get("reason"))
            for kind, fields in bail_events
            if kind == "wgpu_histogram_queue_bail"
        ] == [("wgpu_histogram_queue_bail", "evicted_in_batch")]
        (evidence,) = view.residentHistogramEvidence(payloads)
        assert evidence.tile_number == 1
        evidence.wait_completed()
        counts, _bounds = evidence.readback.resolve()
        assert int(counts.sum()) == 16 * 24

        # The frontier shield is submission-scoped: no permanent pins.
        remaining = small.page_table.resident_keys()
        assert remaining
        assert not any(small.page_table.is_pinned(key) for key in remaining)
        assert set(small.page_table.eviction_candidates()) == set(remaining)

        # Dropped evidence retries via the normal re-queue machinery: a
        # commit that fits the pool delivers tile 0's evidence after all.
        solo_geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        report = _commit(view, solo_geometry, {0: payloads[0]}, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0}
        (retried,) = view.residentHistogramEvidence({0: payloads[0]})
        assert retried.tile_number == 0
        retried.wait_completed()
        counts, _bounds = retried.readback.resolve()
        assert int(counts.sum()) == 16 * 24
    finally:
        view.close()


def test_resident_histogram_obligation_survives_camera_only_coverage_reopen(qt_app):
    """Resident content must not dispatch/resolve again for a camera retarget."""

    view = _shown_view(qt_app)
    try:
        image = np.linspace(-2.0, 5.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("camera-histogram", 0))}
        obligation = ("content-and-mapping", 1)

        view.setResidentHistogramEvidenceRequired(True, obligation)
        _commit(view, geometry, payloads, levels=(-2.0, 5.0))
        (evidence,) = view.residentHistogramEvidence(payloads)
        evidence.wait_completed()
        evidence.readback.resolve()
        view.acceptResidentHistogramEvidence((evidence.evidence_key,))

        # Closing/reopening the coverage phase is what a camera-only retarget
        # does. The completed content+mapping evidence remains authoritative.
        view.setResidentHistogramEvidenceRequired(False)
        view.getView().setRange(xRange=(2, 28), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        view.setResidentHistogramEvidenceRequired(True, obligation)
        _commit(view, geometry, payloads, levels=(-2.0, 5.0))

        assert view.residentHistogramEvidence(payloads) == ()
        assert view._wgpu_executor.histogram_dispatches_total == 1
        assert view._wgpu_executor.histogram_readback_resolves_total == 1
    finally:
        view.close()


def test_phase1_windowable_rgb_uses_resident_alpha_histogram_signal(qt_app):
    from arrayscope.display.shader_mapping import ShaderDisplayMode, ShaderMapping

    view = _shown_view(qt_app)
    try:
        histogram = np.linspace(2.0, 8.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        image = np.full((20, 30, 3), 0.5, dtype=np.float32)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("g6a-windowed-rgb", 0),
                shader_mapping=ShaderMapping(display_mode=ShaderDisplayMode.RGB_WINDOWED),
                histogram_data=histogram,
            )
        }
        view.setResidentHistogramEvidenceRequired(True)

        report = _commit(view, geometry, payloads, levels=(2.0, 8.0))

        assert report.presented_tiles == frozenset({0})
        (evidence,) = view.residentHistogramEvidence(payloads)
        evidence.wait_completed()
        counts, bounds = evidence.readback.resolve()
        assert bounds == pytest.approx((2.0, 8.0))
        assert int(counts.sum()) == histogram.size
    finally:
        view.close()


def test_coarse_payload_falls_back_then_native_payload_refines_same_plane(qt_app):
    view = _shown_view(qt_app)
    try:
        source_shape = (512, 512)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        coarse = _lod_payload(
            0,
            np.full((32, 32), 0.25, np.float32),
            base_source_id="lod-plane",
            # Rung labels describe quality role, not pyramid exponent.  The
            # physical factor is the executor's LOD owner (ADR 0050).
            level=3,
            factor=16,
            source_shape=source_shape,
            payload_source_shape=(32, 32),
        )
        fine = _lod_payload(
            0,
            np.full(source_shape, 0.8, np.float32),
            base_source_id="lod-plane",
            level=0,
            source_shape=source_shape,
        )

        report = _commit(view, geometry, {0: coarse}, levels=(0.0, 1.0))
        assert report.texture_uploads == 1
        assert report.presented_identities == {0: coarse.tile_identity}
        assert view._wgpu_executor._bound_planes[0].max_lod == 4
        assert view._wgpu_tile_instances()[0].lod_level == 0
        assert view._wgpu_tile_instances()[0].src_size == (512.0, 512.0)
        view.getView().setRange(xRange=(0, 512), yRange=(0, 512), padding=0)
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 64, atol=2)

        coarse_keys = set(view._wgpu_executor.page_table.resident_keys())
        report = _commit(view, geometry, {0: fine}, levels=(0.0, 1.0))
        assert report.texture_uploads == 4
        assert report.presented_identities == {0: fine.tile_identity}
        assert view._wgpu_executor._bound_planes[0].max_lod == 4
        assert coarse_keys <= set(view._wgpu_executor.page_table.resident_keys())
        assert all(view._wgpu_executor.page_table.is_pinned(key) for key in coarse_keys)
        assert {
            key.document_generation for key in view._wgpu_executor.page_table.resident_keys()
        } == {view._wgpu_executor._bound_planes[0].document_generation}
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 204, atol=2)
    finally:
        view.close()


def test_phase_vector_reduced_page_renders_cancellation_black_and_coherence_bright(qt_app):
    """Resultant magnitude, not angle alone, owns phase-vector intensity."""

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload, PageBackedPresentation
    from arrayscope.display.pyramid import materialize_lod_page, plan_source_grid_pages
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
        TexturePlaneKind,
    )

    def payload(values, content):
        plan = plan_source_grid_pages(
            content_key=(content,),
            valid_source_rect_yx=(0, 2, 0, 2),
            reduction_yx=(1, 1),
            stored_page_shape=(1, 1),
            dtype="complex64",
            representation="complex_rg32f",
            reducer="phase_vector",
        )[0]
        page = materialize_lod_page(values, source_origin_yx=(0, 0), plan=plan)
        lod = LodInfo(1, 2, (2, 2), (1, 1), 0)
        return DisplayTilePayload(
            0,
            0,
            page.values,
            None,
            (content, 0),
            texture_data=page.values,
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            lod=lod,
            shader_mapping=ShaderMapping(
                component=ShaderComponent.ANGLE,
                display_mode=ShaderDisplayMode.PHASE_COLOR,
            ),
            page_backing=PageBackedPresentation((plan,), (page,), (0, 2, 0, 2), lod),
        )

    cancellation = payload(
        np.asarray([[1.0 + 0.0j, -1.0 + 0.0j], [0.0j, 0.0j]], np.complex64),
        "phase-cancellation",
    )
    coherent = payload(np.ones((2, 2), np.complex64), "phase-coherence")
    geometry = _montage_geometry((2, 2), 1, 1, loaded=1)
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _commit(view, geometry, {0: cancellation}, levels=(0.0, 1.0))
        view.getView().setRange(xRange=(0, 2), yRange=(0, 2), padding=0)
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 0, atol=2)

        _commit(view, geometry, {0: coherent}, levels=(0.0, 1.0))
        _rerender_internal(view)
        assert int(np.max(_center_pixel(view)[:3])) > 100
    finally:
        view.close()


def _preview_lod_payload(tile_number, image, *, base_source_id, level, source_shape):
    payload = _lod_payload(
        tile_number,
        image,
        base_source_id=base_source_id,
        level=level,
        source_shape=source_shape,
    )
    return replace(
        payload,
        quality="preview",
        semantic_data=None,
        semantic_histogram_data=None,
        tile_identity=replace(payload.tile_identity, quality="preview"),
    )


@pytest.mark.parametrize(
    ("preview_shape", "level", "expected_pages"),
    [
        pytest.param((21, 21), 4, 2, id="two-levels-coarser"),
        pytest.param((6, 6), 6, 1, id="large-montage-dynamic-level"),
    ],
)
def test_large_preview_montage_uses_shared_pages_with_per_tile_ack_truth(
    qt_app,
    preview_shape,
    level,
    expected_pages,
):
    """Ring 4 helper: compact preview residency must still acknowledge each tile."""

    from arrayscope.display.model.montage_levels import TileLevelStats

    source_shape = (336, 336)
    tile_count = 272
    geometry = _montage_geometry(source_shape, 17, 16, loaded=tile_count)
    payloads = {
        tile: replace(
            _preview_lod_payload(
                tile,
                np.full(preview_shape, float(tile + 1), np.float32),
                base_source_id=("preview-atlas", tile),
                level=level,
                source_shape=source_shape,
            ),
            level_stats=TileLevelStats(
                source_index=tile,
                bounds=(float(tile + 1), float(tile + 1)),
                sample=np.asarray([float(tile + 1)], dtype=np.float32),
            ),
        )
        for tile in range(tile_count)
    }
    view = _shown_view(qt_app, texture_codec="off")
    try:
        view.setResidentHistogramEvidenceRequired(True, ("preview-atlas", 1))
        dispatches_before = (
            0 if view._wgpu_executor is None else view._wgpu_executor.histogram_dispatches_total
        )

        report = _commit(view, geometry, payloads, levels=(0.0, float(tile_count + 1)))

        assert report.texture_uploads == expected_pages
        assert report.presented_tiles == frozenset(payloads)
        assert report.presented_identities == {
            tile: payload.tile_identity for tile, payload in payloads.items()
        }
        committed = view._wgpu_committed
        assert len(view._wgpu_executor._bound_planes) == expected_pages
        assert (
            len({key for info in committed["tiles"].values() for key in info["page_keys"]})
            == expected_pages
        )
        assert {info["plane_index"] for info in committed["tiles"].values()} == set(
            range(expected_pages)
        )
        assert {instance.lod_level for instance in view._wgpu_tile_instances()} == {0}
        assert {instance.src_size for instance in view._wgpu_tile_instances()} == {
            (float(preview_shape[1]), float(preview_shape[0]))
        }
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_preview_atlas_tiles"] == tile_count
        assert diagnostics["wgpu_preview_atlas_pages"] == expected_pages
        assert view.residentHistogramEvidence(payloads) == ()
        assert view._wgpu_executor.histogram_dispatches_total == dispatches_before
        region = committed["tiles"][144]["world_rect"]
        view.getView().setRange(
            xRange=(region[0], region[0] + region[2]),
            yRange=(region[1], region[1] + region[3]),
            padding=0,
        )
        _rerender_internal(view)
        expected = round(145.0 / float(tile_count + 1) * 255.0)
        assert np.allclose(_center_pixel(view)[:3], expected, atol=2)
    finally:
        view.close()


def test_partial_preview_cohort_does_not_rebuild_a_growing_atlas(qt_app):
    """Atlas construction belongs to the complete required-scope boundary."""

    source_shape = (336, 336)
    geometry = _montage_geometry(source_shape, 17, 16, loaded=272)
    payloads = {
        tile: _preview_lod_payload(
            tile,
            np.full((21, 21), float(tile + 1), np.float32),
            base_source_id=("partial-preview-atlas", tile),
            level=4,
            source_shape=source_shape,
        )
        for tile in range(64)
    }
    view = _shown_view(qt_app, texture_codec="off")
    try:
        report = _commit(view, geometry, payloads, levels=(0.0, 272.0))

        assert report.texture_uploads == len(payloads)
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_preview_atlas_tiles"] == 0
        assert diagnostics["wgpu_preview_atlas_pages"] == 0
    finally:
        view.close()


def test_exact_refinement_preserves_shared_preview_pages_and_mixed_ack_truth(qt_app):
    """Exact tiles overlay one retained atlas; untouched tiles keep preview truth."""

    source_shape = (336, 336)
    tile_count = 272
    geometry = _montage_geometry(source_shape, 17, 16, loaded=tile_count)
    preview = {
        tile: _preview_lod_payload(
            tile,
            np.full((21, 21), float(tile + 1), np.float32),
            base_source_id=("mixed-preview-atlas", tile),
            level=4,
            source_shape=source_shape,
        )
        for tile in range(tile_count)
    }
    view = _shown_view(qt_app, texture_codec="off")
    try:
        first = _commit(view, geometry, preview, levels=(0.0, 300.0))
        assert first.texture_uploads == 2
        atlas_keys = {
            key for info in view._wgpu_committed["tiles"].values() for key in info["page_keys"]
        }
        refined = {
            tile: _lod_payload(
                tile,
                np.full(source_shape, float(280 + tile), np.float32),
                base_source_id=("mixed-preview-atlas", tile),
                level=0,
                source_shape=source_shape,
            )
            for tile in (0, 144)
        }
        mixed = {**preview, **refined}

        second = _commit(view, geometry, mixed, levels=(0.0, 300.0))

        assert second.texture_uploads == 8
        assert second.presented_tiles == frozenset(mixed)
        assert second.presented_identities == {
            tile: payload.tile_identity for tile, payload in mixed.items()
        }
        assert atlas_keys <= set(view._wgpu_executor.page_table.resident_keys())
        preview_keys = {
            key
            for tile, info in view._wgpu_committed["tiles"].items()
            if tile not in refined
            for key in info["page_keys"]
        }
        assert preview_keys == atlas_keys
        assert len(view._wgpu_executor._bound_planes) == 4
        assert {instance.lod_level for instance in view._wgpu_tile_instances()} == {0}
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_preview_atlas_tiles"] == tile_count - len(refined)
        assert diagnostics["wgpu_preview_atlas_pages"] == 2
        region = view._wgpu_committed["tiles"][0]["world_rect"]
        view.getView().setRange(
            xRange=(region[0], region[0] + region[2]),
            yRange=(region[1], region[1] + region[3]),
            padding=0,
        )
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], round(280.0 / 300.0 * 255.0), atol=2)
    finally:
        view.close()


def test_complex_preview_montage_uses_two_shared_rg_pages(qt_app):
    """Complex preview values remain raw and shader-mapped in the shared atlas."""

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
        TexturePlaneKind,
    )

    source_shape = (336, 336)
    tile_count = 272
    geometry = _montage_geometry(source_shape, 17, 16, loaded=tile_count)
    mapping = ShaderMapping(
        component=ShaderComponent.ABS,
        display_mode=ShaderDisplayMode.COMPLEX,
    )
    payloads = {}
    for tile in range(tile_count):
        values = np.full((21, 21), 3.0 + 4.0j, np.complex64)
        identity = TileIdentity(
            document_generation=("complex-preview-atlas", 1),
            operation_key=("abs",),
            source_index=tile,
            image_axes=(0, 1),
            axis_flips=(False, False),
            channel="complex",
            complex_mapping=("complex", "abs"),
            texture_kind=TexturePlaneKind.COMPLEX_RG32F,
            semantic_generation=("complex-preview", tile),
            lod=TileLodIdentity(level=4, factor=16),
            quality="fallback",
        )
        payloads[tile] = DisplayTilePayload(
            tile,
            tile,
            values,
            None,
            ("complex-preview-atlas", tile, "lod", 16, 4, 0),
            source_shape=source_shape,
            lod=LodInfo(
                level=4,
                factor=16,
                source_shape=source_shape,
                texture_shape=(21, 21),
            ),
            shader_mapping=mapping,
            quality="preview",
            tile_identity=identity,
        )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))

        assert report.texture_uploads == 2
        assert report.presented_tiles == frozenset(payloads)
        assert {plane.representation for plane in view._wgpu_executor._bound_planes} == {
            "complex_rg32f"
        }
        region = view._wgpu_committed["tiles"][144]["world_rect"]
        view.getView().setRange(
            xRange=(region[0], region[0] + region[2]),
            yRange=(region[1], region[1] + region[3]),
            padding=0,
        )
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 128, atol=2)
    finally:
        view.close()


def test_coarser_mean_payload_generates_from_resident_fine_pages_zero_upload(qt_app):
    from arrayscope.display.pyramid import reduce_box_mean
    from arrayscope.gpu.keys import REDUCER_MEAN

    # Pin the codec OFF: this asserts the GPU LOD-reduction of resident fine
    # pages is bit-exact (rtol 1e-6).  On an ASTC-capable device AUTO now
    # compresses even this random-noise source (ASTC 4x4 clears the 40 dB gate
    # where BC4 declined), so the reduction would be codec-lossy — irrelevant to
    # what this test checks (raw GPU reduce, zero upload).
    view = _shown_view(qt_app, texture_codec="off")
    try:
        source_shape = (512, 512)
        rng = np.random.default_rng(606)
        fine_values = rng.standard_normal(source_shape, dtype=np.float32)
        coarse_values = reduce_box_mean(fine_values, (4, 4))
        # Deliberately hostile descriptor payload: the live path must ignore
        # these CPU bytes and derive the requested page from resident L0.
        coarse_payload_values = np.full(coarse_values.shape, 123.0, np.float32)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        fine = _lod_payload(
            0,
            fine_values,
            base_source_id="gpu-generated-lod-plane",
            level=0,
            source_shape=source_shape,
        )
        coarse = _lod_payload(
            0,
            coarse_payload_values,
            base_source_id="gpu-generated-lod-plane",
            level=2,
            source_shape=source_shape,
        )

        first = _commit(view, geometry, {0: fine}, levels=(-5.0, 5.0))
        assert first.texture_uploads == 4
        second = _commit(view, geometry, {0: coarse}, levels=(-5.0, 5.0))

        assert second.texture_uploads == 0
        assert second.presented_identities == {0: coarse.tile_identity}
        (generated_key,) = view._wgpu_committed["tiles"][0]["page_keys"]
        assert generated_key.lod.reducer == REDUCER_MEAN
        assert generated_key.lod.level == 2
        assert view._wgpu_executor.page_table.lookup(generated_key) is not None
        view._wgpu_executor.device.queue.on_submitted_work_done_sync()
        gpu_page = view._wgpu_executor.read_resident_page(generated_key)
        np.testing.assert_allclose(
            gpu_page[: coarse_values.shape[0], : coarse_values.shape[1]],
            coarse_values,
            rtol=1e-6,
            atol=1e-6,
        )
        assert not np.any(gpu_page[coarse_values.shape[0] :, :])
        assert not np.any(gpu_page[:, coarse_values.shape[1] :])
    finally:
        view.close()


def test_non_power_of_two_payload_factor_is_rejected_loudly(qt_app):
    view = _shown_view(qt_app)
    try:
        source_shape = (48, 48)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        payload = _lod_payload(
            0,
            np.zeros((16, 16), np.float32),
            base_source_id="bad-lod-factor",
            level=2,
            factor=3,
            source_shape=source_shape,
        )

        with pytest.raises(NotImplementedError, match=r"power-of-two.*factor 3"):
            _commit(view, geometry, {0: payload}, levels=(0.0, 1.0))
    finally:
        view.close()


def test_partial_residency_acknowledges_only_resident_tiles(qt_app):
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One-layer scalar pool: the second tile's upload must evict the
        # first tile's page inside the same submission.
        small = WgpuPlaneExecutor(pool_layers={"scalar_r32f": 1}, device=_shared_wgpu_device())
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small

        geometry = _montage_geometry((20, 30), 2, 1, loaded=2)
        payloads = {
            0: _payload(0, np.zeros((20, 30), np.float32), source_id=("partial", 0)),
            1: _payload(1, np.ones((20, 30), np.float32), source_id=("partial", 1)),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {1}
        assert report.presented_identities == {1: ("partial", 1)}
    finally:
        view.close()


def test_display_aid_flags_ride_every_mapping_rebuild_and_toggle_live(qt_app):
    """Stage-A pixel-grid / clip flags must survive every DisplayMapping rebuild
    (full commit AND the partial level/LUT rebuilds) and toggle live without an
    upload — the regression the copy-forward rebuild sites would otherwise hit."""

    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _view_class("wgpu")(pixel_grid=True)
    view.resize(320, 260)
    view.show()
    try:
        # Constructor flag seeds the initial mapping.
        assert view._wgpu_mapping_state.pixel_grid is True
        assert view._wgpu_mapping_state.clip_indicator is False

        image = np.full((16, 24), 3.0 + 4.0j, dtype=np.complex64)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-aids", 1),
                shader_mapping=ShaderMapping(
                    component=ShaderComponent.ABS, display_mode=ShaderDisplayMode.COMPLEX
                ),
            )
        }
        # A full commit rebuilds the mapping from scratch — the flag must ride it.
        _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert view._wgpu_mapping_state.pixel_grid is True

        # Live toggle of the clip indicator: shader-uniform only, zero upload.
        before = view._wgpu_executor.uploads_total
        view.setWgpuClipIndicatorEnabled(True)
        assert view._wgpu_executor.uploads_total == before
        assert view._wgpu_mapping_state.clip_indicator is True
        assert view._wgpu_mapping_state.pixel_grid is True

        # A level change goes through the PARTIAL copy-forward rebuild; both
        # flags must survive it (they were dropped before the wiring fix).
        view._apply_preview_levels_to_display((0.0, 5.0), final=True)
        assert view._wgpu_mapping_state.level_hi == pytest.approx(5.0)
        assert view._wgpu_mapping_state.pixel_grid is True
        assert view._wgpu_mapping_state.clip_indicator is True

        # Turning the grid off live flips exactly that flag.
        view.setWgpuPixelGridEnabled(False)
        assert view._wgpu_mapping_state.pixel_grid is False
        assert view._wgpu_mapping_state.clip_indicator is True
        assert view.wgpuPixelGridEnabled() is False
        assert view.wgpuClipIndicatorEnabled() is True
    finally:
        view.close()


def test_complex_tile_mode_switch_is_zero_upload_with_physical_truth(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _shown_view(qt_app)
    try:
        value = 3.0 + 4.0j  # magnitude exactly 5
        image = np.full((16, 24), value, dtype=np.complex64)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)

        def mapping(component):
            return ShaderMapping(component=component, display_mode=ShaderDisplayMode.COMPLEX)

        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-complex", 1),
                shader_mapping=mapping(ShaderComponent.ABS),
            )
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1
        assert view._wgpu_mapping_state.mode == "magnitude"
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        # |3+4j| = 5 → g = 0.5 → grayscale 128 (nearest LUT entry).
        assert np.allclose(_center_pixel(view), (128, 128, 128, 255), atol=2)

        # Mode switch (same content identity): physically zero-upload.
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-complex", 1),
                shader_mapping=mapping(ShaderComponent.REAL),
            )
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 0
        assert view._wgpu_mapping_state.mode == "real"
        _rerender_internal(view)
        # Re(3+4j) = 3 → g = 0.3 → grayscale round(76.5) ∈ {76, 77}.
        assert np.allclose(_center_pixel(view), (76, 76, 76, 255), atol=2)

        # Levels switch through the shared preview driver: zero-upload too.
        before = view._wgpu_executor.uploads_total
        view._apply_preview_levels_to_display((0.0, 5.0), final=True)
        assert view._wgpu_executor.uploads_total == before
        assert view._wgpu_mapping_state.level_hi == pytest.approx(5.0)
    finally:
        view.close()


def test_complex_montage_acknowledges_only_resident_content_planes(qt_app):
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One physical complex layer for two requested ContentPlanes: the
        # second upload evicts the first, so only tile 1 can be acknowledged.
        small = WgpuPlaneExecutor(pool_layers={"complex_rg32f": 1}, device=_shared_wgpu_device())
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small

        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        payloads = {
            0: _payload(
                0,
                np.full((16, 24), 3.0 + 4.0j, np.complex64),
                source_id=("complex-partial", 0),
            ),
            1: _payload(
                1,
                np.full((16, 24), 6.0 + 8.0j, np.complex64),
                source_id=("complex-partial", 1),
            ),
        }
        report = _commit(view, geometry, payloads, levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {1}
        assert report.presented_identities == {1: ("complex-partial", 1)}
        assert len(view._wgpu_executor.bound_planes) == 2
        assert all(
            plane.representation == "complex_rg32f" for plane in view._wgpu_executor.bound_planes
        )
    finally:
        view.close()


def test_complex_montage_mode_switch_is_zero_upload_per_tile(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        images = {
            0: np.full((16, 24), 3.0 + 4.0j, np.complex64),
            1: np.full((16, 24), 6.0 + 8.0j, np.complex64),
        }

        def payloads(component):
            mapping = ShaderMapping(component=component, display_mode=ShaderDisplayMode.COMPLEX)
            return {
                tile: _payload(
                    tile,
                    image,
                    source_id=("complex-montage", tile),
                    shader_mapping=mapping,
                )
                for tile, image in images.items()
            }

        report = _commit(view, geometry, payloads(ShaderComponent.ABS), levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.presented_identities == {
            0: ("complex-montage", 0),
            1: ("complex-montage", 1),
        }
        assert report.texture_uploads == 2
        assert len(view._wgpu_executor.bound_planes) == 2
        view.getView().setRange(xRange=(0, 48), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        h, w = target.shape[:2]
        # Each tile samples its own ContentPlane: magnitudes 5 and 10.
        assert np.allclose(target[h // 2, w // 4], (128, 128, 128, 255), atol=2)
        assert np.allclose(target[h // 2, 3 * w // 4], (255, 255, 255, 255), atol=2)

        # Same per-tile content identities, new component uniform: both tiles
        # remain physically acknowledged without another texture upload.
        report = _commit(view, geometry, payloads(ShaderComponent.REAL), levels=(0.0, 10.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.presented_identities == {
            0: ("complex-montage", 0),
            1: ("complex-montage", 1),
        }
        assert report.texture_uploads == 0
        assert view._wgpu_mapping_state.mode == "real"
        _rerender_internal(view)
        target = view._wgpu_executor.read_target()
        # Uniform-only mode switch exposes real components 3 and 6.
        assert np.allclose(target[h // 2, w // 4], (76, 76, 76, 255), atol=2)
        assert np.allclose(target[h // 2, 3 * w // 4], (153, 153, 153, 255), atol=2)
    finally:
        view.close()


def test_complex_phase_color_uses_phase_lut(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
        default_phase_lut,
    )

    view = _shown_view(qt_app)
    try:
        phase = np.pi / 2
        image = np.full((16, 24), np.exp(1j * phase), dtype=np.complex64)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-phase", 1),
                shader_mapping=ShaderMapping(
                    component=ShaderComponent.ANGLE,
                    display_mode=ShaderDisplayMode.PHASE_COLOR,
                ),
            )
        }
        report = _commit(view, geometry, payloads, levels=(-np.pi, np.pi))
        assert set(report.presented_tiles) == {0}
        assert view._wgpu_mapping_state.mode == "phase"
        assert view._wgpu_mapping_state.lut is not None
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        # g = (π/2 + π) / 2π = 0.75 → nearest phase-LUT entry 191.
        expected = default_phase_lut()[191]
        assert np.allclose(_center_pixel(view)[:3], expected, atol=2)
    finally:
        view.close()


def test_magnitude_modulated_phase_color_matches_cpu_oracle_and_switches_zero_upload(qt_app):
    from dataclasses import replace

    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
        cpu_display_rgba,
        default_phase_lut,
    )

    view = _shown_view(qt_app)
    try:
        phase = np.pi / 3.0
        image = np.full((16, 24), 0.5 * np.exp(1j * phase), dtype=np.complex64)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)
        scalar_mapping = ShaderMapping(
            component=ShaderComponent.ABS,
            display_mode=ShaderDisplayMode.COMPLEX,
        )
        phase_mapping = replace(
            scalar_mapping,
            display_mode=ShaderDisplayMode.PHASE_COLOR,
        )
        source_id = ("wgpu-phase-modulated", 1)

        report = _commit(
            view,
            geometry,
            {
                0: _payload(
                    0,
                    image,
                    source_id=source_id,
                    shader_mapping=scalar_mapping,
                )
            },
            levels=(0.0, 1.0),
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1

        report = _commit(
            view,
            geometry,
            {
                0: _payload(
                    0,
                    image,
                    source_id=source_id,
                    shader_mapping=phase_mapping,
                )
            },
            levels=(0.0, 1.0),
        )
        assert report.texture_uploads == 0
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        expected_mapping = replace(
            phase_mapping,
            levels=(0.0, 1.0),
            lut_data=default_phase_lut(),
        )
        expected = cpu_display_rgba(image, expected_mapping)[0, 0]
        assert np.allclose(_center_pixel(view), expected, atol=3)
    finally:
        view.close()


def test_float_rgb_acknowledges_only_physically_resident_packed_pages(qt_app):
    from arrayscope.display.shader_mapping import ShaderDisplayMode, ShaderMapping
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.keys import RGB_WINDOWED_RGBA32F
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        small = WgpuPlaneExecutor(
            pool_layers={RGB_WINDOWED_RGBA32F: 1},
            device=_shared_wgpu_device(),
        )
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required, **_kwargs: small
        geometry = _montage_geometry((20, 30), 2, 1, loaded=2)
        mapping = ShaderMapping(display_mode=ShaderDisplayMode.RGB_WINDOWED)
        payloads = {
            tile: _payload(
                tile,
                np.full((20, 30, 3), 0.25 + 0.25 * tile, np.float32),
                source_id=("wgpu-float-rgb-partial", tile),
                shader_mapping=mapping,
                histogram_data=np.full((20, 30), 0.5, np.float32),
            )
            for tile in (0, 1)
        }

        report = _commit(
            view,
            geometry,
            payloads,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
        )

        assert report.texture_uploads == 2
        assert set(report.presented_tiles) == {1}
        assert report.presented_identities == {1: ("wgpu-float-rgb-partial", 1)}
        assert {key.representation for key in small.page_table.resident_keys()} == {
            RGB_WINDOWED_RGBA32F
        }
    finally:
        view.close()


def test_log_and_symlog_scale_switch_is_zero_upload(qt_app):
    from arrayscope.display.shader_mapping import ShaderMapping, ShaderScale

    view = _shown_view(qt_app)
    try:
        image = np.full((16, 24), 100.0, dtype=np.float32)
        geometry = _montage_geometry((16, 24), 1, 1, loaded=1)

        def payload(scale, *, symlog_constant=0.0):
            return {
                0: _payload(
                    0,
                    image,
                    source_id=("wgpu-scale", 1),
                    shader_mapping=ShaderMapping(scale=scale, symlog_constant=symlog_constant),
                )
            }

        report = _commit(
            view,
            geometry,
            payload(ShaderScale.LOG),
            levels=(0.0, 4.0),
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1
        assert view._wgpu_mapping_state.scale == "log"
        view.getView().setRange(xRange=(0, 24), yRange=(0, 16), padding=0)
        _rerender_internal(view)
        # log10(100) = 2 in [0, 4] -> nearest grayscale entry 128.
        assert np.allclose(_center_pixel(view), (128, 128, 128, 255), atol=2)

        report = _commit(
            view,
            geometry,
            payload(ShaderScale.SYMLOG, symlog_constant=1.0),
            levels=(0.0, 2.0),
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 0
        assert view._wgpu_mapping_state.scale == "symlog"
        assert view._wgpu_mapping_state.symlog_constant == pytest.approx(1.0)
        _rerender_internal(view)
        # symlog(100, C=1) = log10(11), mapped through [0, 2].
        expected = round(np.log10(11.0) / 2.0 * 255.0)
        assert np.allclose(_center_pixel(view), (*([expected] * 3), 255), atol=2)
    finally:
        view.close()


def test_rgb_display_ready_tile_renders_raw_bytes(qt_app):
    view = _shown_view(qt_app)
    try:
        color = np.array([10, 200, 60], np.uint8)
        image = np.broadcast_to(color, (20, 30, 3)).copy()
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("wgpu-rgb", 1))}
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0), rgb_already_windowed=True)
        assert set(report.presented_tiles) == {0}
        view.getView().setRange(xRange=(0, 30), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        # Display-ready bytes: levels/LUT bypassed, rendered as-is.
        assert (_center_pixel(view) == (*color, 255)).all()
    finally:
        view.close()


def test_float_rgb_windowing_matches_cpu_reference_and_levels_switch_is_zero_upload(qt_app):
    from arrayscope.display.image_upload import rgb_display_for_levels
    from arrayscope.display.shader_mapping import (
        ShaderDisplayMode,
        ShaderMapping,
        TexturePlaneKind,
        pack_texture_data,
    )

    view = _shown_view(qt_app)
    try:
        color = np.array([0.25, 0.5, 1.0], np.float32)
        image = np.broadcast_to(color, (20, 30, 3)).copy()
        histogram = np.full((20, 30), 0.5, np.float32)
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {
            0: _payload(
                0,
                image,
                source_id=("wgpu-float-rgb", 1),
                shader_mapping=ShaderMapping(display_mode=ShaderDisplayMode.RGB_WINDOWED),
                histogram_data=histogram,
            )
        }

        report = _commit(
            view,
            geometry,
            payloads,
            levels=(0.0, 1.0),
            rgb_already_windowed=False,
        )
        assert set(report.presented_tiles) == {0}
        assert report.texture_uploads == 1
        view.getView().setRange(xRange=(0, 30), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        base = pack_texture_data(image, TexturePlaneKind.RGB8)
        expected = rgb_display_for_levels(base, histogram, (0.0, 1.0))[0, 0]
        assert np.allclose(_center_pixel(view), (*expected, 255), atol=2)

        before = view._wgpu_executor.uploads_total
        view._apply_preview_levels_to_display((0.0, 0.5), final=True)
        assert view._wgpu_executor.uploads_total == before
        _rerender_internal(view)
        expected = rgb_display_for_levels(base, histogram, (0.0, 0.5))[0, 0]
        assert np.allclose(_center_pixel(view), (*expected, 255), atol=2)
    finally:
        view.close()


def test_out_of_scope_commits_reject_loudly(qt_app):
    from arrayscope.display.shader_mapping import (
        ShaderComponent,
        ShaderDisplayMode,
        ShaderMapping,
    )

    view = _shown_view(qt_app)
    try:
        scalar = np.zeros((20, 30), np.float32)
        cplx = np.zeros((20, 30), np.complex64)
        geometry2 = _montage_geometry((20, 30), 2, 1, loaded=2)
        geometry1 = _montage_geometry((20, 30), 1, 1, loaded=1)

        # Mixed representations in one commit.
        with pytest.raises(NotImplementedError, match="one texture representation"):
            _commit(
                view,
                geometry2,
                {
                    0: _payload(0, scalar, source_id=("rej", 2)),
                    1: _payload(1, cplx.copy(), source_id=("rej", 3)),
                },
                levels=(0.0, 1.0),
            )
        # Display-ready promises are still strict: float RGB is not silently
        # quantized into the bypass pool. The supported float path is the
        # separately tested windowable-RGB representation.
        with pytest.raises(NotImplementedError, match="display-ready"):
            _commit(
                view,
                geometry1,
                {
                    0: _payload(
                        0,
                        np.zeros((20, 30, 3), np.float32),
                        source_id=("rej", 5),
                    )
                },
                levels=(0.0, 1.0),
                rgb_already_windowed=True,
            )
        # Phase-color has honest phase and magnitude variants only. A real
        # component plus cyclic hue has no established backend semantics.
        with pytest.raises(NotImplementedError, match="phase or magnitude"):
            _commit(
                view,
                geometry1,
                {
                    0: _payload(
                        0,
                        cplx,
                        source_id=("rej", 6),
                        shader_mapping=ShaderMapping(
                            component=ShaderComponent.REAL,
                            display_mode=ShaderDisplayMode.PHASE_COLOR,
                        ),
                    )
                },
                levels=(0.0, 1.0),
            )
        # The rejected commits must not have left a half-presented surface.
        assert view.montageDisplayMode() == "none"
    finally:
        view.close()


def test_warm_tiled_residency_accepts_the_commit_plan_contract(qt_app):
    """Regression: the live warm path must consume _wgpu_commit_plan's full
    return contract.  The 3c-prep branch unpacked two values while the
    rejection-lift branch grew the plan to four — no offscreen test drove
    warmTiledResidency, so only the real-Wayland journey matrix caught the
    ValueError.  This pins the seam offscreen."""

    import numpy as np

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        images = {
            tile: np.linspace(0.0, 1.0, 16 * 24, dtype=np.float32).reshape(16, 24) + tile
            for tile in (0, 1)
        }
        payloads = {
            tile: _payload(tile, image, source_id=f"warm-src-{tile}")
            for tile, image in images.items()
        }
        _commit(view, geometry, {0: payloads[0]}, levels=(0.0, 2.0))
        resident_before = len(view._wgpu_executor.page_table.resident_keys())
        view.warmTiledResidency(
            payloads={1: payloads[1]},
            geometry=geometry,
            levels=(0.0, 2.0),
        )
        assert len(view._wgpu_executor.page_table.resident_keys()) >= resident_before
    finally:
        view.close()


def test_atomic_warm_batches_reserve_the_complete_successor(qt_app):
    """A cropped 50-tile successor must survive bounded hidden warming.

    This mirrors the dogfood failure rather than using one-page toy tiles:
    reversing the displayed axes produced 336x100 scalar tiles (two pages
    each), and the complete predecessor remained bound while the successor
    arrived in two-tile batches.
    """

    from arrayscope.display.model.frame import TilePresentationDelta
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        tile_count = 50
        geometry = _montage_geometry((336, 100), 10, 5, loaded=tile_count)
        predecessor = {
            tile: _payload(
                tile,
                np.full((336, 100), float(tile + 1), dtype=np.float32),
                source_id=("crop-generation", 0, "source", tile),
            )
            for tile in range(tile_count)
        }
        payloads = {
            tile: _payload(
                tile,
                np.full((336, 100), float(tile + 101), dtype=np.float32),
                source_id=("crop-generation", 1, "source", tile),
            )
            for tile in range(tile_count)
        }
        delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts=payloads,
            active_tiles=tuple(range(tile_count)),
            planned_tiles=tuple(range(tile_count)),
            atomic_handoff=True,
        )
        view._wgpu_executor = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 1}, device=_shared_wgpu_device()
        )
        _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 200.0),
            histogramRange=(0.0, 200.0),
            montage_tile_payloads=predecessor,
        )

        for start in range(0, tile_count, 2):
            view.warmTiledResidency(
                payloads={
                    tile: payloads[tile] for tile in range(start, min(start + 2, tile_count))
                },
                geometry=geometry,
                levels=(0.0, 200.0),
                tile_delta=delta,
            )

        assert view._wgpu_executor.pool_budget("scalar_r32f") >= 200
        assert all(view.tiledPayloadResident(payload) for payload in payloads.values())
        report = _present_tiled(
            view,
            np.zeros(geometry.display_shape, dtype=np.float32),
            geometry=geometry,
            levels=(0.0, 200.0),
            histogramRange=(0.0, 200.0),
            montage_tile_payloads=payloads,
            tile_delta=delta,
        )
        assert report.presented_tiles == frozenset(payloads)
    finally:
        view.close()


def test_residency_predicate_reuses_binding_within_page_table_generation(
    qt_app,
    monkeypatch,
):
    """Pacing may query one payload hundreds of times without rebuilding keys."""

    import arrayscope.display.wgpu_imageview2d as wgpu_view

    payload = _payload(
        0,
        np.arange(100 * 336, dtype=np.float32).reshape(100, 336),
        source_id=("resident-predicate", "long-semantic-identity", tuple(range(100))),
    )
    view = _shown_view(qt_app)
    try:
        _commit(
            view,
            _montage_geometry((100, 336), 1, 1, loaded=1),
            {0: payload},
            levels=(0.0, float(100 * 336)),
        )
        calls = 0
        original = wgpu_view._wgpu_payload_binding

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(wgpu_view, "_wgpu_payload_binding", counted)
        for _ in range(100):
            assert view.tiledPayloadResident(payload)

        assert calls == 1
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_residency_binding_cache_misses"] == 1
        assert diagnostics["wgpu_residency_binding_cache_hits"] == 99
    finally:
        view.close()


def test_atomic_warm_owns_successor_pages_until_the_bound_plane_swap(qt_app):
    """Unrelated residency churn must evict stale pages, not the successor."""

    from arrayscope.display.model.frame import TilePresentationDelta
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 4, 1, loaded=4)

        def generation(name, count):
            return {
                tile: _payload(
                    tile,
                    np.full((16, 24), float(tile + 1), dtype=np.float32),
                    source_id=(name, tile),
                )
                for tile in range(count)
            }

        predecessor = generation("bound-predecessor", 4)
        stale = generation("stale-cache", 4)
        successor = generation("atomic-successor", 4)
        competitor = generation("competing-warm", 8)
        delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts=successor,
            active_tiles=tuple(successor),
            planned_tiles=tuple(successor),
            atomic_handoff=True,
        )
        view._wgpu_executor = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 16},
            device=_shared_wgpu_device(),
        )
        present_kwargs = {
            "geometry": geometry,
            "levels": (0.0, 10.0),
            "histogramRange": (0.0, 10.0),
        }
        image = np.zeros(geometry.display_shape, dtype=np.float32)
        _present_tiled(
            view,
            image,
            montage_tile_payloads=predecessor,
            **present_kwargs,
        )
        # A second binding pins the already-resident predecessor pages, just
        # like the settled frame in the dogfood trace.
        _present_tiled(
            view,
            image,
            montage_tile_payloads=predecessor,
            **present_kwargs,
        )
        view.warmTiledResidency(payloads=stale)
        for start in range(0, 4, 2):
            view.warmTiledResidency(
                payloads={tile: successor[tile] for tile in range(start, start + 2)},
                tile_delta=delta,
            )
        # Make the unrelated cache rows newer than the successor. With no
        # transaction owner, the following eight-page warm evicts all four
        # successor pages and reproduces the physical-zero handoff.
        view.warmTiledResidency(payloads=stale)
        view.warmTiledResidency(payloads=competitor)

        assert all(view.tiledPayloadResident(payload) for payload in successor.values())
        uploads_before = view._wgpu_executor.uploads_total
        report = _present_tiled(
            view,
            image,
            montage_tile_payloads=successor,
            tile_delta=delta,
            **present_kwargs,
        )
        assert report.presented_tiles == frozenset(successor)
        assert view._wgpu_executor.uploads_total == uploads_before
    finally:
        view.close()


def test_atomic_warm_grows_pool_for_bound_predecessor_and_successor(qt_app):
    """The hidden successor expands residency without re-uploading its predecessor."""

    from arrayscope.display.model.frame import TilePresentationDelta
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 4, 1, loaded=4)

        def generation(name):
            return {
                tile: _payload(
                    tile,
                    np.full((16, 24), float(tile + 1), dtype=np.float32),
                    source_id=(name, tile),
                )
                for tile in range(4)
            }

        predecessor = generation("bound-predecessor")
        successor = generation("atomic-successor")
        delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts=successor,
            active_tiles=tuple(successor),
            planned_tiles=tuple(successor),
            atomic_handoff=True,
        )
        executor = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 6},
            device=_shared_wgpu_device(),
        )
        view._wgpu_executor = executor
        present_kwargs = {
            "geometry": geometry,
            "levels": (0.0, 10.0),
            "histogramRange": (0.0, 10.0),
        }
        image = np.zeros(geometry.display_shape, dtype=np.float32)
        _present_tiled(
            view,
            image,
            montage_tile_payloads=predecessor,
            **present_kwargs,
        )
        _present_tiled(
            view,
            image,
            montage_tile_payloads=predecessor,
            **present_kwargs,
        )
        uploads_after_predecessor = executor.uploads_total

        for start in range(0, 4, 2):
            view.warmTiledResidency(
                payloads={tile: successor[tile] for tile in range(start, start + 2)},
                tile_delta=delta,
            )

        assert view._wgpu_executor is executor
        assert executor.pool_budget("scalar_r32f") >= 8
        assert all(view.tiledPayloadResident(payload) for payload in predecessor.values())
        assert all(view.tiledPayloadResident(payload) for payload in successor.values())
        assert executor.uploads_total == uploads_after_predecessor + 4
    finally:
        view.close()


def test_display_axis_crop_rebinds_full_source_pages_without_upload(qt_app):
    """A cropped montage tile samples the full plane uploaded by its predecessor.

    This is the physical contract behind fast displayed-axis scrolling: the
    crop changes ``TileInstance.src_origin``; it does not rename or re-upload
    source pages already resident from the earlier full montage.
    """

    from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor

    source = np.arange(336 * 336, dtype=np.float32).reshape(336, 336)
    content_key = ("doc", "windowless-view", "montage-source", 12)
    full = DisplayTilePayload(
        0,
        12,
        source,
        None,
        ("full-window-wrapper",),
        source_anchor=PayloadSourceAnchor(
            content_key,
            (0, 336, 0, 336),
            plane_shape=(336, 336),
        ),
    )
    cropped = DisplayTilePayload(
        0,
        12,
        source[:, 94:194],
        None,
        ("cropped-window-wrapper", 94, 194),
        source_anchor=PayloadSourceAnchor(
            content_key,
            (0, 336, 94, 194),
            plane_shape=(336, 336),
        ),
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _commit(
            view,
            _montage_geometry((336, 336), 1, 1, loaded=1),
            {0: full},
            levels=(0.0, float(source.max())),
        )
        assert view._wgpu_last_report_uploads == 4
        resident_before = frozenset(view._wgpu_executor.page_table.resident_keys())

        _commit(
            view,
            _montage_geometry((336, 100), 1, 1, loaded=1),
            {0: cropped},
            levels=(0.0, float(source.max())),
        )

        assert view._wgpu_last_report_uploads == 0
        assert resident_before <= frozenset(view._wgpu_executor.page_table.resident_keys())
        tile = view._wgpu_committed["tiles"][0]
        assert tile["src_origin"] == (94.0, 0.0)
        assert tile["src_size"] == (100.0, 336.0)
        assert tile["plane_identity"] == ("wgpu-source-plane", content_key)
    finally:
        view.close()


def test_odd_aligned_exact_reduced_crop_binds_resident_native_pages(qt_app):
    """Global reduction edge bins must not force a crop-local upload.

    A 100-sample crop beginning at source row 93 spans 51 global factor-2
    bins, even though a locally reduced 100-sample plane would contain 50.
    Once the exact native source pages are resident, the reduced successor
    samples those pages directly and the odd alignment is only a UV change.
    """

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import (
        DisplayTilePayload,
        PageBackedPresentation,
        PayloadSourceAnchor,
    )
    from arrayscope.display.pyramid import materialize_source_grid_pages, plan_source_grid_pages

    source = np.arange(336 * 336, dtype=np.float32).reshape(336, 336)
    content_key = ("doc", "windowless-view", "montage-source", 12)
    full_lod = LodInfo(
        level=2,
        factor=4,
        source_shape=(336, 336),
        texture_shape=(84, 84),
    )
    full = DisplayTilePayload(
        0,
        12,
        source.reshape(84, 4, 84, 4).mean(axis=(1, 3)),
        None,
        ("full-window-wrapper",),
        lod=full_lod,
        source_anchor=PayloadSourceAnchor(
            content_key,
            (0, 336, 0, 336),
            plane_shape=(336, 336),
        ),
        native_residency_data=source,
    )
    crop_rect = (93, 193, 0, 336)
    plans = plan_source_grid_pages(
        content_key=("page-doc", "page-op"),
        valid_source_rect_yx=crop_rect,
        reduction_yx=(1, 1),
        stored_page_shape=(256, 256),
        dtype="float32",
        representation="scalar_r32f",
        reducer="mean",
    )
    pages = materialize_source_grid_pages(
        source[93:193],
        source_origin_yx=(93, 0),
        plans=plans,
    )
    crop_lod = LodInfo(
        level=1,
        factor=2,
        source_shape=(100, 336),
        texture_shape=(51, 168),
    )
    reduced_crop = DisplayTilePayload(
        0,
        12,
        pages[0].values,
        None,
        ("cropped-reduced-window-wrapper", 93, 193),
        lod=crop_lod,
        quality="exact",
        source_anchor=PayloadSourceAnchor(
            content_key,
            crop_rect,
            plane_shape=(336, 336),
        ),
        page_backing=PageBackedPresentation(plans, pages, crop_rect, crop_lod),
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        view.warmTiledResidency(payloads={0: full})
        uploads_before = view._wgpu_executor.uploads_total
        assert uploads_before == 4

        _commit(
            view,
            _montage_geometry((100, 336), 1, 1, loaded=1),
            {0: reduced_crop},
            levels=(0.0, float(source.max())),
        )

        assert view._wgpu_executor.uploads_total == uploads_before
        tile = view._wgpu_committed["tiles"][0]
        assert tile["src_origin"] == (0.0, 93.0)
        assert tile["src_size"] == (336.0, 100.0)
        assert tile["lod_level"] == 0
        assert tile["plane_identity"] == ("wgpu-source-plane", content_key)
    finally:
        view.close()


def test_cold_odd_aligned_reduced_window_uploads_with_global_bin_offset(qt_app):
    """A cold non-montage slice may start and end inside global LOD bins.

    The 100-row window occupies eight factor-16 bins on the source grid,
    although a locally reduced 100-row array would have seven rows.  With no
    native source plane resident, WGPU must preserve that global-bin offset
    in a padded local binding instead of rejecting valid page-backed geometry.
    """

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import (
        DisplayTilePayload,
        PageBackedPresentation,
        PayloadSourceAnchor,
    )
    from arrayscope.display.pyramid import materialize_source_grid_pages, plan_source_grid_pages

    source = np.arange(336 * 336, dtype=np.float32).reshape(336, 336)
    content_key = ("doc", "single-slice", 1)
    source_rect = (94, 194, 0, 336)
    plans = plan_source_grid_pages(
        content_key=("page-doc", "page-op", 1),
        valid_source_rect_yx=source_rect,
        reduction_yx=(4, 4),
        stored_page_shape=(256, 256),
        dtype="float32",
        representation="scalar_r32f",
        reducer="mean",
    )
    pages = materialize_source_grid_pages(
        source[94:194],
        source_origin_yx=(94, 0),
        plans=plans,
    )
    lod = LodInfo(
        level=4,
        factor=16,
        source_shape=(100, 336),
        texture_shape=(8, 21),
    )
    payload = DisplayTilePayload(
        0,
        1,
        pages[0].values,
        None,
        ("cold-odd-aligned-window", 1),
        lod=lod,
        quality="exact",
        source_anchor=PayloadSourceAnchor(
            content_key,
            source_rect,
            plane_shape=(336, 336),
        ),
        page_backing=PageBackedPresentation(plans, pages, source_rect, lod),
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _commit(
            view,
            _montage_geometry((100, 336), 1, 1, loaded=1),
            {0: payload},
            levels=(0.0, float(source.max())),
        )

        assert view._wgpu_last_report_uploads == 1
        tile = view._wgpu_committed["tiles"][0]
        assert tile["src_origin"] == (0.0, 14.0)
        assert tile["src_size"] == (336.0, 100.0)
        assert tile["lod_level"] == 4
        assert tile["plane_identity"][0] == "wgpu-content-plane"
    finally:
        view.close()


def test_misaligned_reduced_crop_without_page_backing_fails_loudly():
    """A cold crop with neither backing nor residency has no honest fallback."""

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor
    from arrayscope.display.wgpu_imageview2d import _wgpu_payload_binding
    from arrayscope.gpu.wgpu_executor import SCALAR_R32F

    source_rect = (94, 194, 0, 336)
    lod = LodInfo(
        level=4,
        factor=16,
        source_shape=(100, 336),
        texture_shape=(8, 21),
    )
    payload = DisplayTilePayload(
        0,
        1,
        np.zeros((8, 21), dtype=np.float32),
        None,
        ("cold-unbacked-misaligned-window", 1),
        lod=lod,
        quality="exact",
        source_anchor=PayloadSourceAnchor(
            ("doc", "unbacked-single-slice", 1),
            source_rect,
            plane_shape=(336, 336),
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        _wgpu_payload_binding(
            payload,
            np.asarray(payload.texture_data),
            representation=SCALAR_R32F,
            mapping_mode="real",
            resident_keys=(),
        )

    message = str(excinfo.value)
    assert "does not match its native LOD ladder" in message
    assert "canonical source-plane resident levels=none" in message


def test_cold_wide_odd_aligned_reduced_window_uploads_all_local_pages(qt_app):
    """A valid multi-page cold crop falls back to a packed local upload."""

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import (
        DisplayTilePayload,
        PageBackedPresentation,
        PayloadSourceAnchor,
    )
    from arrayscope.display.pyramid import materialize_source_grid_pages, plan_source_grid_pages

    source = np.arange(100 * 4500, dtype=np.float32).reshape(100, 4500)
    content_key = ("doc", "wide-single-slice", 1)
    source_rect = (94, 194, 125, 4625)
    plans = plan_source_grid_pages(
        content_key=("page-doc", "wide-page-op", 1),
        valid_source_rect_yx=source_rect,
        reduction_yx=(4, 4),
        stored_page_shape=(256, 256),
        dtype="float32",
        representation="scalar_r32f",
        reducer="mean",
    )
    pages = materialize_source_grid_pages(
        source,
        source_origin_yx=(94, 125),
        plans=plans,
    )
    texture_shape = (
        max(plan.stored_rect_yx[1] for plan in plans)
        - min(plan.stored_rect_yx[0] for plan in plans),
        max(plan.stored_rect_yx[3] for plan in plans)
        - min(plan.stored_rect_yx[2] for plan in plans),
    )
    assert len(plans) == 2
    lod = LodInfo(
        level=4,
        factor=16,
        source_shape=(100, 4500),
        texture_shape=texture_shape,
    )
    payload = DisplayTilePayload(
        0,
        1,
        pages[0].values,
        None,
        ("cold-wide-odd-aligned-window", 1),
        lod=lod,
        quality="exact",
        source_anchor=PayloadSourceAnchor(
            content_key,
            source_rect,
            plane_shape=(336, 5000),
        ),
        page_backing=PageBackedPresentation(plans, pages, source_rect, lod),
    )
    view = _shown_view(qt_app, texture_codec="off")
    try:
        _commit(
            view,
            _montage_geometry((100, 4500), 1, 1, loaded=1),
            {0: payload},
            levels=(0.0, float(source.max())),
        )

        assert view._wgpu_last_report_uploads == 2
        tile = view._wgpu_committed["tiles"][0]
        assert tile["src_origin"] == (13.0, 14.0)
        assert tile["src_size"] == (4500.0, 100.0)
        assert tile["lod_level"] == 4
        assert tile["plane_identity"][0] == "wgpu-content-plane"
    finally:
        view.close()


def test_cold_crop_local_pages_do_not_alias_across_source_windows(qt_app):
    """A window-invariant source identity must not alias crop-local texels.

    Canonical source pages are allowed to survive a displayed-axis scroll,
    because their coordinates remain source-global.  A cold fallback page is
    different: its texel (0, 0) belongs to the current crop.  Reusing that
    page for a shifted crop silently presents the predecessor pixels while
    every semantic/backend identity claims the successor is current.
    """

    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload, PayloadSourceAnchor

    content_key = ("doc", "windowless-source", 7)
    source_id = ("montage-tile", content_key, 7)
    lod = LodInfo(level=0, factor=1, source_shape=(32, 48), texture_shape=(32, 48))

    def payload(start: int, value: float):
        return DisplayTilePayload(
            0,
            7,
            np.full((32, 48), value, dtype=np.float32),
            None,
            source_id,
            lod=lod,
            quality="exact",
            source_anchor=PayloadSourceAnchor(
                content_key,
                (start, start + 32, 20, 68),
                plane_shape=(128, 128),
            ),
        )

    view = _shown_view(qt_app, texture_codec="off")
    try:
        geometry = _montage_geometry((32, 48), 1, 1, loaded=1)
        first = payload(10, 0.2)
        second = payload(11, 0.8)

        first_report = _commit(view, geometry, {0: first}, levels=(0.0, 1.0))
        first_plane = view._wgpu_committed["tiles"][0]["plane_identity"]
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view), (51, 51, 51, 255), atol=2)

        second_report = _commit(view, geometry, {0: second}, levels=(0.0, 1.0))
        second_plane = view._wgpu_committed["tiles"][0]["plane_identity"]
        _rerender_internal(view)

        assert first_report.texture_uploads == 1
        assert second_report.texture_uploads == 1
        assert second_plane != first_plane
        assert np.allclose(_center_pixel(view), (204, 204, 204, 255), atol=2)
    finally:
        view.close()


def test_pan_reuses_tile_instances_and_skips_the_instance_upload(qt_app):
    """Panning must cost O(1), not O(tiles).

    Instances are world-space, so only the camera uniform moves. Locking
    identity (not equality) pins both halves: the view must not rebuild the
    tuple, and the executor must not repack/re-upload it.
    """

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((32, 32), 4, 4, loaded=16)
        image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        payloads = {i: _payload(i, image, source_id=f"pan-src-{i}") for i in range(16)}
        _commit(view, geometry, payloads, levels=(0.0, 1.0))
        executor = view._wgpu_executor

        vb = view.getView()
        vb.setRange(xRange=(0, 128), yRange=(0, 128), padding=0)
        instances = view._wgpu_tile_instances()
        assert len(instances) == 16
        # World space, straight off the montage layout: the second column of
        # a 32-wide tile grid starts at world x=32, not at some [0, 1] share.
        assert instances[1].dst_rect == (32.0, 0.0, 32.0, 32.0)
        # Inversion lives in the camera now, so extents stay positive.
        assert instances[1].src_size == (32.0, 32.0)
        executor._set_tiles(instances)
        writes_before = executor._tiles

        for offset in (5.0, 17.5, 40.0):
            vb.setRange(xRange=(offset, offset + 64), yRange=(0, 64), padding=0)
            panned = view._wgpu_tile_instances()
            assert panned is instances, "pan rebuilt the instance tuple"
            executor._set_tiles(panned)
            assert executor._tiles is writes_before, "pan re-uploaded the instance buffer"

        # A real commit still refreshes them.
        _commit(view, geometry, payloads, levels=(0.0, 0.5))
        assert view._wgpu_tile_instances() is not instances
    finally:
        view.close()


def test_axis_inversion_mirrors_content_and_redraws_without_commit(qt_app):
    """Dogfood bugs 2026-07-18: (1) flips only took effect after the next
    commit — the view listened to sigRangeChanged but not sigStateChanged;
    (2) xInverted was ignored by the camera rect math entirely, so drawn
    content disagreed with the ViewBox's (correctly inverted) interaction
    mapping — drags and zoom rects landed on mirrored features."""

    from arrayscope.gpu.command_protocol import PresentGeneration, UpdateTileInstances

    view = _shown_view(qt_app)
    try:
        # One tile whose left half is dark and right half is bright.
        image = np.zeros((32, 64), dtype=np.float32)
        image[:, 32:] = 1.0
        geometry = _montage_geometry((32, 64), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id="flip-src")}
        _commit(view, geometry, payloads, levels=(0.0, 1.0))
        view.getView().setRange(xRange=(0, 64), yRange=(0, 32), padding=0)

        def render_columns():
            # The camera now carries the inversion (instances are world
            # space), so it is part of every draw exactly as production
            # sends it — re-read per call to pick the flip up.
            view._submit_wgpu(
                (
                    view._wgpu_camera_command(),
                    UpdateTileInstances(view._wgpu_tile_instances()),
                    PresentGeneration(999),
                )
            )
            target = view._wgpu_executor.read_target().astype(np.float32)
            h, w = target.shape[:2]
            row = target[h // 2, :, 0]
            return float(row[: w // 4].mean()), float(row[-w // 4 :].mean())

        left, right = render_columns()
        assert right > left + 50  # bright half on the right pre-flip

        uploads_before = view._wgpu_executor.uploads_total
        draws_before = int(getattr(view, "_wgpu_draw_count", 0) or 0)
        view._wgpu_canvas_update_pending = False
        view.getView().invertX(True)
        qt_app.processEvents()
        # Bug 1: the inversion toggle alone must request a redraw (pending
        # flag armed, or the ondemand draw already ran).
        assert bool(getattr(view, "_wgpu_canvas_update_pending", False)) or (
            int(getattr(view, "_wgpu_draw_count", 0) or 0) > draws_before
        )
        # Bug 2: the drawn content must mirror horizontally, with zero uploads.
        left, right = render_columns()
        assert left > right + 50, f"content not mirrored: left={left} right={right}"
        assert view._wgpu_executor.uploads_total == uploads_before
    finally:
        view.close()


# ---- present-method selection (queue row 3 screen experiment) ---------------


def test_screen_request_falls_back_to_bitmap_off_wayland(qt_app):
    """Anywhere the screen path cannot exist, the view keeps the bitmap
    canvas, records a loud reason, and every commit contract still holds."""

    view = _view_class("wgpu")(present_method="screen")
    try:
        assert view.wgpuPresentMethod() == "bitmap"
        reason = view.wgpuPresentMethodFallbackReason()
        assert "wayland" in reason
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_present_method"] == "bitmap"
        assert diagnostics["wgpu_present_method_fallback_reason"] == reason
        assert diagnostics["wgpu_screen_presents"] == 0

        canvas = np.linspace(0.0, 1.0, 20 * 30, dtype=np.float32).reshape(20, 30)
        report = _present_tiled(
            view,
            canvas,
            histogramData=canvas.copy(),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        )
        assert set(report.presented_tiles) == {0}
    finally:
        view.close()


def test_present_method_request_normalizes_unknown_values(qt_app):
    view = _view_class("wgpu")(present_method="garbage")
    try:
        # Unknown values normalize to the bitmap default; nothing "fell back".
        assert view.wgpuPresentMethod() == "bitmap"
        assert view.wgpuPresentMethodFallbackReason() == ""
    finally:
        view.close()


def test_factory_routes_present_method_setting_to_wgpu_view(qt_app):
    from arrayscope.app.settings_state import settings_from_mapping
    from arrayscope.display.image_view_factory import create_image_view

    settings = settings_from_mapping(
        {"image_rendering_backend": "wgpu", "wgpu_present_method": "screen"}
    )
    messages = []
    view = create_image_view(settings, notify=messages.append)
    try:
        assert view.surface_kind == "wgpu"
        assert view._wgpu_present_method_requested == "screen"
        # Offscreen ring: the fallback is loud through the notify channel.
        if view.wgpuPresentMethod() == "bitmap":
            assert any("screen presentation unavailable" in m for m in messages)
    finally:
        view.close()


def test_grab_presented_framebuffer_is_screen_path_only(qt_app):
    view = _view_class("wgpu")()
    try:
        # Bitmap path: the Qt widget grab is already honest; the physical
        # capture must decline so harnesses keep using the widget grab.
        assert view.grabPresentedFramebuffer() is None
    finally:
        view.close()


def test_auto_present_method_resolves_to_bitmap_off_wayland(qt_app):
    """AUTO means "screen where the measured native-Wayland path exists" —
    everywhere else it quietly resolves to bitmap (reason still recorded in
    diagnostics; no fallback warning, unlike the explicit screen pin)."""

    view = _view_class("wgpu")(present_method="auto")
    try:
        assert view.wgpuPresentMethod() == "bitmap"
        diagnostics = view.wgpuPresentationDiagnostics()
        assert diagnostics["wgpu_present_method_requested"] == "auto"
        assert "wayland" in diagnostics["wgpu_present_method_fallback_reason"]
    finally:
        view.close()


def test_factory_auto_resolution_to_bitmap_is_not_a_warning(qt_app):
    from arrayscope.app.settings_state import settings_from_mapping
    from arrayscope.display.image_view_factory import create_image_view

    settings = settings_from_mapping(
        {"image_rendering_backend": "wgpu", "wgpu_present_method": "auto"}
    )
    messages = []
    view = create_image_view(settings, notify=messages.append)
    try:
        assert view.surface_kind == "wgpu"
        assert view._wgpu_present_method_requested == "auto"
        # AUTO resolving to bitmap offscreen is the rule working, not a
        # fallback worth a status message.
        assert not any("unavailable" in message for message in messages)
    finally:
        view.close()


def test_executor_pool_growth_preserves_atlas_upload_currency(qt_app):
    """Growing page capacity must not rebuild unrelated executor resources."""

    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        view._wgpu_executor = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 1}, device=_shared_wgpu_device()
        )
        trackers = [
            name
            for name in vars(view)
            if name.startswith("_wgpu_") and name.endswith("_atlas_uploaded_version")
        ]
        # Guard the guard: if the trackers get renamed this test must not
        # silently pass by checking nothing.
        assert "_wgpu_widget_atlas_uploaded_version" in trackers
        assert "_wgpu_glyph_atlas_uploaded_version" in trackers

        for name in trackers:
            setattr(view, name, 7)  # pretend every atlas is already uploaded

        original = view._wgpu_executor
        grown = view._ensure_wgpu_executor({"scalar_r32f": 8})
        assert grown is original
        assert grown.pool_budget("scalar_r32f") >= 8
        assert grown.pool_allocated_layers("scalar_r32f") >= 8
        assert grown.pool_grows_total == 1
        assert {name: getattr(view, name) for name in trackers} == dict.fromkeys(trackers, 7)
    finally:
        view.close()


def test_warm_residency_pool_growth_preserves_overlay_geometry(qt_app):
    """A hidden warm may grow page pools without replacing overlay state."""

    from arrayscope.core.roi import RoiKind
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        geometry = _montage_geometry((16, 24), 2, 1, loaded=2)
        images = {
            tile: np.linspace(0.0, 1.0, 16 * 24, dtype=np.float32).reshape(16, 24) + tile
            for tile in (0, 1)
        }
        payloads = {
            tile: _payload(tile, image, source_id=f"warm-overlay-src-{tile}")
            for tile, image in images.items()
        }
        view._wgpu_executor = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 1}, device=_shared_wgpu_device()
        )
        _commit(view, geometry, {0: payloads[0]}, levels=(0.0, 2.0))

        # Non-empty overlay geometry: a ROI.  createRoi syncs + submits it, so
        # the flag lands clean and the CURRENT executor already has it.
        view.createRoi(RoiKind.RECTANGLE, rect=(2.0, 2.0, 8.0, 8.0), color=(40, 220, 80))
        assert view._wgpu_overlay_geometry, "precondition: overlay geometry populated"
        assert view._wgpu_overlay_geometry_dirty is False

        original = view._wgpu_executor
        original_geometry = original._overlay_geometry

        view.warmTiledResidency(
            payloads=payloads,
            geometry=geometry,
            levels=(0.0, 2.0),
        )

        grown = view._wgpu_executor
        assert grown is original
        assert grown.pool_budget("scalar_r32f") >= 2
        assert grown._overlay_geometry == original_geometry
        assert grown._overlay_geometry == view._wgpu_overlay_geometry
        assert view._wgpu_overlay_geometry_dirty is False
    finally:
        view.close()


def test_camera_is_latched_when_the_paced_draw_runs_not_when_it_is_requested(qt_app, monkeypatch):
    """Input arriving inside the pacer's deferral window still makes the frame.

    The frame-pacing design in docs/redesign/wgpu-frame-pacing-2026-07-21.md
    only pays off if moving the draw later in the refresh cycle also moves
    the input sample later.  That is true iff the camera is read when the
    draw RUNS rather than when it is REQUESTED — otherwise deferring a draw
    to just before vblank would present state that is now a frame stale, and
    the pacer would trade smoothness for latency instead of winning both.

    This asserts the property directly rather than by reading the call
    graph: hold the pacer's deferral window open, move the camera inside it,
    and check which range reached the executor.
    """

    from arrayscope.gpu.command_protocol import SetOverlayCamera

    view = _shown_view(qt_app)
    try:
        image = np.zeros((64, 64), dtype=np.float32)
        image[16:48, 16:48] = 1.0
        geometry = _montage_geometry(image.shape, 1, 1, loaded=1)
        _commit(
            view,
            geometry,
            {0: _payload(0, image, source_id="latch-oracle")},
            levels=(0.0, 1.0),
        )

        # Stand in for the paced timer: capture the draw instead of running
        # it, so the window between "redraw requested" and "draw runs" can be
        # held open for as long as the test needs.
        deferred = []
        monkeypatch.setattr(
            view._wgpu_canvas,
            "request_draw",
            lambda *args, **kwargs: deferred.append(view._on_wgpu_draw),
        )
        submitted = []
        real_submit = view._submit_wgpu
        monkeypatch.setattr(
            view,
            "_submit_wgpu",
            lambda commands, **kwargs: (
                submitted.append(commands),
                real_submit(commands, **kwargs),
            )[1],
        )

        at_request = (0.0, 64.0)
        at_draw = (12.0, 76.0)

        # A camera move schedules a frame through sigRangeChanged...
        view._wgpu_canvas_update_pending = False
        view.getView().setRange(xRange=at_request, yRange=(0, 64), padding=0)
        assert deferred, "the camera move never reached the canvas pacer"

        # ...and the user keeps panning while that frame is still only
        # scheduled.  This second move is deliberately COALESCED into the
        # already-pending draw rather than scheduling its own, which is the
        # case that matters: if the coalesced move were lost, a fast pan
        # would present a trail of stale cameras.
        view.getView().setRange(xRange=at_draw, yRange=(0, 64), padding=0)
        assert len(deferred) == 1, "the second move should coalesce, not queue"
        submitted.clear()
        deferred.pop()()

        cameras = [
            command
            for commands in submitted
            for command in commands
            if isinstance(command, SetOverlayCamera)
        ]
        assert cameras, "the paced draw submitted no camera"
        x0, _y0, x1, _y1 = (float(value) for value in cameras[-1].world_rect)

        assert (x0, x1) == pytest.approx(at_draw, abs=0.51), (
            f"the frame presented the camera as it was when the draw was "
            f"REQUESTED ({at_request}), not as it was when the draw RAN "
            f"({at_draw}): got ({x0}, {x1}). Latching at request time would "
            f"make every paced frame one input late, and deferring the draw "
            f"toward vblank would then add latency instead of removing it."
        )
    finally:
        view.close()


def _pure_grey_midtones(target):
    """Count opaque pixels that are grey and strictly between the LUT extremes.

    A binary source through a linear (0, 1) window and a grayscale LUT can only
    produce LUT[0] or LUT[255] under point sampling: ``g`` is 0 or 1, so the
    index is 0 or 255.  An intermediate grey is therefore proof that more than
    one source texel contributed to that pixel, and nothing else in the frame
    can produce one.
    """

    rgb = target[..., :3].astype(np.int32)
    grey = (rgb[..., 0] == rgb[..., 1]) & (rgb[..., 1] == rgb[..., 2])
    return int((grey & (rgb[..., 0] > 40) & (rgb[..., 0] < 215)).sum())


def _draw_zoomed_out(view, world_rect):
    """Re-present the committed tiles under an EXPLICIT camera.

    Not ``_rerender_internal``: that reuses ``_wgpu_camera_command()``, which
    reads pyqtgraph's ``viewRange()`` — and offscreen, with no real window
    geometry, that fit is not a stable way to ask for a specific zoom.  The
    mapping still comes from the view (and therefore from app settings); only
    the camera is pinned, so "how far zoomed out" is the test's to state.
    """

    from arrayscope.gpu.command_protocol import SetOverlayCamera, UpdateTileInstances

    view._submit_wgpu(
        (
            SetOverlayCamera(world_rect, x_inverted=False, y_inverted=True),
            UpdateTileInstances(view._wgpu_tile_instances()),
        )
    )
    return view._wgpu_executor.read_target()


def test_app_settings_default_renders_a_minified_montage_filtered(qt_app):
    """The default-ON gate for the C1 minification filter.

    Every other C1 oracle drives the shader directly.  This one drives the
    DEFAULT: a view built the way the app builds it — ``AppSettingsState``
    defaults through ``create_image_view`` — rendering a montage zoomed out far
    enough that a screen pixel covers several source texels.

    The source is a per-texel checkerboard under a (0, 1) window, so a point
    sample can only emit black or white.  Any intermediate grey is proof the
    footprint was averaged.

    Turning ``wgpu_minification_filter`` off anywhere along the chain — the
    dataclass default, ``settings_from_mapping``, the factory, or the view's
    mapping rebuilds — turns this red.
    """

    from arrayscope.app.settings_state import AppSettingsState, settings_from_mapping
    from arrayscope.display.image_view_factory import create_image_view

    # The default has two owners — the dataclass field and the literal
    # ``settings_from_mapping`` falls back to when the key is absent (the
    # convention every sibling setting follows).  Assert both, or a drift
    # between them leaves one of the two uncovered.
    assert AppSettingsState().wgpu_minification_filter is True
    # Only the backend is pinned; every other setting is the shipped default.
    settings = settings_from_mapping({"image_rendering_backend": "wgpu"})
    assert settings.wgpu_minification_filter is True, "the app default is the thing under test"

    view = create_image_view(settings)
    try:
        view.resize(400, 400)
        view.show()
        assert view.wgpuMinificationFilterEnabled() is True
        assert view._wgpu_mapping_state.minification_filter is True

        tile = np.indices((700, 700)).sum(axis=0).astype(np.float32) % 2.0
        geometry = _montage_geometry((700, 700), 2, 2, loaded=4)
        payloads = {i: _payload(i, tile, source_id=("minify-default", i)) for i in range(4)}
        _commit(view, geometry, payloads, levels=(0.0, 1.0))

        # The whole 1400-texel montage across a 768-px target: 1.82 source
        # texels per screen pixel.  Deliberately not a round ratio — at exactly
        # 2.0 the taps land on texel boundaries, where which side they fall on
        # is an f32 tie rather than a filter result.
        montage = (0.0, 0.0, 1400.0, 1400.0)
        filtered = _draw_zoomed_out(view, montage)

        # Same frame, same camera, aid explicitly off: the flag is the only
        # difference, so every changed pixel is the filter's.
        view.setWgpuMinificationFilterEnabled(False)
        point_sampled = _draw_zoomed_out(view, montage)

        assert _pure_grey_midtones(point_sampled) == 0, "a point sample of binary data is binary"
        assert _pure_grey_midtones(filtered) > 100_000
        changed = int(np.any(filtered != point_sampled, axis=-1).sum())
        assert changed > 100_000, changed
    finally:
        view.close()
