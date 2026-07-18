"""Targeted wgpu live-view tests: montage acks + complex/RGB commit modes.

Offscreen, adapter-skip pattern (mirrors ``test_imagesurface_contract``):
these pin the queue row 3(b) montage/complex slice — physical-truth per-tile
acknowledgement, content-keyed zero-upload behavior across montage scrolls
and mode/levels switches, the phase LUT, RGB display-ready bytes, and the
loud rejections at the honest scope boundary.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from tests.display.test_imageview2d import _present_tiled, _view_class


def _wgpu_adapter_available() -> bool:
    try:
        import wgpu
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        try:
            # Vulkan-only instance BEFORE the first adapter request: letting
            # the probe create an all-backends instance makes GL adapter
            # enumeration re-init EGL, which SIGABRTs in workers that hold
            # live vispy GL state (gate-B Tier 0; full-suite crash 2026-07-18).
            set_instance_extras(backends=["Vulkan"])
        except RuntimeError:
            pass  # instance already exists
        wgpu.gpu.request_adapter_sync(power_preference="low-power")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _wgpu_adapter_available(), reason="no wgpu adapter on this machine"
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


def _payload(tile_number, image, *, source_id, shader_mapping=None, texture_kind=None):
    from arrayscope.display.model.frame import DisplayTilePayload

    return DisplayTilePayload(
        tile_number,
        tile_number,
        image,
        None,
        source_id,
        shader_mapping=shader_mapping,
        texture_kind=texture_kind,
    )


def _lod_payload(
    tile_number,
    image,
    *,
    base_source_id,
    level,
    source_shape,
    payload_source_shape=None,
):
    from arrayscope.display.lod import LodInfo
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.model.tile_identity import TileIdentity, TileLodIdentity
    from arrayscope.display.shader_mapping import TexturePlaneKind

    factor = 1 << int(level)
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


def _shown_view(qt_app):
    view = _view_class("wgpu")()
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

    view._submit_wgpu((UpdateTileInstances(view._wgpu_camera_tiles()),))


def _center_pixel(view):
    target = view._wgpu_executor.read_target()
    h, w = target.shape[:2]
    return target[h // 2, w // 2]


def test_montage_commit_acks_per_tile_and_scrolls_zero_upload(qt_app):
    view = _shown_view(qt_app)
    try:
        rng = np.random.default_rng(11)
        images = {
            name: rng.random((20, 30), dtype=np.float32) for name in ("p0", "p1", "p2")
        }
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
        assert report.texture_uploads == 0

        # Montage scroll: p1 moves to tile 0, new content p2 enters tile 1.
        # Only the genuinely new plane uploads; p1 stays warm across rebind.
        scrolled = {
            0: _payload(0, images["p1"], source_id=("wgpu-montage", "p1")),
            1: _payload(1, images["p2"], source_id=("wgpu-montage", "p2")),
        }
        report = _commit(view, geometry, scrolled, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.texture_uploads == 1

        # Scroll back: every plane was seen before — zero upload.
        report = _commit(view, geometry, payloads, levels=(0.0, 1.0))
        assert set(report.presented_tiles) == {0, 1}
        assert report.texture_uploads == 0
    finally:
        view.close()


def test_coarse_payload_falls_back_then_native_payload_refines_same_plane(qt_app):
    view = _shown_view(qt_app)
    try:
        source_shape = (512, 512)
        geometry = _montage_geometry(source_shape, 1, 1, loaded=1)
        coarse = _lod_payload(
            0,
            np.full((256, 256), 0.25, np.float32),
            base_source_id="lod-plane",
            level=1,
            source_shape=source_shape,
            payload_source_shape=(256, 256),
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
        assert view._wgpu_executor._bound_planes[0].max_lod == 1
        assert view._wgpu_camera_tiles()[0].lod_level == 0
        assert view._wgpu_camera_tiles()[0].src_size == (512.0, 512.0)
        view.getView().setRange(xRange=(0, 512), yRange=(0, 512), padding=0)
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 64, atol=2)

        coarse_keys = set(view._wgpu_executor.page_table.resident_keys())
        report = _commit(view, geometry, {0: fine}, levels=(0.0, 1.0))
        assert report.texture_uploads == 4
        assert report.presented_identities == {0: fine.tile_identity}
        assert view._wgpu_executor._bound_planes[0].max_lod == 1
        assert coarse_keys <= set(view._wgpu_executor.page_table.resident_keys())
        assert all(view._wgpu_executor.page_table.is_pinned(key) for key in coarse_keys)
        assert {
            key.document_generation
            for key in view._wgpu_executor.page_table.resident_keys()
        } == {view._wgpu_executor._bound_planes[0].document_generation}
        _rerender_internal(view)
        assert np.allclose(_center_pixel(view)[:3], 204, atol=2)
    finally:
        view.close()


def test_partial_residency_acknowledges_only_resident_tiles(qt_app):
    from arrayscope.display.wgpu_imageview2d import _shared_wgpu_device
    from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor

    view = _shown_view(qt_app)
    try:
        # One-layer scalar pool: the second tile's upload must evict the
        # first tile's page inside the same submission.
        small = WgpuPlaneExecutor(
            pool_layers={"scalar_r32f": 1}, device=_shared_wgpu_device()
        )
        view._wgpu_executor = small
        view._ensure_wgpu_executor = lambda required: small

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
            return ShaderMapping(
                component=component, display_mode=ShaderDisplayMode.COMPLEX
            )

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
                    shader_mapping=ShaderMapping(
                        scale=scale, symlog_constant=symlog_constant
                    ),
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
        assert np.allclose(
            _center_pixel(view), (*([expected] * 3), 255), atol=2
        )
    finally:
        view.close()


def test_rgb_display_ready_tile_renders_raw_bytes(qt_app):
    view = _shown_view(qt_app)
    try:
        color = np.array([10, 200, 60], np.uint8)
        image = np.broadcast_to(color, (20, 30, 3)).copy()
        geometry = _montage_geometry((20, 30), 1, 1, loaded=1)
        payloads = {0: _payload(0, image, source_id=("wgpu-rgb", 1))}
        report = _commit(
            view, geometry, payloads, levels=(0.0, 1.0), rgb_already_windowed=True
        )
        assert set(report.presented_tiles) == {0}
        view.getView().setRange(xRange=(0, 30), yRange=(0, 20), padding=0)
        _rerender_internal(view)
        # Display-ready bytes: levels/LUT bypassed, rendered as-is.
        assert (_center_pixel(view) == (*color, 255)).all()
    finally:
        view.close()


def test_out_of_scope_commits_reject_loudly(qt_app):
    view = _shown_view(qt_app)
    try:
        scalar = np.zeros((20, 30), np.float32)
        cplx = np.zeros((20, 30), np.complex64)
        rgb = np.zeros((20, 30, 3), np.uint8)
        geometry2 = _montage_geometry((20, 30), 2, 1, loaded=2)
        geometry1 = _montage_geometry((20, 30), 1, 1, loaded=1)

        # Complex montage (>1 tile) is row 3c work.
        with pytest.raises(NotImplementedError, match="complex"):
            _commit(
                view,
                geometry2,
                {
                    0: _payload(0, cplx, source_id=("rej", 0)),
                    1: _payload(1, cplx.copy(), source_id=("rej", 1)),
                },
                levels=(0.0, 1.0),
            )
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
        # RGB without display-ready semantics would need shader windowing.
        with pytest.raises(NotImplementedError, match="display-ready"):
            _commit(
                view,
                geometry1,
                {0: _payload(0, rgb, source_id=("rej", 4))},
                levels=(0.0, 1.0),
                rgb_already_windowed=False,
            )
        # Float RGB does not fit rgb8 cleanly — do not guess.
        with pytest.raises(NotImplementedError, match="rgb8"):
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
        # The rejected commits must not have left a half-presented surface.
        assert view.montageDisplayMode() == "none"
    finally:
        view.close()
