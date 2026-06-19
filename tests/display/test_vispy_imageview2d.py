import numpy as np
import pytest


pytest.importorskip("vispy")


def _montage_geometry():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    return DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )


def _single_tile_montage_geometry():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    return DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,),
    )


def _shifted_montage_geometry():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    return DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_origin_x=3,
        montage_origin_y=0,
        montage_tile_states=(
            MontageTileState.LOADED,
            MontageTileState.LOADED,
            MontageTileState.LOADED,
            MontageTileState.LOADED,
        ),
    )


def test_factory_constructs_vispy_backend(qt_app):
    from arrayscope.app.settings_state import AppSettingsState, ImageRenderingBackendChoice
    from arrayscope.display.image_view_factory import create_image_view

    view = create_image_view(AppSettingsState(image_rendering_backend=ImageRenderingBackendChoice.VISPY))
    try:
        assert type(view).__name__ == "VisPyImageView2D"
        assert view.rendering_backend_name == "vispy"
    finally:
        view.close()


def test_scalar_presentation_does_not_mutate_frozen_visual(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 64 * 48, dtype=np.float32).reshape(48, 64)
    try:
        view.setImagePresentation(data, histogramData=data, levels=(0.1, 0.9), histogramRange=(0.0, 1.0))

        timing = view.lastImageUploadTiming()
        assert timing.mode == "vispy_full"
        assert timing.rgb_window_ms == 0.0
        assert view._vispy_main_data_id is not None
    finally:
        view.close()


def test_scalar_level_preview_updates_clim_without_rgb_work(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    try:
        view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        view._apply_histogram_preview_levels((0.25, 0.75))

        timing = view.lastImageUploadTiming()
        assert timing.mode == "vispy_level_preview"
        assert timing.rgb_window_ms == 0.0
        assert timing.visible_bytes == 0
        assert tuple(float(value) for value in view._vispy_image.clim) == (0.25, 0.75)
    finally:
        view.close()


def test_windowed_rgb_presentation_uses_shader_path(qt_app, monkeypatch):
    import arrayscope.display.vispy_imageview2d as vispy_view
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    def fail_cpu_window(*args, **kwargs):
        raise AssertionError("VisPy windowed RGB path should not CPU-window RGB display data")

    monkeypatch.setattr(vispy_view, "rgb_display_for_levels", fail_cpu_window)
    view = VisPyImageView2D()
    rgb = np.full((8, 9, 3), 200, dtype=np.uint8)
    magnitude = np.linspace(0.0, 1.0, 72, dtype=np.float64).reshape(8, 9)
    try:
        view.setImagePresentation(
            rgb,
            histogramData=magnitude,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
        )
        view._apply_histogram_preview_levels((0.5, 1.0))

        assert view._is_windowed_rgb_vispy_main()
        assert view._vispy_windowed_image.levels == (0.5, 1.0)
        timing = view.lastImageUploadTiming()
        assert timing.rgb_window_ms == 0.0
        assert timing.visible_bytes == 0
    finally:
        view.close()


def test_vispy_complex_windowed_rgb_render_has_visible_signal(qt_app):
    from pyqtgraph.Qt import QtGui
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    y, x = np.mgrid[-1.0:1.0:48j, -1.0:1.0:48j]
    complex_data = (x + 1j * y).astype(np.complex64)
    from arrayscope.display.slice_engine import complex_to_rgb

    rgb, magnitude = complex_to_rgb(complex_data)
    from arrayscope.display.image_upload import rgb_display_for_levels

    expected = rgb_display_for_levels(rgb, magnitude, (float(np.nanmin(magnitude)), float(np.nanmax(magnitude))))
    assert len(np.unique(expected.reshape((-1, 3))[::8], axis=0)) > 8
    view = VisPyImageView2D()
    try:
        view.resize(360, 260)
        view.show()
        view.setImagePresentation(
            rgb,
            histogramData=magnitude,
            levels=(float(np.nanmin(magnitude)), float(np.nanmax(magnitude))),
            histogramRange=(float(np.nanmin(magnitude)), float(np.nanmax(magnitude))),
            rgb_already_windowed=False,
        )
        for _ in range(20):
            qt_app.processEvents()

        pixmap = view.grab()
        assert not pixmap.isNull()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[..., :3]
        center = pixels[pixels.shape[0] // 5 : pixels.shape[0] * 4 // 5, pixels.shape[1] // 5 : pixels.shape[1] * 4 // 5]
        assert int(center.max()) > 32
        assert float(center.mean()) > 5.0
        assert len(np.unique(center.reshape((-1, 3))[:: max(1, center.size // 4096)], axis=0)) > 1
    finally:
        view.close()


def test_vispy_raster_complex_display_image_renders_color(qt_app):
    from pyqtgraph.Qt import QtGui
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.slice_engine import make_image
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    y, x = np.mgrid[-1.0:1.0:48j, -1.0:1.0:48j]
    complex_data = (x + 1j * y).astype(np.complex64)
    state = ViewState.from_shape(complex_data.shape).with_channel(ChannelMode.COMPLEX)
    display = make_image(complex_data, state)
    levels = (float(np.nanmin(display.histogram_data)), float(np.nanmax(display.histogram_data)))

    view = VisPyImageView2D()
    try:
        view.resize(360, 260)
        view.show()
        view.setImagePresentation(
            display.data,
            histogramData=display.histogram_data,
            levels=levels,
            histogramRange=levels,
            rgb_already_windowed=display.rgb_already_windowed,
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
        )
        for _ in range(20):
            qt_app.processEvents()

        pixmap = view.grab()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[..., :3]
        center = pixels[pixels.shape[0] // 5 : pixels.shape[0] * 4 // 5, pixels.shape[1] // 5 : pixels.shape[1] * 4 // 5]
        assert int(center.max()) > 32
        assert float(center.mean()) > 5.0
        assert len(np.unique(center.reshape((-1, 3))[:: max(1, len(center.reshape((-1, 3))) // 512)], axis=0)) > 4
    finally:
        view.close()


def test_vispy_raster_mapped_complex_level_change_updates_uniform_without_texture_upload(qt_app):
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.slice_engine import make_shader_image_from_slab
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    class Request:
        def __init__(self, view_state):
            self.view_state = view_state
            self.ranged_axes = ()

    data = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1j).astype(np.complex64)
    state = ViewState.from_shape(data.shape).with_channel(ChannelMode.COMPLEX)
    display = make_shader_image_from_slab(data, Request(state))

    view = VisPyImageView2D()
    try:
        view.setImagePresentation(
            display.data,
            histogramData=display.histogram_data,
            levels=(0.0, 16.0),
            histogramRange=(0.0, 16.0),
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
        )
        visual = view._vispy_windowed_image
        assert visual.upload_count == 1

        view.setImagePresentation(
            display.data,
            histogramData=display.histogram_data,
            levels=(2.0, 8.0),
            histogramRange=(0.0, 16.0),
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
        )

        assert visual.upload_count == 1
        assert visual.levels == (2.0, 8.0)
    finally:
        view.close()


def test_gpu_mapped_visual_shader_supports_raw_complex_components():
    from arrayscope.display.backends.vispy.raster import GpuMappedImageVisual

    shader = GpuMappedImageVisual._fragment_shader

    assert "uniform float u_component_mode" in shader
    assert "float complex_component" in shader
    assert "if (u_component_mode > 2.5)" in shader
    assert "gl_FragColor = vec4(color, 1.0);" in shader


def test_gpu_mapped_visual_cached_complex_component_change_updates_uniform_without_upload():
    from arrayscope.display.shader_mapping import ShaderComponent, ShaderMapping, TexturePlaneKind
    from arrayscope.display.backends.vispy.raster import GpuMappedImageVisual

    visual = object.__new__(GpuMappedImageVisual)
    visual._scalar_texture = object()
    visual.scalar_source_id = ("source", "complex_rg32f")
    visual._mode = 2.0
    visual._scale_mode = 0.0
    visual._symlog_constant = 0.0
    visual._component_mode = 0.0
    visual.upload_count = 3
    visual._set_lut_texture = lambda lut: None
    visual.set_levels = lambda levels, count=True: setattr(visual, "_levels", tuple(float(value) for value in levels))

    visual.set_mapped_data(
        np.ones((2, 2), dtype=np.complex64),
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        levels=(0.0, 1.0),
        source_id=("source", "complex_rg32f"),
        shader_mapping=ShaderMapping(component=ShaderComponent.IMAG),
    )

    assert visual.upload_count == 3
    assert visual._component_mode == 1.0
    assert visual._levels == (0.0, 1.0)


def test_vispy_complex_windowed_rgb_preserves_high_magnitude_scale(qt_app):
    from pyqtgraph.Qt import QtGui
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.slice_engine import complex_to_rgb

    y, x = np.mgrid[-1.0:1.0:64j, -1.0:1.0:64j]
    phase = np.arctan2(y, x)
    magnitude = (10_000.0 + 2_000.0 * np.cos(phase)).astype(np.float32)
    complex_data = (magnitude * np.exp(1j * phase)).astype(np.complex64)
    rgb, scalar = complex_to_rgb(complex_data)
    levels = (9_000.0, 12_000.0)

    view = VisPyImageView2D()
    try:
        view.resize(360, 260)
        view.show()
        view.setImagePresentation(
            rgb,
            histogramData=scalar,
            levels=levels,
            histogramRange=(float(np.nanmin(scalar)), float(np.nanmax(scalar))),
            rgb_already_windowed=False,
        )
        for _ in range(20):
            qt_app.processEvents()

        visual = view._vispy_windowed_image
        assert getattr(visual._scalar_texture, "_internalformat", None) == "r32f"
        assert getattr(visual._scalar_texture, "_format", None) == "red"
        assert visual._scalar_texture.shape[-1] == 1

        pixmap = view.grab()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[..., :3]
        center = pixels[pixels.shape[0] // 4 : pixels.shape[0] * 3 // 4, pixels.shape[1] // 4 : pixels.shape[1] * 3 // 4]
        assert int(center.max()) > 64
        assert float(center.mean()) > 12.0
    finally:
        view.close()


def test_vispy_tile_layer_uses_windowed_visuals_and_positions_loaded_tiles(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    rgb = np.full((2, 5, 3), 180, dtype=np.uint8)
    magnitude = np.arange(10, dtype=np.float32).reshape(2, 5)
    try:
        view.setMontageTileLayerPresentation(
            rgb,
            histogramData=magnitude,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
        )

        states = view._vispy_tile_visuals
        assert set(states) == {0, 1}
        assert all(state.windowed_rgb for state in states.values())
        assert states[0].windowed_visual.visible
        assert tuple(float(value) for value in states[0].windowed_visual.transform.translate[:2]) == (0.0, 0.0)
        assert tuple(float(value) for value in states[1].windowed_visual.transform.translate[:2]) == (3.0, 0.0)
    finally:
        view.close()


def test_vispy_tile_layer_bounds_cover_full_montage_not_viewport_canvas(qt_app):
    from arrayscope.display.viewport import ViewportPolicy
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    rgb = np.full((2, 5, 3), 180, dtype=np.uint8)
    magnitude = np.arange(10, dtype=np.float32).reshape(2, 5)
    try:
        view.resize(420, 220)
        view.show()
        view.setMontageTileLayerPresentation(
            rgb,
            histogramData=magnitude,
            histogramPlotData=None,
            geometry=_shifted_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            viewport_policy=ViewportPolicy.FIT_ONCE,
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={index: ("tile", index) for index in range(4)},
        )

        rect = view._vispy_bounds_item.rect()
        assert (rect.left(), rect.top()) == (0.0, 0.0)
        assert (rect.width(), rect.height()) == (11.0, 2.0)
        assert view._vispy_display_shape == (2, 11)
        assert view.viewport_controller.last_display_shape == (2, 11)
        assert view._current_image_world_rect() == (0.0, 0.0, 10.0, 1.0)
    finally:
        view.close()


def test_vispy_tile_layer_level_preview_updates_uniforms_without_upload(qt_app, monkeypatch):
    import arrayscope.display.vispy_imageview2d as vispy_view
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    def fail_cpu_window(*args, **kwargs):
        raise AssertionError("VisPy tile shader level preview should not CPU-window RGB tiles")

    monkeypatch.setattr(vispy_view, "rgb_display_for_levels", fail_cpu_window)
    view = VisPyImageView2D()
    rgb = np.full((2, 5, 3), 180, dtype=np.uint8)
    magnitude = np.arange(10, dtype=np.float32).reshape(2, 5)
    try:
        view.setMontageTileLayerPresentation(
            rgb,
            histogramData=magnitude,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
        )
        view._apply_histogram_preview_levels((4.0, 9.0))

        timing = view.lastImageUploadTiming()
        assert timing.mode == "vispy_level_preview"
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 2
        assert timing.tile_layer_rgb_window_tiles == 0
        assert timing.tile_layer_upload_ms == 0.0
        assert timing.visible_bytes == 0
        assert all(state.windowed_visual.levels == (4.0, 9.0) for state in view._vispy_tile_visuals.values())
    finally:
        view.close()


def test_vispy_tile_layer_clean_flush_skips_existing_visual_uploads(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    rgb = np.full((2, 5, 3), 180, dtype=np.uint8)
    magnitude = np.arange(10, dtype=np.float32).reshape(2, 5)
    sources = {0: ("tile", 0), 1: ("tile", 1)}
    try:
        kwargs = dict(
            histogramData=magnitude,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            rgb_already_windowed=False,
            montage_tile_source_ids=sources,
        )
        view.setMontageTileLayerPresentation(rgb, montage_dirty_tiles=None, **kwargs)
        view.setMontageTileLayerPresentation(rgb, montage_dirty_tiles=(), **kwargs)

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 2
        assert timing.tile_layer_upload_ms == 0.0
        assert timing.visible_bytes == 0
    finally:
        view.close()


def test_vispy_direct_tiled_payloads_use_batched_gpu_layer(qt_app):
    from pyqtgraph.Qt import QtGui
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    left = np.full((2, 2, 3), 180, dtype=np.uint8)
    right = np.full((2, 2, 3), 90, dtype=np.uint8)
    left_hist = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    right_hist = np.array([[1.0, 0.5], [0.25, 0.0]], dtype=np.float32)
    payloads = {
        0: DisplayTilePayload(0, 0, left, left_hist, ("tile", 0)),
        1: DisplayTilePayload(1, 1, right, right_hist, ("tile", 1)),
    }
    placeholder = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (2, 5, 3))
    try:
        view.resize(360, 240)
        view.show()
        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
            montage_tile_payloads=payloads,
        )

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 2
        assert timing.tile_layer_rgb_window_tiles == 0
        assert not any(state.visible for state in view._vispy_tile_visuals.values())
        assert view._vispy_gpu_montage_layer.visual.visible
        for _ in range(20):
            qt_app.processEvents()
        pixmap = view.grab()
        assert not pixmap.isNull()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[..., :3]
        center = pixels[pixels.shape[0] // 5 : pixels.shape[0] * 4 // 5, pixels.shape[1] // 5 : pixels.shape[1] * 4 // 5]
        assert int(center.max()) > 16
        assert float(center.mean()) > 2.0
    finally:
        view.close()


def test_vispy_direct_tiled_complex_display_images_render_nonblank(qt_app):
    from pyqtgraph.Qt import QtGui
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.slice_engine import make_image
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    base = np.array([[1 + 0j, 1j], [-1 + 0j, -1j]], dtype=np.complex64)
    state = ViewState.from_shape(base.shape).with_channel(ChannelMode.COMPLEX)
    left = make_image(base, state)
    right = make_image(base * (1.0 + 0.25j), state)
    payloads = {
        0: DisplayTilePayload(
            tile_number=0,
            source_index=0,
            image=left.data,
            histogram_data=left.histogram_data,
            source_id=("complex", 0),
            texture_data=left.semantic_data,
            texture_kind=left.texture_kind,
            semantic_data=left.semantic_data,
            semantic_histogram_data=left.histogram_data,
            shader_mapping=left.shader_mapping,
        ),
        1: DisplayTilePayload(
            tile_number=1,
            source_index=1,
            image=right.data,
            histogram_data=right.histogram_data,
            source_id=("complex", 1),
            texture_data=right.semantic_data,
            texture_kind=right.texture_kind,
            semantic_data=right.semantic_data,
            semantic_histogram_data=right.histogram_data,
            shader_mapping=right.shader_mapping,
        ),
    }
    placeholder = np.zeros((2, 5, 3), dtype=np.uint8)
    view = VisPyImageView2D()
    try:
        view.resize(360, 240)
        view.show()
        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, float(np.nanmax(left.histogram_data))),
            histogramRange=(0.0, float(np.nanmax(left.histogram_data))),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("complex", 0), 1: ("complex", 1)},
            montage_tile_payloads=payloads,
        )
        for _ in range(20):
            qt_app.processEvents()

        pixmap = view.grab()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[..., :3]
        center = pixels[pixels.shape[0] // 5 : pixels.shape[0] * 4 // 5, pixels.shape[1] // 5 : pixels.shape[1] * 4 // 5]
        assert int(center.max()) > 16
        assert float(center.mean()) > 2.0
        sampled = center.reshape((-1, 3))[:: max(1, len(center.reshape((-1, 3))) // 512)]
        chroma = np.max(np.abs(sampled.astype(np.int16) - sampled.mean(axis=1, keepdims=True).astype(np.int16)), axis=1)
        assert int(chroma.max()) > 16
        assert int(np.count_nonzero(chroma > 16)) > 0
    finally:
        view.close()


def test_vispy_direct_tiled_clean_and_dirty_counters(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2, 3), 90, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 1)),
    }
    placeholder = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (2, 5, 3))
    kwargs = dict(
        histogramData=None,
        histogramPlotData=None,
        geometry=_montage_geometry(),
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
    )
    try:
        view.setMontageTileLayerPresentation(
            placeholder,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
            montage_tile_payloads=payloads,
            **kwargs,
        )
        view.setMontageTileLayerPresentation(
            placeholder,
            montage_dirty_tiles=(),
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
            montage_tile_payloads=payloads,
            **kwargs,
        )
        clean = view.lastImageUploadTiming()
        assert clean.tile_layer_items_updated == 0
        assert clean.tile_layer_items_skipped == 2
        assert clean.visible_bytes == 0
        assert clean.tile_layer_resident_items >= 2
        assert clean.tile_layer_storage_capacity >= 2
        assert clean.tile_layer_estimated_gpu_bytes > 0
        assert clean.tile_layer_cpu_shadow_bytes == 0

        dirty_payloads = dict(payloads)
        dirty_payloads[1] = DisplayTilePayload(
            1,
            1,
            np.full((2, 2, 3), 128, dtype=np.uint8),
            np.ones((2, 2), dtype=np.float32),
            ("tile", 1, "dirty"),
        )
        view.setMontageTileLayerPresentation(
            placeholder,
            montage_dirty_tiles=(1,),
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1, "dirty")},
            montage_tile_payloads=dirty_payloads,
            **kwargs,
        )
        dirty = view.lastImageUploadTiming()
        assert dirty.tile_layer_items_updated == 1
        assert dirty.tile_layer_items_skipped == 1
        assert dirty.visible_bytes > 0
        assert dirty.tile_layer_texture_uploads >= 1
        assert dirty.tile_layer_texture_upload_bytes > 0
    finally:
        view.close()


def test_vispy_direct_tiled_shader_mapping_change_updates_uniform_without_texture_upload(qt_app):
    from arrayscope.display.shader_mapping import ShaderComponent, ShaderMapping, TexturePlaneKind
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    texture = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    histogram = np.abs(texture).astype(np.float32)
    source_id = ("complex-source", 0)
    first = DisplayTilePayload(
        0,
        0,
        texture,
        histogram,
        source_id,
        texture_data=texture,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=texture,
        semantic_histogram_data=histogram,
        shader_mapping=ShaderMapping(component=ShaderComponent.ABS),
    )
    second = DisplayTilePayload(
        0,
        0,
        texture,
        histogram,
        source_id,
        texture_data=texture,
        texture_kind=TexturePlaneKind.COMPLEX_RG32F,
        semantic_data=texture,
        semantic_histogram_data=histogram,
        shader_mapping=ShaderMapping(component=ShaderComponent.IMAG),
    )
    placeholder = np.zeros((2, 2), dtype=np.float32)
    view = VisPyImageView2D()
    try:
        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=_single_tile_montage_geometry(),
            levels=(0.0, 10.0),
            histogramRange=(0.0, 10.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: source_id},
            montage_tile_payloads={0: first},
        )
        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=_single_tile_montage_geometry(),
            levels=(0.0, 10.0),
            histogramRange=(0.0, 10.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=(),
            montage_tile_source_ids={0: source_id},
            montage_tile_payloads={0: second},
        )

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_texture_uploads == 0
        assert timing.tile_layer_texture_upload_bytes == 0
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_shader_uniform_updates > 0
    finally:
        view.close()


def test_vispy_direct_tiled_fit_syncs_camera_immediately(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    geometry = _montage_geometry()
    payloads = {
        0: DisplayTilePayload(0, 0, np.ones((2, 2), dtype=np.float32), None, ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.ones((2, 2), dtype=np.float32) * 2.0, None, ("tile", 1)),
    }
    placeholder = np.zeros((2, 5), dtype=np.float32)
    view = VisPyImageView2D()
    try:
        view.resize(360, 240)
        view.show()
        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=geometry,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
            montage_tile_payloads=payloads,
        )
        view.getView().setRange(xRange=(1.0, 2.0), yRange=(0.0, 1.0), padding=0)
        qt_app.processEvents()

        view.setFitLocked(True)
        x_range, y_range = view.getView().viewRange()
        expected = ((0.0, 4.0), (0.0, 1.0))

        np.testing.assert_allclose(x_range, expected[0], atol=1e-6)
        np.testing.assert_allclose(y_range, expected[1], atol=1e-6)
        assert view._vispy_view.camera.aspect is None
        assert view._vispy_camera_key[:2] == expected

        view.oneToOne()
        assert view._vispy_view.camera.aspect == 1.0
    finally:
        view.close()


def test_vispy_first_class_tiled_new_semantic_state_reuses_resident_textures(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2, 3), 90, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 1)),
    }

    def delta(revision: int):
        return TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=revision,
            upserts=payloads,
            active_tiles=(0, 1),
            planned_tiles=(0, 1),
            near_tiles=(0, 1),
        )

    kwargs = dict(
        geometry=_montage_geometry(),
        histogramPlotData=None,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        tile_residency_budget_bytes=64 * 1024 * 1024,
    )
    try:
        view.setTiledMontagePresentation(tile_state=TilePresentationState(payloads), tile_delta=delta(1), **kwargs)
        first = view.lastImageUploadTiming()
        assert first.tile_layer_items_updated == 2
        assert first.visible_bytes > 0

        view.setTiledMontagePresentation(tile_state=TilePresentationState(payloads), tile_delta=delta(2), **kwargs)
        clean = view.lastImageUploadTiming()

        assert clean.tile_layer_resident_items == 2
        assert clean.tile_layer_items_updated == 0
        assert clean.tile_layer_items_skipped == 2
        assert clean.tile_layer_texture_uploads == 0
        assert clean.visible_bytes == 0
    finally:
        view.close()


def test_vispy_first_class_tiled_shifted_window_reuses_resident_sources(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=4, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 11),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=4, rows=1, gap=1),
        montage_tile_states=(
            MontageTileState.LOADED,
            MontageTileState.LOADED,
            MontageTileState.LOADED,
            MontageTileState.LOADED,
        ),
    )
    view = VisPyImageView2D()
    sources = {index: ("source", index) for index in range(4)}
    initial_payloads = {
        index: DisplayTilePayload(
            index,
            index,
            np.full((2, 2), float(index), dtype=np.float32),
            None,
            sources[index],
        )
        for index in range(4)
    }
    shifted_payloads = {
        0: DisplayTilePayload(0, 2, np.full((2, 2), 2.0, dtype=np.float32), None, sources[2]),
        1: DisplayTilePayload(1, 3, np.full((2, 2), 3.0, dtype=np.float32), None, sources[3]),
    }

    def delta(revision: int, payloads, active_tiles):
        return TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=revision,
            upserts=payloads,
            active_tiles=tuple(active_tiles),
            planned_tiles=(0, 1, 2, 3),
            near_tiles=(0, 1, 2, 3),
        )

    kwargs = dict(
        geometry=geometry,
        histogramPlotData=None,
        levels=(0.0, 4.0),
        histogramRange=(0.0, 4.0),
        rgb_already_windowed=False,
        tile_residency_budget_bytes=64 * 1024 * 1024,
    )
    try:
        view.setTiledMontagePresentation(
            tile_state=TilePresentationState(initial_payloads),
            tile_delta=delta(1, initial_payloads, (0, 1, 2, 3)),
            **kwargs,
        )
        first = view.lastImageUploadTiming()
        assert first.tile_layer_items_updated == 4
        assert first.tile_layer_resident_items == 4

        view.setTiledMontagePresentation(
            tile_state=TilePresentationState(shifted_payloads),
            tile_delta=delta(2, shifted_payloads, (0, 1)),
            **kwargs,
        )
        shifted = view.lastImageUploadTiming()

        assert shifted.tile_layer_items_updated == 0
        assert shifted.tile_layer_items_skipped == 2
        assert shifted.tile_layer_texture_uploads == 0
        assert shifted.visible_bytes == 0
        assert shifted.tile_layer_resident_items == 4
    finally:
        view.close()


def test_vispy_first_class_tiled_warms_loaded_near_sources_after_visible_commit(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 3)).with_montage_axis(2, columns=3, indices=(0, 1, 2), text=":"),
        display_shape=(2, 8),
        montage=MontageGeometry(indices=(0, 1, 2), tile_shape=(2, 2), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED, MontageTileState.LOADED),
    )
    payloads = {
        index: DisplayTilePayload(index, index, np.full((2, 2), float(index), dtype=np.float32), None, ("source", index))
        for index in range(3)
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0,),
        planned_tiles=(0, 1, 2),
        near_tiles=(0, 1, 2),
        near_tile_source_ids={index: payload.source_id for index, payload in payloads.items()},
    )
    view = VisPyImageView2D()
    try:
        view.setTiledMontagePresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=delta,
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )
        visible = view.lastImageUploadTiming()
        assert visible.tile_layer_items_updated == 1
        assert visible.tile_layer_resident_items == 1

        for _ in range(10):
            qt_app.processEvents()

        warm = view._last_vispy_warm_tile_stats
        assert warm is not None
        assert warm.items_updated == 2
        assert warm.resident_items == 3
        assert view.lastImageUploadTiming().visible_bytes == visible.visible_bytes
    finally:
        view.close()


def test_vispy_direct_tiled_level_change_skips_structural_refresh(qt_app, monkeypatch):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 0.25, dtype=np.float32), None, ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 0.75, dtype=np.float32), None, ("tile", 1)),
    }
    sources = {tile: payload.source_id for tile, payload in payloads.items()}
    placeholder = np.broadcast_to(np.zeros((1, 1), dtype=np.float32), (2, 5))
    try:
        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=None,
            montage_tile_source_ids=sources,
            montage_tile_payloads=payloads,
        )

        def fail_structure(*_args, **_kwargs):
            raise AssertionError("level-only commit repeated structural display work")

        monkeypatch.setattr(view, "_update_profile_line_bounds", fail_structure)
        monkeypatch.setattr(view, "_updateAspectRatio", fail_structure)
        monkeypatch.setattr(view, "_sync_vispy_montage_bounds", fail_structure)
        monkeypatch.setattr(view, "_apply_viewport_policy", fail_structure)
        monkeypatch.setattr(view, "_update_histogram_for_vispy", fail_structure)

        view.setMontageTileLayerPresentation(
            placeholder,
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.2, 0.8),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=(),
            montage_tile_source_ids=sources,
            montage_tile_payloads=payloads,
        )

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_items_updated == 0
        assert timing.visible_bytes == 0
        assert view._vispy_gpu_montage_layer._levels == (0.2, 0.8)
    finally:
        view.close()


def test_vispy_direct_tiled_scalar_atlas_preserves_high_dynamic_range(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((3, 4, 31)).with_montage_axis(2, columns=3, indices=(10, 20, 30), text=":"),
        display_shape=(3, 14),
        montage=MontageGeometry(indices=(10, 20, 30), tile_shape=(3, 4), columns=3, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED, MontageTileState.LOADED),
    )
    payloads = {
        0: DisplayTilePayload(0, 10, np.full((3, 4), 250.0, dtype=np.float32), None, ("tile", 10)),
        1: DisplayTilePayload(1, 20, np.full((3, 4), 1000.0, dtype=np.float32), None, ("tile", 20)),
        2: DisplayTilePayload(2, 30, np.full((3, 4), 4096.0, dtype=np.float32), None, ("tile", 30)),
    }
    try:
        view.setMontageTileLayerPresentation(
            np.zeros((3, 14), dtype=np.float32),
            histogramData=None,
            histogramPlotData=None,
            geometry=geometry,
            levels=(0.0, 4096.0),
            histogramRange=(0.0, 4096.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={index: payload.source_id for index, payload in payloads.items()},
            montage_tile_payloads=payloads,
        )

        layer = view._vispy_gpu_montage_layer
        pool = layer._pool
        assert pool.cpu_shadow_bytes == 0
        assert pool.resident_count == 3
        assert pool.scalar_texture._format == "red"
        assert pool.scalar_texture._internalformat == "r32f"
        assert tuple(pool.scalar_texture.shape[-1:]) == (1,)

        vertices = layer.visual.vertex_data.reshape((-1, 6, 2))
        expected_origins = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(vertices[:, 0, :], expected_origins)
        assert layer.visual.mode_data.reshape((-1, 6))[:, 0].tolist() == [0.0, 0.0, 0.0]
    finally:
        view.close()


def test_vispy_mouse_bridge_emits_qpointf_for_pyqtgraph_scene(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    class Event:
        pos = (5.0, 7.0)

    view = VisPyImageView2D()
    received = []
    view.view.scene().sigMouseMoved.connect(received.append)
    try:
        view._on_vispy_mouse_move(Event())

        assert received
        assert isinstance(received[-1], QtCore.QPointF)
        mapped = view.view.mapSceneToView(received[-1])
        assert isinstance(mapped, QtCore.QPointF)
    finally:
        view.close()


def test_vispy_canvas_is_passive_for_pyqtgraph_interaction(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 40 * 40, dtype=np.float32).reshape(40, 40)
    try:
        assert view._vispy_view.camera.interactive is False
        assert view._vispy_view.camera.flip == (False, True, False)
        assert view._vispy_canvas_native.testAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        view.resize(420, 260)
        view.show()
        view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        view.getView().setRange(xRange=(0.0, 40.0), yRange=(0.0, 40.0), padding=0)
        assert view.getView().viewRange()[0][1] > view.getView().viewRange()[0][0]

        view.getView().invertY(False)
        view.getView().invertX(True)
        view._sync_vispy_camera_to_view()
        assert view._vispy_view.camera.flip == (True, False, False)

        view.getView().invertY(True)
        view.getView().invertX(False)
        view._sync_vispy_camera_to_view()
        assert view._vispy_view.camera.flip == (False, True, False)
    finally:
        view.close()


def test_vispy_widget_overlays_are_parented_above_gl_surface(qt_app):
    from pyqtgraph.Qt import QtWidgets
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    class Hud(QtWidgets.QLabel):
        def show_text_near(self, text, pos):
            self.setText(str(text))
            self.move(pos)
            self.show()

    view = VisPyImageView2D()
    hud = Hud()
    try:
        view.setHudWidget(hud)
        view.setEvaluationOverlay(True, "Rendering")
        view.setRoiInfoText("Rectangle 1: n=4 mean=1")

        assert hud.parentWidget() is view._display_container
        assert view._evaluation_overlay.parentWidget() is view._display_container
        assert view._roi_info_panel.parentWidget() is view._display_container
        assert not view._evaluation_overlay.isHidden()
        assert not view._roi_info_panel.isHidden()
    finally:
        view.close()


def test_vispy_roi_visuals_mirror_pyqtgraph_rois(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    try:
        view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("rectangle", rect=(3.0, 4.0, 8.0, 6.0), color=(255, 32, 16))

        visual = view._vispy_roi_visuals.get(selection.id)
        assert visual is not None
        assert visual.visible
        assert visual.order == 10_000
        handle_visuals = view._vispy_roi_handle_visuals.get(selection.id)
        assert handle_visuals is not None
        assert len(handle_visuals) == 1
        assert all(handle.visible for handle in handle_visuals)
        assert all(handle.order == 10_001 for handle in handle_visuals)

        view.highlightRoi(selection.id)
        assert view._vispy_roi_visuals[selection.id].visible
        assert all(handle.visible for handle in view._vispy_roi_handle_visuals[selection.id])
        assert view._vispy_roi_cursor_for_point(6.0, 7.0).shape() == QtCore.Qt.CursorShape.SizeAllCursor
        assert view._vispy_roi_cursor_for_point(11.0, 10.0).shape() == QtCore.Qt.CursorShape.SizeFDiagCursor

        assert view.removeRoi(selection.id)
        assert selection.id not in view._vispy_roi_visuals
        assert selection.id not in view._vispy_roi_handle_visuals
    finally:
        view.close()


def test_vispy_roi_visuals_update_during_live_region_changes(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    changed = []
    view.roiChanged.connect(lambda roi_id, geometry: changed.append((roi_id, geometry)))
    try:
        view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("rectangle", rect=(3.0, 4.0, 8.0, 6.0), color=(255, 32, 16))
        item, _selection = view._roi_items[selection.id]
        handle_visual = view._vispy_roi_handle_visuals[selection.id][0]

        item.setPos(5.0, 7.0)
        qt_app.processEvents()

        assert len(changed) == 1
        live_selection = dict((roi.id, roi) for roi in view.roiSelections())[selection.id]
        assert live_selection.geometry.rect[:2] == (5.0, 7.0)
        assert view._vispy_roi_visuals[selection.id].visible
        assert view._vispy_roi_handle_visuals[selection.id][0].visible
        assert view._vispy_roi_handle_visuals[selection.id][0] is handle_visual
    finally:
        view.close()


def test_vispy_line_roi_has_reused_endpoint_handles_and_hover_cursor(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    try:
        view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("line", points=((3.0, 4.0), (11.0, 8.0)), color=(40, 190, 255))
        marker = view._vispy_roi_handle_visuals[selection.id][0]

        assert marker.visible
        assert view._vispy_roi_cursor_for_point(3.0, 4.0).shape() == QtCore.Qt.CursorShape.SizeAllCursor
        assert view._vispy_roi_cursor_for_point(7.0, 6.0).shape() == QtCore.Qt.CursorShape.SizeAllCursor

        item, _selection = view._roi_items[selection.id]
        item.setPos(1.0, 2.0)
        qt_app.processEvents()

        assert view._vispy_roi_handle_visuals[selection.id][0] is marker
    finally:
        view.close()


def test_vispy_freehand_drawing_preview_reuses_one_visual(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        view._set_roi_drawing_preview("roi_freehand", ((1.0, 1.0), (3.0, 2.0)))
        preview = view._vispy_roi_drawing_preview
        assert preview is not None
        assert preview.visible

        view._set_roi_drawing_preview("roi_freehand", ((1.0, 1.0), (3.0, 2.0), (5.0, 4.0)))
        assert view._vispy_roi_drawing_preview is preview

        view._set_roi_drawing_preview(None, ())
        assert not preview.visible
    finally:
        view.close()


def test_vispy_profile_marker_has_vispy_crosshair_visuals(qt_app):
    from pyqtgraph.Qt import QtCore
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    moved = []
    view.set_profile_marker_callback(lambda x, y: moved.append((x, y)))
    try:
        view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        view.setProfileMarker(8.0, 9.0, visible=True)

        assert {"profile_v", "profile_h", "profile_handle_x", "profile_handle_y", "profile_handle_dot"} <= set(view._vispy_profile_visuals)
        assert all(visual.visible for visual in view._vispy_profile_visuals.values())
        assert view._vispy_profile_visuals["profile_handle_dot"].order == 10_002
        assert view._vispy_profile_cursor_for_point(8.0, 9.0).shape() == QtCore.Qt.CursorShape.OpenHandCursor

        view._profile_handle.setPos(10.0, 11.0)
        qt_app.processEvents()

        assert moved
        assert moved[-1] == (10.0, 11.0)
        assert view.profileMarkerPosition() == (10.0, 11.0)
        assert all(visual.visible for visual in view._vispy_profile_visuals.values())
        assert view._vispy_profile_cursor_for_point(10.0, 11.0).shape() == QtCore.Qt.CursorShape.OpenHandCursor

        view.hideProfileMarker()
        assert all(not visual.visible for visual in view._vispy_profile_visuals.values())
    finally:
        view.close()


def test_vispy_montage_tile_overlays_have_vispy_placeholder_visuals(qt_app):
    from arrayscope.display.imageview2d import MontageTileOverlay
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        overlays = (
            MontageTileOverlay(0, 0, 4, 4, "loading", "Loading"),
            MontageTileOverlay(5, 0, 4, 4, "skipped", "Skipped"),
        )
        view.setMontageTileOverlays(overlays)

        assert view.montageTileOverlayCount() == 2
        assert len(view._vispy_overlay_visuals) == 2
        assert all(visual.visible for visual in view._vispy_overlay_visuals)
        visuals = tuple(view._vispy_overlay_visuals)

        view.setMontageTileOverlays(overlays)
        assert tuple(view._vispy_overlay_visuals) == visuals

        view.clearMontageTileOverlays()
        assert view.montageTileOverlayCount() == 0
        assert tuple(view._vispy_overlay_visuals) == visuals
        assert all(not visual.visible for visual in view._vispy_overlay_visuals)
    finally:
        view.close()
