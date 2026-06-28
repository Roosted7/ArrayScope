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


def _present_vispy_tiled(
    view,
    placeholder=None,
    *,
    histogramData=None,
    histogramPlotData=None,
    geometry=None,
    levels,
    histogramRange,
    viewport_policy=None,
    rgb_already_windowed=False,
    montage_dirty_tiles=None,
    montage_tile_source_ids=None,
    montage_tile_payloads=None,
    shader_mapping=None,
    texture_kind=None,
    semantic_data=None,
    tile_delta=None,
    tile_residency_budget_bytes=64 * 1024 * 1024,
    frame_plan=None,
):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.viewport import ViewportPolicy

    if geometry is None:
        geometry = _single_vispy_tile_geometry(placeholder)
    if histogramPlotData is None and histogramData is not None:
        histogramPlotData = histogramData
    payloads = dict(montage_tile_payloads or {})
    if not payloads and placeholder is not None:
        image = np.asarray(placeholder)
        histogram = None if histogramData is None else np.asarray(histogramData)
        texture = None if semantic_data is None else np.asarray(semantic_data)
        if texture is None and image.ndim == 3 and image.shape[-1] in (3, 4) and not bool(rgb_already_windowed):
            texture = histogram
        payloads[0] = DisplayTilePayload(
            0,
            0,
            image,
            histogram,
            ("test-single-tile", id(image), tuple(image.shape), str(image.dtype)),
            texture_data=texture,
            texture_kind=texture_kind,
            semantic_data=texture,
            semantic_histogram_data=histogram,
            shader_mapping=shader_mapping,
        )
    histogram_source = histogramPlotData
    histogram_key = (
        id(histogram_source),
        tuple(np.shape(histogram_source)) if histogram_source is not None else None,
        None if histogram_source is None else str(np.asarray(histogram_source).dtype),
        (float(histogramRange[0]), float(histogramRange[1])),
    )
    histogram_revisions = getattr(view, "_test_histogram_revisions", None)
    if histogram_revisions is None:
        histogram_revisions = {}
        view._test_histogram_revisions = histogram_revisions
    if histogram_key not in histogram_revisions:
        histogram_revisions[histogram_key] = len(histogram_revisions) + 1
    histogram_revision = int(histogram_revisions[histogram_key])
    if tile_delta is None:
        if montage_dirty_tiles is None:
            upserts = payloads
        elif montage_dirty_tiles == ():
            upserts = {}
        else:
            upserts = {int(tile): payloads[int(tile)] for tile in montage_dirty_tiles if int(tile) in payloads}
        revision = 1 if upserts else 2
        tile_delta = TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=revision,
            histogram_revision=histogram_revision,
            viewport_revision=revision,
            upserts=upserts,
            active_tiles=tuple(payloads),
            planned_tiles=tuple(payloads),
        )
    return view.setTiledPresentation(
        geometry=geometry,
        tile_state=TilePresentationState(payloads),
        tile_delta=tile_delta,
        histogramPlotData=histogramPlotData,
        levels=levels,
        histogramRange=histogramRange,
        viewport_policy=ViewportPolicy.PRESERVE if viewport_policy is None else viewport_policy,
        rgb_already_windowed=rgb_already_windowed,
        shader_mapping=shader_mapping,
        tile_residency_budget_bytes=tile_residency_budget_bytes,
        frame_plan=frame_plan,
    )


def _single_vispy_tile_geometry(canvas):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState

    shape = tuple(int(value) for value in np.shape(canvas)[:2])
    return DisplayGeometry(
        view_state=ViewState.from_shape(shape).with_image_axes(0, 1),
        display_shape=shape,
        montage=MontageGeometry(indices=(0,), tile_shape=shape, columns=1, rows=1, gap=0),
        montage_tile_states=(MontageTileState.LOADED,),
    )


def test_factory_constructs_vispy_backend(qt_app):
    from arrayscope.app.settings_state import AppSettingsState, ImageRenderingBackendChoice
    from arrayscope.display.image_view_factory import create_image_view

    view = create_image_view(AppSettingsState(image_rendering_backend=ImageRenderingBackendChoice.VISPY))
    try:
        assert type(view).__name__ == "VisPySurface"
        assert view.surface.capabilities.name == "vispy"
    finally:
        view.close()


def test_vispy_surface_exposes_lifecycle_contract(qt_app):
    from arrayscope.display.backends import surface_for_view
    from arrayscope.display.backends.vispy.surface import VisPySurface
    from arrayscope.display.viewport import ViewportPolicy

    view = VisPySurface()
    try:
        surface = surface_for_view(view)

        assert surface.widget is view
        assert surface.capabilities.name == "vispy"
        assert surface.capabilities.native_pointer_interaction is False
        assert surface.interaction_event_owner() == "shared-controller"
        assert not view._paints_qgraphics_scene()
        surface.apply_camera((2, 3), ViewportPolicy.PRESERVE)
        diagnostics = surface.presentation_diagnostics()
        assert diagnostics["backend"] == "vispy"
        assert diagnostics["interaction_event_owner"] == "shared-controller"
        assert diagnostics["native_pointer_interaction"] is False

        view._vispy_pending_warm_tile_payloads = {1: object()}
        surface.reset_surface("test-context-loss")
        assert view._vispy_pending_warm_tile_payloads == {}
        assert surface.presentation_diagnostics()["last_reset_reason"] == "test-context-loss"
        surface.teardown_surface()
        surface.teardown_surface()
    finally:
        view.close()


def test_vispy_manual_resize_uses_shared_viewport_transaction(qt_app):
    from arrayscope.display.imageview2d import ArrayScopeGraphicsView
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        assert isinstance(view.graphicsView, ArrayScopeGraphicsView)
        view.resize(520, 420)
        view.show()
        qt_app.processEvents()
        view.setImage(np.zeros((100, 100), dtype=np.float32))
        qt_app.processEvents()
        view.getView().setRange(xRange=(-50.0, 150.0), yRange=(-40.0, 160.0), padding=0)
        qt_app.processEvents()
        view.viewport_controller.mode = ViewportMode.USER
        before_size = view.graphicsView.viewport().size()
        before_range = view.getView().viewRange()

        view.resize(340, 420)
        qt_app.processEvents()

        after_size = view.graphicsView.viewport().size()
        after_range = view.getView().viewRange()
        before_x_units = (before_range[0][1] - before_range[0][0]) / before_size.width()
        before_y_units = (before_range[1][1] - before_range[1][0]) / before_size.height()
        after_x_units = (after_range[0][1] - after_range[0][0]) / after_size.width()
        after_y_units = (after_range[1][1] - after_range[1][0]) / after_size.height()
        assert after_x_units == pytest.approx(before_x_units)
        assert after_y_units == pytest.approx(before_y_units)
        assert view._vispy_camera_key[0] == pytest.approx(tuple(after_range[0]))
        assert view._vispy_camera_key[1] == pytest.approx(tuple(after_range[1]))
    finally:
        view.close()


def test_vispy_background_pan_updates_camera_without_graphics_scene_drag(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        view.resize(520, 420)
        view.show()
        qt_app.processEvents()
        view.setImage(np.zeros((100, 100), dtype=np.float32))
        view.getView().setRange(xRange=(0.0, 100.0), yRange=(0.0, 100.0), padding=0)
        qt_app.processEvents()
        before_range = view.getView().viewRange()

        viewport = view.graphicsView.viewport()
        press = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(120.0, 120.0),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(160.0, 120.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        release = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonRelease,
            QtCore.QPointF(160.0, 120.0),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        def fail_hover_hit_test(_point):
            raise AssertionError("hover hit-test during active pan")

        assert view.eventFilter(viewport, press)
        view._interaction_target_at = fail_hover_hit_test
        assert view.eventFilter(viewport, move)
        after_range = view.getView().viewRange()
        assert after_range[0] != before_range[0]
        assert view._vispy_camera_key[0] == pytest.approx(tuple(after_range[0]))
        assert view.eventFilter(viewport, release)
    finally:
        view.close()


def test_vispy_background_pan_matches_flipped_viewbox_axes(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        view.resize(520, 420)
        view.show()
        qt_app.processEvents()
        view.setImage(np.zeros((100, 100), dtype=np.float32))
        view.getView().invertX(True)
        view.getView().invertY(False)
        view.getView().setRange(xRange=(0.0, 100.0), yRange=(0.0, 100.0), padding=0)
        qt_app.processEvents()
        before_range = view.getView().viewRange()
        viewport = view.graphicsView.viewport()

        press = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QPointF(120.0, 120.0),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        move = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(160.0, 140.0),
            QtCore.Qt.MouseButton.NoButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )

        assert view.eventFilter(viewport, press)
        assert view.eventFilter(viewport, move)
        after_range = view.getView().viewRange()
        assert after_range[0][0] > before_range[0][0]
        assert after_range[1][0] > before_range[1][0]
        assert view._vispy_camera_key[0] == pytest.approx(tuple(after_range[0]))
        assert view._vispy_camera_key[1] == pytest.approx(tuple(after_range[1]))
    finally:
        view.close()


def test_vispy_wheel_zoom_updates_camera_without_graphics_scene_wheel(qt_app):
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    class WheelEvent:
        def __init__(self):
            self.accepted = False

        def type(self):
            return QtCore.QEvent.Type.Wheel

        def position(self):
            return QtCore.QPointF(180.0, 140.0)

        def angleDelta(self):
            return QtCore.QPoint(0, 120)

        def accept(self):
            self.accepted = True

    view = VisPyImageView2D()
    try:
        view.resize(520, 420)
        view.show()
        qt_app.processEvents()
        view.setImage(np.zeros((100, 100), dtype=np.float32))
        view.getView().setRange(xRange=(0.0, 100.0), yRange=(0.0, 100.0), padding=0)
        qt_app.processEvents()
        before_range = view.getView().viewRange()
        event = WheelEvent()

        assert view.eventFilter(view.graphicsView.viewport(), event)
        after_range = view.getView().viewRange()
        assert event.accepted
        assert after_range[0][1] - after_range[0][0] < before_range[0][1] - before_range[0][0]
        assert view._vispy_camera_key[0] == pytest.approx(tuple(after_range[0]))
    finally:
        view.close()


def test_scalar_presentation_uses_tiled_visual(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 64 * 48, dtype=np.float32).reshape(48, 64)
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.1, 0.9), histogramRange=(0.0, 1.0))

        timing = view.lastImageUploadTiming()
        assert timing.mode == "vispy_tile_layer"
        assert timing.rgb_window_ms == 0.0
        assert view._vispy_gpu_montage_layer.last_stats.visible_items == 1
        assert not view._vispy_image.visible
    finally:
        view.close()


def test_scalar_level_preview_updates_clim_without_rgb_work(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        view._apply_histogram_preview_levels((0.25, 0.75))

        timing = view.lastImageUploadTiming()
        assert timing.mode == "vispy_level_preview"
        assert timing.rgb_window_ms == 0.0
        assert timing.visible_bytes == 0
        assert view._vispy_gpu_montage_layer._levels == (0.25, 0.75)
    finally:
        view.close()


def test_vispy_histogram_drag_flushes_preview_without_debounce(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        view.histogram.setLevels(0.2, 0.8)
        view._on_histogram_levels_changed()

        timing = view.lastImageUploadTiming()
        assert timing.mode == "vispy_level_preview"
        assert not view._histogram_preview_controller.timer.isActive()
        assert view._histogram_preview_controller.pending_levels is None
        assert view._vispy_gpu_montage_layer._levels == (0.2, 0.8)
    finally:
        view.close()


def test_vispy_tile_redraw_coalesces_canvas_updates_but_keeps_draw_wait(qt_app, monkeypatch):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    update_calls = []
    try:
        monkeypatch.setattr(view._vispy_canvas, "update", lambda *args, **kwargs: update_calls.append((args, kwargs)))

        view._request_vispy_tile_layer_redraw()
        view._request_vispy_tile_layer_redraw()

        diagnostics = view.vispyPresentationDiagnostics()
        assert diagnostics["tile_presentation_request_count"] == 2
        assert diagnostics["tile_presentation_draw_pending"] is True
        assert diagnostics["canvas_update_request_count"] == 1
        for _ in range(5):
            qt_app.processEvents()
        assert update_calls == [((), {})]

        view._on_vispy_draw()
        diagnostics = view.vispyPresentationDiagnostics()
        assert diagnostics["tile_presentation_draw_count"] == 2
        assert diagnostics["tile_presentation_draw_pending"] is False
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
        _present_vispy_tiled(view,
            rgb,
            histogramData=magnitude,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
        )
        view._apply_histogram_preview_levels((0.5, 1.0))

        assert view._vispy_gpu_montage_layer._levels == (0.5, 1.0)
        assert view._vispy_gpu_montage_layer.last_stats.shader_uniform_updates >= 1
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
        _present_vispy_tiled(view,
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


def test_vispy_gpu_mapped_complex_display_image_renders_color(qt_app):
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
        _present_vispy_tiled(view,
            display.data,
            histogramData=display.histogram_data,
            levels=levels,
            histogramRange=levels,
            rgb_already_windowed=display.rgb_already_windowed,
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
            montage_dirty_tiles=(),
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


def test_vispy_gpu_mapped_complex_level_change_updates_uniform_without_texture_upload(qt_app):
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
        _present_vispy_tiled(view,
            display.data,
            histogramData=display.histogram_data,
            levels=(0.0, 16.0),
            histogramRange=(0.0, 16.0),
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
        )
        layer = view._vispy_gpu_montage_layer
        visual = layer._visuals_by_page[0]
        assert layer.last_stats.texture_uploads >= 1

        _present_vispy_tiled(view,
            display.data,
            histogramData=display.histogram_data,
            levels=(2.0, 8.0),
            histogramRange=(0.0, 16.0),
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
            montage_dirty_tiles=(),
        )

        assert layer.last_stats.texture_uploads == 0
        assert tuple(float(value) for value in visual._levels) == (2.0, 8.0)
    finally:
        view.close()


def test_gpu_mapped_visual_shader_supports_raw_complex_components():
    from arrayscope.display.backends.vispy.gpu_mapped_visual import GpuMappedImageVisual

    shader = GpuMappedImageVisual._fragment_shader

    assert "uniform float u_component_mode" in shader
    assert "float complex_component" in shader
    assert "if (u_component_mode > 2.5)" in shader
    assert "scalar = length(z);" in shader
    assert "gl_FragColor = vec4(color, 1.0);" in shader


def test_gpu_mapped_visual_cached_complex_component_change_updates_uniform_without_upload():
    from arrayscope.display.shader_mapping import ShaderComponent, ShaderMapping, TexturePlaneKind
    from arrayscope.display.backends.vispy.gpu_mapped_visual import GpuMappedImageVisual

    visual = object.__new__(GpuMappedImageVisual)
    visual._scalar_texture = object()
    visual.scalar_source_id = ("source", "complex_rg32f")
    visual._mode = 2.0
    visual._scale_mode = 0.0
    visual._symlog_constant = 0.0
    visual._component_mode = 0.0
    visual.upload_count = 3
    visual._lut_key = None
    visual._shader_mapping_key = None
    visual.shader_uniform_update_count = 0
    visual.update = lambda: None
    visual._set_lut_texture = lambda lut, key=None: False
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


def test_gpu_mapped_visual_lut_change_is_uniform_only_for_cached_scalar_source():
    from arrayscope.display.shader_mapping import ShaderMapping, TexturePlaneKind
    from arrayscope.display.backends.vispy.gpu_mapped_visual import GpuMappedImageVisual

    visual = object.__new__(GpuMappedImageVisual)
    visual._scalar_texture = object()
    visual.scalar_source_id = ("source", "scalar_r32f")
    visual._mode = 1.0
    visual._scale_mode = 0.0
    visual._symlog_constant = 0.0
    visual._component_mode = 0.0
    visual._lut_key = None
    visual._shader_mapping_key = None
    visual.upload_count = 4
    visual.shader_uniform_update_count = 0
    visual.update = lambda: None
    uploaded_luts = []
    visual._set_lut_texture = lambda lut, key=None: uploaded_luts.append(np.array(lut, copy=True)) or True
    visual.set_levels = lambda levels, count=True: setattr(visual, "_levels", tuple(float(value) for value in levels))

    mapping = ShaderMapping(lut_data=np.array([[0, 0, 255], [255, 0, 0]], dtype=np.uint8))
    visual.set_mapped_data(
        np.ones((2, 2), dtype=np.float32),
        texture_kind=TexturePlaneKind.SCALAR_R32F,
        levels=(0.0, 1.0),
        source_id=("source", "scalar_r32f"),
        shader_mapping=mapping,
    )

    assert visual.upload_count == 4
    assert visual.shader_uniform_update_count == 1
    assert len(uploaded_luts) == 1
    np.testing.assert_array_equal(uploaded_luts[0], mapping.lut_data)


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
        _present_vispy_tiled(view,
            rgb,
            histogramData=scalar,
            levels=levels,
            histogramRange=(float(np.nanmin(scalar)), float(np.nanmax(scalar))),
            rgb_already_windowed=False,
        )
        for _ in range(20):
            qt_app.processEvents()

        visual = view._vispy_gpu_montage_layer._visuals_by_page[0]
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


def test_vispy_rejects_direct_tile_layer_presentation(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        assert not hasattr(view, "setMontageTileLayerPresentation")
    finally:
        view.close()


def test_vispy_tiled_mode_layers_tiles_above_base_visual(qt_app):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 0.25, dtype=np.float32), None, ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 0.75, dtype=np.float32), None, ("tile", 1)),
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    try:
        view.setTiledPresentation(
            geometry=_montage_geometry(),
            tile_state=TilePresentationState(payloads),
            tile_delta=delta,
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        )

        base_order = int(getattr(view._vispy_image, "order", 0))
        tile_orders = tuple(int(getattr(visual, "order", 0)) for visual in view._vispy_gpu_montage_layer._visuals_by_page)
        assert view._vispy_image.visible is False
        assert view._vispy_windowed_image.visible is False
        assert tile_orders
        assert min(tile_orders) > base_order
    finally:
        view.close()


def test_vispy_tiled_overlay_clear_waits_for_presenting_draw(qt_app, monkeypatch):
    from arrayscope.display.overlays import MontageTileOverlay
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    update_calls = []
    try:
        monkeypatch.setattr(view._vispy_canvas, "update", lambda *args, **kwargs: update_calls.append((args, kwargs)))
        view.setMontageTileOverlays((MontageTileOverlay(0, 0, 2, 2, "loading", "loading"),))
        assert view.montageTileOverlayCount() == 1
        update_calls.clear()

        view._vispy_tile_presentation_request_count = 1
        view._vispy_tile_presentation_draw_count = 0
        view.clearMontageTileOverlays()

        assert view.montageTileOverlayCount() == 1
        assert view._vispy_pending_overlay_clear_request_count == 1
        overlay_orders = [
            int(getattr(visual, "order", 0))
            for visual in view._vispy_overlay_visuals
            if bool(getattr(visual, "visible", False))
        ]
        assert overlay_orders
        assert max(overlay_orders) < 10

        view._on_vispy_draw()

        assert view.montageTileOverlayCount() == 0
        assert view._vispy_pending_overlay_clear_request_count is None
        assert update_calls == []
        assert all(not bool(getattr(visual, "visible", False)) for visual in view._vispy_overlay_visuals)
    finally:
        view.close()


def test_vispy_tile_layer_bounds_cover_full_montage_not_viewport_canvas(qt_app):
    from arrayscope.display.viewport import ViewportPolicy
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    payloads = {
        index: DisplayTilePayload(index, index, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", index))
        for index in range(4)
    }
    try:
        view.resize(420, 220)
        view.show()
        _present_vispy_tiled(view,
            np.zeros((2, 5, 3), dtype=np.uint8),
            histogramData=None,
            histogramPlotData=None,
            geometry=_shifted_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            viewport_policy=ViewportPolicy.FIT_ONCE,
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={index: payload.source_id for index, payload in payloads.items()},
            montage_tile_payloads=payloads,
        )

        rect = view._vispy_bounds_item.rect()
        assert (rect.left(), rect.top()) == (0.0, 0.0)
        assert (rect.width(), rect.height()) == (11.0, 2.0)
        assert view._vispy_display_shape == (2, 11)
        assert view.viewport_controller.last_display_shape == (2, 11)
        assert view._current_image_world_rect() == (0.0, 0.0, 10.0, 1.0)
    finally:
        view.close()


def test_vispy_typed_tiled_single_plane_uses_frame_plan_geometry(qt_app):
    from arrayscope.core.scheduler import FrameTarget
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.backend_contract import VISPY_CAPABILITIES
    from arrayscope.display.frame_planner import FramePlanner
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.viewport import ViewportPolicy
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    state = ViewState.from_shape((4, 4)).with_image_axes(0, 1)
    frame_plan = FramePlanner(internal_tile_shape=(2, 2)).plan(
        target=FrameTarget("semantic", "viewport", "presentation", "exact-visible"),
        view_state=state,
        display_shape=(4, 4),
        backend_capabilities=VISPY_CAPABILITIES,
    )
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    payloads = {
        region.region_id: DisplayTilePayload(
            region.region_id,
            region.region_id,
            image[region.data_slices],
            image[region.data_slices],
            ("single", region.region_id),
        )
        for region in frame_plan.regions
    }
    tile_state = TilePresentationState(payloads, revision=1)
    tile_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=0,
        target_revision=1,
        upserts=payloads,
        active_tiles=frame_plan.active_region_ids,
        planned_tiles=frame_plan.planned_region_ids,
        near_tiles=frame_plan.near_region_ids,
    )
    view = VisPyImageView2D()
    try:
        report = view.setTiledPresentation(
            geometry=frame_plan.geometry,
            tile_state=tile_state,
            tile_delta=tile_delta,
            histogramPlotData=None,
            levels=(0.0, 15.0),
            histogramRange=(0.0, 15.0),
            viewport_policy=ViewportPolicy.PRESERVE,
            tile_residency_budget_bytes=1024 * 1024,
            frame_plan=frame_plan,
        )

        stats = view._vispy_gpu_montage_layer.last_stats
        visual = view._vispy_gpu_montage_layer._visuals_by_page[0]
        assert view.montageDisplayMode() == "vispy_tile_layer"
        assert sorted(report.presented_tiles) == [0, 1, 2, 3]
        assert sorted(report.accepted_upserts(tile_delta)) == [0, 1, 2, 3]
        assert stats.visible_items == 4
        assert stats.resident_items == 4
        assert visual.vertex_data.shape == (24, 2)
        assert (float(visual.vertex_data[:, 0].max()), float(visual.vertex_data[:, 1].max())) == (4.0, 4.0)
    finally:
        view.close()


def test_vispy_tile_layer_level_preview_updates_uniforms_without_upload(qt_app, monkeypatch):
    import arrayscope.display.vispy_imageview2d as vispy_view
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    def fail_cpu_window(*args, **kwargs):
        raise AssertionError("VisPy tile shader level preview should not CPU-window RGB tiles")

    monkeypatch.setattr(vispy_view, "rgb_display_for_levels", fail_cpu_window)
    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 1)),
    }
    try:
        _present_vispy_tiled(view,
            np.zeros((2, 5, 3), dtype=np.uint8),
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            rgb_already_windowed=False,
            montage_dirty_tiles=None,
            montage_tile_source_ids={tile: payload.source_id for tile, payload in payloads.items()},
            montage_tile_payloads=payloads,
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
        assert view._vispy_gpu_montage_layer._levels == (4.0, 9.0)
    finally:
        view.close()


def test_vispy_tile_layer_clean_flush_skips_existing_visual_uploads(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2, 3), 180, dtype=np.uint8), np.ones((2, 2), dtype=np.float32), ("tile", 1)),
    }
    sources = {tile: payload.source_id for tile, payload in payloads.items()}
    try:
        kwargs = dict(
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
            rgb_already_windowed=False,
            montage_tile_source_ids=sources,
            montage_tile_payloads=payloads,
        )
        _present_vispy_tiled(view,np.zeros((2, 5, 3), dtype=np.uint8), montage_dirty_tiles=None, **kwargs)
        _present_vispy_tiled(view,np.zeros((2, 5, 3), dtype=np.uint8), montage_dirty_tiles=(), **kwargs)

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
        _present_vispy_tiled(view,
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
        assert not hasattr(view, "_vispy_tile_visuals")
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


def test_vispy_direct_tiled_upsert_reuses_uploaded_resident_payload(qt_app):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    image = np.full((2, 2), 3.0, dtype=np.float32)
    payloads = {0: DisplayTilePayload(0, 0, image, image, ("tile", 0))}
    proposed = TilePresentationState(payloads)
    initial = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
    )
    retry = TilePresentationDelta(
        structure_revision=1,
        payload_revision=2,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        base_revision=1,
        target_revision=2,
        upserts=payloads,
        active_tiles=(0,),
        planned_tiles=(0,),
        near_tiles=(0,),
    )
    try:
        kwargs = dict(
            geometry=_single_tile_montage_geometry(),
            histogramPlotData=None,
            levels=(0.0, 4.0),
            histogramRange=(0.0, 4.0),
            rgb_already_windowed=False,
        )
        view.setTiledPresentation(
            tile_state=proposed,
            tile_delta=initial,
            **kwargs,
        )
        assert view._vispy_gpu_montage_layer.last_stats.items_updated == 1

        view.setTiledPresentation(
            tile_state=proposed,
            tile_delta=retry,
            **kwargs,
        )

        stats = view._vispy_gpu_montage_layer.last_stats
        assert stats.items_updated == 0
        assert stats.items_skipped == 1
        assert stats.texture_uploads == 0
    finally:
        view.close()


def test_vispy_direct_tiled_hides_previous_windowed_main_visual(qt_app):
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.model.frame import DisplayTilePayload
    from arrayscope.display.slice_engine import make_shader_image_from_slab
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    class Request:
        def __init__(self, view_state):
            self.view_state = view_state
            self.ranged_axes = ()

    data = (np.arange(4, dtype=np.float32).reshape(2, 2) + 1j).astype(np.complex64)
    state = ViewState.from_shape(data.shape).with_channel(ChannelMode.COMPLEX)
    display = make_shader_image_from_slab(data, Request(state))
    payload = DisplayTilePayload(
        0,
        0,
        display.data,
        display.histogram_data,
        ("tile", 0),
        texture_data=display.semantic_data,
        texture_kind=display.texture_kind,
        semantic_data=display.semantic_data,
        semantic_histogram_data=display.histogram_data,
        shader_mapping=display.shader_mapping,
    )
    view = VisPyImageView2D()
    try:
        _present_vispy_tiled(view,
            display.data,
            histogramData=display.histogram_data,
            levels=(0.0, 4.0),
            histogramRange=(0.0, 4.0),
            shader_mapping=display.shader_mapping,
            texture_kind=display.texture_kind,
            semantic_data=display.semantic_data,
        )
        assert view._vispy_gpu_montage_layer.last_stats.visible_items == 1

        _present_vispy_tiled(view,
            np.zeros((2, 2), dtype=np.float32),
            histogramData=None,
            histogramPlotData=None,
            geometry=_single_tile_montage_geometry(),
            levels=(0.0, 4.0),
            histogramRange=(0.0, 4.0),
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0)},
            montage_tile_payloads={0: payload},
        )

        assert not view._vispy_image.visible
        assert not view._vispy_windowed_image.visible
        assert view._vispy_gpu_montage_layer.last_stats.visible_items == 1
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
        _present_vispy_tiled(view,
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


def test_vispy_direct_tiled_scalar_presented_tiles_render_nonblack_without_level_interaction(qt_app):
    from pyqtgraph.Qt import QtGui
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    left = np.linspace(0.2, 0.8, 64, dtype=np.float32).reshape(8, 8)
    right = np.linspace(0.8, 0.2, 64, dtype=np.float32).reshape(8, 8)
    payloads = {
        0: DisplayTilePayload(0, 0, left, left, ("scalar", 0)),
        1: DisplayTilePayload(1, 1, right, right, ("scalar", 1)),
    }
    view = VisPyImageView2D()
    try:
        view.resize(360, 240)
        view.show()
        report = view.setTiledPresentation(
            geometry=_montage_geometry(),
            tile_state=TilePresentationState(payloads),
            tile_delta=TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0, 1),
                planned_tiles=(0, 1),
            ),
            histogramPlotData=np.concatenate([left.ravel(), right.ravel()]),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )
        assert report.presented_tiles == frozenset({0, 1})

        for _ in range(20):
            qt_app.processEvents()

        pixmap = view.grab()
        image = pixmap.toImage().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), dtype=np.uint8).reshape(image.height(), image.width(), 4)[..., :3]
        center = pixels[pixels.shape[0] // 5 : pixels.shape[0] * 4 // 5, pixels.shape[1] // 5 : pixels.shape[1] * 4 // 5]
        assert int(center.max()) > 16
        assert float(center.mean()) > 2.0
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
        _present_vispy_tiled(view,
            placeholder,
            montage_dirty_tiles=None,
            montage_tile_source_ids={0: ("tile", 0), 1: ("tile", 1)},
            montage_tile_payloads=payloads,
            **kwargs,
        )
        _present_vispy_tiled(view,
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
        _present_vispy_tiled(view,
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


def test_vispy_direct_tiled_respects_delta_active_tiles(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 1.0, dtype=np.float32), None, ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 2.0, dtype=np.float32), None, ("tile", 1)),
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(1,),
        planned_tiles=(0, 1),
        near_tiles=(0, 1),
    )
    try:
        _present_vispy_tiled(view,
            np.zeros((2, 5), dtype=np.float32),
            histogramData=None,
            histogramPlotData=None,
            geometry=_montage_geometry(),
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
            montage_dirty_tiles=None,
            montage_tile_source_ids={tile: payload.source_id for tile, payload in payloads.items()},
            montage_tile_payloads=payloads,
            tile_delta=delta,
        )

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_visible_items == 1
        assert timing.tile_layer_items_updated == 1
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
        _present_vispy_tiled(view,
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
        _present_vispy_tiled(view,
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
        _present_vispy_tiled(view,
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
        expected = ((0.0, 5.0), (0.0, 2.0))

        np.testing.assert_allclose(x_range, expected[0], atol=1e-6)
        np.testing.assert_allclose(y_range, expected[1], atol=1e-6)
        assert view._vispy_view.camera.aspect is None
        assert view._vispy_camera_key[:2] == expected

        view.oneToOne()
        assert view._vispy_view.camera.aspect == 1.0
    finally:
        view.close()


def test_vispy_constraints_use_full_montage_world_for_shifted_tiles(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=1),
        montage_origin_x=0,
        montage_origin_y=3,
    )
    view = VisPyImageView2D()
    try:
        view.resize(360, 240)
        view.show()
        _present_vispy_tiled(view,
            np.zeros((2, 5), dtype=np.float32),
            histogramData=np.zeros((2, 5), dtype=np.float32),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            geometry=geometry,
        )

        view.getView().setRange(xRange=(0.0, 5.0), yRange=(-20.0, -15.0), padding=0)
        qt_app.processEvents()
        _x_range, y_range = view.getView().viewRange()

        assert y_range == pytest.approx([-4.75, 0.25])
        assert view._vispy_camera_key[1] == pytest.approx((-4.75, 0.25))
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

    def delta(revision: int, *, upserts):
        return TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=revision,
            upserts=upserts,
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
        view.setTiledPresentation(tile_state=TilePresentationState(payloads), tile_delta=delta(1, upserts=payloads), **kwargs)
        first = view.lastImageUploadTiming()
        assert first.tile_layer_items_updated == 2
        assert first.visible_bytes > 0

        view.setTiledPresentation(tile_state=TilePresentationState(payloads), tile_delta=delta(2, upserts={}), **kwargs)
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

    def delta(revision: int, payloads, active_tiles, *, upserts=None):
        return TilePresentationDelta(
            structure_revision=revision,
            payload_revision=revision,
            visibility_revision=revision,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=revision,
            upserts=payloads if upserts is None else upserts,
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
        view.setTiledPresentation(
            tile_state=TilePresentationState(initial_payloads),
            tile_delta=delta(1, initial_payloads, (0, 1, 2, 3)),
            **kwargs,
        )
        first = view.lastImageUploadTiming()
        assert first.tile_layer_items_updated == 4
        assert first.tile_layer_resident_items == 4

        view.setTiledPresentation(
            tile_state=TilePresentationState(shifted_payloads),
            tile_delta=delta(2, shifted_payloads, (0, 1), upserts={}),
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


def test_vispy_tiled_reuses_resident_texture_after_layer_clear(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 3.0, dtype=np.float32), None, ("stable-source", 0)),
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
            active_tiles=(0,),
            planned_tiles=(0,),
        )

    kwargs = dict(
        geometry=_single_tile_montage_geometry(),
        tile_state=TilePresentationState(payloads),
        histogramPlotData=None,
        levels=(0.0, 4.0),
        histogramRange=(0.0, 4.0),
        rgb_already_windowed=False,
        tile_residency_budget_bytes=64 * 1024 * 1024,
    )
    try:
        view.setTiledPresentation(tile_delta=delta(1), **kwargs)
        first = view.lastImageUploadTiming()
        assert first.tile_layer_texture_uploads >= 1

        view.clearMontageTileLayer()
        view.setTiledPresentation(tile_delta=delta(2), **kwargs)
        second = view.lastImageUploadTiming()

        assert second.tile_layer_items_skipped == 1
        assert second.tile_layer_texture_uploads == 0
        assert second.tile_layer_texture_upload_bytes == 0
        assert second.visible_bytes == 0
    finally:
        view.close()


def test_vispy_first_class_tiled_loading_only_commit_hides_previous_active_tiles(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    loaded = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    loading = DisplayGeometry(
        view_state=loaded.view_state,
        display_shape=(2, 5),
        montage=loaded.montage,
        montage_tile_states=(MontageTileState.LOADING, MontageTileState.LOADING),
    )
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 1.0, dtype=np.float32), None, ("source", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 2.0, dtype=np.float32), None, ("source", 1)),
    }
    first_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
        near_tiles=(0, 1),
    )
    empty_delta = TilePresentationDelta(
        structure_revision=2,
        payload_revision=2,
        visibility_revision=2,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=2,
        active_tiles=(),
        planned_tiles=(0, 1),
        near_tiles=(0, 1),
    )
    view = VisPyImageView2D()
    try:
        view.setTiledPresentation(
            geometry=loaded,
            tile_state=TilePresentationState(payloads),
            tile_delta=first_delta,
            histogramPlotData=None,
            levels=(100.0, 200.0),
            histogramRange=(0.0, 1000.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )
        assert view.lastImageUploadTiming().tile_layer_active_pages > 0
        assert tuple(float(value) for value in view.getLevels()) == (100.0, 200.0)
        assert tuple(float(value) for value in view.getHistogramDataBounds()) == (0.0, 1000.0)

        view.setTiledPresentation(
            geometry=loading,
            tile_state=TilePresentationState({}),
            tile_delta=empty_delta,
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_visible_items == 0
        assert timing.tile_layer_active_pages == 0
        assert timing.tile_layer_resident_items == 2
        assert not any(visual.visible for visual in view._vispy_gpu_montage_layer._visuals_by_page)
        assert tuple(float(value) for value in view.getLevels()) == (100.0, 200.0)
        assert tuple(float(value) for value in view.getHistogramDataBounds()) == (0.0, 1000.0)
    finally:
        view.close()


def test_vispy_direct_tiled_loading_payloads_are_submitted_for_backend_ack(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    loaded_geometry = _montage_geometry()
    loading_geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 128)).with_montage_axis(2, columns=2, indices=(100, 101), text="100:102"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(100, 101), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADING, MontageTileState.LOADING),
    )
    first_payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 1.0, dtype=np.float32), None, ("source", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 2.0, dtype=np.float32), None, ("source", 1)),
    }
    next_payloads = {
        0: DisplayTilePayload(0, 100, np.full((2, 2), 100.0, dtype=np.float32), None, ("source", 100)),
        1: DisplayTilePayload(1, 101, np.full((2, 2), 101.0, dtype=np.float32), None, ("source", 101)),
    }
    first_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=first_payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
        near_tiles=(0, 1),
        near_tile_source_ids={index: payload.source_id for index, payload in first_payloads.items()},
    )
    next_delta = TilePresentationDelta(
        structure_revision=2,
        payload_revision=2,
        visibility_revision=2,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=2,
        upserts=next_payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
        near_tiles=(0, 1),
        near_tile_source_ids={index: payload.source_id for index, payload in next_payloads.items()},
    )
    view = VisPyImageView2D()
    try:
        view.setTiledPresentation(
            geometry=loaded_geometry,
            tile_state=TilePresentationState(first_payloads),
            tile_delta=first_delta,
            histogramPlotData=None,
            levels=(0.0, 2.0),
            histogramRange=(0.0, 2.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        view.setTiledPresentation(
            geometry=loading_geometry,
            tile_state=TilePresentationState(next_payloads),
            tile_delta=next_delta,
            histogramPlotData=None,
            levels=(0.0, 101.0),
            histogramRange=(0.0, 101.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        timing = view.lastImageUploadTiming()
        assert timing.tile_layer_visible_items == 2
        assert timing.tile_layer_items_updated == 2
        assert timing.tile_layer_texture_uploads == 2
        assert view._vispy_gpu_montage_layer.last_stats.presented_tiles == (0, 1)
    finally:
        view.close()


def test_vispy_scalar_tiled_geometry_retry_preserves_previous_frame(qt_app):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    geometry = _montage_geometry()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 0.25, dtype=np.float32), None, ("source", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 0.75, dtype=np.float32), None, ("source", 1)),
    }
    first_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    retry_delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts={},
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    view = VisPyImageView2D()
    try:
        canvas_updates = []
        original_update = view._vispy_canvas.update
        view._vispy_canvas.update = lambda *args, **kwargs: canvas_updates.append(True) or original_update(*args, **kwargs)
        previous = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        _present_vispy_tiled(view,previous, histogramData=previous, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        assert view._vispy_gpu_montage_layer.last_stats.visible_items == 1
        canvas_updates.clear()

        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=first_delta,
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        assert report.presented_tiles == frozenset({0, 1})
        assert not view._vispy_image.visible
        assert canvas_updates

        retry_report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=retry_delta,
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        assert retry_report.presented_tiles == frozenset({0, 1})
        assert not view._vispy_image.visible
    finally:
        view.close()


def test_vispy_tile_level_preview_updates_all_pages_without_deferred_retry(qt_app):
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    geometry = _montage_geometry()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 0.25, dtype=np.float32), None, ("source", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 0.75, dtype=np.float32), None, ("source", 1)),
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    view = VisPyImageView2D()
    try:
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=delta,
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )
        layer = view._vispy_gpu_montage_layer
        layer._ensure_visual_count(3)
        layer._pool.tile_slots = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
        layer._visible_items = 3
        canvas_updates = []
        original_update = view._vispy_canvas.update
        view._vispy_canvas.update = lambda *args, **kwargs: canvas_updates.append(True) or original_update(*args, **kwargs)

        view._apply_histogram_preview_levels((0.2, 0.8))
        stats = layer.last_stats

        assert stats.presented_tiles == (0, 1, 2)
        assert tuple(float(value) for value in layer._levels) == (0.2, 0.8)
        assert all(tuple(getattr(visual, "_levels", ())) == (0.2, 0.8) for visual in layer._visuals_by_page[:3])
        assert canvas_updates
    finally:
        view.close()


def test_vispy_first_typed_tiled_commit_applies_payload_pixels_and_levels_before_autolevel(qt_app):
    from arrayscope.core.view_state import ChannelMode, ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.montage import MontageTileState
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState
    from arrayscope.display.slice_engine import make_image
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    source = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64)
    state = ViewState.from_shape(source.shape).with_channel(ChannelMode.COMPLEX)
    left = make_image(source, state)
    right = make_image(source * (1.0 + 0.5j), state)
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_channel(ChannelMode.COMPLEX).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    payloads = {
        0: DisplayTilePayload(
            0,
            0,
            left.data,
            left.histogram_data,
            ("payload", 0),
            texture_data=left.semantic_data,
            texture_kind=left.texture_kind,
            semantic_data=left.semantic_data,
            semantic_histogram_data=left.histogram_data,
            shader_mapping=left.shader_mapping,
        ),
        1: DisplayTilePayload(
            1,
            1,
            right.data,
            right.histogram_data,
            ("payload", 1),
            texture_data=right.semantic_data,
            texture_kind=right.texture_kind,
            semantic_data=right.semantic_data,
            semantic_histogram_data=right.histogram_data,
            shader_mapping=right.shader_mapping,
        ),
    }
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=payloads,
        active_tiles=(0, 1),
        planned_tiles=(0, 1),
    )
    view = VisPyImageView2D()
    try:
        report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=delta,
            histogramPlotData=np.concatenate([left.histogram_data.ravel(), right.histogram_data.ravel()]),
            levels=(0.0, float(np.nanmax(right.histogram_data))),
            histogramRange=(0.0, float(np.nanmax(right.histogram_data))),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        layer = view._vispy_gpu_montage_layer
        assert report.presented_tiles == frozenset({0, 1})
        assert layer.last_stats.presented_tiles == (0, 1)
        clean_delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts={},
            active_tiles=(0, 1),
            planned_tiles=(0, 1),
        )
        retry_report = view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=clean_delta,
            histogramPlotData=np.concatenate([left.histogram_data.ravel(), right.histogram_data.ravel()]),
            levels=(0.0, float(np.nanmax(right.histogram_data))),
            histogramRange=(0.0, float(np.nanmax(right.histogram_data))),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )

        assert retry_report.presented_tiles == frozenset({0, 1})
        assert layer.last_stats.presented_tiles == (0, 1)
        assert tuple(float(value) for value in layer._levels) == (0.0, float(np.nanmax(right.histogram_data)))
        assert layer._pool.source_ids
        for payload in payloads.values():
            assert layer._pool.source_ids.get(("source", payload.source_id)) == payload.source_id
        assert layer._shader_mapping is not None
        assert any(len(getattr(visual, "vertex_data", ())) > 0 for visual in layer._visuals_by_page)
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
        view.setTiledPresentation(
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

        assert getattr(view, "_last_vispy_warm_tile_stats", None) is None

        clean_delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=2,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            active_tiles=(0,),
            planned_tiles=(0, 1, 2),
            near_tiles=(0, 1, 2),
            near_tile_source_ids={index: payload.source_id for index, payload in payloads.items()},
        )
        view.setTiledPresentation(
            geometry=geometry,
            tile_state=TilePresentationState(payloads),
            tile_delta=clean_delta,
            histogramPlotData=None,
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=64 * 1024 * 1024,
        )
        clean_visible = view.lastImageUploadTiming()

        for _ in range(10):
            qt_app.processEvents()

        warm = view._last_vispy_warm_tile_stats
        assert warm is not None
        assert warm.items_updated == 2
        assert warm.resident_items == 3
        assert view.lastImageUploadTiming().visible_bytes == clean_visible.visible_bytes
    finally:
        view.close()


def test_vispy_warm_residency_schedule_is_constant_time_when_busy(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    class NoIterationDict(dict):
        def items(self):
            raise AssertionError("busy warm scheduling should not inspect payloads")

        def values(self):
            raise AssertionError("busy warm scheduling should not inspect payloads")

        def __iter__(self):
            raise AssertionError("busy warm scheduling should not inspect payloads")

    existing = {0: object()}
    view = VisPyImageView2D()
    try:
        view._vispy_pending_warm_tile_payloads = existing
        view._schedule_vispy_warm_tile_residency(
            NoIterationDict({1: object()}),
            geometry=_montage_geometry(),
            rgb_already_windowed=False,
            tile_delta=None,
            tile_residency_budget_bytes=0,
        )

        assert view._vispy_pending_warm_tile_payloads is existing
    finally:
        view.close()


def test_vispy_warm_residency_schedule_keeps_caller_payload_mapping(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    payloads = {1: object(), 2: object()}
    view = VisPyImageView2D()
    try:
        view._schedule_vispy_warm_tile_residency(
            payloads,
            geometry=_montage_geometry(),
            rgb_already_windowed=False,
            tile_delta=None,
            tile_residency_budget_bytes=0,
        )

        assert view._vispy_pending_warm_tile_payloads is payloads
    finally:
        view.close()


def test_vispy_direct_tiled_histogram_only_commit_refreshes_histogram(qt_app, monkeypatch):
    from types import SimpleNamespace

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    payloads = {
        0: DisplayTilePayload(0, 0, np.full((2, 2), 0.25, dtype=np.float32), None, ("tile", 0)),
        1: DisplayTilePayload(1, 1, np.full((2, 2), 0.75, dtype=np.float32), None, ("tile", 1)),
    }
    sources = {tile: payload.source_id for tile, payload in payloads.items()}
    placeholder = np.broadcast_to(np.zeros((1, 1), dtype=np.float32), (2, 5))
    calls = []
    try:
        _present_vispy_tiled(view,
            placeholder,
            histogramData=None,
            histogramPlotData=np.arange(4, dtype=np.float32),
            geometry=_montage_geometry(),
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            montage_dirty_tiles=None,
            montage_tile_source_ids=sources,
            montage_tile_payloads=payloads,
        )
        monkeypatch.setattr(view, "_update_histogram_for_vispy", lambda *args, **kwargs: calls.append(args))
        view.render_coordinator = SimpleNamespace(interactive_active=True, has_pending_render=False)

        _present_vispy_tiled(view,
            placeholder,
            histogramData=None,
            histogramPlotData=np.arange(4, dtype=np.float32) + 10.0,
            geometry=_montage_geometry(),
            levels=(0.0, 1.0),
            histogramRange=(10.0, 13.0),
            montage_dirty_tiles=(),
            montage_tile_source_ids=sources,
            montage_tile_payloads=payloads,
        )
        _present_vispy_tiled(view,
            placeholder,
            histogramData=None,
            histogramPlotData=np.arange(4, dtype=np.float32) + 20.0,
            geometry=_montage_geometry(),
            levels=(0.0, 1.0),
            histogramRange=(20.0, 23.0),
            montage_dirty_tiles=(),
            montage_tile_source_ids=sources,
            montage_tile_payloads=payloads,
        )

        timing = view.lastImageUploadTiming()
        assert not calls
        assert view._vispy_histogram_update_pending
        view.render_coordinator = SimpleNamespace(interactive_active=False, has_pending_render=False)
        view._flush_pending_vispy_histogram_update()
        assert calls
        assert np.asarray(calls[-1][1])[0] == pytest.approx(20.0)
        assert timing.tile_layer_items_updated == 0
        assert timing.tile_layer_items_skipped == 2
        assert timing.tile_layer_texture_uploads == 0
        assert tuple(float(value) for value in view.getHistogramDataBounds()) == (20.0, 23.0)
    finally:
        view.close()


def test_vispy_single_tile_histogram_comes_from_payload_after_montage(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    try:
        montage = np.arange(4, dtype=np.float32)
        _present_vispy_tiled(
            view,
            np.zeros((2, 5), dtype=np.float32),
            histogramData=None,
            histogramPlotData=montage,
            geometry=_montage_geometry(),
            levels=(0.0, 3.0),
            histogramRange=(0.0, 3.0),
        )

        single_hist = np.full((2, 2), 42.0, dtype=np.float32)
        payloads = {
            0: DisplayTilePayload(0, 0, np.zeros((2, 2), dtype=np.float32), single_hist, ("single", 0)),
        }
        _present_vispy_tiled(
            view,
            np.zeros((2, 2), dtype=np.float32),
            histogramData=None,
            histogramPlotData=None,
            geometry=_single_tile_montage_geometry(),
            montage_tile_payloads=payloads,
            levels=(40.0, 45.0),
            histogramRange=(40.0, 45.0),
        )
        view._flush_pending_vispy_histogram_update()

        assert view.histogramSource is None
        np.testing.assert_array_equal(view.histogramPlotSource, single_hist)
        np.testing.assert_array_equal(view.histogramImageItem.image, single_hist)
    finally:
        view.close()


def test_vispy_tiled_histogram_refreshes_when_source_changes_with_same_revision(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = VisPyImageView2D()
    try:
        left = DisplayTilePayload(
            0,
            0,
            np.zeros((2, 2), dtype=np.float32),
            np.arange(4, dtype=np.float32).reshape(2, 2),
            ("montage", 0),
        )
        right = DisplayTilePayload(
            1,
            1,
            np.zeros((2, 2), dtype=np.float32),
            np.arange(4, 8, dtype=np.float32).reshape(2, 2),
            ("montage", 1),
        )
        montage_payloads = {0: left, 1: right}
        montage_delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts=montage_payloads,
            active_tiles=(0, 1),
            planned_tiles=(0, 1),
        )
        view.setTiledPresentation(
            geometry=_montage_geometry(),
            tile_state=TilePresentationState(montage_payloads),
            tile_delta=montage_delta,
            histogramPlotData=np.concatenate([left.histogram_data.ravel(), right.histogram_data.ravel()]),
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
        )
        view._flush_pending_vispy_histogram_update()
        assert view.histogramImageItem.image is not None

        single_hist = np.full((2, 2), 42.0, dtype=np.float32)
        single_payloads = {
            0: DisplayTilePayload(0, 0, np.zeros((2, 2), dtype=np.float32), single_hist, ("single", 0)),
        }
        single_delta = TilePresentationDelta(
            structure_revision=1,
            payload_revision=1,
            visibility_revision=1,
            level_revision=1,
            histogram_revision=1,
            viewport_revision=1,
            upserts=single_payloads,
            active_tiles=(0,),
            planned_tiles=(0,),
        )
        view.setTiledPresentation(
            geometry=_single_tile_montage_geometry(),
            tile_state=TilePresentationState(single_payloads),
            tile_delta=single_delta,
            histogramPlotData=None,
            levels=(0.0, 9.0),
            histogramRange=(0.0, 9.0),
        )
        view._flush_pending_vispy_histogram_update()

        np.testing.assert_array_equal(view.histogramImageItem.image, single_hist)
    finally:
        view.close()


def test_vispy_tiled_histogram_refreshes_same_shape_same_revision_plot_data(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState

    view = VisPyImageView2D()
    try:
        first_hist = np.arange(4, dtype=np.float32).reshape(2, 2)
        second_hist = np.full((2, 2), 9.0, dtype=np.float32)
        first_payloads = {
            0: DisplayTilePayload(0, 0, np.zeros((2, 2), dtype=np.float32), first_hist, ("same-source", 0)),
        }
        second_payloads = {
            0: DisplayTilePayload(0, 0, np.zeros((2, 2), dtype=np.float32), second_hist, ("same-source", 0)),
        }
        for payloads in (first_payloads, second_payloads):
            delta = TilePresentationDelta(
                structure_revision=1,
                payload_revision=1,
                visibility_revision=1,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=payloads,
                active_tiles=(0,),
                planned_tiles=(0,),
            )
            view.setTiledPresentation(
                geometry=_single_tile_montage_geometry(),
                tile_state=TilePresentationState(payloads),
                tile_delta=delta,
                histogramPlotData=None,
                levels=(0.0, 10.0),
                histogramRange=(0.0, 10.0),
            )
            view._flush_pending_vispy_histogram_update()

        np.testing.assert_array_equal(view.histogramImageItem.image, second_hist)
    finally:
        view.close()


def test_vispy_payload_histogram_does_not_replace_tile_shader_source(qt_app, monkeypatch):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D
    from arrayscope.display.model.frame import DisplayTilePayload

    view = VisPyImageView2D()
    observed = []
    original = view._update_vispy_tile_layer

    def record_update(*args, **kwargs):
        observed.append(kwargs.get("histogram_data"))
        return original(*args, **kwargs)

    monkeypatch.setattr(view, "_update_vispy_tile_layer", record_update)
    try:
        single_hist = np.full((2, 2), 42.0, dtype=np.float32)
        payloads = {
            0: DisplayTilePayload(0, 0, np.ones((2, 2), dtype=np.float32), single_hist, ("single", 0)),
        }
        _present_vispy_tiled(
            view,
            np.zeros((2, 2), dtype=np.float32),
            histogramData=None,
            histogramPlotData=None,
            geometry=_single_tile_montage_geometry(),
            montage_tile_payloads=payloads,
            levels=(40.0, 45.0),
            histogramRange=(40.0, 45.0),
        )

        assert observed == [None]
        assert view._vispy_gpu_montage_layer.last_stats.presented_tiles == (0,)
    finally:
        view.close()


def test_vispy_frame_presentation_refreshes_histogram_curve(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    try:
        data = np.arange(100, dtype=np.float32).reshape(10, 10)

        _present_vispy_tiled(view,
            data,
            histogramData=data,
            histogramPlotData=None,
            levels=(0.0, 99.0),
            histogramRange=(0.0, 99.0),
        )
        qt_app.processEvents()

        x, y = view.histogram.item.plot.getData()
        assert x is not None and len(x) > 0
        assert y is not None and len(y) > 0
        assert getattr(view.histogramImageItem, "image", None) is not None
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
        _present_vispy_tiled(view,
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

        _present_vispy_tiled(view,
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
        _present_vispy_tiled(view,
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
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
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


def test_vispy_roi_visuals_do_not_register_pyqtgraph_scene_items(qt_app):
    from arrayscope.display.interaction import InteractionTarget
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("rectangle", rect=(3.0, 4.0, 8.0, 6.0), color=(255, 32, 16))

        pyqt_item, _selection = view._roi_items[selection.id]
        assert pyqt_item is None
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
        state = view.interaction_controller.set_hover(
            InteractionTarget("roi", object_id=selection.id, part="handle", geometry_kind="rectangle", handle_index=0),
            point=(11.0, 10.0),
        )
        view.sync_interaction_state(state)
        assert view._vispy_hovered_roi_id == selection.id

        assert view.removeRoi(selection.id)
        assert selection.id not in view._vispy_roi_visuals
        assert selection.id not in view._vispy_roi_handle_visuals
    finally:
        view.close()


def test_vispy_polyline_roi_removal_does_not_touch_pyqtgraph_scene(qt_app):
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("polyline", points=((3.0, 4.0), (8.0, 6.0), (12.0, 5.0)))

        pyqt_item, stored = view._roi_items[selection.id]
        assert pyqt_item is None
        assert stored == selection
        assert selection.id in view._vispy_roi_visuals

        assert view.removeRoi(selection.id)
        assert selection.id not in view._roi_items
        assert selection.id not in view._vispy_roi_visuals

        selection = view.createRoi("polyline", points=((2.0, 2.0), (4.0, 8.0), (10.0, 10.0)))
        view.clearRois()
        assert selection.id not in view._roi_items
        assert selection.id not in view._vispy_roi_visuals
    finally:
        view.close()


def test_vispy_roi_visuals_update_during_live_region_changes(qt_app):
    from arrayscope.display.interaction import InteractionTarget
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    changed = []
    view.roiChanged.connect(lambda roi_id, geometry: changed.append((roi_id, geometry)))
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("rectangle", rect=(3.0, 4.0, 8.0, 6.0), color=(255, 32, 16))
        handle_visual = view._vispy_roi_handle_visuals[selection.id][0]

        view._begin_pointer_capture(
            InteractionTarget("roi", object_id=selection.id, part="body", geometry_kind="rectangle"),
            (6.0, 7.0),
        )
        result = view.interaction_controller.update_capture((8.0, 10.0))
        view._apply_drag_result(result)

        assert len(changed) == 1
        live_selection = dict((roi.id, roi) for roi in view.roiSelections())[selection.id]
        assert live_selection.geometry.rect[:2] == (5.0, 7.0)
        assert view._vispy_roi_visuals[selection.id].visible
        assert view._vispy_roi_handle_visuals[selection.id][0].visible
        assert view._vispy_roi_handle_visuals[selection.id][0] is handle_visual
    finally:
        view.close()


def test_vispy_line_roi_has_reused_endpoint_handles_and_hover_cursor(qt_app):
    from arrayscope.display.interaction import InteractionTarget
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        selection = view.createRoi("line", points=((3.0, 4.0), (11.0, 8.0)), color=(40, 190, 255))
        marker = view._vispy_roi_handle_visuals[selection.id][0]

        assert marker.visible
        state = view.interaction_controller.set_hover(
            InteractionTarget("roi", object_id=selection.id, part="handle", geometry_kind="line", handle_index=0),
            point=(3.0, 4.0),
        )
        view.sync_interaction_state(state)
        assert view._vispy_hovered_roi_id == selection.id

        view._begin_pointer_capture(
            InteractionTarget("roi", object_id=selection.id, part="body", geometry_kind="line"),
            (7.0, 6.0),
        )
        result = view.interaction_controller.update_capture((8.0, 8.0))
        view._apply_drag_result(result)

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
    from arrayscope.display.interaction import InteractionTarget
    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    view = VisPyImageView2D()
    data = np.linspace(0.0, 1.0, 20 * 24, dtype=np.float32).reshape(20, 24)
    moved = []
    view.set_profile_marker_callback(lambda x, y: moved.append((x, y)))
    try:
        _present_vispy_tiled(view,data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
        view.setProfileMarker(8.0, 9.0, visible=True)

        assert view._profile_vline.scene() is None
        assert view._profile_hline.scene() is None
        assert view._profile_handle.scene() is None
        assert not view._profile_vline.isVisible()
        assert not view._profile_hline.isVisible()
        assert not view._profile_handle.isVisible()
        assert {"profile_v", "profile_h", "profile_handle_x", "profile_handle_y", "profile_handle_dot"} <= set(view._vispy_profile_visuals)
        assert all(visual.visible for visual in view._vispy_profile_visuals.values())
        assert view._vispy_profile_visuals["profile_handle_dot"].order == 10_002
        state = view.interaction_controller.set_hover(InteractionTarget("profile", part="center"), point=(8.0, 9.0))
        view.sync_interaction_state(state)
        assert view._vispy_profile_hover_part == "center"

        view._begin_pointer_capture(InteractionTarget("profile", part="center"), (8.0, 9.0))
        result = view.interaction_controller.update_capture((10.0, 11.0))
        view._apply_drag_result(result)

        assert moved
        assert moved[-1] == (10.0, 11.0)
        assert view.profileMarkerPosition() == (10.0, 11.0)
        assert all(visual.visible for visual in view._vispy_profile_visuals.values())
        assert view._vispy_profile_hover_part == "center"

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
        assert tuple(int(getattr(visual, "order", 0)) for visual in view._vispy_overlay_visuals) == (5, 6)
        view._vispy_gpu_montage_layer._ensure_visual_count(1)
        view._vispy_gpu_montage_layer._visuals_by_page[0].visible = True
        tile_order = int(getattr(view._vispy_gpu_montage_layer._visuals_by_page[0], "order", 0))
        assert tile_order > max(int(getattr(visual, "order", 0)) for visual in view._vispy_overlay_visuals)
        assert view.vispyPresentationDiagnostics()["overlays_above_tiles"] is False
        visuals = tuple(view._vispy_overlay_visuals)

        view.setMontageTileOverlays(overlays)
        assert tuple(view._vispy_overlay_visuals) == visuals

        view.clearMontageTileOverlays()
        assert view.montageTileOverlayCount() == 0
        assert tuple(view._vispy_overlay_visuals) == visuals
        assert all(not visual.visible for visual in view._vispy_overlay_visuals)
    finally:
        view.close()
