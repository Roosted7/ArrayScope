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

        item.setPos(5.0, 7.0)
        qt_app.processEvents()

        assert changed
        live_selection = dict((roi.id, roi) for roi in view.roiSelections())[selection.id]
        assert live_selection.geometry.rect[:2] == (5.0, 7.0)
        assert view._vispy_roi_visuals[selection.id].visible
        assert view._vispy_roi_handle_visuals[selection.id][0].visible
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
        assert len(view._vispy_overlay_visuals) == 4
        assert all(visual.visible for visual in view._vispy_overlay_visuals)

        view.clearMontageTileOverlays()
        assert view.montageTileOverlayCount() == 0
        assert view._vispy_overlay_visuals == []
    finally:
        view.close()
