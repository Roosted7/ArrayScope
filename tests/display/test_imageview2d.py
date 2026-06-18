import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def test_profile_marker_callback_replacement_and_programmatic_move(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    calls = []

    view.set_profile_marker_callback(lambda x, y: calls.append(("first", x, y)))
    view.set_profile_marker_callback(lambda x, y: calls.append(("second", x, y)))

    view.setProfileMarker(1, 2, visible=True)
    assert calls == []

    view._profile_vline.setValue(3)
    assert len(calls) == 1
    assert calls[0][0] == "second"

    view.clear_profile_marker_callback()
    view._profile_vline.setValue(4)
    assert len(calls) == 1
    view.close()


def test_evaluation_overlay_and_stale_opacity(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4))
    view.show()
    qt_app.processEvents()

    view.setImageStale(True)
    view.setEvaluationOverlay(True, "Updating view...")

    assert view.imageItem.opacity() == 0.55
    assert view._evaluation_overlay.isVisible()
    assert view._evaluation_overlay.text() == "Updating view..."

    view.setImageStale(False)
    view.setEvaluationOverlay(False)
    assert view.imageItem.opacity() == 1.0
    assert not view._evaluation_overlay.isVisible()
    view.close()


def test_montage_tile_overlays_reuse_single_graphics_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D, MontageTileOverlay

    view = ImageView2D()
    view.setImage(np.zeros((8, 8), dtype=float))
    first = (MontageTileOverlay(0, 0, 4, 4, "loading", "Loading"),)
    second = (
        MontageTileOverlay(0, 0, 4, 4, "loading", "Loading"),
        MontageTileOverlay(4, 0, 4, 4, "skipped", "Skipped"),
    )

    view.setMontageTileOverlays(first)
    item = view._montage_tile_overlay_item
    view.setMontageTileOverlays(second)

    assert view._montage_tile_overlay_item is item
    assert view.montageTileOverlayCount() == 2
    assert len(view._montage_tile_overlay_items) == 1
    view.close()


def test_update_image_data_fast_preserves_levels_and_view_range(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((8, 8), dtype=float), levels=(0.0, 10.0))
    view.getView().setRange(xRange=(1, 5), yRange=(2, 6), padding=0)
    before_range = view.getView().viewRange()
    autolevel_calls = []
    monkeypatch.setattr(view, "autoLevels", lambda: autolevel_calls.append(True))

    view.updateImageDataFast(np.ones((8, 8), dtype=float), histogramData=np.ones((8, 8), dtype=float), levels=(0.0, 10.0))

    assert autolevel_calls == []
    assert tuple(view.image.shape) == (8, 8)
    assert view.getLevels() == (0.0, 10.0)
    assert view.getView().viewRange() == before_range
    view.close()


def test_image_presentation_keeps_levels_and_histogram_range_separate(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImagePresentation(
        np.zeros((4, 4), dtype=float),
        histogramData=np.zeros((4, 4), dtype=float),
        levels=(2.0, 8.0),
        histogramRange=(0.0, 10.0),
    )

    assert tuple(float(value) for value in view.getLevels()) == (2.0, 8.0)
    assert view.getHistogramDataBounds() == (0.0, 10.0)

    view.updateImagePresentationFast(
        np.full((4, 4), 1000.0, dtype=float),
        histogramData=np.full((4, 4), 1000.0, dtype=float),
        levels=(2.0, 8.0),
        histogramRange=(0.0, 10.0),
    )

    assert tuple(float(value) for value in view.getLevels()) == (2.0, 8.0)
    assert view.getHistogramDataBounds() == (0.0, 10.0)
    view.close()


def test_update_image_data_fast_accepts_display_ready_rgb(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((4, 4, 3), dtype=np.uint8), levels=(0.0, 1.0), histogramData=np.zeros((4, 4)))
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)

    view.updateImageDataFast(rgb, histogramData=np.ones((4, 4)), levels=(0.0, 1.0), rgb_already_windowed=True)

    assert view._rgbBaseImage is None
    np.testing.assert_array_equal(view.imageDisp, rgb)
    view.close()


def test_separate_histogram_plot_source_drives_histogram_without_replacing_value_source(qt_app):
    from arrayscope.display.geometry import ViewPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    local = np.arange(16, dtype=float).reshape(4, 4)
    plot = np.linspace(100.0, 200.0, 25)
    view.setImagePresentation(
        local,
        histogramData=local.copy(),
        histogramPlotData=plot,
        levels=(110.0, 190.0),
        histogramRange=(100.0, 200.0),
    )

    assert view.histogramPlotSource is not None
    assert view.histogramImageItem.image.shape == (5, 5)
    assert view.valueAtDisplayMapping(ViewPointMapping(view_x=3, view_y=2, canvas_x=3, canvas_y=2, local_x=3, local_y=2, array_index=(2, 3))) == local[2, 3]
    assert tuple(float(value) for value in view.imageItem.levels) == (110.0, 190.0)
    view.histogram.setLevels(120.0, 180.0)
    view._on_histogram_levels_changed()
    assert tuple(float(value) for value in view.imageItem.levels) == (120.0, 180.0)
    view.close()


def test_repeated_fast_updates_do_not_rebind_same_histogram_item(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.ones((4, 4), dtype=float)
    view.setImagePresentation(data, histogramData=data, histogramPlotData=np.arange(16, dtype=float), levels=(0.0, 2.0), histogramRange=(0.0, 2.0))
    calls = []
    monkeypatch.setattr(view.histogram, "setImageItem", lambda item: calls.append(item))

    view.updateImagePresentationFast(data, histogramData=data, histogramPlotData=np.arange(16, dtype=float), levels=(0.0, 2.0), histogramRange=(0.0, 2.0))
    view.updateImagePresentationFast(data, histogramData=data, histogramPlotData=np.arange(16, dtype=float), levels=(0.0, 2.0), histogramRange=(0.0, 2.0))

    assert calls == []
    view.close()


def test_programmatic_scalar_presentation_uploads_visible_image_once(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    calls = []
    original = view.imageItem.setImage

    def recording_set_image(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(view.imageItem, "setImage", recording_set_image)
    data = np.arange(16, dtype=float).reshape(4, 4)

    view.setImagePresentation(data, histogramData=data, levels=(2.0, 12.0), histogramRange=(0.0, 15.0))

    assert len(calls) == 1
    assert tuple(float(value) for value in view.imageItem.levels) == (2.0, 12.0)
    view.close()


def test_same_memory_fast_refresh_uses_no_new_data_path(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.zeros((4, 4), dtype=float)
    view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    calls = []
    original = view.imageItem.setImage

    def recording_set_image(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(view.imageItem, "setImage", recording_set_image)
    data[:] = 2.0

    view.updateImagePresentationFast(data, histogramData=data, levels=(0.0, 2.0), histogramRange=(0.0, 2.0))

    assert calls
    assert calls[0][0][0] is None
    assert view.lastImageUploadTiming().fast_same_object is True
    view.close()


def test_tile_layer_presentation_creates_positioned_tile_items_and_syncs_levels(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=float).reshape(2, 5)
    hist = canvas.copy()

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=np.arange(16, dtype=float),
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
    )

    assert view.montageDisplayMode() == "tile_layer"
    assert not view.imageItem.isVisible()
    states = view._montage_tile_layer.states
    assert set(states) == {0, 1}
    assert states[0].item.pos().x() == 0.0
    assert states[1].item.pos().x() == 3.0

    view.histogram.setLevels(2.0, 8.0)
    view._on_histogram_levels_changed()

    assert tuple(float(value) for value in states[0].item.levels) == (2.0, 8.0)
    view.clearMontageTileLayer()
    assert view.montageDisplayMode() == "canvas"
    assert view.imageItem.isVisible()
    view.close()



def test_tile_layer_direct_payloads_avoid_canvas_slicing(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState
    from arrayscope.window.display_frame import DisplayTilePayload

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    placeholder = np.broadcast_to(np.zeros((1, 1), dtype=np.float32), (2, 5))
    left = np.array([[1, 2], [3, 4]], dtype=np.float32)
    right = np.array([[5, 6], [7, 8]], dtype=np.float32)
    payloads = {
        0: DisplayTilePayload(0, 0, left, left * 10, ("payload", 0)),
        1: DisplayTilePayload(1, 1, right, right * 10, ("payload", 1)),
    }

    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 80.0),
        histogramRange=(0.0, 80.0),
        montage_tile_payloads=payloads,
        montage_tile_source_ids={0: ("payload", 0), 1: ("payload", 1)},
    )

    states = view._montage_tile_layer.states
    assert set(states) == {0, 1}
    np.testing.assert_array_equal(states[0].item.image, left)
    np.testing.assert_array_equal(states[1].item.image, right)
    assert states[1].item.pos().x() == 3.0

    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(10.0, 70.0),
        histogramRange=(0.0, 80.0),
        montage_dirty_tiles=(),
        montage_tile_payloads=payloads,
        montage_tile_source_ids={0: ("payload", 0), 1: ("payload", 1)},
    )

    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_items_skipped == 2
    assert timing.visible_bytes == 0
    assert tuple(float(value) for value in states[0].item.levels) == (10.0, 70.0)
    view.close()

def test_montage_canvas_presentation_sets_image_item_world_origin(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=1),
        montage_origin_x=0,
        montage_origin_y=3,
    )

    view.setImagePresentation(
        np.zeros((2, 5), dtype=float),
        histogramData=np.zeros((2, 5), dtype=float),
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        image_origin=(geometry.montage_origin_x, geometry.montage_origin_y),
    )

    assert view.imageItem.pos().x() == 0.0
    assert view.imageItem.pos().y() == 3.0
    view.close()


def test_tile_layer_items_use_world_positions_when_canvas_origin_shifted(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 4)).with_montage_axis(2, columns=2, indices=(0, 1, 2, 3), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1, 2, 3), tile_shape=(2, 2), columns=2, rows=2, gap=1),
        montage_origin_x=0,
        montage_origin_y=3,
        montage_tile_states=(MontageTileState.UNLOADED, MontageTileState.UNLOADED, MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=float).reshape(2, 5)

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=canvas.copy(),
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
    )

    states = view._montage_tile_layer.states
    assert set(states) == {2, 3}
    assert states[2].item.pos().y() == 3.0
    assert states[3].item.pos().x() == 3.0
    assert states[3].item.pos().y() == 3.0
    view.close()


def test_graphics_layer_z_order_places_tools_above_tiles(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D, MontageTileOverlay
    from arrayscope.display.layers import Z_MONTAGE_LOADING_OVERLAY, Z_PROFILE_MARKER, Z_ROI, Z_TILE_IMAGE
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,),
    )
    canvas = np.zeros((2, 2), dtype=float)
    view.setMontageTileLayerPresentation(canvas, histogramData=canvas, histogramPlotData=None, geometry=geometry, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    roi = view.createRoi(RoiKind.RECTANGLE, rect=(0, 0, 1, 1))
    view.setProfileMarker(0, 0, visible=True)
    view.setMontageTileOverlays((MontageTileOverlay(0, 0, 1, 1, "loading", "Loading"),))

    tile_z = view._montage_tile_layer.states[0].item.zValue()
    roi_z = view._roi_items[roi.id][0].zValue()
    assert tile_z == Z_TILE_IMAGE
    assert roi_z == Z_ROI
    assert view._profile_handle.zValue() == Z_PROFILE_MARKER
    assert view._montage_tile_overlay_item.zValue() == Z_MONTAGE_LOADING_OVERLAY
    assert tile_z < roi_z < view._profile_handle.zValue() < view._montage_tile_overlay_item.zValue()
    view.close()


def test_inactive_tile_layer_items_are_removed_from_scene(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry_loaded = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.zeros((2, 5), dtype=float)
    view.setMontageTileLayerPresentation(canvas, histogramData=canvas, histogramPlotData=None, geometry=geometry_loaded, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    removed_item = view._montage_tile_layer.states[1].item
    geometry_one = DisplayGeometry(
        view_state=geometry_loaded.view_state,
        display_shape=(2, 5),
        montage=geometry_loaded.montage,
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.UNLOADED),
    )

    view.setMontageTileLayerPresentation(canvas, histogramData=canvas, histogramPlotData=None, geometry=geometry_one, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

    assert 1 not in view._montage_tile_layer.states
    assert removed_item.scene() is None
    view.close()


def test_tile_layer_clean_commit_skips_tile_and_histogram_uploads(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=np.float32).reshape(2, 5)
    hist = canvas.copy()
    plot = np.arange(16, dtype=np.float32)

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=plot,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
    )

    calls = []
    for state in view._montage_tile_layer.states.values():
        monkeypatch.setattr(state.item, "setImage", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(view.histogramImageItem, "setImage", lambda *args, **kwargs: calls.append((args, kwargs)))

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=plot,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
        montage_dirty_tiles=(),
    )

    timing = view.lastImageUploadTiming()
    assert calls == []
    assert timing.tile_layer_visible_items == 2
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_items_skipped == 2
    assert timing.tile_layer_upload_ms == 0.0
    assert timing.visible_bytes == 0
    assert timing.histogram_bytes == 0
    view.close()


def test_tile_layer_dirty_commit_updates_only_dirty_tile(qt_app, monkeypatch):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.arange(10, dtype=np.float32).reshape(2, 5)
    hist = canvas.copy()
    view.setMontageTileLayerPresentation(canvas, histogramData=hist, histogramPlotData=None, geometry=geometry, levels=(0.0, 9.0), histogramRange=(0.0, 9.0))

    calls = []
    for tile_number, state in view._montage_tile_layer.states.items():
        monkeypatch.setattr(state.item, "setImage", lambda *args, tile_number=tile_number, **kwargs: calls.append(tile_number))

    canvas[:, 3:5] = 99
    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 9.0),
        histogramRange=(0.0, 9.0),
        montage_dirty_tiles=(1,),
    )

    assert calls == [1]
    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_items_updated == 1
    assert timing.tile_layer_items_skipped == 1
    view.close()


def test_rgb_tile_layer_clean_commit_does_not_rewindow(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    view.setMontageTileLayerPresentation(canvas, histogramData=hist, histogramPlotData=None, geometry=geometry, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

    for state in view._montage_tile_layer.states.values():
        assert state.rgb_base is not None
        assert state.rgb_base.dtype == np.float32

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        montage_dirty_tiles=(),
    )

    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_rgb_window_tiles == 0
    assert timing.rgb_window_ms == 0.0
    assert timing.tile_layer_rgb_window_ms == 0.0
    assert timing.tile_layer_upload_ms == 0.0
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_items_skipped == 2
    view.close()


def test_rgb_tile_layer_level_change_rewindows_cached_bases(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    view.setMontageTileLayerPresentation(canvas, histogramData=hist, histogramPlotData=None, geometry=geometry, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

    before = {tile: np.array(state.item.image, copy=True) for tile, state in view._montage_tile_layer.states.items()}
    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.5, 1.0),
        histogramRange=(0.0, 1.0),
        montage_dirty_tiles=(),
    )

    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_rgb_window_tiles == 2
    assert timing.tile_layer_items_updated == 2
    assert timing.tile_layer_rgb_window_ms > 0.0
    assert timing.tile_layer_upload_ms > 0.0
    for tile, state in view._montage_tile_layer.states.items():
        assert not np.array_equal(state.item.image, before[tile])
    view.close()


def test_rgb_tile_layer_pruned_source_cache_rewindows_from_canvas(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    view._montage_tile_layer._rgb_source_cache_budget_bytes = 1
    view.setMontageTileLayerPresentation(canvas, histogramData=hist, histogramPlotData=None, geometry=geometry, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))

    assert all(state.rgb_base is None for state in view._montage_tile_layer.states.values())
    before = {tile: np.array(state.item.image, copy=True) for tile, state in view._montage_tile_layer.states.items()}
    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.5, 1.0),
        histogramRange=(0.0, 1.0),
        montage_dirty_tiles=(),
    )

    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_rgb_window_tiles == 2
    assert timing.tile_layer_items_updated == 2
    for tile, state in view._montage_tile_layer.states.items():
        assert not np.array_equal(state.item.image, before[tile])
    view.close()


def test_rgb_tile_layer_live_level_change_rewindows_pruned_sources(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    view._montage_tile_layer._rgb_source_cache_budget_bytes = 1
    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )

    assert all(state.rgb_base is None for state in view._montage_tile_layer.states.values())
    before = {tile: np.array(state.item.image, copy=True) for tile, state in view._montage_tile_layer.states.items()}
    view.setLevels(0.5, 1.0)

    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_rgb_window_tiles == 2
    assert timing.tile_layer_items_updated == 2
    assert timing.tile_layer_items_skipped == 0
    for tile, state in view._montage_tile_layer.states.items():
        assert tuple(state.levels) == (0.5, 1.0)
        assert not np.array_equal(state.item.image, before[tile])

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.5, 1.0),
        histogramRange=(0.0, 1.0),
        montage_dirty_tiles=(),
    )
    timing = view.lastImageUploadTiming()
    assert timing.tile_layer_items_updated == 0
    assert timing.tile_layer_rgb_window_tiles == 0
    view.close()


def test_tile_layer_inactive_tile_is_removed_immediately(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    loaded = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 2)).with_montage_axis(2, columns=2, indices=(0, 1), text=":"),
        display_shape=(2, 5),
        montage=MontageGeometry(indices=(0, 1), tile_shape=(2, 2), columns=2, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.LOADED),
    )
    canvas = np.full((2, 5, 3), 200, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(2, 5)
    view.setMontageTileLayerPresentation(canvas, histogramData=hist, histogramPlotData=None, geometry=loaded, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    assert view._montage_tile_layer.states[1].rgb_base is not None
    removed_item = view._montage_tile_layer.states[1].item

    hidden = DisplayGeometry(
        view_state=loaded.view_state,
        display_shape=loaded.display_shape,
        montage=loaded.montage,
        montage_tile_states=(MontageTileState.LOADED, MontageTileState.UNLOADED),
    )
    view.setMontageTileLayerPresentation(canvas, histogramData=hist, histogramPlotData=None, geometry=hidden, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), montage_dirty_tiles=())
    assert 1 not in view._montage_tile_layer.states
    assert removed_item.scene() is None
    view.close()


def test_display_ready_rgb_tile_layer_level_change_keeps_uint8_item_levels(qt_app):
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.montage import MontageTileState

    view = ImageView2D()
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((2, 2, 1)).with_montage_axis(2, columns=1, indices=(0,), text=":"),
        display_shape=(2, 2),
        montage=MontageGeometry(indices=(0,), tile_shape=(2, 2), columns=1, rows=1, gap=1),
        montage_tile_states=(MontageTileState.LOADED,),
    )
    canvas = np.full((2, 2, 3), 128, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(2, 2)
    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
    )

    view.setMontageTileLayerPresentation(
        canvas,
        histogramData=hist,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.5, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
        montage_dirty_tiles=(),
    )

    state = view._montage_tile_layer.states[0]
    assert tuple(float(value) for value in state.item.levels) == (0.0, 255.0)
    assert view.lastImageUploadTiming().tile_layer_items_updated == 0
    view.close()


def test_complex_rgb_histogram_levels_rewindow_display(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    magnitude = np.linspace(0.0, 1.0, 16, dtype=float).reshape(4, 4)
    view.setImagePresentation(
        rgb,
        histogramData=magnitude,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
    )
    before = np.array(view.imageDisp, copy=True)

    view.setLevels(0.5, 1.0)

    assert view._rgbBaseImage is not None
    assert not np.array_equal(view.imageDisp, before)
    view.close()


def test_display_ready_rgb_histogram_levels_do_not_rewindow_display(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 16, dtype=float).reshape(4, 4)
    view.setImagePresentation(
        rgb,
        histogramData=hist,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
    )
    before = np.array(view.imageDisp, copy=True)

    view.setLevels(0.5, 1.0)

    assert view._rgbBaseImage is None
    np.testing.assert_array_equal(view.imageDisp, before)
    view.close()


def test_scalar_image_upload_passes_levels_to_image_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 12.0))

    assert tuple(float(value) for value in view.imageItem.levels) == (2.0, 12.0)
    view.updateImageDataFast(np.ones((4, 4), dtype=float), levels=(3.0, 9.0))
    assert tuple(float(value) for value in view.imageItem.levels) == (3.0, 9.0)
    view.close()


def test_scalar_histogram_level_drag_updates_display_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(0.0, 15.0))

    view.histogram.setLevels(2.0, 8.0)
    view._on_histogram_levels_changed()

    assert tuple(float(value) for value in view.imageItem.levels) == (2.0, 8.0)
    view.close()


def test_value_at_display_mapping_ignores_mismatched_histogram_source(qt_app):
    from arrayscope.display.geometry import ViewPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.ones((4, 4), dtype=float), histogramData=np.ones((4, 4), dtype=float), levels=(0.0, 2.0))
    view.histogramSource = np.ones((8, 8), dtype=float)

    value = view.valueAtDisplayMapping(ViewPointMapping(view_x=0, view_y=0, canvas_x=0, canvas_y=0, local_x=0, local_y=0, array_index=(0, 0)))

    assert value is None
    view.close()


def test_value_at_display_mapping_uses_display_coordinates_for_montage(qt_app):
    from arrayscope.display.geometry import ViewPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.arange(12, dtype=float).reshape(3, 4)
    view.setImage(data, histogramData=data.copy(), levels=(0.0, 12.0))

    value = view.valueAtDisplayMapping(
        ViewPointMapping(
            view_x=3,
            view_y=2,
            canvas_x=3,
            canvas_y=2,
            local_x=0,
            local_y=0,
            array_index=(0, 0, 1),
            tile_number=1,
            montage_axis=2,
            montage_index=1,
        )
    )

    assert value == data[2, 3]
    view.close()


def test_profile_marker_bounds_update_when_image_shape_changes(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((8, 10), dtype=float))
    view.setProfileMarker(9, 7, visible=True)
    assert view._profile_vline.maxRange == (0, 9)
    assert view._profile_hline.maxRange == (0, 7)

    view.setImage(np.zeros((4, 5), dtype=float))
    assert view._profile_vline.maxRange == (0, 4)
    assert view._profile_hline.maxRange == (0, 3)
    assert view.profileMarkerPosition() == (4.0, 3.0)
    view.close()


def test_profile_marker_lines_hide_when_marker_center_leaves_view(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((20, 20), dtype=float))
    view.getView().setRange(xRange=(0, 10), yRange=(0, 10), padding=0)
    view.setProfileMarker(5, 5, visible=True)

    assert view._profile_handle.isVisible()
    assert view._profile_vline.isVisible()
    assert view._profile_hline.isVisible()

    view.getView().setRange(xRange=(10, 19), yRange=(10, 19), padding=0)

    assert not view._profile_handle.isVisible()
    assert not view._profile_vline.isVisible()
    assert not view._profile_hline.isVisible()
    assert view.profileMarkerPosition() == (5.0, 5.0)
    view.close()


def test_imageview_creates_polyline_and_freehand_rois(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((10, 12), dtype=float))
    created = []
    changed = []
    view.roiCreated.connect(lambda selection: created.append(selection))
    view.roiChanged.connect(lambda roi_id, geometry: changed.append((roi_id, geometry)))

    polyline = view.createRoi(RoiKind.POLYLINE, points=((1, 1), (5, 2), (8, 7)))
    freehand = view.createRoi(RoiKind.FREEHAND_POLYGON, points=((2, 2), (7, 2), (7, 8), (2, 8)))

    assert len(created) == 2
    assert polyline.geometry.kind.value == RoiKind.POLYLINE.value
    assert freehand.geometry.kind.value == RoiKind.FREEHAND_POLYGON.value
    assert freehand.geometry.points[0] == freehand.geometry.points[-1]
    assert len(view.roiSelections()) == 2

    item, _selection = view._roi_items[polyline.id]
    item.setPos(1, 0)
    view._on_roi_item_changed(polyline.id)
    assert changed[-1][0] == polyline.id
    assert changed[-1][1].points[0] == (2.0, 1.0)

    assert view.removeRoi(polyline.id)
    assert len(view.roiSelections()) == 1
    view.clearRois()
    assert view.roiSelections() == ()
    view.close()


def test_imageview_freehand_requires_points(qt_app):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((10, 12), dtype=float))

    with pytest.raises(ValueError, match="freehand ROI requires a drag path"):
        view.createRoi(RoiKind.FREEHAND_POLYGON)
    view.close()


def test_persistent_polyline_and_freehand_tools_do_not_start_drawing(qt_app):
    from pyqtgraph.Qt import QtCore, QtGui

    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((10, 12), dtype=float))
    view.setInspectionTool("roi_freehand")
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(1, 1),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )

    assert not view._handle_roi_drawing_event(event)
    assert view.beginRoiDrawingOnce("roi_freehand")
    assert view._handle_roi_drawing_event(event)
    view.cancelPendingRoiDrawing()
    view.close()


def test_imageview_inspection_tool_validation(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setInspectionTool("roi_freehand")
    assert view.inspectionTool() == "roi_freehand"

    with pytest.raises(ValueError, match="unknown inspection tool"):
        view.setInspectionTool("bad")
    view.close()


def test_imageview_preserves_view_range_for_same_shape_by_default(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((8, 10), dtype=float))
    view.getView().setRange(xRange=(2, 5), yRange=(1, 6), padding=0)
    before = view.getView().viewRange()

    view.setImage(np.ones((8, 10), dtype=float))
    after = view.getView().viewRange()

    assert after == before
    view.close()


def test_imageview_resets_view_range_when_shape_changes(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from arrayscope.display.viewport import ViewportPolicy

    view = ImageView2D()
    view.setImage(np.zeros((8, 10), dtype=float))
    view.getView().setRange(xRange=(2, 5), yRange=(1, 6), padding=0)

    view.setImage(np.ones((4, 5), dtype=float), viewport_policy=ViewportPolicy.RESET_FOR_NEW_SHAPE)
    after = view.getView().viewRange()

    assert after != [[2.0, 5.0], [1.0, 6.0]]
    view.close()


def test_programmatic_presentation_does_not_emit_user_level_signal(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    user_calls = []
    view.userLevelsChanged.connect(lambda: user_calls.append(True))

    view.setImagePresentation(
        np.zeros((4, 4), dtype=float),
        histogramData=np.zeros((4, 4), dtype=float),
        levels=(2.0, 8.0),
        histogramRange=(0.0, 10.0),
    )
    view.updateImagePresentationFast(
        np.ones((4, 4), dtype=float),
        histogramData=np.ones((4, 4), dtype=float),
        levels=(1.0, 9.0),
        histogramRange=(0.0, 10.0),
    )

    assert user_calls == []
    view.close()


def test_explicit_set_levels_emits_user_level_signal(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.zeros((4, 4), dtype=float), levels=(0.0, 1.0))
    user_calls = []
    view.userLevelsChanged.connect(lambda: user_calls.append(True))

    view.setLevels(2.0, 8.0)

    assert user_calls == [True]
    assert tuple(float(value) for value in view.getLevels()) == (2.0, 8.0)
    view.close()


def test_histogram_drag_preview_emits_user_level_once_on_finish(qt_app):
    from arrayscope.display.imageview2d import ImageView2D
    from pyqtgraph.Qt import QtCore

    view = ImageView2D()
    view.setImage(np.zeros((4, 4), dtype=float), levels=(0.0, 1.0))
    user_calls = []
    view.userLevelsChanged.connect(lambda: user_calls.append(True))

    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.1, 0.9)
    view._on_histogram_levels_changed()
    with QtCore.QSignalBlocker(view.histogram.item):
        view.histogram.setLevels(0.2, 0.8)
    view._on_histogram_levels_changed()
    view._on_histogram_level_change_finished()

    assert user_calls == [True]
    assert tuple(float(value) for value in view.getLevels()) == (0.2, 0.8)
    view.close()


def test_full_presentation_accepts_display_ready_rgb(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    hist = np.linspace(0.0, 1.0, 16, dtype=float).reshape(4, 4)

    view.setImagePresentation(
        rgb,
        histogramData=hist,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=True,
    )

    assert view._rgbBaseImage is None
    np.testing.assert_array_equal(view.imageDisp, rgb)
    view.close()
