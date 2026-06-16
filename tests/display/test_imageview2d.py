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


def test_scalar_image_upload_passes_levels_to_image_item(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.arange(16, dtype=float).reshape(4, 4), levels=(2.0, 12.0))

    assert tuple(float(value) for value in view.imageItem.levels) == (2.0, 12.0)
    view.updateImageDataFast(np.ones((4, 4), dtype=float), levels=(3.0, 9.0))
    assert tuple(float(value) for value in view.imageItem.levels) == (3.0, 9.0)
    view.close()


def test_value_at_display_mapping_ignores_mismatched_histogram_source(qt_app):
    from arrayscope.display.geometry import DisplayPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    view.setImage(np.ones((4, 4), dtype=float), histogramData=np.ones((4, 4), dtype=float), levels=(0.0, 2.0))
    view.histogramSource = np.ones((8, 8), dtype=float)

    value = view.valueAtDisplayMapping(DisplayPointMapping(display_x=0, display_y=0, local_x=0, local_y=0, array_index=(0, 0)))

    assert value is None
    view.close()


def test_value_at_display_mapping_uses_display_coordinates_for_montage(qt_app):
    from arrayscope.display.geometry import DisplayPointMapping
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    data = np.arange(12, dtype=float).reshape(3, 4)
    view.setImage(data, histogramData=data.copy(), levels=(0.0, 12.0))

    value = view.valueAtDisplayMapping(
        DisplayPointMapping(
            display_x=3,
            display_y=2,
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
    assert polyline.geometry.kind == RoiKind.POLYLINE
    assert freehand.geometry.kind == RoiKind.FREEHAND_POLYGON
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
