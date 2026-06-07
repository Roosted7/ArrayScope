import os

import numpy as np

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


def test_imageview_inspection_tool_validation(qt_app):
    import pytest

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
