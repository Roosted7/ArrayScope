import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.ui.helpers import clear_arrayscope_settings, process_events


def _window(qtbot, data=None):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    win = ArrayScopeWindow(np.arange(32 * 32, dtype=float).reshape(32, 32) if data is None else data)
    qtbot.addWidget(win)
    win.resize(900, 640)
    win.show()
    process_events(qtbot)
    return win


def test_roi_at_image_point_finds_topmost_roi(qtbot):
    from arrayscope.core.roi import RoiKind

    win = _window(qtbot)
    selection = win.img_view.createRoi(RoiKind.RECTANGLE, rect=(4, 4, 10, 8))
    process_events(qtbot)

    hit = win._roi_at_image_point((8.0, 8.0))
    assert hit is not None and str(hit.id) == str(selection.id)
    assert win._roi_at_image_point((30.0, 30.0)) is None
    assert win._roi_at_image_point(None) is None


def test_update_roi_selection_renames_and_recolors(qtbot):
    from arrayscope.core.roi import RoiKind

    win = _window(qtbot)
    selection = win.img_view.createRoi(RoiKind.RECTANGLE, rect=(4, 4, 10, 8))
    process_events(qtbot)

    win._update_roi_selection(selection.id, label="Lesion", color=(10, 200, 30))
    process_events(qtbot)

    updated = win.roi_store.get(str(selection.id))
    assert updated is not None
    assert updated.label == "Lesion"
    assert updated.color == (10, 200, 30)
    mirrored = [s for s in win.img_view.roiSelections() if str(s.id) == str(selection.id)]
    assert mirrored and mirrored[0].label == "Lesion"


def test_hud_context_rows_describe_hovered_roi(qtbot):
    from arrayscope.core.roi import RoiKind
    from arrayscope.display.interaction import InteractionTarget

    win = _window(qtbot)
    selection = win.img_view.createRoi(RoiKind.RECTANGLE, rect=(4, 4, 10, 8))
    process_events(qtbot)

    controller = win.img_view.interaction_controller
    win.img_view.sync_interaction_state(
        controller.set_hover(InteractionTarget("roi", object_id=str(selection.id)), point=(8.0, 8.0))
    )
    rows = win._hud_context_rows()
    assert rows, "hovering a ROI should produce HUD context rows"
    assert selection.label in rows[0][1]

    win.img_view.sync_interaction_state(controller.clear_hover())
    assert win._hud_context_rows() == ()


def test_pixel_hud_renders_multiple_rows(qtbot):
    from arrayscope.ui.hud import PixelHud
    from pyqtgraph.Qt import QtCore

    hud = PixelHud()
    qtbot.addWidget(hud)
    hud.show_rows_near((("crop", "Rectangle 1 · rectangle"), (None, "(3, 4) = 1.5")), QtCore.QPointF(10, 10))
    assert hud.isVisible()
    assert "Rectangle 1" in hud.text()
    assert "(3, 4) = 1.5" in hud.text()

    hud.show_text_near("(1, 1) = 2.0", QtCore.QPointF(5, 5))
    assert hud.text() == "(1, 1) = 2.0"
