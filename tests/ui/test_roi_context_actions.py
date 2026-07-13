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


def test_line_edit_bubble_applies_text(qtbot):
    from arrayscope.ui.bubbles import LineEditBubble
    from pyqtgraph.Qt import QtCore

    accepted = []
    bubble = LineEditBubble(None, initial="old", on_accept=accepted.append)
    qtbot.addWidget(bubble)
    bubble.open_at(QtCore.QPoint(100, 100), focus_widget=bubble.edit)
    bubble.edit.setText("new name")
    bubble.edit.returnPressed.emit()
    assert accepted == ["new name"]


def test_color_swatch_bubble_picks_color(qtbot):
    from arrayscope.core.roi_store import DEFAULT_ROI_COLORS
    from arrayscope.ui.bubbles import ColorSwatchBubble
    from pyqtgraph.Qt import QtCore, QtWidgets

    picked = []
    bubble = ColorSwatchBubble(None, colors=DEFAULT_ROI_COLORS, current=DEFAULT_ROI_COLORS[0], on_accept=picked.append)
    qtbot.addWidget(bubble)
    bubble.open_at(QtCore.QPoint(100, 100))
    swatches = [b for b in bubble.findChildren(QtWidgets.QToolButton) if b.objectName() == "ColorSwatchButton"]
    assert len(swatches) == len(DEFAULT_ROI_COLORS)
    swatches[1].click()
    assert picked == [tuple(DEFAULT_ROI_COLORS[1][:3])]


def test_roi_overlay_panel_renders_structured_rows(qtbot):
    from arrayscope.display.roi_items import MovableInfoPanel

    panel = MovableInfoPanel()
    qtbot.addWidget(panel)
    panel.set_rows((("1", "rectangle", "n=36", "mean=4.2"), ("Lesion", "line", "n=25", "")))
    panel.show()
    text = panel.text()
    assert "n=36" in text
    assert "Lesion" in text
    panel.setText("plain fallback")
    assert panel.text() == "plain fallback"


def test_profile_badges_track_dock_visibility(qtbot):
    win = _window(qtbot, data=np.random.default_rng(0).normal(size=(24, 24, 4)))
    badge = win.dimension_strip.chip(0).index_badge
    live = win.widgets["buttons"]["display"]["live_profile"]

    live.setChecked(True)
    process_events(qtbot, count=12)
    assert win.profile_dock.isVisible()
    assert badge.isChecked(), "profile axis badge should highlight while the dock is visible"

    win.layout_manager.set_profile_dock_visible_from_user(False)
    process_events(qtbot, count=12)
    assert not badge.isChecked(), "badges must un-highlight when the profile dock closes"
    # Live profile state and the crosshair survive the dock closing.
    assert live.isChecked()
    assert win.img_view.profileMarkerPosition() is not None

    win.layout_manager.set_profile_dock_visible_from_user(True)
    process_events(qtbot, count=12)
    assert badge.isChecked(), "reopening the dock restores the badge highlight"


def test_live_profile_context_click_recovers_from_stale_checked_state(qtbot):
    win = _window(qtbot, data=np.random.default_rng(0).normal(size=(24, 24, 4)))
    live = win.widgets["buttons"]["display"]["live_profile"]
    live.setChecked(True)
    process_events(qtbot, count=12)

    # Simulate the stale state: button checked but the marker was cleared.
    win.renderer._clear_live_profile_marker()
    process_events(qtbot, count=4)
    assert live.isChecked()
    assert win.img_view.profileMarkerPosition() is None

    # One context-menu activation must fully re-enable live profile.
    win._set_live_profile_from_context(True, (5.0, 5.0))
    process_events(qtbot, count=12)
    assert win.img_view.profileMarkerPosition() == (5.0, 5.0)
