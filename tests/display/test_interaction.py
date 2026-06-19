from __future__ import annotations

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.display.interaction import (
    CursorIntent,
    DisplayInteractionController,
    InteractionTarget,
    PointerPhase,
    hit_test_display_overlays,
)


def test_one_shot_drawing_state_is_owned_by_shared_controller():
    controller = DisplayInteractionController()

    controller.arm_drawing("roi_freehand")
    assert controller.state.phase is PointerPhase.DRAWING_ARMED
    assert controller.state.cursor_intent is CursorIntent.CROSSHAIR
    assert controller.begin_drawing((1, 2))
    assert not controller.append_drawing_point((1.1, 2.1), minimum_distance=1.0)
    assert controller.append_drawing_point((3, 4), minimum_distance=1.0)

    result = controller.finish_drawing()

    assert result.tool == "roi_freehand"
    assert result.points == ((1.0, 2.0), (3.0, 4.0))
    assert controller.state.phase is PointerPhase.IDLE
    assert controller.state.pending_draw_tool is None


def test_hover_cursor_intent_is_backend_neutral():
    controller = DisplayInteractionController()

    controller.set_hover(InteractionTarget("profile", part="vertical"))
    assert controller.state.cursor_intent is CursorIntent.RESIZE_HORIZONTAL

    controller.set_hover(InteractionTarget("roi", "r1", "handle", "rectangle", handle_index=0))
    assert controller.state.cursor_intent is CursorIntent.RESIZE_DIAGONAL

    controller.begin_capture(InteractionTarget("roi", "r1", "body", "rectangle"), (2, 3))
    assert controller.state.cursor_intent is CursorIntent.CLOSED_HAND
    assert controller.end_capture().object_id == "r1"


def test_hit_test_prioritizes_profile_then_topmost_roi():
    bottom = RoiSelection(
        id="bottom",
        label="Bottom",
        geometry=RoiGeometry(RoiKind.RECTANGLE, rect=(0.0, 0.0, 10.0, 10.0)),
    )
    top = RoiSelection(
        id="top",
        label="Top",
        geometry=RoiGeometry(RoiKind.LINE, points=((0.0, 5.0), (10.0, 5.0))),
    )

    profile = hit_test_display_overlays(
        (5.0, 5.0),
        roi_selections=(bottom, top),
        profile_position=(5.0, 5.0),
        profile_bounds=(0.0, 0.0, 10.0, 10.0),
        tolerance=0.5,
    )
    roi = hit_test_display_overlays(
        (7.0, 5.0),
        roi_selections=(bottom, top),
        tolerance=0.5,
    )

    assert profile.kind == "profile" and profile.part == "center"
    assert roi.kind == "roi" and roi.object_id == "top" and roi.part == "outline"
