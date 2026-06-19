from __future__ import annotations

from arrayscope.core.roi import RoiGeometry, RoiKind, RoiSelection
from arrayscope.display.interaction import (
    CursorIntent,
    DisplayInteractionController,
    InteractionTarget,
    PointerPhase,
    drag_profile_position,
    drag_roi_geometry,
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
    assert controller.end_capture().target.object_id == "r1"


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


def test_rectangle_handle_drag_is_computed_by_shared_controller():
    geometry = RoiGeometry(RoiKind.RECTANGLE, rect=(2.0, 3.0, 4.0, 5.0))
    target = InteractionTarget("roi", "r1", "handle", "rectangle", handle_index=0)
    controller = DisplayInteractionController()

    controller.begin_capture(target, (6.0, 8.0), roi_geometry=geometry)
    update = controller.update_capture((9.0, 11.0))
    finished = controller.end_capture()

    assert update.geometry.rect == (2.0, 3.0, 7.0, 8.0)
    assert finished.geometry == update.geometry
    assert finished.delta == (3.0, 3.0)
    assert controller.state.phase is PointerPhase.IDLE


def test_line_endpoint_and_body_drags_share_pure_geometry_rules():
    geometry = RoiGeometry(RoiKind.LINE, points=((1.0, 2.0), (5.0, 6.0)))
    endpoint = InteractionTarget("roi", "line", "handle", "line", handle_index=1)
    body = InteractionTarget("roi", "line", "body", "line")

    resized = drag_roi_geometry(geometry, endpoint, origin=(5.0, 6.0), point=(8.0, 9.0))
    moved = drag_roi_geometry(geometry, body, origin=(0.0, 0.0), point=(2.0, -1.0))

    assert resized.points == ((1.0, 2.0), (8.0, 9.0))
    assert moved.points == ((3.0, 1.0), (7.0, 5.0))


def test_profile_drag_parts_constrain_semantic_axes():
    position = (4.0, 7.0)

    assert drag_profile_position(position, InteractionTarget("profile", part="vertical"), delta=(3.0, 9.0)) == (7.0, 7.0)
    assert drag_profile_position(position, InteractionTarget("profile", part="horizontal"), delta=(3.0, 9.0)) == (4.0, 16.0)
    assert drag_profile_position(position, InteractionTarget("profile", part="center"), delta=(3.0, 9.0)) == (7.0, 16.0)
