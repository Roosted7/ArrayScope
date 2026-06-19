"""Qt-free pointer, hover, and drawing interaction state.

Render backends may draw overlays differently, but they must consume one
semantic interaction state. During migration, native PyQtGraph ROI items still
own their drag mechanics; hover and one-shot drawing already use this shared
state, and native drag capture can move here incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import hypot
from typing import Iterable

from arrayscope.display.overlay_hit_test import RoiHit, hit_test_roi


ALLOWED_INSPECTION_TOOLS = frozenset(
    {"cursor", "profile", "roi_line", "roi_rectangle", "roi_polyline", "roi_freehand"}
)
ONE_SHOT_DRAW_TOOLS = frozenset({"roi_polyline", "roi_freehand"})
CROSSHAIR_TOOLS = frozenset({"profile", "roi_line", "roi_rectangle"})


class PointerPhase(Enum):
    IDLE = "idle"
    DRAWING_ARMED = "drawing_armed"
    DRAWING = "drawing"
    DRAGGING = "dragging"


class CursorIntent(Enum):
    DEFAULT = "default"
    CROSSHAIR = "crosshair"
    MOVE = "move"
    OPEN_HAND = "open_hand"
    CLOSED_HAND = "closed_hand"
    RESIZE_HORIZONTAL = "resize_horizontal"
    RESIZE_VERTICAL = "resize_vertical"
    RESIZE_DIAGONAL = "resize_diagonal"


@dataclass(frozen=True)
class InteractionTarget:
    """Semantic overlay part below the pointer or held by a drag."""

    kind: str
    object_id: str | None = None
    part: str = "body"
    geometry_kind: str | None = None
    handle_index: int | None = None
    segment_index: int | None = None


@dataclass(frozen=True)
class DrawingResult:
    tool: str
    points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class DisplayInteractionState:
    tool: str = "cursor"
    phase: PointerPhase = PointerPhase.IDLE
    pointer: tuple[float, float] | None = None
    hover: InteractionTarget | None = None
    capture: InteractionTarget | None = None
    pending_draw_tool: str | None = None
    drawing_points: tuple[tuple[float, float], ...] = ()
    revision: int = 0

    @property
    def cursor_intent(self) -> CursorIntent:
        if self.phase is PointerPhase.DRAGGING:
            return CursorIntent.CLOSED_HAND
        if self.pending_draw_tool is not None or self.phase in {PointerPhase.DRAWING_ARMED, PointerPhase.DRAWING}:
            return CursorIntent.CROSSHAIR
        if self.tool in CROSSHAIR_TOOLS:
            return CursorIntent.CROSSHAIR
        target = self.hover
        if target is None:
            return CursorIntent.DEFAULT
        if target.kind == "profile":
            if target.part == "center":
                return CursorIntent.OPEN_HAND
            if target.part == "vertical":
                return CursorIntent.RESIZE_HORIZONTAL
            if target.part == "horizontal":
                return CursorIntent.RESIZE_VERTICAL
        if target.kind == "roi":
            if target.part == "handle" and target.geometry_kind == "rectangle":
                return CursorIntent.RESIZE_DIAGONAL
            return CursorIntent.MOVE
        return CursorIntent.DEFAULT


class DisplayInteractionController:
    """Own semantic pointer state independently of a graphics library."""

    def __init__(self, state: DisplayInteractionState | None = None):
        self._state = state or DisplayInteractionState()

    @property
    def state(self) -> DisplayInteractionState:
        return self._state

    def set_tool(self, tool: str) -> DisplayInteractionState:
        tool = str(tool)
        if tool not in ALLOWED_INSPECTION_TOOLS:
            raise ValueError(f"unknown inspection tool: {tool}")
        return self._replace(tool=tool, hover=None, capture=None)

    def set_pointer(self, point: tuple[float, float] | None) -> DisplayInteractionState:
        normalized = None if point is None else (float(point[0]), float(point[1]))
        return self._replace(pointer=normalized)

    def set_hover(self, target: InteractionTarget | None, *, point: tuple[float, float] | None = None) -> DisplayInteractionState:
        normalized = None if point is None else (float(point[0]), float(point[1]))
        updates = {"hover": target}
        if point is not None:
            updates["pointer"] = normalized
        return self._replace(**updates)

    def clear_hover(self) -> DisplayInteractionState:
        return self._replace(hover=None)

    def arm_drawing(self, tool: str) -> DisplayInteractionState:
        tool = str(tool)
        if tool not in ONE_SHOT_DRAW_TOOLS:
            raise ValueError(f"one-shot drawing is not supported for tool: {tool}")
        return self._replace(
            phase=PointerPhase.DRAWING_ARMED,
            pending_draw_tool=tool,
            drawing_points=(),
            capture=None,
            hover=None,
        )

    def begin_drawing(self, point: tuple[float, float]) -> bool:
        if self._state.pending_draw_tool is None:
            return False
        normalized = (float(point[0]), float(point[1]))
        self._replace(
            phase=PointerPhase.DRAWING,
            pointer=normalized,
            drawing_points=(normalized,),
            hover=None,
        )
        return True

    def append_drawing_point(self, point: tuple[float, float], *, minimum_distance: float = 0.0) -> bool:
        if self._state.phase is not PointerPhase.DRAWING:
            return False
        normalized = (float(point[0]), float(point[1]))
        points = self._state.drawing_points
        if points and _point_distance(points[-1], normalized) < max(0.0, float(minimum_distance)):
            self._replace(pointer=normalized)
            return False
        self._replace(pointer=normalized, drawing_points=points + (normalized,))
        return True

    def finish_drawing(self) -> DrawingResult | None:
        if self._state.phase is not PointerPhase.DRAWING or self._state.pending_draw_tool is None:
            return None
        result = DrawingResult(self._state.pending_draw_tool, self._state.drawing_points)
        self._replace(
            phase=PointerPhase.IDLE,
            pending_draw_tool=None,
            drawing_points=(),
            capture=None,
            hover=None,
        )
        return result

    def cancel_drawing(self) -> DisplayInteractionState:
        return self._replace(
            phase=PointerPhase.IDLE,
            pending_draw_tool=None,
            drawing_points=(),
            capture=None,
            hover=None,
        )

    def begin_capture(self, target: InteractionTarget, point: tuple[float, float]) -> DisplayInteractionState:
        normalized = (float(point[0]), float(point[1]))
        return self._replace(
            phase=PointerPhase.DRAGGING,
            pointer=normalized,
            capture=target,
            hover=target,
        )

    def end_capture(self) -> InteractionTarget | None:
        captured = self._state.capture
        self._replace(phase=PointerPhase.IDLE, capture=None)
        return captured

    def _replace(self, **changes) -> DisplayInteractionState:
        candidate = replace(self._state, **changes)
        if candidate == self._state:
            return self._state
        self._state = replace(candidate, revision=self._state.revision + 1)
        return self._state


def hit_test_display_overlays(
    point: tuple[float, float],
    *,
    roi_selections: Iterable[object] = (),
    profile_position: tuple[float, float] | None = None,
    profile_bounds: tuple[float, float, float, float] | None = None,
    tolerance: float,
) -> InteractionTarget | None:
    """Return the highest-priority interactive overlay target."""

    x, y = (float(point[0]), float(point[1]))
    tolerance = max(0.0, float(tolerance))
    if profile_position is not None and profile_bounds is not None:
        px, py = (float(profile_position[0]), float(profile_position[1]))
        x0, y0, x1, y1 = (float(value) for value in profile_bounds)
        if abs(x - px) <= tolerance and abs(y - py) <= tolerance:
            return InteractionTarget("profile", part="center")
        if min(y0, y1) <= y <= max(y0, y1) and abs(x - px) <= tolerance:
            return InteractionTarget("profile", part="vertical")
        if min(x0, x1) <= x <= max(x0, x1) and abs(y - py) <= tolerance:
            return InteractionTarget("profile", part="horizontal")

    for selection in reversed(tuple(roi_selections)):
        geometry = getattr(selection, "geometry", None)
        if geometry is None:
            continue
        hit = hit_test_roi(geometry, (x, y), tolerance=tolerance)
        if hit is None:
            continue
        return _roi_target(selection, geometry, hit)
    return None


def _roi_target(selection, geometry, hit: RoiHit) -> InteractionTarget:
    kind = getattr(getattr(geometry, "kind", None), "value", getattr(geometry, "kind", None))
    return InteractionTarget(
        kind="roi",
        object_id=str(getattr(selection, "id", "")) or None,
        part=str(hit.part),
        geometry_kind=None if kind is None else str(kind),
        handle_index=hit.handle_index,
        segment_index=hit.segment_index,
    )


def _point_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


__all__ = [
    "ALLOWED_INSPECTION_TOOLS",
    "CursorIntent",
    "DisplayInteractionController",
    "DisplayInteractionState",
    "DrawingResult",
    "InteractionTarget",
    "PointerPhase",
    "hit_test_display_overlays",
]
