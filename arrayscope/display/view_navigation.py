"""Backend-neutral view navigation range math."""

from __future__ import annotations

import math
from dataclasses import dataclass

ViewRange = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class PanGesture:
    start_pixel: tuple[float, float]
    start_range: ViewRange
    viewport_size: tuple[float, float]
    x_inverted: bool = False
    y_inverted: bool = True


def copy_view_range(view_range) -> ViewRange:
    return (
        (float(view_range[0][0]), float(view_range[0][1])),
        (float(view_range[1][0]), float(view_range[1][1])),
    )


def begin_pan(
    start_pixel, view_range, viewport_size, *, x_inverted: bool = False, y_inverted: bool = True
) -> PanGesture:
    return PanGesture(
        start_pixel=(float(start_pixel[0]), float(start_pixel[1])),
        start_range=copy_view_range(view_range),
        viewport_size=(max(1.0, float(viewport_size[0])), max(1.0, float(viewport_size[1]))),
        x_inverted=bool(x_inverted),
        y_inverted=bool(y_inverted),
    )


def pan_view_range(gesture: PanGesture, current_pixel) -> ViewRange:
    current_x, current_y = (float(current_pixel[0]), float(current_pixel[1]))
    start_x, start_y = gesture.start_pixel
    width, height = gesture.viewport_size
    x_range, y_range = gesture.start_range
    dx = current_x - start_x
    dy = current_y - start_y
    x_span = float(x_range[1]) - float(x_range[0])
    y_span = float(y_range[1]) - float(y_range[0])
    x_direction = 1.0 if gesture.x_inverted else -1.0
    y_direction = -1.0 if gesture.y_inverted else 1.0
    x_shift = x_direction * dx * x_span / width
    y_shift = y_direction * dy * y_span / height
    return (
        (float(x_range[0]) + x_shift, float(x_range[1]) + x_shift),
        (float(y_range[0]) + y_shift, float(y_range[1]) + y_shift),
    )


def scale_zoom_view_range(view_range, focus, scale: float) -> ViewRange:
    """Scale the view span about ``focus`` by ``scale`` (<1 zooms in)."""
    scale = float(scale)
    focus_x, focus_y = (float(focus[0]), float(focus[1]))
    x_range, y_range = copy_view_range(view_range)
    return (
        (focus_x + (x_range[0] - focus_x) * scale, focus_x + (x_range[1] - focus_x) * scale),
        (focus_y + (y_range[0] - focus_y) * scale, focus_y + (y_range[1] - focus_y) * scale),
    )


def drag_zoom_view_range(
    view_range, focus, pixel_delta: float, *, sensitivity: float = 0.005
) -> ViewRange:
    """Zoom smoothly from a vertical drag (<0 zooms in, >0 zooms out)."""

    return scale_zoom_view_range(
        view_range, focus, math.exp(float(pixel_delta) * float(sensitivity))
    )


def wheel_zoom_view_range(
    view_range, focus, wheel_steps: float, *, step_scale: float = 0.9
) -> ViewRange:
    scale = float(step_scale) ** float(wheel_steps)
    return scale_zoom_view_range(view_range, focus, scale)


# Clamp a single pinch increment's scale factor.  Native gestures arrive as a
# stream of tiny increments, so these bounds only guard pathological values.
PINCH_SCALE_MIN = 0.2
PINCH_SCALE_MAX = 5.0


def pinch_zoom_view_range(view_range, focus, gesture_value: float) -> ViewRange:
    """Zoom about ``focus`` from a native pinch increment.

    Qt reports magnification as an incremental fraction: fingers apart
    (``gesture_value > 0``) magnifies the content, so the world span shrinks
    by ``1 / (1 + gesture_value)``.
    """
    denominator = 1.0 + float(gesture_value)
    if denominator <= 0.0:
        scale = PINCH_SCALE_MAX
    else:
        scale = min(PINCH_SCALE_MAX, max(PINCH_SCALE_MIN, 1.0 / denominator))
    return scale_zoom_view_range(view_range, focus, scale)


def scroll_pan_view_range(
    view_range, pixel_delta, viewport_size, *, x_inverted: bool = False, y_inverted: bool = True
) -> ViewRange:
    """Pan the view from a two-finger scroll translation.

    The content follows the finger translation (direct manipulation), so this
    reuses the drag-pan math: scroll and mouse-drag then agree on direction and
    honour the same axis inversions.
    """
    gesture = begin_pan(
        (0.0, 0.0), view_range, viewport_size, x_inverted=x_inverted, y_inverted=y_inverted
    )
    return pan_view_range(gesture, (float(pixel_delta[0]), float(pixel_delta[1])))
