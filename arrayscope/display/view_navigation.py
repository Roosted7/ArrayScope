"""Backend-neutral view navigation range math."""

from __future__ import annotations

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


def wheel_zoom_view_range(
    view_range, focus, wheel_steps: float, *, step_scale: float = 0.9
) -> ViewRange:
    scale = float(step_scale) ** float(wheel_steps)
    focus_x, focus_y = (float(focus[0]), float(focus[1]))
    x_range, y_range = copy_view_range(view_range)
    return (
        (focus_x + (x_range[0] - focus_x) * scale, focus_x + (x_range[1] - focus_x) * scale),
        (focus_y + (y_range[0] - focus_y) * scale, focus_y + (y_range[1] - focus_y) * scale),
    )
