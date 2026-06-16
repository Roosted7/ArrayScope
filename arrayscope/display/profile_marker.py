"""Profile marker geometry helpers for ImageView2D."""

from __future__ import annotations


class ProfileMarkerOwner:
    """Small owner for profile-marker bounds calculations."""

    def __init__(self):
        self.bounds_rect = None

    def set_bounds_rect(self, rect) -> None:
        if rect is None:
            self.bounds_rect = None
            return
        x0, y0, x1, y1 = rect
        self.bounds_rect = (float(x0), float(y0), float(x1), float(y1))

    def current_bounds(self, image_shape) -> tuple[float, float, float, float]:
        height, width = image_shape[:2]
        x0, y0, x1, y1 = self.bounds_rect or (0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1)))
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        return (float(x0), float(y0), float(x1), float(y1))

    def clamp_point(self, image_shape, x, y) -> tuple[float, float]:
        x0, y0, x1, y1 = self.current_bounds(image_shape)
        return (max(x0, min(float(x), x1)), max(y0, min(float(y), y1)))
