"""Viewport update policy for 2D image display."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class ViewportPolicy(Enum):
    PRESERVE = "preserve"
    FIT_ONCE = "fit_once"
    RESET_FOR_NEW_SHAPE = "reset_for_new_shape"


class ViewportMode(Enum):
    AUTO_UNTOUCHED = "auto_untouched"
    USER = "user"
    FIT = "fit"
    ONE_TO_ONE = "one_to_one"


class ViewportIntent(Enum):
    PRESERVE = "preserve"
    FIT = "fit"
    ONE_TO_ONE = "one_to_one"
    RESET_FOR_NEW_SHAPE = "reset_for_new_shape"


@dataclass
class ViewportController:
    mode: ViewportMode = ViewportMode.AUTO_UNTOUCHED
    last_display_shape: tuple[int, int] | None = None
    last_display_rect: tuple[float, float, float, float] | None = None

    def note_user_range_changed(self):
        if self.mode not in (ViewportMode.FIT, ViewportMode.ONE_TO_ONE):
            self.mode = ViewportMode.USER

    def apply_after_image(self, view_box, image_shape, viewport_size, *, policy=ViewportPolicy.PRESERVE, display_rect=None):
        image_shape = tuple(int(v) for v in image_shape[:2])
        display_rect = _display_rect(image_shape, display_rect)
        previous_shape = self.last_display_shape
        previous_rect = self.last_display_rect
        self.last_display_shape = image_shape
        self.last_display_rect = display_rect
        intent = _intent_from_policy(policy)

        if intent == ViewportIntent.FIT:
            self.mode = ViewportMode.FIT
            _fit(view_box, display_rect=display_rect)
            return
        if intent == ViewportIntent.ONE_TO_ONE:
            _set_one_to_one(view_box, image_shape, viewport_size, display_rect=display_rect)
            self.mode = ViewportMode.USER
            return

        rect_changed_only = previous_shape == image_shape and previous_rect != display_rect
        shape_changed = previous_shape is None or previous_shape != image_shape
        if intent == ViewportIntent.RESET_FOR_NEW_SHAPE or previous_shape is None:
            if self.mode == ViewportMode.USER and previous_shape is not None:
                _preserve_center_for_shape(view_box, image_shape, display_rect=display_rect)
            else:
                self.mode = ViewportMode.AUTO_UNTOUCHED
                _fit(view_box, display_rect=display_rect)
            return

        if rect_changed_only and intent == ViewportIntent.PRESERVE:
            return

        if self.mode == ViewportMode.FIT and shape_changed:
            _fit(view_box, display_rect=display_rect)

    def fit(self, view_box):
        self.mode = ViewportMode.FIT
        _fit(view_box, display_rect=self.last_display_rect)

    def set_fit_locked(self, view_box, enabled: bool):
        if enabled:
            self.fit(view_box)
        elif self.mode == ViewportMode.FIT:
            self.mode = ViewportMode.USER

    def is_fit_locked(self) -> bool:
        return self.mode == ViewportMode.FIT

    def one_to_one(self, view_box, image_shape, viewport_size, display_rect=None):
        _set_one_to_one(view_box, image_shape, viewport_size, display_rect=_display_rect(image_shape, display_rect))
        self.mode = ViewportMode.USER

    def resize(self, view_box, image_shape, viewport_size, display_rect=None):
        if image_shape is None:
            return
        display_rect = _display_rect(tuple(int(v) for v in image_shape[:2]), display_rect or self.last_display_rect)
        if self.mode == ViewportMode.FIT:
            _fit(view_box, display_rect=display_rect)


def _intent_from_policy(policy):
    if isinstance(policy, ViewportIntent):
        return policy
    if policy == ViewportPolicy.FIT_ONCE:
        return ViewportIntent.FIT
    if policy == ViewportPolicy.RESET_FOR_NEW_SHAPE:
        return ViewportIntent.RESET_FOR_NEW_SHAPE
    return ViewportIntent.PRESERVE


def _fit(view_box, *, display_rect=None):
    if display_rect is None:
        view_box.autoRange(padding=0)
        return
    x0, y0, x1, y1 = display_rect
    view_box.setRange(xRange=(float(x0), float(x1)), yRange=(float(y0), float(y1)), padding=0)


def _set_one_to_one(view_box, image_shape, viewport_size, *, display_rect=None):
    height, width = image_shape
    display_rect = _display_rect((height, width), display_rect)
    x0, y0, x1, y1 = display_rect
    view_range = view_box.viewRange()
    cx = (float(view_range[0][0]) + float(view_range[0][1])) * 0.5 if view_range else (x0 + x1) * 0.5
    cy = (float(view_range[1][0]) + float(view_range[1][1])) * 0.5 if view_range else (y0 + y1) * 0.5
    viewport_width = max(1.0, float(viewport_size.width()))
    viewport_height = max(1.0, float(viewport_size.height()))
    half_w = viewport_width * 0.5
    half_h = viewport_height * 0.5
    view_box.setRange(xRange=(cx - half_w, cx + half_w), yRange=(cy - half_h, cy + half_h), padding=0)


def _preserve_center_for_shape(view_box, image_shape, *, display_rect=None):
    height, width = image_shape
    x0, y0, x1, y1 = _display_rect((height, width), display_rect)
    view_range = view_box.viewRange()
    x_span = float(view_range[0][1]) - float(view_range[0][0])
    y_span = float(view_range[1][1]) - float(view_range[1][0])
    cx = max(float(x0), min(float(x1), (float(view_range[0][0]) + float(view_range[0][1])) * 0.5))
    cy = max(float(y0), min(float(y1), (float(view_range[1][0]) + float(view_range[1][1])) * 0.5))
    view_box.setRange(xRange=(cx - x_span * 0.5, cx + x_span * 0.5), yRange=(cy - y_span * 0.5, cy + y_span * 0.5), padding=0)


def _display_rect(image_shape, display_rect=None) -> tuple[float, float, float, float]:
    height, width = tuple(int(v) for v in image_shape[:2])
    if display_rect is None:
        return (0.0, 0.0, float(max(0, width - 1)), float(max(0, height - 1)))
    x0, y0, x1, y1 = display_rect
    return (float(x0), float(y0), float(x1), float(y1))
