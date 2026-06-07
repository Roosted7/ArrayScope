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

    def note_user_range_changed(self):
        if self.mode not in (ViewportMode.FIT, ViewportMode.ONE_TO_ONE):
            self.mode = ViewportMode.USER

    def apply_after_image(self, view_box, image_shape, viewport_size, *, policy=ViewportPolicy.PRESERVE):
        image_shape = tuple(int(v) for v in image_shape[:2])
        previous_shape = self.last_display_shape
        self.last_display_shape = image_shape
        intent = _intent_from_policy(policy)

        if intent == ViewportIntent.FIT:
            self.mode = ViewportMode.FIT
            _fit(view_box)
            return
        if intent == ViewportIntent.ONE_TO_ONE:
            self.mode = ViewportMode.ONE_TO_ONE
            _set_one_to_one(view_box, image_shape, viewport_size)
            return

        shape_changed = previous_shape is None or previous_shape != image_shape
        if intent == ViewportIntent.RESET_FOR_NEW_SHAPE or previous_shape is None:
            if self.mode == ViewportMode.USER and previous_shape is not None:
                _preserve_center_for_shape(view_box, image_shape)
            else:
                self.mode = ViewportMode.AUTO_UNTOUCHED
                _fit(view_box)
            return

        if self.mode == ViewportMode.FIT and shape_changed:
            _fit(view_box)
        elif self.mode == ViewportMode.ONE_TO_ONE:
            _set_one_to_one(view_box, image_shape, viewport_size)

    def fit(self, view_box):
        self.mode = ViewportMode.FIT
        _fit(view_box)

    def one_to_one(self, view_box, image_shape, viewport_size):
        self.mode = ViewportMode.ONE_TO_ONE
        _set_one_to_one(view_box, image_shape, viewport_size)

    def resize(self, view_box, image_shape, viewport_size):
        if image_shape is None:
            return
        if self.mode == ViewportMode.FIT:
            _fit(view_box)
        elif self.mode == ViewportMode.ONE_TO_ONE:
            _set_one_to_one(view_box, tuple(int(v) for v in image_shape[:2]), viewport_size)


def _intent_from_policy(policy):
    if isinstance(policy, ViewportIntent):
        return policy
    if policy == ViewportPolicy.FIT_ONCE:
        return ViewportIntent.FIT
    if policy == ViewportPolicy.RESET_FOR_NEW_SHAPE:
        return ViewportIntent.RESET_FOR_NEW_SHAPE
    return ViewportIntent.PRESERVE


def _fit(view_box):
    view_box.autoRange(padding=0)


def _set_one_to_one(view_box, image_shape, viewport_size):
    height, width = image_shape
    view_range = view_box.viewRange()
    cx = (float(view_range[0][0]) + float(view_range[0][1])) * 0.5 if view_range else (width - 1) * 0.5
    cy = (float(view_range[1][0]) + float(view_range[1][1])) * 0.5 if view_range else (height - 1) * 0.5
    viewport_width = max(1.0, float(viewport_size.width()))
    viewport_height = max(1.0, float(viewport_size.height()))
    half_w = viewport_width * 0.5
    half_h = viewport_height * 0.5
    view_box.setRange(xRange=(cx - half_w, cx + half_w), yRange=(cy - half_h, cy + half_h), padding=0)


def _preserve_center_for_shape(view_box, image_shape):
    height, width = image_shape
    view_range = view_box.viewRange()
    x_span = float(view_range[0][1]) - float(view_range[0][0])
    y_span = float(view_range[1][1]) - float(view_range[1][0])
    cx = max(0.0, min(float(width - 1), (float(view_range[0][0]) + float(view_range[0][1])) * 0.5))
    cy = max(0.0, min(float(height - 1), (float(view_range[1][0]) + float(view_range[1][1])) * 0.5))
    view_box.setRange(xRange=(cx - x_span * 0.5, cx + x_span * 0.5), yRange=(cy - y_span * 0.5, cy + y_span * 0.5), padding=0)
