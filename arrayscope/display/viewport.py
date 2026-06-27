"""Viewport update policy for 2D image display."""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
import math


MIN_VIEWPORT_CONTENT_FRACTION = 0.05


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


def coerce_viewport_policy(
    policy: ViewportPolicy | ViewportIntent | str,
    auto_range: bool | None = None,
) -> ViewportPolicy | ViewportIntent:
    """Normalize legacy ``autoRange`` and semantic viewport requests once.

    Rendering backends must not maintain independent translations of the
    legacy boolean.  ``True`` means a one-time fit request; persistent Fit is
    represented explicitly by :class:`ViewportIntent.FIT`/controller state.
    """

    if auto_range is not None:
        return ViewportPolicy.FIT_ONCE if bool(auto_range) else ViewportPolicy.PRESERVE
    if isinstance(policy, (ViewportPolicy, ViewportIntent)):
        return policy
    value = str(policy)
    try:
        return ViewportPolicy(value)
    except ValueError:
        return ViewportIntent(value)


@dataclass
class ViewportController:
    mode: ViewportMode = ViewportMode.AUTO_UNTOUCHED
    last_display_shape: tuple[int, int] | None = None
    last_display_rect: tuple[float, float, float, float] | None = None
    last_viewport_size: tuple[float, float] | None = None
    last_auto_view_range: tuple[tuple[float, float], tuple[float, float]] | None = None

    def note_user_range_changed(self, view_range=None):
        if self.mode not in (ViewportMode.FIT, ViewportMode.ONE_TO_ONE):
            if view_range is not None and self.mode == ViewportMode.AUTO_UNTOUCHED and self.is_near_auto(view_range):
                return
            self.mode = ViewportMode.USER

    def apply_after_image(self, view_box, image_shape, viewport_size, *, policy=ViewportPolicy.PRESERVE, display_rect=None):
        image_shape = tuple(int(v) for v in image_shape[:2])
        display_rect = _display_rect(image_shape, display_rect)
        previous_shape = self.last_display_shape
        previous_rect = self.last_display_rect
        self.last_display_shape = image_shape
        self.last_display_rect = display_rect
        self.last_viewport_size = _viewport_size_tuple(viewport_size)
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
            elif self.mode == ViewportMode.FIT:
                # Fit is a persistent interaction mode, not a one-shot range.
                # Do not silently demote it when a slice/montage changes shape.
                _fit(view_box, display_rect=display_rect)
            else:
                self.mode = ViewportMode.AUTO_UNTOUCHED
                self._auto_square_fit(view_box, viewport_size, display_rect=display_rect)
            return

        if rect_changed_only and intent == ViewportIntent.PRESERVE:
            if self.mode == ViewportMode.FIT:
                _fit(view_box, display_rect=display_rect)
            elif self.mode == ViewportMode.AUTO_UNTOUCHED:
                self._auto_square_fit(view_box, viewport_size, display_rect=display_rect)
            return

        if self.mode == ViewportMode.FIT and shape_changed:
            _fit(view_box, display_rect=display_rect)
        elif self.mode == ViewportMode.AUTO_UNTOUCHED and shape_changed:
            self._auto_square_fit(view_box, viewport_size, display_rect=display_rect)

    def fit(self, view_box):
        self.mode = ViewportMode.FIT
        _fit(view_box, display_rect=self.last_display_rect)

    def set_fit_locked(self, view_box, enabled: bool, *, image_shape=None, viewport_size=None, display_rect=None):
        if enabled:
            self.fit(view_box)
        elif self.mode == ViewportMode.FIT:
            if viewport_size is None:
                self.mode = ViewportMode.USER
                return
            shape = (1, 1) if image_shape is None else tuple(int(value) for value in image_shape[:2])
            rect = _display_rect(shape, display_rect or self.last_display_rect)
            self.mode = ViewportMode.AUTO_UNTOUCHED
            self._auto_square_fit(view_box, viewport_size, display_rect=rect)

    def is_fit_locked(self) -> bool:
        return self.mode == ViewportMode.FIT

    def is_auto_active(self) -> bool:
        return self.mode == ViewportMode.AUTO_UNTOUCHED

    def promote_near_auto(self, view_range=None) -> bool:
        if self.mode == ViewportMode.FIT:
            return False
        if self.is_near_auto(view_range):
            self.mode = ViewportMode.AUTO_UNTOUCHED
            return True
        return self.is_auto_active()

    def one_to_one(self, view_box, image_shape, viewport_size, display_rect=None):
        _set_one_to_one(view_box, image_shape, viewport_size, display_rect=_display_rect(image_shape, display_rect))
        self.mode = ViewportMode.USER

    def resize(
        self,
        view_box,
        image_shape,
        viewport_size,
        display_rect=None,
        *,
        previous_viewport_size=None,
        base_view_range=None,
    ):
        if image_shape is None:
            return None
        previous_viewport_size = (
            self.last_viewport_size
            if previous_viewport_size is None
            else _viewport_size_tuple(previous_viewport_size)
        )
        self.last_viewport_size = _viewport_size_tuple(viewport_size)
        display_rect = _display_rect(tuple(int(v) for v in image_shape[:2]), display_rect or self.last_display_rect)
        if base_view_range is None:
            base_view_range = view_box.viewRange()
        if self.mode == ViewportMode.FIT:
            _fit(view_box, display_rect=display_rect)
            return _normalize_view_range(view_box.viewRange())
        elif self.mode == ViewportMode.AUTO_UNTOUCHED and self._current_view_is_near_auto(view_box, base_view_range=base_view_range):
            self._auto_square_fit(view_box, viewport_size, display_rect=display_rect)
            return _normalize_view_range(view_box.viewRange())
        elif self.mode == ViewportMode.AUTO_UNTOUCHED:
            self.mode = ViewportMode.USER
            return _preserve_screen_zoom_for_resize(
                view_box,
                previous_viewport_size,
                self.last_viewport_size,
                base_view_range=base_view_range,
            )
        elif self.mode == ViewportMode.USER:
            return _preserve_screen_zoom_for_resize(
                view_box,
                previous_viewport_size,
                self.last_viewport_size,
                base_view_range=base_view_range,
            )
        return None

    def is_near_auto(self, view_range=None, *, tolerance_fraction: float = 0.04) -> bool:
        target = self.last_auto_view_range
        if target is None:
            return self.mode == ViewportMode.AUTO_UNTOUCHED
        try:
            current = target if view_range is None else _normalize_view_range(view_range)
        except Exception:
            return False
        return view_ranges_near(current, target, tolerance_fraction=tolerance_fraction)

    def _auto_square_fit(self, view_box, viewport_size, *, display_rect=None):
        view_range = square_pixel_fit_view_range(display_rect, viewport_size)
        view_box.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        self.last_auto_view_range = view_range

    def _current_view_is_near_auto(self, view_box, *, base_view_range=None) -> bool:
        if self.last_auto_view_range is None:
            return True
        try:
            return self.is_near_auto(view_box.viewRange() if base_view_range is None else base_view_range)
        except Exception:
            return False


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
    view_range = _display_rect_view_range(display_rect)
    view_box.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)


def _display_rect_view_range(display_rect) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, y0, x1, y1 = _display_rect((1, 1), display_rect)
    return ((float(x0), float(x1)), (float(y0), float(y1)))


def square_pixel_fit_view_range(display_rect, viewport_size) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a square-pixel view range that contains ``display_rect``."""

    x0, y0, x1, y1 = _display_rect((1, 1), display_rect)
    width = max(1e-9, abs(float(x1) - float(x0)))
    height = max(1e-9, abs(float(y1) - float(y0)))
    viewport_width = max(1.0, float(viewport_size.width()))
    viewport_height = max(1.0, float(viewport_size.height()))
    viewport_aspect = viewport_width / viewport_height
    content_aspect = width / height
    fitted_width = width
    fitted_height = height
    if viewport_aspect > content_aspect:
        fitted_width = height * viewport_aspect
    elif viewport_aspect < content_aspect:
        fitted_height = width / viewport_aspect
    cx = (float(x0) + float(x1)) * 0.5
    cy = (float(y0) + float(y1)) * 0.5
    return (
        (cx - fitted_width * 0.5, cx + fitted_width * 0.5),
        (cy - fitted_height * 0.5, cy + fitted_height * 0.5),
    )


def view_ranges_near(first, second, *, tolerance_fraction: float = 0.04) -> bool:
    first = _normalize_view_range(first)
    second = _normalize_view_range(second)
    tolerance_fraction = max(0.0, float(tolerance_fraction))
    for axis in (0, 1):
        span = max(
            abs(float(first[axis][1]) - float(first[axis][0])),
            abs(float(second[axis][1]) - float(second[axis][0])),
            1.0,
        )
        tolerance = span * tolerance_fraction
        if abs(float(first[axis][0]) - float(second[axis][0])) > tolerance:
            return False
        if abs(float(first[axis][1]) - float(second[axis][1])) > tolerance:
            return False
    return True


def _normalize_view_range(view_range) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(view_range[0][0]), float(view_range[0][1])),
        (float(view_range[1][0]), float(view_range[1][1])),
    )


def _viewport_size_tuple(viewport_size) -> tuple[float, float]:
    if isinstance(viewport_size, (tuple, list)) and len(viewport_size) >= 2:
        return (max(1.0, float(viewport_size[0])), max(1.0, float(viewport_size[1])))
    return (max(1.0, float(viewport_size.width())), max(1.0, float(viewport_size.height())))


def constrain_view_range(
    view_range,
    content_rect,
    *,
    previous_view_range=None,
    min_visible_fraction: float = MIN_VIEWPORT_CONTENT_FRACTION,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a view range constrained to keep content recoverable.

    The constraint is intentionally simple and per-axis: the content must span
    at least ``min_visible_fraction`` of the viewport after zoom-out, and at
    least that fraction of the smaller of content/view spans must overlap after
    panning.
    """

    x_range, y_range = view_range
    x0, y0, x1, y1 = content_rect
    fraction = max(1e-9, min(1.0, float(min_visible_fraction)))
    return (
        _constrain_axis_range(
            x_range,
            (x0, x1),
            fraction,
            previous_axis_range=None if previous_view_range is None else previous_view_range[0],
        ),
        _constrain_axis_range(
            y_range,
            (y0, y1),
            fraction,
            previous_axis_range=None if previous_view_range is None else previous_view_range[1],
        ),
    )


def _constrain_axis_range(
    view_axis_range,
    content_axis_range,
    min_visible_fraction: float,
    *,
    previous_axis_range=None,
) -> tuple[float, float]:
    start = float(view_axis_range[0])
    end = float(view_axis_range[1])
    content_start, content_end = sorted((float(content_axis_range[0]), float(content_axis_range[1])))
    content_span = content_end - content_start
    if content_span <= 0.0 or not _finite_values(start, end, content_start, content_end):
        return (start, end)

    direction = 1.0 if end >= start else -1.0
    view_start, view_end = sorted((start, end))
    span = view_end - view_start
    if span <= 0.0:
        span = content_span * min_visible_fraction
        center = (view_start + view_end) * 0.5
        view_start = center - span * 0.5
        view_end = center + span * 0.5

    max_span = content_span / min_visible_fraction
    if span > max_span:
        span = max_span
        center = _range_center(previous_axis_range)
        if center is None:
            center = (view_start + view_end) * 0.5
        view_start = center - span * 0.5
        view_end = center + span * 0.5

    required_overlap = min_visible_fraction * min(content_span, span)
    min_view_start = content_start + required_overlap - span
    max_view_start = content_end - required_overlap
    constrained_start = min(max(view_start, min_view_start), max_view_start)
    constrained_end = constrained_start + span

    if direction < 0:
        return (constrained_end, constrained_start)
    return (constrained_start, constrained_end)


def _finite_values(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _range_center(axis_range) -> float | None:
    if axis_range is None:
        return None
    try:
        start = float(axis_range[0])
        end = float(axis_range[1])
    except Exception:
        return None
    if not _finite_values(start, end):
        return None
    return (start + end) * 0.5


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


def screen_zoom_resize_view_range(view_range, previous_viewport_size, current_viewport_size):
    if previous_viewport_size is None or current_viewport_size is None:
        return None
    previous_width, previous_height = previous_viewport_size
    current_width, current_height = current_viewport_size
    if previous_width <= 0.0 or previous_height <= 0.0 or current_width <= 0.0 or current_height <= 0.0:
        return None
    try:
        view_range = _normalize_view_range(view_range)
    except Exception:
        return None
    x_span = float(view_range[0][1]) - float(view_range[0][0])
    y_span = float(view_range[1][1]) - float(view_range[1][0])
    if not _finite_values(x_span, y_span):
        return None
    cx = (float(view_range[0][0]) + float(view_range[0][1])) * 0.5
    cy = (float(view_range[1][0]) + float(view_range[1][1])) * 0.5
    next_x_span = x_span * (float(current_width) / float(previous_width))
    next_y_span = y_span * (float(current_height) / float(previous_height))
    return (
        (cx - next_x_span * 0.5, cx + next_x_span * 0.5),
        (cy - next_y_span * 0.5, cy + next_y_span * 0.5),
    )


def _preserve_screen_zoom_for_resize(view_box, previous_viewport_size, current_viewport_size, *, base_view_range=None):
    view_range = screen_zoom_resize_view_range(
        view_box.viewRange() if base_view_range is None else base_view_range,
        previous_viewport_size,
        current_viewport_size,
    )
    if view_range is None:
        return None
    view_box.setRange(
        xRange=view_range[0],
        yRange=view_range[1],
        padding=0,
    )
    return view_range


def _display_rect(image_shape, display_rect=None) -> tuple[float, float, float, float]:
    height, width = tuple(int(v) for v in image_shape[:2])
    if display_rect is None:
        return (0.0, 0.0, float(max(1, width)), float(max(1, height)))
    x0, y0, x1, y1 = display_rect
    return (float(x0), float(y0), float(x1), float(y1))
