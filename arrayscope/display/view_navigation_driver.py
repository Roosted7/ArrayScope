"""Qt event driver for backend-neutral viewport navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyqtgraph.Qt import QtCore, QtGui

from arrayscope.display.interaction import point_inside_rect
from arrayscope.display.view_navigation import (
    PanGesture,
    begin_pan,
    copy_view_range,
    drag_zoom_view_range,
    pan_view_range,
    pinch_zoom_view_range,
    scroll_pan_view_range,
    wheel_zoom_view_range,
)

# Compatibility calibration for angleDelta(). Some Wayland touchpads report
# both a very small pixel delta and an accelerated angle delta; preferring the
# calibrated angle per axis keeps that acceleration and the manually tuned feel.
# Pixel deltas remain the fallback for sub-step and pixel-only motion.
_ANGLE_SCROLL_PIXELS_PER_STEP = 240.0
_PIXEL_SCROLL_GAIN_LIMIT = 2.0
_MIDDLE_DRAG_DIRECTION_THRESHOLD_PX = 6.0
_MIDDLE_DRAG_INDEX_STEP_PX = 10.0


@dataclass
class _MiddleDrag:
    start_pixel: tuple[float, float]
    start_range: tuple[tuple[float, float], tuple[float, float]]
    focus: tuple[float, float]
    target_axis: int | None
    mode: str | None = None
    applied_steps: int = 0
    provisional_zoom_applied: bool = False


class QtViewNavigationDriver:
    """Translate plain Qt pan/zoom gestures into canonical ViewBox ranges."""

    def __init__(self, owner):
        self._owner = owner
        self._pan: PanGesture | None = None

    def is_active(self) -> bool:
        return self._pan is not None

    def handle_event(self, event) -> bool:
        owner = self._owner
        if getattr(owner, "image", None) is None or owner.viewport_controller.is_fit_locked():
            self._pan = None
            return False
        event_type = event.type()
        if (
            event_type == QtCore.QEvent.Type.MouseButtonPress
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            point = owner._event_overlay_point(event)
            if point is None:
                return False
            if not point_inside_rect(point, owner._current_image_viewport_rect()):
                return False
            target_at = getattr(owner, "_interaction_target_at", None)
            if callable(target_at) and target_at(point) is not None:
                return False
            position = owner._event_position(event)
            viewport = owner.graphicsView.viewport()
            view_state = getattr(owner.view, "state", {}) or {}
            self._pan = begin_pan(
                (position.x(), position.y()),
                owner.view.viewRange(),
                (viewport.width(), viewport.height()),
                x_inverted=bool(view_state.get("xInverted", False)),
                y_inverted=bool(view_state.get("yInverted", True)),
            )
            _release_viewport_continuity(owner)
            event.accept()
            return True
        if event_type == QtCore.QEvent.Type.MouseMove and self._pan is not None:
            buttons = event.buttons()
            if not bool(buttons & QtCore.Qt.MouseButton.LeftButton):
                self._pan = None
                return False
            position = owner._event_position(event)
            view_range = pan_view_range(self._pan, (position.x(), position.y()))
            owner.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
            # The pan consumes this MouseMove (accept + return True), so
            # pyqtgraph's GraphicsScene never emits sigMouseMoved and the HUD
            # hover readout would freeze at pan-start.  Re-emit the hover at the
            # moved position, exactly like the ROI/profile pointer driver does
            # via the same owner hook (QtPointerInteractionDriver).
            notify_drag = getattr(owner, "_notify_pointer_drag_moved", None)
            if callable(notify_drag):
                notify_drag(event)
            event.accept()
            return True
        if event_type == QtCore.QEvent.Type.MouseButtonRelease:
            if self._pan is None:
                return False
            self._pan = None
            event.accept()
            return True
        if event_type == QtCore.QEvent.Type.Wheel:
            self._pan = None
            if not self._apply_wheel_zoom(event):
                return False
            event.accept()
            return True
        return False

    def _apply_wheel_zoom(self, event) -> bool:
        angle_delta = event.angleDelta()
        delta = float(angle_delta.y() if not angle_delta.isNull() else 0.0)
        if delta == 0.0:
            return False
        focus = self._owner._event_display_point(event)
        if focus is None:
            return False
        view_range = wheel_zoom_view_range(self._owner.view.viewRange(), focus, delta / 120.0)
        _release_viewport_continuity(self._owner)
        self._owner.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        return True


class QtGestureNavigationDriver:
    """Translate native touchpad input into canonical ViewBox ranges.

    Native pinch zooms about the gesture point. Two-finger scroll pans by
    direct manipulation, or zooms about the cursor while Ctrl is held. Plain
    mouse-wheel events remain owned by the backend's existing wheel path.

    Qt's incremental deltas already contain platform acceleration and momentum.
    This driver applies each update exactly once and keeps no competing velocity
    or animation state.
    """

    def __init__(self, owner):
        self._owner = owner
        self._middle_drag: _MiddleDrag | None = None

    def handle_event(self, event) -> bool:
        owner = self._owner
        if getattr(owner, "image", None) is None:
            return False
        event_type = event.type()
        if event_type in {
            QtCore.QEvent.Type.MouseButtonPress,
            QtCore.QEvent.Type.MouseMove,
            QtCore.QEvent.Type.MouseButtonRelease,
        }:
            handled = self._handle_middle_drag(event)
        elif owner.viewport_controller.is_fit_locked():
            return False
        elif event_type == QtCore.QEvent.Type.NativeGesture:
            handled = self._handle_native_gesture(event)
        elif event_type == QtCore.QEvent.Type.Wheel:
            handled = self._handle_scroll(event)
        else:
            return False
        if handled:
            event.accept()
        return handled

    def is_middle_drag_active(self) -> bool:
        return self._middle_drag is not None

    def cancel_middle_drag(self) -> None:
        if self._middle_drag is not None:
            self._finish_middle_drag(commit_provisional=False)

    def _handle_middle_drag(self, event) -> bool:
        event_type = event.type()
        middle = QtCore.Qt.MouseButton.MiddleButton
        if event_type == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() != middle:
                return False
            position = self._owner._event_position(event)
            focus = self._owner._event_display_point(event)
            if focus is None:
                return False
            self._middle_drag = _MiddleDrag(
                start_pixel=(float(position.x()), float(position.y())),
                start_range=copy_view_range(self._owner.view.viewRange()),
                focus=(float(focus[0]), float(focus[1])),
                target_axis=self._owner._middle_drag_target_axis(),
            )
            return True

        gesture = self._middle_drag
        if gesture is None:
            return False
        if event_type == QtCore.QEvent.Type.MouseButtonRelease:
            if event.button() != middle:
                return False
            self._finish_middle_drag()
            return True
        if event_type != QtCore.QEvent.Type.MouseMove:
            return False
        if not bool(event.buttons() & middle):
            self._finish_middle_drag()
            return False

        position = self._owner._event_position(event)
        dx = float(position.x()) - gesture.start_pixel[0]
        dy = float(position.y()) - gesture.start_pixel[1]
        if gesture.mode is None:
            if dy != 0.0 and not self._owner.viewport_controller.is_fit_locked():
                provisional_range = drag_zoom_view_range(gesture.start_range, gesture.focus, dy)
                self._set_middle_range(provisional_range, provisional=True)
                gesture.provisional_zoom_applied = True
            if math.hypot(dx, dy) < _MIDDLE_DRAG_DIRECTION_THRESHOLD_PX:
                return True
            gesture.mode = "index" if abs(dx) > abs(dy) else "zoom"
            if gesture.mode == "index":
                if gesture.provisional_zoom_applied:
                    self._set_middle_range(gesture.start_range, provisional=True)
                    gesture.provisional_zoom_applied = False
                self._owner._set_middle_drag_dimension_active(True)
            elif not self._owner.viewport_controller.is_fit_locked():
                gesture.provisional_zoom_applied = False
                _release_viewport_continuity(self._owner)

        if gesture.mode == "zoom":
            if self._owner.viewport_controller.is_fit_locked():
                self._owner._show_fit_mode_interaction_reminder()
                return True
            view_range = drag_zoom_view_range(gesture.start_range, gesture.focus, dy)
            self._set_middle_range(view_range)
            return True

        if gesture.target_axis is None:
            return True
        desired_steps = math.trunc(dx / _MIDDLE_DRAG_INDEX_STEP_PX)
        step_delta = desired_steps - gesture.applied_steps
        if step_delta:
            gesture.applied_steps = desired_steps
            self._owner._step_middle_drag_dimension(gesture.target_axis, step_delta)
        return True

    def _set_middle_range(self, view_range, *, provisional: bool = False) -> None:
        owner = self._owner
        applying = owner._viewport_applying
        if provisional:
            owner._viewport_applying = True
        try:
            owner.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
        finally:
            owner._viewport_applying = applying

    def _finish_middle_drag(self, *, commit_provisional: bool = True) -> None:
        gesture = self._middle_drag
        if gesture is not None and gesture.provisional_zoom_applied:
            if commit_provisional:
                _release_viewport_continuity(self._owner)
                current_range = copy_view_range(self._owner.view.viewRange())
                self._owner.viewport_controller.note_user_range_changed(current_range)
                self._owner._enforce_viewport_constraints()
            else:
                self._set_middle_range(gesture.start_range, provisional=True)
        self._owner._set_middle_drag_dimension_active(False)
        self._middle_drag = None

    def _handle_native_gesture(self, event) -> bool:
        gesture_type = event.gestureType()
        if gesture_type == QtCore.Qt.NativeGestureType.ZoomNativeGesture:
            value = float(event.value())
            if value == 0.0:
                return False
            focus = self._owner._event_display_point(event)
            if focus is None:
                return False
            view_range = pinch_zoom_view_range(self._owner.view.viewRange(), focus, value)
        elif gesture_type == QtCore.Qt.NativeGestureType.PanNativeGesture:
            delta = event.delta()
            if delta.isNull():
                return False
            view_range = self._pan_range((float(delta.x()), float(delta.y())))
        else:
            return False
        self._commit_range(view_range)
        return True

    def _handle_scroll(self, event) -> bool:
        if not _is_touchpad_scroll(event):
            return False
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            return self._zoom_scroll(event)
        delta = _scroll_pixels(event)
        if delta == (0.0, 0.0):
            return False
        self._commit_range(self._pan_range(delta))
        return True

    def _pan_range(self, delta) -> tuple[tuple[float, float], tuple[float, float]]:
        owner = self._owner
        viewport = owner.graphicsView.viewport()
        view_state = getattr(owner.view, "state", {}) or {}
        return scroll_pan_view_range(
            owner.view.viewRange(),
            delta,
            (viewport.width(), viewport.height()),
            x_inverted=bool(view_state.get("xInverted", False)),
            y_inverted=bool(view_state.get("yInverted", True)),
        )

    def _zoom_scroll(self, event) -> bool:
        steps = _scroll_steps(event)
        if steps == 0.0:
            return False
        focus = self._owner._event_display_point(event)
        if focus is None:
            return False
        view_range = wheel_zoom_view_range(self._owner.view.viewRange(), focus, steps)
        self._commit_range(view_range)
        return True

    def _commit_range(self, view_range) -> None:
        owner = self._owner
        _release_viewport_continuity(owner)
        owner.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)


def _is_touchpad_scroll(event) -> bool:
    pixel_delta = getattr(event, "pixelDelta", None)
    if not callable(pixel_delta):
        # Minimal wheel-event adapters belong to the established backend path.
        return False
    device_getter = getattr(event, "device", None)
    device = device_getter() if callable(device_getter) else None
    angle_delta = event.angleDelta()
    nonzero_angles = [value for value in (angle_delta.x(), angle_delta.y()) if value]
    if nonzero_angles and all(value % 120 == 0 for value in nonzero_angles):
        # Classic wheels report 15-degree notches (120 eighth-degrees). Some
        # Wayland stacks attach the touchpad device identity to those events,
        # so discrete-wheel semantics must win over the reported device type.
        return False
    # Neither pixelDelta nor ScrollPhase uniquely identifies a touchpad on
    # Wayland: high-resolution mice may provide both. Preserve the incumbent
    # mouse-wheel zoom unless Qt identifies a TouchPad with fine-grained input.
    return device is not None and device.type() == QtGui.QInputDevice.DeviceType.TouchPad


def _scroll_pixels(event) -> tuple[float, float]:
    pixel_delta = event.pixelDelta()
    angle_delta = event.angleDelta()

    def axis_pixels(pixel: int, angle: int) -> float:
        pixel_value = float(pixel)
        if not angle:
            return pixel_value
        calibrated = float(angle) / 120.0 * _ANGLE_SCROLL_PIXELS_PER_STEP
        if pixel_value == 0.0 or calibrated * pixel_value <= 0.0:
            return calibrated
        # The two Qt representations can differ sharply on Wayland. Keep the
        # native pixel motion as the direct-manipulation baseline, then admit
        # accelerated angle motion only up to a bounded gain.
        magnitude = min(abs(calibrated), abs(pixel_value) * _PIXEL_SCROLL_GAIN_LIMIT)
        return magnitude if calibrated > 0.0 else -magnitude

    return (
        axis_pixels(pixel_delta.x(), angle_delta.x()),
        axis_pixels(pixel_delta.y(), angle_delta.y()),
    )


def _scroll_steps(event) -> float:
    angle_delta = event.angleDelta()
    angle = angle_delta.y() if angle_delta.y() else angle_delta.x()
    if angle:
        return float(angle) / 120.0
    pixel_delta = event.pixelDelta()
    pixels = pixel_delta.y() if pixel_delta.y() else pixel_delta.x()
    return float(pixels) / _ANGLE_SCROLL_PIXELS_PER_STEP


def _release_viewport_continuity(owner) -> None:
    try:
        window = owner.window()
    except Exception:
        window = None
    release = getattr(window, "_release_viewport_continuity", None)
    if callable(release):
        release()


__all__ = ["QtGestureNavigationDriver", "QtViewNavigationDriver"]
