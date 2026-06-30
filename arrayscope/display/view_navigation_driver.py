"""Qt event driver for backend-neutral viewport navigation."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore

from arrayscope.display.interaction import point_inside_rect
from arrayscope.display.view_navigation import PanGesture, begin_pan, pan_view_range, wheel_zoom_view_range


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
        if event_type == QtCore.QEvent.Type.MouseButtonPress and event.button() in {
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.MiddleButton,
        }:
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
            if not bool(buttons & (QtCore.Qt.MouseButton.LeftButton | QtCore.Qt.MouseButton.MiddleButton)):
                self._pan = None
                return False
            position = owner._event_position(event)
            view_range = pan_view_range(self._pan, (position.x(), position.y()))
            owner.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)
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


def _release_viewport_continuity(owner) -> None:
    try:
        window = owner.window()
    except Exception:
        window = None
    release = getattr(window, "_release_viewport_continuity", None)
    if callable(release):
        release()


__all__ = ["QtViewNavigationDriver"]
