"""Qt pointer-event driver for shared display interaction state."""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.display.interaction import (
    DisplayInteractionController,
    InteractionTarget,
    PointerPhase,
    hit_test_display_overlays,
)


class QtPointerInteractionDriver:
    """Translate Qt pointer events into backend-neutral interaction state."""

    def __init__(self, owner, controller: DisplayInteractionController):
        self._owner = owner
        self._controller = controller

    def handle_event(self, event) -> bool:
        event_type = event.type()
        state = self._controller.state
        if event_type == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self._owner._event_overlay_point(event)
            target = self.target_at(point)
            if point is None or target is None:
                return False
            if not self._owner._begin_pointer_capture(target, point):
                return False
            event.accept()
            return True
        if event_type == QtCore.QEvent.Type.MouseMove and state.phase is PointerPhase.DRAGGING:
            if not self._left_button_is_down(event):
                self.cancel("button-lost")
                event.accept()
                return True
            point = self._owner._event_image_point(event)
            if point is not None:
                result = self._controller.update_capture(point)
                self._owner._apply_drag_result(result)
                self._owner.sync_interaction_state(self._controller.state)
            event.accept()
            return True
        if event_type == QtCore.QEvent.Type.MouseButtonRelease and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if state.phase is not PointerPhase.DRAGGING:
                return False
            result = self._controller.end_capture()
            self._owner._apply_drag_result(result)
            self._owner.sync_interaction_state(self._controller.state)
            event.accept()
            return True
        if event_type == QtCore.QEvent.Type.MouseMove:
            point = self._owner._event_overlay_point(event)
            target = self.target_at(point)
            self._owner.sync_interaction_state(self._controller.set_hover(target, point=point))
            return False
        if event_type == QtCore.QEvent.Type.Leave:
            if state.phase is PointerPhase.DRAGGING:
                if not self._left_button_is_down(event):
                    self.cancel("button-lost")
                return False
            self._owner.sync_interaction_state(self._controller.clear_hover())
            return False
        return False

    def target_at(self, point: tuple[float, float] | None) -> InteractionTarget | None:
        if point is None:
            return None
        state = self._controller.state
        if state.pending_draw_tool is not None or state.phase is PointerPhase.DRAWING:
            return None
        profile_position = self._owner.profileMarkerPosition()
        profile_bounds = self._owner._current_profile_bounds() if profile_position is not None else None
        tolerance = self._hit_tolerance()
        roi_candidates = self._owner.roiHitCandidates(point, tolerance=tolerance)
        return hit_test_display_overlays(
            point,
            roi_selections=roi_candidates,
            roi_selections_topmost=True,
            profile_position=profile_position,
            profile_bounds=profile_bounds,
            tolerance=tolerance,
        )

    def cancel(self, reason: str) -> None:
        state = self._controller.cancel_active(reason)
        self._owner._set_roi_drawing_preview(None, ())
        self._owner.sync_interaction_state(state)

    def cancel_active_capture_for_frame_replacement(self) -> None:
        if self._controller.state.phase is not PointerPhase.IDLE:
            self.cancel("frame-replacement")

    def _hit_tolerance(self) -> float:
        try:
            x_range, y_range = self._owner.view.viewRange()
            viewport = self._owner.graphicsView.viewport()
            x_per_pixel = abs(float(x_range[1]) - float(x_range[0])) / max(1, int(viewport.width()))
            y_per_pixel = abs(float(y_range[1]) - float(y_range[0])) / max(1, int(viewport.height()))
            return max(x_per_pixel, y_per_pixel) * 8.0
        except Exception:
            return 2.0

    @staticmethod
    def _left_button_is_down(event) -> bool:
        try:
            buttons = event.buttons()
        except Exception:
            buttons = QtWidgets.QApplication.mouseButtons()
        return bool(buttons & QtCore.Qt.MouseButton.LeftButton)


__all__ = ["QtPointerInteractionDriver"]
