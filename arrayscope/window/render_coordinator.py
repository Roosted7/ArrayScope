"""Render request coalescing for high-frequency UI interaction."""

from __future__ import annotations

from dataclasses import dataclass

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph.Qt as Qt


@dataclass(frozen=True)
class RenderRequest:
    reason: str
    force_autolevel: bool = False
    interactive: bool = False
    target_key: object = None


class RenderCoordinator(Qt.QtCore.QObject):
    def __init__(
        self,
        window,
        *,
        interactive_interval_ms: int = 32,
        quiet_interval_ms: int = 80,
        busy_retry_ms: int = 40,
    ):
        super().__init__(window)
        self._window = window
        self._interactive_interval_ms = max(0, int(interactive_interval_ms))
        self._quiet_interval_ms = max(1, int(quiet_interval_ms))
        self._busy_retry_ms = max(1, int(busy_retry_ms))
        self._pending_request: RenderRequest | None = None
        self._pending_side_work_cancel = False
        self._interactive_active = False
        self.requested = 0
        self.flushed = 0
        self.coalesced = 0
        self.deferred_side_panel_refreshes = 0
        self.immediate_cache_flushes = 0
        self.presentation_backpressure_skips = 0
        self._connected_presentation_view = None

        # Timer category: UI cosmetic. Qt event-turn barrier / bounded coalescer. This coalesces bursts of
        # non-cached interactive work without making cache hits wait.
        self._render_timer = Qt.QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._flush_timer)

        # Timer category: UI cosmetic. User-interaction quiet detector. It only gates side-panel refreshes;
        # render semantics are guarded by the latest pending request and the
        # render-generation checks in the window.
        self._quiet_timer = Qt.QtCore.QTimer(self)
        self._quiet_timer.setSingleShot(True)
        self._quiet_timer.timeout.connect(self._quiet_timer_elapsed)
        self._connect_presentation_draw_signal()

    @property
    def interactive_active(self) -> bool:
        return bool(self._interactive_active)

    @property
    def has_pending_render(self) -> bool:
        return self._pending_request is not None

    def has_equivalent_pending(
        self,
        *,
        reason: str,
        force_autolevel: bool = False,
        interactive: bool = False,
        target_key=None,
    ) -> bool:
        return self._pending_request == RenderRequest(
            reason=str(reason),
            force_autolevel=bool(force_autolevel),
            interactive=bool(interactive),
            target_key=target_key,
        )

    def request(
        self,
        *,
        reason: str,
        force_autolevel: bool = False,
        interactive: bool = False,
        target_key=None,
    ) -> bool:
        if getattr(self._window, "_closing", False):
            return False
        self._connect_presentation_draw_signal()
        request = RenderRequest(
            reason=str(reason),
            force_autolevel=bool(force_autolevel),
            interactive=bool(interactive),
            target_key=target_key,
        )
        if self._pending_request == request:
            return False
        self.requested += 1
        if self._pending_request is not None:
            self.coalesced += 1
        self._pending_request = request
        if interactive:
            if not self._interactive_active:
                self._interactive_active = True
                self._notify_interaction_state_changed()
            self._pending_side_work_cancel = True
            if self._presentation_draw_pending():
                self.presentation_backpressure_skips += 1
                self._quiet_timer.start(self._quiet_interval_ms)
                return
            # Cache/materialization and presentation supersession are render
            # concerns. Probing every visible tile here made a tiny input
            # action block on evaluator cache locks for up to 44 ms. Cancel
            # stale side work cheaply and let the coalesced flush decide what
            # pixels/evaluation can be reused.
            self._quiet_timer.start(self._quiet_interval_ms)
            if not self._render_timer.isActive():
                self._render_timer.start(self._interactive_interval_ms)
            return True
        self._render_timer.start(0)
        return True

    def flush_now(self) -> None:
        self._render_timer.stop()
        self._flush_timer()

    def cancel_pending(self) -> None:
        self._pending_request = None
        self._pending_side_work_cancel = False
        self._render_timer.stop()
        self._quiet_timer.stop()
        if self._interactive_active:
            self._interactive_active = False
            self._notify_interaction_state_changed()

    def _notify_interaction_state_changed(self) -> None:
        notify = getattr(self._window, "_note_interaction_state_changed", None)
        if callable(notify):
            try:
                notify()
            except Exception:
                pass

    def _interactive_cache_hit(self) -> bool:
        predicate = getattr(self._window, "_interactive_frame_cache_hit", None)
        if not callable(predicate):
            return False
        try:
            return bool(predicate())
        except Exception:
            return False

    def _interactive_render_supersedes_presentation(self, reason: str) -> bool:
        predicate = getattr(self._window, "_interactive_render_supersedes_presentation", None)
        if not callable(predicate):
            return False
        try:
            return bool(predicate(reason=reason))
        except Exception:
            return False

    def _visible_work_busy(self) -> bool:
        visible = getattr(self._window, "visible_evaluation_controller", None)
        return bool(visible is not None and visible.is_busy())

    def _connect_presentation_draw_signal(self) -> None:
        view = getattr(self._window, "img_view", None)
        if view is None or view is self._connected_presentation_view:
            return
        signal = getattr(view, "presentationDrawn", None)
        if signal is None:
            return
        try:
            signal.connect(self._on_presentation_drawn)
        except (TypeError, RuntimeError):
            return
        self._connected_presentation_view = view

    def _presentation_draw_pending(self) -> bool:
        view = getattr(self._window, "img_view", None)
        predicate = getattr(view, "presentationDrawPending", None)
        if callable(predicate):
            try:
                return bool(predicate())
            except Exception:
                return False
        return False

    def _on_presentation_drawn(self) -> None:
        if self._pending_request is None or self._render_timer.isActive():
            return
        if self._presentation_draw_pending():
            return
        self.immediate_cache_flushes += 1
        self._render_timer.start(0)

    def _flush_timer(self) -> None:
        self._connect_presentation_draw_signal()
        request = self._pending_request
        self._pending_request = None
        if request is None or getattr(self._window, "_closing", False):
            return
        if request.interactive and self._pending_side_work_cancel:
            self._pending_side_work_cancel = False
            self._window._cancel_render_dependent_work_for_interactive_change()
        self.flushed += 1
        self._window.render(
            reason=request.reason,
            force_autolevel=request.force_autolevel,
            defer_side_panels=bool(request.interactive and self._interactive_active),
        )

    def _quiet_timer_elapsed(self) -> None:
        was_interactive = self._interactive_active
        self._interactive_active = False
        if was_interactive:
            self._notify_interaction_state_changed()
        if getattr(self._window, "_closing", False):
            return
        if self.has_pending_render:
            if self._presentation_draw_pending():
                request = self._pending_request
                reason = "" if request is None else request.reason
                if self._interactive_cache_hit() or not self._interactive_render_supersedes_presentation(reason):
                    self._quiet_timer.start(self._busy_retry_ms)
                    return
            self.immediate_cache_flushes += 1
            self._render_timer.start(0)
            return
        if self._visible_work_busy():
            self._quiet_timer.start(self._busy_retry_ms)
            return
        if getattr(self._window, "_deferred_side_panel_refresh_pending", False):
            self.deferred_side_panel_refreshes += 1
            self._window._run_deferred_side_panel_refresh(reason="interactive-quiet")
