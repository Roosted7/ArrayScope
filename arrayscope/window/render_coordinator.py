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


class RenderCoordinator(Qt.QtCore.QObject):
    def __init__(
        self,
        window,
        *,
        interactive_interval_ms: int = 16,
        quiet_interval_ms: int = 80,
        busy_retry_ms: int = 40,
    ):
        super().__init__(window)
        self._window = window
        self._interactive_interval_ms = max(0, int(interactive_interval_ms))
        self._quiet_interval_ms = max(1, int(quiet_interval_ms))
        self._busy_retry_ms = max(1, int(busy_retry_ms))
        self._pending_request: RenderRequest | None = None
        self._interactive_active = False
        self.requested = 0
        self.flushed = 0
        self.coalesced = 0
        self.deferred_side_panel_refreshes = 0
        self.immediate_cache_flushes = 0
        self.presentation_backpressure_skips = 0
        self._connected_presentation_view = None

        # Qt event-turn barrier / bounded coalescer. This coalesces bursts of
        # non-cached interactive work without making cache hits wait.
        self._render_timer = Qt.QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._flush_timer)

        # User-interaction quiet detector. It only gates side-panel refreshes;
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

    def request(self, *, reason: str, force_autolevel: bool = False, interactive: bool = False) -> None:
        if getattr(self._window, "_closing", False):
            return
        self._connect_presentation_draw_signal()
        self.requested += 1
        if self._pending_request is not None:
            self.coalesced += 1
        self._pending_request = RenderRequest(
            reason=str(reason),
            force_autolevel=bool(force_autolevel),
            interactive=bool(interactive),
        )
        if interactive:
            self._interactive_active = True
            cache_hit = self._interactive_cache_hit()
            supersedes_presentation = False if cache_hit else self._interactive_render_supersedes_presentation(reason)
            if self._presentation_draw_pending():
                self.presentation_backpressure_skips += 1
                if cache_hit or not supersedes_presentation:
                    self._quiet_timer.start(self._quiet_interval_ms)
                    return
            if not cache_hit:
                self._window._cancel_render_dependent_work_for_interactive_change()
            self._quiet_timer.start(self._quiet_interval_ms)
            if cache_hit:
                self.immediate_cache_flushes += 1
                self._render_timer.start(0)
            elif supersedes_presentation:
                self._render_timer.stop()
                self._flush_timer()
            elif not self._render_timer.isActive():
                self._render_timer.start(self._interactive_interval_ms)
            return
        self._render_timer.start(0)

    def flush_now(self) -> None:
        self._render_timer.stop()
        self._flush_timer()

    def cancel_pending(self) -> None:
        self._pending_request = None
        self._render_timer.stop()
        self._quiet_timer.stop()
        self._interactive_active = False

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
        self.flushed += 1
        self._window.render(
            reason=request.reason,
            force_autolevel=request.force_autolevel,
            defer_side_panels=bool(request.interactive and self._interactive_active),
        )

    def _quiet_timer_elapsed(self) -> None:
        self._interactive_active = False
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
