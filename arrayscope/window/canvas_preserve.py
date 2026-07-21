"""Canvas-size preservation for managed panel transitions."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.app.settings_state import PanelResizeBehavior
from arrayscope.core.runtime_diagnostics import CanvasPreserveRuntimeDiagnostics

logger = logging.getLogger(__name__)

_CANVAS_PRESERVE_ATTEMPTS = 4
_CANVAS_PRESERVE_RETRY_MS = 16
_CANVAS_PRESERVE_TOLERANCE_PX = 1
_CANVAS_STRONG_PRESERVE_HOLD_MS = 250
_CANVAS_STRONG_PRESERVE_NUDGE_MS = 16
_EVENT_LIMIT = 20


@dataclass(frozen=True)
class WindowSizeConstraints:
    widget_minimum: Qt.QtCore.QSize
    widget_maximum: Qt.QtCore.QSize
    window_minimum: Qt.QtCore.QSize | None = None
    window_maximum: Qt.QtCore.QSize | None = None


@dataclass(frozen=True)
class CanvasPreserveEvent:
    kind: str
    generation: int
    message: str


@dataclass(frozen=True)
class CanvasPreserveSnapshot:
    active: bool = False
    generation: int = 0
    mode: str = "best_effort"
    platform: str = ""
    last_transition: str = ""
    last_result: str = "none"
    target_canvas_size: tuple[int, int] | None = None
    final_canvas_size: tuple[int, int] | None = None
    final_window_size: tuple[int, int] | None = None
    last_delta: tuple[int, int] | None = None
    attempts_used: int = 0
    strong_used: bool = False
    strong_available: bool = False
    constraints_active: bool = False
    events: tuple[str, ...] = ()


class CanvasPreserveController:
    def __init__(self, layout_manager):
        self.layout_manager = layout_manager
        self.window = layout_manager.window
        self._generation = 0
        self._strong_preserve_constraints: WindowSizeConstraints | None = None
        self._events = deque(maxlen=_EVENT_LIMIT)
        self._active = False
        self._last_transition = ""
        self._last_result = "none"
        self._target_canvas_size = None
        self._final_canvas_size = None
        self._final_window_size = None
        self._last_delta = None
        self._attempts_used = 0
        self._strong_used = False
        self._strong_available = False
        self._mode = PanelResizeBehavior.BEST_EFFORT.value
        self._platform = ""
        # Canvas pixels the screen ceiling forced the canvas to give up and
        # that it is still owed.  See `_settle_canvas_debt`.
        self._canvas_debt = Qt.QtCore.QSize(0, 0)
        self._clamp_bound = False

    @property
    def generation(self) -> int:
        return int(self._generation)

    @property
    def constraints_active(self) -> bool:
        return self._strong_preserve_constraints is not None

    def cancel(self) -> None:
        self._generation += 1
        self._active = False
        self._last_result = "cancel"
        self._record("cancel", f"gen={self._generation}")
        self._release_strong_preserve_constraints(force=True)

    def _single_shot(self, interval_ms: int, callback) -> None:
        """Schedule a window-lifetime-scoped timer callback.

        The window is passed as the receiver context so Qt drops the callback
        when the window's C++ object is destroyed. Without this, a queued
        retry could fire after close and touch a deleted ``QMainWindow``
        (observed as a segfault during teardown). Generation guards protect
        against *stale* callbacks; the receiver context protects against
        *dead* receivers — both are required.
        """

        # Timer category: UI cosmetic. Bounded layout retry preserving canvas
        # size; the receiver context drops it with the window.
        Qt.QtCore.QTimer.singleShot(interval_ms, self.window, callback)

    def run(
        self,
        transition,
        *,
        preserve_canvas: bool,
        allow_strong: bool = True,
        transition_name: str = "",
    ) -> None:
        mode = self._current_mode()
        platform = self._qt_platform_name()
        self._mode = mode.value
        self._platform = platform
        self._last_transition = str(transition_name or "panel")
        self._strong_available = bool(
            allow_strong
            and mode == PanelResizeBehavior.STRONG_WAYLAND
            and _is_wayland_platform(platform)
        )
        if not self._canvas_preserve_enabled(preserve_canvas, mode):
            transition()
            self.layout_manager.refresh_view_geometry()
            self._active = False
            self._last_result = "skipped: off" if mode == PanelResizeBehavior.OFF else "skipped"
            self._record(
                "skip",
                f"gen={self._generation} mode={mode.value} transition={self._last_transition}",
            )
            return

        self._release_strong_preserve_constraints(force=True)
        central = self.window.centralWidget()
        # Aim at the size the canvas is OWED, not merely the size it has: if
        # the screen ceiling made an earlier transition shrink it, this is
        # the transition that can hand those pixels back (closing that dock
        # frees exactly the room that was missing).  Without this the clamp
        # is lossy in one direction and the canvas ratchets down every
        # show/hide cycle -- measured 662 -> 392 px across a single one.
        target_canvas_size = Qt.QtCore.QSize(
            int(central.width()) + int(self._canvas_debt.width()),
            int(central.height()) + int(self._canvas_debt.height()),
        )
        start_window_size = Qt.QtCore.QSize(self.window.size())
        dock_extents = self.layout_manager._visible_managed_dock_extents()
        size_constraints = self._capture_window_size_constraints()
        self._generation += 1
        generation = self._generation
        self._active = True
        self._last_result = "active"
        self._target_canvas_size = _size_tuple(target_canvas_size)
        self._final_canvas_size = None
        self._final_window_size = None
        self._last_delta = None
        self._attempts_used = 0
        self._strong_used = False
        self._clamp_bound = False
        self._record(
            "start",
            (
                f"gen={generation} transition={self._last_transition} mode={mode.value} platform={platform or 'n/a'} "
                f"target={_size_text(target_canvas_size)} window={_size_text(self.window.size())} docks={len(dock_extents)}"
            ),
        )
        transition()
        # A dock this transition just revealed has to join the replay list,
        # or it is sized once against the pre-growth window and never again.
        dock_extents = self.layout_manager.extents_including_newly_shown_docks(dock_extents)
        self.layout_manager._activate_main_window_layout()
        self.layout_manager._restore_visible_dock_extents(dock_extents)
        # Qt event-turn barrier guarded by `generation`. The correction needs
        # the post-transition viewport size; remove when Qt exposes a reliable
        # dock-layout-complete signal.
        self._single_shot(
            0,
            lambda: self._correct_canvas_size(
                target_canvas_size,
                attempts=_CANVAS_PRESERVE_ATTEMPTS,
                generation=generation,
                dock_extents=dock_extents,
                size_constraints=size_constraints,
                strong_used=False,
                start_window_size=start_window_size,
                allow_strong=bool(allow_strong),
            ),
        )

    def diagnostics(self) -> CanvasPreserveRuntimeDiagnostics:
        snapshot = CanvasPreserveSnapshot(
            active=bool(self._active),
            generation=int(self._generation),
            mode=str(self._mode),
            platform=str(self._platform),
            last_transition=str(self._last_transition),
            last_result=str(self._last_result),
            target_canvas_size=self._target_canvas_size,
            final_canvas_size=self._final_canvas_size,
            final_window_size=self._final_window_size,
            last_delta=self._last_delta,
            attempts_used=int(self._attempts_used),
            strong_used=bool(self._strong_used),
            strong_available=bool(self._strong_available),
            constraints_active=self.constraints_active,
            events=tuple(f"{event.kind} {event.message}" for event in self._events),
        )
        return CanvasPreserveRuntimeDiagnostics(**snapshot.__dict__)

    def _canvas_preserve_enabled(self, preserve_canvas: bool, mode: PanelResizeBehavior) -> bool:
        win = self.window
        return (
            bool(preserve_canvas)
            and mode != PanelResizeBehavior.OFF
            and not win.isMaximized()
            and not win.isFullScreen()
            and win.centralWidget() is not None
        )

    def _current_mode(self) -> PanelResizeBehavior:
        settings = getattr(self.window, "app_settings", None)
        mode = getattr(settings, "panel_resize_behavior", PanelResizeBehavior.BEST_EFFORT)
        if not isinstance(mode, PanelResizeBehavior):
            try:
                mode = PanelResizeBehavior(str(mode))
            except Exception:
                mode = PanelResizeBehavior.BEST_EFFORT
        return mode

    def _correct_canvas_size(
        self,
        target_canvas_size,
        *,
        attempts: int,
        generation: int,
        dock_extents,
        size_constraints: WindowSizeConstraints | None = None,
        strong_used: bool = False,
        start_window_size: Qt.QtCore.QSize | None = None,
        allow_strong: bool = True,
    ) -> None:
        if generation != self._generation:
            return
        win = self.window
        central = win.centralWidget()
        if central is None:
            self._finish(generation, "unsettled")
            return
        if attempts <= 0:
            # Out of attempts.  If the SCREEN CEILING is why the canvas never
            # reached its target, book the shortfall as debt so the next
            # transition can repay it; anything else (a widget minimum, a
            # compositor that refused the size) is not ours to remember and
            # would inflate every later target.
            self._book_canvas_debt(generation)
            self._finish(generation, "best_effort_unsettled")
            self._release_strong_preserve_constraints(generation)
            return
        layout = win.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.layout_manager._restore_visible_dock_extents(dock_extents)
        current = central.size()
        dx = int(target_canvas_size.width()) - int(current.width())
        dy = int(target_canvas_size.height()) - int(current.height())
        self._last_delta = (dx, dy)
        self._final_canvas_size = _size_tuple(current)
        self._final_window_size = _size_tuple(win.size())
        self._attempts_used = max(
            self._attempts_used, _CANVAS_PRESERVE_ATTEMPTS - int(attempts) + 1
        )
        self._record(
            "correct",
            (
                f"gen={generation} attempts={attempts} current={_size_text(current)} "
                f"target={_size_text(target_canvas_size)} delta={dx}x{dy} window={_size_text(win.size())}"
            ),
        )
        if abs(dx) <= _CANVAS_PRESERVE_TOLERANCE_PX and abs(dy) <= _CANVAS_PRESERVE_TOLERANCE_PX:
            # Target met: whatever was owed has been repaid.
            self._canvas_debt = Qt.QtCore.QSize(0, 0)
            self.layout_manager.refresh_view_geometry()
            if (
                self._strong_path_allowed(allow_strong)
                and start_window_size is not None
                and win.size() != start_window_size
            ):
                self._poke_window_resize_commit(generation, size_constraints=size_constraints)
                return
            if (
                self._current_mode() == PanelResizeBehavior.STRONG_WAYLAND
                and allow_strong
                and not _is_wayland_platform(self._platform)
            ):
                self._record(
                    "strong_skipped", f"gen={generation} platform={self._platform or 'n/a'}"
                )
            self._finish(generation, "settled")
            self._schedule_strong_preserve_release(generation)
            return
        minimum = win.minimumSize()
        new_width = max(int(minimum.width()), int(win.width()) + dx)
        new_height = max(int(minimum.height()), int(win.height()) + dy)
        # Preserving the canvas means growing the window by whatever the dock
        # took.  Unbounded, that walks the window straight off the screen --
        # measured 1000x700 -> 1658x700 on a 1400x900 work area when the
        # inspection dock opened, and -> 1000x1267 for the profile dock.
        # The screen is a hard ceiling: past it the "preserved" pixels are
        # ones the user cannot see.  Clamping here (BEFORE the QSize handed
        # to the strong-preserve pin, or Wayland would re-assert the
        # off-screen size against the compositor) makes the canvas yield
        # only once there is genuinely nowhere left to grow -- the correction
        # then simply stops converging and finishes best-effort, which is
        # exactly the intended "shrink the viewport, but only then".
        clamped_width, clamped_height = self.layout_manager.clamp_to_available_screen(
            new_width, new_height, minimum=minimum
        )
        if (clamped_width, clamped_height) != (new_width, new_height):
            self._record(
                "clamp",
                (
                    f"gen={generation} wanted={new_width}x{new_height} "
                    f"clamped={clamped_width}x{clamped_height}"
                ),
            )
            self._clamp_bound = True
            new_width, new_height = clamped_width, clamped_height
        if self._clamp_bound and (new_width, new_height) == (
            int(win.width()),
            int(win.height()),
        ):
            # Pinned at the screen ceiling with the canvas still short: the
            # window cannot grow and no number of retries will change that.
            # Stop now instead of burning the remaining attempts (4 x 16 ms)
            # re-measuring an unreachable target, and say so honestly rather
            # than reporting the generic best-effort failure.
            self._book_canvas_debt(generation)
            self._finish(generation, "clamped")
            self._release_strong_preserve_constraints(generation)
            self.layout_manager.refresh_view_geometry()
            return
        if new_width != win.width() or new_height != win.height():
            use_strong = (
                self._strong_path_allowed(allow_strong) and attempts == 1 and not strong_used
            )
            if use_strong:
                self._apply_strong_preserve_constraints(
                    Qt.QtCore.QSize(new_width, new_height),
                    generation,
                    restore_constraints=size_constraints,
                )
            elif (
                attempts == 1
                and self._current_mode() == PanelResizeBehavior.STRONG_WAYLAND
                and allow_strong
            ):
                self._record(
                    "strong_skipped", f"gen={generation} platform={self._platform or 'n/a'}"
                )
            win.resize(new_width, new_height)
            self._record("resize", f"gen={generation} window={new_width}x{new_height}")
            self.layout_manager._restore_visible_dock_extents(dock_extents)
        next_attempts = (
            attempts
            if attempts == 1 and self.constraints_active and not strong_used
            else attempts - 1
        )
        # Qt layout retry guarded by `generation` and bounded attempt count.
        # This handles compositor-delayed resize application.
        self._single_shot(
            _CANVAS_PRESERVE_RETRY_MS,
            lambda: self._correct_canvas_size(
                target_canvas_size,
                attempts=next_attempts,
                generation=generation,
                dock_extents=dock_extents,
                size_constraints=size_constraints,
                strong_used=strong_used or attempts == 1,
                start_window_size=start_window_size,
                allow_strong=allow_strong,
            ),
        )

    def _book_canvas_debt(self, generation: int) -> None:
        """Remember canvas pixels the SCREEN CEILING withheld, so they can be
        handed back by a later transition.

        Only the ceiling creates debt.  Any other reason the canvas fell
        short -- a widget minimum, a compositor that refused the size -- is
        not ours to remember, and booking it would inflate every later
        target.
        """

        if not self._clamp_bound:
            return
        dx_left, dy_left = self._last_delta or (0, 0)
        self._canvas_debt = Qt.QtCore.QSize(max(0, int(dx_left)), max(0, int(dy_left)))
        self._record("debt", f"gen={generation} owed={_size_text(self._canvas_debt)}")

    def _strong_path_allowed(self, allow_strong: bool) -> bool:
        return (
            bool(allow_strong)
            and self._current_mode() == PanelResizeBehavior.STRONG_WAYLAND
            and _is_wayland_platform(self._platform)
        )

    def _poke_window_resize_commit(
        self,
        generation: int,
        *,
        size_constraints: WindowSizeConstraints | None = None,
    ) -> None:
        if generation != self._generation:
            return
        current_size = Qt.QtCore.QSize(self.window.size())
        self._record("commit_poke", f"gen={generation} window={_size_text(current_size)}")
        self._apply_strong_preserve_constraints(
            current_size,
            generation,
            restore_constraints=size_constraints,
            reason="commit-poke",
        )
        self.window.resize(current_size)
        self.window.updateGeometry()
        self.window.update()
        handle = self.window.windowHandle()
        if handle is not None:
            handle.resize(current_size)
            handle.requestUpdate()
        # Wayland commit nudge guarded by `generation`; remove when the strong
        # preserve path no longer needs compositor acknowledgement nudges.
        self._single_shot(
            _CANVAS_STRONG_PRESERVE_NUDGE_MS,
            lambda: self._nudge_window_resize_commit(generation, target_size=current_size),
        )

    def _nudge_window_resize_commit(self, generation: int, *, target_size: Qt.QtCore.QSize) -> None:
        if generation != self._generation or self._strong_preserve_constraints is None:
            return
        nudge_size = Qt.QtCore.QSize(target_size.width() + 1, target_size.height())
        self._record(
            "commit_nudge",
            f"gen={generation} nudge={_size_text(nudge_size)} target={_size_text(target_size)}",
        )
        self.window.setMinimumSize(nudge_size)
        self.window.setMaximumSize(nudge_size)
        handle = self.window.windowHandle()
        if handle is not None:
            handle.setMinimumSize(nudge_size)
            handle.setMaximumSize(nudge_size)
        self.window.resize(nudge_size)
        if handle is not None:
            handle.resize(nudge_size)
            handle.requestUpdate()
        # Wayland commit nudge completion guarded by `generation`.
        self._single_shot(
            _CANVAS_STRONG_PRESERVE_NUDGE_MS,
            lambda: self._finish_window_resize_commit_nudge(generation, target_size=target_size),
        )

    def _finish_window_resize_commit_nudge(
        self, generation: int, *, target_size: Qt.QtCore.QSize
    ) -> None:
        if generation != self._generation or self._strong_preserve_constraints is None:
            return
        self._record("commit_nudge_finish", f"gen={generation} target={_size_text(target_size)}")
        self.window.setMinimumSize(target_size)
        self.window.setMaximumSize(target_size)
        handle = self.window.windowHandle()
        if handle is not None:
            handle.setMinimumSize(target_size)
            handle.setMaximumSize(target_size)
        self.window.resize(target_size)
        self.window.updateGeometry()
        self.window.update()
        if handle is not None:
            handle.resize(target_size)
            handle.requestUpdate()
        self._finish(generation, "settled")
        self._schedule_strong_preserve_release(generation)

    def _capture_window_size_constraints(self) -> WindowSizeConstraints:
        handle = self.window.windowHandle()
        return WindowSizeConstraints(
            widget_minimum=Qt.QtCore.QSize(self.window.minimumSize()),
            widget_maximum=Qt.QtCore.QSize(self.window.maximumSize()),
            window_minimum=Qt.QtCore.QSize(handle.minimumSize()) if handle is not None else None,
            window_maximum=Qt.QtCore.QSize(handle.maximumSize()) if handle is not None else None,
        )

    def _apply_strong_preserve_constraints(
        self,
        target_size: Qt.QtCore.QSize,
        generation: int,
        *,
        restore_constraints: WindowSizeConstraints | None = None,
        reason: str = "final-retry",
    ) -> None:
        if generation != self._generation:
            return
        if self._strong_preserve_constraints is None:
            self._strong_preserve_constraints = (
                restore_constraints or self._capture_window_size_constraints()
            )
        self._strong_used = True
        self._record(
            "strong_apply", f"gen={generation} reason={reason} fixed={_size_text(target_size)}"
        )
        self.window.setMinimumSize(target_size)
        self.window.setMaximumSize(target_size)
        handle = self.window.windowHandle()
        if handle is not None:
            handle.setMinimumSize(target_size)
            handle.setMaximumSize(target_size)

    def _schedule_strong_preserve_release(self, generation: int) -> None:
        if self._strong_preserve_constraints is None:
            return
        # User/layout timeout guarded by `generation`. It releases temporary
        # size constraints after the compositor has had time to apply them.
        self._single_shot(
            _CANVAS_STRONG_PRESERVE_HOLD_MS,
            lambda: self._release_strong_preserve_constraints(generation),
        )

    def _release_strong_preserve_constraints(
        self, generation: int | None = None, *, force: bool = False
    ) -> None:
        if not force and generation != self._generation:
            return
        constraints = self._strong_preserve_constraints
        if constraints is None:
            return
        self._strong_preserve_constraints = None
        self._record(
            "strong_release",
            (
                f"gen={self._generation} restore_min={_size_text(constraints.widget_minimum)} "
                f"restore_max={_size_text(constraints.widget_maximum)}"
            ),
        )
        self.window.setMinimumSize(constraints.widget_minimum)
        self.window.setMaximumSize(constraints.widget_maximum)
        handle = self.window.windowHandle()
        if handle is not None:
            if constraints.window_minimum is not None:
                handle.setMinimumSize(constraints.window_minimum)
            if constraints.window_maximum is not None:
                handle.setMaximumSize(constraints.window_maximum)

    def _finish(self, generation: int, result: str) -> None:
        if generation != self._generation:
            return
        central = self.window.centralWidget()
        self._active = False
        self._last_result = str(result)
        self._final_canvas_size = None if central is None else _size_tuple(central.size())
        self._final_window_size = _size_tuple(self.window.size())
        self._record(
            "settled" if result == "settled" else "unsettled", f"gen={generation} result={result}"
        )

    def _qt_platform_name(self) -> str:
        app = QtWidgets.QApplication.instance()
        return "" if app is None else str(app.platformName()).lower()

    def _record(self, kind: str, message: str) -> None:
        event = CanvasPreserveEvent(str(kind), int(self._generation), str(message))
        self._events.append(event)
        logger.debug("preserve-canvas %s", event.message)


def _is_wayland_platform(platform: str) -> bool:
    return "wayland" in str(platform).lower()


def _size_tuple(size) -> tuple[int, int]:
    return (int(size.width()), int(size.height()))


def _size_text(size) -> str:
    return f"{int(size.width())}x{int(size.height())}"
