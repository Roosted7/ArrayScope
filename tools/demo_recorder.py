#!/usr/bin/env python
"""Automated README demo recorder for ArrayScope.

Records scripted feature walkthroughs against the real ArrayScope window and
encodes them as optimized GIF, animated AVIF, and H.264 MP4 for the README.
Like tools/ui_gallery.py it uses the PyQtGraph backend on the Qt "offscreen"
platform with a private QSettings name, so it runs headless and never touches
user settings.

A synthetic mouse cursor (large, white with a black outline, click ripples)
and caption pills are composited onto every captured frame, because the
offscreen platform renders no real pointer. Every interaction target is
resolved from live widget geometry at run time — there are no hard-coded
pixel coordinates — so the scenarios keep working as the UI evolves.

Usage:
    python tools/demo_recorder.py                 # record + encode everything
    python tools/demo_recorder.py --list          # list scenarios
    python tools/demo_recorder.py --only fft      # substring filter
    python tools/demo_recorder.py --smoke         # fast, tiny run (CI guard)
    python tools/demo_recorder.py --keep-frames   # keep PNG frame dumps

Output: docs/media/<scenario>.{gif,avif,mp4} (frames under tests/artifacts).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# ``python tools/demo_recorder.py`` puts tools/ at sys.path[0]. Ensure this
# working tree wins before importing any ArrayScope helper.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import contextlib
import itertools

from arrayscope.tools.interaction_budget import (
    INTERACTION_SETTLE_HARD_LIMIT_S,
    bounded_interaction_settle_timeout_s,
)
from arrayscope.tools.presentation_settlement import (
    presentation_is_settled,
    presentation_settlement_diagnostic,
)

DEFAULT_FRAMES_ROOT = REPO_ROOT / "tests" / "artifacts" / "demos"
DEFAULT_MEDIA_OUT = REPO_ROOT / "docs" / "media"
RECORD_FPS = 30
# GIF is the universal fallback: keep it small. AVIF is the primary inline
# format (all current browsers animate it) and stays near-MP4-sized even on
# noisy content, where animated WebP ballooned to 10+ MB.
GIF_FPS = 12
GIF_WIDTH = 880
GIF_LOSSY = 70
AVIF_FPS = 20
AVIF_WIDTH = 880
AVIF_CRF = 34


# --------------------------------------------------------------------------
# Scenario registry
# --------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {}


def scenario(name, *, theme="dark"):
    def register(fn):
        SCENARIOS[name] = {"fn": fn, "theme": theme}
        return fn

    return register


# --------------------------------------------------------------------------
# Synthetic data (deterministic, visually structured; kept in sync by eye
# with tools/ui_gallery.py, duplicated so the two tools stay independent)
# --------------------------------------------------------------------------


def _phantom2d(n=384):
    import numpy as np

    y, x = np.mgrid[0:n, 0:n].astype(np.float64) / n
    img = 0.35 * x + 0.15 * y
    for cx, cy, s, a in (
        (0.32, 0.4, 0.05, 1.0),
        (0.7, 0.3, 0.02, 0.8),
        (0.55, 0.68, 0.09, 0.6),
        (0.8, 0.8, 0.01, 1.4),
    ):
        img += a * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * s)))
    rng = np.random.default_rng(7)
    img += rng.normal(scale=0.02, size=img.shape)
    return img


def _volume3d(nx=96, ny=96, nz=40):
    import numpy as np

    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    z = np.linspace(-3, 3, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vol = np.exp(-((X**2) / 2 + (Y**2) / 3 + (Z**2) / 1.5))
    vol += 0.4 * np.exp(-(((X - 1.2) ** 2 + (Y + 0.8) ** 2 + Z**2) / 0.4))
    vol += 0.25 * np.exp(-(((X + 1.4) ** 2 + (Y - 1.1) ** 2 + (Z - 0.6) ** 2) / 0.2))
    return vol


# --------------------------------------------------------------------------
# Cursor / overlay drawing
# --------------------------------------------------------------------------

CURSOR_SCALE = 1.7
RIPPLE_SECONDS = 0.45
CAPTION_FADE_S = 0.2
CAPTION_LEAD_S = 0.3
CAPTION_BAND_HEIGHT = 78
CAPTION_FONT_PX = 22

# Classic arrow outline in a ~12x19 unit box, hotspot at (0, 0).
_CURSOR_POLY = (
    (0.0, 0.0),
    (0.0, 16.0),
    (4.4, 12.8),
    (7.2, 19.0),
    (10.0, 17.8),
    (7.2, 11.6),
    (12.0, 11.6),
)


def _draw_cursor(painter, pos, pressed):
    from pyqtgraph.Qt import QtCore, QtGui

    scale = CURSOR_SCALE * (0.86 if pressed else 1.0)
    poly = QtGui.QPolygonF(
        [QtCore.QPointF(pos.x() + x * scale, pos.y() + y * scale) for x, y in _CURSOR_POLY]
    )
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    # Soft shadow first, then black outline, then white body: visible on any
    # background, and unmistakably larger than a native pointer.
    shadow = QtGui.QPolygonF([QtCore.QPointF(p.x() + 2.0, p.y() + 2.5) for p in poly])
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, 90))
    painter.drawPolygon(shadow)
    outline = QtGui.QPen(QtGui.QColor(0, 0, 0, 235), 2.6)
    outline.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(outline)
    painter.setBrush(QtGui.QColor(255, 255, 255, 250))
    painter.drawPolygon(poly)


def _draw_ripple(painter, pos, age_fraction):
    from pyqtgraph.Qt import QtCore, QtGui

    radius = 7.0 + 24.0 * age_fraction
    alpha = int(230 * (1.0 - age_fraction))
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    painter.setPen(QtGui.QPen(QtGui.QColor(255, 193, 7, alpha), 3.2))
    painter.drawEllipse(pos, radius, radius)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(255, 193, 7, alpha // 2))
    painter.drawEllipse(pos, 3.5, 3.5)


def _wrap_caption_lines(metrics, text: str, max_width: float, max_lines: int = 2):
    from pyqtgraph.Qt import QtCore

    words = text.split()
    elide_width = max(1, int(max_width))
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for index, word in enumerate(words[1:], start=1):
        candidate = f"{current} {word}"
        if metrics.horizontalAdvance(candidate) <= max_width:
            current = candidate
            continue
        if len(lines) == max_lines - 1:
            remainder = " ".join([current, *words[index:]])
            lines.append(
                metrics.elidedText(remainder, QtCore.Qt.TextElideMode.ElideRight, elide_width)
            )
            return lines
        lines.append(metrics.elidedText(current, QtCore.Qt.TextElideMode.ElideRight, elide_width))
        current = word
    lines.append(metrics.elidedText(current, QtCore.Qt.TextElideMode.ElideRight, elide_width))
    return lines


def _draw_caption_band(painter, width, content_height, text, alpha):
    from pyqtgraph.Qt import QtCore, QtGui

    band = QtCore.QRectF(0.0, float(content_height), float(width), float(CAPTION_BAND_HEIGHT))
    painter.fillRect(band, QtGui.QColor(18, 18, 22, 255))
    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 42), 1.0))
    painter.drawLine(band.topLeft(), band.topRight())
    if not text or alpha <= 0:
        return
    font = QtGui.QFont(painter.font())
    font.setPixelSize(CAPTION_FONT_PX)
    font.setWeight(QtGui.QFont.Weight.DemiBold)
    painter.setFont(font)
    metrics = QtGui.QFontMetricsF(font)
    lines = _wrap_caption_lines(metrics, text, width - 48.0)
    line_height = metrics.height()
    block_height = line_height * len(lines)
    y = content_height + (CAPTION_BAND_HEIGHT - block_height) / 2.0
    painter.setPen(QtGui.QColor(245, 245, 245, int(255 * alpha)))
    for line in lines:
        rect = QtCore.QRectF(24.0, y, width - 48.0, line_height)
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, line)
        y += line_height


def _ease(t: float) -> float:
    # Smooth cubic ease-in-out.
    return 3 * t * t - 2 * t * t * t


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------


class Recorder:
    """Drives one ArrayScope window and captures a fixed-fps frame timeline.

    All durations are in seconds of *output video time*; ``speed`` divides
    them (smoke runs use a high speed factor to stay cheap in CI).
    """

    def __init__(
        self,
        app,
        frames_dir: Path,
        *,
        scenario_name: str,
        fps: int = RECORD_FPS,
        speed: float = 1.0,
    ):
        self.app = app
        self.frames_dir = frames_dir
        self.scenario_name = scenario_name
        self.fps = int(fps)
        self.speed = float(speed)
        self.win = None
        self.frame_index = 0
        self.cursor = None  # QPointF in window coordinates
        self.cursor_pressed = False
        self._ripples = []  # list[(frame_index_born, QPointF)]
        self._caption = ""
        self._caption_born_frame = 0
        self._events: list[dict] = []
        self._open_caption_event: dict | None = None
        self._overlay_widget = None
        self._overlay_top_left = None

    # -- lifecycle ---------------------------------------------------------

    def window(self, data, size=(1100, 780), **kwargs):
        from pyqtgraph.Qt import QtCore

        from arrayscope.window import ArrayScopeWindow

        win = ArrayScopeWindow(data, **kwargs)
        # Encoded video needs a constant frame size: opening docks must
        # squeeze the central widget, never grow the window.
        win.setFixedSize(*size)
        win.show()
        self.win = win
        self.settle()
        # Start with the cursor parked near the lower-right of the image so
        # the first motion reads as "entering the frame".
        self.cursor = QtCore.QPointF(size[0] * 0.62, size[1] * 0.55)
        return win

    def close(self):
        if self.win is not None:
            with contextlib.suppress(Exception):
                self.win.close()
            self._pump_events(0.05)
            self.win = None

    # -- settle (same strictness as tools/ui_gallery.py) ------------------

    def _window_busy(self) -> bool:
        win = self.win
        if not presentation_is_settled(win):
            return True
        try:
            if win._resource_governor_work_active():
                return True
        except Exception:
            pass
        overlay = getattr(getattr(win, "img_view", None), "_evaluation_overlay", None)
        return bool(overlay is not None and overlay.isVisible())

    def settle(self, timeout=INTERACTION_SETTLE_HARD_LIMIT_S, quiet_checks=6):
        timeout = bounded_interaction_settle_timeout_s(timeout)
        deadline = time.monotonic() + timeout
        quiet = 0
        while time.monotonic() < deadline:
            self.app.processEvents()
            # QWidget::grab drives the offscreen backing store through the
            # exact paint path used for captures; without it a resize can
            # retain draw-pending state until the next shot.
            self.win.img_view.grab()
            self.app.processEvents()
            if self._window_busy():
                quiet = 0
            else:
                quiet += 1
                if quiet >= quiet_checks:
                    return True
            time.sleep(0.02)
        raise TimeoutError(
            "demo interaction did not settle within "
            f"{timeout:.3f}s: {presentation_settlement_diagnostic(self.win)!r}"
        )

    # -- frame capture -----------------------------------------------------

    def _pump_events(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while True:
            self.app.processEvents()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.004, remaining))

    def capture(self, *, pump: bool = True) -> None:
        from pyqtgraph.Qt import QtGui

        if pump:
            # Let asynchronous evaluation advance roughly in video time so
            # progressive rendering shows up naturally in the recording.
            self._pump_events(1.0 / self.fps)
        window_image = self.win.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
        image = QtGui.QImage(
            window_image.width(),
            window_image.height() + CAPTION_BAND_HEIGHT,
            QtGui.QImage.Format.Format_RGB32,
        )
        image.fill(QtGui.QColor(18, 18, 22))
        painter = QtGui.QPainter(image)
        try:
            painter.drawImage(0, 0, window_image)
            if self._overlay_widget is not None:
                overlay = (
                    self._overlay_widget.grab()
                    .toImage()
                    .convertToFormat(QtGui.QImage.Format.Format_ARGB32)
                )
                if overlay.isNull() or overlay.width() < 2 or overlay.height() < 2:
                    raise RuntimeError(
                        f"popup overlay grabbed blank from {type(self._overlay_widget).__name__}"
                    )
                painter.drawImage(self._overlay_top_left, overlay)
            caption_age = (self.frame_index - self._caption_born_frame) / self.fps
            alpha = min(1.0, caption_age / CAPTION_FADE_S) if self._caption else 0.0
            _draw_caption_band(
                painter,
                image.width(),
                window_image.height(),
                self._caption,
                alpha,
            )
            live = []
            for born, pos in self._ripples:
                age = (self.frame_index - born) / self.fps
                if age <= RIPPLE_SECONDS:
                    _draw_ripple(painter, pos, max(0.0, age / RIPPLE_SECONDS))
                    live.append((born, pos))
            self._ripples = live
            if self.cursor is not None:
                _draw_cursor(painter, self.cursor, self.cursor_pressed)
        finally:
            painter.end()
        path = self.frames_dir / f"frame_{self.frame_index:05d}.png"
        if not image.save(str(path), "PNG"):
            raise RuntimeError(f"failed to save {path}")
        self.frame_index += 1

    def _frames_for(self, seconds: float) -> int:
        return max(1, round(seconds / self.speed * self.fps))

    def hold(self, seconds: float) -> None:
        for _ in range(self._frames_for(seconds)):
            self.capture()

    def hold_until_settled(self, max_seconds: float, tail_s: float = 0.6) -> None:
        """Capture frames while asynchronous work completes (progressive
        rendering stays visible), then a short settled tail."""
        for _ in range(self._frames_for(max_seconds)):
            self.capture()
            if not self._window_busy():
                break
        self.settle()
        self.hold(tail_s)

    def caption(self, text: str) -> None:
        if self._open_caption_event is not None:
            self._open_caption_event["end_frame"] = self.frame_index
        self._caption = text
        self._caption_born_frame = self.frame_index
        event = {
            "kind": "caption",
            "text": text,
            "start_frame": self.frame_index,
            "end_frame": None,
        }
        self._events.append(event)
        self._open_caption_event = event
        self.hold(CAPTION_LEAD_S)

    def _widget_target(self, widget, rel) -> dict | None:
        if widget is None:
            return None
        rect = widget.rect()
        top_left = self._map_widget_point(widget, rect.topLeft())
        point = self.widget_point(widget, rel)
        return {
            "widget_type": type(widget).__name__,
            "object_name": widget.objectName(),
            "rect": [top_left.x(), top_left.y(), rect.width(), rect.height()],
            "point": [point.x(), point.y()],
        }

    def _map_widget_point(self, widget, local_point):
        from pyqtgraph.Qt import QtCore

        overlay = self._overlay_widget
        if overlay is not None and (widget is overlay or overlay.isAncestorOf(widget)):
            in_overlay = widget.mapTo(overlay, local_point)
            return self._overlay_top_left + QtCore.QPointF(in_overlay)
        return QtCore.QPointF(widget.mapTo(self.win, local_point))

    def _show_overlay(self, widget, anchor_widget, *, placement: str) -> None:
        from pyqtgraph.Qt import QtCore

        widget.ensurePolished()
        widget.adjustSize()
        widget.resize(widget.sizeHint())
        anchor_top_left = anchor_widget.mapTo(self.win, anchor_widget.rect().topLeft())
        anchor_rect = anchor_widget.rect()
        if placement == "above":
            x = anchor_top_left.x()
            y = anchor_top_left.y() - widget.height() - 4
        elif placement == "below":
            x = anchor_top_left.x()
            y = anchor_top_left.y() + anchor_rect.height() + 4
        else:
            raise ValueError(f"unknown overlay placement {placement!r}")
        x = max(4, min(x, self.win.width() - widget.width() - 4))
        y = max(4, min(y, self.win.height() - widget.height() - 4))
        self._overlay_widget = widget
        self._overlay_top_left = QtCore.QPointF(float(x), float(y))

    def _overlay_action_target(self, menu, action) -> dict:
        geometry = menu.actionGeometry(action)
        point = self._overlay_top_left + geometry.center()
        return {
            "widget_type": "QMenuAction",
            "object_name": str(action.data() or ""),
            "rect": [
                self._overlay_top_left.x() + geometry.x(),
                self._overlay_top_left.y() + geometry.y(),
                geometry.width(),
                geometry.height(),
            ],
            "point": [point.x(), point.y()],
        }

    def open_overlay_menu(self, anchor_widget, menu, *, placement: str, label: str) -> None:
        self.click(
            anchor_widget,
            invoke=lambda: self._show_overlay(menu, anchor_widget, placement=placement),
            settle_after=False,
            label=label,
        )

    def open_overlay_widget(self, widget, anchor_widget, *, placement: str, label: str):
        self.transition(
            label,
            lambda: self._show_overlay(widget, anchor_widget, placement=placement),
            settle_after=False,
            hold_s=0.55,
        )

    def edit_overlay_value(self, widget, value, *, label: str) -> None:
        target = self._widget_target(widget, (0.5, 0.5))
        event = self._start_action("edit", label, target=target)
        self.move_to(self.widget_point(widget), seconds=0.45)
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        widget.setValue(value)
        event["activate_frame"] = self.frame_index
        self.hold(0.6)
        self._finish_action(event)

    def click_overlay_widget(self, widget, *, label: str) -> None:
        def activate():
            widget.click()
            self._overlay_widget = None
            self._overlay_top_left = None

        self.click(widget, invoke=activate, label=label)

    def choose_overlay_action(
        self,
        menu,
        action,
        *,
        label: str | None = None,
        move_s: float = 0.5,
        readable_hold_s: float = 0.65,
    ) -> None:
        """Choose a real QAction from a composited offscreen menu.

        Qt's offscreen plugin renders popup surfaces separately from the
        window, and may dismiss a QMenu as soon as it is shown. The recorder
        therefore grabs the built production menu into the window capture,
        while the chosen QAction and its signal remain the real action path.
        """
        from pyqtgraph.Qt import QtCore

        if menu is not self._overlay_widget:
            raise RuntimeError("menu choice requires the menu to be the active overlay")
        geometry = menu.actionGeometry(action)
        if not geometry.isValid() or geometry.isEmpty():
            raise RuntimeError(f"menu action {action.text()!r} has no visible geometry")
        target = self._overlay_action_target(menu, action)
        event = self._start_action(
            "click",
            label or f"choose {action.text().replace('&', '')}",
            target=target,
        )
        menu.setActiveAction(action)
        point = QtCore.QPointF(*target["point"])
        self.move_to(point, seconds=move_s)
        self.hold(readable_hold_s)
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        self._ripples.append((self.frame_index, QtCore.QPointF(self.cursor)))
        self.cursor_pressed = True
        self.hold(0.1)
        action.trigger()
        event["activate_frame"] = self.frame_index
        self.cursor_pressed = False
        self._overlay_widget = None
        self._overlay_top_left = None
        self.settle()
        self.hold(0.3)
        self._finish_action(event)

    def preview_overlay_action(
        self,
        menu,
        action,
        *,
        label: str,
        move_s: float = 0.5,
        readable_hold_s: float = 1.2,
    ) -> None:
        """Highlight a real action without invoking a modal/file-writing path."""
        from pyqtgraph.Qt import QtCore

        if menu is not self._overlay_widget:
            raise RuntimeError("action preview requires the menu to be the active overlay")
        target = self._overlay_action_target(menu, action)
        event = self._start_action("preview", label, target=target)
        menu.setActiveAction(action)
        self.move_to(QtCore.QPointF(*target["point"]), seconds=move_s)
        # Record where the pointer actually rests so the inspector can prove
        # the highlight and the cursor name the same entry.
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        event["activate_frame"] = self.frame_index
        self.hold(readable_hold_s)
        self._finish_action(event)
        self._overlay_widget = None
        self._overlay_top_left = None

    def choose_combo_item(
        self, combo, data, *, label: str, control_label: str, after_trigger=None
    ) -> None:
        """Show a faithful composited choice list and activate the combo item."""
        from pyqtgraph.Qt import QtCore, QtWidgets

        menu = QtWidgets.QMenu(self.win)
        chosen = None
        for index in range(combo.count()):
            action = menu.addAction(combo.itemIcon(index), combo.itemText(index))
            action.setCheckable(True)
            action.setChecked(index == combo.currentIndex())
            model_index = combo.model().index(index, combo.modelColumn(), combo.rootModelIndex())
            action.setEnabled(
                bool(combo.model().flags(model_index) & QtCore.Qt.ItemFlag.ItemIsEnabled)
            )
            action.setData(combo.itemData(index))
            action.triggered.connect(
                lambda checked=False, index=index: combo.setCurrentIndex(index)
            )
            if combo.itemData(index) == data:
                chosen = action
        if chosen is None:
            raise RuntimeError(f"combo item {data!r} not found")
        if after_trigger is not None:
            chosen.triggered.connect(lambda _checked=False: after_trigger())
        self.open_overlay_menu(
            combo,
            menu,
            placement="below",
            label=f"open {control_label} choices",
        )
        self.choose_overlay_action(menu, chosen, label=label)

    @staticmethod
    def _widget_label(widget) -> str:
        """Name a control the way a reviewer would.

        ``text()`` is only a label on buttons and actions. On an editor it is
        the current *content*, which would name the same field "12" in one
        run and "16:112" in the next — unreadable as evidence.
        """
        from pyqtgraph.Qt import QtGui, QtWidgets

        if widget is None:
            return ""
        names = ["accessibleName"]
        if isinstance(widget, QtWidgets.QAbstractButton | QtGui.QAction):
            names.append("text")
        names.append("title")
        for attr in names:
            value = getattr(widget, attr, None)
            if callable(value):
                value = value()
            if value:
                return str(value).replace("&", "")
        return widget.objectName() or type(widget).__name__

    def _start_action(self, kind: str, label: str, *, target: dict | None = None) -> dict:
        event = {
            "kind": kind,
            "label": label,
            "caption": self._caption,
            "start_frame": self.frame_index,
            "press_frame": None,
            "activate_frame": None,
            "end_frame": None,
            "target": target,
            "cursor_at_press": None,
        }
        self._events.append(event)
        return event

    def _finish_action(self, event: dict) -> None:
        event["end_frame"] = self.frame_index

    def transition(self, label: str, invoke, *, settle_after: bool = True, hold_s: float = 0.3):
        """Record an honest non-pointer transition.

        Some documented states have no visible control in the current layout.
        They must not borrow a click ripple from an unrelated cursor position.
        """
        event = self._start_action("transition", label)
        invoke()
        event["activate_frame"] = self.frame_index
        if settle_after:
            self.settle()
        self.hold(hold_s)
        self._finish_action(event)

    def begin_gesture(self, label: str) -> dict:
        from pyqtgraph.Qt import QtCore

        event = self._start_action("drag", label)
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        self._ripples.append((self.frame_index, QtCore.QPointF(self.cursor)))
        self.cursor_pressed = True
        return event

    def finish_gesture(self, event: dict, *, settle_after: bool = True, hold_s: float = 0.3):
        self.cursor_pressed = False
        event["activate_frame"] = max(event["start_frame"], self.frame_index - 1)
        if settle_after:
            self.settle()
        self.hold(hold_s)
        self._finish_action(event)

    def write_manifest(self) -> Path:
        if self._open_caption_event is not None:
            self._open_caption_event["end_frame"] = self.frame_index
            self._open_caption_event = None
        size = self.win.size() if self.win is not None else None
        manifest = {
            "schema_version": 1,
            "scenario": self.scenario_name,
            "record_fps": self.fps,
            "speed": self.speed,
            "frame_count": self.frame_index,
            "window_size": [size.width(), size.height()] if size is not None else None,
            "capture_size": (
                [size.width(), size.height() + CAPTION_BAND_HEIGHT] if size is not None else None
            ),
            "events": self._events,
        }
        path = self.frames_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path

    # -- cursor targeting --------------------------------------------------

    def widget_point(self, widget, rel=(0.5, 0.5)):
        from pyqtgraph.Qt import QtCore

        rect = widget.rect()
        local = QtCore.QPointF(rect.width() * rel[0], rect.height() * rel[1])
        return self._map_widget_point(widget, local.toPoint())

    def view_fraction_point(self, fx: float, fy: float):
        """Image-space point at a fractional position of the current view
        range (robust when the presented layout — e.g. a montage grid — is
        not trivially predictable from the data shape)."""
        (x0, x1), (y0, y1) = self.win.img_view.getView().viewRange()
        return (x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy)

    def image_point(self, x: float, y: float):
        """Window coordinates of an image-space point (matches the mapping
        used by the interaction tests)."""
        from pyqtgraph.Qt import QtCore

        view = self.win.img_view
        scene_pos = view.getView().mapViewToScene(QtCore.QPointF(float(x), float(y)))
        local = view.graphicsView.mapFromScene(scene_pos)
        return QtCore.QPointF(view.graphicsView.viewport().mapTo(self.win, local))

    def move_to(self, target, seconds=0.7, on_point=None) -> None:
        """Ease the cursor to ``target`` (a window-space QPointF, or a widget
        whose center is used). ``on_point`` is called with each intermediate
        window-space position — use it to feed hover/marker updates."""
        from pyqtgraph.Qt import QtCore

        if hasattr(target, "mapTo"):
            target = self.widget_point(target)
        start = QtCore.QPointF(self.cursor)
        steps = self._frames_for(seconds)
        for i in range(1, steps + 1):
            t = _ease(i / steps)
            self.cursor = QtCore.QPointF(
                start.x() + (target.x() - start.x()) * t,
                start.y() + (target.y() - start.y()) * t,
            )
            if on_point is not None:
                on_point(self.cursor)
            self.capture()

    def click(
        self,
        widget=None,
        rel=(0.5, 0.5),
        invoke=None,
        move_s=0.6,
        settle_after=True,
        label=None,
    ) -> None:
        """Move to ``widget`` (optional), show a press with a click ripple,
        then trigger the real behavior — ``invoke`` when given, otherwise the
        widget's own ``click()``/``trigger()``."""
        event = self._start_action(
            "click",
            label or self._widget_label(widget) or "scripted transition",
            target=self._widget_target(widget, rel),
        )
        if widget is not None:
            self.move_to(self.widget_point(widget, rel), seconds=move_s)
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = (
            [self.cursor.x(), self.cursor.y()] if self.cursor is not None else None
        )
        self._ripples.append((self.frame_index, self.cursor))
        self.cursor_pressed = True
        self.hold(0.1)
        if invoke is not None:
            invoke()
        elif widget is not None:
            if hasattr(widget, "click"):
                widget.click()
            elif hasattr(widget, "trigger"):
                widget.trigger()
            else:
                raise TypeError(f"no way to activate {widget!r}")
        event["activate_frame"] = self.frame_index
        self.cursor_pressed = False
        if settle_after:
            self.settle()
        self.hold(0.25)
        self._finish_action(event)

    def toolbar_action_widget(self, toolbar, action):
        widget = toolbar.widgetForAction(action)
        if widget is None:
            raise RuntimeError(f"toolbar widget for {action.text()!r} not found")
        return widget

    def hover_image(self, points, seconds=2.0) -> None:
        """Sweep the cursor through image-space points, delivering real mouse
        move events to the viewport so hover readouts update."""
        from pyqtgraph.Qt import QtCore, QtGui

        view = self.win.img_view
        action_event = self._start_action("hover", "hover image values")
        # Piecewise-linear sweep through the given image points.
        segments = max(1, len(points) - 1)
        per_segment = max(2, self._frames_for(seconds) // segments)
        for a, b in itertools.pairwise(points) if segments > 1 else ((points[0], points[0]),):
            for i in range(per_segment):
                t = _ease((i + 1) / per_segment)
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                scene_pos = view.getView().mapViewToScene(QtCore.QPointF(x, y))
                local = QtCore.QPointF(view.graphicsView.mapFromScene(scene_pos))
                mouse_event = QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseMove,
                    local,
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
                view.eventFilter(view.graphicsView.viewport(), mouse_event)
                self.cursor = QtCore.QPointF(
                    view.graphicsView.viewport().mapTo(self.win, local.toPoint())
                )
                self.capture()
        action_event["activate_frame"] = self.frame_index - 1
        self._finish_action(action_event)

    def zoom_image(self, factor: float, center_xy, seconds=1.2) -> None:
        """Smoothly zoom the viewport by ``factor`` around an image point,
        with the cursor parked on that point (wheel-zoom look)."""
        # Plain "x": the inspector's default bitmap font has no U+00D7, and a
        # tofu box in the evidence sheet is worse than an ASCII multiplier.
        event = self._start_action("zoom", f"zoom image {factor:.2f}x")
        vb = self.win.img_view.getView()
        steps = self._frames_for(seconds)
        per_step = factor ** (1.0 / steps)
        for _ in range(steps):
            vb.scaleBy((1.0 / per_step, 1.0 / per_step), center_xy)
            self.cursor = self.image_point(*center_xy)
            self.capture()
        event["activate_frame"] = self.frame_index - 1
        self._finish_action(event)

    def pan_image(self, dx: float, dy: float, seconds=1.2) -> None:
        """Drag-pan the viewport by an image-space offset with a pressed
        cursor riding the image."""
        from pyqtgraph.Qt import QtCore

        event = self._start_action("drag", "pan image")
        vb = self.win.img_view.getView()
        anchor = QtCore.QPointF(self.cursor)
        self.cursor_pressed = True
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        self._ripples.append((self.frame_index, anchor))
        steps = self._frames_for(seconds)
        for i in range(steps):
            t = _ease((i + 1) / steps) - _ease(i / steps)
            vb.translateBy((-dx * t, -dy * t))
            self.capture()
        self.cursor_pressed = False
        event["activate_frame"] = self.frame_index - 1
        self._finish_action(event)

    def type_into(
        self,
        line_edit,
        text: str,
        on_text=None,
        per_char_s=0.12,
        *,
        label: str | None = None,
        control_label: str | None = None,
    ) -> None:
        """Click into a line edit and type ``text`` one character at a time.
        ``on_text`` (if given) is called with the full text once at the end —
        wire it to the window handler the edit would notify."""
        control_label = control_label or self._widget_label(line_edit)
        event = self._start_action(
            "type",
            label or f"type {text!r} into {control_label}",
            target=self._widget_target(line_edit, (0.5, 0.5)),
        )
        self.click(
            line_edit,
            invoke=lambda: line_edit.setFocus(),
            settle_after=False,
            label=f"focus {control_label}",
        )
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        line_edit.selectAll()
        for i in range(len(text)):
            line_edit.setText(text[: i + 1])
            self.hold(per_char_s)
        if on_text is not None:
            on_text(text)
        event["activate_frame"] = self.frame_index
        self.settle()
        self._finish_action(event)

    def scrub_text_values(
        self,
        line_edit,
        values,
        *,
        on_text,
        per_value_s: float,
        label: str,
        control_label: str | None = None,
    ) -> None:
        """Animate successive real edit values under one manifested action."""
        self.click(
            line_edit,
            invoke=line_edit.setFocus,
            settle_after=False,
            label=f"focus {control_label or self._widget_label(line_edit)}",
        )
        event = self._start_action(
            "scrub",
            label,
            target=self._widget_target(line_edit, (0.5, 0.5)),
        )
        event["press_frame"] = self.frame_index
        event["cursor_at_press"] = [self.cursor.x(), self.cursor.y()]
        for value in values:
            text = str(value)
            line_edit.setText(text)
            on_text(text)
            self.hold(per_value_s)
        event["activate_frame"] = max(event["start_frame"], self.frame_index - 1)
        self.settle()
        self._finish_action(event)


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@scenario("showcase")
def s_showcase(rec: Recorder):
    win = rec.window(_volume3d(128, 128, 48), size=(1100, 780))
    rec.caption("asc(data) — a 128×128×48 volume, one call away")
    rec.hold(2.0)

    rec.caption("Hover for live coordinates and values")
    rec.move_to(rec.image_point(30, 64), seconds=0.7)
    rec.hover_image([(30, 64), (74, 44), (96, 82), (64, 64)], seconds=3.2)
    rec.hold(0.4)

    rec.caption("Scrub through any dimension")
    chip = win.dimension_strip.chip(2)
    rec.scrub_text_values(
        chip.slice_edit,
        list(range(24, 43, 2)) + list(range(42, 9, -3)) + list(range(10, 25, 2)),
        on_text=lambda text: win._on_slice_text_changed(2, text),
        per_value_s=0.09,
        label="scrub axis 2 slices",
        control_label="the axis 2 slice range",
    )
    rec.hold(0.6)

    rec.caption("Crop a displayed axis with an explicit range")
    axis0_edit = win.dimension_strip.chip(0).slice_edit
    rec.type_into(
        axis0_edit,
        "16:112",
        on_text=lambda text: win._on_slice_text_changed(0, text),
        label="crop axis 0 to 16:112",
        control_label="the axis 0 slice range",
    )
    rec.hold_until_settled(1.5, tail_s=0.8)
    rec.caption("Return to the full axis with ':'")
    rec.type_into(
        axis0_edit,
        ":",
        on_text=lambda text: win._on_slice_text_changed(0, text),
        label="restore full axis 0",
        control_label="the axis 0 slice range",
    )
    rec.hold_until_settled(1.5, tail_s=0.6)

    rec.caption("Re-map the image axes with one click")
    rec.click(win.dimension_strip.chip(2).y_button)
    rec.hold_until_settled(2.0, tail_s=1.2)
    rec.click(win.dimension_strip.chip(0).y_button)
    rec.hold_until_settled(2.0, tail_s=1.0)

    rec.caption("Zoom and pan freely — true 1:1 and fit are one click away")
    rec.move_to(rec.image_point(74, 44), seconds=0.6)
    rec.zoom_image(3.0, (74, 44), seconds=1.6)
    rec.hold(0.4)
    rec.pan_image(18, 10, seconds=1.2)
    rec.hold(0.4)
    one_to_one_widget = rec.toolbar_action_widget(
        win.display_toolbar, win.display_toolbar.one_to_one_action
    )
    rec.click(one_to_one_widget, label="show image at true 1:1")
    rec.hold_until_settled(1.5, tail_s=0.8)
    fit_widget = rec.toolbar_action_widget(win.display_toolbar, win.display_toolbar.fit_action)
    rec.click(fit_widget)
    rec.hold_until_settled(2.0, tail_s=1.6)


@scenario("fft")
def s_fft(rec: Recorder):
    from pyqtgraph.Qt import QtWidgets

    win = rec.window(_phantom2d(384), size=(1100, 760))
    rec.caption("A phantom image — let's look at its k-space")
    rec.hold(1.8)

    rec.caption("Add centered FFT steps — the source array is never modified")
    chip0 = win.dimension_strip.chip(0)
    anchor0 = chip0.ops_button.mapToGlobal(chip0.ops_button.rect().topLeft())
    menu0 = win._build_operation_context_menu(0, anchor0)
    fft0 = next(action for action in menu0.actions() if action.data() == "centered_fft")
    rec.open_overlay_menu(
        chip0.ops_button,
        menu0,
        placement="above",
        label="open axis 0 operation menu",
    )
    rec.choose_overlay_action(menu0, fft0, label="choose centered FFT on axis 0")
    rec.hold_until_settled(2.5, tail_s=0.6)
    chip1 = win.dimension_strip.chip(1)
    anchor1 = chip1.ops_button.mapToGlobal(chip1.ops_button.rect().topLeft())
    menu1 = win._build_operation_context_menu(1, anchor1)
    fft1 = next(action for action in menu1.actions() if action.data() == "centered_fft")
    rec.open_overlay_menu(
        chip1.ops_button,
        menu1,
        placement="above",
        label="open axis 1 operation menu",
    )
    rec.choose_overlay_action(menu1, fft1, label="choose centered FFT on axis 1")
    rec.hold_until_settled(2.5, tail_s=1.0)

    rec.caption("Complex data: switch to magnitude on a log scale")
    combo = win.display_toolbar.channel_combo
    scale = win.display_toolbar.scale_combo
    rec.choose_combo_item(scale, "log", label="choose log scale", control_label="scale")
    rec.choose_combo_item(combo, "abs", label="choose magnitude", control_label="channel")
    rec.hold_until_settled(2.0, tail_s=1.6)

    rec.caption("…or phase")
    rec.choose_combo_item(combo, "angle", label="choose phase", control_label="channel")
    rec.hold_until_settled(2.0, tail_s=1.6)

    rec.caption("Every step is reversible — toggle it in the operation stack")

    rec.choose_combo_item(
        combo,
        "abs",
        label="return to logarithmic magnitude",
        control_label="channel",
        after_trigger=lambda: scale.setCurrentIndex(scale.findData("log")),
    )
    rec.settle()
    operation_list = win.operation_dock.operation_list
    first_item = operation_list.item(0)
    first_row = operation_list.itemWidget(first_item)
    first_toggle = next(iter(first_row.findChildren(QtWidgets.QCheckBox)), None)
    if first_toggle is None:
        raise RuntimeError("first operation row has no enable checkbox")
    rec.click(first_toggle, label="disable first FFT")
    rec.hold_until_settled(2.5, tail_s=1.0)
    first_item = operation_list.item(0)
    first_row = operation_list.itemWidget(first_item)
    first_toggle = next(iter(first_row.findChildren(QtWidgets.QCheckBox)), None)
    if first_toggle is None:
        raise RuntimeError("rebuilt operation row has no enable checkbox")
    rec.click(first_toggle, label="enable first FFT", move_s=0.2)
    rec.hold_until_settled(2.5, tail_s=1.6)


@scenario("montage")
def s_montage(rec: Recorder):
    vol = _volume3d(112, 112, 24)
    win = rec.window(vol, size=(1100, 780))
    rec.caption("A 24-slice volume — one slice at a time…")
    rec.hold(1.8)

    rec.caption("Type ':' to montage the whole dimension")
    chip = win.dimension_strip.chip(2)
    rec.type_into(
        chip.slice_edit,
        ":",
        on_text=lambda text: win._on_slice_text_changed(2, text),
        per_char_s=0.45,
        label="enter montage range on axis 2",
        control_label="the axis 2 slice range",
    )
    rec.hold(0.55)
    rec.caption("Tiles are evaluated and presented progressively")
    rec.hold_until_settled(6.0, tail_s=1.6)

    rec.caption("The montage is a real viewport — zoom into any tile")
    center = rec.view_fraction_point(0.38, 0.4)
    rec.move_to(rec.image_point(*center), seconds=0.8)
    rec.zoom_image(2.6, center, seconds=1.5)
    rec.hold(0.8)
    fit_widget = rec.toolbar_action_widget(win.display_toolbar, win.display_toolbar.fit_action)
    rec.click(fit_widget)
    rec.hold_until_settled(2.5, tail_s=1.8)


@scenario("roi")
def s_roi(rec: Recorder):
    import dataclasses

    from arrayscope.core.roi import RoiKind

    win = rec.window(_phantom2d(256), size=(1100, 820))
    rec.caption("Draw ROIs to inspect regions")
    rec.hold(1.4)

    # Rectangle ROI, animated corner-drag: create small, then grow the
    # geometry each frame while the cursor rides the dragged corner.
    x0, y0, x1, y1 = 62.0, 78.0, 148.0, 152.0
    rec.move_to(rec.image_point(x0, y0), seconds=0.8)
    selection = win.img_view.createRoi(RoiKind.RECTANGLE, rect=(x0, y0, 2.0, 2.0))
    rectangle_gesture = rec.begin_gesture("draw rectangle ROI")
    steps = rec._frames_for(1.4)
    for i in range(1, steps + 1):
        t = _ease(i / steps)
        rect = (x0, y0, (x1 - x0) * t, (y1 - y0) * t)
        geometry = dataclasses.replace(selection.geometry, rect=rect)
        win.img_view._set_roi_geometry(selection.id, geometry, emit=i == steps)
        rec.cursor = rec.image_point(x0 + rect[2], y0 + rect[3])
        rec.capture()
    rec.finish_gesture(rectangle_gesture, hold_s=0.6)

    rec.caption("Statistics and histograms update live")
    rec.transition("open inspection panel", win._show_inspection_dock)
    rec.hold_until_settled(2.5, tail_s=1.8)

    rec.caption("Trace a line profile through the features")
    polyline_points = ((30.0, 220.0), (118.0, 96.0), (216.0, 190.0))
    rec.move_to(rec.image_point(*polyline_points[0]), seconds=0.6)
    polyline = win.img_view.createRoi(
        RoiKind.POLYLINE,
        points=(polyline_points[0], polyline_points[0]),
    )
    polyline_gesture = rec.begin_gesture("draw polyline profile")
    drawn = [polyline_points[0]]
    for destination in polyline_points[1:]:
        start = drawn[-1]
        segment_frames = rec._frames_for(0.8)
        for index in range(1, segment_frames + 1):
            t = _ease(index / segment_frames)
            point = (
                start[0] + (destination[0] - start[0]) * t,
                start[1] + (destination[1] - start[1]) * t,
            )
            geometry = dataclasses.replace(polyline.geometry, points=(*drawn, point))
            win.img_view._set_roi_geometry(polyline.id, geometry, emit=False)
            rec.cursor = rec.image_point(*point)
            rec.capture()
        drawn.append(destination)
    geometry = dataclasses.replace(polyline.geometry, points=tuple(drawn))
    win.img_view._set_roi_geometry(polyline.id, geometry, emit=True)
    rec.finish_gesture(polyline_gesture)
    rec.hold_until_settled(2.5, tail_s=1.6)

    # Give the profile finale room: put the inspection panel away again.
    rec.transition("close inspection panel", win.inspection_dock.close)
    rec.hold(0.4)

    rec.caption("Live profiles follow the pointer")
    rec.transition(
        "enable live pointer profile",
        lambda: win.widgets["buttons"]["display"]["live_profile"].setChecked(True),
    )

    for x, y in ((60.0, 60.0), (120.0, 120.0), (200.0, 96.0)):
        rec.move_to(rec.image_point(x, y), seconds=0.7)
        win.img_view.setProfileMarker(x, y, visible=True)
        win._on_profile_marker_moved(x, y)
        win._update_live_profile_from_pending_pos()
        rec.hold_until_settled(1.5, tail_s=0.3)
    rec.hold(1.6)


@scenario("workflow")
def s_workflow(rec: Recorder):
    win = rec.window(_volume3d(96, 96, 32), size=(1100, 780))
    rec.caption("Parameterized operations stay visible and reversible")
    rec.hold(1.2)

    rec.caption("Choose Crop axis… from the real dimension menu")
    chip = win.dimension_strip.chip(0)
    anchor = chip.ops_button.mapToGlobal(chip.ops_button.rect().topLeft())
    menu = win._build_operation_context_menu(0, anchor)
    crop_action = next((action for action in menu.actions() if action.data() == "crop"), None)
    if crop_action is None:
        raise RuntimeError("axis operation menu has no crop action")
    rec.open_overlay_menu(
        chip.ops_button,
        menu,
        placement="above",
        label="open axis 0 operation menu",
    )
    rec.choose_overlay_action(menu, crop_action, label="choose crop axis 0")

    popup = win._operation_params_popup
    if popup is None:
        raise RuntimeError("crop action did not open its parameter popup")
    rec.caption("Edit start and stop; the derived shape updates in place")
    rec.open_overlay_widget(
        popup,
        chip.ops_button,
        placement="above",
        label="show crop parameter popup",
    )
    rec.edit_overlay_value(popup._spins["start"], 16, label="set crop start to 16")
    rec.edit_overlay_value(popup._spins["stop"], 80, label="set crop stop to 80")
    rec.click_overlay_widget(popup._confirm, label="apply crop")
    rec.hold_until_settled(2.0, tail_s=1.0)
    from arrayscope.operations.registry import operation_id_for

    if len(win.document.steps) != 1 or operation_id_for(win.document.steps[0].operation) != "crop":
        raise RuntimeError("crop parameter flow did not commit one crop step")

    rec.caption("Right-click a dimension to export video or PNG frames")
    export_menu = win._build_export_context_menu(2)
    export_action = next(
        (
            action
            for action in export_menu.actions()
            if action.text().startswith("Export along this dimension")
        ),
        None,
    )
    if export_action is None:
        raise RuntimeError("dimension actions have no export entry")
    export_anchor = win.dimension_strip.chip(2).slice_edit
    rec.open_overlay_menu(
        export_anchor,
        export_menu,
        placement="above",
        label="open dimension actions",
    )
    rec.preview_overlay_action(
        export_menu,
        export_action,
        label="preview video and PNG export",
    )
    rec.hold(1.0)


# --------------------------------------------------------------------------
# Child runner
# --------------------------------------------------------------------------


def run_child(name: str, frames_root: Path, *, fps: int, speed: float) -> None:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    app = pg.mkQApp()
    app.setOrganizationName("ArrayScope")
    # Unique per scenario: children run concurrently and must not share a
    # QSettings file.
    app.setApplicationName(f"ArrayScopeDemoRecorder.{name}")
    app.setStyle("Fusion")

    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("theme", SCENARIOS[name]["theme"])
    settings.setValue("image_rendering_backend", "pyqtgraph")
    settings.setValue("first_run_hints_dismissed", True)
    settings.sync()

    frames_dir = frames_root / name
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    rec = Recorder(app, frames_dir, scenario_name=name, fps=fps, speed=speed)
    try:
        SCENARIOS[name]["fn"](rec)
    finally:
        rec.write_manifest()
        rec.close()
        settings.clear()
        settings.sync()
    if rec.frame_index == 0:
        raise RuntimeError(f"scenario {name} captured no frames")
    print(f"{name}: {rec.frame_index} frames")


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}… failed:\n{proc.stderr[-2000:]}")


def encode(
    name: str, frames_dir: Path, out_dir: Path, formats: set[str], fps: int
) -> dict[str, Path]:
    pattern = str(frames_dir / "frame_%05d.png")
    src = ["-framerate", str(fps), "-i", pattern]
    outputs: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    if "mp4" in formats:
        out = out_dir / f"{name}.mp4"
        _run(
            [
                "ffmpeg",
                "-y",
                *src,
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out),
            ]
        )
        outputs["mp4"] = out

    if "avif" in formats:
        out = out_dir / f"{name}.avif"
        _run(
            [
                "ffmpeg",
                "-y",
                *src,
                "-vf",
                f"fps={min(fps, AVIF_FPS)},scale={AVIF_WIDTH}:-2:flags=lanczos",
                "-c:v",
                "libaom-av1",
                "-crf",
                str(AVIF_CRF),
                "-b:v",
                "0",
                "-cpu-used",
                "6",
                "-row-mt",
                "1",
                "-f",
                "avif",
                "-loop",
                "0",
                str(out),
            ]
        )
        outputs["avif"] = out

    if "gif" in formats:
        out = out_dir / f"{name}.gif"
        palette = frames_dir / "palette.png"
        gif_filters = f"fps={min(fps, GIF_FPS)},scale={GIF_WIDTH}:-1:flags=lanczos"
        _run(
            ["ffmpeg", "-y", *src, "-vf", f"{gif_filters},palettegen=stats_mode=diff", str(palette)]
        )
        _run(
            [
                "ffmpeg",
                "-y",
                *src,
                "-i",
                str(palette),
                "-lavfi",
                f"{gif_filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                str(out),
            ]
        )
        if shutil.which("gifsicle"):
            _run(["gifsicle", "-O3", f"--lossy={GIF_LOSSY}", str(out), "-o", str(out)])
        outputs["gif"] = out

    return outputs


# --------------------------------------------------------------------------
# Parent orchestration
# --------------------------------------------------------------------------


def _spawn(name: str, frames_root: Path, fps: int, speed: float) -> tuple[str, bool, str]:
    env = dict(os.environ)
    env.update({"PYQTGRAPH_QT_LIB": "PySide6", "QT_QPA_PLATFORM": "offscreen"})
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        name,
        "--frames",
        str(frames_root),
        "--fps",
        str(fps),
        "--speed",
        str(speed),
    ]
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(REPO_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return name, False, f"demo child process watchdog expired: {exc}"
    return name, proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", metavar="SCENARIO", help=argparse.SUPPRESS)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES_ROOT, help="frame dump root")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_MEDIA_OUT, help="encoded media output dir"
    )
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument(
        "--formats", default="gif,avif,mp4", help="comma list: gif,avif,mp4 or 'none'"
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="re-encode existing frame dumps (implies --keep-frames)",
    )
    parser.add_argument("--fps", type=int, default=RECORD_FPS)
    parser.add_argument(
        "--speed", type=float, default=1.0, help="divide all durations (fast test runs)"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="fast tiny run: --speed 6 --fps 8 --formats none"
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument(
        "--no-inspect",
        action="store_true",
        help="skip final-format storyboards/report after encoding",
    )
    parser.add_argument(
        "--inspection-out",
        type=Path,
        default=REPO_ROOT / "tests" / "artifacts" / "demo-inspection",
        help="inspection artifact directory",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.speed = max(args.speed, 6.0)
        args.fps = min(args.fps, 8)
        args.formats = "none"

    if args.child:
        run_child(args.child, args.frames, fps=args.fps, speed=args.speed)
        return 0

    if args.list:
        for name in SCENARIOS:
            print(name)
        return 0

    names = [n for n in SCENARIOS if not args.only or args.only in n]
    if not names:
        print(f"no scenario matches {args.only!r}", file=sys.stderr)
        return 2

    formats = {f for f in args.formats.split(",") if f and f != "none"}
    unknown = formats - {"gif", "avif", "mp4"}
    if unknown:
        print(f"unknown formats: {sorted(unknown)}", file=sys.stderr)
        return 2

    args.frames.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []
    started = time.monotonic()
    if args.encode_only:
        args.keep_frames = True
        missing = [n for n in names if not (args.frames / n / "frame_00000.png").exists()]
        if missing:
            print(
                f"no frame dumps for {missing} under {args.frames} (record first with --keep-frames)",
                file=sys.stderr,
            )
            return 2
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_spawn, name, args.frames, args.fps, args.speed): name for name in names
            }
            for future in concurrent.futures.as_completed(futures):
                name, ok, log = future.result()
                print(f"[{'ok' if ok else 'FAIL'}] record {name}")
                if not ok:
                    failures.append((name, log))

    recorded = [n for n in names if n not in {f[0] for f in failures}]
    for name in recorded:
        if not formats:
            continue
        manifest = args.frames / name / "manifest.json"
        if not manifest.exists():
            failures.append((name, f"recording produced no manifest at {manifest}"))
            print(f"[FAIL] manifest {name}")
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest, args.out / f"{name}.timeline.json")
        try:
            outputs = encode(name, args.frames / name, args.out, formats, args.fps)
        except RuntimeError as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] encode {name}")
            continue
        sizes = ", ".join(
            f"{kind} {path.stat().st_size / 1e6:.2f} MB" for kind, path in sorted(outputs.items())
        )
        print(f"[ok] encode {name}: {sizes}")
        if not args.keep_frames:
            shutil.rmtree(args.frames / name, ignore_errors=True)

    inspection_failure = ""
    if formats and not failures and not args.no_inspect:
        inspect_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "demo_inspector.py"),
            "--media-root",
            str(args.out),
            "--out",
            str(args.inspection_out),
            "--expected-formats",
            ",".join(
                format_name for format_name in ("mp4", "avif", "gif") if format_name in formats
            ),
            "--strict",
        ]
        if args.only:
            inspect_cmd.extend(["--only", args.only])
        try:
            inspected = subprocess.run(
                inspect_cmd,
                capture_output=True,
                text=True,
                timeout=900,
                cwd=str(REPO_ROOT),
                check=False,
            )
            inspection_log = (inspected.stdout + inspected.stderr).strip()
            if inspected.returncode != 0:
                inspection_failure = inspection_log or "demo inspector returned non-zero"
                print("[FAIL] inspect final media")
            else:
                print("[ok] inspect final media")
                if inspection_log:
                    print(inspection_log)
        except subprocess.TimeoutExpired as exc:
            inspection_failure = f"demo inspection watchdog expired: {exc}"
            print("[FAIL] inspect final media")

    print(
        f"\n{len(names) - len(failures)}/{len(names)} scenarios succeeded in {time.monotonic() - started:.0f}s"
    )
    for name, log in failures:
        print(f"\n--- {name} ---\n{log[-2000:]}")
    if inspection_failure:
        print(f"\n--- final-media inspection ---\n{inspection_failure[-4000:]}")
    return 1 if failures or inspection_failure else 0


if __name__ == "__main__":
    sys.exit(main())
