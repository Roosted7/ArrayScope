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
GIF_WIDTH = 760
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
    for cx, cy, s, a in ((0.32, 0.4, 0.05, 1.0), (0.7, 0.3, 0.02, 0.8), (0.55, 0.68, 0.09, 0.6), (0.8, 0.8, 0.01, 1.4)):
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
CAPTION_FADE_S = 0.25

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
    poly = QtGui.QPolygonF([QtCore.QPointF(pos.x() + x * scale, pos.y() + y * scale) for x, y in _CURSOR_POLY])
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


def _draw_caption(painter, width, height, text, alpha):
    from pyqtgraph.Qt import QtCore, QtGui

    if not text or alpha <= 0:
        return
    font = QtGui.QFont(painter.font())
    font.setPointSizeF(11.5)
    font.setWeight(QtGui.QFont.Weight.DemiBold)
    painter.setFont(font)
    metrics = QtGui.QFontMetricsF(font)
    text_width = metrics.horizontalAdvance(text)
    pad_x, pad_y = 16.0, 8.0
    pill_w = text_width + 2 * pad_x
    pill_h = metrics.height() + 2 * pad_y
    rect = QtCore.QRectF((width - pill_w) / 2.0, height - pill_h - 14.0, pill_w, pill_h)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, int(38 * alpha)), 1.0))
    painter.setBrush(QtGui.QColor(18, 18, 22, int(215 * alpha)))
    painter.drawRoundedRect(rect, pill_h / 2.0, pill_h / 2.0)
    painter.setPen(QtGui.QColor(245, 245, 245, int(255 * alpha)))
    painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)


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

    def __init__(self, app, frames_dir: Path, *, fps: int = RECORD_FPS, speed: float = 1.0):
        self.app = app
        self.frames_dir = frames_dir
        self.fps = int(fps)
        self.speed = float(speed)
        self.win = None
        self.frame_index = 0
        self.cursor = None  # QPointF in window coordinates
        self.cursor_pressed = False
        self._ripples = []  # list[(frame_index_born, QPointF)]
        self._caption = ""
        self._caption_born_frame = 0

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
            try:
                self.win.close()
            except Exception:
                pass
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
        if overlay is not None and overlay.isVisible():
            return True
        return False

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
        image = self.win.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
        painter = QtGui.QPainter(image)
        try:
            caption_age = (self.frame_index - self._caption_born_frame) / self.fps
            alpha = min(1.0, caption_age / CAPTION_FADE_S) if self._caption else 0.0
            _draw_caption(painter, image.width(), image.height(), self._caption, alpha)
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
        self._caption = text
        self._caption_born_frame = self.frame_index

    # -- cursor targeting --------------------------------------------------

    def widget_point(self, widget, rel=(0.5, 0.5)):
        from pyqtgraph.Qt import QtCore

        rect = widget.rect()
        local = QtCore.QPointF(rect.width() * rel[0], rect.height() * rel[1])
        return QtCore.QPointF(widget.mapTo(self.win, local.toPoint()))

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

    def click(self, widget=None, rel=(0.5, 0.5), invoke=None, move_s=0.6, settle_after=True) -> None:
        """Move to ``widget`` (optional), show a press with a click ripple,
        then trigger the real behavior — ``invoke`` when given, otherwise the
        widget's own ``click()``/``trigger()``."""
        if widget is not None:
            self.move_to(self.widget_point(widget, rel), seconds=move_s)
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
        self.cursor_pressed = False
        if settle_after:
            self.settle()
        self.hold(0.25)

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
        # Piecewise-linear sweep through the given image points.
        segments = max(1, len(points) - 1)
        per_segment = max(2, self._frames_for(seconds) // segments)
        for a, b in zip(points, points[1:]) if segments > 1 else ((points[0], points[0]),):
            for i in range(per_segment):
                t = _ease((i + 1) / per_segment)
                x = a[0] + (b[0] - a[0]) * t
                y = a[1] + (b[1] - a[1]) * t
                scene_pos = view.getView().mapViewToScene(QtCore.QPointF(x, y))
                local = QtCore.QPointF(view.graphicsView.mapFromScene(scene_pos))
                event = QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseMove,
                    local,
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.MouseButton.NoButton,
                    QtCore.Qt.KeyboardModifier.NoModifier,
                )
                view.eventFilter(view.graphicsView.viewport(), event)
                self.cursor = QtCore.QPointF(view.graphicsView.viewport().mapTo(self.win, local.toPoint()))
                self.capture()

    def zoom_image(self, factor: float, center_xy, seconds=1.2) -> None:
        """Smoothly zoom the viewport by ``factor`` around an image point,
        with the cursor parked on that point (wheel-zoom look)."""
        vb = self.win.img_view.getView()
        steps = self._frames_for(seconds)
        per_step = factor ** (1.0 / steps)
        for _ in range(steps):
            vb.scaleBy((1.0 / per_step, 1.0 / per_step), center_xy)
            self.cursor = self.image_point(*center_xy)
            self.capture()

    def pan_image(self, dx: float, dy: float, seconds=1.2) -> None:
        """Drag-pan the viewport by an image-space offset with a pressed
        cursor riding the image."""
        from pyqtgraph.Qt import QtCore

        vb = self.win.img_view.getView()
        anchor = QtCore.QPointF(self.cursor)
        self.cursor_pressed = True
        self._ripples.append((self.frame_index, anchor))
        steps = self._frames_for(seconds)
        for i in range(steps):
            t = _ease((i + 1) / steps) - _ease(i / steps)
            vb.translateBy((-dx * t, -dy * t))
            self.capture()
        self.cursor_pressed = False

    def type_into(self, line_edit, text: str, on_text=None, per_char_s=0.12) -> None:
        """Click into a line edit and type ``text`` one character at a time.
        ``on_text`` (if given) is called with the full text once at the end —
        wire it to the window handler the edit would notify."""
        self.click(line_edit, invoke=lambda: line_edit.setFocus(), settle_after=False)
        line_edit.selectAll()
        for i in range(len(text)):
            line_edit.setText(text[: i + 1])
            self.hold(per_char_s)
        if on_text is not None:
            on_text(text)
        self.settle()


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
    rec.move_to(chip.slice_edit, seconds=0.8)
    for index in list(range(24, 43, 2)) + list(range(42, 9, -3)) + list(range(10, 25, 2)):
        chip.slice_edit.setText(str(index))
        win._on_slice_text_changed(2, str(index))
        rec.hold(0.09)
    rec.settle()
    rec.hold(0.6)

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
    fit_widget = rec.toolbar_action_widget(win.display_toolbar, win.display_toolbar.fit_action)
    rec.click(fit_widget)
    rec.hold_until_settled(2.0, tail_s=1.6)


@scenario("fft")
def s_fft(rec: Recorder):
    win = rec.window(_phantom2d(384), size=(1280, 800))
    rec.caption("A phantom image — let's look at its k-space")
    rec.hold(1.8)

    rec.caption("Add centered FFT steps — the source array is never modified")
    chip0 = win.dimension_strip.chip(0)
    rec.click(chip0.ops_button, invoke=lambda: win._append_operation("centered_fft", dim=0))
    rec.hold_until_settled(2.5, tail_s=0.6)
    chip1 = win.dimension_strip.chip(1)
    rec.click(chip1.ops_button, invoke=lambda: win._append_operation("centered_fft", dim=1))
    rec.hold_until_settled(2.5, tail_s=1.0)

    rec.caption("Complex data: switch to magnitude on a log scale")
    combo = win.display_toolbar.channel_combo
    scale = win.display_toolbar.scale_combo
    # Magnitude on a linear scale is a single bright DC pixel (near-black
    # frame); make that beat brief and get to the log view quickly.
    rec.click(combo, invoke=lambda: combo.setCurrentIndex(combo.findData("abs")))
    rec.hold_until_settled(1.5, tail_s=0.15)
    rec.click(scale, invoke=lambda: scale.setCurrentIndex(scale.findData("log")), move_s=0.4)
    rec.hold_until_settled(2.0, tail_s=1.6)

    rec.caption("…or phase")
    rec.click(combo, invoke=lambda: combo.setCurrentIndex(combo.findData("angle")))
    rec.hold_until_settled(2.0, tail_s=1.6)

    rec.caption("Every step is reversible — toggle it in the operation stack")

    def _back_to_log_magnitude():
        # The phase channel auto-switches the scale to linear; restore both
        # so the toggle contrast is star-pattern vs one-axis stripes.
        combo.setCurrentIndex(combo.findData("abs"))
        scale.setCurrentIndex(scale.findData("log"))

    rec.click(combo, invoke=_back_to_log_magnitude, move_s=0.3)
    rec.settle()
    dock_body = win.operation_dock.widget()
    rec.move_to(dock_body, seconds=0.8)
    rec.click(dock_body, rel=(0.15, 0.25), invoke=lambda: win.set_operation_enabled(0, False), move_s=0.3)
    rec.hold_until_settled(2.5, tail_s=1.0)
    rec.click(dock_body, rel=(0.15, 0.25), invoke=lambda: win.set_operation_enabled(0, True), move_s=0.2)
    rec.hold_until_settled(2.5, tail_s=1.6)


@scenario("montage")
def s_montage(rec: Recorder):
    vol = _volume3d(112, 112, 24)
    win = rec.window(vol, size=(1100, 780))
    rec.caption("A 24-slice volume — one slice at a time…")
    rec.hold(1.8)

    rec.caption("Type ':' to montage the whole dimension")
    chip = win.dimension_strip.chip(2)
    rec.type_into(chip.slice_edit, ":", on_text=lambda text: win._on_slice_text_changed(2, text))
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

    win = rec.window(_phantom2d(256), size=(1280, 900))
    rec.caption("Draw ROIs to inspect regions")
    rec.hold(1.4)

    # Rectangle ROI, animated corner-drag: create small, then grow the
    # geometry each frame while the cursor rides the dragged corner.
    x0, y0, x1, y1 = 62.0, 78.0, 148.0, 152.0
    rec.move_to(rec.image_point(x0, y0), seconds=0.8)
    selection = win.img_view.createRoi(RoiKind.RECTANGLE, rect=(x0, y0, 2.0, 2.0))
    rec.cursor_pressed = True
    rec._ripples.append((rec.frame_index, rec.cursor))
    steps = rec._frames_for(1.4)
    for i in range(1, steps + 1):
        t = _ease(i / steps)
        rect = (x0, y0, (x1 - x0) * t, (y1 - y0) * t)
        geometry = dataclasses.replace(selection.geometry, rect=rect)
        win.img_view._set_roi_geometry(selection.id, geometry, emit=i == steps)
        rec.cursor = rec.image_point(x0 + rect[2], y0 + rect[3])
        rec.capture()
    rec.cursor_pressed = False
    rec.settle()
    rec.hold(0.6)

    rec.caption("Statistics and histograms update live")
    rec.click(None, invoke=win._show_inspection_dock, settle_after=True)
    rec.hold_until_settled(2.5, tail_s=1.8)

    rec.caption("Trace a line profile through the features")
    win.img_view.createRoi(RoiKind.POLYLINE, points=((30.0, 220.0), (118.0, 96.0), (216.0, 190.0)))
    rec.move_to(rec.image_point(216, 190), seconds=1.0)
    rec.hold_until_settled(2.5, tail_s=1.6)

    # Give the profile finale room: put the inspection panel away again.
    rec.click(None, invoke=lambda: win.inspection_dock.close(), settle_after=True)
    rec.hold(0.4)

    rec.caption("Live profiles follow the pointer")
    win.widgets["buttons"]["display"]["live_profile"].setChecked(True)
    rec.settle()

    for x, y in ((60.0, 60.0), (120.0, 120.0), (200.0, 96.0)):
        rec.move_to(rec.image_point(x, y), seconds=0.7)
        win.img_view.setProfileMarker(x, y, visible=True)
        win._on_profile_marker_moved(x, y)
        win._update_live_profile_from_pending_pos()
        rec.hold_until_settled(1.5, tail_s=0.3)
    rec.hold(1.6)


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

    rec = Recorder(app, frames_dir, fps=fps, speed=speed)
    try:
        SCENARIOS[name]["fn"](rec)
    finally:
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}… failed:\n{proc.stderr[-2000:]}")


def encode(name: str, frames_dir: Path, out_dir: Path, formats: set[str], fps: int) -> dict[str, Path]:
    pattern = str(frames_dir / "frame_%05d.png")
    src = ["-framerate", str(fps), "-i", pattern]
    outputs: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    if "mp4" in formats:
        out = out_dir / f"{name}.mp4"
        _run(
            ["ffmpeg", "-y", *src, "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
             "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)]
        )
        outputs["mp4"] = out

    if "avif" in formats:
        out = out_dir / f"{name}.avif"
        _run(
            ["ffmpeg", "-y", *src,
             "-vf", f"fps={min(fps, AVIF_FPS)},scale={AVIF_WIDTH}:-2:flags=lanczos",
             "-c:v", "libaom-av1", "-crf", str(AVIF_CRF), "-b:v", "0",
             "-cpu-used", "6", "-row-mt", "1", "-f", "avif", "-loop", "0", str(out)]
        )
        outputs["avif"] = out

    if "gif" in formats:
        out = out_dir / f"{name}.gif"
        palette = frames_dir / "palette.png"
        gif_filters = f"fps={min(fps, GIF_FPS)},scale={GIF_WIDTH}:-1:flags=lanczos"
        _run(["ffmpeg", "-y", *src, "-vf", f"{gif_filters},palettegen=stats_mode=diff", str(palette)])
        _run(
            ["ffmpeg", "-y", *src, "-i", str(palette),
             "-lavfi", f"{gif_filters} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
             str(out)]
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
        sys.executable, str(Path(__file__).resolve()),
        "--child", name, "--frames", str(frames_root),
        "--fps", str(fps), "--speed", str(speed),
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
    except subprocess.TimeoutExpired as exc:
        return name, False, f"demo child process watchdog expired: {exc}"
    return name, proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", metavar="SCENARIO", help=argparse.SUPPRESS)
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES_ROOT, help="frame dump root")
    parser.add_argument("--out", type=Path, default=DEFAULT_MEDIA_OUT, help="encoded media output dir")
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument("--formats", default="gif,avif,mp4", help="comma list: gif,avif,mp4 or 'none'")
    parser.add_argument("--encode-only", action="store_true", help="re-encode existing frame dumps (implies --keep-frames)")
    parser.add_argument("--fps", type=int, default=RECORD_FPS)
    parser.add_argument("--speed", type=float, default=1.0, help="divide all durations (fast test runs)")
    parser.add_argument("--smoke", action="store_true", help="fast tiny run: --speed 6 --fps 8 --formats none")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--keep-frames", action="store_true")
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
            print(f"no frame dumps for {missing} under {args.frames} (record first with --keep-frames)", file=sys.stderr)
            return 2
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(_spawn, name, args.frames, args.fps, args.speed): name for name in names}
            for future in concurrent.futures.as_completed(futures):
                name, ok, log = future.result()
                print(f"[{'ok' if ok else 'FAIL'}] record {name}")
                if not ok:
                    failures.append((name, log))

    recorded = [n for n in names if n not in {f[0] for f in failures}]
    for name in recorded:
        if not formats:
            continue
        try:
            outputs = encode(name, args.frames / name, args.out, formats, args.fps)
        except RuntimeError as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] encode {name}")
            continue
        sizes = ", ".join(f"{kind} {path.stat().st_size / 1e6:.2f} MB" for kind, path in sorted(outputs.items()))
        print(f"[ok] encode {name}: {sizes}")
        if not args.keep_frames:
            shutil.rmtree(args.frames / name, ignore_errors=True)

    print(f"\n{len(names) - len(failures)}/{len(names)} scenarios succeeded in {time.monotonic() - started:.0f}s")
    for name, log in failures:
        print(f"\n--- {name} ---\n{log[-2000:]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
