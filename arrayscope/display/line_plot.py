"""Line plot UI component for ArrayScope."""

from __future__ import annotations

import math

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph as pg
import pyqtgraph.Qt as Qt

from arrayscope.display.slice_engine import make_line
from arrayscope.ui.toasts import show_status_message


class AxisConstrainedViewBox(pg.ViewBox):
    def __init__(self, owner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner = owner

    def wheelEvent(self, ev):
        if self._owner is None or not self._owner.is_line_plot_mode():
            return super().wheelEvent(ev)

        modifiers = ev.modifiers()
        ctrl = modifiers & Qt.QtCore.Qt.KeyboardModifier.ControlModifier
        shift = modifiers & Qt.QtCore.Qt.KeyboardModifier.ShiftModifier
        delta = ev.delta() if hasattr(ev, "delta") else ev.angleDelta().y()
        if delta == 0:
            return

        step = 1.0015 ** abs(delta)
        scale_factor = step if delta < 0 else 1.0 / step
        (x0, x1), (y0, y1) = self.viewRange()
        xspan = x1 - x0
        yspan = y1 - y0

        try:
            if hasattr(ev, "scenePos"):
                scene_pos = ev.scenePos()
            elif hasattr(ev, "scenePosition"):
                scene_pos = ev.scenePosition()
            else:
                scene_pos = ev.pos()
            anchor_pt = self.mapSceneToView(scene_pos)
            ax = anchor_pt.x()
            ay = anchor_pt.y()
        except Exception:
            ax = (x0 + x1) * 0.5
            ay = (y0 + y1) * 0.5

        if not (x0 <= ax <= x1):
            ax = (x0 + x1) * 0.5
        if not (y0 <= ay <= y1):
            ay = (y0 + y1) * 0.5

        fx = 0 if xspan == 0 else (ax - x0) / max(1e-12, xspan)
        fy = 0 if yspan == 0 else (ay - y0) / max(1e-12, yspan)

        if ctrl and not shift:
            new_xspan = xspan * scale_factor
            x0n = ax - fx * new_xspan
            self.setXRange(x0n, x0n + new_xspan, padding=0)
        elif shift and not ctrl:
            new_yspan = yspan * scale_factor
            y0n = ay - fy * new_yspan
            self.setYRange(y0n, y0n + new_yspan, padding=0)
        else:
            new_xspan = xspan * scale_factor
            new_yspan = yspan * scale_factor
            x0n = ax - fx * new_xspan
            y0n = ay - fy * new_yspan
            self.setXRange(x0n, x0n + new_xspan, padding=0)
            self.setYRange(y0n, y0n + new_yspan, padding=0)
        ev.accept()


class LinePlotController:
    def __init__(self, owner):
        self.owner = owner
        self.widget = pg.PlotWidget(viewBox=AxisConstrainedViewBox(owner))
        self.widget.showGrid(x=True, y=True, alpha=0.25)
        pg.setConfigOptions(antialias=True)

        self.current_line_data = None
        self.crosshair = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((150, 0, 0, 180), width=2))
        self.crosshair.setVisible(False)
        self.widget.addItem(self.crosshair)
        self.widget.scene().sigMouseMoved.connect(self.on_hover)

        self.curve = None
        self.curves = []
        self.phase_strip = None
        self.legend = None
        self.base_range = None
        self.base_pen_width = 3.0
        self.min_pen = 2.0
        self.max_pen = 6.0
        self.color = (50, 100, 200)
        self.plot_style = "line"
        self.widget.getViewBox().sigRangeChanged.connect(self.on_range_changed)

    def update(self, data, view_state):
        self.update_line_result(make_line(data, view_state), view_state)

    def update_line_result(self, line_result, view_state, y_range=None):
        self.update_line_results(((line_result, view_state, ""),), y_range=y_range)

    def update_line_results(self, line_entries, y_range=None, phase_strip_data=None):
        self.widget.clear()
        self.widget.showGrid(x=True, y=True, alpha=0.25)
        self.widget.addItem(self.crosshair)
        self.crosshair.setVisible(False)
        self.curves = []
        self.curve = None
        self.phase_strip = None
        self.legend = None
        self.current_line_data = None

        try:
            entries = tuple(line_entries)
            if not entries:
                self.current_line_data = None
                return
            if len(entries) > 1:
                self.legend = self.widget.addLegend(offset=(8, 8))
            colors = ((50, 100, 200), (220, 90, 50), (40, 150, 90), (160, 90, 210), (220, 160, 40))
            first_axis = None
            for index, (line_result, view_state, label) in enumerate(entries):
                line_data = np.asarray(line_result.data)
                if line_data.ndim != 1:
                    show_status_message(self.owner, f"Expected 1D profile, got {line_data.ndim}D shape {line_data.shape}")
                    continue
                if self.current_line_data is None:
                    self.current_line_data = line_data
                if first_axis is None:
                    first_axis = view_state.line_axis
                color = colors[index % len(colors)]
                if self.plot_style == "bar" and len(entries) == 1:
                    x = np.arange(len(line_data))
                    item = pg.BarGraphItem(x=x, height=line_data, width=0.8, brush=pg.mkBrush(*color, 180), pen=pg.mkPen(*color))
                    self.widget.addItem(item)
                else:
                    pen = pg.mkPen(color=color, width=2)
                    item = self.widget.plot(line_data, pen=pen, name=label or f"dim {view_state.line_axis}")
                self.curves.append(item)
                if self.curve is None:
                    self.curve = item
                if self.base_range is None:
                    self.base_range = max(1.0, len(line_data) - 1)
            if phase_strip_data is not None:
                self._add_phase_strip(np.asarray(phase_strip_data))
            if y_range is not None:
                self.widget.getViewBox().setYRange(float(y_range[0]), float(y_range[1]), padding=0)
            else:
                self.widget.enableAutoRange(axis="y")
            if first_axis is not None:
                self.widget.setLabel("bottom", f"Index along dim {first_axis}")

        except Exception as e:
            show_status_message(self.owner, f"Line plot update failed: {e}")
            self.current_line_data = None

    def _add_phase_strip(self, phase_data):
        if phase_data.ndim != 1 or phase_data.size == 0:
            return
        normalized = ((phase_data + np.pi) / (2.0 * np.pi)).reshape(1, -1)
        self.phase_strip = pg.ImageItem(normalized)
        self.phase_strip.setLookupTable(_phase_lut())
        self.phase_strip.setLevels((0, 1))
        self.phase_strip.setOpacity(0.85)
        self.phase_strip.setRect(Qt.QtCore.QRectF(0, -0.08, max(1, phase_data.size - 1), 0.06))
        self.widget.addItem(self.phase_strip)

    def toggle_style(self):
        self.plot_style = "bar" if self.plot_style == "line" else "line"
        self.owner.update_line_plot()

    def hide_crosshair(self):
        if self.crosshair is not None and self.crosshair.isVisible():
            self.crosshair.setVisible(False)

    def on_range_changed(self, _viewbox, ranges):
        if self.curve is None or self.base_range is None:
            return
        if self.plot_style == "bar":
            return

        try:
            x_range, _ = ranges
            x_span = max(1e-9, x_range[1] - x_range[0])
            scale = self.base_range / x_span
            adj = math.log10(1 + scale)
            new_width = self.base_pen_width * adj
            new_width = max(self.min_pen, min(self.max_pen, new_width))
            current_pen = self.curve.opts["pen"]
            if current_pen.widthF() != new_width:
                new_pen = pg.mkPen(self.color, width=new_width)
                self.curve.setPen(new_pen)
        except Exception:
            return

    def on_hover(self, pos):
        if self.current_line_data is None:
            return

        view_box = self.widget.getViewBox()
        if view_box.sceneBoundingRect().contains(pos):
            mouse_point = view_box.mapSceneToView(pos)
            idx = int(round(mouse_point.x()))
            if 0 <= idx < len(self.current_line_data):
                value = self.current_line_data[idx]
                label = self.owner.widgets["labels"]["pixelValue"]
                text = _format_hover_value(idx, value)
                if hasattr(label, "set_pixel_status"):
                    label.set_pixel_status(text)
                else:
                    label.setText(text)
                self.crosshair.setValue(idx)
                if not self.crosshair.isVisible():
                    self.crosshair.setVisible(True)
            else:
                self.hide_crosshair()
        else:
            self.hide_crosshair()


def _format_hover_value(idx, value):
    decimal_places = _decimal_places(abs(value)) if np.isfinite(value) else 3
    if decimal_places > 5:
        return f"[{idx}] = {value:.3e}"
    dp = min(8, max(0, decimal_places))
    return f"[{idx}] = {value:.{dp}f}"


def _decimal_places(number):
    if isinstance(number, (int, np.integer)):
        return 0
    return int(max(1, (number.as_integer_ratio()[1]).bit_length()))


def _phase_lut():
    positions = np.linspace(0.0, 1.0, 256)
    hsv = np.zeros((256, 4), dtype=np.ubyte)
    for index, hue in enumerate(positions):
        color = pg.hsvColor(float(hue), sat=1.0, val=1.0)
        hsv[index] = [color.red(), color.green(), color.blue(), 255]
    return hsv
