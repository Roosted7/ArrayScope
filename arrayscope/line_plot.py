"""Line plot UI component for ArrayScope."""

from __future__ import annotations

import math

import numpy as np

from .qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph as pg
import pyqtgraph.Qt as Qt

from .slice_engine import make_line


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
        pg.setConfigOptions(antialias=True)

        self.current_line_data = None
        self.crosshair = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((150, 0, 0, 180), width=2))
        self.crosshair.setVisible(False)
        self.widget.addItem(self.crosshair)
        self.widget.scene().sigMouseMoved.connect(self.on_hover)

        self.curve = None
        self.base_range = None
        self.base_pen_width = 3.0
        self.min_pen = 2.0
        self.max_pen = 6.0
        self.color = (50, 100, 200)
        self.plot_style = "line"
        self.widget.getViewBox().sigRangeChanged.connect(self.on_range_changed)

    def update(self, data, view_state):
        self.update_line_result(make_line(data, view_state), view_state)

    def update_line_result(self, line_result, view_state):
        self.widget.clear()
        self.widget.addItem(self.crosshair)
        self.crosshair.setVisible(False)

        try:
            line_data = line_result.data

            if line_data.ndim == 1:
                self.current_line_data = line_data

                if self.plot_style == "bar":
                    x = np.arange(len(line_data))
                    brush = pg.mkBrush(50, 100, 200, 180)
                    pen = pg.mkPen(50, 100, 200)
                    self.curve = pg.BarGraphItem(x=x, height=line_data, width=0.8, brush=brush, pen=pen)
                    self.widget.addItem(self.curve)
                    self.owner.tab_widget.setTabText(1, "Bar Plot")
                else:
                    pen = pg.mkPen(color=(50, 100, 200), width=2)
                    self.curve = self.widget.plot(line_data, pen=pen, name="")
                    self.owner.tab_widget.setTabText(1, "Line Plot")

                if self.base_range is None:
                    x_min = 0
                    x_max = len(line_data) - 1
                    self.base_range = max(1.0, (x_max - x_min))
            else:
                print(f"Warning: Expected 1D data but got {line_data.ndim}D data with shape {line_data.shape}")

            self.widget.setLabel("bottom", f"Index along dim {view_state.line_axis}")

        except Exception as e:
            print(f"Line plot update failed: {e}")
            self.current_line_data = None

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
        except Exception as e:
            print(f"Thickness update failed: {e}")

    def on_hover(self, pos):
        if not self.owner.is_line_plot_mode():
            return
        if self.current_line_data is None:
            return

        view_box = self.widget.getViewBox()
        if view_box.sceneBoundingRect().contains(pos):
            mouse_point = view_box.mapSceneToView(pos)
            idx = int(round(mouse_point.x()))
            if 0 <= idx < len(self.current_line_data):
                value = self.current_line_data[idx]
                self.owner.widgets["labels"]["pixelValue"].setText(_format_hover_value(idx, value))
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
