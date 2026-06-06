"""Colormap factories for ArrayScope."""

from __future__ import annotations

import numpy as np

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui


def gray_colormap():
    """Create a grayscale colormap matching pyqtgraph's built-in default."""
    return pg.ColorMap(pos=[0.0, 1.0], color=[[0, 0, 0, 255], [255, 255, 255, 255]])


def phase_colormap():
    """Return pyqtgraph's PAL-relaxed phase map, falling back to cyclic HSV."""
    try:
        colormap = pg.colormap.get("PAL-relaxed")
        if colormap is not None:
            return colormap
    except Exception:
        pass
    return hsv_phase_colormap()


def hsv_phase_colormap():
    """Create a cyclic HSV phase colormap."""
    positions = np.linspace(0.0, 1.0, 257)
    colors = []
    for pos in positions:
        qcolor = QtGui.QColor()
        qcolor.setHsvF(pos % 1.0, 1.0, 1.0, 1.0)
        colors.append([qcolor.red(), qcolor.green(), qcolor.blue(), 255])
    return pg.ColorMap(pos=positions, color=np.array(colors))


def d3_warm_colormap():
    """Create D3.js interpolateWarm colormap."""
    return _d3_cubehelix_colormap(start_hue=-100, end_hue=80)


def d3_cool_colormap():
    """Create D3.js interpolateCool colormap."""
    return _d3_cubehelix_colormap(start_hue=260, end_hue=80)


def named_colormap(colormap_name):
    if colormap_name == "gray":
        return gray_colormap()
    if colormap_name == "PAL-relaxed":
        return phase_colormap()
    if colormap_name == "d3-warm":
        return d3_warm_colormap()
    if colormap_name == "d3-cool":
        return d3_cool_colormap()
    return pg.colormap.get(colormap_name)


def _d3_cubehelix_colormap(start_hue, end_hue):
    # D3 uses cubehelix interpolation with long linear hue interpolation.
    colors = []
    positions = []
    n_samples = 256

    a_const = -0.14861
    b_const = +1.78277
    c_const = -0.29227
    d_const = -0.90649
    e_const = +1.97294

    for i in range(n_samples):
        t = i / (n_samples - 1)
        hue = start_hue + t * (end_hue - start_hue)
        saturation = 0.75 + t * (1.50 - 0.75)
        lightness = 0.35 + t * (0.8 - 0.35)

        h_rad = (hue + 120) * np.pi / 180
        amp = saturation * lightness * (1 - lightness)
        cosh = np.cos(h_rad)
        sinh = np.sin(h_rad)

        red = lightness + amp * (a_const * cosh + b_const * sinh)
        green = lightness + amp * (c_const * cosh + d_const * sinh)
        blue = lightness + amp * (e_const * cosh)

        colors.append(
            (
                int(np.clip(red * 255, 0, 255)),
                int(np.clip(green * 255, 0, 255)),
                int(np.clip(blue * 255, 0, 255)),
            )
        )
        positions.append(t)

    return pg.ColorMap(pos=np.array(positions), color=np.array(colors))
