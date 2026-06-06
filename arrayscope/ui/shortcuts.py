"""Keyboard shortcut mappings used by the main window."""

from __future__ import annotations

import pyqtgraph.Qt as Qt


COLORMAP_SHORTCUTS = {
    Qt.QtCore.Qt.Key.Key_1: "gray",
    Qt.QtCore.Qt.Key.Key_2: "viridis",
    Qt.QtCore.Qt.Key.Key_3: "plasma",
    Qt.QtCore.Qt.Key.Key_4: "PAL-relaxed",
    Qt.QtCore.Qt.Key.Key_5: "cividis",
    Qt.QtCore.Qt.Key.Key_6: "CET-CBL1",
    Qt.QtCore.Qt.Key.Key_7: "d3-cool",
    Qt.QtCore.Qt.Key.Key_8: "d3-warm",
}


def colormap_name_for_key(key):
    return COLORMAP_SHORTCUTS.get(key)
