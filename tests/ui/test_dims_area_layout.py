"""Layout of the dimension-strip area: resize handle + multi-window sync button.

Two behaviours are pinned here:

* the " . . . " drag handle below the dimension chips is only shown when
  there is more than one chip row to trade height between; a single-row strip
  hides it so it does not waste vertical space (issue: always-visible divider);
* the multi-window sync (link) button lives right-aligned in the right column
  beneath the histogram, NOT inside the dimension-chip row, and keeps a margin
  no smaller than the chips' maximum inter-chip spacing (issue: sync button
  eating horizontal space in the chip row).
"""

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from arrayscope.ui.dimension_strip import DimensionStrip
from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings

pytest.importorskip("pytestqt")

_SETTLE_TIMEOUT_MS = min(4000, INTERACTION_SETTLE_HARD_LIMIT_MS)


@pytest.fixture
def make_window(qtbot):
    windows = []

    def make(data):
        from arrayscope.window import ArrayScopeWindow

        win = ArrayScopeWindow(data)
        windows.append(win)
        qtbot.addWidget(win)
        win.show()
        qtbot.waitExposed(win)
        return win

    _clear_arrayscope_settings()
    yield make
    for win in windows:
        win.close()


def _is_descendant(widget, ancestor):
    node = widget.parentWidget() if widget is not None else None
    while node is not None:
        if node is ancestor:
            return True
        node = node.parentWidget()
    return False


def test_resize_handle_hidden_for_single_row(qtbot, make_window):
    # A 2D array is two chips; a comfortably wide window fits them on one row.
    win = make_window(np.arange(8 * 6, dtype=float).reshape(8, 6))
    win.resize(1200, 800)
    qtbot.waitUntil(lambda: win.dimension_strip.row_metrics()[0] == 1, timeout=_SETTLE_TIMEOUT_MS)
    win._sync_dims_area_height()
    assert win.dimension_strip.row_metrics()[0] == 1
    assert not win._dims_resize_handle.isVisible()


def test_resize_handle_visible_for_multiple_rows(qtbot, make_window):
    # Six dimensions in a deliberately narrow window are forced to wrap.
    win = make_window(np.arange(2 * 2 * 2 * 2 * 2 * 3, dtype=float).reshape(2, 2, 2, 2, 2, 3))
    win.resize(420, 800)
    qtbot.waitUntil(lambda: win.dimension_strip.row_metrics()[0] > 1, timeout=_SETTLE_TIMEOUT_MS)
    win._sync_dims_area_height()
    assert win.dimension_strip.row_metrics()[0] > 1
    assert win._dims_resize_handle.isVisible()


def test_sync_button_lives_in_right_column_not_chip_row(qtbot, make_window):
    win = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win.resize(1200, 800)
    qtbot.wait(50)
    button = win.sync_dims_button
    # Not parented under the scrollable chip strip.
    assert not _is_descendant(button, win.dims_scroll)
    assert not _is_descendant(button, win.dimension_strip)
    # Right-aligned: its right edge sits at/after the chip strip's right edge.
    strip_right = win.dimension_strip.mapToGlobal(
        win.dimension_strip.rect().topRight()
    ).x()
    button_right = button.mapToGlobal(button.rect().topRight()).x()
    assert button_right >= strip_right - 1


def test_sync_button_margin_at_least_chip_spacing(qtbot, make_window):
    win = make_window(np.arange(8 * 6 * 4, dtype=float).reshape(8, 6, 4))
    win.resize(1200, 800)
    qtbot.wait(50)
    bar = win._sync_dims_bar
    left, _top, right, _bottom = bar._layout.getContentsMargins()
    # The floor equals the chips' maximum inter-chip spacing.
    assert min(left, right) >= DimensionStrip.PREFERRED_CHIP_SPACING
