"""The histogram value-axis view must survive index changes.

A manual zoom/pan of the histogram's value axis is a deliberate choice: an
index-driven presentation refresh must leave it alone. Only an explicit levels
reset (double-click / Auto button, funnelled through
``reset_histogram_view_range``) re-fits the view to the data bounds.

The gating lives on the shared ``ImageView2D`` base, so exercising the
pyqtgraph backend covers the wgpu/vispy backends too (they call the same
``_apply_presentation_histogram_range`` helper).
"""

import numpy as np
import pytest

from tests.display.test_imageview2d import _present_tiled  # reuse the tiled harness

pytest.importorskip("pytestqt")


def _hist_value_range(view):
    return tuple(view.histogram.item.vb.viewRange()[1])


def test_manual_histogram_view_survives_index_change_until_reset(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_tiled(
            view,
            np.zeros((8, 8), dtype=float),
            histogramData=np.zeros((8, 8), dtype=float),
            levels=(0.0, 10.0),
            histogramRange=(0.0, 10.0),
        )

        # The user zooms the histogram's value axis by hand.
        view.histogram.item.vb.setYRange(3.0, 6.0, padding=0)
        assert view._user_histogram_view_dirty
        user_range = _hist_value_range(view)

        # An index change delivers a fresh frame with different data bounds; the
        # manual view must NOT be yanked back to those bounds.
        _present_tiled(
            view,
            np.ones((8, 8), dtype=float),
            histogramData=np.ones((8, 8), dtype=float),
            levels=(0.0, 10.0),
            histogramRange=(0.0, 100.0),
        )
        assert _hist_value_range(view) == pytest.approx(user_range, abs=1e-6)

        # An explicit levels reset re-fits and clears the manual flag.
        view.reset_histogram_view_range(0.0, 100.0)
        assert not view._user_histogram_view_dirty
        assert _hist_value_range(view) != pytest.approx(user_range, abs=1e-6)

        # Once reset, later presentations auto-fit their data bounds again.
        _present_tiled(
            view,
            np.ones((8, 8), dtype=float),
            histogramData=np.ones((8, 8), dtype=float),
            levels=(0.0, 10.0),
            histogramRange=(0.0, 50.0),
        )
        after_reset = _hist_value_range(view)
        assert after_reset != pytest.approx(user_range, abs=1e-6)
    finally:
        view.close()


def test_first_presentation_auto_fits_without_user_interaction(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_tiled(
            view,
            np.zeros((8, 8), dtype=float),
            histogramData=np.zeros((8, 8), dtype=float),
            levels=(0.0, 10.0),
            histogramRange=(0.0, 10.0),
        )
        assert not view._user_histogram_view_dirty
        low, high = _hist_value_range(view)
        # The value axis brackets the data bounds (allowing pyqtgraph padding).
        assert low <= 0.0 + 1e-6
        assert high >= 10.0 - 1e-6
    finally:
        view.close()
