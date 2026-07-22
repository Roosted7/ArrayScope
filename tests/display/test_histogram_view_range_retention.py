"""The histogram value-axis view must survive index changes.

A manual zoom/pan of the histogram's value axis is a deliberate choice: an
index-driven presentation refresh must leave it alone. Only an explicit levels
reset (double-click / Auto button, funnelled through
``reset_histogram_view_range``) re-fits the view to the data bounds.

The pure policy lives in ``histogram_view_range`` and its Qt application lives
in ``HistogramDisplayController``. Exercising the shared controller through the
pyqtgraph view covers the wgpu/vispy backends too.
"""

import numpy as np
import pytest

from tests.display.test_imageview2d import _present_tiled  # reuse the tiled harness

pytest.importorskip("pytestqt")


def _hist_value_range(view):
    return tuple(view.histogram.item.vb.viewRange()[1])


def _hist_view_is_manual(view):
    return view._histogram_display_controller.view_range_is_manual


def _set_manual_hist_value_range(view, low: float, high: float):
    vb = view.histogram.item.vb
    vb.setYRange(float(low), float(high), padding=0)
    vb.sigRangeChangedManually.emit((False, True))


def _present_histogram_range(view, low: float, high: float):
    _present_tiled(
        view,
        np.zeros((8, 8), dtype=float),
        histogramData=np.zeros((8, 8), dtype=float),
        levels=(float(low), float(high)),
        histogramRange=(float(low), float(high)),
    )


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
        _set_manual_hist_value_range(view, 3.0, 6.0)
        assert _hist_view_is_manual(view)
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
        assert not _hist_view_is_manual(view)
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
        assert not _hist_view_is_manual(view)
        low, high = _hist_value_range(view)
        # The value axis brackets the data bounds (allowing pyqtgraph padding).
        assert low <= 0.0 + 1e-6
        assert high >= 10.0 - 1e-6
    finally:
        view.close()


def test_reset_reuses_startup_auto_range_and_keeps_following_presentations(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_histogram_range(view, 0.0, 100.0)
        startup_range = _hist_value_range(view)

        _set_manual_hist_value_range(view, 20.0, 40.0)
        assert _hist_view_is_manual(view)
        view.reset_histogram_view_range(0.0, 100.0)

        assert not _hist_view_is_manual(view)
        assert _hist_value_range(view) == pytest.approx(startup_range, abs=1e-6)

        _present_histogram_range(view, 0.0, 200.0)
        assert not _hist_view_is_manual(view)
        assert _hist_value_range(view)[1] > startup_range[1]
    finally:
        view.close()


def test_auto_range_has_small_zero_floor_and_larger_ceiling_margin(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        # A strictly positive observed minimum still keeps zero as the floor;
        # this covers magnitude data and unsigned/non-negative source types.
        _present_histogram_range(view, 20.0, 100.0)
        low, high = _hist_value_range(view)

        assert low == pytest.approx(-1.0, abs=1e-6)
        assert high == pytest.approx(110.0, abs=1e-6)
    finally:
        view.close()


def test_auto_range_uses_full_margin_below_negative_data(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_histogram_range(view, -100.0, 50.0)
        low, high = _hist_value_range(view)

        assert low == pytest.approx(-115.0, abs=1e-6)
        assert high == pytest.approx(65.0, abs=1e-6)
    finally:
        view.close()


def test_auto_range_hysteresis_is_independent_for_each_side(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_histogram_range(view, -100.0, 100.0)
        initial = _hist_value_range(view)

        # Small contractions do not make the plot breathe with auto-level noise.
        _present_histogram_range(view, -80.0, 80.0)
        assert _hist_value_range(view) == pytest.approx(initial, abs=1e-6)

        # Crossing below 75% contracts only that side.
        _present_histogram_range(view, -70.0, 80.0)
        lower_contracted = _hist_value_range(view)
        assert lower_contracted[0] == pytest.approx(-85.0, abs=1e-6)
        assert lower_contracted[1] == pytest.approx(initial[1], abs=1e-6)

        # Content almost touching the retained ceiling expands only the top.
        _present_histogram_range(view, -70.0, 118.0)
        expanded = _hist_value_range(view)
        assert expanded[0] == pytest.approx(lower_contracted[0], abs=1e-6)
        assert expanded[1] == pytest.approx(136.8, abs=1e-6)
    finally:
        view.close()


def test_fresh_histogram_disables_pyqtgraph_value_auto_pan(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        vb = view.histogram.item.vb
        assert vb.state["autoRange"][vb.YAxis] is False
    finally:
        view.close()


def test_same_data_bounds_do_not_resize_range_during_level_interaction(qt_app, monkeypatch):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_histogram_range(view, 0.0, 100.0)
        calls = []
        original = view.histogram.item.setHistogramRange

        def record_range(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(view.histogram.item, "setHistogramRange", record_range)
        view.histogram.setLevels(20.0, 80.0)
        _present_histogram_range(view, 0.0, 100.0)

        assert calls == []
    finally:
        view.close()


def test_auto_reset_survives_plot_axis_refresh_then_hysteresis_changes_one_endpoint(qt_app):
    from arrayscope.display.imageview2d import ImageView2D

    view = ImageView2D()
    try:
        _present_histogram_range(view, -100.0, 100.0)
        _set_manual_hist_value_range(view, -40.0, 40.0)
        view.reset_histogram_view_range(-100.0, 100.0)
        reset_range = _hist_value_range(view)

        # Histogram count-axis auto-ranging is plot maintenance, not a manual
        # edit of the value-axis view.
        view.histogram.item.vb.setXRange(0.0, 200.0, padding=0)
        assert not _hist_view_is_manual(view)

        _present_histogram_range(view, -70.0, 100.0)
        contracted = _hist_value_range(view)
        assert contracted[0] == pytest.approx(-87.0, abs=1e-6)
        assert contracted[1] == pytest.approx(reset_range[1], abs=1e-6)
    finally:
        view.close()
