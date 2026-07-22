"""Regression: the montage histogram must be populated for BOTH backends.

Field defect (2026-07): the PyQtGraph histogram panel rendered empty while the
VisPy one worked.  A tiled montage has no single bound ``ImageItem``, and the
aggregate ``histogram_plot_data`` derived from level stats is not published on
every backend/commit — so the CPU-LUT histogram had no data source and stayed
empty.  The fix feeds the histogram from the committed tile PAYLOADS
(``semantic_data``, the ADR-0050 semantic pixels), the same backend-agnostic
source of truth VisPy uses.  These tests fail on the pre-fix tree for pyqtgraph.
"""

import numpy as np
import pytest

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings


def _select_image_backend(name: str) -> None:
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings()
    settings.setValue("image_rendering_backend", str(name))
    settings.sync()


def _histogram_curve(win):
    x, y = win.img_view.histogram.item.plot.getData()
    if x is None or y is None:
        return None
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return None
    return x, y


@pytest.mark.parametrize("backend", ["pyqtgraph", "vispy"])
def test_montage_histogram_has_data_points(qtbot, backend):
    _clear_arrayscope_settings()
    _select_image_backend(backend)
    from arrayscope.window import ArrayScopeWindow

    rng = np.random.default_rng(1234)
    data = (rng.standard_normal((128, 128)).astype(np.float32) * 100.0) + 500.0
    data_min = float(np.min(data))
    data_max = float(np.max(data))

    win = ArrayScopeWindow(data)
    win.show()
    qtbot.addWidget(win)
    try:
        # Wait on the actual histogram-ready condition (the plot curve becoming
        # non-empty), not a fixed sleep — the histogram job is async.
        qtbot.waitUntil(
            lambda: _histogram_curve(win) is not None,
            timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
        )
        curve = _histogram_curve(win)
        assert curve is not None
        x, y = curve

        # Non-vacuous: the curve must actually describe the data, not merely
        # exist.  Finite bin centres, a non-degenerate span, counts present,
        # and bins that lie within the data's real value range.
        assert x.size > 0
        assert np.all(np.isfinite(x))
        assert float(np.max(x)) > float(np.min(x))
        assert float(np.nansum(y)) > 0.0
        margin = (data_max - data_min) * 0.05 + 1.0
        assert float(np.min(x)) >= data_min - margin
        assert float(np.max(x)) <= data_max + margin
    finally:
        win.close()
        _select_image_backend("pyqtgraph")
