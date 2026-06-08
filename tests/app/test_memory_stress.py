import numpy as np
import pytest

from arrayscope.core.memory_budget import MONTAGE_BUDGET_BYTES, estimate_montage_bytes
from arrayscope.display.levels import finite_bounds


def test_memory_estimate_blocks_large_montage_without_allocation():
    nbytes = estimate_montage_bytes((8192, 8192), 128, np.float32, histogram=True, columns=16)

    assert nbytes > MONTAGE_BUDGET_BYTES


def test_large_level_bounds_uses_sampling():
    data = np.arange(250_000, dtype=float).reshape(500, 500)

    bounds = finite_bounds(data, exact_limit=1_000, max_samples=10_000)

    assert bounds is not None
    assert bounds[0] == 0.0
    assert bounds[1] <= float(data.max())


def test_montage_viewport_canvas_rss_stays_bounded(qtbot, monkeypatch):
    psutil = pytest.importorskip("psutil")
    from tests.ui.helpers import clear_arrayscope_settings, process_events
    import arrayscope.window.render as render_module
    from arrayscope.window import ArrayScopeWindow

    budget = 8 * 1024 * 1024
    monkeypatch.setattr(render_module, "VISIBLE_RENDER_BUDGET_BYTES", budget)
    clear_arrayscope_settings()
    data = np.zeros((256, 256, 64), dtype=np.float32)
    process = psutil.Process()
    before = process.memory_info().rss
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    rss_samples = []
    try:
        process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(64)), text=":"))
        win.render(reason="rss-stress")
        qtbot.waitUntil(lambda: getattr(win, "_current_montage_canvas", None) is not None, timeout=5000)
        rss_samples.append(process.memory_info().rss)
        for row in (1, 3, 5):
            y0 = row * (256 + 1)
            win.img_view.getView().setRange(xRange=(0, 256), yRange=(y0, y0 + 256), padding=0)
            win.update_montage_view()
            process_events(qtbot, count=80)
            rss_samples.append(process.memory_info().rss)

        tolerance = budget + 128 * 1024 * 1024
        assert max(rss_samples) - before < tolerance
        assert max(rss_samples) - min(rss_samples) < tolerance
    finally:
        win.close()
