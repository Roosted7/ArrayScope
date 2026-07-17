import numpy as np
import pytest

from arrayscope.core.memory_budget import DEFAULT_MONTAGE_RESIDENCY_BUDGET_BYTES, estimate_montage_tile_grid_bytes
from arrayscope.display.levels import finite_bounds
from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS


def test_memory_estimate_blocks_large_montage_without_allocation():
    nbytes = estimate_montage_tile_grid_bytes((8192, 8192), 128, np.float32, histogram=True, columns=16)

    assert nbytes > DEFAULT_MONTAGE_RESIDENCY_BUDGET_BYTES


def test_large_level_bounds_uses_sampling():
    data = np.arange(250_000, dtype=float).reshape(500, 500)

    bounds = finite_bounds(data, exact_limit=1_000, max_samples=10_000)

    assert bounds is not None
    assert bounds[0] == 0.0
    assert bounds[1] <= float(data.max())


def test_montage_tile_residency_rss_stays_bounded(qtbot):
    psutil = pytest.importorskip("psutil")
    from pytestqt.exceptions import TimeoutError as QtBotTimeoutError
    from tests.ui.helpers import clear_arrayscope_settings
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.tools.presentation_settlement import (
        presentation_is_settled,
        presentation_settlement_diagnostic,
        presentation_target_token,
    )
    from arrayscope.window import ArrayScopeWindow

    budget = 8 * 1024 * 1024
    clear_arrayscope_settings()
    data = np.zeros((256, 256, 64), dtype=np.float32)
    process = psutil.Process()
    win = ArrayScopeWindow(data)
    win.app_settings = AppSettingsState(
        theme=win.app_settings.theme,
        prefetch_nearby_slices=win.app_settings.prefetch_nearby_slices,
        panel_resize_behavior=win.app_settings.panel_resize_behavior,
        fft_backend=win.app_settings.fft_backend,
        fft_workers=win.app_settings.fft_workers,
        memory_profile=win.app_settings.memory_profile,
        render_memory_budget_mb=8,
    )
    qtbot.addWidget(win)
    rss_samples = []
    accounted_samples = []
    try:
        win.show()
        try:
            qtbot.waitUntil(
                lambda: presentation_is_settled(win, require_quiescent=True),
                timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
            )
        except QtBotTimeoutError as exc:
            raise AssertionError(presentation_settlement_diagnostic(win)) from exc
        first_settled_rss = process.memory_info().rss
        # Constructing the Qt/pyqtgraph backend establishes a large, one-time
        # process high-water mark which is unrelated to montage residency.
        # The first physical presentation is the startup boundary: measure the
        # viewport walk after it so this gate detects tile/page ownership rather
        # than backend construction or an in-flight initial frame.
        window_baseline = first_settled_rss

        def wait_and_sample_residency(*, previous_target=None):
            def current_target_is_settled():
                current_target = presentation_target_token(win)
                if current_target is None:
                    return False
                if previous_target is not None and current_target == previous_target:
                    return False
                return presentation_is_settled(
                    win,
                    expected_target=current_target,
                    require_quiescent=True,
                )

            try:
                qtbot.waitUntil(
                    current_target_is_settled,
                    timeout=INTERACTION_SETTLE_HARD_LIMIT_MS,
                )
            except QtBotTimeoutError as exc:
                raise AssertionError(presentation_settlement_diagnostic(win)) from exc
            session = win._frame_session
            page_cache = session.lod_page_cache
            display_cache = win.operation_evaluator.display_cache_diagnostics()
            assert page_cache.bytes_used <= page_cache.max_bytes
            assert display_cache.bytes_used <= display_cache.max_bytes
            rss_samples.append(process.memory_info().rss)
            accounted_samples.append(page_cache.bytes_used + display_cache.bytes_used)

        initial_target = presentation_target_token(win)
        assert initial_target is not None
        win._set_view_state(win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(64)), text=":"))
        win.render(reason="rss-stress")
        wait_and_sample_residency(previous_target=initial_target)
        for row in (1, 3, 5):
            previous_target = presentation_target_token(win)
            assert previous_target is not None
            y0 = row * (256 + 1)
            win.img_view.getView().setRange(xRange=(0, 256), yRange=(y0, y0 + 256), padding=0)
            win.update_image_view()
            wait_and_sample_residency(previous_target=previous_target)

        tolerance = budget + 128 * 1024 * 1024
        assert max(accounted_samples) <= budget
        assert max(rss_samples) - window_baseline < tolerance
        assert max(rss_samples) - min(rss_samples) < tolerance
    finally:
        win.close()
