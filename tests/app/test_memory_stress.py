import numpy as np

from arrayscope.core.memory_budget import (
    DEFAULT_VISIBLE_RENDER_BUDGET_BYTES,
    estimate_display_image_bytes,
)
from arrayscope.display.levels import finite_bounds
from tests.settle import settle_window


def test_memory_estimate_blocks_large_montage_tile_without_allocation():
    # The montage guard in ``window/frame_controller`` is per tile: a tile whose
    # estimate exceeds the single-tile budget is skipped rather than computed.
    nbytes = estimate_display_image_bytes((8192, 16384), np.float32, histogram=True)

    assert nbytes > DEFAULT_VISIBLE_RENDER_BUDGET_BYTES


def test_large_level_bounds_uses_sampling():
    data = np.arange(250_000, dtype=float).reshape(500, 500)

    bounds = finite_bounds(data, exact_limit=1_000, max_samples=10_000)

    assert bounds is not None
    assert bounds[0] == 0.0
    assert bounds[1] <= float(data.max())


def test_montage_tile_residency_reuses_its_working_set_on_a_second_walk(qtbot):
    """Re-walking the same semantic windows must not grow the accounted set.

    This is the half of the old ``..._rss_stays_bounded`` gate that says
    something about this code: a cache keyed by transient session or view
    objects grows again on the second traversal, and a correctly keyed one
    reuses or evicts inside the first traversal's high-water mark.  It is a
    comparison between two numbers this run produced, so it means the same thing
    on a loaded machine as on an idle one.

    The other half — process RSS, and cache bytes against the installed caps —
    is gone from pytest, because every threshold it used is a fraction of
    ``psutil.virtual_memory().available`` at the moment of the run.  A second
    workload on the box shrinks the caps under an already-resident working set
    and the gate fails (``assert 2948096 <= 2423808``, measured 2026-08-01 under
    three concurrent suites) without anything having changed here.  That
    measurement now lives in
    ``arrayscope.tools.montage_residency_memory_benchmark``, which records the
    policy inputs beside the result so a number from a busy machine is
    identifiable rather than silently comparable.
    """

    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.core.memory_policy import MIN_RENDER_CAP_MB
    from arrayscope.tools.presentation_settlement import (
        presentation_is_settled,
        presentation_target_token,
    )
    from arrayscope.window import ArrayScopeWindow
    from tests.ui.helpers import clear_arrayscope_settings

    # Exercise the smallest render budget the application will actually adopt.
    # The page cache and display cache remain independent owners: the former is
    # governed by this render cap, while the latter has its own policy budget.
    render_budget_mb = MIN_RENDER_CAP_MB
    clear_arrayscope_settings()
    data = np.zeros((256, 256, 64), dtype=np.float32)
    win = ArrayScopeWindow(data)
    win.app_settings = AppSettingsState(
        theme=win.app_settings.theme,
        prefetch_nearby_slices=win.app_settings.prefetch_nearby_slices,
        panel_resize_behavior=win.app_settings.panel_resize_behavior,
        fft_backend=win.app_settings.fft_backend,
        fft_workers=win.app_settings.fft_workers,
        memory_profile=win.app_settings.memory_profile,
        render_memory_budget_mb=render_budget_mb,
    )
    qtbot.addWidget(win)
    accounted_samples = []
    try:
        win.show()
        settle_window(win, lambda: presentation_is_settled(win, require_quiescent=True))

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

            settle_window(win, current_target_is_settled)
            session = win._frame_session
            page_cache = session.lod_page_cache
            display_cache = win.operation_evaluator.display_cache_diagnostics()
            accounted_samples.append(page_cache.bytes_used + display_cache.bytes_used)

        initial_target = presentation_target_token(win)
        assert initial_target is not None
        win._set_view_state(
            win.view_state.with_montage_axis(2, columns=8, indices=tuple(range(64)), text=":")
        )
        win.render(reason="rss-stress")
        wait_and_sample_residency(previous_target=initial_target)
        rows = (1, 3, 5)
        for _pass in range(2):
            for row in rows:
                previous_target = presentation_target_token(win)
                assert previous_target is not None
                y0 = row * (256 + 1)
                win.img_view.getView().setRange(
                    xRange=(0, 256),
                    yRange=(y0, y0 + 256),
                    padding=0,
                )
                win.update_image_view()
                wait_and_sample_residency(previous_target=previous_target)

        # The first traversal is allowed to populate the working set. Repeating
        # the exact same semantic windows must reuse or evict within that
        # high-water mark; a cache keyed by transient session/view objects would
        # grow again on the second traversal.
        first_walk = accounted_samples[1 : 1 + len(rows)]
        second_walk = accounted_samples[1 + len(rows) :]
        assert len(first_walk) == len(second_walk) == len(rows)
        assert max(second_walk) <= max(first_walk)
    finally:
        win.close()
