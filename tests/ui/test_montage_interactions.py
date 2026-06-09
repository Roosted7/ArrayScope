import time

import numpy as np
import pytest

from tests.ui.helpers import (
    assert_panel_invariants as _assert_panel_invariants,
    assert_size_close as _assert_size_close,
    clear_arrayscope_settings as _clear_arrayscope_settings,
    panel_body as _panel_body,
    process_events as _process_events,
    view_action as _view_action,
    wait_for_panel_preserve as _wait_for_panel_preserve,
)


def test_montage_roi_gap_source_is_nan(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=2, indices=(0, 1), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=40)

        gap_x = win._current_montage_geometry.tile_width
        source = win._roi_source_image()

        assert np.isnan(source[:, gap_x]).all()
    finally:
        win.close()


def test_montage_status_does_not_remain_computing(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.core.cache_status import CacheStatus
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        assert win.operation_evaluator.last_status.status != CacheStatus.COMPUTING
    finally:
        win.close()


def test_montage_visible_subset_hover_uses_source_index_not_local_tile_zero(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    data = np.zeros((2, 3, 20), dtype=float)
    for index in range(data.shape[2]):
        data[:, :, index] = index
    win = ArrayScopeWindow(data)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=5, indices=tuple(range(20)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)
        win.img_view.getView().setRange(xRange=(0, 2), yRange=(7, 8), padding=0)
        win.update_montage_view()
        qtbot.waitUntil(lambda: win._current_montage_canvas is not None and win._current_montage_canvas.origin_y != 0, timeout=3000)

        canvas = win._current_montage_canvas
        assert canvas.origin_y != 0
        tile_10 = canvas.full_plan.tiles[10]
        context = win.display_geometry.context_for_display_point(
            tile_10.x0 - canvas.origin_x + 1,
            tile_10.y0 - canvas.origin_y + 1,
        )

        assert context is not None
        assert context.context_text.endswith("d2=10")
    finally:
        win.close()


def test_montage_does_not_call_make_montage_for_interactive_render(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.window.render as render_module
    from arrayscope.window import ArrayScopeWindow

    monkeypatch.setattr(render_module, "make_montage", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")), raising=False)
    win = ArrayScopeWindow(np.arange(2 * 3 * 8, dtype=float).reshape(2, 3, 8))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, columns=4, indices=tuple(range(8)), text=":"))
        win.render(reason="test-montage")
        _process_events(qtbot, count=50)

        assert win._current_montage_canvas is not None
    finally:
        win.close()


def test_montage_byte_budget_limits_visible_tiles():
    from arrayscope.core.view_state import ViewState
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.window.render import RenderMixin

    state = ViewState.from_shape((10, 10, 10)).with_montage_axis(2, indices=tuple(range(10)), text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(10)), tile_shape=(10, 10), columns=5, gap=1)

    selected, used, skipped = RenderMixin()._select_visible_montage_tiles_by_budget(
        plan.tiles,
        tile_shape=(10, 10),
        output_dtype=np.float32,
        rgb=False,
        budget_bytes=10 * 10 * 4 * 2,
    )

    assert tuple(tile.source_index for tile in selected) == (0,)
    assert used == 10 * 10 * 8
    assert skipped == 9


def test_visible_render_budget_uses_app_setting():
    from arrayscope.app.settings_state import AppSettingsState
    from arrayscope.window.render import RenderMixin

    mixin = RenderMixin()
    mixin.app_settings = AppSettingsState(render_memory_budget_mb=256)

    assert mixin._visible_render_budget_bytes() == 256 * 1024 * 1024
