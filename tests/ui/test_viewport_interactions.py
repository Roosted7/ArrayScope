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


def test_render_preserves_viewport_for_same_display_shape(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        view = win.img_view.getView()
        view.setRange(xRange=(1, 3), yRange=(1, 3), padding=0)
        before = view.viewRange()

        win._on_slice_index_changed(2, 1)
        _process_events(qtbot, count=20)

        np.testing.assert_allclose(view.viewRange(), before, atol=1e-9)
    finally:
        win.close()


def test_toolbar_fit_and_one_to_one_are_viewport_commands(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.display.viewport import ViewportMode
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(20 * 30, dtype=float).reshape(20, 30))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot, count=20)
        def fail_render(*args, **kwargs):
            raise AssertionError("Fit/1:1 must not render")

        win.render = fail_render
        win.display_toolbar.one_to_one_action.trigger()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.ONE_TO_ONE
        win.display_toolbar.fit_action.trigger()
        _process_events(qtbot, count=20)
        assert win.img_view.viewport_controller.mode == ViewportMode.FIT
        assert not hasattr(win.display_toolbar, "aspect_combo")
    finally:
        win.close()
