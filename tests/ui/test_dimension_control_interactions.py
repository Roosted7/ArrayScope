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


def test_tiled_dimension_x_y_buttons_are_disabled_and_guarded(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1), text=":"))
        win.render(reason="test")
        _process_events(qtbot)

        chip = win.dimension_strip.chip(2)
        assert not chip.x_button.isEnabled()
        assert not chip.y_button.isEnabled()
        before = win.view_state
        win.set_dimension_role("x", 2)
        assert win.view_state == before
    finally:
        win.close()


def test_empty_tiled_slice_text_clears_to_midpoint_scalar(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win._set_view_state(win.view_state.with_montage_axis(2, indices=(0, 1, 2), text=":"))
        win._on_slice_text_changed(2, "")
        _process_events(qtbot)

        assert win.view_state.montage_axis is None
        assert win.view_state.axis_range_indices[2] is None
        assert win.view_state.slice_indices[2] == 2
    finally:
        win.close()


def test_live_profile_from_axis_sets_exactly_one_profile_axis(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.profile_axes = (0, 1)
        win._enable_live_profile_for_axis(2)
        _process_events(qtbot)

        assert win.profile_axes == (2,)
        assert win.view_state.line_axis == 2
        assert win.profile_dock.isVisible()
    finally:
        win.close()
