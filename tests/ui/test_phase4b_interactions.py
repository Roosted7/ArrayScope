import os

import numpy as np

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _process_events(qtbot, count=8):
    for _ in range(count):
        qtbot.wait(10)


def _clear_arrayscope_settings():
    from pyqtgraph.Qt import QtCore

    settings = QtCore.QSettings("ArrayScope", "ArrayScope")
    settings.clear()
    settings.sync()


def _view_action(win, text):
    for action in win.menuBar().actions():
        if action.text() == "View":
            for child in action.menu().actions():
                if child.text() == text:
                    return child
    raise AssertionError(f"View action not found: {text}")


def test_view_menu_uses_managed_dock_actions(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.arange(8 * 9, dtype=float).reshape(8, 9))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        action = _view_action(win, "Inspection")
        action.trigger()
        _process_events(qtbot, count=20)

        assert win.inspection_dock.isVisible()
        assert action.isChecked()
    finally:
        win.close()


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
