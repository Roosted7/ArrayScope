import numpy as np

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings
from tests.ui.helpers import process_events as _process_events

_GRID = "Pixel Grid (when zoomed in)"
_CLIP = "Clipping Indicator"


def _submenu_action(win, menu_text, submenu_text, action_text):
    # Iterate with the parent actions still in scope: a top-level menu created
    # via ``menuBar().addMenu("View")`` keeps no Python reference, so fetching
    # ``action.menu()`` into a temporary and touching it later lets shiboken
    # collect the wrapper (RuntimeError: C++ object already deleted). Mirrors
    # the known-good ``tests.ui.helpers.view_action`` pattern.
    for menu_action in win.menuBar().actions():
        if menu_action.text() != menu_text:
            continue
        for sub in menu_action.menu().actions():
            if sub.text() != submenu_text:
                continue
            for child in sub.menu().actions():
                if child.text() == action_text:
                    return child
    raise AssertionError(f"action not found: {menu_text}/{submenu_text}/{action_text}")


def test_display_aids_menu_exists_and_defaults_off(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        grid = _submenu_action(win, "View", "Display Aids", _GRID)
        clip = _submenu_action(win, "View", "Display Aids", _CLIP)
        assert grid.isCheckable()
        assert clip.isCheckable()
        assert not grid.isChecked()
        assert not clip.isChecked()
        assert win.app_settings.wgpu_pixel_grid is False
        assert win.app_settings.wgpu_clip_indicator is False
    finally:
        win.close()


def test_toggling_display_aids_updates_and_persists_settings(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "View", "Display Aids", _GRID).trigger()
        _submenu_action(win, "View", "Display Aids", _CLIP).trigger()
        _process_events(qtbot)
        assert win.app_settings.wgpu_pixel_grid is True
        assert win.app_settings.wgpu_clip_indicator is True
        assert _submenu_action(win, "View", "Display Aids", _GRID).isChecked()
    finally:
        win.close()

    # A newly opened window restores the persisted toggles from QSettings.
    win2 = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win2)
    try:
        _process_events(qtbot)
        assert win2.app_settings.wgpu_pixel_grid is True
        assert win2.app_settings.wgpu_clip_indicator is True
        assert _submenu_action(win2, "View", "Display Aids", _GRID).isChecked()
        assert _submenu_action(win2, "View", "Display Aids", _CLIP).isChecked()
    finally:
        win2.close()
