"""View > Display Server menu (Linux/Wayland-gated qt_platform setting)."""

import numpy as np

from tests.ui.helpers import (
    clear_arrayscope_settings as _clear_arrayscope_settings,
)
from tests.ui.helpers import (
    process_events as _process_events,
)


def _view_submenu(win, submenu_text):
    for action in win.menuBar().actions():
        if action.text() == "View":
            for child in action.menu().actions():
                if child.text() == submenu_text:
                    return child.menu()
    return None


def _make_window(monkeypatch, wayland_session):
    import arrayscope.app.qt_platform as qt_platform

    if wayland_session:
        monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
        monkeypatch.setattr(qt_platform.sys, "platform", "linux", raising=False)
    else:
        monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    from arrayscope.window import ArrayScopeWindow

    return ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))


def test_display_server_menu_present_on_wayland_sessions(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    win = _make_window(monkeypatch, wayland_session=True)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        menu = _view_submenu(win, "Display Server")
        assert menu is not None
        labels = [a.text() for a in menu.actions() if a.text()]
        assert "Auto (Wayland, X11 on early crash)" in labels
        assert "Force Wayland" in labels
        assert "Force X11 (XWayland)" in labels
        assert any(label.startswith("Active now:") for label in labels)
    finally:
        win.close()


def test_display_server_menu_absent_off_wayland(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    win = _make_window(monkeypatch, wayland_session=False)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert _view_submenu(win, "Display Server") is None
    finally:
        win.close()


def test_selecting_platform_persists_setting(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.app.qt_platform import QtPlatformChoice

    win = _make_window(monkeypatch, wayland_session=True)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        menu = _view_submenu(win, "Display Server")
        action = next(a for a in menu.actions() if a.text() == "Force X11 (XWayland)")
        action.trigger()
        _process_events(qtbot)
        assert win.app_settings.qt_platform == QtPlatformChoice.XCB
        assert win._settings.value("qt_platform") == "xcb"
        # Round-trips through a fresh settings load.
        assert win._load_app_settings().qt_platform == QtPlatformChoice.XCB
    finally:
        win.close()
