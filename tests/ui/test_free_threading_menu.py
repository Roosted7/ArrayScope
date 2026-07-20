"""Performance > Python Free-Threading menu (free-threaded-build-gated)."""

import numpy as np

from tests.ui.helpers import (
    clear_arrayscope_settings as _clear_arrayscope_settings,
)
from tests.ui.helpers import (
    process_events as _process_events,
)


def _performance_submenu(win, submenu_text):
    for action in win.menuBar().actions():
        if action.text() == "Performance":
            for child in action.menu().actions():
                if child.text() == submenu_text:
                    return child.menu()
    return None


def _make_window(monkeypatch, free_threaded_build, gil_enabled=False):
    import arrayscope.app.free_threading as free_threading

    monkeypatch.setattr(free_threading, "interpreter_is_free_threaded", lambda: free_threaded_build)
    monkeypatch.setattr(free_threading, "gil_currently_enabled", lambda: gil_enabled)
    from arrayscope.window import ArrayScopeWindow

    return ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))


def test_menu_offers_choices_on_free_threaded_builds(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    win = _make_window(monkeypatch, free_threaded_build=True)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        menu = _performance_submenu(win, "Python Free-Threading")
        assert menu is not None
        labels = [a.text() for a in menu.actions() if a.text()]
        assert "Enabled (GIL off)" in labels
        assert "Force-disabled (GIL on)" in labels
        assert "Auto-disabled (crashed shortly after launch)" in labels
        assert "Active now: GIL disabled (free threading)" in labels
        # ENABLED is the persisted default and shows as checked.
        enabled = next(a for a in menu.actions() if a.text() == "Enabled (GIL off)")
        assert enabled.isChecked()
        # The supervisor-owned state is shown but never user-selectable.
        auto = next(
            a for a in menu.actions() if a.text() == "Auto-disabled (crashed shortly after launch)"
        )
        assert not auto.isEnabled()
    finally:
        win.close()


def test_menu_explains_requirement_on_regular_builds(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    win = _make_window(monkeypatch, free_threaded_build=False)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        menu = _performance_submenu(win, "Python Free-Threading")
        assert menu is not None
        labels = [a.text() for a in menu.actions() if a.text()]
        assert labels == ["Requires a free-threaded Python build (e.g. 3.14t)"]
        assert not menu.actions()[0].isEnabled()
    finally:
        win.close()


def test_force_disable_persists_setting(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from arrayscope.app.free_threading import FreeThreadingChoice

    win = _make_window(monkeypatch, free_threaded_build=True)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        menu = _performance_submenu(win, "Python Free-Threading")
        action = next(a for a in menu.actions() if a.text() == "Force-disabled (GIL on)")
        action.trigger()
        _process_events(qtbot)
        assert win.app_settings.python_free_threading is FreeThreadingChoice.FORCE_DISABLED
        assert win._settings.value("python_free_threading") == "force_disabled"
        # Round-trips through a fresh settings load.
        assert win._load_app_settings().python_free_threading is FreeThreadingChoice.FORCE_DISABLED
    finally:
        win.close()


def test_supervisor_written_auto_disable_shows_checked(qtbot, monkeypatch):
    _clear_arrayscope_settings()
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.free_threading import FreeThreadingChoice

    # What the crash supervisor persists from outside the QApplication
    # (written here through the test-scoped QSettings store).
    settings = QtCore.QSettings()
    settings.setValue("python_free_threading", FreeThreadingChoice.AUTO_DISABLED.value)
    settings.sync()
    win = _make_window(monkeypatch, free_threaded_build=True)
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert win.app_settings.python_free_threading is FreeThreadingChoice.AUTO_DISABLED
        menu = _performance_submenu(win, "Python Free-Threading")
        auto = next(
            a for a in menu.actions() if a.text() == "Auto-disabled (crashed shortly after launch)"
        )
        assert auto.isChecked()
        # Selecting Enabled re-arms free threading for the next launch.
        enabled = next(a for a in menu.actions() if a.text() == "Enabled (GIL off)")
        enabled.trigger()
        _process_events(qtbot)
        assert win._settings.value("python_free_threading") == "enabled"
    finally:
        _clear_arrayscope_settings()
        win.close()
