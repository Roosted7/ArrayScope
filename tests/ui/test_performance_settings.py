import numpy as np

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings, process_events as _process_events


def _menu(win, text):
    for action in win.menuBar().actions():
        if action.text() == text:
            return action.menu()
    raise AssertionError(f"menu not found: {text}")


def _submenu_action(win, menu_text, submenu_text, action_text):
    menu = _menu(win, menu_text)
    for action in menu.actions():
        if action.text() == submenu_text:
            submenu = action.menu()
            for child in submenu.actions():
                if child.text() == action_text:
                    return child
    raise AssertionError(f"action not found: {menu_text}/{submenu_text}/{action_text}")


def test_performance_menu_exists(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        assert _menu(win, "Performance") is not None
        assert _submenu_action(win, "Performance", "Memory Profile", "Balanced") is not None
    finally:
        win.close()


def test_selecting_fft_workers_updates_settings(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import FFTWorkersChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "FFT Workers", "2").trigger()
        _process_events(qtbot)
        assert win.app_settings.fft_workers == FFTWorkersChoice.TWO
    finally:
        win.close()


def test_render_memory_budget_persists_through_settings(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow
    from arrayscope.app.settings_state import MemoryProfileChoice

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "Render Memory Budget", "1024 MiB").trigger()
        _submenu_action(win, "Performance", "Memory Profile", "Custom").trigger()
        _process_events(qtbot)
        assert win.app_settings.render_memory_budget_mb == 1024
        assert win.app_settings.memory_profile == MemoryProfileChoice.CUSTOM
        assert win._memory_policy().visible_render_budget_bytes == 1024 * 1024 * 1024
    finally:
        win.close()

    second = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(second)
    try:
        _process_events(qtbot)
        assert second.app_settings.render_memory_budget_mb == 1024
        assert second.app_settings.memory_profile == MemoryProfileChoice.CUSTOM
    finally:
        second.close()


def test_selecting_pyfftw_backend_does_not_crash(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import FFTBackendChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "FFT Backend", "pyFFTW").trigger()
        _process_events(qtbot)
        assert win.app_settings.fft_backend == FFTBackendChoice.PYFFTW
    finally:
        win.close()


def test_selecting_memory_profile_recomputes_policy(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import MemoryProfileChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        _submenu_action(win, "Performance", "Memory Profile", "Conservative").trigger()
        _process_events(qtbot)

        assert win.app_settings.memory_profile == MemoryProfileChoice.CONSERVATIVE
        assert win._memory_policy().profile == MemoryProfileChoice.CONSERVATIVE
    finally:
        win.close()
