import numpy as np

from tests.ui.helpers import clear_arrayscope_settings as _clear_arrayscope_settings, process_events as _process_events


def _menu(win, text):
    for action in win.menuBar().actions():
        if action.text() == text:
            return action.menu()
    raise AssertionError(f"menu not found: {text}")


def _action(menu, text):
    for action in menu.actions():
        if action.text() == text:
            return action
    raise AssertionError(f"action not found: {text}")


def test_developer_menu_opens_diagnostics_dialog(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.ui.diagnostics import DiagnosticsDialog
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        developer = _menu(win, "Developer")
        assert _action(developer, "Verify icons") is not None
        _action(developer, "Diagnostics").trigger()
        _process_events(qtbot)

        dialog = getattr(win, "_diagnostics_dialog", None)
        assert isinstance(dialog, DiagnosticsDialog)
        assert dialog.isVisible()
        assert dialog.tabs.tabText(0) == "Memory"
        assert {dialog.tabs.tabText(index) for index in range(dialog.tabs.count())} == {
            "Memory",
            "Caches",
            "Schedulers",
            "Render",
            "Canvas Preserve",
            "Montage",
            "FFT",
            "Operations",
            "All",
        }
        dialog.tabs.setCurrentWidget(dialog._section_edits["All"])
        text = dialog.text_edit.toPlainText()
        for heading in ("Memory", "Caches", "Schedulers", "Render", "Canvas Preserve", "Montage", "FFT", "Operations"):
            assert heading in text
        assert dialog.refresh_button.isCheckable()
        assert dialog.refresh_button.isChecked()
    finally:
        win.close()


def test_opening_diagnostics_twice_reuses_dialog(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        first = win._diagnostics_dialog
        win.open_diagnostics_dialog()

        assert win._diagnostics_dialog is first
        assert not hasattr(getattr(win, "layout_manager", object()), "_diagnostics_dialog")
    finally:
        win.close()


def test_diagnostics_refresh_updates_cache_text(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        dialog.tabs.setCurrentWidget(dialog._section_edits["All"])
        before = dialog.text_edit.toPlainText()
        win.operation_evaluator.image(win.view_state)
        dialog.refresh()
        after = dialog.text_edit.toPlainText()

        assert before != after
        assert "Image:" in after
        assert "entries=1" in after
    finally:
        win.close()


def test_diagnostics_auto_text_toggle_pauses_text_but_not_bars(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        dialog.tabs.setCurrentWidget(dialog._section_edits["All"])
        before = dialog.text_edit.toPlainText()

        dialog.refresh_button.setChecked(False)
        win.operation_evaluator.image(win.view_state)
        dialog.refresh()

        assert dialog.text_edit.toPlainText() == before
        assert "entries=1" in dialog._bars["image"].text()

        dialog.refresh_button.setChecked(True)
        assert "entries=1" in dialog.text_edit.toPlainText()
    finally:
        win.close()


def test_diagnostics_timer_stops_after_close(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        assert dialog._timer.isActive()
        dialog.close()
        _process_events(qtbot)

        assert not dialog._timer.isActive()
    finally:
        win.close()
