import numpy as np
from dataclasses import replace

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
        assert dialog.minimumWidth() <= 480
        assert dialog.width() <= 560
        assert dialog.tabs.tabText(0) == "Realtime"
        assert {dialog.tabs.tabText(index) for index in range(dialog.tabs.count())} == {
            "Realtime",
            "Feedback",
            "Memory",
            "Caches",
            "Schedulers",
            "Render",
            "Canvas Preserve",
            "Montage",
            "Compute",
            "FFT",
            "Operations",
            "All",
        }
        dialog.tabs.setCurrentWidget(dialog._section_edits["All"])
        text = dialog.text_edit.toPlainText()
        for heading in ("Realtime", "Feedback", "Memory", "Caches", "Schedulers", "Render", "Canvas Preserve", "Montage", "Compute", "FFT", "Operations"):
            assert heading in text
        assert dialog.refresh_button.isCheckable()
        assert dialog.refresh_button.isChecked()
        assert dialog._overview_labels["status"].text()
        assert "RSS" in dialog._overview_labels["resources"].text()
        assert "sync" in dialog._overview_labels["render"].text()
        assert dialog._overview_labels["montage"].text()
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


def test_diagnostics_reports_actual_image_backend_separately_from_setting(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.app.settings_state import ImageRenderingBackendChoice
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.app_settings = replace(win.app_settings, image_rendering_backend=ImageRenderingBackendChoice.VISPY)
        snapshot = win.collect_runtime_diagnostics()

        assert snapshot.image_rendering_backend_selected == "vispy"
        assert snapshot.image_rendering_backend_actual == "pyqtgraph"
        assert snapshot.image_rendering_backend == "pyqtgraph"
    finally:
        win.close()


def test_diagnostics_operations_tab_shows_region_planner_details(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5, 6), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win.operation_evaluator.image(win.view_state.with_slice(2, 2))
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        dialog.tabs.setCurrentWidget(dialog._section_edits["Operations"])
        dialog.refresh(force_text=True)
        text = dialog.current_text_edit().toPlainText()

        for expected in ("Final region", "Required input", "Expanded axes", "Transitions", "Stage cache candidates"):
            assert expected in text
        assert "CenteredFFT" in text
    finally:
        win.close()


def test_diagnostics_operations_tab_shows_optimizer_summaries(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5, 6), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2), CenteredIFFT(axis=2)))
        win._set_document(win.operation_coordinator.document)
        win.operation_evaluator.image(win.view_state.with_slice(2, 2))
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        dialog.tabs.setCurrentWidget(dialog._section_edits["Operations"])
        dialog.refresh(force_text=True)
        text = dialog.current_text_edit().toPlainText()

        assert "Count: 2" in text
        assert "Optimized count: 1" in text
        assert "Optimizations:" in text
        assert "removed CenteredFFT/CenteredIFFT axis 2" in text
    finally:
        win.close()


def test_diagnostics_stage_cache_bar_and_cache_text_update(qtbot):
    _clear_arrayscope_settings()
    from arrayscope.operations.pipeline import CenteredFFT
    from arrayscope.window import ArrayScopeWindow

    win = ArrayScopeWindow(np.zeros((4, 5, 6), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.operation_coordinator.load_operations((CenteredFFT(axis=2),))
        win._set_document(win.operation_coordinator.document)
        win.operation_evaluator.image(win.view_state.with_slice(2, 0))
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        dialog.refresh(force_text=True)

        dialog.tabs.setCurrentWidget(dialog._section_edits["Caches"])
        dialog.refresh(force_text=True)
        cache_text = dialog.current_text_edit().toPlainText()
        assert "Stage cache:" in cache_text
        assert "entries=1" in cache_text
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
        assert "RSS" in dialog._overview_labels["resources"].text()

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
