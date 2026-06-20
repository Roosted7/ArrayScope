import json

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


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
        assert dialog.log_button.text() == "Log..."
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


def test_diagnostics_jsonl_logging_writes_start_and_snapshots(qtbot, tmp_path, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.ui.diagnostics as diagnostics_ui
    from arrayscope.window import ArrayScopeWindow

    requested_path = tmp_path / "diagnostics-log"
    written_path = tmp_path / "diagnostics-log.jsonl"
    monkeypatch.setattr(diagnostics_ui, "get_save_file_name", lambda *args, **kwargs: (str(requested_path), "JSON Lines (*.jsonl)"))

    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog

        dialog.log_button.click()

        assert dialog.log_button.text() == "Stop"
        assert dialog._logger is not None
        records = _read_jsonl(written_path)
        assert [record["event"] for record in records] == ["start", "snapshot"]
        assert records[0]["schema_version"] == 1
        assert records[0]["config"]["derived_shape"] == [4, 5]
        assert records[1]["sequence"] == 1
        assert records[1]["diagnostics"]["derived_dtype"] == "float32"

        win.operation_evaluator.image(win.view_state)
        dialog.refresh()
        records = _read_jsonl(written_path)
        assert [record["event"] for record in records] == ["start", "snapshot", "snapshot"]
        assert records[-1]["sequence"] == 2
        assert records[-1]["diagnostics"]["image_cache"]["entries"] == 1

        dialog.log_button.click()
        assert dialog.log_button.text() == "Log..."
        assert dialog._logger is None
        dialog.refresh()
        assert len(_read_jsonl(written_path)) == 3
    finally:
        win.close()


def test_diagnostics_jsonl_logging_cancel_keeps_logging_inactive(qtbot, tmp_path, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.ui.diagnostics as diagnostics_ui
    from arrayscope.window import ArrayScopeWindow

    monkeypatch.setattr(diagnostics_ui, "get_save_file_name", lambda *args, **kwargs: ("", ""))
    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog

        dialog.log_button.click()

        assert dialog.log_button.text() == "Log..."
        assert dialog._logger is None
        assert not list(tmp_path.iterdir())
    finally:
        win.close()


def test_diagnostics_jsonl_logging_stops_on_close(qtbot, tmp_path, monkeypatch):
    _clear_arrayscope_settings()
    import arrayscope.ui.diagnostics as diagnostics_ui
    from arrayscope.window import ArrayScopeWindow

    path = tmp_path / "diagnostics.jsonl"
    monkeypatch.setattr(diagnostics_ui, "get_save_file_name", lambda *args, **kwargs: (str(path), "JSON Lines (*.jsonl)"))
    win = ArrayScopeWindow(np.zeros((4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        _process_events(qtbot)
        win.open_diagnostics_dialog()
        dialog = win._diagnostics_dialog
        dialog.log_button.click()
        assert dialog._logger is not None

        dialog.close()
        _process_events(qtbot)

        assert not dialog._timer.isActive()
        assert dialog._logger is None
        assert dialog.log_button.text() == "Log..."
        assert [record["event"] for record in _read_jsonl(path)] == ["start", "snapshot"]
    finally:
        win.close()
