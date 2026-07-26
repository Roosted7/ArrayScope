from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.operations import library
from arrayscope.ui.command_palette import CommandPaletteDialog, PaletteCommand
from arrayscope.ui.operation_add_popup import OperationAddPopup
from arrayscope.ui.operation_listing import build_operation_listing
from tests.ui.helpers import clear_arrayscope_settings, process_events


@pytest.fixture(autouse=True)
def _isolated_ops_dir(tmp_path, monkeypatch):
    from arrayscope.app import user_dirs

    ops_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(ops_dir))
    monkeypatch.setattr(user_dirs, "user_operations_directory", lambda: ops_dir)
    library.refresh_user_operations()
    yield
    library.refresh_user_operations()


def test_unavailable_reason_disables_add_popup_chip_menu_and_palette(qtbot):
    from arrayscope.operations.registry import get_operation_entry
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    operation_id = library.create_empty_user_operation()
    entry = get_operation_entry(operation_id)
    reason = entry.unavailable_reason
    assert reason

    popup = OperationAddPopup(
        build_operation_listing(),
        on_accept=lambda *_args: None,
        on_needs_parameters=lambda *_args: None,
    )
    qtbot.addWidget(popup)
    assert not popup.select_operation(operation_id)
    popup.set_expanded(True)
    popup_item = next(
        popup._list.item(row)
        for row in range(popup._list.count())
        if popup._list.item(row).data(QtCore.Qt.ItemDataRole.UserRole + 1) == operation_id
    )
    assert popup_item.toolTip() == reason

    win = ArrayScopeWindow(np.ones((3, 4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    win.show()
    process_events(qtbot)
    menu = win._build_operation_context_menu(0, QtCore.QPoint(0, 0))
    actions = [
        action
        for candidate in (menu, *menu.findChildren(type(menu)))
        for action in candidate.actions()
    ]
    menu_action = next(action for action in actions if action.data() == operation_id)
    assert not menu_action.isEnabled()
    assert menu_action.toolTip() == reason

    palette = CommandPaletteDialog(
        [
            PaletteCommand(
                operation_id,
                entry.label,
                kind="operation",
                enabled=False,
                unavailable_reason=reason,
            )
        ]
    )
    qtbot.addWidget(palette)
    item = palette.list_widget.item(0)
    assert not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
    assert item.toolTip() == reason
    assert palette.selected() == (None, None)
    win.close()


def test_window_palette_uses_library_listing_so_hidden_operations_stay_hidden(
    qtbot,
    monkeypatch,
):
    from arrayscope.window import ArrayScopeWindow, operation_actions

    clear_arrayscope_settings()
    captured = []

    class CapturingPalette:
        def __init__(self, commands, **_kwargs):
            captured.extend(commands)

        def exec(self):
            return QtWidgets.QDialog.DialogCode.Rejected

    monkeypatch.setattr(operation_actions, "CommandPaletteDialog", CapturingPalette)
    win = ArrayScopeWindow(np.ones((3, 4, 5), dtype=np.float32))
    qtbot.addWidget(win)
    try:
        library.set_operation_hidden("reverse", True)
        win.open_command_palette()
        assert "reverse" not in {command.id for command in captured}
    finally:
        library.set_operation_hidden("reverse", False)
        win.close()
