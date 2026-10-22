"""Ring 1: transfer editable pipelines through the existing recipe protocol."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations import library, recipes
from arrayscope.operations.pipeline import Crop, OperationStep, ReverseAxis
from tests.ui.helpers import clear_arrayscope_settings, settle_idle


@pytest.fixture
def window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    clipboard = QtWidgets.QApplication.clipboard()
    previous_clipboard = clipboard.text()
    steps = (OperationStep(Crop(1, 1, 5)), OperationStep(ReverseAxis(0), enabled=False))
    win = ArrayScopeWindow(np.arange(24).reshape(4, 6), steps=steps)
    qtbot.addWidget(win)
    win.show()
    settle_idle(win)
    yield win
    clipboard.setText(previous_clipboard)


def test_recipe_copy_menu_and_keyboard_palette_paste_preserve_the_editable_stack(
    window, qtbot, monkeypatch
):
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    original = window.document.steps
    menu = window.operation_dock.more_button.menu()
    copy_action = next(
        action for action in menu.actions() if action.text() == "Copy Operation Recipe to Clipboard"
    )
    copy_action.trigger()
    copied = QtWidgets.QApplication.clipboard().text()
    assert json.loads(copied) == recipes.recipe_from_steps(original)
    assert window.findChild(QtGui.QAction, "PasteOperationRecipeAction") is not None
    window.clear_operations()
    settle_idle(window)
    selected = []

    def choose_paste():
        from arrayscope.ui.command_palette import CommandPaletteDialog

        palette = QtWidgets.QApplication.activeModalWidget()
        if not isinstance(palette, CommandPaletteDialog):
            selected.append(None)
            return
        try:
            qtbot.keyClicks(palette.search, "paste operation recipe")
            command, _axis = palette.selected()
            selected.append(None if command is None else command.id)
            qtbot.keyPress(palette.search, QtCore.Qt.Key.Key_Return)
        finally:
            # A broken search/accept path must fail this test without leaving
            # the production modal loop open for the process watchdog.
            if palette.isVisible():
                palette.reject()

    window.activateWindow()
    window.setFocus()
    QtCore.QTimer.singleShot(0, window, choose_paste)
    qtbot.keyClick(window, QtCore.Qt.Key.Key_K, modifier=QtCore.Qt.KeyboardModifier.ControlModifier)

    assert selected == ["paste_recipe"]
    assert not warnings
    assert recipes.recipe_from_steps(window.document.steps) == recipes.recipe_from_steps(original)
    assert tuple(window.document.current_shape) == (4, 4)
    settle_idle(window)


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        '{"version": 2, "operations": [{"id": "crop", "axis": 99}]}',
    ],
    ids=["not-a-recipe", "incompatible-axis"],
)
def test_invalid_clipboard_recipe_keeps_the_current_document(window, monkeypatch, text):
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )
    previous = window.document
    QtWidgets.QApplication.clipboard().setText(text)

    assert window.paste_operation_recipe() is False

    assert window.document is previous
    assert window.operation_coordinator.document is previous
    assert warnings
    assert warnings[0][0] == "Recipe Paste Error"


def test_rank_increasing_clipboard_recipe_uses_the_existing_new_window_handoff(
    window, qtbot, monkeypatch
):
    from arrayscope.operations.pipeline import SplitAxis
    from arrayscope.window import ArrayScopeWindow

    before = set(QtWidgets.QApplication.topLevelWidgets())
    base_data = window.base_data
    QtWidgets.QApplication.clipboard().setText(recipes.dumps_recipe((SplitAxis(1, 2),)))

    def refuse_origin_render(*_args, **_kwargs):
        pytest.fail("recipe import rendered the origin after handing it off")

    monkeypatch.setattr(window, "render", refuse_origin_render)

    assert window.paste_operation_recipe() is True

    targets = [
        widget
        for widget in QtWidgets.QApplication.topLevelWidgets()
        if widget not in before and isinstance(widget, ArrayScopeWindow)
    ]
    assert len(targets) == 1
    target = targets[0]
    qtbot.addWidget(target)
    assert target.base_data is base_data
    assert tuple(target.document.current_shape) == (4, 3, 2)
    settle_idle(target)


def test_clipboard_commands_require_the_same_review_as_file_recipes(window, tmp_path, monkeypatch):
    directory = tmp_path / "operations"
    directory.mkdir()
    original_directory = library.user_operations_directory
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(directory))
    wrapper_path = directory / "command.json"
    wrapper_path.write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:clipboard_command",
                "label": "Clipboard command",
                "runtime": "command",
                "command_template": f'"{sys.executable}" {{in}} {{out}}',
                "handoff": "npy",
            }
        )
    )
    library.refresh_user_operations()
    text = json.dumps(
        {"version": 2, "operations": [{"id": "user:clipboard_command", "enabled": True}]}
    )
    try:
        QtWidgets.QApplication.clipboard().setText(text)
        assert window.paste_operation_recipe() is True
        pasted = window.document.steps
        assert len(pasted) == 1
        assert not pasted[0].enabled
        assert json.loads(wrapper_path.read_text())["review"]["required"] is True

        assert library.review_user_operation("user:clipboard_command") is True
        recipe_path = tmp_path / "imported-recipe.json"
        recipe_path.write_text(text)
        assert window.load_operation_recipe_from_path(str(recipe_path)) is True
        loaded = window.document.steps
        assert len(loaded) == 1
        assert not loaded[0].enabled
        assert loaded[0].unavailable_reason == pasted[0].unavailable_reason
        assert json.loads(wrapper_path.read_text())["review"]["required"] is True
    finally:
        monkeypatch.setattr(library, "user_operations_directory", original_directory)
        library.refresh_user_operations()
