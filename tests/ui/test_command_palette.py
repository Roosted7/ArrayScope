"""Focused interaction coverage for the operation catalogue's search owner."""

from __future__ import annotations

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore

from arrayscope.ui.command_palette import CommandPaletteDialog, PaletteCommand


def test_search_routes_arrows_to_enabled_results_and_enter_accepts(qtbot):
    commands = [
        PaletteCommand("alpha", "Operation alpha", kind="operation"),
        PaletteCommand("beta", "Operation beta", kind="operation", enabled=False),
        PaletteCommand("gamma", "Operation gamma", kind="operation"),
    ]
    dialog = CommandPaletteDialog(commands)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.search.setFocus()
    qtbot.waitUntil(dialog.search.hasFocus)

    qtbot.keyClicks(dialog.search, "no match")
    assert dialog.list_widget.item(0).text() == "No matching commands or operations."
    assert not dialog.axis_label.isVisibleTo(dialog)
    assert not dialog.axis_combo.isVisibleTo(dialog)

    dialog.search.clear()
    qtbot.keyClicks(dialog.search, "operation")
    assert dialog.selected()[0].id == "alpha"
    # The disabled middle result must not trap keyboard navigation.
    qtbot.keyPress(dialog.search, QtCore.Qt.Key.Key_Down)
    assert dialog.search.hasFocus()
    assert dialog.selected()[0].id == "gamma"

    qtbot.keyPress(dialog.search, QtCore.Qt.Key.Key_Return)
    assert dialog.result() == dialog.DialogCode.Accepted
