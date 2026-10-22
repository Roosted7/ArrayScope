"""The discoverable selection export and Save shortcut share one action owner."""

import numpy as np
from pyqtgraph.Qt import QtGui

from arrayscope.tools.interaction_budget import INTERACTION_SETTLE_HARD_LIMIT_MS
from tests.settle import settle_window
from tests.ui.helpers import frame_session_settled, use_pyqtgraph_backend


def test_selected_export_menu_and_save_shortcut_share_one_owner(qtbot, monkeypatch):
    from arrayscope.window import ArrayScopeWindow

    calls = []
    monkeypatch.setattr(
        ArrayScopeWindow, "_save_current_numpy_file", lambda self: calls.append(self)
    )
    settings = use_pyqtgraph_backend()
    win = ArrayScopeWindow(np.arange(24).reshape(4, 6))
    qtbot.addWidget(win)
    try:
        win.show()
        settle_window(win, lambda: frame_session_settled(win), label="export menu opens")
        selected = win.findChild(QtGui.QAction, "ExportSelectedArrayAction")
        full = win.findChild(QtGui.QAction, "ExportFullDerivedArrayAction")
        assert selected.text() == "Export Selected Array…"
        assert full.text() == "Export Full Derived Array…"
        assert selected is not full

        selected.trigger()
        win.activateWindow()
        win.setFocus()
        qtbot.waitUntil(win.isActiveWindow, timeout=INTERACTION_SETTLE_HARD_LIMIT_MS)
        combination = QtGui.QKeySequence.keyBindings(QtGui.QKeySequence.StandardKey.Save)[0][0]
        qtbot.keyClick(win, combination.key(), modifier=combination.keyboardModifiers())

        assert calls == [win, win]
    finally:
        win.close()
        settings.clear()
        settings.sync()
