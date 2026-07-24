import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtWidgets

from tests.ui.helpers import clear_arrayscope_settings, process_events

_AXIS_AND_PLAIN_SRC = '''
import numpy as np


def shift(data, axis, amount: int = 1):
    """Roll one axis by amount."""
    return np.roll(data, amount, axis=axis)


def double(data):
    """Double every sample."""
    return data * 2
'''


@pytest.fixture(autouse=True)
def _isolated_ops_dir(tmp_path, monkeypatch):
    from arrayscope.app import user_dirs
    from arrayscope.operations import library

    ops_dir = tmp_path / "operations"
    monkeypatch.setattr(library, "user_operations_directory", lambda: str(ops_dir))
    monkeypatch.setattr(user_dirs, "user_operations_directory", lambda: ops_dir)
    library.refresh_user_operations()
    library.reset_layout()
    for op_id in tuple(library.hidden_operations()):
        library.set_operation_hidden(op_id, False)
    yield
    library.refresh_user_operations()


def _window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    win.show()
    process_events(qtbot)
    return win


def _manager(qtbot, win):
    from arrayscope.ui.operation_manager import OperationManagerDialog

    dialog = OperationManagerDialog(win)
    qtbot.addWidget(dialog)
    dialog.show()
    process_events(qtbot)
    return dialog


def _write_source(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(body)
    return str(path)


def test_tree_shows_system_ops_and_hide_unhide(qtbot):
    from arrayscope.operations import library

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        # Groups are present with system ops beneath them.
        groups = {
            dialog.tree.topLevelItem(i).data(0, 0x0101)
            for i in range(dialog.tree.topLevelItemCount())
        }
        assert "Reduce" in groups
        assert dialog.select_operation("mean")

        dialog.remove_button.click()
        process_events(qtbot)
        assert "mean" in library.hidden_operations()

        # Still listed (hidden) so it can be restored; row is marked.
        assert dialog.select_operation("mean")
        item = dialog.tree.currentItem()
        assert "(hidden)" in item.text(0)

        dialog.unhide_button.click()
        process_events(qtbot)
        assert "mean" not in library.hidden_operations()
    finally:
        dialog.close()
        win.close()


def test_drag_persist_moves_op_between_groups(qtbot):
    from arrayscope.operations import library

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        # Simulate a drop: move "mean" out of Reduce into Complex.
        tree = dialog.tree
        mean_item = None
        complex_group = None
        for gi in range(tree.topLevelItemCount()):
            group_item = tree.topLevelItem(gi)
            if group_item.data(0, 0x0101) == "Complex":
                complex_group = group_item
            for ci in range(group_item.childCount()):
                if group_item.child(ci).data(0, 0x0100) == "mean":
                    mean_item = group_item.takeChild(ci)
                    break
        assert mean_item is not None
        assert complex_group is not None
        complex_group.insertChild(0, mean_item)
        dialog._persist_layout()
        process_events(qtbot)

        groups = dict(library.grouped_operations())
        assert "mean" in {entry.id for entry in groups["Complex"]}
    finally:
        dialog.close()
        win.close()


def test_import_flow_end_to_end(qtbot, tmp_path, monkeypatch):
    from arrayscope.operations import library
    from arrayscope.operations.registry import get_operation_entry
    from arrayscope.ui import operation_manager

    src = _write_source(tmp_path, "ops.py", _AXIS_AND_PLAIN_SRC)

    # The import panel auto-fills from introspection of the first function.
    panel = operation_manager._OperationImportDialog(
        None, src, library.introspect_python_source(src), ["User", "Reduce"]
    )
    qtbot.addWidget(panel)
    assert panel.function_combo.currentText() == "shift"
    assert panel.requires_axis_check.isChecked() is True
    assert panel.description_edit.text() == "Roll one axis by amount."
    assert panel.params_table.rowCount() == 1
    assert panel.params_table.item(0, 0).text() == "amount"
    panel.close()

    # Drive the manager's Add flow: choose the file, accept the panel unchanged.
    monkeypatch.setattr(operation_manager, "get_open_file_name", lambda *a, **k: (src, ""))
    monkeypatch.setattr(
        operation_manager._OperationImportDialog,
        "exec",
        lambda self: QtWidgets.QDialog.DialogCode.Accepted,
    )

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        dialog.add_button.click()
        process_events(qtbot)

        op_id = "user:shift"
        assert op_id in {entry.id for entry in library.grouped_operations()[0][1]} or any(
            op_id == e.id for _g, es in library.grouped_operations() for e in es
        )
        # An imported op copies its code file into the ops directory.
        ops_dir = library.user_operations_directory()
        assert os.path.exists(os.path.join(ops_dir, "shift.py"))

        # Editing the Label via the editor writes through update_user_operation.
        assert dialog.select_operation(op_id)
        dialog.label_edit.setText("Rolled")
        dialog.label_edit.editingFinished.emit()
        process_events(qtbot)
        assert get_operation_entry(op_id).label == "Rolled"
    finally:
        dialog.close()
        win.close()


def test_link_mode_stores_absolute_path(qtbot, tmp_path, monkeypatch):
    from arrayscope.operations import library
    from arrayscope.ui import operation_manager

    src = _write_source(tmp_path, "linked.py", "def scale(data):\n    return data * 3\n")

    monkeypatch.setattr(operation_manager, "get_open_file_name", lambda *a, **k: (src, ""))

    def _accept_link(self):
        self.link_radio.setChecked(True)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(operation_manager._OperationImportDialog, "exec", _accept_link)

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        dialog.add_button.click()
        process_events(qtbot)
        stored = operation_manager._user_op_source_path("user:scale")
        assert stored == os.path.abspath(src)
        # No copy was made into the ops directory.
        assert not os.path.exists(os.path.join(library.user_operations_directory(), "scale.py"))
    finally:
        dialog.close()
        win.close()


def test_reset_all_clears_layout_and_unhides(qtbot, monkeypatch):
    from arrayscope.operations import library

    library.set_operation_hidden("mean", True)
    library.apply_library_layout(group_order=["Complex", "Reduce"])

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        dialog.reset_all_button.click()
        process_events(qtbot)
        assert "mean" not in library.hidden_operations()
        # Layout is back to the default taxonomy order.
        order = [group for group, _entries in library.grouped_operations()]
        assert order.index("Reduce") < order.index("Complex")
    finally:
        dialog.close()
        win.close()


def test_menu_entries_open_dialogs(qtbot):
    win = _window(qtbot)
    try:
        colormap_action = None
        operation_action = None
        for menu_action in win.menuBar().actions():
            if menu_action.text() != "View":
                continue
            for sub in menu_action.menu().actions():
                if sub.text() == "Colormap manager…":
                    colormap_action = sub
                elif sub.text() == "Operation manager…":
                    operation_action = sub
        assert colormap_action is not None
        assert operation_action is not None

        operation_action.trigger()
        process_events(qtbot)
        assert win._operation_manager_dialog is not None
        assert win._operation_manager_dialog.isVisible()
        win._operation_manager_dialog.close()

        colormap_action.trigger()
        process_events(qtbot)
        assert win._colormap_designer_dialog is not None
        win._colormap_designer_dialog.close()
    finally:
        win.close()


def test_problems_group_appears_for_broken_wrapper(qtbot, tmp_path):
    import json

    from arrayscope.operations import library

    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "broken.py").write_text("def oops(data)\n    return data\n")  # syntax error
    (ops_dir / "broken.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:broken",
                "source": {"mode": "import", "path": "broken.py", "callable": "oops"},
            }
        )
    )
    library.refresh_user_operations()

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        groups = [
            dialog.tree.topLevelItem(i).data(0, 0x0101)
            for i in range(dialog.tree.topLevelItemCount())
        ]
        assert "Problems" in groups
    finally:
        dialog.close()
        win.close()
