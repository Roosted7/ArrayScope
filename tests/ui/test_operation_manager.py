import os
import sys

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


def test_new_and_source_picker_stay_in_manager(qtbot, tmp_path, monkeypatch):
    from arrayscope.operations import library
    from arrayscope.operations.registry import get_operation_entry
    from arrayscope.ui import operation_manager

    src = _write_source(tmp_path, "ops.py", _AXIS_AND_PLAIN_SRC)
    monkeypatch.setattr(operation_manager, "get_open_file_name", lambda *a, **k: (src, ""))

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert not hasattr(operation_manager, "_OperationImportDialog")
        dialog.new_button.click()
        process_events(qtbot)
        new_id = dialog.selected_operation_id()
        assert new_id == "user:new_operation"
        assert dialog.label_edit.text() == "New operation"
        assert dialog.source_box.isVisible()

        dialog.source_browse_button.click()
        process_events(qtbot)

        # AST inference fills ordinary editable fields in this same manager.
        assert dialog.callable_combo.currentText() == "shift"
        assert [
            dialog.callable_combo.itemText(i) for i in range(dialog.callable_combo.count())
        ] == [
            "shift",
            "double",
        ]
        assert dialog.requires_axis_check.isChecked() is True
        assert dialog.description_edit.text() == "Roll one axis by amount."
        assert dialog.params_table.rowCount() == 1
        assert dialog.params_table.columnCount() == 7
        assert dialog.params_table.item(0, 0).text() == "amount"
        assert get_operation_entry(new_id).label == "Shift"

        # Copy mode owns an independent source file under the existing entry id.
        ops_dir = library.user_operations_directory()
        assert os.path.exists(os.path.join(ops_dir, "new_operation.py"))

        # Inferred values remain ordinary editable values.
        dialog.label_edit.setText("Rolled")
        dialog.label_edit.editingFinished.emit()
        process_events(qtbot)
        assert get_operation_entry(new_id).label == "Rolled"
    finally:
        dialog.close()
        win.close()


def test_link_mode_stores_absolute_path(qtbot, tmp_path, monkeypatch):
    from arrayscope.operations import library
    from arrayscope.ui import operation_manager

    src = _write_source(tmp_path, "linked.py", "def scale(data):\n    return data * 3\n")

    monkeypatch.setattr(operation_manager, "get_open_file_name", lambda *a, **k: (src, ""))

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        dialog.new_button.click()
        process_events(qtbot)
        operation_id = dialog.selected_operation_id()
        dialog.link_radio.setChecked(True)
        dialog.source_browse_button.click()
        process_events(qtbot)
        stored = library.user_operation_source_path(operation_id)
        assert stored == os.path.abspath(src)
        wrapper = library.user_operation_wrapper(operation_id)
        assert wrapper["source"]["mode"] == "link"
    finally:
        dialog.close()
        win.close()


def test_duplicate_system_selection_becomes_editable_user_copy(qtbot):
    from arrayscope.operations import library, registry

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_operation("conjugate")
        assert dialog.label_edit.isEnabled() is False
        assert dialog.duplicate_button.isEnabled() is True
        assert "Duplicate" in dialog.status_label.text()

        dialog.duplicate_button.click()
        process_events(qtbot)

        operation_id = dialog.selected_operation_id()
        assert operation_id.startswith("user:")
        assert dialog.label_edit.isEnabled() is True
        assert dialog.params_table.isEnabled() is True
        assert registry.get_operation_entry(operation_id).label == "Conjugate copy"
        data = np.array([1 + 2j], dtype=np.complex64)
        assert np.array_equal(
            registry.create_operation(operation_id).apply(data),
            np.conjugate(data),
        )
        assert library.user_operation_wrapper(operation_id)["template"]["kind"] == "native-copy"
    finally:
        dialog.close()
        win.close()


def test_parameter_metadata_parity_and_default_layout(qtbot, tmp_path):
    from arrayscope.operations import library, registry

    src = _write_source(
        tmp_path,
        "bounded.py",
        "def bounded(data, gain: float = 0.5):\n    return data * gain\n",
    )
    operation_id = library.import_custom_operation(src, "bounded")
    library.update_user_operation(
        operation_id,
        parameters=[
            {
                "name": "gain",
                "label": "Gain",
                "kind": "float",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0,
                "step": 0.05,
                "description": "Scale factor.",
            }
        ],
    )

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_operation(operation_id)
        process_events(qtbot)
        assert dialog.params_table.columnCount() == 7
        assert dialog.params_table.item(0, 3).text() == "0.0"
        assert dialog.params_table.item(0, 4).text() == "1.0"
        assert dialog.params_table.item(0, 5).text() == "0.05"
        assert dialog.params_table.item(0, 6).text() == "Scale factor."
        assert dialog.params_table.minimumHeight() >= 200
        assert dialog.params_table.horizontalScrollBar().maximum() == 0

        dialog.params_table.item(0, 6).setText("Editable help.")
        process_events(qtbot)
        assert (
            registry.get_operation_entry(operation_id).parameters[0].description == "Editable help."
        )
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


def test_command_runtime_and_advanced_environment_editor_share_the_manager(qtbot, tmp_path):
    from arrayscope.operations import library, registry

    executable = tmp_path / "copy-array"
    executable.write_text(
        "#!/usr/bin/env python3\nimport shutil, sys\nshutil.copyfile(sys.argv[-2], sys.argv[-1])\n"
    )
    executable.chmod(0o755)

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        dialog.new_button.click()
        process_events(qtbot)
        operation_id = dialog.selected_operation_id()

        dialog.runtime_combo.setCurrentIndex(dialog.runtime_combo.findData("command"))
        dialog.command_template_edit.setText(f'"{executable}" {{in}} {{out}}')
        dialog.command_template_edit.editingFinished.emit()
        process_events(qtbot)

        wrapper = library.user_operation_wrapper(operation_id)
        assert wrapper["runtime"] == "command"
        assert wrapper["command_template"].endswith("{in} {out}")
        assert dialog.source_box.isHidden()
        assert dialog.command_box.isVisible()
        assert registry.get_operation_entry(operation_id).unavailable_reason == ""

        assert not dialog.advanced_panel.isVisible()
        dialog.advanced_button.setChecked(True)
        process_events(qtbot)
        assert dialog.advanced_panel.isVisible()
        dialog.environment_id_edit.setText("recon")
        dialog.environment_name_edit.setText("Recon")
        dialog.environment_kind_combo.setCurrentIndex(
            dialog.environment_kind_combo.findData("interpreter")
        )
        dialog.environment_locator_edit.setText(sys.executable)
        dialog.environment_variables_edit.setPlainText(
            "BART_TOOLBOX_PATH=/opt/bart\nRECON_MODE=test"
        )
        dialog.environment_save_button.click()
        process_events(qtbot)

        records = library.execution_environments()
        assert len(records) == 1
        assert records[0].id == "recon"
        assert dict(records[0].variables)["BART_TOOLBOX_PATH"] == "/opt/bart"
    finally:
        dialog.close()
        win.close()
