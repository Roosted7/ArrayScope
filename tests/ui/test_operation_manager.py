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


def test_duplicate_system_selection_transitions_to_editable_user_copy(qtbot):
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
        assert dialog.label_edit.text() == "Conjugate copy"
        # Generated-source contents and execution parity belong to the library
        # contract; this UI test pins the read-only -> editable transition.
        assert dialog.source_box.isEnabled() is True
    finally:
        dialog.close()
        win.close()


def test_parameter_and_input_slot_tables_preserve_editable_schema(qtbot, tmp_path):
    from arrayscope.operations import library, registry

    bounded_src = _write_source(
        tmp_path,
        "bounded.py",
        "def bounded(data, gain: float = 0.5):\n    return data * gain\n",
    )
    bounded_id = library.import_custom_operation(bounded_src, "bounded")
    library.update_user_operation(
        bounded_id,
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

    blend_src = _write_source(
        tmp_path,
        "blend.py",
        "def blend(data, reference):\n    return data + reference\n",
    )
    blend_id = library.import_custom_operation(blend_src, "blend")
    library.update_user_operation(
        blend_id,
        parameters=[],
        input_slots=[
            {
                "name": "reference",
                "label": "Reference",
                "description": "A second array.",
                "accepts": ["dimension-set", "open-document", "saved-array"],
            }
        ],
    )

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        # Bounds/help and slot accepts/labels are definition metadata, not just
        # display text. Reconstructing either editor must preserve and write
        # back the full schema rather than only its happy-path name/default.
        assert dialog.select_operation(bounded_id)
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
            registry.get_operation_entry(bounded_id).parameters[0].description == "Editable help."
        )

        assert dialog.select_operation(blend_id)
        process_events(qtbot)
        assert dialog.slots_table.rowCount() == 1
        assert dialog.slots_table.item(0, 0).text() == "reference"
        assert "open-document" in dialog.slots_table.item(0, 2).text()

        dialog.slots_table.item(0, 1).setText("Reference data")
        process_events(qtbot)
        assert registry.get_operation_entry(blend_id).input_slots[0].label == "Reference data"
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


# Qt.ItemDataRole.UserRole + 3 -- the manager's problem-path role, in the same
# raw-number style the tests above use for the id (0x0100) and group (0x0101).
_PROBLEM_PATH_ROLE = 0x0103


def _select_problem_row(dialog, basename):
    for index in range(dialog.tree.topLevelItemCount()):
        group = dialog.tree.topLevelItem(index)
        if group.data(0, 0x0101) != "Problems":
            continue
        for child_index in range(group.childCount()):
            child = group.child(child_index)
            path = child.data(0, _PROBLEM_PATH_ROLE)
            if path is not None and os.path.basename(str(path)) == basename:
                dialog.tree.setCurrentItem(child)
                return str(path)
    return None


def test_problems_group_appears_for_broken_wrapper(qtbot, tmp_path, monkeypatch):
    import json

    from pyqtgraph.Qt import QtGui

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
                "label": "Broken",
                "source": {"mode": "import", "path": "broken.py", "callable": "oops"},
                "parameters": [{"name": "amount", "kind": "int", "default": 2}],
                "input_slots": [
                    {"name": "reference", "accepts": ["open-document"], "description": "A second."}
                ],
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

        # Appearing is not the point -- being actionable is. The row selects,
        # reports the loader's message, and puts the ordinary form to work on
        # the raw payload instead of blanking the editor.
        path = _select_problem_row(dialog, "broken.json")
        assert path is not None
        process_events(qtbot)
        assert dialog.selected_operation_id() is None
        assert dialog.selected_problem_path() == path
        assert dialog.repair_box.isVisible()
        assert "syntax error" in dialog.problem_message_label.text()
        assert dialog.label_edit.text() == "Broken"
        assert dialog.label_edit.isEnabled() is True
        assert dialog.remove_button.isEnabled() is True
        assert dialog.open_file_button.isEnabled() is True
        # The payload parses, so there is a form to drive; no raw text pane.
        assert dialog.raw_json_edit.isHidden()
        # Including the parameter and slot tables, which read stored dicts here
        # rather than registry objects.
        assert dialog.params_table.rowCount() == 1
        assert dialog.params_table.item(0, 0).text() == "amount"
        assert dialog.params_table.item(0, 2).text() == "2"
        assert dialog.slots_table.rowCount() == 1
        assert dialog.slots_table.item(0, 0).text() == "reference"
        assert "open-document" in dialog.slots_table.item(0, 2).text()

        # An edit made here saves to the file by path.
        dialog.label_edit.setText("Broken but editable")
        dialog.label_edit.editingFinished.emit()
        process_events(qtbot)
        assert json.loads((ops_dir / "broken.json").read_text())["label"] == "Broken but editable"

        # Editing a stored parameter is an ordinary edit here too.
        dialog.params_table.item(0, 0).setText("gain")
        process_events(qtbot)
        stored = json.loads((ops_dir / "broken.json").read_text())
        assert [parameter["name"] for parameter in stored["parameters"]] == ["gain"]

        # And the add buttons are gated on "may this dialog rewrite the
        # selection", which a Problems row now satisfies -- they used to be
        # gated on the selection having a registered user id, which it cannot.
        assert dialog._editing_definition() is True
        dialog.add_param_button.click()
        dialog.add_slot_button.click()
        process_events(qtbot)
        assert dialog.selected_problem_path() == path

        # Open-file reaches the wrapper -- the file that actually needs editing
        # when the definition did not load.
        opened = []
        monkeypatch.setattr(
            QtGui.QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile())
        )
        dialog.open_file_button.click()
        process_events(qtbot)
        assert opened == [path]

        # And a Problems row survives an unrelated drag: it has no operation id,
        # so the tree's id-shaped normalization would otherwise evict it to the
        # top level on the next drop.
        dialog.tree._normalize()
        assert _select_problem_row(dialog, "broken.json") is not None

        # Fixing what the loader complained about takes the row out of
        # Problems and lands the selection on the now-registered operation.
        (ops_dir / "broken.py").write_text("def oops(data):\n    return data\n")
        library.refresh_user_operations()
        process_events(qtbot)
        groups = [
            dialog.tree.topLevelItem(i).data(0, 0x0101)
            for i in range(dialog.tree.topLevelItemCount())
        ]
        assert "Problems" not in groups
        assert dialog.selected_operation_id() == "user:broken"
    finally:
        dialog.close()
        win.close()


def test_repair_editor_retargets_a_missing_source_file(qtbot, tmp_path, monkeypatch):
    import json

    from arrayscope.operations import library
    from arrayscope.ui import operation_manager

    # "source file not found" is the single most likely way a stored definition
    # stops loading: the file it pointed at moved. Pointing the ordinary source
    # picker at the file's new home has to be the repair, in the manager.
    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "moved.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:moved",
                "label": "Moved",
                "source": {"mode": "import", "path": "gone.py", "callable": "double"},
            }
        )
    )
    replacement = _write_source(tmp_path, "found.py", "def double(data):\n    return data * 2\n")
    monkeypatch.setattr(operation_manager, "get_open_file_name", lambda *a, **k: (replacement, ""))
    library.refresh_user_operations()

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert _select_problem_row(dialog, "moved.json") is not None
        process_events(qtbot)
        assert "source file not found" in dialog.problem_message_label.text()
        assert dialog.source_box.isVisible()
        assert dialog.source_browse_button.isEnabled() is True

        dialog.copy_radio.setChecked(True)
        dialog.source_browse_button.click()
        process_events(qtbot)

        assert not library.user_operation_problems()
        assert dialog.selected_operation_id() == "user:moved"
        # Copy mode really copied, exactly as it does for a loaded operation.
        wrapper = library.user_operation_wrapper("user:moved")
        assert wrapper["source"] == {"mode": "import", "path": "found.py", "callable": "double"}
        assert (ops_dir / "found.py").exists()
    finally:
        dialog.close()
        win.close()


def test_unparseable_wrapper_gets_a_raw_json_editor_and_a_path_keyed_delete(
    qtbot, tmp_path, monkeypatch
):
    import json

    from arrayscope.operations import library

    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "garbled.py").write_text("def double(data):\n    return data * 2\n")
    (ops_dir / "garbled.json").write_text('{"format": "arrayscope-operation", "version":')
    library.refresh_user_operations()

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        path = _select_problem_row(dialog, "garbled.json")
        assert path is not None
        process_events(qtbot)
        # No dict to fill a form from, so the text of the file is the editor.
        assert dialog.repair_box.isVisible()
        assert dialog.raw_json_edit.isVisible()
        assert dialog.raw_json_edit.toPlainText().startswith('{"format"')
        assert dialog.label_edit.isEnabled() is False

        # Saving something that is still not JSON reports it and writes nothing.
        dialog.raw_json_edit.setPlainText("{oops")
        dialog.raw_json_save_button.click()
        process_events(qtbot)
        assert "Not valid JSON" in dialog.problem_message_label.text()
        assert (ops_dir / "garbled.json").read_text().startswith('{"format"')

        # Saving a valid wrapper resolves the row.
        dialog.raw_json_edit.setPlainText(
            json.dumps(
                {
                    "format": "arrayscope-operation",
                    "version": 1,
                    "id": "user:garbled",
                    "label": "Garbled",
                    "source": {"mode": "import", "path": "garbled.py", "callable": "double"},
                }
            )
        )
        dialog.raw_json_save_button.click()
        process_events(qtbot)
        assert not library.user_operation_problems()
        assert dialog.selected_operation_id() == "user:garbled"

        # And a broken definition is deletable from the manager, by path.
        (ops_dir / "again.json").write_text("{not json at all")
        library.refresh_user_operations()
        process_events(qtbot)
        assert _select_problem_row(dialog, "again.json") is not None
        process_events(qtbot)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "question",
            lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes,
        )
        dialog.remove_button.click()
        process_events(qtbot)
        assert not (ops_dir / "again.json").exists()
        assert not library.user_operation_problems()
    finally:
        dialog.close()
        win.close()


def test_command_copy_toggled_to_python_stays_in_the_tree(qtbot, tmp_path):
    from arrayscope.operations import library

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        # Build a command operation the way the manager does, duplicate it, and
        # then use the one control the copy invites: the Runtime combo. This is
        # the reported brick -- the duplicate used to be written without a
        # `source` block, so switching to Python produced a wrapper the loader
        # refuses and the operation vanished into Problems.
        dialog.new_button.click()
        process_events(qtbot)
        command_id = dialog.selected_operation_id()
        dialog.runtime_combo.setCurrentIndex(dialog.runtime_combo.findData("command"))
        dialog.command_template_edit.setText("tool {in} {out}")
        dialog.command_template_edit.editingFinished.emit()
        process_events(qtbot)

        dialog.duplicate_button.click()
        process_events(qtbot)
        copy_id = dialog.selected_operation_id()
        assert copy_id != command_id
        assert library.user_operation_wrapper(copy_id)["runtime"] == "command"

        dialog.runtime_combo.setCurrentIndex(dialog.runtime_combo.findData("python"))
        process_events(qtbot)

        assert not library.user_operation_problems()
        assert dialog.select_operation(copy_id)
        wrapper = library.user_operation_wrapper(copy_id)
        assert wrapper["runtime"] == "python"
        assert wrapper["source"]["callable"]
        groups = [
            dialog.tree.topLevelItem(i).data(0, 0x0101)
            for i in range(dialog.tree.topLevelItemCount())
        ]
        assert "Problems" not in groups
    finally:
        dialog.close()
        win.close()


def test_shell_plus_environment_prefix_does_not_take_down_the_manager(qtbot, tmp_path, monkeypatch):
    import json

    from arrayscope.operations import environments, library

    # Shell execution plus an environment that resolves to a conda prefix makes
    # command_runtime.build_command raise from inside availability(). That call
    # sits under every operation enumeration, so before the guard this did not
    # disable one operation -- the manager could not be constructed at all.
    ops_dir = tmp_path / "operations"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / environments.ENVIRONMENTS_FILE).write_text(
        json.dumps(
            {
                "format": environments.ENVIRONMENTS_FORMAT,
                "version": environments.ENVIRONMENTS_VERSION,
                "environments": [{"id": "recon", "name": "Recon", "conda_env": "recon-env"}],
            }
        )
    )
    (ops_dir / "shelled.json").write_text(
        json.dumps(
            {
                "format": "arrayscope-operation",
                "version": 1,
                "id": "user:shelled",
                "label": "Shelled",
                "runtime": "command",
                "command_template": "tool {in} {out}",
                "shell": True,
                "environment": "recon",
            }
        )
    )
    record = environments.ExecutionEnvironment(id="recon", name="Recon", conda_env="recon-env")
    resolved = environments.ResolvedEnvironment(
        record, {}, None, ("conda", "run", "-n", "recon-env"), interpreter="python"
    )
    monkeypatch.setattr(library, "resolve_execution_environment", lambda _id: (resolved, None))
    library.refresh_user_operations()

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_operation("user:shelled")
        process_events(qtbot)
        assert "Unavailable" in dialog.status_label.text()
        assert "shell execution" in dialog.status_label.text()
        # The whole listing still builds, which is the actual regression.
        from arrayscope.ui.operation_listing import build_operation_listing

        listed = {entry.id for section in build_operation_listing() for entry in section.entries}
        assert "user:shelled" in listed
    finally:
        dialog.close()
        win.close()


def test_unresolvable_environment_reference_is_not_silently_cleared(qtbot, tmp_path):
    from arrayscope.operations import library

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        dialog.new_button.click()
        process_events(qtbot)
        operation_id = dialog.selected_operation_id()
        dialog.runtime_combo.setCurrentIndex(dialog.runtime_combo.findData("command"))
        dialog.command_template_edit.setText("tool {in} {out}")
        dialog.command_template_edit.editingFinished.emit()
        process_events(qtbot)
        # A reference to an environment that has no record: deleted, renamed,
        # or arrived with an imported definition.
        library.update_user_operation(operation_id, environment="ghost")
        process_events(qtbot)

        assert dialog.select_operation(operation_id)
        process_events(qtbot)
        assert dialog.environment_combo.currentData() == "ghost"
        assert "missing" in dialog.environment_combo.currentText()

        # The next unrelated runtime autosave must not persist it away.
        dialog.handoff_combo.setCurrentText("cfl")
        process_events(qtbot)
        assert library.user_operation_wrapper(operation_id)["environment"] == "ghost"
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
