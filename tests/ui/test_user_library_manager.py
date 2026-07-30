"""The manager grammar both libraries speak (ADR 0063).

Everything here used to be tested twice -- once per manager, with the two
copies already disagreeing about what the behavior *was*. It is one
implementation now, so it is tested once, against both dialogs, and the
vocabulary itself is an assertion: if a button, an icon or a marker drifts on
one side, this file fails rather than the drift shipping.
"""

import functools
import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.ui.helpers import (
    auto_confirm_message_boxes,
    clear_arrayscope_settings,
    process_events,
)

_ICON_NAME_PROPERTY = "arrayscope_icon_name"


@pytest.fixture(autouse=True)
def _isolated_libraries(tmp_path, monkeypatch):
    from arrayscope.app import user_dirs
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    monkeypatch.setattr(user_dirs, "user_config_directory", lambda *a, **k: tmp_path)
    monkeypatch.setattr(user_dirs, "user_operations_directory", lambda: tmp_path / "operations")
    monkeypatch.setattr(
        operation_library, "user_operations_directory", lambda: str(tmp_path / "operations")
    )
    monkeypatch.setattr(
        colormap_library, "user_colormap_directory", lambda: str(tmp_path / "colormaps")
    )
    colormap_library.refresh_user_colormaps()
    operation_library.refresh_user_operations()
    operation_library.reset_layout()
    for op_id in tuple(operation_library.hidden_operations()):
        operation_library.set_operation_hidden(op_id, False)
    yield
    colormap_library.refresh_user_colormaps()
    operation_library.refresh_user_operations()


def _window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    win = ArrayScopeWindow(np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6))
    qtbot.addWidget(win)
    win.show()
    process_events(qtbot)
    return win


def _managers(qtbot, win):
    from arrayscope.ui.colormap_manager import ColormapManagerDialog
    from arrayscope.ui.operation_manager import OperationManagerDialog

    dialogs = []
    for factory in (ColormapManagerDialog, OperationManagerDialog):
        dialog = factory(win)
        qtbot.addWidget(dialog)
        dialog.show()
        dialogs.append(dialog)
    process_events(qtbot)
    return dialogs


# ---------------------------------------------------------------------------
# The vocabulary, as an invariant
# ---------------------------------------------------------------------------


def test_both_managers_expose_the_same_buttons_in_the_same_order_with_the_same_icons(qtbot):
    from arrayscope.ui.user_library_manager import LIBRARY_BUTTONS

    win = _window(qtbot)
    dialogs = _managers(qtbot, win)
    try:
        for dialog in dialogs:
            for spec in LIBRARY_BUTTONS:
                button = getattr(dialog, spec.name)
                assert button.objectName() == spec.name
                # Remove is the one dynamic icon (delete vs visibility_off): it
                # follows the action, so its *current* value depends on the
                # selection, and only its two legal values are pinned here.
                expected = (
                    {"delete", "visibility_off"} if spec.name == "remove_button" else {spec.icon}
                )
                assert button.property(_ICON_NAME_PROPERTY) in expected, (dialog, spec.name)
                assert button.toolTip()
            # Filter row and the shared editor actions, same names on both sides.
            assert dialog.filter_combo.objectName() == "LibraryFilterCombo"
            assert dialog.filter_edit.objectName() == "LibraryFilterEdit"
            assert dialog.apply_button.text() == "Apply"
            assert dialog.done_button.text() == "Done"
    finally:
        for dialog in dialogs:
            dialog.close()
        win.close()


def test_the_two_collided_glyphs_are_settled_the_same_way_on_both_sides(qtbot):
    # ``folder_open`` used to mean Import in the colormap designer and
    # Open-folder in the operation manager. One glyph, two meanings, one repo.
    win = _window(qtbot)
    dialogs = _managers(qtbot, win)
    try:
        for dialog in dialogs:
            assert dialog.open_folder_button.property(_ICON_NAME_PROPERTY) == "folder_open"
            assert dialog.import_button.property(_ICON_NAME_PROPERTY) == "upload"
            assert dialog.export_button.property(_ICON_NAME_PROPERTY) == "download"
            # ``data_object`` said nothing about duplication on either side.
            assert dialog.duplicate_button.property(_ICON_NAME_PROPERTY) == "content_copy"
    finally:
        for dialog in dialogs:
            dialog.close()
        win.close()


def test_marker_vocabulary_is_one_vocabulary():
    from arrayscope.ui import user_library_manager as base

    assert base.marker_suffix() == ""
    assert base.marker_suffix(user=True) == "  (user)"
    assert base.marker_suffix(hidden=True, user=True) == "  (hidden) (user)"
    # "modified" replaces "user": a user file that shadows a system definition
    # is the more specific fact about it.
    assert base.marker_suffix(user=True, modified=True) == "  (modified)"
    assert base.marker_suffix(unavailable=True) == "  (unavailable)"


def test_duplicate_naming_is_one_convention():
    from arrayscope.app.user_library import unique_name

    assert unique_name("viridis", []) == "viridis copy"
    assert unique_name("viridis", ["viridis copy"]) == "viridis copy 2"
    assert unique_name("viridis", ["viridis copy", "viridis copy 2"]) == "viridis copy 3"
    # The new-item case takes the plain name when it is free.
    assert unique_name("custom", [], copy_suffix=False) == "custom"
    assert unique_name("custom", ["custom"], copy_suffix=False) == "custom 2"


# ---------------------------------------------------------------------------
# Shared behavior, exercised against both dialogs
# ---------------------------------------------------------------------------


def test_search_narrows_both_trees_and_hides_emptied_groups(qtbot):
    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        for dialog, needle, expected in (
            (colormaps, "turbo", "turbo"),
            (operations, "mean", "mean"),
        ):
            dialog.filter_edit.setText(needle)
            process_events(qtbot)
            visible = _visible_keys(dialog)
            assert expected in visible
            assert len(visible) < len(_all_keys(dialog))
            assert any(
                not dialog.tree.topLevelItem(index).isHidden()
                for index in range(dialog.tree.topLevelItemCount())
            )
            # A group with nothing left in it goes away rather than sitting empty.
            assert all(
                dialog.tree.topLevelItem(index).isHidden()
                or _visible_children(dialog.tree.topLevelItem(index))
                for index in range(dialog.tree.topLevelItemCount())
            )
            dialog.filter_edit.setText("")
            process_events(qtbot)
            assert len(_visible_keys(dialog)) == len(_all_keys(dialog))
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_a_group_renames_in_place_on_both_sides(qtbot):
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        for dialog, old, module in (
            (colormaps, "Perceptual", colormap_library),
            (operations, "Reduce", operation_library),
        ):
            item = _group_item(dialog, old)
            assert item is not None
            assert item.flags() & item.flags().__class__.ItemIsEditable
            item.setText(0, "Mine")
            process_events(qtbot)
            groups = [
                group
                for group, _members in (
                    module.grouped_colormaps()
                    if module is colormap_library
                    else module.grouped_operations()
                )
            ]
            assert "Mine" in groups
            assert old not in groups
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_removing_a_definition_asks_first_on_both_sides(qtbot, monkeypatch):
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    colormap_library.save_user_colormap(
        "mine", colormap_library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
    )
    operation_id = operation_library.create_empty_user_operation()

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        # Declining leaves the definition alone -- the colormap manager used to
        # delete on a single click with no confirmation at all.
        auto_confirm_message_boxes(monkeypatch, answer=False)
        assert colormaps.select_key("mine")
        colormaps.remove_button.click()
        assert operations.select_key(operation_id)
        operations.remove_button.click()
        process_events(qtbot)
        assert colormap_library.find_colormap("mine") is not None
        assert operation_library.user_operation_wrapper(operation_id) is not None

        auto_confirm_message_boxes(monkeypatch, answer=True)
        assert colormaps.select_key("mine")
        colormaps.remove_button.click()
        assert operations.select_key(operation_id)
        operations.remove_button.click()
        process_events(qtbot)
        assert colormap_library.find_colormap("mine") is None
        assert operation_library.user_operation_wrapper(operation_id) is None
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_revert_undoes_the_last_destructive_change_on_both_sides(qtbot, monkeypatch):
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    auto_confirm_message_boxes(monkeypatch)
    colormap_library.save_user_colormap(
        "mine", colormap_library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
    )
    operation_id = operation_library.create_empty_user_operation()

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        for dialog, key in ((colormaps, "mine"), (operations, operation_id)):
            assert dialog.select_key(key)
            process_events(qtbot)
            # The button only exists when it leads somewhere.
            assert not dialog.revert_button.isVisible()
            dialog.remove_button.click()
            process_events(qtbot)
            assert dialog.revert_button.isVisible()
            dialog.revert_button.click()
            process_events(qtbot)
            assert dialog.select_key(key)
            assert not dialog.revert_button.isVisible()
        assert colormap_library.find_colormap("mine") is not None
        # A user operation restored *without* its code file would be a Problems
        # row, so the snapshot has to carry the whole definition.
        assert operation_library.user_operation_problems() == []
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_reset_all_restores_the_arrangement_and_everything_hidden(qtbot, monkeypatch):
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    auto_confirm_message_boxes(monkeypatch)
    colormap_library.set_builtin_hidden("turbo", True)
    operation_library.set_operation_hidden("mean", True)
    colormap_library.apply_library_layout(["User"], {"viridis": "User"}, {"viridis": 0})
    operation_library.apply_library_layout(op_groups={"fftshift": "User"})

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        colormaps.reset_all_button.click()
        operations.reset_all_button.click()
        process_events(qtbot)
        assert colormap_library.hidden_builtins() == frozenset()
        assert operation_library.hidden_operations() == frozenset()
        assert colormap_library.effective_group(colormap_library.find_colormap("viridis")) != "User"
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_a_manager_unregisters_its_listener_when_it_closes(qtbot):
    # The registry is process-global, so a listener that outlives its dialog
    # turns the next library mutation into a call on a deleted C++ object.
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    assert colormaps.refresh_tree in colormap_library._listeners._callbacks
    assert operations.refresh_tree in operation_library._listeners._callbacks
    colormaps.close()
    operations.close()
    process_events(qtbot)
    assert colormaps.refresh_tree not in colormap_library._listeners._callbacks
    assert operations.refresh_tree not in operation_library._listeners._callbacks
    win.close()


def test_a_problems_row_survives_a_drag_that_touches_nothing_else(qtbot, tmp_path):
    # The Problems group is not part of the arrangement; the id-shaped
    # normalization would otherwise evict every one of its rows on the next
    # unrelated drop.
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    (tmp_path / "colormaps").mkdir(parents=True, exist_ok=True)
    (tmp_path / "colormaps" / "broken.json").write_text('{"name": "broken", "stops":')
    (tmp_path / "operations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "operations" / "broken.json").write_text('{"format": "arrayscope-operation",')
    colormap_library.refresh_user_colormaps()
    operation_library.refresh_user_operations()

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        for dialog in (colormaps, operations):
            paths = _problem_paths(dialog)
            assert len(paths) == 1
            dialog.tree._normalize()
            assert _problem_paths(dialog) == paths
            # And it is addressable: selectable, with its file reachable.
            assert dialog.select_problem_path(paths[0])
            assert dialog.remove_button.isEnabled()
            assert dialog.open_file_button.isEnabled()
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def _pick_file(path, *_args, **_kwargs):
    """Stand in for the platform file dialog, which cannot answer offscreen."""

    return path, ""


def test_new_and_duplicate_produce_an_editable_definition_on_both_sides(qtbot):
    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        for dialog in (colormaps, operations):
            before = set(_all_keys(dialog))
            dialog.new_button.click()
            process_events(qtbot)
            created = dialog.selected_key()
            assert created is not None
            assert created not in before

            dialog.duplicate_button.click()
            process_events(qtbot)
            duplicated = dialog.selected_key()
            assert duplicated is not None
            assert duplicated != created
        # The one duplicate-suffix convention, on the surface a user reads.
        assert colormaps.selected_key() == "custom copy"
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_export_then_import_round_trips_on_both_sides(qtbot, tmp_path, monkeypatch):
    from arrayscope.display import colormap_library
    from arrayscope.ui import colormap_manager, operation_manager

    colormap_library.save_user_colormap(
        "mine", colormap_library.SEQUENTIAL, ((0.0, (1, 2, 3)), (1.0, (250, 240, 230)))
    )
    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        for dialog, module, name in (
            (colormaps, colormap_manager, "mine.json"),
            (operations, operation_manager, "operation.json"),
        ):
            target = str(tmp_path / name)
            chosen = functools.partial(_pick_file, target)
            monkeypatch.setattr(module, "get_save_file_name", chosen)
            monkeypatch.setattr(module, "get_open_file_name", chosen)
            if dialog is colormaps:
                assert dialog.select_key("mine")
            else:
                dialog.new_button.click()
                process_events(qtbot)
            exported_from = dialog.selected_key()
            dialog.export_button.click()
            process_events(qtbot)
            assert os.path.exists(target)

            dialog.import_button.click()
            process_events(qtbot)
            imported = dialog.selected_key()
            assert imported is not None
            # A fresh identity, never an overwrite of what it came from.
            assert imported != exported_from
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_open_file_reaches_the_stored_definition_on_both_sides(qtbot, monkeypatch):
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    colormap_library.save_user_colormap(
        "mine", colormap_library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
    )
    operation_id = operation_library.create_empty_user_operation()

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    opened = []
    try:
        for dialog, key in ((colormaps, "mine"), (operations, operation_id)):
            monkeypatch.setattr(type(dialog), "open_path", lambda _self, path: opened.append(path))
            assert dialog.select_key(key)
            process_events(qtbot)
            assert dialog.open_file_button.isEnabled()
            dialog.open_file_button.click()
            process_events(qtbot)
        assert len(opened) == 2
        assert all(os.path.exists(path) for path in opened)
        # A colormap opens its stored JSON; an operation opens its code file --
        # in both cases the file the user would edit by hand.
        assert opened[0].endswith(".json")
        assert opened[1].endswith(".py")
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


def test_the_pinned_section_is_one_checkbox_on_both_sides(qtbot):
    from arrayscope.display import colormap_library
    from arrayscope.operations import library as operation_library

    win = _window(qtbot)
    colormaps, operations = _managers(qtbot, win)
    try:
        # "Favorites" (a drag target) and "Common" (a checkbox) were the same
        # concept reached two different ways. Both are a checkbox now, in the
        # same place in the editor.
        assert "pinned" in colormaps.common_check.text()
        assert "pinned" in operations.common_check.text()

        assert colormaps.select_key("turbo")
        process_events(qtbot)
        assert not colormaps.common_check.isChecked()
        colormaps.common_check.setChecked(True)
        process_events(qtbot)
        info = colormap_library.find_colormap("turbo")
        assert colormap_library.effective_group(info) == colormap_library.FAVORITES_GROUP

        assert operations.select_key("crop")
        process_events(qtbot)
        operations.common_check.setChecked(True)
        process_events(qtbot)
        assert "crop" in operation_library.effective_common_ids()
    finally:
        for dialog in (colormaps, operations):
            dialog.close()
        win.close()


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------


def _rows(dialog):
    from arrayscope.ui.user_library_manager import ITEM_ROLE

    for group_index in range(dialog.tree.topLevelItemCount()):
        group_item = dialog.tree.topLevelItem(group_index)
        for child_index in range(group_item.childCount()):
            child = group_item.child(child_index)
            key = child.data(0, ITEM_ROLE)
            if key is not None:
                yield child, str(key)


def _all_keys(dialog):
    return [key for _item, key in _rows(dialog)]


def _visible_keys(dialog):
    return [key for item, key in _rows(dialog) if not item.isHidden()]


def _visible_children(group_item):
    return [
        group_item.child(index)
        for index in range(group_item.childCount())
        if not group_item.child(index).isHidden()
    ]


def _group_item(dialog, group):
    from arrayscope.ui.user_library_manager import GROUP_ROLE

    for index in range(dialog.tree.topLevelItemCount()):
        item = dialog.tree.topLevelItem(index)
        if item.data(0, GROUP_ROLE) == group:
            return item
    return None


def _problem_paths(dialog):
    from arrayscope.ui.user_library_manager import PROBLEM_PATH_ROLE

    paths = []
    for group_index in range(dialog.tree.topLevelItemCount()):
        group_item = dialog.tree.topLevelItem(group_index)
        for child_index in range(group_item.childCount()):
            path = group_item.child(child_index).data(0, PROBLEM_PATH_ROLE)
            if path is not None:
                paths.append(str(path))
    return paths
