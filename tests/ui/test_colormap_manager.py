"""What is genuinely the *colormap* manager's own.

The tree, the button bar, the filter, group rename, confirmation, Revert and
the Problems group are the shared grammar and are tested once, against both
managers, in ``test_user_library_manager.py`` (ADR 0063).
"""

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


@pytest.fixture(autouse=True)
def _isolated_user_dir(tmp_path, monkeypatch):
    from arrayscope.display import colormap_library as library

    monkeypatch.setattr(library, "user_colormap_directory", lambda: str(tmp_path / "colormaps"))
    library.refresh_user_colormaps()
    yield
    library.refresh_user_colormaps()


def _window(qtbot):
    from arrayscope.window import ArrayScopeWindow

    clear_arrayscope_settings()
    win = ArrayScopeWindow(np.arange(16 * 16, dtype=float).reshape(16, 16))
    qtbot.addWidget(win)
    win.show()
    process_events(qtbot)
    return win


def _manager(qtbot, win):
    from arrayscope.ui.colormap_manager import ColormapManagerDialog

    dialog = ColormapManagerDialog(win)
    qtbot.addWidget(dialog)
    dialog.show()
    process_events(qtbot)
    return dialog


def test_rename_moves_instead_of_duplicating(qtbot):
    from arrayscope.display import colormap_library as library

    library.save_user_colormap(
        "imported", library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
    )
    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_key("imported")
        dialog.name_edit.setText("renamed")
        # Switching away triggers the dirty-snapshot autosave, with rename.
        assert dialog.select_key("gray")
        process_events(qtbot)
        names = {info.name for info in library.user_colormaps()}
        assert "renamed" in names
        assert "imported" not in names
    finally:
        dialog.close()
        win.close()


def test_a_builtin_is_read_only_until_the_user_says_override(qtbot):
    from arrayscope.display import colormap_library as library

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_key("viridis")
        process_events(qtbot)
        # It used to be silently editable: the first keystroke stored an
        # override the user never asked for. Now the editor is read-only and
        # the affordance is explicit.
        assert not dialog.name_edit.isEnabled()
        assert not dialog.kind_combo.isEnabled()
        assert dialog.override_button.isVisible()
        assert not dialog.unhide_button.isEnabled()

        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData(library.DIVERGING))
        assert dialog.select_key("gray")
        process_events(qtbot)
        assert not library.overrides_builtin("viridis")

        assert dialog.select_key("viridis")
        dialog.override_button.click()
        process_events(qtbot)
        assert library.overrides_builtin("viridis")
        assert dialog.name_edit.isEnabled()
        assert not dialog.override_button.isVisible()
        assert dialog.unhide_button.isEnabled()  # Restore brings the system one back
        assert "(modified)" in dialog.tree.currentItem().text(0)

        dialog.unhide_button.click()
        process_events(qtbot)
        assert not library.overrides_builtin("viridis")
    finally:
        dialog.close()
        win.close()


def test_apply_disabled_for_incompatible_kind(qtbot):
    from arrayscope.display import colormap_library as library

    win = _window(qtbot)  # real-valued data => scalar family
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_key("RomaO") or dialog.select_key("PAL-relaxed")
        process_events(qtbot)
        assert dialog.kind_combo.currentData() == library.CYCLIC
        assert not dialog.apply_button.isEnabled()
        assert dialog.conflict_label.text()
        assert dialog.select_key("viridis")
        process_events(qtbot)
        assert dialog.apply_button.isEnabled()
        assert not dialog.conflict_label.text()
    finally:
        dialog.close()
        win.close()


def test_kind_filter_hides_non_matching(qtbot):
    from arrayscope.display import colormap_library as library
    from arrayscope.ui.user_library_manager import ITEM_ROLE

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        dialog.filter_combo.setCurrentIndex(dialog.filter_combo.findData(library.CYCLIC))
        process_events(qtbot)
        visible = []
        tree = dialog.tree
        for gi in range(tree.topLevelItemCount()):
            group_item = tree.topLevelItem(gi)
            for ci in range(group_item.childCount()):
                child = group_item.child(ci)
                if not child.isHidden():
                    visible.append(str(child.data(0, ITEM_ROLE)))
        assert "viridis" not in visible
        assert "RomaO" in visible
    finally:
        dialog.close()
        win.close()


def test_an_unreadable_map_is_repaired_where_it_is_listed(qtbot, tmp_path, monkeypatch):
    # ADR 0062's invariant, applied to colormaps: a file the library cannot read
    # used to vanish with no message anywhere.
    from arrayscope.display import colormap_library as library

    auto_confirm_message_boxes(monkeypatch)
    directory = tmp_path / "colormaps"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "garbled.json").write_text('{"name": "garbled", "stops":')
    library.refresh_user_colormaps()

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        path = str(directory / "garbled.json")
        assert dialog.select_problem_path(path)
        process_events(qtbot)
        assert dialog.repair_box.isVisible()
        assert "Unreadable colormap file" in dialog.problem_message_label.text()
        assert dialog.raw_json_edit.toPlainText().startswith('{"name"')

        # Saving text that is still not a JSON object says so instead of writing
        # a knowingly-broken file.
        dialog.raw_json_edit.setPlainText("[1, 2, 3]")
        dialog.raw_json_save_button.click()
        process_events(qtbot)
        assert "JSON object" in dialog.problem_message_label.text()
        assert library.user_colormap_problems()

        dialog.raw_json_edit.setPlainText(
            '{"name": "garbled", "kind": "sequential", '
            '"stops": [[0.0, [0, 0, 0]], [1.0, [255, 255, 255]]]}'
        )
        dialog.raw_json_save_button.click()
        process_events(qtbot)
        assert library.user_colormap_problems() == []
        # The selection follows the definition to where it now lives.
        assert dialog.selected_key() == "garbled"
    finally:
        dialog.close()
        win.close()


def test_an_unavailable_builtin_is_listed_greyed_and_explains_itself(qtbot, monkeypatch):
    from arrayscope.display import colormap_library as library

    monkeypatch.setattr(library, "_crameri_stops_cache", {})
    monkeypatch.setattr(library, "_resolve_builtin_stops", lambda spec: None if spec else spec)

    win = _window(qtbot)
    dialog = _manager(qtbot, win)
    try:
        assert dialog.select_key("Batlow")
        process_events(qtbot)
        assert "(unavailable)" in dialog.tree.currentItem().text(0)
        assert "cmcrameri" in dialog.source_label.text()
        # Nothing that would produce a map it cannot draw.
        assert not dialog.override_button.isEnabled()
        assert not dialog.export_button.isEnabled()
        assert not dialog.duplicate_button.isEnabled()
    finally:
        dialog.close()
        win.close()
