import os

import numpy as np
import pytest

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.ui.helpers import clear_arrayscope_settings, process_events


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


def _designer(qtbot, win):
    from arrayscope.ui.colormap_designer import ColormapDesignerDialog

    dialog = ColormapDesignerDialog(win)
    qtbot.addWidget(dialog)
    dialog.show()
    process_events(qtbot)
    return dialog


def _select(dialog, name):
    for row in range(dialog.list_widget.count()):
        if dialog.list_widget.item(row).data(0x0100) == name:  # UserRole
            dialog.list_widget.setCurrentRow(row)
            return True
    return False


def test_rename_moves_instead_of_duplicating(qtbot):
    from arrayscope.display import colormap_library as library

    library.save_user_colormap("imported", library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255))))
    win = _window(qtbot)
    dialog = _designer(qtbot, win)
    try:
        assert _select(dialog, "imported")
        dialog.name_edit.setText("renamed")
        # Switching away triggers the autosave-with-rename path.
        assert _select(dialog, "gray")
        process_events(qtbot)
        names = {info.name for info in library.user_colormaps()}
        assert "renamed" in names
        assert "imported" not in names
    finally:
        dialog.close()
        win.close()


def test_builtin_edit_creates_override_and_reset_restores(qtbot):
    from arrayscope.display import colormap_library as library

    win = _window(qtbot)
    dialog = _designer(qtbot, win)
    try:
        assert _select(dialog, "viridis")
        assert not dialog.reset_button.isVisible()
        # Edit the kind => dirty; switching away autosaves an override.
        dialog.kind_combo.setCurrentIndex(dialog.kind_combo.findData(library.DIVERGING))
        assert _select(dialog, "gray")
        process_events(qtbot)
        assert library.overrides_builtin("viridis")
        assert _select(dialog, "viridis")
        process_events(qtbot)
        assert dialog.reset_button.isVisible()
        dialog.reset_button.click()
        process_events(qtbot)
        assert not library.overrides_builtin("viridis")
    finally:
        dialog.close()
        win.close()


def test_apply_disabled_for_incompatible_kind(qtbot):
    from arrayscope.display import colormap_library as library

    win = _window(qtbot)  # real-valued data => scalar family
    dialog = _designer(qtbot, win)
    try:
        assert _select(dialog, "RomaO") or _select(dialog, "PAL-relaxed")
        process_events(qtbot)
        assert dialog.kind_combo.currentData() == library.CYCLIC
        assert not dialog.apply_button.isEnabled()
        assert dialog.conflict_label.text()
        assert _select(dialog, "viridis")
        process_events(qtbot)
        assert dialog.apply_button.isEnabled()
        assert not dialog.conflict_label.text()
    finally:
        dialog.close()
        win.close()


def test_delete_builtin_hides_and_reset_restores(qtbot):
    from arrayscope.display import colormap_library as library

    win = _window(qtbot)
    dialog = _designer(qtbot, win)
    try:
        assert _select(dialog, "turbo")
        dialog.delete_button.click()
        process_events(qtbot)
        assert "turbo" not in {info.name for info in library.list_colormaps()}
        assert _select(dialog, "turbo")  # still listed (greyed) in the designer
        dialog.reset_button.click()
        process_events(qtbot)
        assert "turbo" in {info.name for info in library.list_colormaps()}
    finally:
        dialog.close()
        win.close()
