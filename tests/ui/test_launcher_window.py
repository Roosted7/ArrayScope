"""The empty-state launcher must open files and DICOM folders without a mouse drag."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.launcher_window import LauncherWindow


def _launcher(qtbot, callback, **kwargs):
    win = LauncherWindow(
        callback,
        supported_suffixes=(".npy", ".nii.gz"),
        name_filter="Arrays (*.npy *.nii.gz)",
        **kwargs,
    )
    qtbot.addWidget(win)
    win.show()
    return win


@pytest.mark.parametrize("opened", [False, True], ids=["refused", "opened"])
def test_folder_button_uses_directory_picker_and_keeps_refused_open(
    qtbot, monkeypatch, tmp_path, opened
):
    paths = []
    folder = tmp_path / "DICOM study"
    folder.mkdir()
    win = _launcher(qtbot, lambda path: paths.append(path) or opened)
    button = win.findChild(QtWidgets.QPushButton, "openFolderButton")
    assert button is not None, "The advertised DICOM folder route needs a visible picker"

    def choose_folder(dialog):
        assert dialog.fileMode() == QtWidgets.QFileDialog.FileMode.Directory
        assert dialog.testOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", choose_folder)
    monkeypatch.setattr(QtWidgets.QFileDialog, "selectedFiles", lambda _dialog: [str(folder)])
    qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)

    assert paths == [folder]
    assert win.isVisible() is not opened


def test_standard_open_shortcut_opens_every_selected_file(qtbot, monkeypatch, tmp_path):
    files = [tmp_path / "first.npy", tmp_path / "second.nii.gz"]
    paths = []
    win = _launcher(qtbot, lambda path: paths.append(path) or path == files[0])

    def choose_files(dialog):
        assert dialog.fileMode() == QtWidgets.QFileDialog.FileMode.ExistingFiles
        return QtWidgets.QDialog.DialogCode.Accepted

    monkeypatch.setattr(QtWidgets.QFileDialog, "exec", choose_files)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "selectedFiles", lambda _dialog: list(map(str, files))
    )
    win.activateWindow()
    win.setFocus()
    qtbot.waitUntil(win.isActiveWindow)
    shortcut = QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Open)[0]
    qtbot.keyPress(win, shortcut.key(), modifier=shortcut.keyboardModifiers())

    assert paths == files
    assert not win.isVisible()


def test_cancelling_either_picker_keeps_launcher_available(qtbot, monkeypatch):
    paths = []
    win = _launcher(qtbot, lambda path: paths.append(path) or True)
    monkeypatch.setattr(
        QtWidgets.QFileDialog, "exec", lambda _dialog: QtWidgets.QDialog.DialogCode.Rejected
    )
    for object_name in ("openFilesButton", "openFolderButton"):
        button = win.findChild(QtWidgets.QPushButton, object_name)
        assert button is not None
        assert button.accessibleName()
        qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
        assert win.isVisible()
    assert paths == []


@pytest.mark.parametrize("opened", [False, True], ids=["refused", "opened"])
def test_sample_card_opens_the_requested_example_from_the_keyboard(qtbot, opened):
    sample = SimpleNamespace(
        id="volume",
        title="Volume",
        description="Explore a three-dimensional object.",
        hint="Try slicing and montage.",
    )
    requested = []
    win = _launcher(
        qtbot,
        lambda _path: pytest.fail("An example must use its sample callback"),
        samples=(sample,),
        sample_callback=lambda sample_id: requested.append(sample_id) or opened,
    )
    button = win.findChild(QtWidgets.QPushButton, "sample_volume")
    assert button.accessibleName() == "Explore Volume"
    assert sample.hint in button.accessibleDescription()
    button.setFocus()
    qtbot.keyClick(button, QtCore.Qt.Key.Key_Space)

    assert requested == ["volume"]
    assert win.isVisible() is not opened
    if not opened:
        assert win.status_label.isVisible()
        assert "Could not open" in win.status_label.text()


def test_recent_files_show_the_location_and_keyboard_resume_respects_missing_paths(qtbot, tmp_path):
    available = tmp_path / "study.npy"
    available.write_bytes(b"test")
    missing = tmp_path / "moved.npy"
    attempts = []
    win = _launcher(
        qtbot,
        lambda path: attempts.append(path) or len(attempts) > 1,
        recent_paths=(missing, available),
    )
    missing_item = win.recent_list.item(0)
    assert "Not found" in missing_item.text()
    assert not missing_item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
    assert "no longer available" in missing_item.toolTip()
    available_item = win.recent_list.item(1)
    assert str(available.parent) in available_item.text()
    win.recent_list.itemActivated.emit(missing_item)
    assert attempts == []

    win.recent_list.setCurrentItem(available_item)
    win.recent_list.setFocus()
    qtbot.keyPress(win.recent_list, QtCore.Qt.Key.Key_Return)
    assert attempts == [available]
    assert win.isVisible()
    assert win.status_label.isVisible()

    qtbot.keyPress(win.recent_list, QtCore.Qt.Key.Key_Return)
    assert attempts == [available, available]
    assert not win.isVisible()


def test_keyboard_focus_reveals_example_actions_on_a_small_display(qtbot):
    from arrayscope.app.sample_data import SAMPLES

    win = _launcher(qtbot, lambda _path: False, samples=SAMPLES, sample_callback=lambda _id: False)
    win.resize(520, 480)
    first = win.findChild(QtWidgets.QPushButton, "openFilesButton")
    last = win.findChild(QtWidgets.QPushButton, f"sample_{SAMPLES[-1].id}")
    win.activateWindow()
    first.setFocus()
    qtbot.waitUntil(first.hasFocus)
    for _ in range(8):
        if last.hasFocus():
            break
        qtbot.keyClick(QtWidgets.QApplication.focusWidget(), QtCore.Qt.Key.Key_Tab)
    assert last.hasFocus()
    viewport = win.findChild(QtWidgets.QScrollArea, "launcherBody").viewport()
    assert viewport.rect().contains(last.mapTo(viewport, last.rect().center()))
