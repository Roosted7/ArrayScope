"""Qt file-dialog helpers that avoid problematic native dialog backends."""

from __future__ import annotations

from pyqtgraph.Qt import QtWidgets


def get_save_file_name(parent, title, directory="", name_filter="All files (*)"):
    dialog = QtWidgets.QFileDialog(parent, title, directory, name_filter)
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return "", ""
    selected_filter = dialog.selectedNameFilter()
    files = dialog.selectedFiles()
    return (files[0] if files else ""), selected_filter


def get_open_file_name(parent, title, directory="", name_filter="All files (*)"):
    dialog = QtWidgets.QFileDialog(parent, title, directory, name_filter)
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
    dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return "", ""
    selected_filter = dialog.selectedNameFilter()
    files = dialog.selectedFiles()
    return (files[0] if files else ""), selected_filter


def get_existing_directory(parent, title, directory=""):
    dialog = QtWidgets.QFileDialog(parent, title, directory)
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
    dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
    dialog.setOption(QtWidgets.QFileDialog.Option.ShowDirsOnly, True)
    dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return ""
    files = dialog.selectedFiles()
    return files[0] if files else ""
