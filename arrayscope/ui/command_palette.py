"""Searchable command and operation palette."""

from __future__ import annotations

from dataclasses import dataclass

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtWidgets

from arrayscope.ui.icons import material_icon


@dataclass(frozen=True)
class PaletteCommand:
    id: str
    label: str
    kind: str = "command"
    requires_axis: bool = False
    icon: str = "search"


class CommandPaletteDialog(QtWidgets.QDialog):
    def __init__(self, commands, axis_choices=(), default_axis=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Command Palette")
        self.setObjectName("CommandPaletteDialog")
        self._commands = tuple(commands)
        self._filtered = self._commands

        layout = QtWidgets.QVBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Type a command or operation")
        layout.addWidget(self.search)

        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget, 1)

        axis_row = QtWidgets.QHBoxLayout()
        axis_row.addWidget(QtWidgets.QLabel("Axis"))
        self.axis_combo = QtWidgets.QComboBox()
        for label, axis in axis_choices:
            self.axis_combo.addItem(label, axis)
        if default_axis is not None:
            index = self.axis_combo.findData(int(default_axis))
            if index >= 0:
                self.axis_combo.setCurrentIndex(index)
        axis_row.addWidget(self.axis_combo, 1)
        layout.addLayout(axis_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self.search.textChanged.connect(self._refresh)
        self.list_widget.currentRowChanged.connect(lambda _row: self._sync_axis_visibility())
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())
        self._refresh("")
        self.resize(520, 420)

    def selected(self):
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._filtered):
            return None, None
        command = self._filtered[row]
        axis = self.axis_combo.currentData() if command.requires_axis else None
        return command, axis

    def _refresh(self, text):
        words = tuple(part.lower() for part in text.split() if part)
        self._filtered = tuple(
            command for command in self._commands if all(word in command.label.lower() or word in command.id.lower() for word in words)
        )
        self.list_widget.clear()
        for command in self._filtered:
            prefix = "op" if command.kind == "operation" else "cmd"
            item = QtWidgets.QListWidgetItem(material_icon(command.icon), f"{prefix}: {command.label}")
            self.list_widget.addItem(item)
        if self._filtered:
            self.list_widget.setCurrentRow(0)
        self._sync_axis_visibility()

    def _sync_axis_visibility(self):
        command, _axis = self.selected()
        visible = bool(command and command.requires_axis)
        self.axis_combo.setVisible(visible)
        label = self.axis_combo.parentWidget()
        if label is not None:
            label.setVisible(True)
