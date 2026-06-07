"""Qt model for ROI inspection state."""

from __future__ import annotations

import pyqtgraph.Qt as Qt


class RoiTableModel(Qt.QtCore.QAbstractTableModel):
    HEADERS = ("ROI", "Kind", "Count", "Mean", "Std", "Min", "Max", "Visible")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []

    def set_rows(self, rows):
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent=Qt.QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=Qt.QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.QtCore.Qt.ItemDataRole.DisplayRole):
        if role == Qt.QtCore.Qt.ItemDataRole.DisplayRole and orientation == Qt.QtCore.Qt.Orientation.Horizontal:
            return self.HEADERS[int(section)]
        return None

    def data(self, index, role=Qt.QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        column = index.column()
        if role == Qt.QtCore.Qt.ItemDataRole.UserRole:
            return row["id"]
        if role == Qt.QtCore.Qt.ItemDataRole.DisplayRole:
            return row["values"][column]
        if role == Qt.QtCore.Qt.ItemDataRole.CheckStateRole and column == 7:
            return Qt.QtCore.Qt.CheckState.Checked if row["enabled"] else Qt.QtCore.Qt.CheckState.Unchecked
        if role == Qt.QtCore.Qt.ItemDataRole.BackgroundRole:
            color = row.get("color")
            if color is not None:
                return Qt.QtGui.QColor(*color, 36)
        if role == Qt.QtCore.Qt.ItemDataRole.DecorationRole and column == 0:
            color = row.get("color")
            if color is not None:
                pixmap = Qt.QtGui.QPixmap(12, 12)
                pixmap.fill(Qt.QtGui.QColor(*color))
                return pixmap
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 7:
            flags |= Qt.QtCore.Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def roi_id_for_row(self, row):
        if 0 <= int(row) < len(self._rows):
            return self._rows[int(row)]["id"]
        return None
