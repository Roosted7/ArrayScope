"""Colormap designer: browse, edit, create, import and export colormaps."""

from __future__ import annotations

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.display import colormap_library as library
from arrayscope.ui.file_dialogs import get_open_file_name, get_save_file_name
from arrayscope.ui.icons import material_icon, set_button_icon
from arrayscope.ui.toasts import show_status_message


_KIND_LABELS = (
    ("Sequential", library.SEQUENTIAL),
    ("Diverging", library.DIVERGING),
    ("Cyclic (phase-safe)", library.CYCLIC),
)


def _preview_icon(name: str, width: int = 96, height: int = 14) -> QtGui.QIcon:
    try:
        lut = library.get_colormap(name).getLookupTable(0.0, 1.0, width, alpha=False)
    except Exception:
        return QtGui.QIcon()
    pixmap = QtGui.QPixmap(width, height)
    painter = QtGui.QPainter(pixmap)
    for index, rgb in enumerate(lut):
        painter.fillRect(index, 0, 1, height, QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
    painter.end()
    return QtGui.QIcon(pixmap)


class ColormapDesignerDialog(QtWidgets.QDialog):
    """Non-modal designer; user maps persist and shadow built-ins by name."""

    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self.setWindowTitle("Colormaps")
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.resize(680, 420)

        root = QtWidgets.QHBoxLayout(self)

        # -- left: library list ------------------------------------------
        left = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setIconSize(QtCore.QSize(96, 14))
        self.list_widget.setMinimumWidth(230)
        left.addWidget(self.list_widget, 1)
        list_buttons = QtWidgets.QHBoxLayout()
        self.new_button = QtWidgets.QToolButton(self)
        set_button_icon(self.new_button, "add", tooltip="New colormap")
        self.duplicate_button = QtWidgets.QToolButton(self)
        set_button_icon(self.duplicate_button, "data_object", tooltip="Duplicate selected")
        self.import_button = QtWidgets.QToolButton(self)
        set_button_icon(self.import_button, "folder_open", tooltip="Import (.json, .mat, .csv, .txt, .npy)")
        self.export_button = QtWidgets.QToolButton(self)
        set_button_icon(self.export_button, "download", tooltip="Export selected as JSON")
        self.delete_button = QtWidgets.QToolButton(self)
        set_button_icon(self.delete_button, "delete", tooltip="Delete user colormap")
        for button in (self.new_button, self.duplicate_button, self.import_button, self.export_button, self.delete_button):
            list_buttons.addWidget(button)
        list_buttons.addStretch(1)
        left.addLayout(list_buttons)
        root.addLayout(left)

        # -- right: editor -------------------------------------------------
        right = QtWidgets.QVBoxLayout()
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit(self)
        form.addRow("Name", self.name_edit)
        self.kind_combo = QtWidgets.QComboBox(self)
        for label, kind in _KIND_LABELS:
            self.kind_combo.addItem(label, kind)
        self.kind_combo.setToolTip("Cyclic maps wrap smoothly and are required for phase/complex display")
        form.addRow("Kind", self.kind_combo)
        right.addLayout(form)

        right.addWidget(QtWidgets.QLabel("Stops — drag to move, double-click to recolor, click the bar to add:"))
        self.gradient = pg.GradientWidget(orientation="bottom")
        self.gradient.setMinimumHeight(58)
        right.addWidget(self.gradient)

        self.source_label = QtWidgets.QLabel("")
        self.source_label.setObjectName("OperationsMetaLabel")
        right.addWidget(self.source_label)
        right.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("Save && Apply", self)
        set_button_icon(self.apply_button, "done")
        self.apply_button.setToolTip("Save as a user colormap and apply it to the current view")
        self.save_button = QtWidgets.QPushButton("Save", self)
        set_button_icon(self.save_button, "save")
        actions.addWidget(self.save_button)
        actions.addWidget(self.apply_button)
        right.addLayout(actions)
        root.addLayout(right, 1)

        self.list_widget.currentItemChanged.connect(lambda *_: self._load_selected())
        self.new_button.clicked.connect(self._new_map)
        self.duplicate_button.clicked.connect(self._duplicate_map)
        self.import_button.clicked.connect(self._import_map)
        self.export_button.clicked.connect(self._export_map)
        self.delete_button.clicked.connect(self._delete_map)
        self.save_button.clicked.connect(lambda: self._save(apply=False))
        self.apply_button.clicked.connect(lambda: self._save(apply=True))

        self._reload_list(select=getattr(window, "current_colormap", None))

    # ------------------------------------------------------------------

    def _reload_list(self, select=None):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for info in library.list_colormaps():
            suffix = "  (user)" if info.source == "user" else ""
            item = QtWidgets.QListWidgetItem(_preview_icon(info.name), f"{info.name}{suffix}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, info.name)
            item.setToolTip(f"{info.kind} · {info.source}")
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        if select is not None:
            for row in range(self.list_widget.count()):
                if self.list_widget.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == str(select):
                    self.list_widget.setCurrentRow(row)
                    break
            else:
                self.list_widget.setCurrentRow(0)
        else:
            self.list_widget.setCurrentRow(0)
        self._load_selected()

    def _selected_name(self):
        item = self.list_widget.currentItem()
        return None if item is None else str(item.data(QtCore.Qt.ItemDataRole.UserRole))

    def _load_selected(self):
        name = self._selected_name()
        if name is None:
            return
        info = library.find_colormap(name)
        if info is None:
            return
        self.name_edit.setText(info.name)
        index = self.kind_combo.findData(info.kind)
        self.kind_combo.setCurrentIndex(max(0, index))
        self._set_gradient_stops(library.colormap_stops(info.name))
        self.delete_button.setEnabled(info.source == "user")
        if info.source == "user":
            self.source_label.setText("User colormap — saving overwrites it.")
        else:
            self.source_label.setText("Built-in colormap — saving creates a user copy under the chosen name.")

    def _set_gradient_stops(self, stops):
        positions = [position for position, _color in stops]
        colors = [tuple(color) + (255,) for _position, color in stops]
        self.gradient.item.setColorMap(pg.ColorMap(positions, colors))

    def _gradient_stops(self):
        colormap = self.gradient.item.colorMap()
        positions, colors = colormap.getStops(mode=pg.ColorMap.BYTE)
        return tuple(
            (float(position), (int(color[0]), int(color[1]), int(color[2])))
            for position, color in zip(positions, colors)
        )

    # ------------------------------------------------------------------

    def _save(self, *, apply: bool):
        name = self.name_edit.text().strip()
        if not name:
            show_status_message(self._window, "Colormap name must not be empty.", timeout=2500)
            return
        kind = self.kind_combo.currentData()
        try:
            library.save_user_colormap(name, kind, self._gradient_stops())
        except Exception as exc:
            show_status_message(self._window, f"Could not save colormap: {exc}", timeout=3500)
            return
        self._reload_list(select=name)
        if apply:
            setter = getattr(self._window, "_set_display_colormap", None)
            if callable(setter):
                setter(name, user_selected=True, request_render=True)
        show_status_message(self._window, f"Saved colormap “{name}”.", timeout=2500)

    def _new_map(self, _checked=False):
        self.name_edit.setText(self._unique_name("custom"))
        self.kind_combo.setCurrentIndex(0)
        self._set_gradient_stops(((0.0, (0, 0, 0)), (1.0, (255, 255, 255))))
        self.source_label.setText("New colormap — press Save to keep it.")

    def _duplicate_map(self, _checked=False):
        name = self._selected_name()
        if name is None:
            return
        self.name_edit.setText(self._unique_name(name))
        self.source_label.setText("Copy — press Save to keep it.")

    def _unique_name(self, base):
        existing = {info.name for info in library.list_colormaps()}
        if base not in existing:
            return base
        for index in range(2, 100):
            candidate = f"{base}-{index}"
            if candidate not in existing:
                return candidate
        return f"{base}-copy"

    def _import_map(self, _checked=False):
        path, _ = get_open_file_name(
            self,
            "Import colormap",
            "",
            "Colormaps (*.json *.mat *.csv *.txt *.npy);;All files (*)",
        )
        if not path:
            return
        try:
            info = library.import_colormap_file(path, kind=self.kind_combo.currentData())
        except Exception as exc:
            show_status_message(self._window, f"Import failed: {exc}", timeout=4000)
            return
        self._reload_list(select=info.name)
        show_status_message(self._window, f"Imported colormap “{info.name}”.", timeout=2500)

    def _export_map(self, _checked=False):
        name = self._selected_name()
        if name is None:
            return
        path, _ = get_save_file_name(self, "Export colormap", f"{name}.json", "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            library.export_colormap(name, path)
        except Exception as exc:
            show_status_message(self._window, f"Export failed: {exc}", timeout=4000)
            return
        show_status_message(self._window, f"Exported {path}", timeout=2500)

    def _delete_map(self, _checked=False):
        name = self._selected_name()
        if name is None:
            return
        info = library.find_colormap(name)
        if info is None or info.source != "user":
            return
        library.delete_user_colormap(name)
        self._reload_list()
        show_status_message(self._window, f"Deleted colormap “{name}”.", timeout=2500)
