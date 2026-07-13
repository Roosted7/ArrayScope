"""Colormap designer: browse, edit, create, import and export colormaps.

Interaction model:

- Edits save automatically when you switch maps or close the dialog — no
  silent loss. Renaming a user map moves it (no stale pre-rename copy).
- Built-ins are editable too: saving stores a user override under the same
  name; ``Reset`` restores the system definition. ``Revert`` undoes the most
  recent change made in this session. Each button only appears when it
  leads to a different state.
- ``Apply`` saves, applies to the current view and closes; ``Done`` saves
  and closes. Apply is disabled (with an explanation) when the selected
  kind cannot be shown in the current viewing mode.
"""

from __future__ import annotations

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.display import colormap_library as library
from arrayscope.display.colormap_policy import colormap_family
from arrayscope.ui.file_dialogs import get_open_file_name, get_save_file_name
from arrayscope.ui.icons import set_button_icon
from arrayscope.ui.toasts import show_status_message


_KIND_LABELS = (
    ("Sequential", library.SEQUENTIAL),
    ("Diverging", library.DIVERGING),
    ("Cyclic (phase-safe)", library.CYCLIC),
)
_SORT_MODES = (("Group", "group"), ("Name", "name"), ("Kind", "kind"))


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
    def __init__(self, window):
        super().__init__(window)
        self._window = window
        self.setWindowTitle("Colormaps")
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.resize(720, 460)
        self._loaded = None  # (name, kind, stops, source)
        self._revert_snapshot = None  # (name, payload-or-None) set on each save
        self._suppress_dirty = False

        root = QtWidgets.QHBoxLayout(self)

        # -- left: library list ---------------------------------------------
        left = QtWidgets.QVBoxLayout()
        sort_row = QtWidgets.QHBoxLayout()
        sort_row.addWidget(QtWidgets.QLabel("Sort by"))
        self.sort_combo = QtWidgets.QComboBox(self)
        for label, mode in _SORT_MODES:
            self.sort_combo.addItem(label, mode)
        sort_row.addWidget(self.sort_combo)
        sort_row.addStretch(1)
        left.addLayout(sort_row)
        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setIconSize(QtCore.QSize(96, 14))
        self.list_widget.setMinimumWidth(250)
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
        set_button_icon(self.delete_button, "delete", tooltip="Delete (built-ins are hidden and can be restored)")
        for button in (self.new_button, self.duplicate_button, self.import_button, self.export_button, self.delete_button):
            list_buttons.addWidget(button)
        list_buttons.addStretch(1)
        left.addLayout(list_buttons)
        root.addLayout(left)

        # -- right: editor -----------------------------------------------------
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
        self.source_label.setWordWrap(True)
        right.addWidget(self.source_label)
        self.conflict_label = QtWidgets.QLabel("")
        self.conflict_label.setObjectName("OperationsMetaLabel")
        self.conflict_label.setWordWrap(True)
        right.addWidget(self.conflict_label)
        right.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        self.revert_button = QtWidgets.QPushButton("Revert", self)
        set_button_icon(self.revert_button, "undo")
        self.revert_button.setToolTip("Undo the most recent change made in this session")
        self.reset_button = QtWidgets.QPushButton("Reset", self)
        set_button_icon(self.reset_button, "reset_wrench")
        self.reset_button.setToolTip("Restore the built-in system definition")
        actions.addWidget(self.revert_button)
        actions.addWidget(self.reset_button)
        actions.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("Apply", self)
        set_button_icon(self.apply_button, "done")
        self.apply_button.setToolTip("Save, apply to the current view, and close")
        self.done_button = QtWidgets.QPushButton("Done", self)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.done_button)
        right.addLayout(actions)
        root.addLayout(right, 1)

        self.sort_combo.currentIndexChanged.connect(lambda *_: self._reload_list(select=self._selected_name()))
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        self.new_button.clicked.connect(self._new_map)
        self.duplicate_button.clicked.connect(self._duplicate_map)
        self.import_button.clicked.connect(self._import_map)
        self.export_button.clicked.connect(self._export_map)
        self.delete_button.clicked.connect(self._delete_map)
        self.revert_button.clicked.connect(self._revert)
        self.reset_button.clicked.connect(self._reset)
        self.apply_button.clicked.connect(self._apply_and_close)
        self.done_button.clicked.connect(self._done)
        self.kind_combo.currentIndexChanged.connect(lambda *_: self._sync_action_states())
        self.name_edit.textEdited.connect(lambda *_: self._sync_action_states())

        self._reload_list(select=getattr(window, "current_colormap", None))

    # ------------------------------------------------------------------
    # List handling
    # ------------------------------------------------------------------

    def _reload_list(self, select=None):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        mode = self.sort_combo.currentData() or "group"
        infos = list(library.list_colormaps(include_hidden=True))
        if mode == "name":
            infos.sort(key=lambda info: info.name.lower())
            self._add_items(infos)
        elif mode == "kind":
            for kind in library.KINDS:
                subset = sorted((i for i in infos if i.kind == kind), key=lambda i: i.name.lower())
                if subset:
                    self._add_header(kind.capitalize())
                    self._add_items(subset)
        else:
            for group, subset in library.grouped_colormaps(include_hidden=True):
                self._add_header(group)
                self._add_items(subset)
        self.list_widget.blockSignals(False)
        target = str(select) if select is not None else None
        selected = False
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == target:
                self.list_widget.setCurrentRow(row)
                selected = True
                break
        if not selected:
            for row in range(self.list_widget.count()):
                if self.list_widget.item(row).data(QtCore.Qt.ItemDataRole.UserRole):
                    self.list_widget.setCurrentRow(row)
                    break
        self._load_selected()

    def _add_header(self, text):
        item = QtWidgets.QListWidgetItem(str(text))
        item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        self.list_widget.addItem(item)

    def _add_items(self, infos):
        for info in infos:
            suffix = ""
            if info.hidden:
                suffix = "  (hidden)"
            elif info.source == "user" and library.overrides_builtin(info.name):
                suffix = "  (modified)"
            elif info.source == "user":
                suffix = "  (user)"
            item = QtWidgets.QListWidgetItem(_preview_icon(info.name), f"{info.name}{suffix}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, info.name)
            item.setToolTip(f"{info.kind} · {info.group}")
            if info.hidden:
                item.setForeground(self.palette().brush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text))
            self.list_widget.addItem(item)

    def _selected_name(self):
        item = self.list_widget.currentItem()
        if item is None:
            return None
        value = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return None if value is None else str(value)

    # ------------------------------------------------------------------
    # Editor state
    # ------------------------------------------------------------------

    def _editor_state(self):
        return (
            self.name_edit.text().strip(),
            self.kind_combo.currentData(),
            self._gradient_stops(),
        )

    def _editor_dirty(self) -> bool:
        if self._loaded is None:
            return False
        name, kind, stops, _source = self._loaded
        current_name, current_kind, current_stops = self._editor_state()
        return bool(current_name) and (current_name, current_kind, current_stops) != (name, kind, stops)

    def _on_selection_changed(self, *_args):
        self._autosave_pending_edits()
        self._load_selected()

    def _load_selected(self):
        name = self._selected_name()
        if name is None:
            return
        info = library.find_colormap(name)
        if info is None:
            return
        stops = library._normalize_stops(library.colormap_stops(info.name, points=17))
        self._suppress_dirty = True
        try:
            self.name_edit.setText(info.name)
            self.kind_combo.setCurrentIndex(max(0, self.kind_combo.findData(info.kind)))
            self._set_gradient_stops(stops)
        finally:
            self._suppress_dirty = False
        self._loaded = (info.name, info.kind, self._gradient_stops(), info.source)
        self._sync_action_states()

    def _set_gradient_stops(self, stops):
        positions = [position for position, _color in stops]
        colors = [tuple(color) + (255,) for _position, color in stops]
        self.gradient.item.setColorMap(pg.ColorMap(positions, colors))

    def _gradient_stops(self):
        colormap = self.gradient.item.colorMap()
        positions, colors = colormap.getStops(mode=pg.ColorMap.BYTE)
        return library._normalize_stops(
            (float(position), (int(color[0]), int(color[1]), int(color[2])))
            for position, color in zip(positions, colors)
        )

    # ------------------------------------------------------------------
    # Saving / revert / reset
    # ------------------------------------------------------------------

    def _snapshot_before_change(self, name):
        info = library.find_colormap(name)
        payload = None
        if info is not None and info.source == "user":
            payload = (info.kind, info.stops)
        self._revert_snapshot = (str(name), payload, str(name) in library.hidden_builtins())

    def _autosave_pending_edits(self):
        """Edits persist when leaving a map; renames move the file."""
        if self._suppress_dirty or not self._editor_dirty():
            return
        loaded_name, _kind, _stops, loaded_source = self._loaded
        new_name, new_kind, new_stops = self._editor_state()
        self._snapshot_before_change(new_name)
        try:
            library.save_user_colormap(new_name, new_kind, new_stops)
        except Exception as exc:
            show_status_message(self._window, f"Could not save colormap: {exc}", timeout=3500)
            return
        if loaded_source == "user" and new_name != loaded_name:
            library.delete_user_colormap(loaded_name)
        self._loaded = (new_name, new_kind, new_stops, "user")
        self._refresh_previews(select=new_name)

    def _refresh_previews(self, select=None):
        self.list_widget.blockSignals(True)
        self._reload_list(select=select or self._selected_name())
        self.list_widget.blockSignals(False)

    def _revert(self, _checked=False):
        if self._revert_snapshot is None:
            return
        name, payload, was_hidden = self._revert_snapshot
        self._revert_snapshot = None
        if payload is None:
            library.delete_user_colormap(name)
        else:
            kind, stops = payload
            library.save_user_colormap(name, kind, stops)
        if was_hidden != (name in library.hidden_builtins()):
            library.set_builtin_hidden(name, was_hidden)
        self._loaded = None
        self._reload_list(select=name if library.find_colormap(name) else None)
        show_status_message(self._window, f"Reverted the last change to “{name}”.", timeout=2500)

    def _reset(self, _checked=False):
        name = self._selected_name()
        if name is None:
            return
        self._snapshot_before_change(name)
        if library.reset_builtin(name):
            self._loaded = None
            self._reload_list(select=name)
            show_status_message(self._window, f"Restored “{name}” to the system definition.", timeout=2500)

    def _apply_and_close(self, _checked=False):
        self._autosave_pending_edits()
        name = self.name_edit.text().strip() or self._selected_name()
        if name:
            setter = getattr(self._window, "_set_display_colormap", None)
            if callable(setter):
                setter(name, user_selected=True, request_render=True)
        self.accept()

    def _done(self, _checked=False):
        self._autosave_pending_edits()
        self.accept()

    def closeEvent(self, event):
        self._autosave_pending_edits()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Button states
    # ------------------------------------------------------------------

    def _sync_action_states(self):
        name = self._selected_name()
        info = None if name is None else library.find_colormap(name)

        # Reset: only when the selection deviates from a system definition.
        resettable = info is not None and (library.overrides_builtin(info.name) or info.hidden)
        # Revert: only when this session recorded a change to undo — and not
        # when it would land on the same state Reset produces.
        revertable = self._revert_snapshot is not None
        if revertable and resettable and info is not None:
            snap_name, payload, was_hidden = self._revert_snapshot
            if snap_name == info.name and payload is None and not was_hidden:
                revertable = False
        self.revert_button.setVisible(revertable)
        self.reset_button.setVisible(resettable)
        self.delete_button.setEnabled(info is not None and not (info.source == "builtin" and info.hidden))

        # Apply: blocked when the kind cannot be presented right now.
        window = self._window
        family = None
        try:
            family = colormap_family(window.view_state.channel)
        except Exception:
            pass
        kind = self.kind_combo.currentData()
        compatible = family is None or kind in library.kinds_for_family(family)
        self.apply_button.setEnabled(compatible)
        if compatible:
            self.conflict_label.setText("")
        elif family == "phase":
            self.conflict_label.setText(
                "The current view shows complex/phase data, which needs a cyclic colormap — Apply is disabled."
            )
        else:
            self.conflict_label.setText(
                "Cyclic colormaps are reserved for complex/phase display — Apply is disabled for the current view."
            )

        if info is None:
            self.source_label.setText("")
        elif info.hidden:
            self.source_label.setText("Hidden built-in — Reset restores it to the library.")
        elif library.overrides_builtin(info.name):
            self.source_label.setText("Modified built-in — edits persist as your override; Reset restores the system definition.")
        elif info.source == "user":
            self.source_label.setText("User colormap — edits save automatically when you switch maps or close.")
        else:
            self.source_label.setText("Built-in colormap — editing it stores your override under the same name.")

    # ------------------------------------------------------------------
    # List actions
    # ------------------------------------------------------------------

    def _new_map(self, _checked=False):
        self._autosave_pending_edits()
        name = self._unique_name("custom")
        library.save_user_colormap(name, library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255))))
        self._reload_list(select=name)

    def _duplicate_map(self, _checked=False):
        self._autosave_pending_edits()
        source = self._selected_name()
        if source is None:
            return
        info = library.find_colormap(source)
        name = self._unique_name(source)
        library.save_user_colormap(name, info.kind, library.colormap_stops(source, points=17))
        self._reload_list(select=name)

    def _unique_name(self, base):
        existing = {info.name for info in library.list_colormaps(include_hidden=True)}
        if base not in existing:
            return base
        for index in range(2, 100):
            candidate = f"{base}-{index}"
            if candidate not in existing:
                return candidate
        return f"{base}-copy"

    def _import_map(self, _checked=False):
        self._autosave_pending_edits()
        path, _ = get_open_file_name(
            self,
            "Import colormap",
            "",
            "Colormaps (*.json *.mat *.csv *.txt *.npy);;All files (*)",
        )
        if not path:
            return
        try:
            info = library.import_colormap_file(path)
        except Exception as exc:
            show_status_message(self._window, f"Import failed: {exc}", timeout=4000)
            return
        self._reload_list(select=info.name)
        show_status_message(
            self._window,
            f"Imported “{info.name}” as {info.kind} — adjust the kind if the guess is wrong.",
            timeout=3500,
        )

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
        if info is None:
            return
        self._snapshot_before_change(name)
        if info.source == "user":
            library.delete_user_colormap(name)
            message = f"Deleted colormap “{name}”."
            if library.builtin_group_for(name) is not None:
                message = f"Removed the override — “{name}” is back to the system definition."
        else:
            library.set_builtin_hidden(name, True)
            message = f"Hid built-in “{name}” — select it and press Reset to restore."
        self._loaded = None
        self._reload_list()
        show_status_message(self._window, message, timeout=3000)
