"""Colormap manager: browse, arrange, edit, create, import and export.

The colormap half of the shared manager grammar
(:mod:`arrayscope.ui.user_library_manager`, ADR 0063). The tree, the button bar
and its icons, the filter row, the markers, Revert, the Problems group, group
rename and layout persistence all live in the base; this module contributes the
gradient editor and what "apply" means for a colormap.

Interaction model:

- The library tree is drag-reorderable: maps move within and between groups,
  groups reorder and rename inline; the arrangement persists and drives the
  toolbar picker. The Favorites section (top, unnamed) holds the go-to maps.
- The *kind* (sequential/diverging/cyclic) is metadata, not a group -- use the
  filter above the tree to narrow the list, or type to search by name.
- Built-in definitions are read-only, exactly like a system operation:
  ``Override built-in`` makes an editable user copy under the same name and
  ``Restore`` puts the system definition back. The override used to happen
  silently on the first keystroke, which is how a user modified a built-in
  without ever deciding to.
- Edits to a user map save automatically when you switch maps or close -- no
  silent loss. Renaming one moves it (no stale pre-rename copy). ``Revert``
  undoes the most recent change made in this session.
- ``Apply`` saves, applies to the current view and closes; ``Done`` saves and
  closes. Apply is disabled (with an explanation) when the selected kind cannot
  be shown in the current viewing mode.
"""

from __future__ import annotations

import contextlib
import json
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.display import colormap_library as library
from arrayscope.display.colormap_policy import colormap_family
from arrayscope.ui.file_dialogs import get_open_file_name, get_save_file_name
from arrayscope.ui.icons import set_button_icon
from arrayscope.ui.user_library_manager import ITEM_ROLE, UserLibraryManagerDialog, unique_name

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


class ColormapManagerDialog(UserLibraryManagerDialog):
    ITEM_NOUN = "colormap"
    FILTER_CHOICES = (("All kinds", None), *_KIND_LABELS)
    # Favorites is the unnamed section pickers render ungrouped at the top;
    # renaming it would turn it into an ordinary group and lose that meaning.
    PROTECTED_GROUPS = (library.FAVORITES_GROUP,)
    TREE_MINIMUM_WIDTH = 260
    TREE_ICON_SIZE = QtCore.QSize(96, 14)

    def __init__(self, window):
        self._loaded = None  # (name, kind, stops, source)
        self._loaded_problem = None
        super().__init__(window, title="Colormaps")
        self.resize(760, 520)
        self.setMinimumSize(700, 460)
        self.start_listening()
        selected = getattr(window, "current_colormap", None)
        if selected is not None:
            self.select_key(str(selected))

    # ------------------------------------------------------------------
    # Editor pane
    # ------------------------------------------------------------------

    def build_editor(self, root):
        right_widget = QtWidgets.QWidget(self)
        right = QtWidgets.QVBoxLayout(right_widget)
        self.editor_scroll = QtWidgets.QScrollArea(self)
        self.editor_scroll.setWidgetResizable(True)
        self.editor_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.editor_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.editor_scroll.setWidget(right_widget)

        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit(self)
        form.addRow("Name", self.name_edit)
        self.kind_combo = QtWidgets.QComboBox(self)
        for label, kind in _KIND_LABELS:
            self.kind_combo.addItem(label, kind)
        self.kind_combo.setToolTip(
            "Cyclic maps wrap smoothly and are required for phase/complex display"
        )
        form.addRow("Kind", self.kind_combo)
        right.addLayout(form)

        right.addWidget(
            QtWidgets.QLabel("Stops — drag to move, double-click to recolor, click the bar to add:")
        )
        self.gradient = pg.GradientWidget(orientation="bottom")
        self.gradient.setMinimumHeight(58)
        right.addWidget(self.gradient)

        # -- repair pane (Problems rows only) ------------------------------
        # A colormap file the library could not read is edited here, in the same
        # column, rather than being silently dropped. There is no form to drive
        # from a payload with no usable stops, so the honest editor is the text
        # itself (ADR 0062's second repair shape).
        self.repair_box = QtWidgets.QGroupBox("Repair this stored colormap", self)
        repair_layout = QtWidgets.QVBoxLayout(self.repair_box)
        self.problem_message_label = QtWidgets.QLabel("", self)
        self.problem_message_label.setObjectName("OperationsProblemLabel")
        self.problem_message_label.setWordWrap(True)
        self.problem_message_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        repair_layout.addWidget(self.problem_message_label)
        self.problem_path_label = QtWidgets.QLabel("", self)
        self.problem_path_label.setObjectName("OperationsMetaLabel")
        self.problem_path_label.setWordWrap(True)
        self.problem_path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        repair_layout.addWidget(self.problem_path_label)
        self.raw_json_edit = QtWidgets.QPlainTextEdit(self)
        self.raw_json_edit.setPlaceholderText("Colormap JSON")
        self.raw_json_edit.setMinimumHeight(160)
        repair_layout.addWidget(self.raw_json_edit, 1)
        raw_buttons = QtWidgets.QHBoxLayout()
        self.raw_json_save_button = QtWidgets.QPushButton("Save colormap JSON", self)
        raw_buttons.addWidget(self.raw_json_save_button)
        raw_buttons.addStretch(1)
        repair_layout.addLayout(raw_buttons)
        self.repair_box.setVisible(False)
        right.addWidget(self.repair_box)

        self.source_label = QtWidgets.QLabel("", self)
        self.source_label.setObjectName("OperationsMetaLabel")
        self.source_label.setWordWrap(True)
        right.addWidget(self.source_label)
        self.conflict_label = QtWidgets.QLabel("", self)
        self.conflict_label.setObjectName("OperationsMetaLabel")
        self.conflict_label.setWordWrap(True)
        right.addWidget(self.conflict_label)

        self.override_button = QtWidgets.QPushButton("Override built-in", self)
        set_button_icon(self.override_button, "content_copy")
        self.override_button.setToolTip(
            "Store an editable copy under this name; Restore puts the system definition back"
        )
        override_row = QtWidgets.QHBoxLayout()
        override_row.addWidget(self.override_button)
        override_row.addStretch(1)
        right.addLayout(override_row)
        right.addStretch(1)

        actions = QtWidgets.QHBoxLayout()
        actions.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("Apply", self)
        set_button_icon(self.apply_button, "done")
        self.apply_button.setToolTip("Save, apply to the current view, and close")
        self.done_button = QtWidgets.QPushButton("Done", self)
        actions.addWidget(self.apply_button)
        actions.addWidget(self.done_button)
        right.addLayout(actions)
        root.addWidget(self.editor_scroll, 1)

        self.override_button.clicked.connect(self._override_builtin)
        self.raw_json_save_button.clicked.connect(self._save_raw_colormap)
        self.apply_button.clicked.connect(self._apply_and_close)
        self.done_button.clicked.connect(self.accept)
        self.kind_combo.currentIndexChanged.connect(lambda *_: self._sync_action_states())
        self.name_edit.textEdited.connect(lambda *_: self._sync_action_states())

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def populate_tree(self):
        for group, infos in library.grouped_colormaps(
            include_hidden=True, include_unavailable=True
        ):
            group_item = self.group_item(
                group, label=library.FAVORITES_LABEL if group == library.FAVORITES_GROUP else group
            )
            for info in infos:
                group_item.addChild(
                    self.item_row(
                        info.name,
                        info.name,
                        icon=_preview_icon(info.name),
                        tooltip=info.unavailable_reason
                        or f"{info.kind} · {'user' if info.source == 'user' else 'built-in'}",
                        hidden=info.hidden,
                        user=info.source == "user",
                        modified=info.source == "user" and library.overrides_builtin(info.name),
                        unavailable=bool(info.unavailable_reason),
                    )
                )
            group_item.setExpanded(True)

    def row_matches(self, item, text, choice, group):
        info = library.find_colormap(str(item.data(0, ITEM_ROLE)))
        if choice is not None and (info is None or info.kind != choice):
            return False
        return not text or text in item.text(0).casefold()

    # Compatibility names for callers that speak colormaps rather than the
    # shared vocabulary.
    def select_map(self, name: str) -> bool:
        return self.select_key(name)

    def selected_map_name(self):
        return self.selected_key()

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
        if self._loaded is None or self._loaded[3] != "user":
            # A built-in is read-only until Override built-in says otherwise, so
            # there is no such thing as a pending edit to one.
            return False
        name, kind, stops, _source = self._loaded
        current_name, current_kind, current_stops = self._editor_state()
        return bool(current_name) and (current_name, current_kind, current_stops) != (
            name,
            kind,
            stops,
        )

    def load_editor(self):
        if self._reloading:
            return
        problem_path = self.selected_problem_path()
        self._loaded_problem = (
            self.problem_for_path(problem_path) if problem_path is not None else None
        )
        if self._loaded_problem is not None:
            self._load_repair_editor(self._loaded_problem)
            return
        self.repair_box.setVisible(False)
        name = self.selected_key()
        info = None if name is None else library.find_colormap(name)
        if info is None:
            self._loaded = None
            self._sync_action_states()
            return
        try:
            stops = library._normalize_stops(library.colormap_stops(info.name, points=17))
        except Exception:
            # An unavailable built-in has no table to sample; the row still has
            # to be selectable, because that is where its reason is explained.
            stops = ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
        self._suppress = True
        try:
            self.name_edit.setText(info.name)
            self.kind_combo.setCurrentIndex(max(0, self.kind_combo.findData(info.kind)))
            self._set_gradient_stops(stops)
        finally:
            self._suppress = False
        self._loaded = (info.name, info.kind, self._gradient_stops(), info.source)
        self._sync_action_states()

    def _load_repair_editor(self, problem):
        self.repair_box.setVisible(True)
        self._loaded = None
        self.problem_message_label.setText(
            problem.message
            if problem.payload is not None
            # A bare JSON decoder message says nothing about what it is a
            # message *about*.
            else f"Unreadable colormap file — {problem.message}"
        )
        self.problem_path_label.setText(problem.path)
        self._suppress = True
        try:
            try:
                with open(problem.path) as handle:
                    self.raw_json_edit.setPlainText(handle.read())
            except OSError as exc:
                self.raw_json_edit.setPlainText("")
                self.problem_path_label.setText(f"{problem.path} — {exc}")
        finally:
            self._suppress = False
        self.source_label.setText(
            "Stored but not loaded — fix it here and save, open it in an editor, or remove it."
        )
        self.conflict_label.setText("")
        self._sync_action_states()

    def _save_raw_colormap(self):
        problem = self._loaded_problem
        if problem is None:
            return
        try:
            payload = json.loads(self.raw_json_edit.toPlainText())
        except ValueError as exc:
            self.problem_message_label.setText(f"Not valid JSON yet — {exc}")
            return
        if not isinstance(payload, dict):
            self.problem_message_label.setText(
                "A colormap must be a JSON object (a single {…} document)."
            )
            return
        # Adopt the name the user just typed *before* the rescan: a repair that
        # works removes this row, and the selection has to follow the definition.
        claimed = payload.get("name")
        self._loaded_problem = problem._replace(
            payload=payload, item_id=str(claimed) if isinstance(claimed, str) and claimed else None
        )
        if library.update_user_colormap_file(problem.path, payload):
            self.status("Saved the colormap and rescanned.", timeout=2500)

    def repaired_item_id(self):
        return self._loaded_problem.item_id if self._loaded_problem else None

    def _set_gradient_stops(self, stops):
        positions = [position for position, _color in stops]
        colors = [(*tuple(color), 255) for _position, color in stops]
        self.gradient.item.setColorMap(pg.ColorMap(positions, colors))

    def _gradient_stops(self):
        colormap = self.gradient.item.colorMap()
        positions, colors = colormap.getStops(mode=pg.ColorMap.BYTE)
        return library._normalize_stops(
            (float(position), (int(color[0]), int(color[1]), int(color[2])))
            for position, color in zip(positions, colors, strict=False)
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def autosave_pending_edits(self):
        if self._suppress or not self._editor_dirty():
            return
        loaded_name, _kind, _stops, loaded_source = self._loaded
        new_name, new_kind, new_stops = self._editor_state()
        self.snapshot_before_change(new_name)
        # Adopt the saved state *before* the library call: saving notifies, and
        # the notification lands back here as a tree rebuild.
        self._loaded = (new_name, new_kind, new_stops, "user")
        try:
            library.save_user_colormap(new_name, new_kind, new_stops)
        except Exception as exc:
            self._loaded = (loaded_name, _kind, _stops, loaded_source)
            self.status(f"Could not save colormap: {exc}", timeout=3500)
            return
        if loaded_source == "user" and new_name != loaded_name:
            library.delete_user_colormap(loaded_name)
        self.refresh_tree()
        self.select_key(new_name)

    def _override_builtin(self, _checked=False):
        """Make the read-only built-in an editable user copy under the same name."""

        name = self.selected_key()
        info = None if name is None else library.find_colormap(name)
        if info is None or info.source == "user":
            return
        self.snapshot_before_change(name)
        library.save_user_colormap(name, info.kind, library.colormap_stops(name, points=17))
        self.refresh_tree()
        self.select_key(name)
        self.status(f"“{name}” is now yours to edit — Restore puts the built-in back.")

    def _apply_and_close(self, _checked=False):
        self.save_pending_edits()
        name = self.name_edit.text().strip() or self.selected_key()
        if name:
            setter = getattr(self._window, "_set_display_colormap", None)
            if callable(setter):
                setter(name, user_selected=True, request_render=True)
        self.accept()

    # ------------------------------------------------------------------
    # Button states
    # ------------------------------------------------------------------

    def _sync_action_states(self):
        problem_path = self.selected_problem_path()
        name = self.selected_key()
        info = None if name is None else library.find_colormap(name)
        is_user = info is not None and info.source == "user"
        is_builtin = info is not None and info.source == "builtin"

        for widget in (self.name_edit, self.kind_combo, self.gradient):
            widget.setEnabled(is_user)
        self.override_button.setVisible(bool(is_builtin and not info.hidden))
        self.override_button.setEnabled(bool(is_builtin and not info.unavailable_reason))

        restorable = info is not None and (library.overrides_builtin(info.name) or info.hidden)
        self.unhide_button.setEnabled(bool(restorable))
        self.duplicate_button.setEnabled(info is not None and not info.unavailable_reason)
        self.export_button.setEnabled(info is not None and not info.unavailable_reason)
        self.remove_button.setEnabled(
            problem_path is not None
            or (info is not None and not (info.source == "builtin" and info.hidden))
        )
        self._set_remove_affordance(destructive=problem_path is not None or is_user)
        self.open_file_button.setEnabled(problem_path is not None or is_user)
        self.sync_revert_button()

        family = None
        with contextlib.suppress(Exception):
            family = colormap_family(self._window.view_state.channel)
        kind = self.kind_combo.currentData()
        compatible = family is None or kind in library.kinds_for_family(family)
        self.apply_button.setEnabled(bool(compatible and info is not None))
        if compatible:
            self.conflict_label.setText("")
        elif family == "phase":
            self.conflict_label.setText(
                "The current view shows complex/phase data, which needs a cyclic colormap "
                "— Apply is disabled."
            )
        else:
            self.conflict_label.setText(
                "Cyclic colormaps are reserved for complex/phase display — Apply is disabled "
                "for the current view."
            )

        if problem_path is not None:
            return
        if info is None:
            self.source_label.setText("")
        elif info.unavailable_reason:
            self.source_label.setText(f"Unavailable — {info.unavailable_reason}")
        elif info.hidden:
            self.source_label.setText("Hidden built-in — Restore returns it to the library.")
        elif library.overrides_builtin(info.name):
            self.source_label.setText(
                "Modified built-in — your override; Restore brings back the system definition."
            )
        elif info.source == "user":
            self.source_label.setText(
                "User colormap — edits save automatically when you switch maps or close."
            )
        else:
            self.source_label.setText(
                "Built-in colormap — read-only. Override built-in makes an editable copy."
            )

    # ------------------------------------------------------------------
    # The shared library contract
    # ------------------------------------------------------------------

    def library_directory(self):
        return library.user_colormap_directory()

    def library_add_listener(self, callback):
        library.add_library_listener(callback)

    def library_remove_listener(self, callback):
        library.remove_library_listener(callback)

    def library_problems(self):
        return library.user_colormap_problems()

    def library_remove_problem_file(self, path):
        return library.remove_user_colormap_file(path)

    def library_apply_layout(self, group_order, groups, order):
        library.apply_library_layout(group_order, groups, order)

    def library_rename_group(self, old, new):
        return library.rename_group(old, new)

    def library_is_hidden(self, key):
        return str(key) in library.hidden_builtins()

    def library_snapshot_payload(self, key):
        info = library.find_colormap(key)
        return (info.kind, info.stops) if info is not None and info.source == "user" else None

    def library_restore_snapshot(self, key, payload, was_hidden):
        if payload is None:
            library.delete_user_colormap(key)
        else:
            kind, stops = payload
            library.save_user_colormap(key, kind, stops)
        if was_hidden != (key in library.hidden_builtins()):
            library.set_builtin_hidden(key, was_hidden)
        self._loaded = None

    def library_new(self):
        name = unique_name("custom", [info.name for info in self._all_infos()], copy_suffix=False)
        library.save_user_colormap(
            name, library.SEQUENTIAL, ((0.0, (0, 0, 0)), (1.0, (255, 255, 255)))
        )
        self.refresh_tree()
        self.select_key(name)

    def library_duplicate(self):
        source = self.selected_key()
        info = None if source is None else library.find_colormap(source)
        if info is None:
            return
        name = unique_name(source, [existing.name for existing in self._all_infos()])
        library.save_user_colormap(name, info.kind, library.colormap_stops(source, points=17))
        self.refresh_tree()
        self.select_key(name)

    def library_import(self):
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
            self.status(f"Import failed: {exc}", timeout=4000)
            return
        self.refresh_tree()
        self.select_key(info.name)
        self.status(
            f"Imported “{info.name}” as {info.kind} — adjust the kind if the guess is wrong.",
            timeout=3500,
        )

    def library_export(self):
        name = self.selected_key()
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
            self.status(f"Export failed: {exc}", timeout=4000)
            return
        self.status(f"Exported {path}", timeout=2500)

    def library_delete_or_hide(self):
        name = self.selected_key()
        info = None if name is None else library.find_colormap(name)
        if info is None:
            return
        if info.source == "user":
            question = f"Delete the colormap “{name}” and its file?"
            if library.builtin_group_for(name) is not None:
                question = f"Remove your override of “{name}” and restore the built-in?"
        else:
            question = f"Hide the built-in “{name}”? Restore brings it back."
        if not self.confirm("Remove colormap", question):
            return
        self.snapshot_before_change(name)
        if info.source == "user":
            library.delete_user_colormap(name)
            message = f"Deleted colormap “{name}”."
            if library.builtin_group_for(name) is not None:
                message = f"Removed the override — “{name}” is back to the system definition."
        else:
            library.set_builtin_hidden(name, True)
            message = f"Hid built-in “{name}” — select it and press Restore."
        self._loaded = None
        self.status(message)

    def library_restore(self):
        name = self.selected_key()
        if name is None:
            return
        self.snapshot_before_change(name)
        if library.reset_builtin(name):
            self._loaded = None
            self.status(f"Restored “{name}” to the system definition.", timeout=2500)

    def library_open_file(self):
        problem_path = self.selected_problem_path()
        if problem_path is not None:
            self.open_path(problem_path)
            return
        name = self.selected_key()
        info = None if name is None else library.find_colormap(name)
        if info is None or info.source != "user":
            return
        path = library.user_colormap_path(name)
        if not os.path.exists(path):
            self.status("Could not locate the colormap file.", timeout=3000)
            return
        self.open_path(path)

    def library_reset_all(self):
        for name in tuple(library.hidden_builtins()):
            library.set_builtin_hidden(name, False)
        library.reset_layout()

    @staticmethod
    def _all_infos():
        return library.list_colormaps(include_hidden=True, include_unavailable=True)
