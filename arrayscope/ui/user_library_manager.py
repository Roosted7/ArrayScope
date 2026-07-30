"""One manager grammar for every user-editable library (ADR 0063).

ArrayScope has two libraries the user browses, rearranges, extends and breaks:
colormaps and operations.  Their managers were written eighteen months apart
and had converged by copy-paste on the same tree widget and the same layout
persistence while diverging on everything a person actually notices --
one had Revert and a filter, the other had Problems, confirmation and
Restore; the same ``folder_open`` glyph meant Import on one side and
Open-folder on the other; the same concept was "Favorites" here and "Common"
there.

This module is the shared half: the tree, the button bar *in a fixed order
with a fixed icon vocabulary*, the marker vocabulary, the filter row, group
rename, layout persistence, the Problems group, the one-step session Revert,
listener registration and selection preservation.  Each manager subclasses it
and contributes only its editor pane and its library calls.

The vocabulary is not documentation, it is :data:`LIBRARY_BUTTONS`, and
``tests/ui/test_user_library_manager.py`` asserts both dialogs expose exactly
it, in order.  Drift fails a test rather than accumulating.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.ui.icons import material_icon, set_button_icon
from arrayscope.ui.toasts import show_status_message

# The item's own identity (a colormap name, an operation id).
ITEM_ROLE = QtCore.Qt.ItemDataRole.UserRole
GROUP_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
# Set on a group that is not part of the arrangement -- today only "Problems".
VIRTUAL_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2
# A Problems row's identity is its file path, never an item id: the id lives
# inside a payload the loader could not accept.  ADR 0062.
PROBLEM_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole + 3

PROBLEMS_GROUP = "Problems"

# One vocabulary of row markers, in this order, italic + grey for the two that
# mean "this row is not currently in play".
MARKER_HIDDEN = "(hidden)"
MARKER_USER = "(user)"
MARKER_MODIFIED = "(modified)"
MARKER_UNAVAILABLE = "(unavailable)"


class ButtonSpec(NamedTuple):
    """One button of the shared bar: attribute name, icon, tooltip, visible text."""

    name: str
    icon: str
    tooltip: str
    text: str = ""


# THE button bar, in THE order, with THE icons.  Two rows: what makes a
# definition, then what happens to the selected one.
#
# Two collisions are settled here, repo-wide.  ``folder_open`` means *open the
# folder* on both sides -- it used to mean Import in the colormap designer,
# which is why Import is ``upload``.  And ``data_object`` (a JSON-ish glyph)
# was Duplicate on both sides while saying nothing about duplication; Duplicate
# is ``content_copy``.
CREATE_BUTTONS: tuple[ButtonSpec, ...] = (
    ButtonSpec("new_button", "add", "New {noun}", "New"),
    ButtonSpec("duplicate_button", "content_copy", "Duplicate the selected {noun}", "Duplicate"),
    ButtonSpec("import_button", "upload", "Import a {noun} from a file"),
    ButtonSpec("export_button", "download", "Export the selected {noun} to a file"),
)
ITEM_BUTTONS: tuple[ButtonSpec, ...] = (
    # The delete icon is dynamic (see ``_set_remove_affordance``): hiding a
    # system definition is reversible, so a trash can overstates it.
    ButtonSpec("remove_button", "delete", "Delete the selected {noun}"),
    ButtonSpec("unhide_button", "visibility", "Restore the selected system {noun}"),
    ButtonSpec("open_file_button", "edit", "Open the stored file"),
    ButtonSpec("open_folder_button", "folder_open", "Open the {noun} folder"),
    ButtonSpec("revert_button", "undo", "Undo the most recent change made in this session"),
    ButtonSpec("reset_all_button", "refresh", "Reset the arrangement and restore everything"),
)
LIBRARY_BUTTONS: tuple[ButtonSpec, ...] = CREATE_BUTTONS + ITEM_BUTTONS


def marker_suffix(
    *, hidden: bool = False, user: bool = False, modified: bool = False, unavailable: bool = False
) -> str:
    """The row suffix for one item, in the one vocabulary both managers speak."""

    markers = []
    if hidden:
        markers.append(MARKER_HIDDEN)
    if modified:
        markers.append(MARKER_MODIFIED)
    elif user:
        markers.append(MARKER_USER)
    if unavailable:
        markers.append(MARKER_UNAVAILABLE)
    return ("  " + " ".join(markers)) if markers else ""


class LibraryTree(QtWidgets.QTreeWidget):
    """Groups as parents, items as draggable children.  Two levels, always."""

    orderEdited = QtCore.Signal()

    def __init__(self, parent=None, *, icon_size: QtCore.QSize | None = None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        if icon_size is not None:
            self.setIconSize(icon_size)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setExpandsOnDoubleClick(False)

    def dropEvent(self, event):
        super().dropEvent(event)
        self._normalize()
        self.orderEdited.emit()

    def _normalize(self):
        """Keep the two-level shape: groups top-level, items under groups."""

        index = 0
        while index < self.topLevelItemCount():
            item = self.topLevelItem(index)
            if item.data(0, VIRTUAL_ROLE):
                # The Problems group is not part of the arrangement: its rows
                # carry a file path rather than an item id, so the id-shaped
                # normalization below would evict every one of them to the top
                # level on the next unrelated drop.
                index += 1
                continue
            if item.data(0, ITEM_ROLE):
                # An item dropped at the top level slides into the nearest group.
                self.takeTopLevelItem(index)
                target = self.topLevelItem(max(0, index - 1))
                if target is not None and not target.data(0, VIRTUAL_ROLE):
                    target.addChild(item)
                    target.setExpanded(True)
                continue
            child_index = 0
            while child_index < item.childCount():
                child = item.child(child_index)
                if child.data(0, ITEM_ROLE):
                    child_index += 1
                    continue
                # A group nested under a group moves back to the top level.
                item.takeChild(child_index)
                self.insertTopLevelItem(self.indexOfTopLevelItem(item) + 1, child)
            index += 1


class UserLibraryManagerDialog(QtWidgets.QDialog):
    """Base for the colormap and operation managers.

    Subclasses supply the library calls (everything named ``library_*``) and
    build their editor pane in :meth:`build_editor`; everything the two
    managers share is here and is therefore identical by construction.
    """

    ITEM_NOUN = "item"
    TREE_HINT = "Drag to rearrange · double-click a group to rename it"
    FILTER_LABEL = "Show"
    # ((label, data), …) -- the first entry is the "everything" choice.
    FILTER_CHOICES: tuple[tuple[str, object], ...] = (("All", None),)
    # Groups whose name is structural rather than chosen, so not renameable.
    PROTECTED_GROUPS: tuple[str, ...] = ()
    TREE_MINIMUM_WIDTH = 240
    TREE_ICON_SIZE: QtCore.QSize | None = None

    def __init__(self, window, *, title: str):
        super().__init__(window)
        self._window = window
        self.setWindowTitle(title)
        self.setWindowFlag(QtCore.Qt.WindowType.Tool, True)
        self.setSizeGripEnabled(True)
        self._reloading = False
        self._suppress = False
        self._autosaving = False
        self._revert_snapshot = None
        self._problems_by_path: dict = {}

        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(self._build_left_column())
        self.build_editor(root)

        self.filter_combo.currentIndexChanged.connect(lambda *_: self.apply_filters())
        self.filter_edit.textChanged.connect(lambda *_: self.apply_filters())
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        self.tree.orderEdited.connect(self.persist_layout)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.new_button.clicked.connect(lambda *_: self._guarded(self.library_new))
        self.duplicate_button.clicked.connect(lambda *_: self._guarded(self.library_duplicate))
        self.import_button.clicked.connect(lambda *_: self._guarded(self.library_import))
        self.export_button.clicked.connect(lambda *_: self.library_export())
        self.remove_button.clicked.connect(lambda *_: self._remove_or_hide())
        self.unhide_button.clicked.connect(lambda *_: self.library_restore())
        self.open_file_button.clicked.connect(lambda *_: self.library_open_file())
        self.open_folder_button.clicked.connect(lambda *_: self._open_folder())
        self.revert_button.clicked.connect(lambda *_: self._revert())
        self.reset_all_button.clicked.connect(lambda *_: self._reset_all())

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_left_column(self) -> QtWidgets.QVBoxLayout:
        left = QtWidgets.QVBoxLayout()

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel(self.FILTER_LABEL, self))
        self.filter_combo = QtWidgets.QComboBox(self)
        self.filter_combo.setObjectName("LibraryFilterCombo")
        for label, data in self.FILTER_CHOICES:
            self.filter_combo.addItem(label, data)
        filter_row.addWidget(self.filter_combo)
        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setObjectName("LibraryFilterEdit")
        self.filter_edit.setPlaceholderText("Search")
        self.filter_edit.setClearButtonEnabled(True)
        filter_row.addWidget(self.filter_edit, 1)
        left.addLayout(filter_row)

        self.tree = LibraryTree(self, icon_size=self.TREE_ICON_SIZE)
        self.tree.setMinimumWidth(self.TREE_MINIMUM_WIDTH)
        left.addWidget(self.tree, 1)
        hint = QtWidgets.QLabel(self.TREE_HINT, self)
        hint.setObjectName("OperationsMetaLabel")
        hint.setWordWrap(True)
        left.addWidget(hint)

        create_row = QtWidgets.QHBoxLayout()
        for spec in CREATE_BUTTONS:
            create_row.addWidget(self._make_button(spec))
        create_row.addStretch(1)
        left.addLayout(create_row)

        item_row = QtWidgets.QHBoxLayout()
        for spec in ITEM_BUTTONS:
            button = self._make_button(spec)
            if spec.name == "revert_button":
                item_row.addStretch(1)
            item_row.addWidget(button)
        left.addLayout(item_row)
        return left

    def _make_button(self, spec: ButtonSpec) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton(self)
        button.setObjectName(spec.name)
        if spec.text:
            button.setText(spec.text)
        set_button_icon(button, spec.icon, tooltip=spec.tooltip.format(noun=self.ITEM_NOUN))
        setattr(self, spec.name, button)
        return button

    def build_editor(self, root: QtWidgets.QHBoxLayout) -> None:
        """Add the manager's editor pane to ``root``.  Subclass responsibility."""

        raise NotImplementedError

    # ------------------------------------------------------------------
    # Teardown -- a listener that outlives its dialog is a call on a corpse
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self.save_pending_edits()
        self.library_remove_listener(self.refresh_tree)
        super().closeEvent(event)

    def accept(self):
        self.save_pending_edits()
        self.library_remove_listener(self.refresh_tree)
        super().accept()

    def reject(self):
        self.library_remove_listener(self.refresh_tree)
        super().reject()

    def start_listening(self) -> None:
        """Register the library listener and build the first tree.

        Called by the subclass at the *end* of its ``__init__``: the listener
        rebuilds the tree, which loads the editor, which needs the subclass's
        widgets to exist.
        """

        self.library_add_listener(self.refresh_tree)
        self.refresh_tree()

    # ------------------------------------------------------------------
    # Tree
    # ------------------------------------------------------------------

    def refresh_tree(self):
        """Rebuild from the library, preserving the selected *definition*."""

        selected = self.selected_key()
        problem_path = self.selected_problem_path()
        # A repair that works makes the row leave Problems, so "keep the
        # selection" has to mean the definition, not the row.
        repaired = self.repaired_item_id()
        self._reloading = True
        try:
            self.tree.clear()
            self.populate_tree()
            self._append_problems_group()
        finally:
            self._reloading = False
        self.apply_filters()
        restored = (
            (selected is not None and self.select_key(selected))
            or (problem_path is not None and self.select_problem_path(problem_path))
            or (repaired is not None and self.select_key(repaired))
        )
        if not restored:
            self._select_first_item()
        self.load_editor()

    def group_item(self, group: str, *, label: str | None = None) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([group if label is None else label])
        item.setData(0, GROUP_ROLE, group)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        flags = (
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsDropEnabled
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        if group not in self.PROTECTED_GROUPS:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        self.tree.addTopLevelItem(item)
        return item

    def item_row(
        self,
        key: str,
        label: str,
        *,
        icon: QtGui.QIcon | None = None,
        tooltip: str = "",
        hidden: bool = False,
        user: bool = False,
        modified: bool = False,
        unavailable: bool = False,
    ) -> QtWidgets.QTreeWidgetItem:
        """One library row, marked and styled in the shared vocabulary."""

        suffix = marker_suffix(hidden=hidden, user=user, modified=modified, unavailable=unavailable)
        item = QtWidgets.QTreeWidgetItem([f"{label}{suffix}"])
        if icon is not None:
            item.setIcon(0, icon)
        item.setData(0, ITEM_ROLE, key)
        item.setToolTip(0, tooltip or label)
        item.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        )
        if hidden or unavailable:
            item.setForeground(
                0,
                self.palette().brush(
                    QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text
                ),
            )
            font = item.font(0)
            font.setItalic(True)
            item.setFont(0, font)
        return item

    def _append_problems_group(self):
        """Stored definitions the library could not load -- as addressable rows."""

        self._problems_by_path = {}
        try:
            problems = self.library_problems()
        except Exception:
            problems = []
        if not problems:
            return
        group_item = self.group_item(PROBLEMS_GROUP)
        group_item.setData(0, VIRTUAL_ROLE, True)
        group_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
        for problem in problems:
            self._problems_by_path[problem.path] = problem
            label = problem.item_id or os.path.basename(problem.path)
            child = QtWidgets.QTreeWidgetItem([label])
            # Only the *group* is virtual.  The row itself is addressable by
            # file path, which is what makes it selectable, editable and
            # removable instead of a dead end (ADR 0062).
            child.setData(0, PROBLEM_PATH_ROLE, problem.path)
            child.setIcon(0, material_icon("warning"))
            child.setToolTip(0, f"{os.path.basename(problem.path)} — {problem.message}")
            child.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            group_item.addChild(child)
        group_item.setExpanded(True)

    def problem_for_path(self, path):
        return self._problems_by_path.get(path)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def selected_key(self):
        item = self.tree.currentItem()
        if item is None or item.data(0, VIRTUAL_ROLE):
            return None
        value = item.data(0, ITEM_ROLE)
        return None if value is None else str(value)

    def selected_problem_path(self):
        """File path of the selected Problems row, or ``None``.

        The sibling of :meth:`selected_key`: every action that used to
        early-return on ``None`` asks this second question before deciding the
        selection is unaddressable.
        """

        item = self.tree.currentItem()
        if item is None:
            return None
        value = item.data(0, PROBLEM_PATH_ROLE)
        return None if value is None else str(value)

    def select_key(self, key) -> bool:
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                if child.data(0, ITEM_ROLE) == str(key):
                    self.tree.setCurrentItem(child)
                    return True
        return False

    def select_problem_path(self, path) -> bool:
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                if child.data(0, PROBLEM_PATH_ROLE) == str(path):
                    self.tree.setCurrentItem(child)
                    return True
        return False

    def _select_first_item(self):
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            if group_item.data(0, VIRTUAL_ROLE):
                continue
            for child_index in range(group_item.childCount()):
                if not group_item.child(child_index).isHidden():
                    self.tree.setCurrentItem(group_item.child(child_index))
                    return

    def _on_selection_changed(self, *_args):
        if self._reloading:
            return
        self.save_pending_edits()
        self.load_editor()

    def save_pending_edits(self) -> None:
        """:meth:`autosave_pending_edits`, guarded against re-entry.

        A save mutates the library, the library notifies its listeners, and this
        dialog *is* a listener -- so the tree rebuilds and the selection is
        restored while the save is still on the stack, which arrives back here
        as a second "selection changed". Without this guard that second pass
        compares the editor (already showing the *next* definition) against the
        snapshot of the one being saved, decides it is dirty, and saves again,
        forever. Registering the listener is what made the colormap manager
        reachable from a second window; this is what keeps that from eating
        itself.
        """

        if self._autosaving:
            return
        self._autosaving = True
        try:
            self.autosave_pending_edits()
        finally:
            self._autosaving = False

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def apply_filters(self):
        text = self.filter_edit.text().strip().casefold()
        choice = self.filter_combo.currentData()
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            group = str(group_item.data(0, GROUP_ROLE) or "")
            visible = 0
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                if child.data(0, PROBLEM_PATH_ROLE) is not None:
                    # A stored definition the library could not load is exactly
                    # what a filter must never hide: it has no metadata to match
                    # on and it is the row the user most needs to reach.
                    hide = bool(text) and text not in child.text(0).casefold()
                else:
                    hide = not self.row_matches(child, text, choice, group)
                child.setHidden(hide)
                visible += 0 if hide else 1
            group_item.setHidden(visible == 0)

    def row_matches(self, item, text: str, choice, group: str) -> bool:
        """Whether one row survives the filter.  Subclasses narrow this."""

        return not text or text in item.text(0).casefold()

    # ------------------------------------------------------------------
    # Layout persistence and group rename
    # ------------------------------------------------------------------

    def tree_layout(self, override: dict | None = None):
        group_order = []
        groups = {}
        order = {}
        for group_index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(group_index)
            if group_item.data(0, VIRTUAL_ROLE):
                continue
            group = str(group_item.data(0, GROUP_ROLE))
            group_order.append(group)
            for child_index in range(group_item.childCount()):
                key = str(group_item.child(child_index).data(0, ITEM_ROLE))
                groups[key] = group
                order[key] = child_index
        if override:
            groups.update(override)
        return group_order, groups, order

    def persist_layout(self, override: dict | None = None):
        group_order, groups, order = self.tree_layout(override)
        self.library_apply_layout(group_order, groups, order)

    def _on_item_changed(self, item, _column):
        """A top-level row was edited in place: that is a group rename."""

        if self._reloading or item.parent() is not None:
            return
        old_group = str(item.data(0, GROUP_ROLE))
        new_group = item.text(0).strip()
        if not new_group or new_group == old_group or old_group in self.PROTECTED_GROUPS:
            self.refresh_tree()
            return
        item.setData(0, GROUP_ROLE, new_group)
        if self.library_rename_group(old_group, new_group):
            self.persist_layout()
        self.refresh_tree()

    # ------------------------------------------------------------------
    # Session revert -- one step, and the button only exists when it leads
    # somewhere
    # ------------------------------------------------------------------

    def snapshot_before_change(self, key) -> None:
        """Remember enough to undo the change that is about to happen."""

        if key is None:
            return
        self._revert_snapshot = (
            str(key),
            self.library_snapshot_payload(str(key)),
            self.library_is_hidden(str(key)),
        )

    def clear_revert_snapshot(self) -> None:
        self._revert_snapshot = None

    def revert_target(self):
        """``(key, payload, was_hidden)`` of the pending undo, or ``None``."""

        return self._revert_snapshot

    def _revert(self):
        if self._revert_snapshot is None:
            return
        key, payload, was_hidden = self._revert_snapshot
        self._revert_snapshot = None
        try:
            self.library_restore_snapshot(key, payload, was_hidden)
        except Exception as exc:
            self.status(f"Could not revert: {exc}", timeout=4000)
            return
        self.refresh_tree()
        # The selection follows what was just put back: the whole point of the
        # undo is to be looking at the thing again.
        self.select_key(key)
        self.status(f"Reverted the last change to “{key}”.", timeout=2500)

    def sync_revert_button(self) -> None:
        self.revert_button.setVisible(self._revert_snapshot is not None)

    def _set_remove_affordance(self, *, destructive: bool) -> None:
        """Delete really deletes; hiding a system definition is reversible.

        The icon follows the action rather than only the tooltip: a trash can
        over "this comes back with Restore" is a lie the user only discovers
        after clicking.
        """

        set_button_icon(
            self.remove_button,
            "delete" if destructive else "visibility_off",
            tooltip=(
                f"Delete this {self.ITEM_NOUN} and its file"
                if destructive
                else f"Hide this {self.ITEM_NOUN} (Restore brings it back)"
            ),
        )

    # ------------------------------------------------------------------
    # Shared actions
    # ------------------------------------------------------------------

    def _guarded(self, action):
        """Save whatever is in the editor before an action rebuilds the tree."""

        self.save_pending_edits()
        action()

    def _remove_or_hide(self):
        problem_path = self.selected_problem_path()
        if problem_path is not None:
            name = os.path.basename(problem_path)
            if not self.confirm(
                f"Remove {self.ITEM_NOUN}",
                f"Remove the stored definition “{name}” and delete its file?",
            ):
                return
            if self.library_remove_problem_file(problem_path):
                self.status(f"Removed “{name}”.", timeout=2500)
            return
        self.library_delete_or_hide()

    def _open_folder(self):
        directory = self.library_directory()
        os.makedirs(directory, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(directory))

    def _reset_all(self):
        if not self.confirm(
            f"Reset {self.ITEM_NOUN}s",
            f"Reset the arrangement to defaults and restore every removed {self.ITEM_NOUN}?",
        ):
            return
        self.library_reset_all()
        self.status(f"Reset {self.ITEM_NOUN}s to defaults.", timeout=2500)

    # ------------------------------------------------------------------
    # Small shared services
    # ------------------------------------------------------------------

    def status(self, message: str, *, timeout: int = 3000) -> None:
        show_status_message(self._window, message, timeout=timeout)

    def confirm(self, title: str, question: str) -> bool:
        """Destroying a definition asks first -- on both sides, always."""

        answer = QtWidgets.QMessageBox.question(self, title, question)
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def open_path(self, path: str) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    # ------------------------------------------------------------------
    # The subclass contract
    # ------------------------------------------------------------------

    def populate_tree(self) -> None:
        """Add every group and row (Problems excluded -- the base adds those)."""

        raise NotImplementedError

    def load_editor(self) -> None:
        """Fill the editor pane from the current selection."""

        raise NotImplementedError

    def autosave_pending_edits(self) -> None:
        """Persist unsaved editor state before the selection or dialog changes."""

    def repaired_item_id(self):
        """Id a just-repaired Problems row now claims, so selection can follow it."""

        return None

    def library_directory(self) -> str:
        raise NotImplementedError

    def library_add_listener(self, callback) -> None:
        raise NotImplementedError

    def library_remove_listener(self, callback) -> None:
        raise NotImplementedError

    def library_problems(self):
        return []

    def library_remove_problem_file(self, path: str) -> bool:
        raise NotImplementedError

    def library_apply_layout(self, group_order, groups, order) -> None:
        raise NotImplementedError

    def library_rename_group(self, old: str, new: str) -> bool:
        raise NotImplementedError

    def library_is_hidden(self, key: str) -> bool:
        return False

    def library_snapshot_payload(self, key: str):
        return None

    def library_restore_snapshot(self, key: str, payload, was_hidden: bool) -> None:
        raise NotImplementedError

    def library_new(self) -> None:
        raise NotImplementedError

    def library_duplicate(self) -> None:
        raise NotImplementedError

    def library_import(self) -> None:
        raise NotImplementedError

    def library_export(self) -> None:
        raise NotImplementedError

    def library_delete_or_hide(self) -> None:
        raise NotImplementedError

    def library_restore(self) -> None:
        raise NotImplementedError

    def library_open_file(self) -> None:
        raise NotImplementedError

    def library_reset_all(self) -> None:
        raise NotImplementedError
