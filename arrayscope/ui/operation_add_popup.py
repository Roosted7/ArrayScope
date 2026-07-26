"""Anchored popup for adding an operation (stage 1 of the add flow).

A compact, grouped operation picker rendered from the
:class:`~arrayscope.ui.operation_listing.ListingSection` list the caller builds
via :func:`~arrayscope.ui.operation_listing.build_operation_listing` (which reads
the operation library). Section headers are non-selectable dividers; op rows
carry an icon, label and description tooltip. The pinned Common section opens
immediately; the remaining groups are an accordion behind "Browse categories…"
so revealing a category never turns the popup into one long catalogue scroll.
Search stays owned by the command palette and is linked from the first row.

When the selected op takes an axis, an axis dropdown appears below the list so
op *and* dimension are chosen in the same popup. For the dimension-chip flow the
axis is fixed by the chip, so the dropdown is hidden entirely.

Activation splits by whether the op has parameters: a parameterless op is
confirmed immediately via ``on_accept``; a parameterized op hands off to
``on_needs_parameters`` so the caller can open the stage-2 params popup anchored
at the same point.
"""

from __future__ import annotations

from collections.abc import Callable

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from arrayscope.operations.registry import OperationEntry
from arrayscope.ui.bubbles import EditBubble
from arrayscope.ui.icons import material_icon

# Item-data roles: ROLE_KIND distinguishes operations, headers, and browse
# controls; ROLE_OP carries an operation id or category title.
_ROLE_KIND = int(QtCore.Qt.ItemDataRole.UserRole)
_ROLE_OP = int(QtCore.Qt.ItemDataRole.UserRole) + 1

_KIND_OP = "op"
_KIND_HEADER = "header"
_KIND_SEARCH = "search"
_KIND_BROWSE = "browse"
_KIND_CATEGORY = "category"
_KIND_COMMON_ONLY = "common_only"


class OperationAddPopup(EditBubble):
    """Grouped operation picker with an inline axis selector."""

    def __init__(
        self,
        sections,
        *,
        axis_choices=(),
        default_axis: int | None = None,
        fixed_axis: int | None = None,
        is_enabled: Callable[[OperationEntry], bool] | None = None,
        on_search: Callable[[], None] | None = None,
        on_accept: Callable[[str, int | None], None],
        on_needs_parameters: Callable[[str, int | None], None],
        parent=None,
    ) -> None:
        super().__init__(parent, icon_name=None)
        # See OperationParamsPopup: a Qt.Popup auto-closes on focus loss, so keep
        # the object alive on close rather than letting WA_DeleteOnClose delete
        # it while a caller still references it.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)
        # Pre-built listing sections (from build_operation_listing, which reads
        # the operation library). The picker only presents them; it never owns
        # the catalogue.
        self._sections = list(sections)
        self._by_id = {entry.id: entry for section in self._sections for entry in section.entries}
        self._fixed_axis = fixed_axis
        self._is_enabled = is_enabled or (lambda _entry: True)
        self._on_search = on_search
        self._on_accept = on_accept
        self._on_needs_parameters = on_needs_parameters
        self._expanded = False
        self._expanded_category: str | None = None

        container = QtWidgets.QWidget(self)
        column = QtWidgets.QVBoxLayout(container)
        self._container = container
        self._column = column
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self._list = QtWidgets.QListWidget(self)
        self._list.setObjectName("OperationAddList")
        # Rows are NOT uniform (headers/toggle are shorter than op rows), so the
        # height is summed row-by-row in _apply_list_height rather than assuming
        # a single row height.
        self._list.setUniformItemSizes(False)
        self._list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self._list.setMinimumWidth(240)
        self._list.setIconSize(QtCore.QSize(16, 16))
        self._list.currentRowChanged.connect(lambda _row: self._sync_axis_visibility())
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemActivated.connect(self._on_item_activated)
        column.addWidget(self._list, 1)

        self._axis_row = QtWidgets.QWidget(self)
        axis_layout = QtWidgets.QHBoxLayout(self._axis_row)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.addWidget(QtWidgets.QLabel("Axis"))
        self._axis_combo = QtWidgets.QComboBox(self._axis_row)
        for label, axis in axis_choices:
            self._axis_combo.addItem(label, int(axis))
        if default_axis is not None:
            index = self._axis_combo.findData(int(default_axis))
            if index >= 0:
                self._axis_combo.setCurrentIndex(index)
        axis_layout.addWidget(self._axis_combo, 1)
        column.addWidget(self._axis_row)

        self.add_widget(container, 1)

        self._rebuild()

    # -- list construction ----------------------------------------------------

    def _rebuild(self) -> None:
        self._list.clear()
        if self._on_search is not None:
            self._add_search_row()

        main_sections = [section for section in self._sections if not section.is_more]
        more_sections = [section for section in self._sections if section.is_more]
        for section in main_sections:
            self._add_header(section.title)
            for entry in section.entries:
                self._add_op_row(entry)

        if more_sections:
            if not self._expanded:
                self._add_browse_row()
            else:
                self._add_header("Browse by category")
                for section in more_sections:
                    is_open = section.title == self._expanded_category
                    self._add_category_row(section.title, len(section.entries), expanded=is_open)
                    if is_open:
                        for entry in section.entries:
                            self._add_op_row(entry)
                self._add_common_only_row()
        self._select_first_op()
        self._apply_list_height()
        self._sync_axis_visibility()
        self._resize_to_content()

    def _resize_to_content(self) -> None:
        """Make the popup follow a rebuilt list in both growth directions."""

        # Qt eagerly grows a visible top-level popup but keeps its old geometry
        # when a fixed-height child shrinks. Activate from the inner layout out
        # before reading the hint, otherwise the hint still describes the
        # category that just closed and leaves a large blank panel.
        layouts = (self._column, self.content_layout, self.layout())
        for layout in layouts:
            layout.invalidate()
        self._container.updateGeometry()
        for layout in layouts:
            layout.activate()
        self.resize(self.sizeHint())

    #: Tallest the row list is allowed to grow before it starts scrolling.
    #: Sized to clear the compact Common listing plus its search and browse
    #: affordances, while a category with many operations scrolls within it.
    _LIST_MAX_HEIGHT = 460

    def _apply_list_height(self) -> None:
        """Size the list to a whole number of rows, capped at a sane maximum.

        Sums the per-row size hints (they differ: headers/toggle are shorter
        than op rows) and stops before a row would spill past the cap, so the
        visible area always ends on a whole-row boundary -- never a half-cut
        row -- and the collapsed listing does not scroll when it fits.
        """

        frame = 2 * self._list.frameWidth()
        accumulated = 0
        for row in range(self._list.count()):
            item = self._list.item(row)
            row_height = item.sizeHint().height()
            if row_height <= 0:
                row_height = self._list.sizeHintForRow(row)
            # Stop before a row spills past the cap so the visible area always
            # ends on a whole-row boundary (never a half-cut row).
            if row > 0 and accumulated + row_height + frame > self._LIST_MAX_HEIGHT:
                break
            accumulated += row_height
        self._list.setFixedHeight(accumulated + frame)

    def _add_header(self, title: str) -> None:
        item = QtWidgets.QListWidgetItem(title.upper())
        item.setData(_ROLE_KIND, _KIND_HEADER)
        item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        font = item.font()
        font.setPointSizeF(max(6.0, font.pointSizeF() - 1.5))
        font.setBold(True)
        item.setFont(font)
        palette = QtWidgets.QApplication.palette()
        item.setForeground(palette.color(QtGui.QPalette.ColorRole.Mid))
        item.setSizeHint(QtCore.QSize(0, 18))
        self._list.addItem(item)

    def _add_op_row(self, entry: OperationEntry) -> None:
        reason = str(entry.unavailable_reason or "")
        label = entry.label.rstrip(".")
        if reason:
            label = f"{label}  (unavailable)"
        item = QtWidgets.QListWidgetItem(material_icon(entry.icon), label)
        item.setData(_ROLE_KIND, _KIND_OP)
        item.setData(_ROLE_OP, entry.id)
        if reason:
            item.setToolTip(reason)
            item.setForeground(
                QtWidgets.QApplication.palette().brush(
                    QtGui.QPalette.ColorGroup.Disabled,
                    QtGui.QPalette.ColorRole.Text,
                )
            )
        elif entry.description:
            item.setToolTip(entry.description)
        if reason or not self._is_enabled(entry):
            item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QtCore.QSize(0, 26))
        self._list.addItem(item)

    def _add_search_row(self) -> None:
        item = QtWidgets.QListWidgetItem(material_icon("search"), "Search all operations…  Ctrl+K")
        item.setData(_ROLE_KIND, _KIND_SEARCH)
        item.setSizeHint(QtCore.QSize(0, 26))
        self._list.addItem(item)

    def _add_browse_row(self) -> None:
        item = QtWidgets.QListWidgetItem(material_icon("expand_more"), "Browse categories…")
        item.setData(_ROLE_KIND, _KIND_BROWSE)
        item.setSizeHint(QtCore.QSize(0, 26))
        self._list.addItem(item)

    def _add_category_row(self, title: str, count: int, *, expanded: bool) -> None:
        icon = "expand_less" if expanded else "chevron_right"
        item = QtWidgets.QListWidgetItem(material_icon(icon), f"{title}  ({count})")
        item.setData(_ROLE_KIND, _KIND_CATEGORY)
        item.setData(_ROLE_OP, title)
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setSizeHint(QtCore.QSize(0, 26))
        self._list.addItem(item)

    def _add_common_only_row(self) -> None:
        item = QtWidgets.QListWidgetItem(material_icon("expand_less"), "Show Common only")
        item.setData(_ROLE_KIND, _KIND_COMMON_ONLY)
        item.setSizeHint(QtCore.QSize(0, 24))
        self._list.addItem(item)

    def _select_first_op(self) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_ROLE_KIND) == _KIND_OP and (
                item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable
            ):
                self._list.setCurrentRow(row)
                return

    # -- interaction ----------------------------------------------------------

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        kind = item.data(_ROLE_KIND)
        if kind == _KIND_SEARCH:
            self._open_search()
        elif kind == _KIND_BROWSE:
            self._toggle_expanded()
        elif kind == _KIND_CATEGORY:
            self._toggle_category(str(item.data(_ROLE_OP)))
        elif kind == _KIND_COMMON_ONLY:
            self.set_expanded(False)

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        kind = item.data(_ROLE_KIND)
        if kind == _KIND_SEARCH:
            self._open_search()
        elif kind == _KIND_BROWSE:
            self._toggle_expanded()
        elif kind == _KIND_CATEGORY:
            self._toggle_category(str(item.data(_ROLE_OP)))
        elif kind == _KIND_COMMON_ONLY:
            self.set_expanded(False)
        elif kind == _KIND_OP:
            self._activate_operation(item.data(_ROLE_OP))

    def _open_search(self) -> None:
        if self._on_search is None:
            return
        self.close()
        self._on_search()

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        if not self._expanded:
            self._expanded_category = None
        self._rebuild()

    def _toggle_category(self, title: str) -> None:
        self._expanded_category = None if self._expanded_category == title else title
        self._rebuild()
        if self._expanded_category is not None:
            self._select_category_row(self._expanded_category)

    def _current_entry(self) -> OperationEntry | None:
        item = self._list.currentItem()
        if item is None or item.data(_ROLE_KIND) != _KIND_OP:
            return None
        return self._by_id.get(item.data(_ROLE_OP))

    def _selected_axis(self, entry: OperationEntry) -> int | None:
        if not entry.requires_axis:
            return None
        if self._fixed_axis is not None:
            return int(self._fixed_axis)
        data = self._axis_combo.currentData()
        return None if data is None else int(data)

    def _sync_axis_visibility(self) -> None:
        if self._fixed_axis is not None:
            visible = False
        else:
            entry = self._current_entry()
            visible = bool(entry and entry.requires_axis)
        changed = self._axis_row.isHidden() == visible
        self._axis_row.setVisible(visible)
        if changed:
            self._resize_to_content()

    def _activate_operation(self, op_id: str) -> None:
        entry = self._by_id.get(op_id)
        if entry is None or entry.unavailable_reason or not self._is_enabled(entry):
            return
        axis = self._selected_axis(entry)
        self.close()
        if entry.parameters:
            self._on_needs_parameters(entry.id, axis)
        else:
            self._on_accept(entry.id, axis)

    def activate_current(self) -> None:
        """Activate the currently selected op row (used by tests / a confirm)."""

        entry = self._current_entry()
        if entry is not None:
            self._activate_operation(entry.id)

    # -- test / introspection helpers ----------------------------------------

    def select_operation(self, op_id: str) -> bool:
        """Move the selection to ``op_id``, revealing it; returns whether it was found.

        Most operations live behind a category row, so selecting by id opens the
        owning category rather than reporting "not found" for an operation that
        plainly exists.
        """

        if self._select_listed_operation(op_id):
            return True
        for section in self._sections:
            if section.is_more and any(entry.id == op_id for entry in section.entries):
                self._expanded = True
                self._expanded_category = section.title
                self._rebuild()
                return self._select_listed_operation(op_id)
        return False

    def _select_listed_operation(self, op_id: str) -> bool:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_ROLE_KIND) == _KIND_OP and item.data(_ROLE_OP) == op_id:
                if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable):
                    return False
                self._list.setCurrentRow(row)
                self._list.scrollToItem(item)
                return True
        return False

    def _select_category_row(self, title: str) -> bool:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_ROLE_KIND) == _KIND_CATEGORY and item.data(_ROLE_OP) == title:
                self._list.setCurrentRow(row)
                self._list.scrollToItem(item)
                return True
        return False

    def visible_section_titles(self) -> list[str]:
        return [
            self._list.item(row).text()
            for row in range(self._list.count())
            if self._list.item(row).data(_ROLE_KIND) == _KIND_HEADER
        ]

    def visible_operation_ids(self) -> list[str]:
        return [
            self._list.item(row).data(_ROLE_OP)
            for row in range(self._list.count())
            if self._list.item(row).data(_ROLE_KIND) == _KIND_OP
        ]

    def visible_category_titles(self) -> list[str]:
        return [
            str(self._list.item(row).data(_ROLE_OP))
            for row in range(self._list.count())
            if self._list.item(row).data(_ROLE_KIND) == _KIND_CATEGORY
        ]

    def set_expanded_category(self, title: str | None) -> None:
        """Reveal the category chooser and optionally one category's entries."""

        self._expanded = True
        self._expanded_category = title
        self._rebuild()
        if title is not None:
            self._select_category_row(title)

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if expanded != self._expanded or (not expanded and self._expanded_category is not None):
            self._expanded = expanded
            if not expanded:
                self._expanded_category = None
            self._rebuild()
