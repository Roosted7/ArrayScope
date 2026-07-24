"""Anchored popup for adding an operation (stage 1 of the add flow).

A compact, grouped operation picker rendered from the
:class:`~arrayscope.ui.operation_listing.ListingSection` list the caller builds
via :func:`~arrayscope.ui.operation_listing.build_operation_listing` (which reads
the operation library). Section headers are non-selectable dividers; op rows
carry an icon, label and description tooltip. The optional backend groups
(SigPy / BART / ...) live behind a "More..." fold-out that expands in place.

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

# Item-data roles: ROLE_KIND distinguishes op rows / headers / the more toggle;
# ROLE_OP carries the operation id for op rows.
_ROLE_KIND = int(QtCore.Qt.ItemDataRole.UserRole)
_ROLE_OP = int(QtCore.Qt.ItemDataRole.UserRole) + 1

_KIND_OP = "op"
_KIND_HEADER = "header"
_KIND_MORE = "more"
_KIND_LESS = "less"


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
        self._on_accept = on_accept
        self._on_needs_parameters = on_needs_parameters
        self._expanded = False

        container = QtWidgets.QWidget(self)
        column = QtWidgets.QVBoxLayout(container)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        self._list = QtWidgets.QListWidget(self)
        self._list.setObjectName("OperationAddList")
        self._list.setUniformItemSizes(False)
        self._list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self._list.setMinimumWidth(240)
        self._list.setMinimumHeight(260)
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
        sections = self._sections
        for section in sections:
            if section.is_more and not self._expanded:
                continue
            self._add_header(section.title)
            for entry in section.entries:
                self._add_op_row(entry)
        has_more = any(section.is_more for section in sections)
        if has_more:
            self._add_toggle_row(collapse=self._expanded)
        self._select_first_op()
        self._sync_axis_visibility()

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
        item = QtWidgets.QListWidgetItem(material_icon(entry.icon), entry.label.rstrip("."))
        item.setData(_ROLE_KIND, _KIND_OP)
        item.setData(_ROLE_OP, entry.id)
        if entry.description:
            item.setToolTip(entry.description)
        if not self._is_enabled(entry):
            item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        item.setSizeHint(QtCore.QSize(0, 26))
        self._list.addItem(item)

    def _add_toggle_row(self, *, collapse: bool) -> None:
        text = "Less" if collapse else "More…"
        icon = "expand_less" if collapse else "expand_more"
        item = QtWidgets.QListWidgetItem(material_icon(icon), text)
        item.setData(_ROLE_KIND, _KIND_LESS if collapse else _KIND_MORE)
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
        if kind in (_KIND_MORE, _KIND_LESS):
            self._toggle_expanded()

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        kind = item.data(_ROLE_KIND)
        if kind in (_KIND_MORE, _KIND_LESS):
            self._toggle_expanded()
        elif kind == _KIND_OP:
            self._activate_operation(item.data(_ROLE_OP))

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._rebuild()

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
            self._axis_row.setVisible(False)
            return
        entry = self._current_entry()
        self._axis_row.setVisible(bool(entry and entry.requires_axis))

    def _activate_operation(self, op_id: str) -> None:
        entry = self._by_id.get(op_id)
        if entry is None:
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
        """Move the selection to ``op_id``; returns whether it was found."""

        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.data(_ROLE_KIND) == _KIND_OP and item.data(_ROLE_OP) == op_id:
                if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsSelectable):
                    return False
                self._list.setCurrentRow(row)
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

    def set_expanded(self, expanded: bool) -> None:
        if bool(expanded) != self._expanded:
            self._toggle_expanded()
