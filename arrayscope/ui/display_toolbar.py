"""Compact, width-adaptive display controls for the ArrayScope main window.

The toolbar degrades gracefully as horizontal space shrinks:

- level 0: icon + text before each dropdown; dropdown entries show icon + text
- level 1: label text hidden, label icons kept (first adaptation)
- level 2: label icons hidden too
- level 3: dropdown entry text hidden, leaving icon-only entries
"""

from __future__ import annotations

import pyqtgraph.Qt as Qt
from pyqtgraph.Qt import QtCore, QtWidgets

from arrayscope.ui.icons import glyph_icon, material_icon, set_action_icon
from arrayscope.ui.widgets import TOOL_BUTTON_STYLE, configure_tool_button


_CHANNEL_ITEMS = (
    ("Complex", "complex", "ℂ"),
    ("Real", "real", "ℝ"),
    ("Abs", "abs", "|z|"),
    ("Imag", "imag", "ℑ"),
    ("Phase", "angle", "φ"),
)
_SCALE_ITEMS = (
    ("Linear", "linear", "lin"),
    ("Log", "log", "log"),
    ("Symlog", "symlog", "±log"),
)
_WINDOW_ITEMS = (
    ("Relative", "relative", "%"),
    ("Absolute", "absolute", "#"),
)
# Sentinel entry that opens the colormap designer instead of selecting.
CUSTOMIZE_COLORMAP = "__customize_colormaps__"

_COLORMAP_ICON_CACHE: dict[str, object] = {}


def _colormap_icon(name):
    from pyqtgraph.Qt import QtGui

    icon = _COLORMAP_ICON_CACHE.get(name)
    if icon is not None:
        return icon
    icon = QtGui.QIcon()
    try:
        from arrayscope.display.colormap_library import get_colormap

        lut = get_colormap(name).getLookupTable(0.0, 1.0, 32, alpha=False)
        pixmap = QtGui.QPixmap(len(lut), 12)
        painter = QtGui.QPainter(pixmap)
        for index, rgb in enumerate(lut):
            painter.fillRect(index, 0, 1, 12, QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
        painter.end()
        icon = QtGui.QIcon(pixmap.scaled(32, 12))
    except Exception:
        pass
    _COLORMAP_ICON_CACHE[name] = icon
    return icon


def _item_icon(value, glyph):
    return glyph_icon(glyph) if glyph else _colormap_icon(value)


class DisplayToolbar(QtWidgets.QToolBar):
    channelChanged = Qt.QtCore.Signal(str)
    scaleChanged = Qt.QtCore.Signal(str)
    colormapChanged = Qt.QtCore.Signal(str)
    colormapCustomizeRequested = Qt.QtCore.Signal()
    fitRequested = Qt.QtCore.Signal(bool)
    oneToOneRequested = Qt.QtCore.Signal()
    windowModeChanged = Qt.QtCore.Signal(str)
    autoWindowRequested = Qt.QtCore.Signal()
    syncWindowToggled = Qt.QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__("Display", parent)
        self.setObjectName("DisplayToolbar")
        self.setMovable(False)
        self.setIconSize(Qt.QtCore.QSize(16, 16))
        self.setStyleSheet(TOOL_BUTTON_STYLE)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self._channel_options_state: tuple[tuple[str, bool], ...] | None = None
        self._compact_level = 0
        self._group_icon_labels: list[tuple[QtWidgets.QLabel, str]] = []
        # QToolBar manages child visibility through the QWidgetActions
        # returned by addWidget; hiding the widgets directly gets undone.
        self._group_icon_actions: list = []
        self._group_text_actions: list = []
        self._combo_items: list[tuple[QtWidgets.QComboBox, tuple[tuple[str, str, str], ...]]] = []

        self.channel_combo = self._add_group("Channel", "layers", _CHANNEL_ITEMS)
        self.channel_combo.currentIndexChanged.connect(self._channel_index_changed)

        self.addSeparator()
        self.scale_combo = self._add_group("Scale", "linear_scale", _SCALE_ITEMS)
        self.scale_combo.currentIndexChanged.connect(
            lambda _i: self.scaleChanged.emit(self.scale_combo.currentData())
        )

        self.addSeparator()
        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(material_icon("palette").pixmap(14, 14))
        icon_label.setToolTip("Color")
        self._group_icon_actions.append(self.addWidget(icon_label))
        self._group_icon_labels.append((icon_label, "palette"))
        text_label = QtWidgets.QLabel("Color")
        self._group_text_actions.append(self.addWidget(text_label))
        # Grouped fold-out picker: submenus per colormap group.
        self.colormap_button = QtWidgets.QToolButton()
        self.colormap_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.colormap_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.colormap_button.setToolTip("Colormap")
        self._colormap_menu = QtWidgets.QMenu(self.colormap_button)
        self.colormap_button.setMenu(self._colormap_menu)
        self._colormap_family = "scalar"
        self._current_colormap = None
        self.addWidget(self.colormap_button)

        self.addSeparator()
        self.fit_action = self.addAction("Fit")
        self.fit_action.setCheckable(True)
        set_action_icon(self.fit_action, "fit_screen")
        self.fit_action.setToolTip("Fit image to viewport")
        self.fit_action.triggered.connect(lambda checked=False: self.fitRequested.emit(bool(checked)))
        self.one_to_one_action = self.addAction("1:1")
        set_action_icon(self.one_to_one_action, "aspect_ratio")
        self.one_to_one_action.setToolTip("Show image at one screen pixel per image pixel")
        self.one_to_one_action.triggered.connect(lambda _checked=False: self.oneToOneRequested.emit())
        for action in (self.fit_action, self.one_to_one_action):
            button = self.widgetForAction(action)
            if button is not None:
                configure_tool_button(button)

        # Expanding center section: the status readout (pixel/crosshair value,
        # slice context) lives between the left view controls and the
        # right-aligned window/level controls.
        self.center_container = QtWidgets.QWidget()
        self.center_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
        )
        self._center_layout = QtWidgets.QHBoxLayout(self.center_container)
        self._center_layout.setContentsMargins(10, 0, 10, 0)
        self._center_layout.setSpacing(10)
        self._center_layout.addStretch(1)
        self._center_layout.addStretch(1)
        self.addWidget(self.center_container)

        # Visible only while status text sits next to it; a lone divider at
        # the right edge is noise.
        self._right_separator = self.addSeparator()
        self.window_combo = self._add_group("Window", "contrast", _WINDOW_ITEMS)
        self.window_combo.currentIndexChanged.connect(
            lambda _i: self.windowModeChanged.emit(self.window_combo.currentData())
        )

        self.auto_window_action = self.addAction("Auto")
        set_action_icon(self.auto_window_action, "tonality")
        self.auto_window_action.setToolTip("Auto window levels")
        self.auto_window_action.triggered.connect(self.autoWindowRequested)
        button = self.widgetForAction(self.auto_window_action)
        if button is not None:
            configure_tool_button(button)

        self.sync_window_action = self.addAction("Sync")
        self.sync_window_action.setCheckable(True)
        set_action_icon(self.sync_window_action, "link")
        self.sync_window_action.setToolTip(
            "Sync window/level with other linked ArrayScope windows (also from separately started sessions)"
        )
        self.sync_window_action.toggled.connect(lambda checked: self.syncWindowToggled.emit(bool(checked)))
        button = self.widgetForAction(self.sync_window_action)
        if button is not None:
            configure_tool_button(button)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def add_center_widget(self, widget) -> None:
        """Add a widget to the centered status section (kept centered in the
        free space between the left and right control groups)."""
        self._center_layout.insertWidget(self._center_layout.count() - 1, widget)

    def set_colormap_options(self, family, *, current=None) -> None:
        """Rebuild the grouped picker menu for the active channel family."""
        from arrayscope.display import colormap_library as library

        self._colormap_family = str(family)
        if current is not None:
            self._current_colormap = str(current)
        menu = self._colormap_menu
        names = tuple(
            info.name
            for _group, infos in library.grouped_colormaps(self._colormap_family)
            for info in infos
        )
        state = (self._colormap_family, names)
        # Binder syncs call this often; never tear the menu down while the
        # user is browsing it, and skip rebuilds when nothing changed.
        if state == getattr(self, "_colormap_menu_state", None) or menu.isVisible():
            if state == getattr(self, "_colormap_menu_state", None):
                self._sync_colormap_button()
                return
        self._colormap_menu_state = state
        menu.clear()
        for group, infos in library.grouped_colormaps(self._colormap_family):
            submenu = menu.addMenu(group)
            for info in infos:
                action = submenu.addAction(_colormap_icon(info.name), info.name)
                action.setCheckable(True)
                action.setChecked(self._current_colormap == info.name)
                action.triggered.connect(
                    lambda _checked=False, name=info.name: self._pick_colormap(name)
                )
        menu.addSeparator()
        customize = menu.addAction(material_icon("edit"), "Customize…")
        customize.setToolTip("Edit, create and import colormaps")
        customize.triggered.connect(self.colormapCustomizeRequested)
        self._sync_colormap_button()

    def _pick_colormap(self, name: str) -> None:
        self._current_colormap = str(name)
        self._sync_colormap_button()
        self.colormapChanged.emit(str(name))

    def _sync_colormap_button(self) -> None:
        name = self._current_colormap or ""
        self.colormap_button.setIcon(_colormap_icon(name) if name else material_icon("palette"))
        self.colormap_button.setText("" if self._compact_level >= 3 else name)
        self.colormap_button.setToolTip(f"Colormap: {name}" if name else "Colormap")
        for group_action in self._colormap_menu.actions():
            submenu = group_action.menu()
            if submenu is None:
                continue
            for action in submenu.actions():
                action.setChecked(action.text() == name)

    def sync_center_separator(self) -> None:
        has_text = False
        for index in range(self._center_layout.count()):
            item_widget = self._center_layout.itemAt(index).widget()
            if item_widget is not None and bool(getattr(item_widget, "text", lambda: "")()):
                has_text = True
                break
        self._right_separator.setVisible(has_text)

    def _add_group(self, label_text, icon_name, items):
        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(material_icon(icon_name).pixmap(14, 14))
        icon_label.setToolTip(label_text)
        self._group_icon_actions.append(self.addWidget(icon_label))
        self._group_icon_labels.append((icon_label, icon_name))

        text_label = QtWidgets.QLabel(label_text)
        self._group_text_actions.append(self.addWidget(text_label))

        combo = QtWidgets.QComboBox()
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        combo.setToolTip(label_text)
        for item in items:
            text, value, glyph = item
            combo.addItem(_item_icon(value, glyph), text, value)
            combo.setItemData(combo.count() - 1, text, QtCore.Qt.ItemDataRole.ToolTipRole)
        self.addWidget(combo)
        self._combo_items.append((combo, tuple(items)))
        return combo

    # ------------------------------------------------------------------
    # Width adaptation
    # ------------------------------------------------------------------

    def compact_level(self) -> int:
        return self._compact_level

    def _apply_compact_level(self, level: int) -> None:
        level = max(0, min(3, int(level)))
        self._compact_level = level
        for action in self._group_text_actions:
            action.setVisible(level < 1)
        for action in self._group_icon_actions:
            action.setVisible(level < 2)
        for combo, items in self._combo_items:
            for index, (text, _value, _glyph) in enumerate(items):
                combo.setItemText(index, "" if level >= 3 else text)
            combo.setMaximumWidth(46 if level >= 3 else 16_777_215)
        if hasattr(self, "colormap_button"):
            self._sync_colormap_button()

    def _required_width(self, level: int) -> int:
        self._apply_compact_level(level)
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        return self.sizeHint().width()

    def adapt_to_width(self, available: int | None = None) -> int:
        """Pick the least-compact level that fits `available` pixels."""
        available = int(self.width() if available is None else available)
        previous = self._compact_level
        chosen = 3
        for level in range(4):
            if self._required_width(level) <= available:
                chosen = level
                break
        if chosen != self._compact_level or chosen != previous:
            self._apply_compact_level(chosen)
        return self._compact_level

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adapt_to_width(event.size().width())

    def refresh_icons(self) -> None:
        """Re-tint label pixmaps and combo item icons for the active palette."""
        for icon_label, name in self._group_icon_labels:
            icon_label.setPixmap(material_icon(name).pixmap(14, 14))
        _COLORMAP_ICON_CACHE.clear()
        for combo, items in self._combo_items:
            for index, (_text, value, glyph) in enumerate(items):
                combo.setItemIcon(index, _item_icon(value, glyph))
        if hasattr(self, "colormap_button"):
            self.set_colormap_options(self._colormap_family, current=self._current_colormap)

    # ------------------------------------------------------------------
    # State API (unchanged)
    # ------------------------------------------------------------------

    def set_channel_options(self, enabled_channels):
        state = tuple(
            (str(self.channel_combo.itemData(index)), bool(enabled_channels.get(self.channel_combo.itemData(index), False)))
            for index in range(self.channel_combo.count())
        )
        if self._channel_options_state == state:
            return
        self._channel_options_state = state
        for index in range(self.channel_combo.count()):
            value = self.channel_combo.itemData(index)
            item = self.channel_combo.model().item(index)
            if item is not None:
                item.setEnabled(bool(enabled_channels.get(value, False)))

    def _channel_index_changed(self, index):
        item = self.channel_combo.model().item(index)
        if item is not None and not item.isEnabled():
            return
        self.channelChanged.emit(self.channel_combo.currentData())

    def set_current(self, *, channel=None, scale=None, aspect=None, window_mode=None, colormap=None):
        if colormap is not None and str(colormap) != self._current_colormap:
            self._current_colormap = str(colormap)
            self._sync_colormap_button()
        for combo, value in (
            (self.channel_combo, channel),
            (self.scale_combo, scale),
            (self.window_combo, window_mode),
        ):
            if value is None:
                continue
            index = combo.findData(value)
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        if aspect is not None:
            blocker = Qt.QtCore.QSignalBlocker(self.fit_action)
            try:
                self.fit_action.setChecked(aspect == "fit")
            finally:
                blocker.unblock()
