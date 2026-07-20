"""Material icon helpers for Qt widgets."""

from __future__ import annotations

import logging

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    from qt_material_icons import MaterialIcon
except Exception:  # pragma: no cover - exercised only before optional assets are installed.
    MaterialIcon = None


_LOGGER = logging.getLogger(__name__)
_ICON_CACHE = {}
_MISSING_LOGGED = set()

_FALLBACK_PIXMAPS = {
    "add": QtWidgets.QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "search": QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView,
    "edit": QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView,
    "delete": QtWidgets.QStyle.StandardPixmap.SP_TrashIcon,
    "delete_sweep": QtWidgets.QStyle.StandardPixmap.SP_TrashIcon,
    "save": QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton,
    "folder_open": QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon,
    "download": QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
    "done": QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton,
    "undo": QtWidgets.QStyle.StandardPixmap.SP_ArrowBack,
    "refresh": QtWidgets.QStyle.StandardPixmap.SP_BrowserReload,
    "warning": QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
    "close": QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton,
    "arrow_upward": QtWidgets.QStyle.StandardPixmap.SP_ArrowUp,
    "arrow_downward": QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
    "arrow_back": QtWidgets.QStyle.StandardPixmap.SP_ArrowBack,
    "arrow_forward": QtWidgets.QStyle.StandardPixmap.SP_ArrowForward,
    "view_quilt": QtWidgets.QStyle.StandardPixmap.SP_FileDialogListView,
    "reset_wrench": QtWidgets.QStyle.StandardPixmap.SP_BrowserReload,
    "open_in_new": QtWidgets.QStyle.StandardPixmap.SP_TitleBarMaxButton,
    "show_chart": QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon,
    "monitor_heart": QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton,
    "data_array": QtWidgets.QStyle.StandardPixmap.SP_FileIcon,
    "data_object": QtWidgets.QStyle.StandardPixmap.SP_FileIcon,
    "functions": QtWidgets.QStyle.StandardPixmap.SP_CommandLink,
    "analytics": QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon,
    "waves": QtWidgets.QStyle.StandardPixmap.SP_MediaPlay,
    "crop": QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton,
    "join_inner": QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton,
    "call_split": QtWidgets.QStyle.StandardPixmap.SP_DialogResetButton,
    "drag_indicator": QtWidgets.QStyle.StandardPixmap.SP_ArrowUp,
    "inventory_2": QtWidgets.QStyle.StandardPixmap.SP_DriveHDIcon,
    "link": QtWidgets.QStyle.StandardPixmap.SP_CommandLink,
}


def material_icon(name: str, *, color: QtGui.QColor | str | None = None) -> QtGui.QIcon:
    """Return a Material Symbols icon, with a small Qt fallback for bootstrapping."""
    resolved = QtGui.QColor(color) if color is not None else _default_icon_color()
    key = (str(name), resolved.name())
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]

    icon = QtGui.QIcon()
    if MaterialIcon is not None:
        try:
            icon = MaterialIcon(name)
            # The raw Material SVGs are black; always tint so glyphs stay
            # visible on dark, light and native palettes alike.
            icon.set_color(resolved)
            palette = _application_palette()
            if palette is not None:
                disabled = palette.color(
                    QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText
                )
                icon.set_color(disabled, QtGui.QIcon.Mode.Disabled)
                # Checked buttons render on the highlight color.
                on_color = palette.color(QtGui.QPalette.ColorRole.HighlightedText)
                icon.set_color(on_color, QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.On)
                icon.set_color(on_color, QtGui.QIcon.Mode.Active, QtGui.QIcon.State.On)
        except Exception:
            icon = QtGui.QIcon()
    if icon.isNull():
        icon = _fallback_icon(name)
        if name not in _MISSING_LOGGED:
            _MISSING_LOGGED.add(name)
            _LOGGER.debug("Using fallback icon for missing/null material icon: %s", name)
    _ICON_CACHE[key] = icon
    return icon


def _application_palette():
    app = QtWidgets.QApplication.instance()
    return None if app is None else app.palette()


def _default_icon_color() -> QtGui.QColor:
    palette = _application_palette()
    if palette is not None:
        return palette.color(QtGui.QPalette.ColorRole.ButtonText)
    return QtGui.QColor("#e8eaed")


def refresh_icon_tints() -> int:
    """Re-tint every icon set through this module for the active palette.

    Icons are baked pixmaps, so a runtime theme switch must re-resolve them.
    Returns the number of refreshed targets.
    """
    _ICON_CACHE.clear()
    app = QtWidgets.QApplication.instance()
    if app is None:
        return 0
    refreshed = 0
    seen_actions = set()
    for widget in app.allWidgets():
        name = widget.property(_ICON_NAME_PROPERTY)
        if name:
            if isinstance(widget, QtWidgets.QAbstractButton):
                widget.setIcon(material_icon(str(name)))
                refreshed += 1
            elif isinstance(widget, QtWidgets.QLabel):
                size = widget.property(_ICON_SIZE_PROPERTY) or 18
                widget.setPixmap(material_icon(str(name)).pixmap(int(size), int(size)))
                refreshed += 1
        for action in widget.actions():
            refreshed += _refresh_action_icon(action, seen_actions)
    return refreshed


def _refresh_action_icon(action, seen) -> int:
    if action in seen:
        return 0
    seen.add(action)
    refreshed = 0
    name = action.property(_ICON_NAME_PROPERTY)
    if name:
        action.setIcon(material_icon(str(name)))
        refreshed += 1
    menu = action.menu()
    if menu is not None:
        for child in menu.actions():
            refreshed += _refresh_action_icon(child, seen)
    return refreshed


_ICON_NAME_PROPERTY = "arrayscope_icon_name"
_ICON_SIZE_PROPERTY = "arrayscope_icon_size"


def glyph_icon(
    text: str, *, color: QtGui.QColor | str | None = None, size: int = 18
) -> QtGui.QIcon:
    """Render a short text glyph (e.g. 'ℝ', 'φ', 'log') as a tinted icon."""
    resolved = QtGui.QColor(color) if color is not None else _default_icon_color()
    key = (f"glyph:{text}", resolved.name())
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    pixmap = QtGui.QPixmap(size * 2, size * 2)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(resolved)
    font = QtGui.QFont()
    font.setBold(True)
    point_size = size * 0.62 if len(text) <= 1 else size * 0.5 if len(text) <= 3 else size * 0.38
    font.setPointSizeF(max(5.0, point_size))
    painter.setFont(font)
    painter.drawText(QtCore.QRectF(0, 0, size, size), QtCore.Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    icon = QtGui.QIcon(pixmap)
    _ICON_CACHE[key] = icon
    return icon


def _fallback_icon(name: str) -> QtGui.QIcon:
    style = QtWidgets.QApplication.style()
    pixmap = _FALLBACK_PIXMAPS.get(name, QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
    return style.standardIcon(pixmap) if style is not None else QtGui.QIcon()


def verify_icon_names(names) -> dict[str, bool]:
    """Return a mapping of icon names to whether they produce a non-null icon."""
    result = {}
    for name in names:
        result[str(name)] = not material_icon(str(name)).isNull()
    return result


def set_button_icon(
    button: QtWidgets.QAbstractButton,
    name: str,
    *,
    icon_size: int = 18,
    tooltip: str | None = None,
    text_beside_icon: bool = True,
) -> None:
    button.setIcon(material_icon(name))
    button.setProperty(_ICON_NAME_PROPERTY, str(name))
    button.setIconSize(QtCore.QSize(icon_size, icon_size))
    if tooltip:
        button.setToolTip(tooltip)
    if isinstance(button, QtWidgets.QToolButton):
        style = (
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            if text_beside_icon and button.text()
            else QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        button.setToolButtonStyle(style)


def set_action_icon(action: QtGui.QAction, name: str) -> None:
    action.setIcon(material_icon(name))
    action.setProperty(_ICON_NAME_PROPERTY, str(name))


def set_label_icon(label: QtWidgets.QLabel, name: str, *, icon_size: int = 18) -> None:
    label.setText("")
    label.setPixmap(material_icon(name).pixmap(icon_size, icon_size))
    label.setProperty(_ICON_NAME_PROPERTY, str(name))
    label.setProperty(_ICON_SIZE_PROPERTY, int(icon_size))


def clear_label_icon(label: QtWidgets.QLabel) -> None:
    label.clear()
