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
}


def material_icon(name: str, *, color: QtGui.QColor | str | None = None) -> QtGui.QIcon:
    """Return a Material Symbols icon, with a small Qt fallback for bootstrapping."""
    color_key = None if color is None else QtGui.QColor(color).name()
    key = (str(name), color_key)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]

    icon = QtGui.QIcon()
    if MaterialIcon is not None:
        try:
            icon = MaterialIcon(name)
            if color is not None:
                icon.set_color(QtGui.QColor(color))
        except Exception:
            icon = QtGui.QIcon()
    if icon.isNull():
        icon = _fallback_icon(name)
        if name not in _MISSING_LOGGED:
            _MISSING_LOGGED.add(name)
            _LOGGER.debug("Using fallback icon for missing/null material icon: %s", name)
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


def set_label_icon(label: QtWidgets.QLabel, name: str, *, icon_size: int = 18) -> None:
    label.setText("")
    label.setPixmap(material_icon(name).pixmap(icon_size, icon_size))


def clear_label_icon(label: QtWidgets.QLabel) -> None:
    label.clear()
