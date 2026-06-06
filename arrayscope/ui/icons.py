"""Material icon helpers for Qt widgets."""

from __future__ import annotations

from arrayscope.app.qt_binding import prefer_pyside6

prefer_pyside6()

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

try:
    from qt_material_icons import MaterialIcon
except Exception:  # pragma: no cover - exercised only before optional assets are installed.
    MaterialIcon = None


_FALLBACK_PIXMAPS = {
    "add": QtWidgets.QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "search": QtWidgets.QStyle.StandardPixmap.SP_FileDialogContentsView,
    "edit": QtWidgets.QStyle.StandardPixmap.SP_FileDialogDetailedView,
    "delete": QtWidgets.QStyle.StandardPixmap.SP_TrashIcon,
    "delete_sweep": QtWidgets.QStyle.StandardPixmap.SP_TrashIcon,
    "save": QtWidgets.QStyle.StandardPixmap.SP_DialogSaveButton,
    "folder_open": QtWidgets.QStyle.StandardPixmap.SP_DirOpenIcon,
    "download": QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
    "undo": QtWidgets.QStyle.StandardPixmap.SP_ArrowBack,
    "refresh": QtWidgets.QStyle.StandardPixmap.SP_BrowserReload,
    "warning": QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
    "close": QtWidgets.QStyle.StandardPixmap.SP_DialogCloseButton,
    "arrow_upward": QtWidgets.QStyle.StandardPixmap.SP_ArrowUp,
    "arrow_downward": QtWidgets.QStyle.StandardPixmap.SP_ArrowDown,
    "arrow_back": QtWidgets.QStyle.StandardPixmap.SP_ArrowBack,
    "arrow_forward": QtWidgets.QStyle.StandardPixmap.SP_ArrowForward,
}


def material_icon(name: str, *, color: QtGui.QColor | str | None = None) -> QtGui.QIcon:
    """Return a Material Symbols icon, with a small Qt fallback for bootstrapping."""
    if MaterialIcon is not None:
        icon = MaterialIcon(name)
        if color is not None:
            icon.set_color(QtGui.QColor(color))
        return icon

    style = QtWidgets.QApplication.style()
    pixmap = _FALLBACK_PIXMAPS.get(name, QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
    return style.standardIcon(pixmap) if style is not None else QtGui.QIcon()


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
