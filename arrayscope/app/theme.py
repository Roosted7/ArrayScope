"""Small built-in Qt palette theme abstraction for ArrayScope."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ThemeChoice(str, Enum):
    SYSTEM = "system"
    NATIVE = "native"
    DARK = "dark"
    LIGHT = "light"


@dataclass(frozen=True)
class ThemeResult:
    requested: ThemeChoice
    applied: ThemeChoice
    backend: str
    warning: Optional[str] = None


def normalize_theme_choice(choice) -> ThemeChoice:
    if isinstance(choice, ThemeChoice):
        return choice
    if choice is None or choice == "auto":
        return ThemeChoice.SYSTEM
    try:
        return ThemeChoice(str(choice).lower())
    except ValueError:
        return ThemeChoice.SYSTEM


def choose_theme_backend(choice, available_backends=()) -> ThemeResult:
    choice = normalize_theme_choice(choice)
    if choice in (ThemeChoice.SYSTEM, ThemeChoice.NATIVE):
        return ThemeResult(choice, choice, "native")
    return ThemeResult(choice, choice, "builtin")


def apply_theme_to_qapplication(app, choice) -> ThemeResult:
    choice = normalize_theme_choice(choice)
    result = choose_theme_backend(choice)
    _apply_builtin_palette(app, result.applied)
    return result


def _apply_builtin_palette(app, choice):
    if app is None:
        return
    try:
        from pyqtgraph.Qt import QtGui, QtWidgets
    except Exception:
        return

    app.setStyleSheet("")
    app.setPalette(app.style().standardPalette())
    if choice in (ThemeChoice.SYSTEM, ThemeChoice.NATIVE):
        return

    if choice == ThemeChoice.DARK:
        palette = QtGui.QPalette(app.palette())
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(37, 37, 37))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(230, 230, 230))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(28, 28, 28))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(230, 230, 230))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(230, 230, 230))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(80, 120, 180))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(230, 230, 230))
        app.setPalette(palette)
    elif choice == ThemeChoice.LIGHT:
        palette = app.style().standardPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(245, 245, 245))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(255, 255, 255))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(238, 238, 238))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(20, 20, 20))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(20, 20, 20))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(245, 245, 245))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(20, 20, 20))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(60, 120, 200))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(255, 255, 255))
        app.setPalette(palette)
