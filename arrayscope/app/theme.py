"""ArrayScope theme engine.

Semantic color tokens + Qt palette + application stylesheet + pyqtgraph
propagation, for a cohesive dark/light appearance across chrome and plots.

Public API kept stable: ``ThemeChoice``, ``ThemeResult``,
``normalize_theme_choice``, ``choose_theme_backend``,
``apply_theme_to_qapplication``. New: ``ThemeTokens``, ``resolve_theme_tokens``
and ``current_theme_tokens`` for widgets that need concrete colors (plot
backgrounds, overlays, level handles).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace
from enum import Enum


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
    warning: str | None = None


@dataclass(frozen=True)
class ThemeTokens:
    """Semantic colors. All values are '#rrggbb' hex strings."""

    name: str
    is_dark: bool
    # Chrome surfaces
    window: str  # main window / toolbar background
    surface: str  # docks, cards, chips
    surface_alt: str  # hover / alternate rows
    base: str  # inputs, lists, tables
    # Text tiers
    text: str
    text_muted: str
    # Lines
    border: str
    border_strong: str
    # Accent
    accent: str
    accent_hover: str
    accent_text: str
    # Data surfaces (pyqtgraph)
    canvas: str  # image letterbox + plot background
    plot_text: str  # axis text / foreground
    plot_grid: str
    histogram_fill: str  # histogram body fill
    level_handle: str  # window/level region lines
    # Feedback
    success: str
    warning_color: str
    error: str
    # Floating chips (HUD, toasts, evaluation overlay)
    overlay_bg: str  # rgba() string
    overlay_text: str


DARK_TOKENS = ThemeTokens(
    name="dark",
    is_dark=True,
    window="#202327",
    surface="#282c31",
    surface_alt="#31363c",
    base="#1a1d20",
    text="#e8eaed",
    text_muted="#9aa0a6",
    border="#3c4147",
    border_strong="#4c5258",
    accent="#5b9bd5",
    accent_hover="#6faee6",
    accent_text="#0e1013",
    canvas="#131518",
    plot_text="#b6bcc2",
    plot_grid="#2e3338",
    histogram_fill="#4a7fb5",
    level_handle="#e8a33d",
    success="#57ab5a",
    warning_color="#d4a72c",
    error="#e5534b",
    overlay_bg="rgba(24, 27, 31, 205)",
    overlay_text="#e8eaed",
)

LIGHT_TOKENS = ThemeTokens(
    name="light",
    is_dark=False,
    window="#f2f3f5",
    surface="#fafbfc",
    surface_alt="#e9ebee",
    base="#ffffff",
    text="#1f2328",
    text_muted="#59626b",
    border="#d0d5da",
    border_strong="#b4bac1",
    accent="#2f6fbd",
    accent_hover="#3d7fce",
    accent_text="#ffffff",
    canvas="#eceef0",
    plot_text="#40464d",
    plot_grid="#d7dbdf",
    histogram_fill="#5b8fc7",
    level_handle="#c07d1a",
    success="#1a7f37",
    warning_color="#9a6700",
    error="#cf222e",
    overlay_bg="rgba(252, 253, 254, 215)",
    overlay_text="#1f2328",
)


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


def _system_prefers_dark(app) -> bool:
    """Best-effort system dark-mode detection."""
    try:
        from pyqtgraph.Qt import QtCore

        hints = app.styleHints() if app is not None else None
        scheme = getattr(hints, "colorScheme", None)
        if callable(scheme):
            value = scheme()
            if value == QtCore.Qt.ColorScheme.Dark:
                return True
            if value == QtCore.Qt.ColorScheme.Light:
                return False
    except Exception:
        pass
    try:
        palette = app.style().standardPalette()
        color = palette.window().color()
        return color.lightnessF() < 0.5
    except Exception:
        return False


def resolve_theme_tokens(choice, app=None) -> ThemeTokens:
    choice = normalize_theme_choice(choice)
    if choice == ThemeChoice.DARK:
        return DARK_TOKENS
    if choice == ThemeChoice.LIGHT:
        return LIGHT_TOKENS
    if choice == ThemeChoice.NATIVE:
        return _tokens_from_native_palette(app)
    return DARK_TOKENS if _system_prefers_dark(app) else LIGHT_TOKENS


def _tokens_from_native_palette(app) -> ThemeTokens:
    """Derive tokens from the OS palette: system colors, ArrayScope styling.

    A bare native style left custom widgets (chips, HUD, checked states)
    unstyled and often unreadable; deriving tokens from the real palette
    keeps the system's colors while restoring consistent chrome.
    """
    try:
        from pyqtgraph.Qt import QtGui

        palette = app.style().standardPalette() if app is not None else QtGui.QPalette()
        role = QtGui.QPalette.ColorRole
        is_dark = palette.color(role.Window).lightnessF() < 0.5
        base_tokens = DARK_TOKENS if is_dark else LIGHT_TOKENS

        def name(color_role):
            return palette.color(color_role).name()

        window = palette.color(role.Window)
        surface_alt = window.lighter(115) if is_dark else window.darker(105)
        border = window.lighter(140) if is_dark else window.darker(120)
        border_strong = window.lighter(170) if is_dark else window.darker(135)
        overlay = QtGui.QColor(window)
        return dataclasses_replace(
            base_tokens,
            name="native",
            is_dark=is_dark,
            window=name(role.Window),
            surface=name(role.Button),
            surface_alt=surface_alt.name(),
            base=name(role.Base),
            text=name(role.WindowText),
            text_muted=palette.color(QtGui.QPalette.ColorGroup.Disabled, role.WindowText).name(),
            border=border.name(),
            border_strong=border_strong.name(),
            accent=name(role.Highlight),
            accent_hover=palette.color(role.Highlight).lighter(112).name(),
            accent_text=name(role.HighlightedText),
            canvas=name(role.Base),
            plot_text=name(role.WindowText),
            overlay_bg=f"rgba({overlay.red()}, {overlay.green()}, {overlay.blue()}, 215)",
            overlay_text=name(role.WindowText),
        )
    except Exception:
        return _system_dark_fallback(app)


def _system_dark_fallback(app) -> ThemeTokens:
    return DARK_TOKENS if _system_prefers_dark(app) else LIGHT_TOKENS


def current_theme_tokens(app=None) -> ThemeTokens:
    """Tokens of the most recently applied theme (dark set as fallback)."""
    if app is None:
        try:
            from pyqtgraph.Qt import QtWidgets

            app = QtWidgets.QApplication.instance()
        except Exception:
            app = None
    tokens = None if app is None else app.property("_arrayscope_theme_tokens")
    return tokens if isinstance(tokens, ThemeTokens) else DARK_TOKENS


def apply_theme_to_qapplication(app, choice) -> ThemeResult:
    choice = normalize_theme_choice(choice)
    result = choose_theme_backend(choice)
    if app is None:
        return result
    tokens = resolve_theme_tokens(choice, app)
    if choice == ThemeChoice.NATIVE:
        # Native keeps the OS palette but still applies the ArrayScope
        # stylesheet built from palette-derived tokens, so custom widgets
        # (chips, HUD, checked states) stay readable.
        app.setPalette(app.style().standardPalette())
    else:
        _apply_palette(app, tokens)
    app.setStyleSheet(build_stylesheet(tokens))
    app.setProperty("_arrayscope_theme_tokens", tokens)
    _apply_pyqtgraph_defaults(tokens)
    return result


def _apply_pyqtgraph_defaults(tokens: ThemeTokens) -> None:
    """New pyqtgraph widgets pick these up at construction time."""
    try:
        import pyqtgraph as pg

        pg.setConfigOptions(background=tokens.canvas, foreground=tokens.plot_text)
    except Exception:
        pass


def _apply_palette(app, tokens: ThemeTokens) -> None:
    try:
        from pyqtgraph.Qt import QtGui
    except Exception:
        return

    c = QtGui.QColor
    palette = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    palette.setColor(role.Window, c(tokens.window))
    palette.setColor(role.WindowText, c(tokens.text))
    palette.setColor(role.Base, c(tokens.base))
    palette.setColor(role.AlternateBase, c(tokens.surface_alt))
    palette.setColor(role.Text, c(tokens.text))
    palette.setColor(role.PlaceholderText, c(tokens.text_muted))
    palette.setColor(role.Button, c(tokens.surface))
    palette.setColor(role.ButtonText, c(tokens.text))
    palette.setColor(role.Highlight, c(tokens.accent))
    palette.setColor(role.HighlightedText, c(tokens.accent_text))
    palette.setColor(role.ToolTipBase, c(tokens.surface))
    palette.setColor(role.ToolTipText, c(tokens.text))
    palette.setColor(role.Link, c(tokens.accent))
    palette.setColor(role.Mid, c(tokens.text_muted))
    palette.setColor(role.Midlight, c(tokens.surface_alt))
    palette.setColor(role.Light, c(tokens.border))
    palette.setColor(role.Dark, c(tokens.border_strong))
    group = QtGui.QPalette.ColorGroup.Disabled
    palette.setColor(group, role.Text, c(tokens.text_muted))
    palette.setColor(group, role.ButtonText, c(tokens.text_muted))
    palette.setColor(group, role.WindowText, c(tokens.text_muted))
    app.setPalette(palette)


def _arrow_asset_paths(tokens: ThemeTokens):
    """Write tiny theme-tinted arrow PNGs for spinbox steppers.

    Qt stylesheets cannot draw primitive arrows once a subcontrol is styled,
    and unstyled native steppers render inconsistently across themes. File
    URLs to generated pixmaps give crisp, consistent arrows everywhere.
    """
    import os
    import tempfile

    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    if QtWidgets.QApplication.instance() is None:
        return None
    cache_dir = os.path.join(tempfile.gettempdir(), "arrayscope-theme-assets")
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for state, color_name in (("", tokens.text), ("_muted", tokens.text_muted)):
        color = QtGui.QColor(color_name)
        for direction in ("up", "down"):
            file_name = f"arrow_{direction}{state}_{color.name().lstrip('#')}.png"
            path = os.path.join(cache_dir, file_name)
            if not os.path.exists(path):
                pixmap = QtGui.QPixmap(16, 16)
                pixmap.setDevicePixelRatio(2.0)
                pixmap.fill(QtCore.Qt.GlobalColor.transparent)
                painter = QtGui.QPainter(pixmap)
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.setBrush(color)
                if direction == "up":
                    points = (
                        QtCore.QPointF(1.2, 5.6),
                        QtCore.QPointF(6.8, 5.6),
                        QtCore.QPointF(4.0, 2.4),
                    )
                else:
                    points = (
                        QtCore.QPointF(1.2, 2.4),
                        QtCore.QPointF(6.8, 2.4),
                        QtCore.QPointF(4.0, 5.6),
                    )
                painter.drawPolygon(QtGui.QPolygonF(points))
                painter.end()
                if not pixmap.save(path, "PNG"):
                    return None
            paths[f"{direction}{state}"] = path.replace(os.sep, "/")
    return paths


def _spin_stepper_stylesheet(tokens: ThemeTokens) -> str:
    try:
        paths = _arrow_asset_paths(tokens)
    except Exception:
        paths = None
    if not paths:
        return ""
    return f"""
/* Reserve room so text never runs under the styled stepper buttons. */
QAbstractSpinBox {{ padding-right: 17px; }}
QAbstractSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border: none;
    background: transparent;
    border-top-right-radius: 3px;
}}
QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border: none;
    background: transparent;
    border-bottom-right-radius: 3px;
}}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {tokens.surface_alt};
}}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{
    background: {tokens.border};
}}
QAbstractSpinBox::up-arrow {{
    image: url({paths["up"]});
    width: 8px;
    height: 8px;
}}
QAbstractSpinBox::down-arrow {{
    image: url({paths["down"]});
    width: 8px;
    height: 8px;
}}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url({paths["up_muted"]});
}}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({paths["down_muted"]});
}}
"""


def build_stylesheet(tokens: ThemeTokens) -> str:
    t = tokens
    return (
        _spin_stepper_stylesheet(tokens)
        + f"""
/* ---------- global chrome ---------- */
QMainWindow, QDialog {{
    background: {t.window};
}}
QToolTip {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.border_strong};
    border-radius: 4px;
    padding: 4px 7px;
    font-size: 9pt;
}}

/* ---------- menus ---------- */
QMenuBar {{
    background: {t.window};
    color: {t.text};
    border-bottom: 1px solid {t.border};
    padding: 1px 4px;
    font-size: 9pt;
}}
QMenuBar::item {{
    padding: 4px 9px;
    border-radius: 4px;
    background: transparent;
}}
QMenuBar::item:selected {{ background: {t.surface_alt}; }}
QMenuBar::item:pressed {{ background: {t.accent}; color: {t.accent_text}; }}
QMenu {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    padding: 4px;
    font-size: 9pt;
}}
QMenu::item {{
    padding: 5px 22px 5px 10px;
    border-radius: 4px;
}}
QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
QMenu::item:disabled {{ color: {t.text_muted}; }}
QMenu::separator {{
    height: 1px;
    background: {t.border};
    margin: 4px 8px;
}}
QMenu::icon {{ padding-left: 6px; }}

/* ---------- toolbar ---------- */
QToolBar {{
    background: {t.window};
    border: none;
    padding: 3px 6px;
    spacing: 4px;
}}
QToolBar::separator {{
    width: 1px;
    background: {t.border};
    margin: 4px 6px;
}}
QToolBar QLabel {{
    color: {t.text_muted};
    font-size: 9pt;
    padding: 0 2px;
}}

/* ---------- buttons ---------- */
QToolButton {{
    background: transparent;
    color: {t.text};
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 9pt;
}}
QToolButton:hover {{ background: {t.surface_alt}; border-color: {t.border}; }}
QToolButton:pressed {{ background: {t.border}; }}
QToolButton:checked {{
    background: {t.accent};
    color: {t.accent_text};
    border-color: {t.accent};
}}
QToolButton:disabled {{ color: {t.text_muted}; }}
QPushButton {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.border_strong};
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 9pt;
}}
QPushButton:hover {{ background: {t.surface_alt}; }}
QPushButton:pressed {{ background: {t.border}; }}
QPushButton:checked {{
    background: {t.accent};
    color: {t.accent_text};
    border-color: {t.accent};
}}
QPushButton:disabled {{ color: {t.text_muted}; background: {t.window}; }}
QPushButton:default {{ border-color: {t.accent}; }}

/* ---------- inputs ---------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QAbstractSpinBox {{
    background: {t.base};
    color: {t.text};
    border: 1px solid {t.border_strong};
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 9pt;
    selection-background-color: {t.accent};
    selection-color: {t.accent_text};
}}
QComboBox:hover, QLineEdit:hover, QAbstractSpinBox:hover {{ border-color: {t.accent}; }}
QComboBox:focus, QLineEdit:focus, QAbstractSpinBox:focus {{ border-color: {t.accent}; }}
QComboBox:disabled, QLineEdit:disabled, QAbstractSpinBox:disabled {{
    color: {t.text_muted};
    background: {t.window};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background: {t.surface};
    color: {t.text};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
    selection-background-color: {t.accent};
    selection-color: {t.accent_text};
    outline: 0;
}}

/* ---------- docks & panels ---------- */
QDockWidget {{
    color: {t.text};
    font-size: 9pt;
}}
QLabel#ManagedDockTitleLabel, QLabel#DetachedPanelMoveHandle {{
    color: {t.text};
    font-size: 9pt;
    font-weight: 600;
    padding: 1px 2px;
}}
QStatusBar {{
    background: {t.window};
    color: {t.text_muted};
    border-top: 1px solid {t.border};
    font-size: 8.5pt;
}}
QStatusBar::item {{ border: none; }}

/* ---------- item views ---------- */
QTableView, QTreeView {{
    background: {t.base};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 4px;
    gridline-color: {t.border};
    alternate-background-color: {t.surface_alt};
    selection-background-color: {t.accent};
    selection-color: {t.accent_text};
    font-size: 9pt;
    outline: 0;
}}
QListView, QListWidget {{
    background: {t.base};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 4px;
    alternate-background-color: {t.surface_alt};
    font-size: 9pt;
    outline: 0;
}}
/* Selected list rows keep readable text (row widgets carry their own
   colors); the accent lives in an edge marker instead of a full fill. */
QListView::item:selected, QListWidget::item:selected {{
    background: {t.surface_alt};
    border-left: 3px solid {t.accent};
    color: {t.text};
}}
QListView::item:hover:!selected, QListWidget::item:hover:!selected {{
    background: {t.surface_alt};
}}
QHeaderView::section {{
    background: {t.surface};
    color: {t.text_muted};
    border: none;
    border-bottom: 1px solid {t.border_strong};
    border-right: 1px solid {t.border};
    padding: 3px 6px;
    font-size: 8.5pt;
    font-weight: 600;
}}

/* ---------- tabs ---------- */
QTabWidget::pane {{
    border: 1px solid {t.border};
    border-radius: 4px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {t.text_muted};
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 4px 10px;
    font-size: 9pt;
}}
QTabBar::tab:selected {{
    background: {t.surface};
    color: {t.text};
    border-color: {t.border};
}}
QTabBar::tab:hover:!selected {{ color: {t.text}; }}

/* ---------- group boxes ---------- */
QGroupBox {{
    font-size: 9pt;
    font-weight: 600;
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 5px;
    margin-top: 1.4ex;
    padding-top: 4pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
}}

/* ---------- scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {t.border_strong};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {t.text_muted}; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {t.border_strong};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {t.text_muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---------- checkboxes / radios ---------- */
QCheckBox, QRadioButton {{
    color: {t.text};
    font-size: 9pt;
    spacing: 5px;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
}}

/* ---------- dimension strip chips ---------- */
QFrame[dimensionChip="true"] {{
    background: {t.surface};
    border: 1px solid {t.border};
    border-radius: 6px;
}}
QFrame[dimensionChip="true"]:focus {{ border-color: {t.accent}; }}
QFrame[dimensionChip="true"][montageAxis="true"] {{ border: 1px solid {t.accent}; }}
QFrame[dimensionChip="true"] QLabel {{
    font-size: 9pt;
    color: {t.text};
}}
QFrame[dimensionChip="true"] QToolButton {{
    padding: 2px 0;
    min-width: 24px;
    font-weight: 600;
}}
QToolButton#DimChipBadge {{
    background: {t.surface_alt};
    color: {t.text_muted};
    font-size: 8.5pt;
    font-weight: 600;
    padding: 2px 0;
    border: none;
    border-top-left-radius: 5px;
    border-bottom-left-radius: 5px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border-right: 1px solid {t.border};
}}
QToolButton#DimChipBadge:hover {{
    background: {t.border};
    color: {t.text};
}}
QToolButton#DimChipBadge:checked {{
    background: {t.accent};
    color: {t.accent_text};
    border-right: 1px solid {t.accent};
}}
QToolButton#DimChipBadge:disabled {{
    color: {t.text_muted};
    background: {t.surface};
}}
QFrame#DimChipSeparator {{
    background: {t.border};
    border: none;
    margin-top: 5px;
    margin-bottom: 5px;
}}

/* ---------- floating chips ---------- */
QLabel#EvaluationOverlay, QFrame#RoiInfoPanel, QFrame#PixelHud {{
    background: {t.overlay_bg};
    color: {t.overlay_text};
    border: 1px solid {t.border_strong};
    border-radius: 5px;
    padding: 5px 8px;
    font-size: 9pt;
}}
QFrame#PixelHud, QFrame#RoiInfoPanel {{ padding: 0; }}
QFrame#RoiInfoPanel QLabel {{
    background: transparent;
    border: none;
    color: {t.overlay_text};
    font-size: 9pt;
}}
QFrame#PixelHud QLabel {{
    background: transparent;
    border: none;
    color: {t.overlay_text};
    font-size: 9pt;
}}
QFrame#PixelHudSeparator {{
    background: {t.border};
    border: none;
    margin: 1px 2px;
}}
QFrame#EditBubble {{
    background: {t.surface};
    border: 1px solid {t.border_strong};
    border-radius: 6px;
}}
QFrame#FirstRunHints {{
    background: {t.overlay_bg};
    border: 1px solid {t.accent};
    border-radius: 6px;
}}
QFrame#FirstRunHints QLabel {{
    background: transparent;
    border: none;
    color: {t.overlay_text};
    font-size: 9pt;
}}
QPushButton#FirstRunHintsDismiss {{
    padding: 2px 10px;
    font-size: 8.5pt;
}}
QFrame#EditBubble QLabel {{ font-size: 9pt; }}
QFrame#RoiInfoDivider {{
    background: {t.border};
    border: none;
    margin: 1px 0;
}}
QLabel#ArrayScopeStatusMessageLabel {{
    color: {t.text_muted};
    font-size: 8.5pt;
}}
QLabel#OperationsMetaLabel {{
    color: {t.text_muted};
    font-size: 8.5pt;
}}
QLabel#ArrayInfoLabel {{
    color: {t.text_muted};
    font-size: 8.5pt;
}}
QToolButton#OperationAxisButton {{
    color: {t.accent};
    font-size: 8pt;
    font-weight: 600;
    padding: 0 3px;
    border: 1px solid transparent;
    border-radius: 3px;
}}
QToolButton#OperationAxisButton:hover {{
    border-color: {t.accent};
    background: {t.surface_alt};
}}

/* ---------- splitters ---------- */
QSplitter::handle {{ background: {t.border}; }}
QSplitter::handle:hover {{ background: {t.accent}; }}
"""
    )
