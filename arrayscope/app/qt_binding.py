"""Qt binding selection for pyqtgraph.

ArrayScope uses pyqtgraph's Qt abstraction directly. This module only chooses
the default binding before pyqtgraph is imported.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_QT_BINDING_MODULES = ("PyQt6", "PySide6", "PyQt5", "PySide2")


def prefer_pyside6() -> None:
    """Make PySide6 the pyqtgraph default unless the process already chose Qt."""
    if os.environ.get("PYQTGRAPH_QT_LIB"):
        return

    if any(module in sys.modules for module in _QT_BINDING_MODULES):
        return

    if importlib.util.find_spec("PySide6") is not None:
        os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"


def pin_wayland_appearance() -> None:
    """Keep the window chrome ArrayScope has always had, whatever the wheel bundles.

    The PyPI PySide6 wheel bundles more Wayland plugins than the conda-forge
    build this project developed under: an ``adwaita`` client-side decoration
    and a ``qgtk3`` platform theme. Qt auto-prefers both on GNOME, so switching
    the dev environment to uv silently changed the titlebar and widget metrics
    (2026-07-30) — the app's face was an accident of which plugins a package
    manager shipped. Pinned here instead: Qt's own ``bradient`` decorations and
    the portal platform theme, which is what every ArrayScope window has looked
    like since the project began.

    Defaults, not mandates — every write is ``setdefault``, so an explicit
    ``QT_WAYLAND_DECORATION``/``QT_QPA_PLATFORMTHEME`` in the environment wins.

    Scoped to where the pin is *right*: Wayland only (X11 titlebars are the
    window manager's, not a Qt plugin's), never headless, and the theme pin is
    GNOME-only — on KDE the compositor draws server-side decorations and Qt's
    own ``kde`` theme is the native look, so pinning there would make the app
    less native, not more. Environment reads happen at QApplication
    construction, not at import, so the pin runs from
    ``launch._prepare_qt_environment()`` — the one pre-QApplication hook — and
    deliberately NOT at module import: an import-time pin fires before a test
    harness can say "offscreen", and a desktop session's ``WAYLAND_DISPLAY``
    then leaks the portal theme into headless processes, which pay real DBus
    round-trips for it (measured 11.6 s → 30.5 s on one settle-budget test).
    """

    if not os.environ.get("WAYLAND_DISPLAY"):
        return
    if os.environ.get("QT_QPA_PLATFORM", "").startswith(("offscreen", "minimal")):
        return
    os.environ.setdefault("QT_WAYLAND_DECORATION", "bradient")
    if "gnome" in os.environ.get("XDG_CURRENT_DESKTOP", "").lower():
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "xdgdesktopportal")
