"""Entry point for the frozen (PyInstaller) ArrayScope bundles.

Thin shell around ``arrayscope.__main__.main`` with the three behaviors a
double-clickable "viewer" build needs and the pip install does not:

- ``multiprocessing.freeze_support()`` before anything else: selector and
  non-blocking windows spawn worker processes, and in a frozen app the spawn
  re-executes this binary.
- Launched with no arguments (double-click, Start Menu, dock): show a file
  dialog instead of the CLI usage error.
- On Windows/macOS, pre-create the QApplication to attach the window icon
  bundled under arrayscope/resources. On Linux the QApplication must NOT be
  created here — arrayscope.__main__ runs the Wayland/xcb supervision logic
  first, and an existing QApplication would disable it; icons come from the
  AppImage/.desktop metadata instead.
"""

from __future__ import annotations

import multiprocessing
import sys

SUPPORTED_PATTERNS = "*.npy *.npz *.h5 *.hdf5 *.mat *.cfl *.rec *.dcm *.nii *.nii.gz *.txt"


def _icon_path():
    from pathlib import Path

    import arrayscope

    return Path(arrayscope.__file__).parent / "resources" / "icons" / "arrayscope-256.png"


def _make_app_with_icon():
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    app = pg.mkQApp()
    icon_file = _icon_path()
    if icon_file.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_file)))
    return app


def _ask_for_file():
    """File-open dialog for the no-arguments launch; None when cancelled."""
    _make_app_with_icon()
    from pyqtgraph.Qt import QtWidgets

    path, _selected = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Open array file - ArrayScope",
        "",
        f"Array data ({SUPPORTED_PATTERNS});;All files (*)",
    )
    return path or None


def main() -> int:
    multiprocessing.freeze_support()

    if len(sys.argv) == 1:
        chosen = _ask_for_file()
        if chosen is None:
            return 0
        sys.argv.append(chosen)
    elif sys.platform in ("win32", "darwin"):
        # Supervision (Linux-only) is inert here, so creating the app early
        # is safe and lets every window inherit the bundled icon.
        _make_app_with_icon()

    from arrayscope.__main__ import main as cli_main

    return cli_main() or 0


if __name__ == "__main__":
    sys.exit(main())
