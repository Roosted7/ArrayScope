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
