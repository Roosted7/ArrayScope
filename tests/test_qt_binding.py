import importlib.util
import os
import subprocess
import sys

import pytest


def test_arrayscope_defaults_pyqtgraph_to_pyside6():
    env = os.environ.copy()
    env.pop("PYQTGRAPH_QT_LIB", None)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import arrayscope, pyqtgraph.Qt as Qt; print(Qt.QT_LIB)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-1] == "PySide6"


def test_arrayscope_respects_explicit_pyqtgraph_binding():
    if importlib.util.find_spec("PyQt5") is None:
        pytest.skip("PyQt5 is not installed")

    env = os.environ.copy()
    env["PYQTGRAPH_QT_LIB"] = "PyQt5"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import arrayscope, pyqtgraph.Qt as Qt; print(Qt.QT_LIB)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.splitlines()[-1] == "PyQt5"
