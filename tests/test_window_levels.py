import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
WINDOW_LEVELS_PATH = ROOT / "arrayscope" / "core" / "window_levels.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.window_levels", WINDOW_LEVELS_PATH)
window_levels = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = window_levels
SPEC.loader.exec_module(window_levels)


def test_absolute_window_reuses_previous_numeric_levels():
    decision = window_levels.choose_window_levels(
        mode="absolute",
        previous_levels=(10, 20),
        previous_bounds=(0, 100),
        current_bounds=(1000, 2000),
        default_levels=(0, 100),
    )

    assert decision.auto_levels is False
    assert decision.levels == (10.0, 20.0)


def test_relative_window_maps_previous_fractions_to_current_bounds():
    decision = window_levels.choose_window_levels(
        mode="relative",
        previous_levels=(25, 75),
        previous_bounds=(0, 100),
        current_bounds=(200, 300),
    )

    assert decision.auto_levels is False
    assert decision.levels == (225.0, 275.0)


def test_relative_window_auto_levels_without_previous_state():
    decision = window_levels.choose_window_levels(
        mode="relative",
        previous_levels=None,
        previous_bounds=None,
        current_bounds=(200, 300),
        default_levels=(-3.14, 3.14),
    )

    assert decision.auto_levels is True
    assert decision.levels == (-3.14, 3.14)


def test_force_auto_overrides_absolute_window_for_channel_or_scale_changes():
    decision = window_levels.choose_window_levels(
        mode="absolute",
        previous_levels=(10, 20),
        previous_bounds=(0, 100),
        current_bounds=(200, 300),
        default_levels=None,
        force_auto=True,
    )

    assert decision.auto_levels is True
    assert decision.levels is None


def test_window_levels_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse(WINDOW_LEVELS_PATH.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
