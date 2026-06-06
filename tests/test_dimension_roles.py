import ast
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
DIMENSION_ROLES_PATH = ROOT / "arrayscope" / "core" / "dimension_roles.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.dimension_roles", DIMENSION_ROLES_PATH)
dimension_roles = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dimension_roles
SPEC.loader.exec_module(dimension_roles)

DimensionRoles = dimension_roles.DimensionRoles


def test_dimension_roles_keep_image_axes_distinct_when_assigning_y_or_x():
    roles = DimensionRoles.from_axes((0, 1), profile_axes=(2,))

    roles = roles.with_image_axis("y", 1)
    assert roles.image_axes == (1, 0)
    assert roles.profile_axes == (2,)

    roles = roles.with_image_axis("x", 1)
    assert roles.image_axes == (0, 1)


def test_dimension_roles_track_profile_axes_as_tuple_for_future_multi_profile():
    roles = DimensionRoles.from_axes((0, 1))

    roles = roles.with_toggled_profile_axis(2).with_toggled_profile_axis(3)
    assert roles.profile_axes == (2, 3)

    roles = roles.with_single_profile_axis(1)
    assert roles.profile_axes == (1,)


def test_dimension_roles_reject_duplicate_image_axes():
    with pytest.raises(ValueError, match="distinct"):
        DimensionRoles.from_axes((0, 0))


def test_dimension_roles_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse(DIMENSION_ROLES_PATH.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
