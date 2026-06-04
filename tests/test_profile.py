import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)


def load_module(name):
    path = ROOT / "arrayscope" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"arrayscope.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


view_state = load_module("view_state")
profile = load_module("profile")

ViewState = view_state.ViewState


def state_for(shape, image_axes=(0, 1), line_axis=2, slices=None):
    return ViewState(
        ndim=len(shape),
        shape=tuple(shape),
        image_axes=image_axes,
        line_axis=line_axis,
        slice_indices=tuple(slices if slices is not None else (0,) * len(shape)),
        axis_flipped=(False,) * len(shape),
        axis_fftshifted=(False,) * len(shape),
    )


def test_profile_state_maps_image_x_to_secondary_and_y_to_primary_axis():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=2)

    profile_state = profile.profile_state_from_image_hover(state, image_x=2, image_y=1)

    assert profile_state.line_axis == 2
    assert profile_state.slice_indices == (1, 2, 0)


def test_profile_state_preserves_line_axis_when_line_axis_is_image_axis():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=1, slices=(0, 0, 3))

    profile_state = profile.profile_state_from_image_hover(state, image_x=2, image_y=1)

    assert profile_state.line_axis == 1
    assert profile_state.slice_indices == (1, 0, 3)


def test_profile_state_handles_reversed_image_axes():
    state = state_for((2, 3, 4, 5), image_axes=(3, 1), line_axis=2, slices=(1, 0, 0, 0))

    profile_state = profile.profile_state_from_image_hover(state, image_x=2, image_y=4)

    assert profile_state.slice_indices == (1, 2, 0, 4)


@pytest.mark.parametrize("image_x,image_y", [(-1, 0), (3, 0), (0, -1), (0, 2)])
def test_profile_state_returns_none_outside_image_bounds(image_x, image_y):
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=2)

    assert profile.profile_state_from_image_hover(state, image_x=image_x, image_y=image_y) is None


def test_profile_y_range_matches_image_window_only_when_requested():
    assert profile.profile_y_range("match_image", (1, 5)) == (1.0, 5.0)
    assert profile.profile_y_range("auto", (1, 5)) is None
    assert profile.profile_y_range("match_image", None) is None


def test_profile_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse((ROOT / "arrayscope" / "profile.py").read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
