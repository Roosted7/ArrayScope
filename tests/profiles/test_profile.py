import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
PACKAGE = types.ModuleType("arrayscope")
PACKAGE.__path__ = [str(ROOT / "arrayscope")]
sys.modules.setdefault("arrayscope", PACKAGE)


MODULE_PATHS = {
    "axis_utils": ("arrayscope.core.axis_utils", ROOT / "arrayscope" / "core" / "axis_utils.py"),
    "cache_status": ("arrayscope.core.cache_status", ROOT / "arrayscope" / "core" / "cache_status.py"),
    "dimension_roles": ("arrayscope.core.dimension_roles", ROOT / "arrayscope" / "core" / "dimension_roles.py"),
    "view_state": ("arrayscope.core.view_state", ROOT / "arrayscope" / "core" / "view_state.py"),
    "window_levels": ("arrayscope.core.window_levels", ROOT / "arrayscope" / "core" / "window_levels.py"),
    "dim_ops": ("arrayscope.operations.dim_ops", ROOT / "arrayscope" / "operations" / "dim_ops.py"),
    "operation_pipeline": ("arrayscope.operations.pipeline", ROOT / "arrayscope" / "operations" / "pipeline.py"),
    "operation_stack": ("arrayscope.operations.stack", ROOT / "arrayscope" / "operations" / "stack.py"),
    "operation_evaluator": ("arrayscope.operations.evaluator", ROOT / "arrayscope" / "operations" / "evaluator.py"),
    "operation_registry": ("arrayscope.operations.registry", ROOT / "arrayscope" / "operations" / "registry.py"),
    "operation_recipes": ("arrayscope.operations.recipes", ROOT / "arrayscope" / "operations" / "recipes.py"),
    "operation_coordinator": ("arrayscope.operations.coordinator", ROOT / "arrayscope" / "operations" / "coordinator.py"),
    "slice_engine": ("arrayscope.display.slice_engine", ROOT / "arrayscope" / "display" / "slice_engine.py"),
    "profile": ("arrayscope.profiles.model", ROOT / "arrayscope" / "profiles" / "model.py"),
    "profile_coordinator": ("arrayscope.profiles.coordinator", ROOT / "arrayscope" / "profiles" / "coordinator.py"),
    "theme": ("arrayscope.app.theme", ROOT / "arrayscope" / "app" / "theme.py"),
    "settings_state": ("arrayscope.app.settings_state", ROOT / "arrayscope" / "app" / "settings_state.py"),
}


def load_module(name):
    module_name, path = MODULE_PATHS[name]
    spec = importlib.util.spec_from_file_location(module_name, path)
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


def test_marker_position_is_clamped_to_image_axes():
    assert profile.clamp_marker_position((2, 3, 4), (0, 1), image_x=10, image_y=-2) == (2, 0)


def test_profile_states_from_marker_supports_multiple_profile_axes():
    state = state_for((2, 3, 4), image_axes=(0, 1), line_axis=2)

    states = profile.profile_states_from_marker(state, image_x=2, image_y=1, profile_axes=(1, 2))

    assert tuple(profile_state.line_axis for profile_state in states) == (1, 2)
    assert states[0].slice_indices == (1, 0, 0)
    assert states[1].slice_indices == (1, 2, 0)


def test_profile_y_range_matches_image_window_only_when_requested():
    assert profile.profile_y_range("match_image", (1, 5)) == (1.0, 5.0)
    assert profile.profile_y_range("auto", (1, 5)) is None
    assert profile.profile_y_range("match_image", None) is None


def test_profile_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse((ROOT / "arrayscope" / "profiles" / "model.py").read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
