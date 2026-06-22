import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


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


load_module("axis_utils")
view_state = load_module("view_state")
load_module("profile")
load_module("slice_engine")
load_module("operation_pipeline")
load_module("cache_status")
operation_evaluator = load_module("operation_evaluator")
profile_coordinator = load_module("profile_coordinator")

ViewState = view_state.ViewState
ArrayDocument = sys.modules["arrayscope.operations.pipeline"].ArrayDocument
OperationEvaluator = operation_evaluator.OperationEvaluator
ProfileCoordinator = profile_coordinator.ProfileCoordinator


def test_profile_coordinator_clamps_and_renders_line_result():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    state = ViewState.from_shape(data.shape).with_line_axis(2)
    coordinator = ProfileCoordinator()
    evaluator = OperationEvaluator(ArrayDocument(data))

    result = coordinator.render_from_marker(
        evaluator,
        state,
        10,
        -2,
        line_axis=2,
        y_range_mode="match_image",
        image_levels=(1, 9),
    )

    assert result.marker_position == (2, 0)
    assert result.view_state.slice_indices == (0, 2, 2)
    assert result.y_range == (1.0, 9.0)
    np.testing.assert_array_equal(result.line_result.data, data[0, 2, :])
