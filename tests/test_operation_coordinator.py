import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]
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
load_module("dim_ops")
load_module("operation_pipeline")
load_module("operation_stack")
load_module("cache_status")
load_module("slice_engine")
load_module("operation_evaluator")
load_module("operation_registry")
operation_coordinator = load_module("operation_coordinator")

OperationCoordinator = operation_coordinator.OperationCoordinator


def test_operation_coordinator_appends_reorders_and_materializes():
    data = np.arange(3 * 4).reshape(3, 4)
    coordinator = OperationCoordinator(data)

    coordinator.append_operation("crop", axis=1, parameters={"start": 1, "stop": 4})
    coordinator.append_operation("reverse", axis=0)

    assert coordinator.shape == (3, 3)
    assert coordinator.operation_shapes() == ((3, 3), (3, 3))

    result = coordinator.evaluator.current_data()
    np.testing.assert_array_equal(result, np.flip(data[:, 1:4], axis=0))

    coordinator.materialize()
    assert coordinator.document.operations == ()
    np.testing.assert_array_equal(coordinator.base_data, result)


def test_operation_coordinator_delete_and_move_validate_against_base_shape():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    coordinator = OperationCoordinator(data)
    coordinator.append_operation("crop", axis=2, parameters={"start": 1, "stop": 4})
    coordinator.append_operation("mean", axis=0)

    with pytest.raises(ValueError, match="out of bounds"):
        coordinator.move(1, -1)

    assert coordinator.shape == (3, 3)

    coordinator.delete(0)
    assert coordinator.shape == (3, 4)
