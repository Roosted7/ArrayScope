import importlib.util
from pathlib import Path

import numpy as np
import pytest

from arrayscope.operations import registry


def _append(coordinator, operation_id, axis=None, parameters=None, *, axes=None):
    """Create then append -- the two steps the Add-operation flow takes."""

    coordinator.append_created_operation(
        registry.create_operation(operation_id, axis=axis, parameters=parameters or {}, axes=axes)
    )


ROOT = Path(__file__).parents[2]


MODULE_PATHS = {
    "axis_utils": ("arrayscope.core.axis_utils", ROOT / "arrayscope" / "core" / "axis_utils.py"),
    "cache_status": (
        "arrayscope.core.cache_status",
        ROOT / "arrayscope" / "core" / "cache_status.py",
    ),
    "view_state": ("arrayscope.core.view_state", ROOT / "arrayscope" / "core" / "view_state.py"),
    "window_levels": (
        "arrayscope.core.window_levels",
        ROOT / "arrayscope" / "core" / "window_levels.py",
    ),
    "dim_ops": ("arrayscope.operations.dim_ops", ROOT / "arrayscope" / "operations" / "dim_ops.py"),
    "operation_pipeline": (
        "arrayscope.operations.pipeline",
        ROOT / "arrayscope" / "operations" / "pipeline.py",
    ),
    "operation_stack": (
        "arrayscope.operations.stack",
        ROOT / "arrayscope" / "operations" / "stack.py",
    ),
    "operation_evaluator": (
        "arrayscope.operations.evaluator",
        ROOT / "arrayscope" / "operations" / "evaluator.py",
    ),
    "operation_registry": (
        "arrayscope.operations.registry",
        ROOT / "arrayscope" / "operations" / "registry.py",
    ),
    "operation_recipes": (
        "arrayscope.operations.recipes",
        ROOT / "arrayscope" / "operations" / "recipes.py",
    ),
    "operation_coordinator": (
        "arrayscope.operations.coordinator",
        ROOT / "arrayscope" / "operations" / "coordinator.py",
    ),
    "slice_engine": (
        "arrayscope.display.slice_engine",
        ROOT / "arrayscope" / "display" / "slice_engine.py",
    ),
    "profile": ("arrayscope.profiles.model", ROOT / "arrayscope" / "profiles" / "model.py"),
    "profile_coordinator": (
        "arrayscope.profiles.coordinator",
        ROOT / "arrayscope" / "profiles" / "coordinator.py",
    ),
    "theme": ("arrayscope.app.theme", ROOT / "arrayscope" / "app" / "theme.py"),
    "settings_state": (
        "arrayscope.app.settings_state",
        ROOT / "arrayscope" / "app" / "settings_state.py",
    ),
}


def load_module(name):
    module_name, _path = MODULE_PATHS[name]
    return importlib.import_module(module_name)


load_module("axis_utils")
load_module("dim_ops")
operation_pipeline = load_module("operation_pipeline")
load_module("operation_stack")
load_module("cache_status")
load_module("slice_engine")
load_module("operation_evaluator")
load_module("operation_registry")
operation_coordinator = load_module("operation_coordinator")

OperationCoordinator = operation_coordinator.OperationCoordinator
CenteredFFT = operation_pipeline.CenteredFFT
OperationStep = operation_pipeline.OperationStep
ReverseAxis = operation_pipeline.ReverseAxis


def test_operation_coordinator_appends_reorders_and_materializes():
    data = np.arange(3 * 4).reshape(3, 4)
    coordinator = OperationCoordinator(data)

    _append(coordinator, "crop", axis=1, parameters={"start": 1, "stop": 4})
    _append(coordinator, "reverse", axis=0)

    assert coordinator.shape == (3, 3)
    assert coordinator.operation_shapes() == ((3, 3), (3, 3))

    result = coordinator.evaluator.document.materialize()
    np.testing.assert_array_equal(result, np.flip(data[:, 1:4], axis=0))

    coordinator.materialize()
    assert coordinator.document.operations == ()
    np.testing.assert_array_equal(coordinator.base_data, result)


def test_operation_coordinator_delete_and_move_validate_against_base_shape():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    coordinator = OperationCoordinator(data)
    _append(coordinator, "crop", axis=2, parameters={"start": 1, "stop": 4})
    _append(coordinator, "mean", axis=0)

    with pytest.raises(ValueError, match="out of bounds"):
        coordinator.move(1, -1)

    assert coordinator.shape == (3, 3)

    coordinator.delete(0)
    assert coordinator.shape == (3, 4)


def test_operation_dtype_estimates_delegate_to_cost_model():
    coordinator = OperationCoordinator(np.zeros((8, 16), dtype=np.float32))
    _append(coordinator, "centered_fft", axis=0)

    assert coordinator.operation_dtype_estimates() == (np.dtype(np.complex64),)


def test_the_coordinator_takes_an_axis_set_from_either_channel():
    """The two ways a surface hands over an axis set, both landing the same op.

    A popup forwards ``form.values()``, which carries the selection under the
    reserved key; a recipe or a direct caller passes ``axes=``. Neither is a
    special case for the coordinator.
    """

    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    from_form = OperationCoordinator(data)
    _append(from_form, "mean_axes", axis=None, parameters={"axes": (0, 2)})

    explicit = OperationCoordinator(data)
    _append(explicit, "mean_axes", axes=(2, 0))

    assert from_form.shape == (3,)
    assert from_form.document.operations == explicit.document.operations
    np.testing.assert_array_equal(
        from_form.evaluator.document.materialize(), data.mean(axis=(0, 2))
    )


def test_the_coordinator_names_an_out_of_range_axis_set_before_evaluating():
    coordinator = OperationCoordinator(np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="out of bounds"):
        _append(coordinator, "centered_fftn", axes=(0, 2))
