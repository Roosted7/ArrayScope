import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


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


load_module("axis_utils")
view_state = load_module("view_state")
load_module("profile")
load_module("slice_engine")
load_module("operation_pipeline")
load_module("cache_status")
operation_evaluator = load_module("operation_evaluator")
profile_coordinator = load_module("profile_coordinator")

ViewState = view_state.ViewState
ArrayDocument = sys.modules["arrayscope.operation_pipeline"].ArrayDocument
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
    assert result.view_state.slice_indices == (0, 2, 0)
    assert result.y_range == (1.0, 9.0)
    np.testing.assert_array_equal(result.line_result.data, data[0, 2, :])
