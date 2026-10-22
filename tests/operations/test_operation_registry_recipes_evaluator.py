import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _operations_from_recipe(module, recipe, base_shape, **kwargs):
    """The enabled operations of a recipe, the way a loader reads them."""

    return tuple(
        step.operation
        for step in module.steps_from_recipe(recipe, base_shape, **kwargs)
        if step.enabled
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


view_state = load_module("view_state")
load_module("slice_engine")
operation_pipeline = load_module("operation_pipeline")
operation_registry = load_module("operation_registry")
operation_recipes = load_module("operation_recipes")
cache_status = load_module("cache_status")
operation_evaluator = load_module("operation_evaluator")

ArrayDocument = operation_pipeline.ArrayDocument
CenteredFFT = operation_pipeline.CenteredFFT
CenteredFFTN = operation_pipeline.CenteredFFTN
MeanOverAxes = operation_pipeline.MeanOverAxes
CombineRealImagAxis = operation_pipeline.CombineRealImagAxis
Conjugate = operation_pipeline.Conjugate
Crop = operation_pipeline.Crop
FFTShift = operation_pipeline.FFTShift
Mean = operation_pipeline.Mean
ReverseAxis = operation_pipeline.ReverseAxis
RootSumSquares = operation_pipeline.RootSumSquares
SplitComplexAxis = operation_pipeline.SplitComplexAxis
OperationEvaluator = operation_evaluator.OperationEvaluator
CacheStatus = cache_status.CacheStatus
ViewState = view_state.ViewState
OperationStep = operation_pipeline.OperationStep


def test_operation_registry_contains_expected_dimension_actions():
    entries = {entry.id: entry for entry in operation_registry.operation_entries()}

    for operation_id in (
        "crop",
        "reverse",
        "conjugate",
        "mean",
        "rss",
        "centered_fft",
        "centered_ifft",
        "fftshift",
        "combine_real_imag",
        "split_complex",
    ):
        assert operation_id in entries
        assert entries[operation_id].label
        assert entries[operation_id].operation_type is not None

    assert entries["crop"].changes_shape is True
    assert [parameter.name for parameter in entries["crop"].parameters] == ["start", "stop"]
    assert entries["conjugate"].requires_axis is False


def test_recipe_json_round_trip_preserves_operation_stack():
    operations = (
        Crop(axis=1, start=1, stop=3),
        ReverseAxis(axis=0),
        Conjugate(),
        CombineRealImagAxis(axis=2),
        SplitComplexAxis(axis=2),
        Mean(axis=1),
        RootSumSquares(axis=0),
        CenteredFFT(axis=0),
        FFTShift(axis=0),
    )

    text = operation_recipes.dumps_recipe(operations)
    loaded = operation_recipes.loads_recipe(text, base_shape=(3, 4, 2))

    assert loaded == operations


def test_recipe_file_round_trip(tmp_path):
    operations = (Crop(axis=0, start=0, stop=2), ReverseAxis(axis=1))
    recipe_path = tmp_path / "recipe.json"

    operation_recipes.save_recipe(recipe_path, operations)
    loaded = operation_recipes.load_recipe(recipe_path, base_shape=(3, 4))

    assert loaded == operations


def test_recipe_v2_preserves_disabled_steps():
    steps = (
        OperationStep(Crop(axis=1, start=1, stop=3), enabled=True),
        OperationStep(Mean(axis=0), enabled=False),
    )

    text = operation_recipes.dumps_recipe(steps)
    loaded_steps = operation_recipes.loads_recipe_steps(text, base_shape=(3, 4))

    assert tuple(step.operation for step in loaded_steps) == tuple(step.operation for step in steps)
    assert tuple(step.enabled for step in loaded_steps) == (True, False)
    assert operation_recipes.loads_recipe(text, base_shape=(3, 4)) == (
        Crop(axis=1, start=1, stop=3),
    )


def test_axis_set_operations_round_trip_through_a_recipe():
    operations = (
        CenteredFFTN(axes=(0, 2)),
        MeanOverAxes(axes=(0, 1)),
    )

    text = operation_recipes.dumps_recipe(operations)

    # The set serializes under its own key, parallel to a single ``axis``.
    assert '"axes"' in text
    assert operation_recipes.loads_recipe(text, base_shape=(3, 4, 2)) == operations


def test_an_axis_set_recipe_is_read_back_as_a_set_whatever_order_it_names():
    recipe = {
        "version": operation_recipes.RECIPE_VERSION,
        "operations": [{"id": "centered_fftn", "axes": [2, 0]}],
    }

    loaded = _operations_from_recipe(operation_recipes, recipe, base_shape=(3, 4, 2))

    # Written backwards, loaded as the same operation -- otherwise two identical
    # pipelines would miss each other's stage-cache entries.
    assert loaded == (CenteredFFTN(axes=(0, 2)),)


def test_a_recipe_written_before_axis_sets_existed_still_loads_unchanged():
    """The compatibility direction: no ``axes`` key anywhere, nothing changes."""

    legacy = {
        "version": operation_recipes.LEGACY_RECIPE_VERSION,
        "operations": [
            {"id": "crop", "axis": 1, "parameters": {"start": 1, "stop": 3}},
            {"id": "conjugate"},
            {"id": "mean", "axis": 0},
        ],
    }

    assert _operations_from_recipe(operation_recipes, legacy, base_shape=(3, 4, 2)) == (
        Crop(axis=1, start=1, stop=3),
        Conjugate(),
        Mean(axis=0),
    )


def test_an_axis_set_recipe_names_a_bad_selection_instead_of_a_shape_error():
    for axes, message in (
        ([], "at least 1"),
        ([1, 1], "more than once"),
        ([0, 5], "out of bounds"),
    ):
        recipe = {
            "version": operation_recipes.RECIPE_VERSION,
            "operations": [{"id": "centered_fftn", "axes": axes}],
        }
        with pytest.raises(ValueError, match=message):
            _operations_from_recipe(operation_recipes, recipe, base_shape=(3, 4, 2))

    # ...and the rank it is checked against is the shape entering THAT step, not
    # the base shape: axis 2 is legal at the start and gone after the reduction.
    after_reduction = {
        "version": operation_recipes.RECIPE_VERSION,
        "operations": [{"id": "mean", "axis": 0}, {"id": "centered_fftn", "axes": [0, 2]}],
    }
    with pytest.raises(ValueError, match=r"operation 1 \(centered_fftn\) is incompatible"):
        _operations_from_recipe(operation_recipes, after_reduction, base_shape=(3, 4, 2))


def test_axis_set_entries_declare_one_selection_kind_and_reject_the_other():
    entries = {entry.id: entry for entry in operation_registry.operation_entries()}

    assert entries["centered_fftn"].requires_axes is True
    assert entries["centered_fft"].requires_axis is True
    assert (entries["conjugate"].requires_axis or entries["conjugate"].requires_axes) is False

    with pytest.raises(ValueError, match="requires an axis set"):
        operation_registry.create_operation("centered_fftn")
    with pytest.raises(ValueError, match="requires an axis"):
        operation_registry.create_operation("centered_fft")

    with pytest.raises(ValueError, match="not both"):
        operation_registry.OperationEntry(
            id="broken",
            label="Broken",
            operation_type=CenteredFFTN,
            requires_axis=True,
            requires_axes=True,
        )


def test_describing_an_axis_set_operation_names_every_axis():
    assert operation_registry.describe_operation(MeanOverAxes(axes=(0, 2))) == "Mean over axes 0, 2"
    assert (
        operation_registry.describe_operation(CenteredFFTN(axes=(1, 2)))
        == "Centered FFT over axes 1, 2"
    )


def test_invalid_recipe_validation_reports_clear_error():
    with pytest.raises(ValueError, match="unsupported recipe version"):
        _operations_from_recipe(
            operation_recipes, {"version": 999, "operations": []}, base_shape=(2, 3)
        )

    invalid_crop = {
        "version": operation_recipes.RECIPE_VERSION,
        "operations": [{"id": "crop", "axis": 1, "parameters": {"start": 2, "stop": 5}}],
    }
    with pytest.raises(ValueError, match=r"operation 0 \(crop\) is incompatible"):
        _operations_from_recipe(operation_recipes, invalid_crop, base_shape=(2, 3))

    invalid_axis_after_reduction = {
        "version": operation_recipes.RECIPE_VERSION,
        "operations": [{"id": "mean", "axis": 0}, {"id": "reverse", "axis": 1}],
    }
    with pytest.raises(ValueError, match=r"operation 1 \(reverse\) is incompatible"):
        _operations_from_recipe(operation_recipes, invalid_axis_after_reduction, base_shape=(2, 3))


def test_document_all_operation_shape_changes():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    document = ArrayDocument(data)
    expected_shapes = {
        Crop(axis=1, start=1, stop=3): (2, 2, 4),
        ReverseAxis(axis=2): (2, 3, 4),
        Conjugate(): (2, 3, 4),
        Mean(axis=0): (3, 4),
        RootSumSquares(axis=2): (2, 3),
        operation_pipeline.Sum(axis=1): (2, 4),
        operation_pipeline.Maximum(axis=1): (2, 4),
        operation_pipeline.Minimum(axis=1): (2, 4),
        operation_pipeline.CenteredFFT(axis=2): (2, 3, 4),
        operation_pipeline.CenteredIFFT(axis=2): (2, 3, 4),
        operation_pipeline.FFTShift(axis=2): (2, 3, 4),
        operation_pipeline.CombineRealImagAxis(axis=0): (1, 3, 4),
    }

    for operation, shape in expected_shapes.items():
        assert document.with_operation(operation).current_shape == shape

    assert ArrayDocument(np.zeros((1, 3, 4), dtype=complex)).with_operation(
        operation_pipeline.SplitComplexAxis(axis=0)
    ).current_shape == (2, 3, 4)

    document = document.with_operation(Crop(axis=2, start=1, stop=4)).with_operation(Mean(axis=0))
    assert document.current_shape == (3, 3)
    assert ArrayDocument(data).current_shape == data.shape


def test_display_cache_invalidates_for_operations_and_view_state():
    data = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(float)
    state = ViewState.from_shape(data.shape)
    evaluator = OperationEvaluator(ArrayDocument(data))

    first = evaluator.image(state)
    second = evaluator.image(state)
    assert first is second
    assert evaluator.image_evaluations == 1

    shifted_state = state.with_slice(2, 0)
    evaluator.image(shifted_state)
    assert evaluator.image_evaluations == 2

    evaluator.set_document(evaluator.document.with_operation(ReverseAxis(axis=0)))
    evaluator.image(state)
    assert evaluator.image_evaluations == 3


def test_display_cache_invalidates_for_step_enabled_flag():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    state = ViewState.from_shape(data.shape)
    document = ArrayDocument(data).with_operation(ReverseAxis(axis=0))
    evaluator = OperationEvaluator(document)

    evaluator.image(state)
    evaluator.set_document(
        ArrayDocument(data, steps=(OperationStep(ReverseAxis(axis=0), enabled=False),))
    )
    evaluator.image(state)

    assert evaluator.image_evaluations == 2


def test_display_cache_status_tracks_display_hit_and_miss():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    state = ViewState.from_shape(data.shape)
    evaluator = OperationEvaluator(ArrayDocument(data))

    evaluator.image(state)
    assert evaluator.last_status.status == CacheStatus.READY

    evaluator.image(state)
    assert evaluator.last_status.status == CacheStatus.CACHED


def test_line_cache_and_base_data_remains_unmodified():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    original = data.copy()
    document = (
        ArrayDocument(data)
        .with_operation(Crop(axis=1, start=1, stop=4))
        .with_operation(ReverseAxis(axis=0))
    )
    evaluator = OperationEvaluator(document)
    state = ViewState.from_shape(document.current_shape).with_line_axis(1).with_slice(0, 0)

    first = evaluator.line(state)
    second = evaluator.line(state)

    assert first is second
    assert evaluator.line_evaluations == 1
    np.testing.assert_array_equal(first.data, data[::-1, 1:4][0, :])
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "module_path",
    [
        ROOT / "arrayscope" / "operations" / "registry.py",
        ROOT / "arrayscope" / "operations" / "recipes.py",
        ROOT / "arrayscope" / "operations" / "evaluator.py",
    ],
)
def test_operation_support_modules_have_no_qt_or_pyqtgraph_imports(module_path):
    tree = ast.parse(module_path.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
