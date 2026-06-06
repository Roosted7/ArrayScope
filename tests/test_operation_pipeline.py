import ast
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


dim_ops = load_module("dim_ops")
operation_pipeline = load_module("operation_pipeline")

ArrayDocument = operation_pipeline.ArrayDocument
CenteredFFT = operation_pipeline.CenteredFFT
CenteredIFFT = operation_pipeline.CenteredIFFT
CombineRealImagAxis = operation_pipeline.CombineRealImagAxis
Conjugate = operation_pipeline.Conjugate
Crop = operation_pipeline.Crop
FFTShift = operation_pipeline.FFTShift
Mean = operation_pipeline.Mean
ReverseAxis = operation_pipeline.ReverseAxis
RootSumSquares = operation_pipeline.RootSumSquares
SplitComplexAxis = operation_pipeline.SplitComplexAxis


def test_crop_reverse_and_conjugate_preserve_base_data_reference_and_values():
    data = (np.arange(3 * 4).reshape(3, 4) + 1j).astype(complex)
    document = (
        ArrayDocument(data)
        .with_operation(Crop(axis=1, start=1, stop=4))
        .with_operation(ReverseAxis(axis=0))
        .with_operation(Conjugate())
    )

    result = document.materialize()

    assert document.base_data is data
    assert document.current_shape == (3, 3)
    expected = np.conjugate(np.flip(data[:, 1:4], axis=0))
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(data, (np.arange(3 * 4).reshape(3, 4) + 1j).astype(complex))


def test_mean_and_root_sum_squares_remove_axes_predictably():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(float)
    document = ArrayDocument(data).with_operation(Mean(axis=1)).with_operation(RootSumSquares(axis=1))

    result = document.materialize()

    assert document.current_shape == (2,)
    expected = np.sqrt(np.sum(np.abs(np.mean(data, axis=1)) ** 2, axis=1))
    np.testing.assert_allclose(result, expected)


def test_centered_fft_and_ifft_operations_wrap_existing_dim_ops():
    data = np.arange(12).reshape(3, 4).astype(float)
    document = ArrayDocument(data).with_operation(CenteredFFT(axis=1)).with_operation(CenteredIFFT(axis=1))

    result = document.materialize()

    assert document.current_shape == data.shape
    np.testing.assert_allclose(result, data)
    np.testing.assert_allclose(
        CenteredFFT(axis=1).apply(data),
        dim_ops.centered_fft(data, axis=1),
    )


def test_fftshift_operation_wraps_existing_dim_ops():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    document = ArrayDocument(data).with_operation(FFTShift(axis=2))

    result = document.materialize()

    assert document.current_shape == data.shape
    np.testing.assert_array_equal(result, dim_ops.apply_fftshift(data, axis=2))


def test_real_complex_axis_operations_wrap_existing_dim_ops():
    data = np.array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], dtype=float)
    combined_document = ArrayDocument(data).with_operation(CombineRealImagAxis(axis=2))
    combined = combined_document.materialize()

    assert combined_document.current_shape == (2, 2, 1)
    np.testing.assert_array_equal(combined, dim_ops.combine_real_imag_axis(data, axis=2))

    split_document = combined_document.with_operation(SplitComplexAxis(axis=2))
    assert split_document.current_shape == data.shape
    np.testing.assert_array_equal(split_document.materialize(), data)


def test_document_is_undoable_by_removing_operations():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    document = ArrayDocument(data).with_operation(Crop(axis=2, start=1, stop=4)).with_operation(Mean(axis=0))

    undone = document.without_last_operation()

    assert document.current_shape == (3, 3)
    assert undone.current_shape == (2, 3, 3)
    assert undone.operations == (Crop(axis=2, start=1, stop=4),)
    np.testing.assert_array_equal(undone.materialize(), data[:, :, 1:4])


def test_axis_and_crop_validation_use_current_derived_shape():
    document = ArrayDocument(np.zeros((2, 3))).with_operation(Mean(axis=0))

    with pytest.raises(ValueError, match="out of bounds"):
        document.with_operation(ReverseAxis(axis=1))

    with pytest.raises(ValueError, match="crop bounds"):
        ArrayDocument(np.zeros((2, 3))).with_operation(Crop(axis=1, start=2, stop=4))


def test_operation_pipeline_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse((ROOT / "arrayscope" / "operations" / "pipeline.py").read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
