import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


DIM_OPS_PATH = Path(__file__).parents[1] / "arrayscope" / "dim_ops.py"
SPEC = importlib.util.spec_from_file_location("arrayscope_dim_ops_for_test", DIM_OPS_PATH)
dim_ops = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dim_ops
SPEC.loader.exec_module(dim_ops)


def test_centered_fft_and_ifft_match_existing_viewer_transform():
    data = np.arange(12).reshape(3, 4).astype(float)

    transformed = dim_ops.centered_fft(data, axis=1)
    restored = dim_ops.centered_ifft(transformed, axis=1)

    expected = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(data), axis=1, norm="ortho"))
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(restored, data)
    assert transformed.shape == data.shape


def test_fftshift_round_trip_preserves_shape_and_values():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    shifted = dim_ops.apply_fftshift(data, axis=2)
    restored = dim_ops.undo_fftshift(shifted, axis=2)

    np.testing.assert_array_equal(shifted, np.fft.fftshift(data, axes=2))
    np.testing.assert_array_equal(restored, data)
    assert shifted.shape == data.shape


def test_combine_real_imag_axis_keeps_singleton_axis():
    data = np.array(
        [
            [[1, 10], [2, 20], [3, 30]],
            [[4, 40], [5, 50], [6, 60]],
        ],
        dtype=float,
    )

    combined = dim_ops.combine_real_imag_axis(data, axis=2)

    assert combined.shape == (2, 3, 1)
    expected = np.array([[1 + 10j, 2 + 20j, 3 + 30j], [4 + 40j, 5 + 50j, 6 + 60j]])
    np.testing.assert_array_equal(combined[:, :, 0], expected)


def test_split_complex_axis_restores_size_2_real_imag_axis():
    data = np.array([[[1 + 10j], [2 + 20j]], [[3 + 30j], [4 + 40j]]])

    split = dim_ops.split_complex_axis(data, axis=2)

    assert split.shape == (2, 2, 2)
    np.testing.assert_array_equal(split[:, :, 0], np.real(data[:, :, 0]))
    np.testing.assert_array_equal(split[:, :, 1], np.imag(data[:, :, 0]))


def test_combine_and_split_reject_invalid_shapes():
    with pytest.raises(ValueError, match="size 2"):
        dim_ops.combine_real_imag_axis(np.zeros((3, 4)), axis=0)

    with pytest.raises(ValueError, match="already complex"):
        dim_ops.combine_real_imag_axis(np.zeros((2, 4), dtype=complex), axis=0)

    with pytest.raises(ValueError, match="must be complex"):
        dim_ops.split_complex_axis(np.zeros((1, 4)), axis=0)

    with pytest.raises(ValueError, match="size 1"):
        dim_ops.split_complex_axis(np.zeros((2, 4), dtype=complex), axis=0)


def test_dim_ops_module_has_no_qt_or_pyqtgraph_imports():
    tree = ast.parse(DIM_OPS_PATH.read_text())

    imported_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.append(node.module.split(".")[0])

    assert "pyqtgraph" not in imported_roots
    assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)
