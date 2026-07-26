"""Value and planning truth for native normalization and reductions."""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations.capabilities import OperationKind
from arrayscope.operations.pipeline import (
    Median,
    Normalize,
    Percentile,
    StandardDeviation,
    Variance,
)
from arrayscope.operations.regions import region_from_index_spec


@pytest.fixture
def complex_data():
    real = np.arange(3 * 5 * 4, dtype=np.float32).reshape(3, 5, 4) - 10
    imag = np.flip(real, axis=1) * np.float32(0.25)
    return (real + 1j * imag).astype(np.complex64)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (StandardDeviation(axis=1), lambda data: np.std(data, axis=1, ddof=1)),
        (Variance(axis=1), lambda data: np.var(data, axis=1, ddof=1)),
        (Median(axis=1), lambda data: np.median(data, axis=1)),
        (
            Percentile(axis=1, q=30.0),
            lambda data: (
                np.percentile(data.real, 30.0, axis=1) + 1j * np.percentile(data.imag, 30.0, axis=1)
            ).astype(np.complex64),
        ),
    ],
)
def test_reductions_match_numpy_references_on_complex64(complex_data, operation, reference):
    result = operation.apply(complex_data)

    np.testing.assert_allclose(result, reference(complex_data), rtol=1e-6, atol=1e-6)
    assert result.shape == operation.output_shape(complex_data.shape) == (3, 4)
    assert result.dtype == operation.output_dtype(complex_data.dtype)


@pytest.mark.parametrize("operation", [StandardDeviation(axis=1), Variance(axis=1)])
def test_std_and_variance_use_sample_ddof_one(operation):
    data = np.array([[1.0, 3.0], [4.0, 8.0]], dtype=np.float32)
    result = operation.apply(data)
    reference_fn = np.std if isinstance(operation, StandardDeviation) else np.var

    np.testing.assert_allclose(result, reference_fn(data, axis=1, ddof=1))
    assert not np.allclose(result, reference_fn(data, axis=1, ddof=0))
    assert result.dtype == np.dtype(np.float32)


def test_percentile_float32_does_not_inherit_numpy_float64_promotion():
    data = np.arange(24, dtype=np.float32).reshape(4, 6)
    operation = Percentile(axis=1, q=25.0)

    result = operation.apply(data)

    np.testing.assert_allclose(result, np.percentile(data, 25.0, axis=1))
    assert result.dtype == operation.output_dtype(data.dtype) == np.dtype(np.float32)


def test_normalize_is_per_axis_l2_and_leaves_zero_lines_zero(complex_data):
    data = complex_data.copy()
    data[0] = 0
    operation = Normalize(axis=1)
    norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=1, keepdims=True))
    reference = np.zeros_like(data)
    np.divide(data, norm, out=reference, where=norm != 0)

    result = operation.apply(data)

    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(result[0], 0)
    assert result.dtype == data.dtype


def test_normalize_expands_selected_axis_for_region_evaluation(complex_data):
    operation = Normalize(axis=1)
    output_region = region_from_index_spec(complex_data.shape, (slice(1, 3), slice(2, 4), 1))
    input_region = operation.required_input_region(complex_data.shape, output_region)
    input_data = complex_data[1:3, :, 1]

    region_result = operation.apply_to_region(
        input_data, input_region=input_region, output_region=output_region
    )
    expected = operation.apply(complex_data)[1:3, 2:4, 1]

    np.testing.assert_allclose(region_result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "operation",
    [
        StandardDeviation(axis=1),
        Variance(axis=1),
        Median(axis=1),
        Percentile(axis=1, q=75),
    ],
)
def test_native_reductions_are_opaque_over_the_reduced_axis(operation):
    capabilities = operation.capabilities((4, 8, 12), np.complex64)

    assert capabilities.kind is OperationKind.REDUCTION
    assert capabilities.blocking_axes == (1,)
    assert capabilities.expands_request_axes == (1,)
    assert 1 not in capabilities.chunkable_axes


def test_invalid_percentile_fails_loudly():
    with pytest.raises(ValueError, match="between 0 and 100"):
        Percentile(axis=0, q=101).apply(np.ones((3, 4), dtype=np.float32))
