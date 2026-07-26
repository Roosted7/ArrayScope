"""Value, dtype, and region truth for the native pointwise toolbox."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from arrayscope.operations.pipeline import (
    Clip,
    HardThreshold,
    ImaginaryPart,
    LogMagnitude,
    Magnitude,
    Offset,
    Phase,
    Power,
    RealPart,
    Scale,
    SoftThreshold,
)
from arrayscope.operations.plugin_conformance import verify_region_conformance


@dataclass(frozen=True)
class _OperationSpec:
    id: str
    operation: object

    def resolve_fn(self, axis, params):
        del axis, params
        return self.operation.apply


@pytest.fixture
def complex_data():
    return np.array(
        [[0, 1 + 2j, -3 + 4j], [0.25 - 0.5j, -2j, 5 - 1j]],
        dtype=np.complex64,
    )


def test_complex_display_maps_match_numpy_and_declare_real_dtype(complex_data):
    operations_and_references = [
        (Magnitude(), np.abs(complex_data)),
        (Phase(), np.angle(complex_data)),
        (RealPart(), np.real(complex_data)),
        (ImaginaryPart(), np.imag(complex_data)),
        (
            LogMagnitude(epsilon=1e-3),
            np.log(np.maximum(np.abs(complex_data), np.float32(1e-3))),
        ),
    ]

    for operation, reference in operations_and_references:
        result = operation.apply(complex_data)
        np.testing.assert_allclose(result, reference)
        assert result.dtype == np.dtype(np.float32)
        assert operation.output_dtype(np.complex64) == np.dtype(np.float32)


def test_scalar_maps_match_numpy_without_promoting_float32():
    data = np.array([[-2.0, -0.25, 0.0, 0.5, 3.0]], dtype=np.float32)
    operations_and_references = [
        (Scale(factor=2.5), data * np.float32(2.5)),
        (Offset(value=-0.75), data + np.float32(-0.75)),
        (Power(exponent=2.0), np.power(data, np.float32(2.0))),
        (Clip(minimum=-1.0, maximum=1.0), np.clip(data, -1.0, 1.0)),
    ]

    for operation, reference in operations_and_references:
        result = operation.apply(data)
        np.testing.assert_allclose(result, reference)
        assert result.dtype == data.dtype
        assert operation.output_dtype(data.dtype) == data.dtype


def test_complex_clip_is_componentwise_and_dtype_preserving(complex_data):
    result = Clip(minimum=-1.0, maximum=2.0).apply(complex_data)
    reference = np.clip(complex_data.real, -1.0, 2.0) + 1j * np.clip(complex_data.imag, -1.0, 2.0)

    np.testing.assert_array_equal(result, reference.astype(np.complex64))
    assert result.dtype == np.dtype(np.complex64)


def test_native_thresholds_match_magnitude_references(complex_data):
    threshold = 1.5
    magnitude = np.abs(complex_data)
    scale = np.zeros_like(magnitude)
    np.divide(
        np.maximum(magnitude - threshold, 0),
        magnitude,
        out=scale,
        where=magnitude != 0,
    )

    soft = SoftThreshold(threshold=threshold).apply(complex_data)
    hard = HardThreshold(threshold=threshold).apply(complex_data)

    np.testing.assert_allclose(soft, complex_data * scale)
    np.testing.assert_array_equal(
        hard, np.where(magnitude >= threshold, complex_data, np.complex64(0))
    )
    assert soft.dtype == hard.dtype == complex_data.dtype


@pytest.mark.parametrize(
    "operation",
    [
        Magnitude(),
        Phase(),
        RealPart(),
        ImaginaryPart(),
        LogMagnitude(epsilon=1e-6),
        Scale(factor=1.25),
        Offset(value=-0.5),
        Power(exponent=2.0),
        Clip(minimum=-1.0, maximum=2.0),
        SoftThreshold(threshold=0.3),
        HardThreshold(threshold=0.3),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_elementwise_claim_passes_property_region_conformance(operation, dtype):
    result = verify_region_conformance(
        _OperationSpec(type(operation).__name__, operation),
        (7, 8, 9),
        dtype,
        rng=np.random.default_rng(20260726),
        samples=18,
    )

    assert result.honored, result.reason


@pytest.mark.parametrize(
    "operation",
    [
        LogMagnitude(epsilon=0.0),
        SoftThreshold(threshold=-0.1),
        HardThreshold(threshold=-0.1),
        Clip(minimum=2.0, maximum=1.0),
    ],
)
def test_invalid_pointwise_parameters_fail_loudly(operation):
    with pytest.raises(ValueError):
        operation.apply(np.ones((2, 3), dtype=np.float32))
