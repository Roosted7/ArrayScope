"""Value, shape, dtype, region, and axis-metadata truth for native structure ops."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from arrayscope.core.axis_info import AxisInfo
from arrayscope.operations.capabilities import OperationKind
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CumulativeSum,
    Difference,
    Gradient,
    Pad,
    Resample,
    Roll,
    Smooth,
    Squeeze,
    Transpose,
)
from arrayscope.operations.regions import region_from_index_spec


@pytest.fixture
def complex_data():
    real = np.arange(4 * 6 * 5, dtype=np.float32).reshape(4, 6, 5)
    return (real + 1j * np.flip(real, axis=1)).astype(np.complex64)


@pytest.mark.parametrize("amount", [-7, -1, 0, 2, 13])
def test_roll_matches_numpy_for_signed_amounts(complex_data, amount):
    operation = Roll(axis=1, amount=amount)

    np.testing.assert_array_equal(
        operation.apply(complex_data), np.roll(complex_data, amount, axis=1)
    )
    assert operation.output_shape(complex_data.shape) == complex_data.shape
    assert operation.output_dtype(complex_data.dtype) == complex_data.dtype


@pytest.mark.parametrize(
    ("mode", "numpy_mode"),
    [(0, "constant"), (1, "edge"), (2, "reflect")],
)
def test_pad_supports_asymmetric_zero_edge_and_reflect(complex_data, mode, numpy_mode):
    operation = Pad(axis=1, before=2, after=3, mode=mode)
    widths = ((0, 0), (2, 3), (0, 0))

    result = operation.apply(complex_data)

    np.testing.assert_array_equal(result, np.pad(complex_data, widths, mode=numpy_mode))
    assert result.shape == operation.output_shape(complex_data.shape) == (4, 11, 5)
    assert result.dtype == complex_data.dtype


@pytest.mark.parametrize("boundary", ["reflect", "edge", "zero"])
def test_smooth_matches_scipy_for_each_named_boundary(complex_data, boundary):
    """The enum names are scipy's on purpose, so scipy is the oracle."""

    operation = Smooth(axis=1, width=3, boundary=boundary, normalize_edges=False)
    scipy_mode = {"reflect": "mirror", "edge": "nearest", "zero": "constant"}[boundary]

    result = operation.apply(complex_data)

    reference = ndimage.uniform_filter1d(
        complex_data.astype(np.complex128), size=3, axis=1, mode=scipy_mode, cval=0.0
    )
    np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-4)
    assert result.shape == operation.output_shape(complex_data.shape) == complex_data.shape
    assert result.dtype == operation.output_dtype(complex_data.dtype) == complex_data.dtype


def test_smooth_normalize_edges_only_changes_the_zero_boundary_ends():
    data = np.ones((5,), dtype=np.float32)

    faded = Smooth(axis=0, width=3, boundary="zero", normalize_edges=False).apply(data)
    kept = Smooth(axis=0, width=3, boundary="zero", normalize_edges=True).apply(data)

    # Without normalization the window averages in the two zeros it padded with.
    np.testing.assert_allclose(faded, [2 / 3, 1, 1, 1, 2 / 3], rtol=1e-6)
    np.testing.assert_allclose(kept, np.ones(5), rtol=1e-6)

    # With a boundary that supplies real samples the flag is inert.
    for boundary in ("reflect", "edge"):
        with_flag = Smooth(axis=0, width=3, boundary=boundary, normalize_edges=True).apply(data)
        without = Smooth(axis=0, width=3, boundary=boundary, normalize_edges=False).apply(data)
        np.testing.assert_array_equal(with_flag, without)


def test_smooth_width_one_is_the_identity_and_integers_widen_to_float32():
    data = np.arange(6, dtype=np.int16)
    operation = Smooth(axis=0, width=1, boundary="reflect", normalize_edges=True)

    result = operation.apply(data)

    np.testing.assert_array_equal(result, data)
    assert result.dtype == operation.output_dtype(data.dtype) == np.dtype(np.float32)


def test_smooth_refuses_a_boundary_it_does_not_know():
    with pytest.raises(ValueError, match="reflect, edge, zero"):
        Smooth(axis=0, width=3, boundary="wrap", normalize_edges=True).apply(np.ones(4))


@pytest.mark.parametrize(("factor", "target"), [(0.6, 4), (1.0, 6), (1.5, 9), (2.25, 14)])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_fractional_resample_has_exact_shape_dtype_and_scipy_values(
    complex_data, factor, target, order
):
    operation = Resample(axis=1, factor=factor, order=order, mode=2)
    zoom = (1.0, target / complex_data.shape[1], 1.0)
    reference = ndimage.zoom(
        complex_data,
        zoom,
        order=order,
        mode="reflect",
        prefilter=order > 1,
        grid_mode=False,
    )

    result = operation.apply(complex_data)

    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)
    assert result.shape == operation.output_shape(complex_data.shape) == (4, target, 5)
    assert result.dtype == operation.output_dtype(complex_data.dtype) == complex_data.dtype


def test_transpose_swaps_arbitrary_axes(complex_data):
    operation = Transpose(axis=0, other_axis=2)

    result = operation.apply(complex_data)

    np.testing.assert_array_equal(result, np.swapaxes(complex_data, 0, 2))
    assert result.shape == operation.output_shape(complex_data.shape) == (5, 6, 4)
    assert result.dtype == complex_data.dtype


def test_squeeze_removes_exactly_one_selected_singleton_axis():
    data = np.arange(20, dtype=np.float32).reshape(4, 1, 5)
    operation = Squeeze(axis=1)

    np.testing.assert_array_equal(operation.apply(data), np.squeeze(data, axis=1))
    assert operation.output_shape(data.shape) == (4, 5)
    with pytest.raises(ValueError, match="size 1"):
        Squeeze(axis=0).apply(data)


@pytest.mark.parametrize(
    ("operation", "reference"),
    [
        (Difference(axis=1), lambda data: np.diff(data, axis=1)),
        (Gradient(axis=1), lambda data: np.gradient(data, axis=1, edge_order=1)),
        (
            CumulativeSum(axis=1),
            lambda data: np.cumsum(data, axis=1, dtype=data.dtype),
        ),
    ],
)
def test_difference_gradient_and_cumulative_sum_match_numpy(complex_data, operation, reference):
    result = operation.apply(complex_data)

    np.testing.assert_allclose(result, reference(complex_data), rtol=1e-6, atol=1e-6)
    assert result.shape == operation.output_shape(complex_data.shape)
    assert result.dtype == operation.output_dtype(complex_data.dtype) == complex_data.dtype


@pytest.mark.parametrize(
    "operation",
    [
        Roll(axis=1, amount=2),
        Pad(axis=1, before=1, after=2, mode=0),
        Resample(axis=1, factor=1.5, order=1, mode=2),
        Smooth(axis=1, width=3, boundary="reflect", normalize_edges=True),
        Difference(axis=1),
        Gradient(axis=1),
        CumulativeSum(axis=1),
    ],
)
def test_blocking_axis_region_paths_match_whole_array_result(complex_data, operation):
    output_shape = operation.output_shape(complex_data.shape)
    output_region = region_from_index_spec(
        output_shape, (slice(1, 3), slice(1, output_shape[1] - 1), 2)
    )
    input_region = operation.required_input_region(complex_data.shape, output_region)
    input_data = complex_data[1:3, :, 2]

    region_result = operation.apply_to_region(
        input_data, input_region=input_region, output_region=output_region
    )
    expected = operation.apply(complex_data)[1:3, 1 : output_shape[1] - 1, 2]

    np.testing.assert_allclose(region_result, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "operation",
    [
        Roll(axis=1, amount=2),
        Pad(axis=1, before=1, after=2, mode=0),
        Resample(axis=1, factor=1.5, order=1, mode=2),
        Transpose(axis=1, other_axis=2),
        Difference(axis=1),
        Gradient(axis=1),
        CumulativeSum(axis=1),
    ],
)
def test_nonpointwise_structure_ops_expand_every_axis_they_mix(operation):
    capabilities = operation.capabilities((4, 6, 5), np.complex64)

    assert capabilities.kind in {OperationKind.TRANSFORM, OperationKind.RESHAPE}
    assert set(capabilities.blocking_axes) == set(capabilities.expands_request_axes)
    assert capabilities.blocking_axes


def test_axis_metadata_tracks_pad_resample_difference_transpose_and_squeeze():
    data = np.zeros((4, 6, 1), dtype=np.float32)
    axes = (
        AxisInfo("x", "X", 4, unit="mm", spacing=2.0, origin=10.0),
        AxisInfo("y", "Y", 6, unit="mm", spacing=3.0, origin=-5.0),
        AxisInfo("singleton", "Singleton", 1),
    )
    document = ArrayDocument(
        data,
        operations=(
            Pad(axis=0, before=2, after=1, mode=0),
            Resample(axis=1, factor=0.5, order=1, mode=2),
            Difference(axis=0),
            Transpose(axis=0, other_axis=1),
            Squeeze(axis=2),
        ),
        axes=axes,
    )

    assert document.shape == (3, 6)
    assert tuple(axis.id for axis in document.current_axes) == ("y", "x")
    assert tuple(axis.size for axis in document.current_axes) == document.shape
    assert document.current_axes[0].spacing == pytest.approx(7.5)
    assert document.current_axes[1].origin == pytest.approx(7.0)


@pytest.mark.parametrize(
    "operation",
    [
        Pad(axis=0, before=-1, after=0, mode=0),
        Pad(axis=0, before=1, after=1, mode=7),
        Resample(axis=0, factor=0.0, order=1, mode=2),
        Resample(axis=0, factor=1.0, order=4, mode=2),
        Resample(axis=0, factor=1.0, order=1, mode=7),
    ],
)
def test_invalid_structure_parameters_fail_loudly(operation):
    with pytest.raises(ValueError):
        operation.apply(np.ones((4, 5), dtype=np.float32))
