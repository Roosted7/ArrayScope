import numpy as np

from arrayscope.operations.optimizer import CastDType, optimize_operations
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CenteredIFFT,
    Conjugate,
    Crop,
    ReverseAxis,
    evaluate,
)


def _types(operations):
    return tuple(type(operation).__name__ for operation in operations)


def test_fft_ifft_pair_becomes_dtype_preserving_cast_for_real_inputs():
    plan = optimize_operations((2, 3, 4), np.float32, (CenteredFFT(axis=2), CenteredIFFT(axis=2)))

    assert _types(plan.operations) == ("CastDType",)
    assert plan.operations[0].dtype == "complex64"
    assert plan.output_shape == (2, 3, 4)
    assert np.dtype(plan.output_dtype) == np.dtype(np.complex64)


def test_ifft_fft_pair_simplifies_too_and_preserves_complex_dtype():
    plan = optimize_operations((2, 3, 4), np.complex64, (CenteredIFFT(axis=2), CenteredFFT(axis=2)))

    assert plan.operations == ()
    assert np.dtype(plan.output_dtype) == np.dtype(np.complex64)
    assert plan.steps[0].kind == "fft_ifft_cancel"


def test_fft_ifft_dtype_policy_for_float64():
    plan = optimize_operations((2, 3, 4), np.float64, (CenteredFFT(axis=2), CenteredIFFT(axis=2)))

    assert _types(plan.operations) == ("CastDType",)
    assert plan.operations[0].dtype == "complex128"
    assert np.dtype(plan.output_dtype) == np.dtype(np.complex128)


def test_optimized_fft_ifft_materialized_value_uses_dtype_preserving_identity_policy():
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    result = evaluate(data, (CenteredFFT(axis=2), CenteredIFFT(axis=2)))

    assert result.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(result.real, data)
    np.testing.assert_array_equal(result.imag, np.zeros_like(data))


def test_reverse_and_conjugate_pairs_are_removed():
    reverse = optimize_operations((2, 3, 4), np.float32, (ReverseAxis(axis=1), ReverseAxis(axis=1)))
    conjugate = optimize_operations((2, 3, 4), np.complex64, (Conjugate(), Conjugate()))

    assert reverse.operations == ()
    assert conjugate.operations == ()


def test_adjacent_same_axis_crops_compose():
    plan = optimize_operations((4, 10, 6), np.float32, (Crop(axis=1, start=2, stop=9), Crop(axis=1, start=3, stop=5)))

    assert _types(plan.operations) == ("Crop",)
    assert plan.operations[0].axis == 1
    assert plan.operations[0].start == 5
    assert plan.operations[0].stop == 7
    assert plan.output_shape == (4, 2, 6)


def test_adjacent_casts_coalesce():
    plan = optimize_operations((2, 3), np.float32, (CastDType("complex64"), CastDType("complex128")))

    assert plan.operations == (CastDType("complex128"),)
    assert np.dtype(plan.output_dtype) == np.dtype(np.complex128)


def test_different_axis_or_non_adjacent_fft_ifft_do_not_simplify():
    different_axis = optimize_operations((2, 3, 4), np.float32, (CenteredFFT(axis=1), CenteredIFFT(axis=2)))
    separated = optimize_operations(
        (2, 3, 4),
        np.float32,
        (CenteredFFT(axis=2), ReverseAxis(axis=0), CenteredIFFT(axis=2)),
    )

    assert _types(different_axis.operations) == ("CenteredFFT", "CenteredIFFT")
    assert _types(separated.operations) == ("CenteredFFT", "ReverseAxis", "CenteredIFFT")


def test_optimization_does_not_mutate_user_steps_or_recipe_operations():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2), CenteredIFFT(axis=2)))
    original_steps = document.steps

    plan = optimize_operations(data.shape, data.dtype, document.enabled_operations)

    assert _types(plan.operations) == ("CastDType",)
    assert document.steps == original_steps
    assert _types(document.enabled_operations) == ("CenteredFFT", "CenteredIFFT")
