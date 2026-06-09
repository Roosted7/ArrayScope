import numpy as np
import pytest

from arrayscope.operations.capabilities import OperationKind
from arrayscope.operations.pipeline import (
    CenteredFFT,
    CenteredIFFT,
    CombineRealImagAxis,
    Conjugate,
    Crop,
    FFTShift,
    Maximum,
    Mean,
    Minimum,
    ReverseAxis,
    RootSumSquares,
    SplitComplexAxis,
    Sum,
)
from arrayscope.operations.registry import operation_entries


def test_every_registered_operation_declares_dtype_and_capabilities():
    shape = (4, 5, 6)
    dtype = np.dtype(np.float32)
    samples = {
        "crop": Crop(axis=1, start=1, stop=4),
        "reverse": ReverseAxis(axis=1),
        "conjugate": Conjugate(),
        "mean": Mean(axis=1),
        "rss": RootSumSquares(axis=1),
        "sum": Sum(axis=1),
        "max": Maximum(axis=1),
        "min": Minimum(axis=1),
        "centered_fft": CenteredFFT(axis=1),
        "centered_ifft": CenteredIFFT(axis=1),
        "fftshift": FFTShift(axis=1),
        "combine_real_imag": CombineRealImagAxis(axis=2),
        "split_complex": SplitComplexAxis(axis=2),
    }

    for entry in operation_entries():
        operation = samples[entry.id]
        op_shape = (4, 5, 2) if entry.id == "combine_real_imag" else (4, 5, 1) if entry.id == "split_complex" else shape
        op_dtype = np.dtype(np.complex64) if entry.id == "split_complex" else dtype
        assert operation.output_dtype(op_dtype) is None or isinstance(np.dtype(operation.output_dtype(op_dtype)), np.dtype)
        capabilities = operation.capabilities(op_shape, op_dtype)
        assert isinstance(capabilities.kind, OperationKind)


def test_fft_declares_blocking_expanded_axis_and_stage_cache():
    capabilities = CenteredFFT(axis=1).capabilities((4, 8, 16), np.float32)

    assert capabilities.kind == OperationKind.TRANSFORM
    assert capabilities.blocking_axes == (1,)
    assert capabilities.expands_request_axes == (1,)
    assert capabilities.cache_stage is True
    assert 1 not in capabilities.chunkable_axes
    assert CenteredIFFT(axis=1).capabilities((4, 8, 16), np.float32).cache_stage is True


@pytest.mark.parametrize("operation", [Mean(axis=1), Sum(axis=1), Maximum(axis=1), Minimum(axis=1), RootSumSquares(axis=1)])
def test_reductions_declare_blocking_and_expanded_axis(operation):
    capabilities = operation.capabilities((4, 8, 16), np.float32)

    assert capabilities.kind == OperationKind.REDUCTION
    assert capabilities.blocking_axes == (1,)
    assert capabilities.expands_request_axes == (1,)


@pytest.mark.parametrize("operation", [Crop(axis=1, start=1, stop=4), ReverseAxis(axis=1), FFTShift(axis=1)])
def test_view_operations_are_fusible_without_blocking_axes(operation):
    capabilities = operation.capabilities((4, 8, 16), np.float32)

    assert capabilities.kind == OperationKind.VIEW
    assert capabilities.blocking_axes == ()
    assert capabilities.cache_stage is False
    assert capabilities.can_fuse is True


def test_complex_conversion_dtype_declarations_match_current_behavior():
    assert CombineRealImagAxis(axis=0).output_dtype(np.float32) == np.dtype(np.complex64)
    assert SplitComplexAxis(axis=0).output_dtype(np.complex64) == np.dtype(np.float32)


def test_capabilities_validate_axes_through_existing_shape_rules():
    with pytest.raises(Exception):
        CenteredFFT(axis=4).capabilities((4, 8, 16), np.float32)
