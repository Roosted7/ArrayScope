import numpy as np

from arrayscope.operations.capabilities import OperationCapabilities, OperationKind
from arrayscope.operations.cost import estimate_operation_cost, estimate_pipeline_cost, operation_output_dtype
from arrayscope.operations.pipeline import (
    CenteredFFT,
    CombineRealImagAxis,
    Crop,
    Mean,
    RootSumSquares,
    SplitComplexAxis,
)


def test_operation_output_dtype_matches_existing_estimates():
    assert operation_output_dtype(np.float32, Mean(axis=0)) == np.mean(np.empty((1,), dtype=np.float32)).dtype
    assert operation_output_dtype(np.float32, RootSumSquares(axis=0)) == np.dtype(np.float32)
    assert operation_output_dtype(np.float32, CenteredFFT(axis=0)) == np.dtype(np.complex64)
    assert operation_output_dtype(np.float32, CombineRealImagAxis(axis=0)) == np.dtype(np.complex64)
    assert operation_output_dtype(np.complex64, SplitComplexAxis(axis=0)) == np.dtype(np.float32)


def test_fft_cost_requires_full_axis_and_has_conservative_peak():
    cost = estimate_operation_cost((4, 8, 16), np.float32, CenteredFFT(axis=1))

    assert cost.kind == "transform"
    assert cost.requires_full_axis == (1,)
    assert cost.expands_request_axes == (1,)
    assert cost.cache_stage is True
    assert cost.estimated_peak_bytes >= cost.estimated_output_bytes * 3
    assert cost.can_chunk
    assert cost.blocking_axes == (1,)
    assert 1 not in cost.chunkable_axes


def test_reduction_cost_removes_axis_and_requires_full_axis():
    cost = estimate_operation_cost((4, 8, 16), np.float32, Mean(axis=1))

    assert cost.kind == "reduction"
    assert cost.output_shape == (4, 16)
    assert cost.requires_full_axis == (1,)


def test_rss_cost_accounts_for_abs_square_temporaries():
    cost = estimate_operation_cost((4, 8, 16), np.complex64, RootSumSquares(axis=0))

    assert cost.kind == "reduction"
    assert cost.output_dtype == np.dtype(np.float32)
    assert cost.temp_multiplier == 3.0
    assert cost.estimated_peak_bytes > cost.estimated_input_bytes


def test_complex_conversion_cost_estimates_output_dtype_and_peak():
    combine = estimate_operation_cost((2, 5), np.float32, CombineRealImagAxis(axis=0))
    split = estimate_operation_cost((1, 5), np.complex64, SplitComplexAxis(axis=0))

    assert combine.output_shape == (1, 5)
    assert combine.output_dtype == np.dtype(np.complex64)
    assert combine.estimated_peak_bytes == combine.estimated_input_bytes + combine.estimated_output_bytes
    assert split.output_shape == (2, 5)
    assert split.output_dtype == np.dtype(np.float32)


def test_pipeline_cost_tracks_peak_across_operations():
    cost = estimate_pipeline_cost((8, 16, 2), np.float32, (Crop(axis=1, start=2, stop=10), CenteredFFT(axis=0), RootSumSquares(axis=2)))
    fft_cost = cost.operation_costs[1]

    assert cost.output_shape == (8, 8)
    assert cost.output_dtype == np.dtype(np.float32)
    assert cost.estimated_peak_bytes >= fft_cost.estimated_peak_bytes


def test_cost_uses_operation_declared_dtype_and_capabilities():
    class CustomOperation:
        def output_shape(self, shape):
            return tuple(shape)

        def output_dtype(self, input_dtype):
            return np.dtype(np.float64)

        def capabilities(self, input_shape, input_dtype=None):
            return OperationCapabilities(
                kind=OperationKind.ELEMENTWISE,
                chunkable_axes=tuple(range(len(input_shape))),
                can_fuse=True,
            )

    cost = estimate_operation_cost((4, 5), np.float32, CustomOperation())

    assert cost.kind == "elementwise"
    assert cost.output_dtype == np.dtype(np.float64)
    assert cost.can_fuse is True


def test_crop_cost_reports_fusion_capability():
    cost = estimate_operation_cost((4, 8, 16), np.float32, Crop(axis=1, start=2, stop=6))

    assert cost.kind == "view"
    assert cost.can_fuse is True
