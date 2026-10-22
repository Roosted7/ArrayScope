"""Coordinate maps must price tiny requests without full-axis index arrays."""

from math import prod

import numpy as np
import pytest

from arrayscope.core.array_source import read_index_spec
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, Crop, ReverseAxis, SplitAxis
from arrayscope.operations.planner import plan_output_region
from arrayscope.operations.regions import index_spec_from_region, region_from_index_spec


@pytest.mark.parametrize("operation", [ReverseAxis(0), Crop(0, 2, 99_999_998), SplitAxis(0, 2)])
def test_hundred_million_axis_plans_allocate_only_requested_coordinate_metadata(
    monkeypatch, operation
):
    class MetadataSource:
        shape = (100_000_000, 2)
        dtype = np.dtype("float32")
        nbytes = prod(shape) * dtype.itemsize

        def read_region(self, *_args, **_kwargs):
            raise AssertionError("metadata plan read payload")

    real_arange = np.arange
    allocations = []

    def bounded_arange(*args, **kwargs):
        requested = range(*[int(value) for value in args])
        allocations.append(len(requested))
        assert len(requested) <= 64, "planner enumerated an entire source axis"
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(np, "arange", bounded_arange)
    document = ArrayDocument(MetadataSource(), operations=(operation,))
    shape = document.current_shape
    region = region_from_index_spec(shape, (slice(0, 4), *((slice(None),) * (len(shape) - 1))))
    prepared = plan_output_region(document, region)
    assert prepared.estimated_peak_bytes is not None
    assert max(allocations, default=0) <= 64


@pytest.mark.parametrize(
    "item", [slice(None), slice(1, 11, 3), slice(11, None, -2), slice(7, 3, 1), [9, 1, 9, 0], 3]
)
@pytest.mark.parametrize("operation", [ReverseAxis(0), Crop(0, 2, 15)])
def test_affine_coordinate_maps_keep_numpy_order_duplicates_and_negative_slices(item, operation):
    values = np.arange(19)
    derived = operation.apply(values)
    region = region_from_index_spec(derived.shape, (item,))
    required = operation.required_input_region(values.shape, region)
    actual = read_index_spec(values, index_spec_from_region(required))
    np.testing.assert_array_equal(actual, derived[item])


def test_reversing_a_full_region_is_metadata_only_even_before_oversized_fft(monkeypatch):
    class MetadataSource:
        shape = (100_000_000, 2)
        dtype = np.dtype("float32")
        nbytes = prod(shape) * dtype.itemsize

        def read_region(self, *_args, **_kwargs):
            raise AssertionError("metadata plan read payload")

    def no_coordinate_array(*_args, **_kwargs):
        raise AssertionError("full-axis affine coordinate map allocated an index vector")

    document = ArrayDocument(MetadataSource(), operations=(ReverseAxis(0), CenteredFFT(0)))
    monkeypatch.setattr(np, "arange", no_coordinate_array)
    region = region_from_index_spec(document.current_shape, (slice(0, 1), slice(None)))
    prepared = plan_output_region(document, region)
    assert prepared.estimated_peak_bytes > 64 * 1024**2
