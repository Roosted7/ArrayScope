"""Canonical regions preserve NumPy negative-slice values and empty bounds."""

import numpy as np
import pytest

from arrayscope.operations.regions import (
    apply_region,
    apply_subregion,
    index_spec_from_region,
    region_contains,
    region_from_index_spec,
    region_nbytes,
    region_shape,
)


@pytest.mark.parametrize(
    ("size", "selection"),
    [
        (5, slice(None, None, -1)),
        (5, slice(3, 0, -1)),
        (7, slice(None, None, -2)),
        (7, slice(5, None, -2)),
        (5, slice(4, -100, -1)),
        (5, slice(100, -100, -3)),
        (5, slice(2, 2, -1)),
        (5, slice(4, -1, -1)),
        (5, slice(-100, None, -1)),
        (5, slice(-100, 2, -1)),
        (0, slice(None, None, -1)),
    ],
)
def test_negative_slice_canonical_values_shapes_bytes_and_keys_match_numpy(size, selection):
    source = np.arange(size, dtype=np.int16)
    region = region_from_index_spec(source.shape, (selection,))
    expected = source[selection]

    np.testing.assert_array_equal(apply_region(source, region), expected)
    assert region_shape(source.shape, region) == expected.shape
    assert region_nbytes(source.shape, source.dtype, region) == expected.nbytes
    assert region_from_index_spec(source.shape, index_spec_from_region(region)) == region


def test_negative_slice_composes_with_point_and_duplicate_cartesian_axes():
    source = np.arange(5 * 6 * 4).reshape(5, 6, 4)
    region = region_from_index_spec(source.shape, (slice(None, None, -2), (4, 1, 4), 2))
    expected = np.take(source[::-2, :, 2], [4, 1, 4], axis=1)

    np.testing.assert_array_equal(apply_region(source, region), expected)
    assert region_shape(source.shape, region) == (3, 3)
    assert region_nbytes(source.shape, source.dtype, region) == expected.nbytes


def test_negative_slices_on_multiple_axes_and_zero_extent_keep_numpy_semantics():
    for shape in ((5, 6, 4), (0, 6, 4)):
        source = np.arange(np.prod(shape)).reshape(shape)
        spec = (slice(None, None, -1), slice(4, None, -2), slice(None, None, -1))
        region = region_from_index_spec(shape, spec)

        np.testing.assert_array_equal(apply_region(source, region), source[spec])
        assert region_shape(shape, region) == source[spec].shape


def test_reversed_source_region_contains_and_extracts_a_reversed_subregion():
    source = np.arange(7)
    complete = region_from_index_spec(source.shape, (slice(None, None, -1),))
    selected = region_from_index_spec(source.shape, (slice(5, None, -2),))

    assert region_contains(complete, selected, source.shape)
    np.testing.assert_array_equal(
        apply_subregion(
            source[::-1], source_region=complete, target_region=selected, shape=source.shape
        ),
        source[5::-2],
    )
