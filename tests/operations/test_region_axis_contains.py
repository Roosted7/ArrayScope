"""`_axis_contains` answers membership arithmetically; semantics must not move.

The stage-plan match loop asks region containment per tile per cache entry
inside one GUI callback, so `_axis_contains` was rewritten from materialized
numpy index arrays (~240 µs per `region_contains` call) to range/set
arithmetic.  These tests pin the rewrite to the exact semantics of the
original vectorized implementation, including its edge cases (empty
progressions, negative steps, stride alignment, and index normalization).
"""

import numpy as np
import pytest

from arrayscope.operations.regions import (
    AxisRegion,
    AxisRegionKind,
    RegionSpec,
    _axis_contains,
    _axis_indices,
    region_contains,
)


def _reference_axis_contains(container: AxisRegion, contained: AxisRegion, size: int) -> bool:
    """The original numpy implementation, kept verbatim as the oracle."""

    container_indices = _axis_indices(container, size)
    contained_indices = _axis_indices(contained, size)
    if isinstance(container_indices, int):
        if isinstance(contained_indices, int):
            return int(container_indices) == int(contained_indices)
        contained_values = tuple(
            int(index) for index in np.asarray(contained_indices, dtype=np.int64).ravel()
        )
        return len(contained_values) == 1 and int(container_indices) == contained_values[0]
    container_values = np.asarray(container_indices, dtype=np.int64).ravel()
    if isinstance(contained_indices, int):
        return bool((container_values == int(contained_indices)).any())
    contained_values = np.asarray(contained_indices, dtype=np.int64).ravel()
    return bool(np.isin(contained_values, container_values).all())


_SIZE = 12

_AXIS_REGIONS = (
    AxisRegion(AxisRegionKind.ALL),
    AxisRegion(AxisRegionKind.POINT, 0),
    AxisRegion(AxisRegionKind.POINT, 5),
    AxisRegion(AxisRegionKind.POINT, 11),
    AxisRegion(AxisRegionKind.POINT, -1),
    AxisRegion(AxisRegionKind.SLICE, (0, 12, 1)),
    AxisRegion(AxisRegionKind.SLICE, (2, 9, 3)),
    AxisRegion(AxisRegionKind.SLICE, (0, None, 2)),
    AxisRegion(AxisRegionKind.SLICE, (0, None, 3)),
    AxisRegion(AxisRegionKind.SLICE, (1, 11, 2)),
    AxisRegion(AxisRegionKind.SLICE, (11, None, -3)),
    AxisRegion(AxisRegionKind.SLICE, (9, 1, -2)),
    AxisRegion(AxisRegionKind.SLICE, (0, 0, 1)),
    AxisRegion(AxisRegionKind.SLICE, (5, 6, 1)),
    AxisRegion(AxisRegionKind.SLICE, (-4, None, 1)),
    AxisRegion(AxisRegionKind.INDICES, ()),
    AxisRegion(AxisRegionKind.INDICES, (1,)),
    AxisRegion(AxisRegionKind.INDICES, (3, 7, 9)),
    AxisRegion(AxisRegionKind.INDICES, tuple(range(12))),
    AxisRegion(AxisRegionKind.INDICES, (2, 4, 6)),
    AxisRegion(AxisRegionKind.INDICES, (-1, 0)),
)


def test_axis_contains_matches_reference_across_kind_matrix():
    checked = 0
    for container in _AXIS_REGIONS:
        for contained in _AXIS_REGIONS:
            expected = _reference_axis_contains(container, contained, _SIZE)
            actual = _axis_contains(container, contained, _SIZE)
            assert actual == expected, (
                f"container={container!r} contained={contained!r}: "
                f"fast path {actual} != reference {expected}"
            )
            checked += 1
    assert checked == len(_AXIS_REGIONS) ** 2


def test_axis_contains_stride_alignment_is_not_endpoint_containment():
    # {0,3,6} has both endpoints on {0,2,4,6} but 3 is not a member; an
    # endpoint-only fast path would wrongly report containment.
    container = AxisRegion(AxisRegionKind.SLICE, (0, 7, 2))
    contained = AxisRegion(AxisRegionKind.SLICE, (0, 7, 3))
    assert _axis_contains(container, contained, _SIZE) is False
    assert _reference_axis_contains(container, contained, _SIZE) is False


def test_axis_contains_out_of_range_point_still_raises():
    container = AxisRegion(AxisRegionKind.ALL)
    contained = AxisRegion(AxisRegionKind.POINT, 40)
    with pytest.raises(IndexError):
        _axis_contains(container, contained, _SIZE)


def test_region_contains_matrix_composes_axis_fast_paths():
    shape = (4, 12)
    container = RegionSpec(
        (AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.INDICES, (0, 1, 2, 5)))
    )
    inside = RegionSpec(
        (AxisRegion(AxisRegionKind.SLICE, (0, 4, 1)), AxisRegion(AxisRegionKind.POINT, 5))
    )
    outside = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.POINT, 4)))
    assert region_contains(container, inside, shape) is True
    assert region_contains(container, outside, shape) is False
