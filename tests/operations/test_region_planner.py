import numpy as np

from arrayscope.operations.pipeline import CenteredFFT, CenteredIFFT
from arrayscope.operations.planner import build_operation_stages, candidate_stage_cache_points
from arrayscope.operations.regions import (
    AxisRegionKind,
    expand_region_axes,
    index_spec_from_region,
    region_from_index_spec,
    region_is_full,
    region_nbytes,
    region_shape,
)


def test_region_spec_roundtrips_to_index_spec():
    region = region_from_index_spec((4, 5, 6), (1, slice(1, 5, 2), np.array([0, 3, 5])))
    spec = index_spec_from_region(region)

    assert spec[0] == 1
    assert spec[1] == slice(1, 5, 2)
    assert spec[2] == (0, 3, 5)
    assert hash(region)


def test_region_shapes_and_nbytes_for_point_slice_indices_all():
    region = region_from_index_spec((4, 5, 6, 7), (1, slice(1, 5, 2), np.array([0, 3, 5]), slice(None)))

    assert region_shape((4, 5, 6, 7), region) == (2, 3, 7)
    assert region_nbytes((4, 5, 6, 7), np.float32, region) == 2 * 3 * 7 * 4
    assert not region_is_full(region)


def test_expand_region_axes_converts_requested_axis_to_all():
    region = region_from_index_spec((4, 5, 6), (1, slice(1, 3), 2))
    expanded = expand_region_axes(region, (0, 2))

    assert expanded.axes[0].kind == AxisRegionKind.ALL
    assert expanded.axes[1].kind == AxisRegionKind.SLICE
    assert expanded.axes[2].kind == AxisRegionKind.ALL


def test_build_operation_stages_includes_base_and_operation_metadata():
    stages = build_operation_stages((4, 5, 6), np.float32, (CenteredFFT(axis=2),))

    assert len(stages) == 2
    assert stages[0].stage_index == 0
    assert stages[1].stage_index == 1
    assert stages[1].capabilities.cache_stage is True


def test_fft_over_sliced_axis_produces_expanded_cache_candidate():
    final_region = region_from_index_spec((4, 5, 6), (slice(None), slice(None), 2))
    candidates = candidate_stage_cache_points((4, 5, 6), np.float32, (CenteredFFT(axis=2),), final_region)

    assert len(candidates) == 1
    assert candidates[0].stage_index == 1
    assert candidates[0].region.axes[2].kind == AxisRegionKind.ALL
    assert candidates[0].priority == "high"


def test_fft_then_ifft_produces_two_ordered_candidates():
    final_region = region_from_index_spec((4, 5, 6), (slice(None), slice(None), 2))
    candidates = candidate_stage_cache_points((4, 5, 6), np.float32, (CenteredFFT(axis=2), CenteredIFFT(axis=2)), final_region)

    assert [candidate.stage_index for candidate in candidates] == [1, 2]
    assert candidates[1].priority == candidates[0].priority
