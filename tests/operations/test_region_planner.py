import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT, CombineRealImagAxis, Crop, FFTShift, Mean, ReverseAxis, SplitComplexAxis
from arrayscope.operations.planner import (
    build_operation_stages,
    candidate_stage_cache_points,
    final_region_for_request,
    plan_region_request,
    required_input_region_for_operation,
)
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, expand_region_axes, index_spec_from_region, region_from_index_spec, region_is_full, region_nbytes, region_shape
from arrayscope.operations.slabs import request_for_export_frame, request_for_image, request_for_line, request_for_scalar


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


def test_final_region_for_requests_matches_view_state():
    state = ViewState.from_shape((4, 5, 6)).with_image_axes(0, 1).with_slice(2, 3)

    image = final_region_for_request(state.shape, request_for_image(state))
    line = final_region_for_request(state.shape, request_for_line(state.with_line_axis(2)))
    scalar = final_region_for_request(state.shape, request_for_scalar(state, (1, 2, 3)))
    export = final_region_for_request(state.shape, request_for_export_frame(state, 2, 4))

    assert [axis.kind for axis in image.axes] == [AxisRegionKind.ALL, AxisRegionKind.ALL, AxisRegionKind.POINT]
    assert [axis.kind for axis in line.axes] == [AxisRegionKind.POINT, AxisRegionKind.POINT, AxisRegionKind.ALL]
    assert all(axis.kind == AxisRegionKind.POINT for axis in scalar.axes)
    assert export.axes[2].value == 4


def test_operation_required_input_region_mappings():
    output = region_from_index_spec((4, 5, 6), (slice(None), 2, 3))

    crop = required_input_region_for_operation(Crop(axis=1, start=1, stop=4), (4, 5, 6), output)
    reverse = required_input_region_for_operation(ReverseAxis(axis=1), (4, 5, 6), output)
    shift = required_input_region_for_operation(FFTShift(axis=2), (4, 5, 6), output)
    fft = required_input_region_for_operation(CenteredFFT(axis=2), (4, 5, 6), output)
    mean = required_input_region_for_operation(Mean(axis=1), (4, 5, 6), region_from_index_spec((4, 6), (slice(None), 3)))

    assert crop.axes[1].value == 3
    assert reverse.axes[1].value == 2
    assert shift.axes[2].value == 0
    assert fft.axes[2].kind == AxisRegionKind.ALL
    assert [axis.kind for axis in mean.axes] == [AxisRegionKind.ALL, AxisRegionKind.ALL, AxisRegionKind.POINT]


def test_complex_region_mappings():
    combine = required_input_region_for_operation(
        CombineRealImagAxis(axis=2),
        (4, 5, 2),
        region_from_index_spec((4, 5, 1), (slice(None), slice(None), 0)),
    )
    split = required_input_region_for_operation(
        SplitComplexAxis(axis=2),
        (4, 5, 1),
        region_from_index_spec((4, 5, 2), (slice(None), slice(None), 1)),
    )

    assert combine.axes[2].kind == AxisRegionKind.ALL
    assert split.axes[2] == AxisRegion(AxisRegionKind.POINT, 0)


def test_plan_region_request_records_transitions_and_candidates():
    data = np.zeros((4, 5, 6), dtype=np.float32)
    document = ArrayDocument(data, operations=(Crop(axis=1, start=1, stop=5), ReverseAxis(axis=0), CenteredFFT(axis=2)))
    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1).with_slice(2, 3)

    plan = plan_region_request(document, request_for_image(state))

    assert [type(transition.operation).__name__ for transition in plan.transitions] == ["Crop", "ReverseAxis", "CenteredFFT"]
    assert plan.required_input_region.axes[2].kind == AxisRegionKind.ALL
    assert len(plan.cache_candidates) == 1
    assert plan.cache_candidates[0].stage_index == 3
    assert plan.cache_candidates[0].region.axes[2].kind == AxisRegionKind.ALL


def test_disabled_operation_steps_are_ignored_by_region_plan():
    from arrayscope.operations.pipeline import OperationStep

    data = np.zeros((4, 5, 6), dtype=np.float32)
    document = ArrayDocument(
        data,
        steps=(OperationStep(Crop(axis=1, start=1, stop=4), enabled=False), OperationStep(CenteredFFT(axis=2), enabled=True)),
    )
    state = ViewState.from_shape(document.current_shape)

    plan = plan_region_request(document, request_for_image(state))

    assert [type(transition.operation).__name__ for transition in plan.transitions] == ["CenteredFFT"]
