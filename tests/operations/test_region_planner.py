import numpy as np
import pytest

from arrayscope.core.view_state import ViewState
from arrayscope.operations import plugins
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    CenteredFFTN,
    CenteredIFFT,
    CenteredIFFTN,
    CombineRealImagAxis,
    Crop,
    FFTShift,
    Mean,
    MeanOverAxes,
    ReverseAxis,
    SplitComplexAxis,
)
from arrayscope.operations.planner import (
    build_operation_stages,
    candidate_stage_cache_points,
    expanded_axes_for_transition,
    final_region_for_request,
    plan_region_request,
    required_input_region_for_operation,
)
from arrayscope.operations.plugins import PluginOperation, PluginOperationSpec
from arrayscope.operations.regions import (
    AxisRegion,
    AxisRegionKind,
    expand_region_axes,
    index_spec_from_region,
    region_from_index_spec,
    region_is_full,
    region_nbytes,
    region_shape,
)
from arrayscope.operations.slabs import (
    request_for_export_frame,
    request_for_image,
    request_for_line,
    request_for_scalar,
)


def test_region_spec_roundtrips_to_index_spec():
    region = region_from_index_spec((4, 5, 6), (1, slice(1, 5, 2), np.array([0, 3, 5])))
    spec = index_spec_from_region(region)

    assert spec[0] == 1
    assert spec[1] == slice(1, 5, 2)
    assert spec[2] == (0, 3, 5)
    assert hash(region)


def test_region_shapes_and_nbytes_for_point_slice_indices_all():
    region = region_from_index_spec(
        (4, 5, 6, 7), (1, slice(1, 5, 2), np.array([0, 3, 5]), slice(None))
    )

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
    candidates = candidate_stage_cache_points(
        (4, 5, 6), np.float32, (CenteredFFT(axis=2),), final_region
    )

    assert len(candidates) == 1
    assert candidates[0].stage_index == 1
    assert candidates[0].region.axes[2].kind == AxisRegionKind.ALL
    assert candidates[0].priority == "highest"
    assert candidates[0].retain is True


def test_fft_then_ifft_produces_two_ordered_candidates():
    final_region = region_from_index_spec((4, 5, 6), (slice(None), slice(None), 2))
    candidates = candidate_stage_cache_points(
        (4, 5, 6), np.float32, (CenteredFFT(axis=2), CenteredIFFT(axis=2)), final_region
    )

    assert [candidate.stage_index for candidate in candidates] == [1, 2]
    assert candidates[0].retain is False
    assert candidates[1].retain is True
    assert candidates[1].priority == "highest"


def test_final_region_for_requests_matches_view_state():
    state = ViewState.from_shape((4, 5, 6)).with_image_axes(0, 1).with_slice(2, 3)

    image = final_region_for_request(state.shape, request_for_image(state))
    line = final_region_for_request(state.shape, request_for_line(state.with_line_axis(2)))
    scalar = final_region_for_request(state.shape, request_for_scalar(state, (1, 2, 3)))
    export = final_region_for_request(state.shape, request_for_export_frame(state, 2, 4))

    assert [axis.kind for axis in image.axes] == [
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
        AxisRegionKind.POINT,
    ]
    assert [axis.kind for axis in line.axes] == [
        AxisRegionKind.POINT,
        AxisRegionKind.POINT,
        AxisRegionKind.ALL,
    ]
    assert all(axis.kind == AxisRegionKind.POINT for axis in scalar.axes)
    assert export.axes[2].value == 4


def test_operation_required_input_region_mappings():
    output = region_from_index_spec((4, 5, 6), (slice(None), 2, 3))

    crop = required_input_region_for_operation(Crop(axis=1, start=1, stop=4), (4, 5, 6), output)
    reverse = required_input_region_for_operation(ReverseAxis(axis=1), (4, 5, 6), output)
    shift = required_input_region_for_operation(FFTShift(axis=2), (4, 5, 6), output)
    fft = required_input_region_for_operation(CenteredFFT(axis=2), (4, 5, 6), output)
    mean = required_input_region_for_operation(
        Mean(axis=1), (4, 5, 6), region_from_index_spec((4, 6), (slice(None), 3))
    )

    assert crop.axes[1].value == 3
    assert reverse.axes[1].value == 2
    assert shift.axes[2].value == 0
    assert fft.axes[2].kind == AxisRegionKind.ALL
    assert [axis.kind for axis in mean.axes] == [
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
        AxisRegionKind.POINT,
    ]


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
    document = ArrayDocument(
        data, operations=(Crop(axis=1, start=1, stop=5), ReverseAxis(axis=0), CenteredFFT(axis=2))
    )
    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1).with_slice(2, 3)

    plan = plan_region_request(document, request_for_image(state))

    assert [type(transition.operation).__name__ for transition in plan.transitions] == [
        "Crop",
        "ReverseAxis",
        "CenteredFFT",
    ]
    assert plan.required_input_region.axes[2].kind == AxisRegionKind.ALL
    assert len(plan.cache_candidates) == 1
    assert plan.cache_candidates[0].stage_index == 3
    assert plan.cache_candidates[0].region.axes[2].kind == AxisRegionKind.ALL


def test_disabled_operation_steps_are_ignored_by_region_plan():
    from arrayscope.operations.pipeline import OperationStep

    data = np.zeros((4, 5, 6), dtype=np.float32)
    document = ArrayDocument(
        data,
        steps=(
            OperationStep(Crop(axis=1, start=1, stop=4), enabled=False),
            OperationStep(CenteredFFT(axis=2), enabled=True),
        ),
    )
    state = ViewState.from_shape(document.current_shape)

    plan = plan_region_request(document, request_for_image(state))

    assert [type(transition.operation).__name__ for transition in plan.transitions] == [
        "CenteredFFT"
    ]


def test_region_plan_uses_optimized_fft_ifft_pair():
    data = np.zeros((4, 5, 6), dtype=np.float32)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2), CenteredIFFT(axis=2)))
    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1).with_slice(2, 3)

    plan = plan_region_request(document, request_for_image(state))

    assert [type(transition.operation).__name__ for transition in plan.transitions] == ["CastDType"]
    assert plan.required_input_region == plan.final_region
    assert plan.cache_candidates == ()
    assert plan.original_operation_count == 2
    assert plan.optimized_operation_count == 1
    assert plan.optimization_steps


def test_region_plan_uses_composed_crop_and_removed_reverse_pair():
    data = np.zeros((4, 10, 6), dtype=np.float32)
    crop_doc = ArrayDocument(
        data, operations=(Crop(axis=1, start=2, stop=9), Crop(axis=1, start=3, stop=5))
    )
    crop_state = ViewState.from_shape(crop_doc.current_shape)
    crop_plan = plan_region_request(crop_doc, request_for_image(crop_state))

    assert [type(transition.operation).__name__ for transition in crop_plan.transitions] == ["Crop"]
    assert crop_plan.optimized_operation_count == 1

    reverse_doc = ArrayDocument(data, operations=(ReverseAxis(axis=1), ReverseAxis(axis=1)))
    reverse_state = ViewState.from_shape(reverse_doc.current_shape)
    reverse_plan = plan_region_request(reverse_doc, request_for_image(reverse_state))

    assert reverse_plan.transitions == ()
    assert reverse_plan.optimized_operation_count == 0


# --- axis-set operations: blocking = the set, chunkable = the complement -----


def test_axis_set_transform_blocks_only_its_own_axes_and_expands_them_together():
    capabilities = CenteredFFTN(axes=(1, 2)).capabilities((4, 8, 16), np.float32)

    assert capabilities.blocking_axes == (1, 2)
    assert capabilities.expands_request_axes == (1, 2)
    # The complement is what "broadcast over the rest" means at the engine
    # level: chunked evaluation walks axis 0 while each chunk carries whole
    # transformed axes.
    assert capabilities.chunkable_axes == (0,)
    assert capabilities.cache_stage is True


def test_axis_set_transform_asks_for_whole_applied_axes_and_slices_the_rest():
    output_region = region_from_index_spec((4, 8, 16), (2, slice(1, 5), slice(None)))

    required = CenteredFFTN(axes=(1, 2)).required_input_region((4, 8, 16), output_region)

    assert [axis.kind for axis in required.axes] == [
        AxisRegionKind.POINT,
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
    ]


def test_axis_set_transform_region_result_matches_the_whole_array_transform():
    rng = np.random.default_rng(0x5A1E)
    data = rng.standard_normal((4, 8, 6)).astype(np.float32)
    operation = CenteredFFTN(axes=(1, 2))
    whole = operation.apply(data)

    # A POINT on a non-applied axis AND a slice on an applied one: the applied
    # axes must be transformed whole and only then cut down, and the point drop
    # must not shift the slab positions the later cuts use.
    output_region = region_from_index_spec((4, 8, 6), (2, slice(2, 6), slice(None)))
    input_region = operation.required_input_region((4, 8, 6), output_region)
    slab = operation.apply_to_region(
        data[2], input_region=input_region, output_region=output_region
    )

    np.testing.assert_array_equal(slab, whole[2, 2:6, :])


def test_axis_set_inverse_transform_takes_the_same_region_path():
    rng = np.random.default_rng(0x1FF7)
    data = rng.standard_normal((4, 8, 6)).astype(np.float32)
    operation = CenteredIFFTN(axes=(1, 2))
    whole = operation.apply(data)

    output_region = region_from_index_spec((4, 8, 6), (2, slice(2, 6), slice(None)))
    input_region = operation.required_input_region((4, 8, 6), output_region)
    slab = operation.apply_to_region(
        data[2], input_region=input_region, output_region=output_region
    )

    np.testing.assert_array_equal(slab, whole[2, 2:6, :])


def test_axis_set_transform_survives_a_point_on_an_applied_axis():
    rng = np.random.default_rng(0xBEEF)
    data = rng.standard_normal((4, 8, 6)).astype(np.float32)
    operation = CenteredFFTN(axes=(0, 2))
    whole = operation.apply(data)

    # Points on BOTH applied axes: taking them drops axes, so the takes have to
    # run from the highest axis down or the second one indexes the wrong slab.
    output_region = region_from_index_spec((4, 8, 6), (1, slice(None), 3))
    input_region = operation.required_input_region((4, 8, 6), output_region)
    slab = operation.apply_to_region(data, input_region=input_region, output_region=output_region)

    np.testing.assert_array_equal(slab, whole[1, :, 3])


def test_axis_set_reduction_plans_and_evaluates_a_multi_axis_mean():
    rng = np.random.default_rng(0xD15C)
    data = rng.standard_normal((4, 5, 6, 3)).astype(np.float32)
    operation = MeanOverAxes(axes=(1, 3))
    document = ArrayDocument(data, operations=(operation,))

    assert document.current_shape == (4, 6)
    # Two reduced input axes with no output counterpart: the R2 bridge has to
    # translate the capability set, not hand it to the two-axis output region.
    assert operation.output_axis_sources((4, 5, 6, 3)) == (0, 2)

    output_region = region_from_index_spec((4, 6), (slice(None), slice(None)))
    input_region = operation.required_input_region((4, 5, 6, 3), output_region)
    assert [axis.kind for axis in input_region.axes] == [
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
    ]
    # Byte-identical, not merely close: the whole region is present, so the
    # region path is the same single ``np.mean`` over the same axis tuple.
    np.testing.assert_array_equal(
        operation.apply_to_region(data, input_region=input_region, output_region=output_region),
        data.mean(axis=(1, 3)),
    )

    state = ViewState.from_shape(document.current_shape)
    evaluated = OperationEvaluator(document).image(state)
    np.testing.assert_allclose(np.asarray(evaluated.data), data.mean(axis=(1, 3)), rtol=1e-6)


def test_scrubbing_a_broadcast_axis_of_an_axis_set_transform_reads_one_slab():
    """The broadcast half: an un-applied axis stays a POINT all the way down.

    This is what "apply over a subset of dims" has to buy. The plan for one
    displayed slice widens only the two transformed axes; the scrubbed axis is
    never expanded, so each step reads and transforms one slab rather than the
    volume -- and the values agree, step for step, with the single-axis chain
    the multi-axis op replaces.
    """

    rng = np.random.default_rng(0xC0FFEE)
    data = rng.standard_normal((6, 5, 4)).astype(np.float32)
    document = ArrayDocument(data, operations=(CenteredFFTN(axes=(1, 2)),))
    control = ArrayDocument(data, operations=(CenteredFFT(axis=1), CenteredFFT(axis=2)))
    state = ViewState.from_shape(document.current_shape).with_image_axes(1, 2)

    plan = plan_region_request(document, request_for_image(state.with_slice(0, 3)))
    assert [axis.kind for axis in plan.required_input_region.axes] == [
        AxisRegionKind.POINT,
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
    ]
    assert plan.transitions[0].expanded_axes == (1, 2)

    evaluator = OperationEvaluator(document)
    control_evaluator = OperationEvaluator(control)
    for index in range(6):
        np.testing.assert_allclose(
            np.asarray(evaluator.image(state.with_slice(0, index)).data),
            np.asarray(control_evaluator.image(state.with_slice(0, index)).data),
            rtol=1e-5,
            atol=1e-5,
        )


# --- rank-changing operations at the capability/region axis-space boundary ---
#
# A capability declares axis sets in INPUT space; a region indexes OUTPUT axes.
# The two coincide for every rank-preserving operation, which is exactly why the
# conflation survived. These pin both directions of a rank change.


def _plugin_operation(operation_id, fn):
    """Install a spec directly (the documented ``_reset_plugin_cache`` seam)."""

    spec = PluginOperationSpec(id=operation_id, label=operation_id, fn=fn, changes_shape=True)
    plugins._SPEC_CACHE[spec.id] = spec
    return PluginOperation(plugin_id=spec.id)


@pytest.fixture
def plugin_spec_cache():
    plugins._reset_plugin_cache()
    yield
    plugins._reset_plugin_cache()


def test_rank_reducing_plugin_op_plans_and_evaluates(plugin_spec_cache):
    # ``expands_request_axes`` used to reach ``expand_region_axes`` verbatim, so
    # an input-space axis 2 met a two-axis output region and plan_slab raised
    # "axis 2 is out of range for region with 2 axes".
    operation = _plugin_operation("demo:mean-axis-1", lambda a: a.mean(axis=1))
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    document = ArrayDocument(data, operations=(operation,))
    assert document.current_shape == (4, 6)

    state = ViewState.from_shape(document.current_shape)
    plan = plan_region_request(document, request_for_image(state))

    # The reduced input axis has no output counterpart, so it expands nothing;
    # the two surviving axes are already demanded whole here.
    assert len(plan.cache_candidates) == 1
    assert len(plan.cache_candidates[0].region.axes) == 2

    evaluated = OperationEvaluator(document).image(state)
    np.testing.assert_array_equal(np.asarray(evaluated.data), data.mean(axis=1))


def test_rank_increasing_plugin_op_expands_every_output_axis(plugin_spec_cache):
    operation = _plugin_operation(
        "demo:stack-4", lambda a: np.stack([a, a + 1, a + 2, a + 3], axis=0)
    )
    data = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    document = ArrayDocument(data, operations=(operation,))
    assert document.current_shape == (4, 5, 6)

    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1).with_slice(2, 3)
    plan = plan_region_request(document, request_for_image(state))

    # Opaque whole-array op: it computed the entire output, so every output axis
    # -- including the created one, which has no input axis at all -- is a
    # legitimate expansion. Declaring only the two INPUT axes left output axis 2
    # pinned to the requested slice.
    assert plan.transitions[0].expanded_axes == (0, 1, 2)
    assert [axis.kind for axis in plan.cache_candidates[0].region.axes] == [
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
        AxisRegionKind.ALL,
    ]

    evaluated = OperationEvaluator(document).image(state)
    expected = np.stack([data, data + 1, data + 2, data + 3], axis=0)[:, :, 3]
    np.testing.assert_array_equal(np.asarray(evaluated.data), expected)


def test_scrubbing_a_broadcast_axis_of_a_slab_mapped_op_runs_one_slab(plugin_spec_cache):
    """The point of slab mapping, counted rather than asserted about.

    A Tier-1 plugin op blocks every axis, so displaying slice ``i`` of a volume
    runs the author's function over the WHOLE array -- every slab, every step.
    Declaring ``maps_over_slabs`` moves the blocking to the selected axes only,
    so the planner hands the operation one slab and the function runs once.
    Five scrub steps: five slab calls, not five times six.
    """

    calls = {"n": 0}

    def demean(slab):
        calls["n"] += 1
        return slab - slab.mean()

    spec = PluginOperationSpec(
        id="demo:demean-slab-counted",
        label="Demean slab",
        fn=demean,
        requires_axes=True,
        maps_over_slabs=True,
    )
    plugins._SPEC_CACHE[spec.id] = spec
    operation = PluginOperation(plugin_id=spec.id, axes=(1, 2))

    rng = np.random.default_rng(0x5CAB)
    data = rng.standard_normal((6, 5, 4)).astype(np.float32)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(operation,)))
    state = ViewState.from_shape((6, 5, 4)).with_image_axes(1, 2)

    # Characterization probes the SLAB callable, so let those calls land before
    # the count that matters starts.
    evaluator.image(state.with_slice(0, 0))

    calls["n"] = 0
    for index in range(1, 6):
        evaluated = np.asarray(evaluator.image(state.with_slice(0, index)).data)
        np.testing.assert_allclose(
            evaluated, data[index] - data[index].mean(), rtol=1e-5, atol=1e-5
        )

    # One slab per step. Under the incumbent whole-array declaration this would
    # have been six slabs per step -- thirty.
    assert calls["n"] == 5


def test_scrubbing_a_created_axis_reuses_the_cached_stage(plugin_spec_cache):
    calls = {"n": 0}

    def stack_four(array):
        calls["n"] += 1
        return np.stack([array, array + 1, array + 2, array + 3], axis=0)

    operation = _plugin_operation("demo:stack-4-counted", stack_four)
    data = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(operation,)))
    state = ViewState.from_shape(evaluator.document.current_shape).with_image_axes(0, 1)

    calls["n"] = 0
    for index in range(6):
        evaluator.image(state.with_slice(2, index))

    # Six scrub steps used to mean six whole-array invocations of an OPAQUE op:
    # the cache candidate pinned the untranslated last output axis, so every
    # step missed. One invocation is the entire point of ``cache_stage``.
    assert calls["n"] == 1


def test_expanded_axes_for_transition_pairs_output_axes_with_their_input_axes():
    # The fallback for an operation that declares no capabilities. It used to
    # zip the two regions positionally, which pairs an output axis with
    # whichever input axis shares its index -- only the same axis when the rank
    # and the order both happen to match.
    class Unmapped:
        """No capabilities and no axis map: nothing can be paired."""

    class Mapped(Unmapped):
        def output_axis_sources(self, input_shape):
            del input_shape
            return (0, 2)

    output_region = region_from_index_spec((4, 6), (slice(None), 3))
    required = region_from_index_spec((4, 5, 6), (slice(None), slice(None), 3))

    assert expanded_axes_for_transition(output_region, required, Mapped(), (4, 5, 6)) == ()
    assert expanded_axes_for_transition(output_region, required, Unmapped(), (4, 5, 6)) == ()

    # Output axis 1 derives from input axis 2, which the input region demands
    # whole; the positional read would have looked at input axis 1 instead.
    whole_last = region_from_index_spec((4, 5, 6), (slice(None), 2, slice(None)))
    assert expanded_axes_for_transition(output_region, whole_last, Mapped(), (4, 5, 6)) == (1,)
