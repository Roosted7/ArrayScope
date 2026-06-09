import ast
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.display.slice_engine import (
    make_export_frame,
    make_image,
    make_image_from_slab,
    make_line,
    make_line_from_slab,
    make_scalar_from_slab,
    apply_channel,
)
from arrayscope.operations.pipeline import (
    ArrayDocument,
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
    OperationStep,
)
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec, apply_subregion
from arrayscope.operations.slabs import evaluate_slab, plan_slab, request_for_export_frame, request_for_image, request_for_line, request_for_scalar
from arrayscope.operations.stage_cache import StageCache


def test_image_snapshot_plans_slab_once(monkeypatch):
    import arrayscope.operations.evaluator as evaluator_module

    calls = []
    original = evaluator_module.plan_slab

    def wrapped(document, request):
        calls.append(request)
        return original(document, request)

    monkeypatch.setattr(evaluator_module, "plan_slab", wrapped)
    document = ArrayDocument(np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6))
    state = ViewState.from_shape(document.current_shape)

    evaluator_module.evaluate_image_snapshot(document, state)

    assert len(calls) == 1


def _assert_image_and_line_match(data, operations):
    document = ArrayDocument(data, operations=operations)
    state = ViewState.from_shape(document.current_shape)
    full = document.materialize()

    image_request = request_for_image(state)
    lazy_image = make_image_from_slab(evaluate_slab(document, image_request), image_request)
    full_image = make_image(full, state)
    np.testing.assert_allclose(lazy_image.data, full_image.data)

    line_request = request_for_line(state)
    lazy_line = make_line_from_slab(evaluate_slab(document, line_request), line_request)
    full_line = make_line(full, state)
    np.testing.assert_allclose(lazy_line.data, full_line.data)


def test_lazy_slab_matches_materialized_image_and_line_for_existing_operations():
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(float)

    for operations in (
        (Crop(axis=1, start=1, stop=4),),
        (ReverseAxis(axis=0),),
        (Mean(axis=1),),
        (RootSumSquares(axis=1),),
        (FFTShift(axis=2),),
        (CenteredFFT(axis=2),),
        (Mean(axis=1), CenteredFFT(axis=1)),
        (Crop(axis=1, start=1, stop=4), ReverseAxis(axis=0), CenteredFFT(axis=2)),
    ):
        _assert_image_and_line_match(data, operations)


def test_stage_cache_backed_slab_matches_uncached_and_reuses_stage():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(document.current_shape)
    cache = StageCache(max_bytes=1024 * 1024, max_entries=8)

    request0 = request_for_image(state.with_slice(2, 0))
    uncached = evaluate_slab(document, request0)
    cached = evaluate_slab(document, request0, stage_cache=cache, document_key=("doc",))
    np.testing.assert_allclose(cached, uncached)
    assert cache.diagnostics().stores == 1

    request1 = request_for_image(state.with_slice(2, 1))
    cached_next = evaluate_slab(document, request1, stage_cache=cache, document_key=("doc",))
    np.testing.assert_allclose(cached_next, evaluate_slab(document, request1))
    assert cache.diagnostics().hits >= 1


def test_simplified_fft_ifft_slab_uses_dtype_preserving_identity_with_and_without_cache():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2), CenteredIFFT(axis=2)))
    state = ViewState.from_shape(document.current_shape)
    request = request_for_image(state.with_slice(2, 1))
    cache = StageCache(max_bytes=1024 * 1024, max_entries=8)

    uncached = evaluate_slab(document, request)
    cached = evaluate_slab(document, request, stage_cache=cache, document_key=("doc",))

    assert uncached.dtype == np.dtype(np.complex64)
    np.testing.assert_array_equal(uncached.real, data[:, :, 1])
    np.testing.assert_array_equal(uncached.imag, np.zeros_like(data[:, :, 1]))
    np.testing.assert_array_equal(cached, uncached)
    assert cache.diagnostics().stores == 0


def test_simplified_reverse_conjugate_and_crop_composition_match_materialized():
    data = (np.arange(4 * 10 * 6, dtype=np.float32).reshape(4, 10, 6) + 1j).astype(np.complex64)
    for operations in (
        (ReverseAxis(axis=1), ReverseAxis(axis=1)),
        (Conjugate(), Conjugate()),
        (Crop(axis=1, start=2, stop=9), Crop(axis=1, start=3, stop=5)),
    ):
        _assert_image_and_line_match(data, operations)


def test_apply_subregion_extracts_from_slice_and_indices_sources():
    data = np.arange(10)
    source_slice = RegionSpec((AxisRegion(AxisRegionKind.SLICE, (2, 9, 2)),))
    target_point = RegionSpec((AxisRegion(AxisRegionKind.POINT, 6),))
    np.testing.assert_array_equal(apply_subregion(data[2:9:2], source_region=source_slice, target_region=target_point, shape=(10,)), 6)

    source_indices = RegionSpec((AxisRegion(AxisRegionKind.INDICES, (1, 3, 7)),))
    target_indices = RegionSpec((AxisRegion(AxisRegionKind.INDICES, (7, 1)),))
    np.testing.assert_array_equal(
        apply_subregion(data[[1, 3, 7]], source_region=source_indices, target_region=target_indices, shape=(10,)),
        np.asarray([7, 1]),
    )


@pytest.mark.parametrize(
    "operations",
    (
        (Crop(axis=1, start=1, stop=4),),
        (ReverseAxis(axis=0),),
        (FFTShift(axis=2),),
        (Conjugate(),),
        (Mean(axis=1),),
        (Sum(axis=1),),
        (Maximum(axis=1),),
        (Minimum(axis=1),),
        (RootSumSquares(axis=1),),
        (CenteredFFT(axis=2),),
        (CenteredIFFT(axis=2),),
        (Crop(axis=1, start=1, stop=4), ReverseAxis(axis=0), CenteredFFT(axis=2)),
        (CenteredFFT(axis=2), Mean(axis=1)),
        (CenteredFFT(axis=2), CenteredIFFT(axis=2)),
    ),
)
def test_planner_backed_slab_matches_materialized_for_registered_operations(operations):
    data = (np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(float) + 1j).astype(np.complex64)
    _assert_image_and_line_match(data, operations)


def test_slab_matches_materialized_after_crop_reverse_same_axis():
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6)
    document = ArrayDocument(
        data,
        operations=(
            Crop(axis=1, start=1, stop=4),
            ReverseAxis(axis=1),
        ),
    )
    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1)

    lazy = make_image_from_slab(evaluate_slab(document, request_for_image(state)), request_for_image(state))
    full = make_image(document.materialize(), state)

    np.testing.assert_array_equal(lazy.data, full.data)


def test_lazy_image_axis_ranges_match_materialized_for_arbitrary_image_axes():
    data = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(float)
    document = ArrayDocument(
        data,
        operations=(
            Crop(axis=1, start=0, stop=3),
            ReverseAxis(axis=3),
            FFTShift(axis=2),
        ),
    )
    full = document.materialize()
    state = (
        ViewState.from_shape(document.current_shape)
        .with_image_axes(3, 1)
        .with_axis_range(3, (4, 2, 1), "4,2,1")
        .with_axis_range(1, (0, 2), "0:2:3")
        .with_slice(0, 1)
        .with_slice(2, 2)
    )

    request = request_for_image(state)
    lazy = make_image_from_slab(evaluate_slab(document, request), request)
    materialized = make_image(full, state)

    np.testing.assert_array_equal(lazy.data, materialized.data)
    assert request.ranged_axes == (3, 1)


def test_lazy_line_axis_range_matches_materialized_non_arithmetic_indices():
    data = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(float)
    document = ArrayDocument(data, operations=(ReverseAxis(axis=1), CenteredFFT(axis=2)))
    full = document.materialize()
    state = (
        ViewState.from_shape(document.current_shape)
        .with_line_axis(2)
        .with_axis_range(2, (4, 1, 3), "4,1,3")
        .with_slice(0, 2)
        .with_slice(1, 1)
    )

    request = request_for_line(state)
    lazy = make_line_from_slab(evaluate_slab(document, request), request)
    materialized = make_line(full, state)

    np.testing.assert_allclose(lazy.data, materialized.data)
    assert request.ranged_axes == (2,)


def test_lazy_slab_matches_materialized_complex_axis_operations():
    real_imag = np.arange(4 * 5 * 2).reshape(4, 5, 2).astype(float)
    _assert_image_and_line_match(real_imag, (CombineRealImagAxis(axis=2),))

    complex_singleton = (np.arange(4 * 5).reshape(4, 5).astype(float) + 1j).reshape(4, 5, 1)
    _assert_image_and_line_match(complex_singleton, (SplitComplexAxis(axis=2),))


def test_lazy_scalar_matches_materialized_value_after_fft_axis_expansion():
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(float)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(document.current_shape)
    index = (1, 2, 3)

    request = request_for_scalar(state, index)
    lazy_value = make_scalar_from_slab(evaluate_slab(document, request), request)
    full_value = document.materialize()[index]

    np.testing.assert_allclose(lazy_value, full_value)
    plan = plan_slab(document, request)
    assert plan.base_shape == (6,)
    assert plan.region_plan is not None
    assert plan.region_plan.final_region.axes[2].kind == AxisRegionKind.POINT
    assert plan.region_plan.required_input_region.axes[2].kind == AxisRegionKind.ALL
    assert len(plan.region_plan.cache_candidates) == 1


def test_plan_slab_includes_region_plan_for_fft_image_request():
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(float)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1).with_slice(2, 3)

    plan = plan_slab(document, request_for_image(state))

    assert plan.region_plan is not None
    assert plan.region_plan.request_kind == "image"
    assert plan.region_plan.final_region.axes[2].kind == AxisRegionKind.POINT
    assert plan.region_plan.required_input_region.axes[2].kind == AxisRegionKind.ALL


def test_simple_slice_only_operations_return_views_and_do_not_modify_base():
    data = np.arange(4 * 5 * 6).reshape(4, 5, 6).astype(float)
    original = data.copy()
    document = ArrayDocument(data, operations=(Crop(axis=1, start=1, stop=4), ReverseAxis(axis=0)))
    state = ViewState.from_shape(document.current_shape)

    slab = evaluate_slab(document, request_for_image(state))

    assert np.shares_memory(slab, data)
    np.testing.assert_array_equal(data, original)


def test_slab_and_cache_modules_have_no_qt_or_pyqtgraph_imports():
    root = Path(__file__).parents[2]
    for path in (
        root / "arrayscope" / "operations" / "slabs.py",
        root / "arrayscope" / "operations" / "cache.py",
    ):
        tree = ast.parse(path.read_text())
        imported_roots = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.append(node.module.split(".")[0])

        assert "pyqtgraph" not in imported_roots
        assert not any(name.startswith(("PyQt", "PySide")) for name in imported_roots)


def test_random_small_arrays_match_materialized_image_export_profile_and_scalar_paths():
    rng = np.random.default_rng(123)
    real_data = rng.normal(size=(4, 5, 6))
    complex_parts = rng.normal(size=(4, 5, 2))
    complex_singleton = (rng.normal(size=(4, 5)) + 1j * rng.normal(size=(4, 5))).reshape(4, 5, 1)

    documents = [
        ArrayDocument(real_data, operations=(Crop(axis=1, start=1, stop=5), ReverseAxis(axis=0), CenteredFFT(axis=2), RootSumSquares(axis=1))),
        ArrayDocument(real_data, operations=(Crop(axis=1, start=0, stop=4), Mean(axis=1), CenteredFFT(axis=1))),
        ArrayDocument(complex_parts, operations=(CombineRealImagAxis(axis=2),)),
        ArrayDocument(complex_singleton, operations=(SplitComplexAxis(axis=2), ReverseAxis(axis=0))),
        ArrayDocument(
            real_data,
            steps=(
                OperationStep(Crop(axis=1, start=1, stop=4), enabled=True),
                OperationStep(ReverseAxis(axis=0), enabled=False),
                OperationStep(CenteredFFT(axis=2), enabled=True),
            ),
        ),
        ArrayDocument(real_data, operations=(CenteredFFT(axis=2), Crop(axis=1, start=1, stop=5), ReverseAxis(axis=0))),
    ]

    for document in documents:
        full = document.materialize()
        state = ViewState.from_shape(document.current_shape)
        if np.iscomplexobj(full):
            channels = (ChannelMode.ABS, ChannelMode.ANGLE, ChannelMode.REAL)
        else:
            channels = (ChannelMode.REAL,)
        for channel in channels:
            state = state.with_channel(channel)
            image_request = request_for_image(state)
            np.testing.assert_allclose(
                make_image_from_slab(evaluate_slab(document, image_request), image_request).data,
                make_image(full, state).data,
                rtol=1e-6,
                atol=1e-6,
            )

            line_request = request_for_line(state)
            np.testing.assert_allclose(
                make_line_from_slab(evaluate_slab(document, line_request), line_request).data,
                make_line(full, state).data,
                rtol=1e-6,
                atol=1e-6,
            )

            if state.image_axes is not None:
                frame_axis = next((axis for axis in range(state.ndim) if axis not in state.image_axes and state.shape[axis] > 1), None)
                if frame_axis is not None:
                    frame_index = min(1, state.shape[frame_axis] - 1)
                    export_request = request_for_export_frame(state, frame_axis, frame_index)
                    np.testing.assert_allclose(
                        make_image_from_slab(evaluate_slab(document, export_request), export_request).data,
                        make_export_frame(full, state, frame_axis, frame_index).data,
                        rtol=1e-6,
                        atol=1e-6,
                    )

            scalar_index = tuple(min(1, size - 1) for size in state.shape)
            scalar_request = request_for_scalar(state, scalar_index)
            np.testing.assert_allclose(
                make_scalar_from_slab(evaluate_slab(document, scalar_request), scalar_request),
                np.asarray(apply_channel(full[scalar_index if scalar_index else ()], channel)).item(),
                rtol=1e-6,
                atol=1e-6,
            )


@st.composite
def _small_document_specs(draw):
    ndim = draw(st.integers(2, 4))
    shape = tuple(draw(st.integers(2, 5)) for _ in range(ndim))
    current_shape = shape
    operations = []
    max_ops = draw(st.integers(0, 5))
    for _ in range(max_ops):
        candidates = ["reverse", "crop", "fftshift", "fft"]
        if len(current_shape) > 1:
            candidates.extend(["mean", "rss"])
        kind = draw(st.sampled_from(candidates))
        axis = draw(st.integers(0, len(current_shape) - 1))
        size = current_shape[axis]
        if kind == "reverse":
            operations.append(ReverseAxis(axis))
        elif kind == "crop":
            start = draw(st.integers(0, size - 1))
            stop = draw(st.integers(start + 1, size))
            operations.append(Crop(axis, start, stop))
            current_shape = current_shape[:axis] + (stop - start,) + current_shape[axis + 1 :]
        elif kind == "fftshift":
            operations.append(FFTShift(axis))
        elif kind == "fft":
            operations.append(CenteredFFT(axis))
        elif kind == "mean" and len(current_shape) > 1:
            operations.append(Mean(axis))
            current_shape = current_shape[:axis] + current_shape[axis + 1 :]
        elif kind == "rss" and len(current_shape) > 1:
            operations.append(RootSumSquares(axis))
            current_shape = current_shape[:axis] + current_shape[axis + 1 :]
        if len(current_shape) < 1:
            break
    return shape, tuple(operations)


@given(_small_document_specs())
@settings(max_examples=80, deadline=None)
def test_hypothesis_lazy_slab_matches_materialized_image_line_scalar_and_export(spec):
    shape, operations = spec
    data = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    document = ArrayDocument(data, operations=operations)
    if len(document.current_shape) < 1:
        return
    full = document.materialize()
    state = ViewState.from_shape(document.current_shape)

    if state.image_axes is not None:
        image_request = request_for_image(state)
        np.testing.assert_allclose(
            make_image_from_slab(evaluate_slab(document, image_request), image_request).data,
            make_image(full, state).data,
            rtol=1e-6,
            atol=1e-6,
        )

    line_request = request_for_line(state)
    np.testing.assert_allclose(
        make_line_from_slab(evaluate_slab(document, line_request), line_request).data,
        make_line(full, state).data,
        rtol=1e-6,
        atol=1e-6,
    )

    scalar_index = tuple(min(1, size - 1) for size in state.shape)
    scalar_request = request_for_scalar(state, scalar_index)
    np.testing.assert_allclose(
        make_scalar_from_slab(evaluate_slab(document, scalar_request), scalar_request),
        np.asarray(apply_channel(full[scalar_index], state.channel)).item(),
        rtol=1e-6,
        atol=1e-6,
    )

    if state.image_axes is not None:
        frame_axis = next((axis for axis in range(state.ndim) if axis not in state.image_axes and state.shape[axis] > 1), None)
        if frame_axis is not None:
            frame_index = min(1, state.shape[frame_axis] - 1)
            export_request = request_for_export_frame(state, frame_axis, frame_index)
            np.testing.assert_allclose(
                make_image_from_slab(evaluate_slab(document, export_request), export_request).data,
                make_export_frame(full, state, frame_axis, frame_index).data,
                rtol=1e-6,
                atol=1e-6,
            )
