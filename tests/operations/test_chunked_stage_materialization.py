import numpy as np
import pytest

from arrayscope.core.compute_policy import ComputeLane, EvaluationContext
from arrayscope.core.view_state import ViewState
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.chunked_stage import (
    materialize_stage_candidate_chunked,
    plan_chunked_stage_materialization,
)
from arrayscope.window.montage_renderer import _montage_stage_chunk_axes
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT, FFTShift
from arrayscope.operations.slabs import materialize_stage_candidate, plan_slab, request_for_image
from arrayscope.operations.stage_cache import StageCache


class _MemoryPolicy:
    stage_cache_budget_bytes = 1024 * 1024


def _fft_plan(shape=(6, 5, 4)):
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=len(shape) - 1),))
    state = ViewState.from_shape(document.current_shape).with_slice(len(shape) - 1, 1)
    plan = plan_slab(document, request_for_image(state))
    candidate = tuple(candidate for candidate in plan.region_plan.cache_candidates if candidate.retain)[-1]
    return document, plan, candidate


def test_fft_axis_stage_chunks_only_non_blocking_axes():
    _document, plan, candidate = _fft_plan()

    chunk_plan = plan_chunked_stage_materialization(plan.region_plan, candidate, _MemoryPolicy(), target_chunk_bytes=320, allowed_chunk_axes=(0, 1))

    assert chunk_plan is not None
    assert 2 in chunk_plan.blocking_axes
    assert 2 not in chunk_plan.chunk_axes
    assert chunk_plan.estimated_chunk_bytes <= 320


def test_chunk_plan_requires_explicit_allowed_axes():
    _document, plan, candidate = _fft_plan()

    assert plan_chunked_stage_materialization(plan.region_plan, candidate, _MemoryPolicy(), target_chunk_bytes=320) is None


def test_chunked_stage_equals_non_chunked_stage():
    document, plan, candidate = _fft_plan()
    context = EvaluationContext(ComputeLane.STAGE, None, 1, _MemoryPolicy())

    expected = materialize_stage_candidate(document, plan.region_plan, candidate, document_key=("doc",), evaluation_context=context)
    actual = materialize_stage_candidate_chunked(
        document,
        plan.region_plan,
        candidate,
        document_key=("doc",),
        evaluation_context=context,
        memory_policy=_MemoryPolicy(),
        target_chunk_bytes=320,
        allowed_chunk_axes=(0, 1),
    )

    np.testing.assert_allclose(actual.data, expected.data)
    assert actual.region == expected.region
    assert actual.stage_index == expected.stage_index


def test_chunked_stage_preserves_fftshift_region_semantics():
    shape = (6, 5, 8)
    data = np.zeros(shape, dtype=np.complex64)
    for index in range(shape[2]):
        data[:, :, index] = 1.0 if index % 2 == 0 else -1.0
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2)))
    state = ViewState.from_shape(document.current_shape).with_slice(2, 0)
    plan = plan_slab(document, request_for_image(state))
    candidate = tuple(candidate for candidate in plan.region_plan.cache_candidates if candidate.retain)[-1]
    context = EvaluationContext(ComputeLane.STAGE, None, 1, _MemoryPolicy())

    expected = materialize_stage_candidate(document, plan.region_plan, candidate, document_key=("doc",), evaluation_context=context)
    actual = materialize_stage_candidate_chunked(
        document,
        plan.region_plan,
        candidate,
        document_key=("doc",),
        evaluation_context=context,
        memory_policy=_MemoryPolicy(),
        target_chunk_bytes=512,
        allowed_chunk_axes=(0, 1),
    )

    np.testing.assert_allclose(actual.data, expected.data)


def test_chunked_stage_cancellation_prevents_cache_store():
    class Token:
        def __init__(self):
            self.calls = 0

        @property
        def cancelled(self):
            self.calls += 1
            return self.calls > 3

    document, plan, candidate = _fft_plan(shape=(8, 5, 4))
    cache = StageCache(max_bytes=1024 * 1024, max_entries=8)

    with pytest.raises(EvaluationCancelled):
        materialize_stage_candidate_chunked(
            document,
            plan.region_plan,
            candidate,
            stage_cache=cache,
            document_key=("doc",),
            cancellation_token=Token(),
            memory_policy=_MemoryPolicy(),
            target_chunk_bytes=320,
            allowed_chunk_axes=(0, 1),
        )

    assert cache.diagnostics().stores == 0


def test_no_chunk_plan_when_only_blocking_axis_can_be_split():
    _document, plan, candidate = _fft_plan(shape=(1, 1, 16))

    chunk_plan = plan_chunked_stage_materialization(plan.region_plan, candidate, _MemoryPolicy(), target_chunk_bytes=16, allowed_chunk_axes=(0, 1))

    assert chunk_plan is None


def test_chunk_plan_respects_memory_threshold():
    _document, plan, candidate = _fft_plan(shape=(12, 4, 4))

    small = plan_chunked_stage_materialization(plan.region_plan, candidate, _MemoryPolicy(), target_chunk_bytes=320, allowed_chunk_axes=(0, 1))
    large = plan_chunked_stage_materialization(plan.region_plan, candidate, _MemoryPolicy(), target_chunk_bytes=1024 * 1024, allowed_chunk_axes=(0, 1))

    assert small is not None
    assert len(small.chunks) > 1
    assert large is None


def test_montage_axis_can_be_allowed_but_fft_blocking_axis_is_not_chunked():
    _document, _plan, _candidate = _fft_plan(shape=(6, 5, 4))
    state = ViewState.from_shape((6, 5, 4)).with_montage_axis(2, columns=2, indices=tuple(range(4)), text=":")
    _document, plan, candidate = _fft_plan(shape=(6, 5, 4))

    assert _montage_stage_chunk_axes(state) == (2,)
    assert plan_chunked_stage_materialization(
        plan.region_plan,
        candidate,
        _MemoryPolicy(),
        target_chunk_bytes=16,
        allowed_chunk_axes=_montage_stage_chunk_axes(state),
    ) is None
