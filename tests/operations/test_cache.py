import numpy as np

from arrayscope.core.cache_status import CacheStatus
from arrayscope.core.view_state import ViewState
from arrayscope.operations.cache import BoundedArrayCache
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.pipeline import ArrayDocument, ReverseAxis


def test_bounded_array_cache_tracks_hits_misses_and_evictions():
    cache = BoundedArrayCache(max_bytes=32, max_entries=2)
    a = np.arange(4, dtype=np.float64)
    b = np.arange(4, dtype=np.float64)

    assert cache.get("a") is None
    cache.put("a", a)
    assert cache.get("a") is a
    cache.put("b", b)

    diagnostics = cache.diagnostics(CacheStatus.READY, "ready")
    assert diagnostics.hits == 1
    assert diagnostics.misses == 1
    assert diagnostics.evictions >= 1
    assert diagnostics.bytes_used <= diagnostics.max_bytes


def test_operation_evaluator_uses_bounded_display_cache_and_invalidates_by_document():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    state = ViewState.from_shape(data.shape)
    evaluator = OperationEvaluator(ArrayDocument(data))

    first = evaluator.image(state)
    second = evaluator.image(state)
    assert first is second
    assert evaluator.image_evaluations == 1
    assert evaluator.derived_evaluations == 0
    assert evaluator.cache_diagnostics().hits == 1

    evaluator.set_document(evaluator.document.with_operation(ReverseAxis(axis=0)))
    evaluator.image(state)
    assert evaluator.image_evaluations == 2
    assert evaluator.derived_evaluations == 0


def test_prefetch_fills_cache_without_incrementing_visible_evaluation_count():
    data = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype(float)
    state = ViewState.from_shape(data.shape).with_slice(2, 1)
    prefetch_state = state.with_slice(2, 2)
    evaluator = OperationEvaluator(ArrayDocument(data))

    evaluator.image(state)
    result = evaluator.prefetch_image_snapshot(evaluator.document, prefetch_state)
    assert evaluator.store_prefetch_image_result(evaluator.document, prefetch_state, None, result) is True
    visible_count = evaluator.image_evaluations
    evaluator.image(prefetch_state)

    assert evaluator.image_evaluations == visible_count
    assert evaluator.cache_diagnostics().status == CacheStatus.CACHED


def test_prefetch_store_uses_document_key_not_array_equality():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    evaluator = OperationEvaluator(ArrayDocument(data))
    other_document = ArrayDocument(data.copy())
    state = ViewState.from_shape(data.shape)
    result = evaluator.prefetch_image_snapshot(other_document, state)

    assert evaluator.store_prefetch_image_result(other_document, state, None, result) is False
