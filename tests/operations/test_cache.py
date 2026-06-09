import numpy as np

from arrayscope.core.cache_status import CacheStatus
from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import MontageTile, RenderedTile
from arrayscope.operations.cache import BoundedArrayCache, _nbytes
from arrayscope.operations.evaluator import OperationEvaluator
from arrayscope.operations.coordinator import OperationCoordinator
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
    assert diagnostics.hit_rate == 0.5
    assert diagnostics.evictions >= 1
    assert diagnostics.bytes_used <= diagnostics.max_bytes


def test_bounded_cache_counts_rendered_tile_bytes():
    tile = MontageTile(
        montage_index=0,
        source_index=0,
        row=0,
        col=0,
        x0=0,
        y0=0,
        width=100,
        height=100,
        view_state=None,
    )
    rendered = RenderedTile(
        tile=tile,
        image=np.zeros((100, 100), dtype=np.float32),
        histogram_data=np.zeros((100, 100), dtype=np.float32),
        eval_ms=0.0,
        slab_shape=(),
        slab_nbytes=None,
    )

    assert rendered.nbytes() == 80_000
    assert _nbytes(rendered) == 80_000

    large_cache = BoundedArrayCache(max_bytes=100_000, max_entries=96)
    large_cache.put("tile", rendered)
    assert large_cache.bytes_used == 80_000
    assert len(large_cache._items) == 1

    small_cache = BoundedArrayCache(max_bytes=1024, max_entries=96)
    small_cache.put("tile", rendered)
    assert small_cache.bytes_used == 0
    assert len(small_cache._items) == 0
    assert small_cache.evictions == 1


def test_bounded_cache_uses_nbytes_protocol_before_fallbacks():
    class ProtocolValue:
        image = np.zeros((100, 100), dtype=np.float32)

        def nbytes(self):
            return 12

    assert _nbytes(ProtocolValue()) == 12


def test_bounded_cache_counts_image_and_histogram_attrs_without_protocol():
    class ImageValue:
        def __init__(self):
            self.image = np.zeros((5, 6), dtype=np.float32)
            self.histogram_data = np.zeros((5, 6), dtype=np.float64)

    assert _nbytes(ImageValue()) == 360


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
    assert evaluator.image_cache_diagnostics().prefetch_stored == 1


def test_prefetch_store_uses_document_key_not_array_equality():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    evaluator = OperationEvaluator(ArrayDocument(data))
    other_document = ArrayDocument(data.copy())
    state = ViewState.from_shape(data.shape)
    result = evaluator.prefetch_image_snapshot(other_document, state)

    assert evaluator.store_prefetch_image_result(other_document, state, None, result) is False
    assert evaluator.image_cache_diagnostics().prefetch_stale == 1


def test_prefetch_diagnostics_track_scheduling_outcomes():
    evaluator = OperationEvaluator(ArrayDocument(np.arange(6).reshape(2, 3)))

    evaluator.note_prefetch_scheduled()
    evaluator.note_prefetch_deduped()
    evaluator.note_prefetch_limited()
    evaluator.note_prefetch_stale()

    diagnostics = evaluator.image_cache_diagnostics()
    assert diagnostics.prefetch_scheduled == 1
    assert diagnostics.prefetch_deduped == 1
    assert diagnostics.prefetch_limited == 1
    assert diagnostics.prefetch_stale == 1


def test_evaluator_diagnostics_counts_degraded_refused_chunked_cancelled():
    evaluator = OperationEvaluator(ArrayDocument(np.arange(6).reshape(2, 3)))

    evaluator.note_render_degraded()
    evaluator.note_render_refused("too large")
    evaluator.note_chunked_evaluation()
    evaluator.note_render_cancelled()

    diagnostics = evaluator.image_cache_diagnostics()
    assert diagnostics.degraded_evaluations == 1
    assert diagnostics.refused_evaluations == 1
    assert diagnostics.chunked_evaluations == 1
    assert diagnostics.cancelled_evaluations == 1


def test_cache_invalidates_when_document_revision_changes_for_same_array_object():
    data = np.arange(3 * 4).reshape(3, 4).astype(float)
    state = ViewState.from_shape(data.shape)
    evaluator = OperationEvaluator(ArrayDocument(data))

    first = evaluator.image(state)
    data[0, 0] = 999
    evaluator.set_document(evaluator.document.with_data_changed())
    second = evaluator.image(state)

    assert evaluator.document.revision == 1
    assert first is not second
    assert second.data[0, 0] == 999


def test_operation_coordinator_marks_base_data_changed_and_replaces_base_data_revision():
    data = np.arange(6).reshape(2, 3)
    coordinator = OperationCoordinator(data)
    initial_revision = coordinator.document.revision

    coordinator.mark_base_data_changed()
    assert coordinator.document.revision == initial_revision + 1

    coordinator.replace_base_data(data)
    assert coordinator.document.revision == initial_revision + 2
