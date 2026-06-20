import numpy as np

from arrayscope.core.cache_status import CacheStatus
from arrayscope.core.view_state import ViewState
from arrayscope.display.montage import MontageTile, RenderedTile
from arrayscope.core.memory_policy import MemoryProfileChoice, compute_memory_policy, SystemMemorySnapshot
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


def test_bounded_array_cache_resize_evicts_to_new_byte_limit():
    cache = BoundedArrayCache(max_bytes=128, max_entries=10)
    cache.put("a", np.zeros(16, dtype=np.float32))
    cache.put("b", np.zeros(16, dtype=np.float32))

    cache.resize(max_bytes=64)

    assert cache.bytes_used <= 64
    assert len(cache._items) == 1
    assert cache.evictions == 1


def test_bounded_array_cache_resize_growing_preserves_entries():
    cache = BoundedArrayCache(max_bytes=64, max_entries=10)
    value = np.zeros(8, dtype=np.float32)
    cache.put("a", value)

    cache.resize(max_bytes=128)

    assert cache.get("a") is value


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


def test_montage_tile_cache_diagnostics_are_separate_from_image_cache():
    data = np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2)
    evaluator = OperationEvaluator(ArrayDocument(data))
    state = ViewState.from_shape(data.shape).with_montage_axis(2, columns=2, indices=(0, 1), text=":")
    from arrayscope.display.montage import make_montage_plan
    from arrayscope.display.slice_engine import DisplayImage
    from arrayscope.operations.evaluator import EvaluationResult

    plan = make_montage_plan(state, axis=2, indices=(0, 1), tile_shape=(2, 2), columns=2)
    image = np.zeros((2, 2), dtype=np.float32)
    evaluator.store_montage_tile_result(
        plan.tiles[0],
        montage_axis=2,
        colormap_lut=None,
        result=EvaluationResult(DisplayImage(image, histogram_data=image.copy()), 0.0, image.shape, int(image.nbytes)),
    )

    assert evaluator.tile_cache_diagnostics().entries == 1
    assert evaluator.image_cache_diagnostics().entries == 0


def test_image_cache_key_ignores_axis_flip_display_transform():
    data = np.zeros((2, 3, 4), dtype=np.float32)
    evaluator = OperationEvaluator(ArrayDocument(data))
    state = ViewState.from_shape(data.shape)
    flipped = state.with_axis_flipped(state.image_axes[0], True)

    assert evaluator.image_key(state) == evaluator.image_key(flipped)


def test_apply_memory_policy_resizes_image_tile_and_profile_caches():
    evaluator = OperationEvaluator(ArrayDocument(np.zeros((2, 3), dtype=np.float32)))
    system = SystemMemorySnapshot(total_bytes=8 * 1024**3, available_bytes=4 * 1024**3, process_rss_bytes=0)
    policy = compute_memory_policy(profile=MemoryProfileChoice.CONSERVATIVE, render_cap_mb=512, input_nbytes=1024, system=system)

    evaluator.apply_memory_policy(policy)

    assert evaluator.image_cache_diagnostics().max_bytes == policy.image_cache_budget_bytes
    assert evaluator.tile_cache_diagnostics().max_bytes == policy.tile_cache_budget_bytes
    assert evaluator.profile_cache_diagnostics().max_bytes == policy.profile_cache_budget_bytes
    assert evaluator.stage_cache_diagnostics().max_bytes == policy.stage_cache_budget_bytes


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


def test_operation_evaluator_clear_cache_clears_stage_cache():
    from arrayscope.operations.pipeline import CenteredFFT

    data = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2),)))
    evaluator.image(ViewState.from_shape(evaluator.document.current_shape).with_slice(2, 0))
    assert evaluator.stage_cache_diagnostics().entries == 1

    evaluator.clear_cache()

    assert evaluator.stage_cache_diagnostics().entries == 0


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
