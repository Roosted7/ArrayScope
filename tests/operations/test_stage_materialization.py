import numpy as np

from arrayscope.core.view_state import ViewState
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT
from arrayscope.operations.slabs import plan_slab, request_for_image
from arrayscope.operations.stage_cache import StageCache, StageValue
from arrayscope.operations.stage_materialization import StageMaterializationManager


def _candidate():
    document = ArrayDocument(np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6), operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(document.current_shape).with_slice(2, 1)
    plan = plan_slab(document, request_for_image(state))
    return document, plan.region_plan.cache_candidates[0]


def test_stage_materialization_singleflight_and_complete():
    document, candidate = _candidate()
    cache = StageCache(max_bytes=1024 * 1024, max_entries=8)
    manager = StageMaterializationManager(cache)
    document_key = ("doc",)

    first = manager.request_stage(document_key, candidate)
    second = manager.request_stage(document_key, candidate)

    assert first.decision == "scheduled"
    assert second.decision == "attached"
    assert manager.diagnostics().in_flight == 1

    value = StageValue(
        data=np.zeros(candidate.shape, dtype=np.complex64),
        region=candidate.region,
        stage_index=candidate.stage_index,
        nbytes=16,
        priority=candidate.priority,
    )
    manager.complete(first.key, value)

    assert manager.diagnostics().in_flight == 0
    assert manager.diagnostics().completed == 1


def test_stage_materialization_refusal_records_consequence():
    _document, candidate = _candidate()
    cache = StageCache(max_bytes=1, max_entries=8)
    manager = StageMaterializationManager(cache)

    result = manager.request_stage(("doc",), candidate)

    assert result.decision == "refused"
    diagnostics = manager.diagnostics()
    assert diagnostics.refused == 1
    assert "recompute" in diagnostics.consequence


def test_stage_materialization_invalidate_document_clears_inflight():
    _document, candidate = _candidate()
    cache = StageCache(max_bytes=1024 * 1024, max_entries=8)
    manager = StageMaterializationManager(cache)
    key = ("doc",)
    manager.request_stage(key, candidate)

    manager.invalidate_document(key)

    assert manager.diagnostics().in_flight == 0
