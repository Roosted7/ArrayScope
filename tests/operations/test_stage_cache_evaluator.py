import numpy as np

from arrayscope.core.memory_policy import MemoryProfileChoice, SystemMemorySnapshot, compute_memory_policy
from arrayscope.core.view_state import ViewState
from arrayscope.operations import dim_ops
from arrayscope.operations.evaluator import OperationEvaluator, evaluate_image_snapshot, stage_document_key
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT, ReverseAxis


def test_fft_over_sliced_axis_stores_and_reuses_expanded_stage(monkeypatch):
    calls = {"fft": 0}
    original = dim_ops.centered_fft

    def counted(data, axis):
        calls["fft"] += 1
        return original(data, axis)

    monkeypatch.setattr(dim_ops, "centered_fft", counted)
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2),)))
    state = ViewState.from_shape(evaluator.document.current_shape)

    evaluator.image(state.with_slice(2, 0))
    diagnostics = evaluator.stage_cache_diagnostics()
    assert diagnostics.stores == 1
    assert diagnostics.entries == 1
    assert calls["fft"] == 1

    evaluator.image(state.with_slice(2, 1))
    diagnostics = evaluator.stage_cache_diagnostics()
    assert calls["fft"] == 1
    assert diagnostics.hits >= 1


def test_fft_ifft_pair_is_simplified_without_transform_or_stage_cache(monkeypatch):
    calls = {"fft": 0, "ifft": 0}
    original_fft = dim_ops.centered_fft
    original_ifft = dim_ops.centered_ifft

    def counted_fft(data, axis):
        calls["fft"] += 1
        return original_fft(data, axis)

    def counted_ifft(data, axis):
        calls["ifft"] += 1
        return original_ifft(data, axis)

    monkeypatch.setattr(dim_ops, "centered_fft", counted_fft)
    monkeypatch.setattr(dim_ops, "centered_ifft", counted_ifft)
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2), CenteredIFFT(axis=2))))
    state = ViewState.from_shape(evaluator.document.current_shape)

    evaluator.image(state.with_slice(2, 0))
    diagnostics = evaluator.stage_cache_diagnostics()
    assert diagnostics.stores == 0
    assert diagnostics.entries == 0
    assert calls == {"fft": 0, "ifft": 0}

    evaluator.image(state.with_slice(2, 1))
    diagnostics = evaluator.stage_cache_diagnostics()
    assert calls == {"fft": 0, "ifft": 0}
    assert diagnostics.entries == 0


def test_stage_cache_invalidates_on_operation_and_revision_changes():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2),)))
    state = ViewState.from_shape(evaluator.document.current_shape)

    evaluator.image(state.with_slice(2, 0))
    assert evaluator.stage_cache_diagnostics().entries == 1

    evaluator.set_document(evaluator.document.with_operation(ReverseAxis(axis=0)))
    assert evaluator.stage_cache_diagnostics().entries == 0

    evaluator.image(ViewState.from_shape(evaluator.document.current_shape).with_slice(2, 0))
    assert evaluator.stage_cache_diagnostics().entries >= 1
    evaluator.set_document(evaluator.document.with_data_changed())
    assert evaluator.stage_cache_diagnostics().entries == 0


def test_stage_cache_respects_memory_policy_shrink():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2),)))
    state = ViewState.from_shape(evaluator.document.current_shape)
    evaluator.image(state.with_slice(2, 0))
    assert evaluator.stage_cache_diagnostics().entries == 1

    system = SystemMemorySnapshot(total_bytes=8 * 1024**3, available_bytes=4 * 1024**3, process_rss_bytes=0)
    policy = compute_memory_policy(profile=MemoryProfileChoice.BALANCED, render_cap_mb=512, input_nbytes=data.nbytes, system=system)
    policy = policy.__class__(**{**policy.__dict__, "stage_cache_budget_bytes": 1})
    evaluator.apply_memory_policy(policy)

    assert evaluator.stage_cache_diagnostics().entries == 0


def test_profile_scalar_export_and_montage_like_requests_reuse_stage(monkeypatch):
    calls = {"fft": 0}
    original = dim_ops.centered_fft

    def counted(data, axis):
        calls["fft"] += 1
        return original(data, axis)

    monkeypatch.setattr(dim_ops, "centered_fft", counted)
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    evaluator = OperationEvaluator(ArrayDocument(data, operations=(CenteredFFT(axis=2),)))
    state = ViewState.from_shape(evaluator.document.current_shape)

    evaluator.image(state.with_slice(2, 0))
    evaluator.scalar(state.with_slice(2, 1), (1, 2, 1))
    evaluator.line(state.with_line_axis(0).with_slice(2, 2))
    evaluator.export_frame(state, 2, 3)
    evaluate_image_snapshot(
        evaluator.document,
        state.with_slice(2, 4),
        stage_cache=evaluator.stage_cache,
        stage_document_key=stage_document_key(evaluator.document),
    )

    assert calls["fft"] == 1
    assert evaluator.stage_cache_diagnostics().hits >= 4
