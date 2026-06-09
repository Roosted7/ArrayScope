import numpy as np
import pytest

from arrayscope.core.view_state import ChannelMode, ViewState
from arrayscope.operations import fft_backend
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.chunked import evaluate_image_snapshot_chunked
from arrayscope.operations.evaluator import EvaluationResult, evaluate_image_snapshot
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, Mean


class _Token:
    def __init__(self):
        self.cancelled = False


def test_chunked_image_matches_exact_raw_slice():
    data = np.arange(8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3)
    document = ArrayDocument(data)
    state = ViewState.from_shape(data.shape)

    exact = evaluate_image_snapshot(document, state)
    chunked = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=3)

    np.testing.assert_array_equal(chunked.value.data, exact.value.data)


def test_chunked_image_matches_exact_fft_pipeline():
    data = np.arange(8 * 9 * 4, dtype=np.float32).reshape(8, 9, 4)
    document = ArrayDocument(data, operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(document.current_shape)

    exact = evaluate_image_snapshot(document, state)
    chunked = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2)

    np.testing.assert_allclose(chunked.value.data, exact.value.data)


def test_chunked_image_matches_exact_reduction_pipeline():
    data = np.arange(8 * 9 * 4, dtype=np.float32).reshape(8, 9, 4)
    document = ArrayDocument(data, operations=(Mean(axis=2),))
    state = ViewState.from_shape(document.current_shape)

    exact = evaluate_image_snapshot(document, state)
    chunked = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2)

    np.testing.assert_allclose(chunked.value.data, exact.value.data)


def test_chunked_complex_rgb_matches_exact():
    data = (np.arange(8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3) + 1j).astype(np.complex64)
    document = ArrayDocument(data)
    state = ViewState.from_shape(data.shape).with_channel(ChannelMode.COMPLEX)

    exact = evaluate_image_snapshot(document, state)
    chunked = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=3)

    np.testing.assert_array_equal(chunked.value.data, exact.value.data)
    np.testing.assert_allclose(chunked.value.histogram_data, exact.value.histogram_data)


def test_chunked_evaluation_stops_when_cancelled_before_next_chunk(monkeypatch):
    import arrayscope.operations.evaluator as evaluator_module

    token = _Token()
    calls = []

    def fake_evaluate(
        document,
        view_state,
        colormap_lut=None,
        cancellation_token=None,
        degraded=False,
        stage_cache=None,
        stage_document_key=None,
    ):
        del colormap_lut, cancellation_token, degraded, stage_cache, stage_document_key
        calls.append(view_state)
        if len(calls) == 1:
            token.cancelled = True
        return EvaluationResult(
            value=evaluate_image_snapshot(document, view_state).value,
            eval_ms=0.0,
            slab_shape=(2, 4),
            slab_nbytes=32,
        )

    monkeypatch.setattr(evaluator_module, "evaluate_image_snapshot", fake_evaluate)
    document = ArrayDocument(np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
    state = ViewState.from_shape(document.current_shape)

    with pytest.raises(EvaluationCancelled):
        evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2, cancellation_token=token)

    assert len(calls) == 1


def test_chunked_cancellation_after_fft_runtime_options_change(monkeypatch):
    import arrayscope.operations.evaluator as evaluator_module

    fft_backend.set_fft_runtime_options(backend="numpy", workers="1")
    token = _Token()
    calls = []

    def fake_evaluate(
        document,
        view_state,
        colormap_lut=None,
        cancellation_token=None,
        degraded=False,
        stage_cache=None,
        stage_document_key=None,
    ):
        del colormap_lut, cancellation_token, degraded, stage_cache, stage_document_key
        calls.append(view_state)
        token.cancelled = True
        return EvaluationResult(
            value=evaluate_image_snapshot(document, view_state).value,
            eval_ms=0.0,
            slab_shape=(2, 4),
            slab_nbytes=32,
        )

    monkeypatch.setattr(evaluator_module, "evaluate_image_snapshot", fake_evaluate)
    document = ArrayDocument(np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
    state = ViewState.from_shape(document.current_shape)

    with pytest.raises(EvaluationCancelled):
        evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2, cancellation_token=token)

    assert len(calls) == 1


def test_chunked_result_reports_chunk_count_and_mode():
    document = ArrayDocument(np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
    state = ViewState.from_shape(document.current_shape)

    result = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2)

    assert result.mode == "chunked"
    assert result.chunk_count == 4
