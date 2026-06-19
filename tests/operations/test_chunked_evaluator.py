import numpy as np
import pytest

from arrayscope.core.view_state import ChannelMode, ScaleMode, ViewState
from arrayscope.operations import fft_backend
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.chunked import evaluate_image_snapshot_chunked
from arrayscope.operations.evaluator import EvaluationResult, evaluate_image_snapshot
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, Mean


class _Token:
    def __init__(self):
        self.cancelled = False


class _CountingToken:
    def __init__(self, cancel_after_checks: int):
        self._checks = 0
        self._cancel_after_checks = int(cancel_after_checks)

    @property
    def cancelled(self):
        self._checks += 1
        return self._checks > self._cancel_after_checks


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


def test_chunked_shader_complex_keeps_raw_texture_and_scaled_histogram():
    data = (np.arange(8 * 9 * 3, dtype=np.float32).reshape(8, 9, 3) + 1j).astype(np.complex64)
    document = ArrayDocument(data)
    state = ViewState.from_shape(data.shape).with_channel(ChannelMode.COMPLEX).with_scale(ScaleMode.LOG)

    chunked = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=3, shader_display=True)

    assert chunked.value.texture_kind == "complex_rg32f"
    assert np.iscomplexobj(chunked.value.data)
    np.testing.assert_array_equal(chunked.value.data, data[:, :, 0])
    np.testing.assert_array_equal(chunked.value.semantic_data, data[:, :, 0])
    np.testing.assert_allclose(chunked.value.histogram_data, np.log10(np.abs(data[:, :, 0])))


def test_chunked_evaluation_stops_when_cancelled_before_next_chunk(monkeypatch):
    del monkeypatch
    token = _CountingToken(cancel_after_checks=2)
    document = ArrayDocument(np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
    state = ViewState.from_shape(document.current_shape)

    with pytest.raises(Exception) as exc_info:
        evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2, cancellation_token=token)
    assert type(exc_info.value).__name__ == "EvaluationCancelled"


def test_chunked_cancellation_after_fft_runtime_options_change(monkeypatch):
    fft_backend.set_fft_runtime_options(backend="numpy", workers="1")
    del monkeypatch
    token = _CountingToken(cancel_after_checks=2)
    document = ArrayDocument(np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
    state = ViewState.from_shape(document.current_shape)

    with pytest.raises(Exception) as exc_info:
        evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2, cancellation_token=token)
    assert type(exc_info.value).__name__ == "EvaluationCancelled"


def test_chunked_result_reports_chunk_count_and_mode():
    document = ArrayDocument(np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
    state = ViewState.from_shape(document.current_shape)

    result = evaluate_image_snapshot_chunked(document, state, chunk_axis=0, chunk_size=2)

    assert result.mode == "chunked"
    assert result.chunk_count == 4
