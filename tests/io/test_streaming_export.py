"""Streaming must preserve exact values and refuse repeated source work."""

from dataclasses import replace
from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

from arrayscope.core.array_source import (
    DEFAULT_SOURCE_READ_BUDGET_BYTES,
    LazySourceArray,
    SourceReadRefused,
    read_index_spec,
)
from arrayscope.io import streaming_export
from arrayscope.io.region_export import RegionExportChanged, RegionExportRequest
from arrayscope.io.streaming_export import (
    StreamingExportRefused,
    export_streaming_region,
    plan_streaming_export,
)
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import (
    ArrayDocument,
    CenteredFFT,
    Crop,
    FFTShift,
    Mean,
    ReverseAxis,
    Scale,
    SplitAxis,
    Squeeze,
    Transpose,
)
from arrayscope.operations.regions import region_from_index_spec


class CountedSource:
    chunk_shape = None
    label = "streaming-counter"

    def __init__(self, values):
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.nbytes = values.nbytes
        self.reads = []
        self.changed = False

    def read_region(self, spec, *, cancellation_token=None):
        self.reads.append(spec)
        return np.array(read_index_spec(self.values, spec), copy=True)

    def validate_snapshot(self):
        if self.changed:
            raise ValueError("source epoch changed")

    def close(self):
        pass


def request_for(tmp_path, source, operations=(), ranges=None, squeeze=False):
    document = ArrayDocument(LazySourceArray(source), operations=operations)
    if ranges is None:
        ranges = tuple((0, size) for size in document.current_shape)
    return RegionExportRequest(document, tmp_path / "result.npy", ranges, squeeze=squeeze)


@pytest.mark.parametrize("reverse", [False, True])
def test_crop_reverse_complex_chunks_match_independent_numpy_and_read_each_sample_once(
    tmp_path, reverse
):
    values = np.arange(23 * 7 * 2).reshape(23, 7, 2).astype(np.complex128)
    values += 1j * values[::-1]
    source = CountedSource(values)
    operations = (Crop(0, 2, 22), Scale(2)) + ((ReverseAxis(0),) if reverse else ())
    request = request_for(tmp_path, source, operations, ((1, 20), (2, 6), (1, 2)), squeeze=True)
    expected = values[2:22] * 2
    if reverse:
        expected = expected[::-1]
    expected = expected[1:20, 2:6, 1]

    prepared = plan_streaming_export(request, chunk_target_bytes=768)
    assert source.reads == []
    assert prepared.chunk_count > 2
    assert prepared.chunk_shape[1:] == expected.shape[1:]
    assert prepared.estimated_peak_bytes <= 768
    result = export_streaming_region(request, prepared=prepared)

    np.testing.assert_array_equal(np.load(result.paths[0]), expected)
    assert result.shape == expected.shape
    touched = []
    for spec in source.reads:
        rows = list(range(*spec[0].indices(values.shape[0])))
        touched.extend(rows)
        assert len(rows) <= prepared.chunk_shape[0]
        assert all(2 <= column < 6 for column in range(*spec[1].indices(7)))
        assert list(range(*spec[2].indices(2))) == [1]
    assert len(touched) == len(set(touched)) == 19
    assert prepared.total_source_read_bytes == expected.nbytes
    assert len(source.reads) == prepared.chunk_count


def test_nonstream_odd_fft_rank_changes_and_reduction_match_numpy(tmp_path):
    values = np.arange(18 * 7 * 4, dtype=np.float64).reshape(18, 1, 7, 4)
    source = CountedSource(values)
    operations = (Crop(0, 1, 18), CenteredFFT(2), Mean(3), Squeeze(1))
    request = request_for(tmp_path, source, operations, ((0, 17), (1, 6)))
    expected = np.fft.ifftshift(
        np.fft.ifft(np.fft.fftshift(values[1:18], axes=2), axis=2, norm="ortho"),
        axes=2,
    ).mean(axis=3)[:, 0, 1:6]

    prepared = plan_streaming_export(request, chunk_target_bytes=8192)
    assert prepared.chunk_count > 1
    assert source.reads == []
    exported = export_streaming_region(request, prepared=prepared)

    np.testing.assert_allclose(np.load(exported.paths[0]), expected, rtol=1e-12, atol=1e-12)
    assert (
        sum(prod(np.shape(read_index_spec(values, spec))) * 8 for spec in source.reads)
        == prepared.total_source_read_bytes
    )
    assert prepared.total_source_read_bytes == 17 * 7 * 4 * 8


@pytest.mark.parametrize(
    ("shape", "squeeze", "target", "expected_shape", "chunk_count"),
    [
        ((), True, 16, (), 1),
        ((1, 1, 1), True, 16, (), 1),
        ((1, 1, 1), False, 16, (1, 1, 1), 1),
        ((5, 1, 1), True, 32, (5,), 3),
        ((0, 3), True, 1, (0, 3), 0),
        ((1, 0, 1), True, 1, (0,), 0),
    ],
)
def test_scalar_empty_and_last_singleton_chunk_keep_global_shape(
    tmp_path, shape, squeeze, target, expected_shape, chunk_count
):
    values = np.arange(prod(shape), dtype=np.float64).reshape(shape)
    source = CountedSource(values)
    request = request_for(tmp_path, source, squeeze=squeeze)
    prepared = plan_streaming_export(request, chunk_target_bytes=target)
    assert prepared.output_shape == expected_shape
    assert prepared.chunk_count == chunk_count

    result = export_streaming_region(request, prepared=prepared)

    np.testing.assert_array_equal(
        np.load(result.paths[0]), np.squeeze(values) if squeeze else values
    )
    assert len(source.reads) == chunk_count
    assert result.shape == expected_shape


@pytest.mark.parametrize("operation", [CenteredFFT(0), Transpose(1, 2)])
def test_full_source_repetition_is_refused_before_any_payload_or_output(tmp_path, operation):
    source = CountedSource(np.arange(32 * 7 * 3, dtype=np.float64).reshape(32, 7, 3))
    request = request_for(tmp_path, source, (operation,))
    with pytest.raises(StreamingExportRefused, match=r"working memory|overlap"):
        plan_streaming_export(request, chunk_target_bytes=8192)
    assert source.reads == []
    assert list(tmp_path.iterdir()) == []


def test_single_chunk_permutation_remains_exact_without_repeated_source_work(tmp_path):
    values = np.arange(4 * 3 * 2, dtype=np.int16).reshape(4, 3, 2)
    source = CountedSource(values)
    request = request_for(tmp_path, source, (Transpose(0, 2),))
    prepared = plan_streaming_export(request, chunk_target_bytes=1024)
    assert prepared.chunk_count == 1
    export_streaming_region(request, prepared=prepared)
    np.testing.assert_array_equal(np.load(request.path), values.swapaxes(0, 2))
    assert len(source.reads) == 1


def test_unregistered_operation_is_refused_without_cost_or_execution_probe(tmp_path):
    class UnknownOperation:
        def output_shape(self, shape):
            return shape

        def output_dtype(self, dtype):
            return dtype

        def capabilities(self, *_args):
            raise AssertionError("unsupported operation cost was probed")

    source = CountedSource(np.zeros((4, 4)))
    request = request_for(tmp_path, source, (UnknownOperation(),))
    with pytest.raises(StreamingExportRefused, match="built-in metadata-only"):
        plan_streaming_export(request)
    assert source.reads == []
    assert not request.path.exists()


@pytest.mark.parametrize("estimate", [None, 4096])
def test_decoded_source_working_cost_is_enforced_metadata_only(tmp_path, estimate):
    source = CountedSource(np.zeros((16, 8), dtype=np.float64))
    source.estimate_read_nbytes = lambda _spec: estimate
    request = request_for(tmp_path, source)
    with pytest.raises(StreamingExportRefused, match=r"known.*cost|working memory"):
        plan_streaming_export(request, chunk_target_bytes=1024)
    assert source.reads == []
    assert list(tmp_path.iterdir()) == []


def test_later_overlapping_chunk_is_refused_before_its_io_and_preserves_old_output(
    tmp_path, monkeypatch
):
    source = CountedSource(np.arange(16 * 4).reshape(16, 4))
    request = request_for(tmp_path, source)
    request.path.write_bytes(b"previous output")
    prepared = plan_streaming_export(request, chunk_target_bytes=128)
    original = streaming_export._plan_chunk

    def corrupted_demand(request, axis, offset, length):
        plan, shape = original(request, axis, offset, length)
        if offset >= 2 * length:
            plan = replace(
                plan,
                required_input_region=region_from_index_spec(
                    source.shape,
                    (slice(0, length), slice(None)),
                ),
            )
        return plan, shape

    monkeypatch.setattr(streaming_export, "_plan_chunk", corrupted_demand)
    with pytest.raises(StreamingExportRefused, match="overlap"):
        export_streaming_region(request, prepared=prepared)
    assert len(source.reads) == 2
    assert request.path.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [request.path]


@pytest.mark.parametrize("failure", ["cancel", "stale-middle", "stale-final"])
def test_cancel_or_changed_source_never_publishes_a_partial_file(tmp_path, failure):
    source = CountedSource(np.arange(16 * 4).reshape(16, 4))
    request = request_for(tmp_path, source)
    request.path.write_bytes(b"previous output")
    prepared = plan_streaming_export(request, chunk_target_bytes=128)
    token = SimpleNamespace(cancelled=False)

    def progress(event):
        threshold = prepared.chunk_count if failure == "stale-final" else 1
        if event.phase == "writing" and len(source.reads) >= threshold:
            if failure == "cancel":
                token.cancelled = True
            else:
                source.changed = True

    with pytest.raises(EvaluationCancelled if failure == "cancel" else RegionExportChanged):
        export_streaming_region(
            request, prepared=prepared, cancellation_token=token, progress_callback=progress
        )
    assert request.path.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [request.path]


def test_observer_or_cancel_after_commit_does_not_report_failed_export(tmp_path):
    source = CountedSource(np.arange(4 * 4).reshape(4, 4))
    request = request_for(tmp_path, source)
    token = SimpleNamespace(cancelled=False)

    def progress(event):
        if event.phase == "complete":
            token.cancelled = True
            raise RuntimeError("observer failed after publication")

    result = export_streaming_region(
        request, chunk_target_bytes=128, cancellation_token=token, progress_callback=progress
    )
    np.testing.assert_array_equal(np.load(result.paths[0]), source.values)


def test_changing_metadata_cost_cannot_trap_minimum_chunk_backoff(tmp_path):
    source = CountedSource(np.zeros((16, 4), dtype=np.float64))
    calls = 0

    def changing_estimate(_spec):
        nonlocal calls
        calls += 1
        if calls > 6:
            raise RuntimeError("minimum-chunk backoff did not terminate")
        return 32 if calls == 1 else 4096

    source.estimate_read_nbytes = changing_estimate
    request = request_for(tmp_path, source)
    with pytest.raises(StreamingExportRefused, match=r"cost changed|working"):
        plan_streaming_export(request, chunk_target_bytes=128)
    assert calls <= 3
    assert source.reads == []
    assert list(tmp_path.iterdir()) == []


def test_streaming_keeps_the_shared_per_read_hard_budget(tmp_path):
    class HugeSource(CountedSource):
        def __init__(self):
            self.shape = (2, DEFAULT_SOURCE_READ_BUDGET_BYTES // 8 + 1)
            self.dtype = np.dtype("float64")
            self.nbytes = prod(self.shape) * self.dtype.itemsize
            self.reads = []
            self.changed = False

        def read_region(self, _spec, *, cancellation_token=None):
            raise AssertionError("an oversized source read reached I/O")

    source = HugeSource()
    request = request_for(tmp_path, source)
    request.path.write_bytes(b"previous output")
    prepared = plan_streaming_export(
        request, chunk_target_bytes=4 * DEFAULT_SOURCE_READ_BUDGET_BYTES
    )
    with pytest.raises(SourceReadRefused) as caught:
        export_streaming_region(request, prepared=prepared)
    assert caught.value.requested_nbytes == source.shape[1] * 8
    assert source.reads == []
    assert request.path.read_bytes() == b"previous output"
    assert list(tmp_path.iterdir()) == [request.path]


def test_fixed_codec_overhead_does_not_force_tiny_repeated_reads(tmp_path):
    source = CountedSource(np.zeros((512, 512), dtype=np.float64))

    def estimate(spec):
        rows = len(range(*spec[0].indices(512)))
        columns = len(range(*spec[1].indices(512)))
        return rows * columns * 8 + 8 * 1024**2

    source.estimate_read_nbytes = estimate
    request = request_for(tmp_path, source)
    prepared = plan_streaming_export(request, chunk_target_bytes=64 * 1024**2)
    assert prepared.chunk_count == 1
    assert prepared.estimated_peak_bytes == 12 * 1024**2
    assert prepared.total_source_read_bytes == source.nbytes
    assert source.reads == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", [FFTShift(0), SplitAxis(0, 2)])
def test_large_cyclic_or_fancy_maps_are_refused_before_coordinate_arrays(
    tmp_path, monkeypatch, operation
):
    class MetadataSource:
        shape = (100_000_000, 2)
        dtype = np.dtype("float32")
        nbytes = prod(shape) * dtype.itemsize

        def read_region(self, *_args, **_kwargs):
            raise AssertionError("unsupported mapping read source pixels")

    document = ArrayDocument(MetadataSource(), operations=(operation,))
    request = RegionExportRequest(
        document,
        tmp_path / "result.npy",
        tuple((0, size) for size in document.current_shape),
    )

    def refuse_coordinate_array(*_args, **_kwargs):
        raise AssertionError("unsupported mapping allocated coordinate metadata")

    monkeypatch.setattr(np, "arange", refuse_coordinate_array)
    with pytest.raises(StreamingExportRefused, match="coordinate mapping"):
        plan_streaming_export(request)
    assert list(tmp_path.iterdir()) == []
