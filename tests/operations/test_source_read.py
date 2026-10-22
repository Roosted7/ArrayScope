import re
from pathlib import Path

import numpy as np
import pytest

from arrayscope.core.array_source import (
    DEFAULT_SOURCE_MATERIALIZE_BUDGET_BYTES,
    DEFAULT_SOURCE_READ_BUDGET_BYTES,
    ArraySource,
    CompositeArraySource,
    LazySourceArray,
    NdArraySource,
    SourceReadRefused,
)
from arrayscope.core.compute_policy import ComputeLane
from arrayscope.core.view_state import ViewState
from arrayscope.operations.cancellation import EvaluationCancelled
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, Crop
from arrayscope.operations.regions import apply_region, region_from_index_spec
from arrayscope.operations.slabs import (
    evaluate_slab_from_plan,
    plan_slab,
    request_for_image,
    request_for_line,
    request_for_scalar,
)
from arrayscope.operations.source_read import read_base_region, source_read_budget_bytes
from arrayscope.operations.stage_cache import StageCache


def evaluate_slab(document, request, **kwargs):
    """Plan then evaluate -- the two steps every production caller takes."""

    return evaluate_slab_from_plan(document, request, plan_slab(document, request), **kwargs)


class _Token:
    def __init__(self, cancelled=False):
        self.cancelled = cancelled


class _CountingSource(NdArraySource):
    def __init__(self, array, **kwargs):
        super().__init__(array, **kwargs)
        self.read_specs = []

    def read_region(self, index_spec, *, cancellation_token=None):
        self.read_specs.append(tuple(index_spec))
        return super().read_region(index_spec, cancellation_token=cancellation_token)


class _WorkingSource(_CountingSource):
    def __init__(self, array, working_bytes):
        super().__init__(array)
        self.working_bytes = working_bytes
        self.estimates = []

    def estimate_read_nbytes(self, index_spec):
        self.estimates.append(tuple(index_spec))
        return self.working_bytes


def _lazy(data):
    return LazySourceArray(_CountingSource(data))


def test_read_base_region_on_plain_array_matches_apply_region():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    region = region_from_index_spec(data.shape, (1, slice(None), (0, 2, 4)))

    assert np.array_equal(read_base_region(data, region), apply_region(data, region))


def test_read_base_region_on_lazy_source_matches_eager():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    lazy = _lazy(data)
    region = region_from_index_spec(data.shape, (slice(1, 3), 2, slice(None)))

    result = read_base_region(lazy, region)

    assert np.array_equal(result, apply_region(data, region))
    assert lazy.source.read_specs == [(slice(1, 3, 1), 2, slice(None))]


def test_read_base_region_refuses_over_budget_reads():
    data = np.zeros((64, 64), dtype=np.float64)
    lazy = _lazy(data)
    region = region_from_index_spec(data.shape, (slice(None), slice(None)))

    with pytest.raises(SourceReadRefused) as excinfo:
        read_base_region(lazy, region, budget_bytes=128)

    assert excinfo.value.requested_nbytes == data.nbytes
    assert excinfo.value.budget_bytes == 128
    assert lazy.source.read_specs == []


def test_decoded_chunk_working_set_is_guarded_before_any_source_read():
    source = _WorkingSource(np.zeros((8, 8), dtype=np.float32), working_bytes=4096)
    lazy = LazySourceArray(source)
    region = region_from_index_spec(lazy.shape, (slice(1, 2), slice(2, 3)))

    with pytest.raises(SourceReadRefused) as refused:
        read_base_region(lazy, region, budget_bytes=128)

    assert refused.value.requested_nbytes == 4096
    assert source.estimates == [(slice(1, 2, 1), slice(2, 3, 1))]
    assert source.read_specs == []


def test_unknown_advertised_working_set_is_refused_before_any_source_read():
    source = _WorkingSource(np.zeros((8, 8), dtype=np.float32), working_bytes=None)
    lazy = LazySourceArray(source)
    region = region_from_index_spec(lazy.shape, (slice(1, 2), slice(2, 3)))

    with pytest.raises(SourceReadRefused, match="unknown") as refused:
        read_base_region(lazy, region, budget_bytes=128)

    assert refused.value.requested_nbytes is None
    assert source.read_specs == []


def test_export_preflight_prices_source_working_set_without_changing_logical_bytes(tmp_path):
    from arrayscope.io.region_export import RegionExportRequest, plan_region_export

    source = _WorkingSource(np.zeros((8, 8), dtype=np.float32), working_bytes=4096)
    request = RegionExportRequest(
        ArrayDocument(LazySourceArray(source)), tmp_path / "selection.npy", ((1, 2), (2, 3))
    )

    plan = plan_region_export(request)

    assert plan.source_read_bytes == 4
    assert plan.output_bytes == 4
    assert plan.estimated_peak_bytes == 4096
    assert source.read_specs == []


def test_legacy_source_protocol_and_logical_budget_fallback_stay_available():
    source = _CountingSource(np.zeros((8, 8), dtype=np.float32))
    lazy = LazySourceArray(source)
    region = region_from_index_spec(lazy.shape, (slice(1, 2), slice(2, 3)))

    assert isinstance(source, ArraySource)
    assert not hasattr(lazy, "estimate_read_nbytes")
    np.testing.assert_array_equal(read_base_region(lazy, region, budget_bytes=4), [[0]])
    assert len(source.read_specs) == 1


def test_composite_source_cannot_hide_unknown_or_oversized_child_working_sets():
    left = _WorkingSource(np.zeros((8, 8), dtype=np.float32), working_bytes=4096)
    right = _WorkingSource(np.ones((8, 8), dtype=np.float32), working_bytes=8192)
    source = LazySourceArray(CompositeArraySource(LazySourceArray(left), LazySourceArray(right)))
    region = region_from_index_spec(source.shape, (slice(1, 2), slice(2, 3)))

    with pytest.raises(SourceReadRefused) as refused:
        read_base_region(source, region, budget_bytes=8192)

    assert refused.value.requested_nbytes == 8192 + 4
    assert left.read_specs == right.read_specs == []
    right.working_bytes = None
    with pytest.raises(SourceReadRefused, match="unknown"):
        read_base_region(source, region, budget_bytes=16384)
    assert left.read_specs == right.read_specs == []


def test_unknown_read_working_set_stays_unknown_through_transform_preflight(tmp_path):
    from arrayscope.io.region_export import RegionExportRequest, plan_region_export

    source = _WorkingSource(np.zeros((8, 8), dtype=np.float32), working_bytes=None)
    request = RegionExportRequest(
        ArrayDocument(LazySourceArray(source), operations=(CenteredFFT(1),)),
        tmp_path / "unknown.npy",
        ((1, 2), (2, 3)),
    )

    plan = plan_region_export(request)

    assert plan.estimated_peak_bytes is None
    assert plan.needs_confirmation
    assert source.read_specs == []


@pytest.mark.parametrize("working_bytes", [0, -1, 1.5])
def test_source_estimate_cannot_undercut_result_bytes_or_report_invalid_counts(working_bytes):
    source = _WorkingSource(np.zeros((8, 8), dtype=np.float32), working_bytes=working_bytes)
    lazy = LazySourceArray(source)
    region = region_from_index_spec(lazy.shape, (slice(1, 2), slice(2, 3)))

    if working_bytes == 0:
        with pytest.raises(SourceReadRefused) as refused:
            read_base_region(lazy, region, budget_bytes=3)
        assert refused.value.requested_nbytes == 4
    else:
        with pytest.raises(ValueError, match="nonnegative integer"):
            read_base_region(lazy, region, budget_bytes=128)
    assert source.read_specs == []


@pytest.mark.parametrize("working_bytes", [None, DEFAULT_SOURCE_MATERIALIZE_BUDGET_BYTES + 1])
def test_implicit_full_materialization_uses_source_working_guard_before_read(working_bytes):
    source = _WorkingSource(np.zeros((2, 3), dtype=np.float32), working_bytes=working_bytes)
    lazy = LazySourceArray(source)

    with pytest.raises(SourceReadRefused):
        np.asarray(lazy)

    assert source.read_specs == []


def test_full_materialization_preserves_convenience_cap_and_explicit_unlimited_semantics():
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = _WorkingSource(values, working_bytes=128)
    lazy = LazySourceArray(source)

    with pytest.raises(SourceReadRefused):
        lazy.materialize(budget_bytes=64)
    assert source.read_specs == []
    np.testing.assert_array_equal(lazy.materialize(budget_bytes=128), values)
    source.working_bytes = None
    np.testing.assert_array_equal(lazy.materialize(budget_bytes=None), values)
    assert len(source.read_specs) == 2


def test_read_base_region_checks_cancellation_before_reading():
    data = np.zeros((8, 8), dtype=np.float32)
    lazy = _lazy(data)
    region = region_from_index_spec(data.shape, (slice(None), slice(None)))

    with pytest.raises(EvaluationCancelled):
        read_base_region(lazy, region, cancellation_token=_Token(cancelled=True))

    assert lazy.source.read_specs == []


class _Policy:
    visible_render_budget_bytes = 3 * DEFAULT_SOURCE_READ_BUDGET_BYTES
    prefetch_budget_bytes = 2 * DEFAULT_SOURCE_READ_BUDGET_BYTES


class _Context:
    def __init__(self, lane, policy=None):
        self.lane = lane
        self.memory_policy = _Policy() if policy is None else policy


def test_source_read_budget_is_lane_aware():
    assert source_read_budget_bytes(None) == DEFAULT_SOURCE_READ_BUDGET_BYTES
    assert (
        source_read_budget_bytes(_Context(ComputeLane.VISIBLE))
        == _Policy.visible_render_budget_bytes
    )
    assert source_read_budget_bytes(_Context(ComputeLane.PREFETCH)) == _Policy.prefetch_budget_bytes


def test_source_read_budget_floors_at_module_default():
    class SmallPolicy:
        visible_render_budget_bytes = 1024
        prefetch_budget_bytes = 1024

    assert (
        source_read_budget_bytes(_Context(ComputeLane.VISIBLE, SmallPolicy()))
        == DEFAULT_SOURCE_READ_BUDGET_BYTES
    )


def test_evaluate_slab_image_line_scalar_match_eager_document():
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    eager = ArrayDocument(data, operations=(Crop(axis=2, start=1, stop=5), CenteredFFT(axis=1)))
    lazy = ArrayDocument(
        _lazy(data), operations=(Crop(axis=2, start=1, stop=5), CenteredFFT(axis=1))
    )
    state = ViewState.from_shape(eager.current_shape)

    for request_factory in (request_for_image, request_for_line):
        request = request_factory(state)
        assert np.allclose(evaluate_slab(lazy, request), evaluate_slab(eager, request))

    scalar_request = request_for_scalar(state, (0,) * len(eager.current_shape))
    assert np.allclose(evaluate_slab(lazy, scalar_request), evaluate_slab(eager, scalar_request))


def test_evaluate_slab_with_stage_cache_matches_eager_document():
    data = np.arange(6 * 8 * 4, dtype=np.float32).reshape(6, 8, 4)
    operations = (CenteredFFT(axis=1),)
    eager = ArrayDocument(data, operations=operations)
    lazy = ArrayDocument(_lazy(data), operations=operations)
    state = ViewState.from_shape(eager.current_shape)
    request = request_for_image(state)

    cache = StageCache(max_bytes=16 * 1024 * 1024)
    first = evaluate_slab(lazy, request, stage_cache=cache, document_key=("doc",))
    second = evaluate_slab(lazy, request, stage_cache=cache, document_key=("doc",))

    assert np.allclose(first, evaluate_slab(eager, request))
    assert np.allclose(second, first)


def test_evaluate_slab_propagates_read_refusal(monkeypatch):
    import arrayscope.operations.source_read as source_read_module

    data = np.zeros((32, 32), dtype=np.float64)
    lazy = ArrayDocument(_lazy(data))
    state = ViewState.from_shape(lazy.current_shape)

    monkeypatch.setattr(source_read_module, "source_read_budget_bytes", lambda context: 16)
    with pytest.raises(SourceReadRefused):
        evaluate_slab(lazy, request_for_image(state))


def test_base_data_reads_go_through_the_source_seam():
    """Guard: evaluation must not index document base data around the budgeted seam."""

    package_root = Path(__file__).resolve().parents[2] / "arrayscope"
    offenders = []
    for path in package_root.rglob("*.py"):
        if path.name == "source_read.py":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"apply_region\(\s*(document|self)\.base_data", text):
            offenders.append(str(path))
        if re.search(r"(document|self)\.base_data\[", text):
            offenders.append(str(path))
    assert offenders == []
