import re
from pathlib import Path

import numpy as np
import pytest

from arrayscope.core.array_source import (
    DEFAULT_SOURCE_READ_BUDGET_BYTES,
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
    evaluate_slab,
    request_for_image,
    request_for_line,
    request_for_scalar,
)
from arrayscope.operations.source_read import read_base_region, source_read_budget_bytes
from arrayscope.operations.stage_cache import StageCache


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
        text = path.read_text()
        if re.search(r"apply_region\(\s*(document|self)\.base_data", text):
            offenders.append(str(path))
        if re.search(r"(document|self)\.base_data\[", text):
            offenders.append(str(path))
    assert offenders == []
