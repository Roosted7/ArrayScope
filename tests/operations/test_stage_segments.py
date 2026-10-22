"""Per-stage shared-memory segments: the lifecycle, and who owns the unlink.

A stage segment is the largest thing this process ever shares — the canonical
volume is 234 MB once an FFT prefix has run — so the interesting claims are all
about its *end*. It must unlink when its cache entry evicts, when the cache is
cleared, when the value dies with a closed window, and at shutdown; and once
unlinked it must be unreachable, not merely forgotten. The one claim about its
middle is that a real child process reads the parent's bytes and cannot write
them.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from arrayscope.core.shared_memory import attach_shared_memory
from arrayscope.core.view_state import ViewState
from arrayscope.kernel.process_backend import ProcessPool
from arrayscope.kernel.remote_call import RemoteCall
from arrayscope.operations import source_transport, stage_segments
from arrayscope.operations.evaluator import (
    OperationEvaluator,
    evaluate_image_snapshot,
    stage_document_key,
)
from arrayscope.operations.pipeline import ArrayDocument, CenteredFFT, CenteredIFFT, FFTShift
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec, StageKey
from arrayscope.operations.stage_segments import (
    AttachedStageCache,
    SegmentUnavailable,
    publish_stage_value,
    published_bytes,
    published_count,
    release_all_stage_segments,
    segment_for_value,
)
from tests.kernel import remote_entries


@pytest.fixture(scope="module")
def pool():
    pool = ProcessPool(processes=1).acquire()
    yield pool
    pool.release()


@pytest.fixture(autouse=True)
def publishing(monkeypatch):
    """What ``window/main.py`` turns on when it chooses the process backend."""

    monkeypatch.setattr(source_transport, "_publication_enabled", True)
    yield
    stage_segments.close_all_attachments()
    release_all_stage_segments()


def _fft_document(extent: int = 12):
    values = np.arange(extent * extent * 4, dtype=np.float32).reshape(extent, extent, 4)
    return ArrayDocument(
        values,
        operations=(CenteredFFT(axis=2), FFTShift(axis=2), CenteredIFFT(axis=2)),
    )


def _stage_a_value(document, *, cache=None):
    """Run one real evaluation so a real stage key and value land in the cache."""

    cache = OperationEvaluator(document).stage_cache if cache is None else cache
    view_state = ViewState.from_shape(document.current_shape).with_image_axes(0, 1)
    evaluate_image_snapshot(
        document,
        view_state,
        shader_display=True,
        provisional_histogram=True,
        stage_cache=cache,
        stage_document_key=stage_document_key(document),
    )
    resident = cache.resident_items()
    assert len(resident) == 1, "the FFT chain is expected to retain exactly one stage"
    return (cache, *resident[0])


def _segment_exists(name: str) -> bool:
    try:
        segment = attach_shared_memory(name)
    except FileNotFoundError:
        return False
    segment.close()
    return True


# ------------------------------------------------------------- publication


def test_storing_a_stage_publishes_it_and_the_segment_holds_the_same_bytes():
    cache, _key, value = _stage_a_value(_fft_document())

    segment = segment_for_value(value)

    assert segment is not None
    assert published_count() == 1
    assert published_bytes() == int(value.data.nbytes)
    np.testing.assert_array_equal(stage_segments.attach_stage_array(segment), value.data)
    assert segment.stage_index == int(value.stage_index)
    assert cache.stores == 1


def test_nothing_is_published_until_the_process_decides_to_run_a_pool(monkeypatch):
    """A threads-only viewer would double every stage and read none of the copies."""

    monkeypatch.setattr(source_transport, "_publication_enabled", False)

    _cache, _key, value = _stage_a_value(_fft_document())

    assert segment_for_value(value) is None
    assert published_count() == 0


def test_a_stage_larger_than_the_caches_envelope_is_not_published():
    """The segment budget is the cache's, kept in sync by evicting together."""

    _cache, _key, value = _stage_a_value(_fft_document())
    assert segment_for_value(value) is not None

    key = StageKey(
        document_key=("starved",),
        operation_prefix=(),
        region=value.region,
        dtype=str(value.data.dtype),
        shape=tuple(value.data.shape),
    )

    # A second publication of the same bytes under a cache that could not hold
    # them: the refusal is the publisher's, and it costs nothing but a copy
    # that never happened.
    assert publish_stage_value(key, value, budget_bytes=int(value.data.nbytes) - 1) is None
    assert published_count() == 1


# ---------------------------------------------------------------- the end


def test_evicting_the_entry_unlinks_its_segment():
    cache, _key, value = _stage_a_value(_fft_document())
    name = segment_for_value(value).name
    assert _segment_exists(name)

    # A budget that no longer holds what is in it: the resize evicts, and the
    # eviction hook is what keeps published bytes inside cached bytes.
    cache.resize(max_bytes=int(value.nbytes) - 1)

    assert cache.evictions == 1
    assert segment_for_value(value) is None
    assert published_bytes() == 0
    assert not _segment_exists(name)


def test_clearing_the_cache_unlinks_everything_it_published():
    """``BoundedCache.clear`` runs no eviction hook, so the release is by hand."""

    cache, _key, value = _stage_a_value(_fft_document())
    name = segment_for_value(value).name

    cache.clear()

    assert published_count() == 0
    assert not _segment_exists(name)


def test_a_stage_value_nobody_holds_any_more_releases_its_own_segment():
    """A closed window's cache is dropped, not cleared: nothing ever evicts.

    ``gc.collect()`` is not test decoration — a ``StageCache`` holds its own
    ``retention_key`` closure and eviction hook, so it is a cycle and dies on a
    collection, never on the last reference. The segment must not wait for the
    process to exit either way, which is what the weakref release is for.
    """

    cache, _key, value = _stage_a_value(_fft_document())
    name = segment_for_value(value).name

    del cache, value
    gc.collect()

    assert published_count() == 0
    assert not _segment_exists(name)


def test_shutdown_releases_every_segment_and_a_released_one_cannot_be_reattached():
    """The killed-child case too: the name is gone, not merely unregistered."""

    _cache, _key, value = _stage_a_value(_fft_document())
    segment = segment_for_value(value)

    release_all_stage_segments()

    assert published_count() == 0
    assert published_bytes() == 0
    assert not _segment_exists(segment.name)
    with pytest.raises(SegmentUnavailable):
        stage_segments.attach_stage_array(segment)


# ------------------------------------------------------------- the middle


def test_a_child_process_reads_the_published_stage_and_cannot_write_it(pool):
    _cache, _key, value = _stage_a_value(_fft_document())
    segment = segment_for_value(value)

    seen = pool.run(RemoteCall(remote_entries.__name__ + ":stage_segment_probe", segment))

    assert seen["pid"] != remote_entries.own_pid(None)
    assert seen["shape"] == tuple(value.data.shape)
    assert seen["dtype"] == str(value.data.dtype)
    assert seen["checksum"] == pytest.approx(complex(value.data.sum()))
    assert seen["writeable"] is False


# ------------------------------------------------------ the child's cache


def _key_for(value, *, stage_index: int, shape=None, region=None):
    return StageKey(
        document_key=("ignored-by-the-attached-cache",),
        operation_prefix=tuple(range(int(stage_index))),
        region=value.region if region is None else region,
        dtype=str(value.data.dtype),
        shape=tuple(value.data.shape) if shape is None else tuple(shape),
    )


def test_the_attached_cache_serves_the_stage_it_was_given_and_nothing_else():
    _cache, key, value = _stage_a_value(_fft_document())
    attached = AttachedStageCache((segment_for_value(value),), max_bytes=1 << 30)

    served = attached.get_containing(_key_for(value, stage_index=len(key.operation_prefix)))

    assert served is not None
    np.testing.assert_array_equal(served.data, value.data)
    assert served.region == value.region
    # A different stage of the same shape and dtype is a different array: the
    # FFT and the shift of it are indistinguishable by everything but position.
    assert attached.get_containing(_key_for(value, stage_index=1)) is None
    assert attached.put(key, value) is False


def test_the_attached_cache_answers_containment_the_way_the_real_one_does():
    _cache, key, value = _stage_a_value(_fft_document())
    attached = AttachedStageCache((segment_for_value(value),), max_bytes=1 << 30)
    inner = RegionSpec((AxisRegion(AxisRegionKind.POINT, 0), *value.region.axes[1:]))

    served = attached.get_containing(
        _key_for(value, stage_index=len(key.operation_prefix), region=inner)
    )

    assert served is not None
    # It hands back the CONTAINING value, exactly as ``StageCache`` does; the
    # evaluation cuts its own subregion out of it.
    assert served.region == value.region
