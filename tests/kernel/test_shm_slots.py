"""The shared-memory slot pool's lifecycle, in one process.

``FREE -> LEASED -> READY -> HELD -> FREE`` is the whole contract: one writer
per slot, immutable from READY, and exhaustion that degrades to pickling
rather than failing. Every illegal transition is refused loudly, because a
silent one would hand two producers the same bytes.
"""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass

import numpy as np
import pytest

from arrayscope.kernel.shm_slots import (
    SLOT_PAYLOAD_MIN_BYTES,
    SlotPool,
    slottable,
    unwrap_payload,
    wrap_payload,
)


@pytest.fixture
def pool():
    pool = SlotPool(slot_bytes=1 << 18, slots=4, context=multiprocessing.get_context("spawn"))
    yield pool
    pool.close()


def big_array(count: int = 40_000) -> np.ndarray:
    return np.arange(count, dtype=np.float32)


def test_lease_publish_take_release_round_trip(pool):
    array = np.arange(64, dtype=np.int32)
    lease = pool.lease()

    descriptor = lease.publish(array)
    taken = pool.take(descriptor)

    assert descriptor.shape == (64,)
    assert descriptor.nbytes == array.nbytes
    assert np.array_equal(taken, array)
    assert not taken.flags.writeable
    del taken
    pool.release(descriptor)


def test_a_slot_is_immutable_between_ready_and_release(pool):
    lease = pool.lease()
    descriptor = lease.publish(np.arange(4, dtype=np.int8))

    with pytest.raises(RuntimeError, match="ready"):
        lease.publish(np.arange(4, dtype=np.int8))
    pool.take(descriptor)
    with pytest.raises(RuntimeError, match="held"):
        pool.take(descriptor)
    pool.release(descriptor)


def test_taking_before_publication_is_refused(pool):
    from arrayscope.kernel.shm_slots import SlotDescriptor

    lease = pool.lease()

    with pytest.raises(RuntimeError, match="leased"):
        pool.take(SlotDescriptor(index=lease.index, nbytes=4, shape=(1,), dtype="<i4"))
    lease.release()


def test_one_writer_per_slot_and_exhaustion_counts_a_fallback(pool):
    leases = [pool.lease() for _ in range(4)]

    assert sorted(lease.index for lease in leases) == [0, 1, 2, 3]
    assert pool.lease() is None
    assert pool.fallbacks == 1
    assert pool.lease() is None
    assert pool.fallbacks == 2
    for lease in leases:
        lease.release()
    assert pool.lease() is not None


def test_released_slots_are_reused_around_the_wrap(pool):
    seen = []
    for _ in range(10):
        lease = pool.lease()
        seen.append(lease.index)
        lease.release()

    assert seen == [index % 4 for index in range(10)]
    assert pool.fallbacks == 0


def test_reclaim_frees_only_one_owner(pool):
    mine = pool.lease(owner=1)
    theirs = pool.lease(owner=2)
    pool.lease(owner=1)

    assert pool.reclaim(1) == 2
    assert pool.reclaim(0) == 0
    theirs.release()
    mine.release()


def test_large_arrays_travel_by_descriptor_and_small_ones_do_not(pool):
    payload = {"big": big_array(), "small": np.arange(4, dtype=np.float32)}
    assert slottable(payload["big"])
    assert not slottable(payload["small"])

    wrapped, leases = wrap_payload(payload, pool)

    assert len(leases) == 1
    assert wrapped["small"] is payload["small"]
    restored, taken = unwrap_payload(wrapped, pool, copy=True)
    assert np.array_equal(restored["big"], payload["big"])
    for index in taken:
        pool.release(index)
    for lease in leases:
        lease.release()


def test_a_dataclass_field_is_rewritten_in_place(pool):
    from arrayscope.display.histogram_plot import HistogramPlotRequest

    request = HistogramPlotRequest(
        data=big_array(),
        source_identity=("id", (40_000,), "float32"),
        histogram_bounds=None,
        visible_value_span=None,
        pixel_extent=100.0,
    )

    wrapped, leases = wrap_payload(request, pool)

    assert len(leases) == 1
    assert not isinstance(wrapped.data, np.ndarray)
    assert wrapped.source_identity == request.source_identity
    restored, taken = unwrap_payload(wrapped, pool, copy=True)
    assert np.array_equal(restored.data, request.data)
    for index in taken:
        pool.release(index)
    for lease in leases:
        lease.release()


def test_an_exhausted_pool_leaves_the_payload_intact(pool):
    held = [pool.lease() for _ in range(4)]
    payload = (big_array(), big_array())

    wrapped, leases = wrap_payload(payload, pool)

    assert leases == ()
    assert pool.fallbacks == 2
    assert all(item is original for item, original in zip(wrapped, payload, strict=True))
    for lease in held:
        lease.release()


def test_a_payload_over_the_slot_size_stays_pickled(pool):
    payload = np.zeros((1 << 18) + 8, dtype=np.uint8)
    assert payload.nbytes > SLOT_PAYLOAD_MIN_BYTES

    wrapped, leases = wrap_payload(payload, pool)

    assert wrapped is payload
    assert leases == ()
    # And it is a counted fallback, not a silent one: a family whose planes
    # outgrow the slot class must show up in the same diagnostic the
    # exhaustion path feeds, or the pool reads as keeping up while every
    # payload pickles.
    assert pool.fallbacks == 1
    assert all(state == 0 for state in pool._states)


@dataclass(frozen=True)
class _ValidatedArray:
    data: np.ndarray

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be an ndarray")


@dataclass(frozen=True)
class _Payload:
    data: object


@pytest.mark.parametrize("nested", [False, True])
def test_a_dataclass_fallback_releases_only_its_own_slots(pool, nested):
    # The earlier payload and an unrelated writer must retain their leases;
    # a rejected nested value must not exhaust the remaining slots either.
    held = pool.lease(owner=7)
    rejected = _ValidatedArray(big_array())
    if nested:
        rejected = _Payload(rejected)
    payload = (big_array(), rejected)
    try:
        wrapped, leases = wrap_payload(payload, pool, owner=1)

        assert len(leases) == 1
        assert wrapped[1] is rejected
        assert pool.fallbacks == 1
        assert sorted(pool._states) == [0, 0, 1, 2]
        restored, taken = unwrap_payload(wrapped, pool, copy=True)
        assert np.array_equal(restored[0], payload[0])
        for index in taken:
            pool.release(index)
        for lease in leases:
            lease.release()
        # Releasing the returned leases cannot reach the unrelated writer.
        assert pool._states[held.index] == 1
    finally:
        held.release()


def test_a_rejected_outer_dataclass_releases_its_successful_nested_fields(pool):
    # _ValidatedArray accepts only an ndarray; its annotation does not enforce
    # the runtime type. Build a holder that permits the original nested object
    # but rejects the rewritten one, after its two children each took a slot.
    @dataclass(frozen=True)
    class _ValidatedPayload:
        first: object
        second: object

        def __post_init__(self):
            if not isinstance(self.second, np.ndarray):
                raise TypeError("second must be an ndarray")

    payload = _ValidatedPayload(_Payload(big_array()), big_array())

    wrapped, leases = wrap_payload(payload, pool)

    assert wrapped is payload
    assert leases == ()
    assert all(state == 0 for state in pool._states)
    assert pool.fallbacks == 1


def test_a_failed_wrap_releases_every_lease_it_acquired(pool):
    class _BrokenPayload(dict):
        def items(self):
            raise RuntimeError("payload traversal failed")

    with pytest.raises(RuntimeError, match="payload traversal failed"):
        wrap_payload((big_array(), _BrokenPayload()), pool)

    assert all(state == 0 for state in pool._states)
    assert pool.fallbacks == 0


def test_a_child_result_fallback_keeps_slots_available(pool, monkeypatch):
    from arrayscope.kernel.process_backend import _run_child_job

    payload = _ValidatedArray(big_array())
    monkeypatch.setattr(
        "arrayscope.kernel.process_backend.resolve_entry", lambda _entry: lambda _request: payload
    )

    # A child deliberately hands its result leases to the parent by descriptor,
    # rather than releasing them itself. Pickled fallbacks have no descriptors
    # for the parent to take, so rollback must happen before the reply is sent.
    for _ in range(pool.slots + 1):
        reply = _run_child_job(("entry", None, False), None, pool, owner=1)
        restored, taken = unwrap_payload(reply[1], pool, copy=True)
        assert reply[0] == "ok"
        assert np.array_equal(restored.data, payload.data)
        assert taken == ()
        assert all(state == 0 for state in pool._states)
    assert pool.fallbacks == pool.slots + 1
