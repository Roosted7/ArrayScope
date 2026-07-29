"""Ring 1. The prepared-upload mailbox must drop stale buffers, never show them.

Pins the correctness rule of the GUI-thread hand-off seam
(`docs/architecture/progressive-render-contract.md`, R5 "the budget is a
bookkeeping budget"): a buffer prepared on a worker is a promise about *which
pixels these are*. Handing one back under a different identity is a torn frame,
so a mismatch must return nothing and let the caller prepare inline.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from arrayscope.display.shader_mapping import ShaderMapping, ShaderScale
from arrayscope.presentation.prepared_uploads import (
    PreparedUploadMailbox,
    cpu_mapping_preparation_variant,
    prepared_upload_key,
)


class _Payload:
    """The two attributes `prepared_upload_key` reads, and nothing else."""

    def __init__(self, tile_identity=None, source_id=None):
        self.tile_identity = tile_identity
        self.source_id = source_id


def _buffer(value: float = 1.0, size: int = 4) -> np.ndarray:
    return np.full((size, size), value, dtype=np.float32)


def test_take_under_the_publishing_key_returns_the_buffer():
    mailbox = PreparedUploadMailbox()
    buffer = _buffer()
    assert mailbox.publish(7, ("id-a", "scalar"), buffer)

    assert mailbox.take(7, ("id-a", "scalar")) is buffer
    assert mailbox.counters().hits == 1


def test_take_under_a_different_key_drops_the_buffer_and_reports_stale():
    """The core rule: a stale prepared buffer is refused, not presented."""

    mailbox = PreparedUploadMailbox()
    mailbox.publish(7, ("id-a", "scalar"), _buffer(1.0))

    # The slot moved on: this commit is submitting a different payload identity.
    assert mailbox.take(7, ("id-b", "scalar")) is None
    counters = mailbox.counters()
    assert counters.stale == 1
    assert counters.hits == 0
    # And the refused buffer is gone, so a later commit cannot resurrect it
    # by asking under the key it was originally prepared for.
    assert mailbox.take(7, ("id-a", "scalar")) is None
    assert mailbox.counters().resident_entries == 0


def test_a_changed_variant_is_as_stale_as_a_changed_identity():
    """Round levels / representation are part of what the buffer promises."""

    mailbox = PreparedUploadMailbox()
    mailbox.publish(3, ("id-a", (0.0, 1.0)), _buffer())

    assert mailbox.take(3, ("id-a", (0.0, 2.0))) is None
    assert mailbox.counters().stale == 1


def test_keep_latest_replaces_an_older_preparation_for_the_same_slot():
    """Under load the newest ready buffer wins; intermediates are not queued."""

    mailbox = PreparedUploadMailbox()
    stale_buffer = _buffer(1.0)
    fresh_buffer = _buffer(2.0)
    mailbox.publish(2, ("id-a", "scalar"), stale_buffer)
    mailbox.publish(2, ("id-b", "scalar"), fresh_buffer)

    assert mailbox.counters().replaced == 1
    # The superseded preparation is unreachable even under its own key.
    assert mailbox.take(2, ("id-a", "scalar")) is None
    mailbox.publish(2, ("id-b", "scalar"), fresh_buffer)
    assert mailbox.take(2, ("id-b", "scalar")) is fresh_buffer


def test_holds_reports_an_exact_waiting_preparation_only():
    """The submitter's dedupe: re-offering an admitted payload must not repeat work.

    Without this, a payload that stays dirty across a governed fill is prepared
    again on every completion drain -- measured at 21 588 assemblies for a
    272-tile montage before the check existed.
    """

    mailbox = PreparedUploadMailbox()
    mailbox.publish(4, ("id-a", "scalar"), _buffer())

    assert mailbox.holds(4, ("id-a", "scalar")) is True
    assert mailbox.holds(4, ("id-b", "scalar")) is False
    assert mailbox.holds(5, ("id-a", "scalar")) is False
    # Asking does not consume: the commit still gets its buffer.
    assert mailbox.take(4, ("id-a", "scalar")) is not None
    assert mailbox.holds(4, ("id-a", "scalar")) is False


def test_a_miss_is_an_ordinary_outcome_and_is_counted_separately():
    mailbox = PreparedUploadMailbox()

    assert mailbox.take(11, ("id-a", "scalar")) is None
    counters = mailbox.counters()
    assert counters.misses == 1
    assert counters.stale == 0


def test_the_budget_bounds_residency_and_never_evicts_the_newest_publish():
    """Prepared buffers are large; the mailbox may not grow without bound."""

    one = _buffer(1.0, size=64)  # 16 KiB
    mailbox = PreparedUploadMailbox(budget_bytes=int(one.nbytes) * 2)
    for slot in range(6):
        assert mailbox.publish(slot, (f"id-{slot}", "scalar"), _buffer(float(slot), size=64))

    counters = mailbox.counters()
    assert counters.resident_bytes <= int(one.nbytes) * 2
    assert counters.evicted > 0
    # The most recent publish survives: it is the one a governed fill wants next.
    assert mailbox.take(5, ("id-5", "scalar")) is not None


def test_a_buffer_larger_than_the_whole_budget_is_declined_not_hoarded():
    mailbox = PreparedUploadMailbox(budget_bytes=1024)

    assert mailbox.publish(0, ("id-a", "scalar"), _buffer(1.0, size=64)) is False
    assert mailbox.counters().resident_entries == 0


def test_discard_and_clear_release_their_bytes():
    mailbox = PreparedUploadMailbox()
    mailbox.publish(1, ("id-a", "scalar"), _buffer())
    mailbox.publish(2, ("id-b", "scalar"), _buffer())

    mailbox.discard(1)
    assert mailbox.counters().resident_entries == 1
    mailbox.clear()
    counters = mailbox.counters()
    assert counters.resident_entries == 0
    assert counters.resident_bytes == 0


def test_concurrent_publishers_and_one_taker_never_yield_a_mismatched_buffer():
    """The hand-off is cross-thread; a take must never return another key's data."""

    mailbox = PreparedUploadMailbox()
    stop = threading.Event()
    mismatches: list[tuple] = []

    def publisher(index: int) -> None:
        while not stop.is_set():
            key = (f"id-{index}", "scalar")
            mailbox.publish(0, key, np.full((8, 8), float(index), dtype=np.float32))

    threads = [threading.Thread(target=publisher, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    try:
        for _ in range(4000):
            for index in range(4):
                taken = mailbox.take(0, (f"id-{index}", "scalar"))
                if taken is not None and float(taken[0, 0]) != float(index):
                    mismatches.append((index, float(taken[0, 0])))
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)

    assert not mismatches


@pytest.mark.parametrize(
    ("payload", "expected_identity"),
    [
        (_Payload(tile_identity="tile-id", source_id="source-id"), "tile-id"),
        (_Payload(tile_identity=None, source_id="source-id"), "source-id"),
    ],
)
def test_the_key_prefers_the_acknowledgement_identity(payload, expected_identity):
    """Semantic and physical payload identities both guard the buffer."""

    assert prepared_upload_key(payload, "scalar") == (
        expected_identity if payload.tile_identity is not None else None,
        payload.source_id,
        "scalar",
    )


def test_same_acknowledgement_identity_with_new_actual_pages_is_stale():
    """Fallback page improvements may share TileIdentity but not pixels."""

    mailbox = PreparedUploadMailbox()
    older = _Payload(tile_identity="same-target", source_id=("actual-pages", "coarse"))
    newer = _Payload(tile_identity="same-target", source_id=("actual-pages", "fine"))
    mailbox.publish(4, prepared_upload_key(older, "scalar"), _buffer(1.0))

    assert mailbox.take(4, prepared_upload_key(newer, "scalar")) is None
    assert mailbox.counters().stale == 1


def test_cpu_mapping_variant_includes_symlog_and_lut_inputs():
    """Equal levels do not imply equal CPU-mapped complex pixels."""

    payload = _Payload(tile_identity="same-target", source_id="same-source")
    payload.shader_mapping = ShaderMapping(
        scale=ShaderScale.SYMLOG,
        symlog_constant=1.0,
        lut_identity=("lut", 1),
    )
    first = cpu_mapping_preparation_variant(payload, (0.0, 2.0))
    payload.shader_mapping = ShaderMapping(
        scale=ShaderScale.SYMLOG,
        symlog_constant=2.0,
        lut_identity=("lut", 2),
    )
    second = cpu_mapping_preparation_variant(payload, (0.0, 2.0))

    assert first != second
