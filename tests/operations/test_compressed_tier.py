"""G7 Phase A: two-level cache correctness, tier recovery, and default-off proof.

The compressed backing tier must be lossless (bit-identical round-trip) and must
recover an evicted value by *decode* rather than by recompute -- that is the
whole point (a decode replaces the expensive miss).  The default path (no tier)
must be byte-identical to today's raw cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from arrayscope.operations.cache import BoundedArrayCache
from arrayscope.operations.compressed_tier import (
    CompressedBackingTier,
    TwoLevelArrayCache,
)


def _sample(dtype, shape=(128, 128)) -> np.ndarray:
    rng = np.random.default_rng(11)
    dtype = np.dtype(dtype)
    if dtype == np.complex64:
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    if dtype == np.int16:
        return (rng.integers(-2000, 2000, size=shape)).astype(np.int16)
    return (rng.standard_normal(shape) * 100.0).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.complex64, np.int16])
@pytest.mark.parametrize("codec", ["zfp", "blosc2"])
def test_tier_round_trip_bit_identical(dtype, codec):
    array = _sample(dtype)
    tier = CompressedBackingTier(max_bytes=10 << 20, codec_name=codec)
    tier.store(("k",), array)
    recovered = tier.load(("k",))
    assert recovered.dtype == np.dtype(dtype)
    assert recovered.shape == array.shape
    # Lossless: bit-for-bit identical, not merely close.
    assert np.array_equal(recovered, array)


def test_evicted_from_raw_is_recovered_by_decode_not_recompute():
    """An item evicted from the raw cache is served from the tier by decode.

    The loader/recompute is spied: on a raw-miss/tier-hit it must NOT be called.
    """

    chunk_bytes = _sample(np.float32).nbytes
    tier = CompressedBackingTier(max_bytes=10 << 20, codec_name="zfp")
    # Raw cache holds only 2 chunks; the tier backs the rest.
    cache = TwoLevelArrayCache.build(
        raw_max_bytes=chunk_bytes * 2, raw_max_entries=100, tier=tier
    )

    values = {i: _sample(np.float32) + i for i in range(6)}
    calls: list[int] = []

    def loader(i):
        def compute():
            calls.append(i)
            return values[i]

        return compute

    # Fill: 6 distinct chunks through a 2-slot raw cache -> 0 and others evict
    # into the tier.
    for i in range(6):
        cache.get_or_compute((i,), loader(i))
    assert calls == list(range(6))  # each computed exactly once so far

    calls.clear()
    # Revisit chunk 0: evicted from raw long ago, but present compressed.
    value, hit = cache.get_or_compute((0,), loader(0))
    assert hit is True
    assert calls == [], "tier recovery must not recompute"
    assert np.array_equal(value, values[0])
    assert cache.tier_recoveries >= 1


def test_default_off_never_invokes_a_codec(monkeypatch):
    """With no tier, the two-level cache is a pass-through and never encodes.

    Guards the byte-identical default: resolve_codec/encode must not be reached.
    """

    import arrayscope.operations.compressed_tier as mod

    def _boom(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("codec used on the default (tier-off) path")

    monkeypatch.setattr(mod, "resolve_codec", _boom)
    monkeypatch.setattr(mod, "get_codec", _boom)

    cache = TwoLevelArrayCache.build(raw_max_bytes=1 << 20, raw_max_entries=8, tier=None)
    array = _sample(np.float32)
    cache.put((0,), array)
    got = cache.get((0,))
    assert np.array_equal(got, array)
    assert cache.tier is None
    assert cache.tier_recoveries == 0


def test_default_bounded_array_cache_byte_identical_without_hook():
    """BoundedArrayCache with no on_evict behaves exactly as before (eviction).

    Proves the additive hook did not change the default cache's eviction path.
    """

    chunk_bytes = _sample(np.float32).nbytes
    cache = BoundedArrayCache(chunk_bytes * 2, max_entries=100)  # no on_evict
    for i in range(5):
        cache.put((i,), _sample(np.float32))
    # Only the last 2 survive; the rest evicted and are gone (no tier).
    assert cache.get((0,)) is None
    assert cache.get((4,)) is not None
    assert cache.evictions == 3


def _smooth(i: int, shape=(256, 256)) -> np.ndarray:
    """Structured int16 plane (a gradient) -- representative of imaging data,
    which zfp compresses well (random noise does not)."""

    ramp = np.linspace(0, 1, shape[1], dtype=np.float32)
    plane = np.outer(np.ones(shape[0], np.float32), ramp) + i * 0.01
    return (plane * 2000.0).astype(np.int16)


def test_tier_ram_win_retains_more_working_set():
    """Under one byte budget the tier retains more entries than raw would."""

    chunk_bytes = _smooth(0).nbytes
    budget = chunk_bytes * 8
    raw = BoundedArrayCache(budget, max_entries=1000)
    tier = CompressedBackingTier(max_bytes=budget, codec_name="zfp")
    for i in range(60):
        raw.put((i,), _smooth(i))
        tier.store((i,), _smooth(i))
    # Same byte budget; the compressed tier keeps strictly more entries.
    assert len(tier) > len(raw._cache)
    diag = tier.diagnostics()
    assert diag.mean_ratio is not None
    assert diag.mean_ratio > 1.2
