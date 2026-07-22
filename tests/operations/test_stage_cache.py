import numpy as np

from arrayscope.operations.compressed_tier import CompressedBackingTier
from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec, StageKey
from arrayscope.operations.stage_cache import StageCache, StageValue


def _key(name, region=None):
    region = RegionSpec((AxisRegion(AxisRegionKind.ALL),)) if region is None else region
    return StageKey(
        document_key=("doc",), operation_prefix=(name,), region=region, dtype="float32", shape=(4,)
    )


def _value(data, *, priority="low", region=None):
    data = np.asarray(data, dtype=np.float32)
    region = RegionSpec((AxisRegion(AxisRegionKind.ALL),)) if region is None else region
    return StageValue(
        data=data, region=region, stage_index=1, nbytes=int(data.nbytes), priority=priority
    )


def test_stage_cache_put_get_and_diagnostics():
    cache = StageCache(max_bytes=1024, max_entries=4)
    key = _key("a")
    value = _value(np.arange(4))

    assert cache.get(key) is None
    assert cache.put(key, value) is True
    assert cache.get(key) is value

    diagnostics = cache.diagnostics()
    assert diagnostics.entries == 1
    assert diagnostics.bytes_used == value.nbytes
    assert diagnostics.hits == 1
    assert diagnostics.misses == 1
    assert diagnostics.hit_rate == 0.5
    assert diagnostics.last_hit


def test_stage_cache_refuses_oversized_stage():
    cache = StageCache(max_bytes=8, max_entries=4)
    assert cache.put(_key("large"), _value(np.arange(4))) is False

    diagnostics = cache.diagnostics()
    assert diagnostics.entries == 0
    assert diagnostics.refused_over_budget == 1
    assert diagnostics.hit_rate is None


def test_stage_cache_resize_and_priority_eviction():
    cache = StageCache(max_bytes=64, max_entries=4)
    low = _value(np.arange(4), priority="low")
    high = _value(np.arange(4), priority="highest")
    cache.put(_key("low"), low)
    cache.put(_key("high"), high)

    cache.resize(max_bytes=20)

    assert cache.get(_key("low")) is None
    assert cache.get(_key("high")) is high
    assert cache.diagnostics().evictions == 1


def test_stage_cache_lru_within_equal_priority():
    cache = StageCache(max_bytes=64, max_entries=2)
    first = _value(np.arange(2), priority="medium")
    second = _value(np.arange(2), priority="medium")
    third = _value(np.arange(2), priority="medium")
    cache.put(_key("first"), first)
    cache.put(_key("second"), second)
    assert cache.get(_key("first")) is first
    cache.put(_key("third"), third)

    assert cache.get(_key("second")) is None
    assert cache.get(_key("first")) is first
    assert cache.get(_key("third")) is third


def test_stage_cache_clear_preserves_counters_and_clear_counters_resets():
    cache = StageCache(max_bytes=1024, max_entries=4)
    cache.put(_key("a"), _value(np.arange(2)))
    cache.get(_key("a"))
    cache.clear()

    diagnostics = cache.diagnostics()
    assert diagnostics.entries == 0
    assert diagnostics.hits == 1

    cache.clear_counters()
    diagnostics = cache.diagnostics()
    assert diagnostics.hits == 0
    assert diagnostics.misses == 0
    assert diagnostics.stores == 0


def test_stage_cache_get_containing_returns_broader_region():
    cache = StageCache(max_bytes=1024, max_entries=4)
    full = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.ALL)))
    point = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.POINT, 2)))
    key_full = StageKey(("doc",), ("fft",), full, "float32", (4, 5))
    key_point = StageKey(("doc",), ("fft",), point, "float32", (4, 5))
    value = StageValue(np.zeros((4, 5), dtype=np.float32), full, 1, 80, "high")
    cache.put(key_full, value)

    assert cache.get_containing(key_point) is value
    assert cache.diagnostics().hits == 1


def test_stage_cache_resident_snapshot_does_not_wait_for_mutation_lock():
    import threading

    cache = StageCache(max_bytes=1024, max_entries=4)
    full = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.ALL)))
    point = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.POINT, 2)))
    key_full = StageKey(("doc",), ("fft",), full, "float32", (4, 5))
    key_point = StageKey(("doc",), ("fft",), point, "float32", (4, 5))
    value = StageValue(np.zeros((4, 5), dtype=np.float32), full, 1, 80, "high")
    cache.put(key_full, value)
    completed = threading.Event()
    observed = []

    def read_snapshot():
        observed.append((cache.resident_items(), cache.peek_containing_resident(key_point)))
        completed.set()

    with cache._lock:
        reader = threading.Thread(target=read_snapshot)
        reader.start()
        assert completed.wait(timeout=5.0)
    reader.join(timeout=5.0)

    assert observed == [(((key_full, value),), value)]


def test_stage_cache_snapshot_tracks_eviction_and_clear():
    cache = StageCache(max_bytes=64, max_entries=2)
    first = _value(np.arange(4), priority="low")
    second = _value(np.arange(4), priority="highest")
    cache.put(_key("first"), first)
    cache.put(_key("second"), second)

    cache.resize(max_bytes=20)
    assert cache.resident_items() == ((_key("second"), second),)

    cache.clear()
    assert cache.resident_items() == ()


def test_stage_cache_retention_score_prefers_hot_visible_expensive_stage():
    cache = StageCache(max_bytes=40, max_entries=4)
    cheap_prefetch = StageValue(
        data=np.arange(4, dtype=np.float32),
        region=RegionSpec((AxisRegion(AxisRegionKind.ALL),)),
        stage_index=1,
        nbytes=16,
        priority="high",
        recompute_cost=0.0,
        prefetch_only=True,
    )
    expensive_visible = StageValue(
        data=np.arange(4, dtype=np.float32),
        region=RegionSpec((AxisRegion(AxisRegionKind.ALL),)),
        stage_index=1,
        nbytes=16,
        priority="high",
        recompute_cost=500.0,
        visible_reuse=True,
    )
    cache.put(_key("prefetch"), cheap_prefetch)
    cache.put(_key("visible"), expensive_visible)
    assert cache.get(_key("visible")) is expensive_visible

    cache.put(_key("new"), _value(np.arange(4), priority="high"))

    assert cache.get(_key("prefetch")) is None
    assert cache.get(_key("visible")) is expensive_visible


def test_stage_cache_in_flight_claim_and_publish():
    cache = StageCache(max_bytes=1024, max_entries=4)
    key = _key("a")

    assert cache.begin_compute(key) is True
    assert cache.begin_compute(key) is False

    value = _value(np.arange(4))
    cache.finish_compute(key, value)
    finished, waited = cache.wait_for_compute(key)
    # Entry is gone after finish: nothing in flight means the caller may claim.
    assert finished is True
    assert waited is None
    assert cache.begin_compute(key) is True
    cache.finish_compute(key, None)

    diagnostics = cache.diagnostics()
    assert diagnostics.compute_claims == 2


def test_stage_cache_wait_receives_published_value_and_times_out():
    import threading

    cache = StageCache(max_bytes=1024, max_entries=4)
    key = _key("a")
    value = _value(np.arange(4))
    assert cache.begin_compute(key) is True

    results = []
    waiter = threading.Thread(
        target=lambda: results.append(cache.wait_for_compute(key, poll_s=0.01))
    )
    waiter.start()
    cache.finish_compute(key, value)
    waiter.join(timeout=5)
    assert results == [(True, value)]
    assert cache.diagnostics().compute_wait_reuses == 1

    # A failed computer publishes None: waiters see finished-without-value.
    assert cache.begin_compute(key) is True
    finisher = threading.Timer(0.05, cache.finish_compute, args=(key, None))
    finisher.start()
    finished, waited = cache.wait_for_compute(key, poll_s=0.01)
    assert finished is True
    assert waited is None

    # Timeout leaves the claim in place and reports not-finished.
    assert cache.begin_compute(key) is True
    finished, waited = cache.wait_for_compute(key, poll_s=0.01, timeout_s=0.05)
    assert finished is False
    assert waited is None
    cache.finish_compute(key, None)


# --- G7 compressed backing tier (host-cache codec) --------------------------
def _tier_cache(raw_bytes, tier_bytes, *, codec="zfp", max_entries=64):
    tier = CompressedBackingTier(max_bytes=tier_bytes, codec_name=codec)
    return StageCache(
        max_bytes=raw_bytes,
        max_entries=max_entries,
        tier=tier,
        total_max_bytes=raw_bytes + tier_bytes,
    )


def test_stage_tier_off_is_byte_identical_default():
    cache = StageCache(max_bytes=1024, max_entries=4)
    assert cache.tier is None
    diag = cache.diagnostics()
    assert diag.tier_engaged is False
    assert diag.tier_recoveries == 0
    assert diag.max_bytes == 1024  # no envelope split when the tier is off


def test_stage_tier_recovers_evicted_stage_by_decode():
    chunk = _value(np.arange(64, dtype=np.float32)).nbytes  # 256 B
    cache = _tier_cache(raw_bytes=chunk, tier_bytes=chunk * 8)  # raw holds ~1 entry
    keys = [_key(f"s{i}") for i in range(6)]
    vals = [_value(np.arange(64, dtype=np.float32) + i) for i in range(6)]
    for k, v in zip(keys, vals, strict=True):
        assert cache.put(k, v) is True

    # keys[0] was evicted from the raw hot cache long ago -> served by a tier
    # decode, bit-identical, not a None miss.
    recovered = cache.get(keys[0])
    assert recovered is not None
    assert np.array_equal(recovered.data, vals[0].data)

    diag = cache.diagnostics()
    assert diag.tier_engaged is True
    assert diag.tier_recoveries >= 1
    assert diag.max_bytes == chunk + chunk * 8  # raw+tier envelope
    assert diag.bytes_used <= diag.max_bytes


def test_stage_tier_compressed_entries_stay_out_of_containment_and_snapshot():
    full = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.ALL)))
    point = RegionSpec((AxisRegion(AxisRegionKind.ALL), AxisRegion(AxisRegionKind.POINT, 2)))
    key_full = StageKey(("doc",), ("fft",), full, "float32", (8, 8))
    key_point = StageKey(("doc",), ("fft",), point, "float32", (8, 8))
    array = np.arange(64, dtype=np.float32).reshape(8, 8)
    value = StageValue(array, full, 1, int(array.nbytes), "low")

    cache = _tier_cache(raw_bytes=int(array.nbytes), tier_bytes=int(array.nbytes) * 4)
    assert cache.put(key_full, value) is True
    # Evict key_full into the tier with an unrelated-prefix filler.
    filler_key = StageKey(("doc",), ("filler",), full, "float32", (8, 8))
    assert (
        cache.put(filler_key, StageValue(array.copy(), full, 1, int(array.nbytes), "low")) is True
    )

    # The containing entry now lives compressed in the tier: containment matching
    # and the lock-free resident snapshot must NOT see it (no hidden decode).
    assert cache.get_containing(key_point) is None
    assert cache.peek_containing_resident(key_point) is None
    assert all(k != key_full for k, _v in cache.resident_items())

    # But an exact-key get still recovers it by decode (bit-identical).
    recovered = cache.get(key_full)
    assert recovered is not None
    assert np.array_equal(recovered.data, array)
    assert cache.diagnostics().tier_recoveries >= 1
