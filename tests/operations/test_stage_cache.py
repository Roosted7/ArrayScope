import numpy as np

from arrayscope.operations.regions import AxisRegion, AxisRegionKind, RegionSpec, StageKey
from arrayscope.operations.stage_cache import StageCache, StageValue


def _key(name, region=None):
    region = RegionSpec((AxisRegion(AxisRegionKind.ALL),)) if region is None else region
    return StageKey(document_key=("doc",), operation_prefix=(name,), region=region, dtype="float32", shape=(4,))


def _value(data, *, priority="low", region=None):
    data = np.asarray(data, dtype=np.float32)
    region = RegionSpec((AxisRegion(AxisRegionKind.ALL),)) if region is None else region
    return StageValue(data=data, region=region, stage_index=1, nbytes=int(data.nbytes), priority=priority)


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
