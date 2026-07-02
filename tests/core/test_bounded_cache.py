"""Focused tests for the one bounded-cache core (roadmap Y3)."""

from arrayscope.core.bounded_cache import BoundedCache


def test_lru_eviction_by_entry_count():
    cache = BoundedCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # refresh recency
    cache.put("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.evictions == 1


def test_byte_budget_eviction_and_accounting():
    cache = BoundedCache(max_bytes=100)
    cache.put("a", "x", nbytes=60)
    cache.put("b", "y", nbytes=60)
    assert cache.bytes_used <= 100
    assert len(cache) == 1
    assert cache.get("a") is None
    assert cache.get("b") == "y"


def test_replace_updates_byte_accounting():
    cache = BoundedCache(max_bytes=100)
    cache.put("a", "x", nbytes=40)
    cache.put("a", "y", nbytes=10)
    assert cache.bytes_used == 10
    assert len(cache) == 1


def test_retention_key_evicts_smallest_first():
    cache = BoundedCache(max_entries=2, retention_key=lambda key, value: value)
    cache.put("low", 1)
    cache.put("high", 9)
    cache.put("mid", 5)
    assert "low" not in cache
    assert cache.peek("high") == 9
    assert cache.peek("mid") == 5


def test_peek_does_not_touch_recency_or_counters():
    cache = BoundedCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.peek("a") == 1
    assert cache.hits == 0 and cache.misses == 0
    cache.put("c", 3)  # "a" is still oldest because peek did not refresh
    assert "a" not in cache


def test_resize_evicts_to_new_budget():
    cache = BoundedCache(max_entries=8)
    for index in range(5):
        cache.put(index, index)
    cache.resize(max_entries=2)
    assert len(cache) == 2
    assert cache.peek(3) == 3 and cache.peek(4) == 4


def test_counters_and_clear():
    cache = BoundedCache(max_entries=4)
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.get("missing") is None
    cache.note_hit()
    cache.note_miss()
    assert (cache.hits, cache.misses) == (2, 2)
    cache.clear_counters()
    assert (cache.hits, cache.misses, cache.evictions) == (0, 0, 0)
    cache.clear()
    assert len(cache) == 0 and cache.bytes_used == 0


def test_unbounded_dimensions_do_not_evict():
    cache = BoundedCache()
    for index in range(100):
        cache.put(index, index, nbytes=1000)
    assert len(cache) == 100
    assert cache.evictions == 0
