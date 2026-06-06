import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
PATH = ROOT / "arrayscope" / "core" / "cache_status.py"
SPEC = importlib.util.spec_from_file_location("arrayscope.core.cache_status", PATH)
cache_status = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache_status
SPEC.loader.exec_module(cache_status)


def test_cache_status_for_hit_miss_and_error():
    assert cache_status.cache_status_for_hit(True).status == cache_status.CacheStatus.CACHED
    assert cache_status.cache_status_for_hit(False).status == cache_status.CacheStatus.COMPUTING
    assert cache_status.cache_status_for_hit(False, has_error=True).status == cache_status.CacheStatus.ERROR
    assert cache_status.cache_status_ready().status == cache_status.CacheStatus.READY
    assert cache_status.cache_status_error(ValueError("bad")).status == cache_status.CacheStatus.ERROR
    assert cache_status.cache_status_prefetching().status == cache_status.CacheStatus.PREFETCHING
    assert cache_status.cache_status_stale_ignored().status == cache_status.CacheStatus.STALE_IGNORED


def test_cache_diagnostics_snapshot_defaults_and_values():
    snapshot = cache_status.CacheDiagnosticsSnapshot(
        cache_status.CacheStatus.READY,
        "ready",
        entries=2,
        bytes_used=128,
        max_bytes=256,
        hits=3,
        misses=4,
        evictions=1,
        last_eval_ms=12.5,
    )

    assert snapshot.status == cache_status.CacheStatus.READY
    assert snapshot.entries == 2
    assert snapshot.bytes_used == 128
    assert snapshot.last_eval_ms == 12.5
