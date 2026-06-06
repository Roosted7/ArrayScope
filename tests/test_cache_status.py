import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
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
