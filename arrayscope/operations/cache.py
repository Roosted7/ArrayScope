"""Bounded LRU cache for display evaluation results."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from arrayscope.core.bounded_cache import BoundedCache
from arrayscope.core.cache_status import CacheDiagnosticsSnapshot, CacheStatus


class BoundedArrayCache:
    def __init__(self, max_bytes: int, max_entries: int):
        self._cache = BoundedCache(max_bytes=int(max_bytes), max_entries=int(max_entries))
        self.last_eval_ms = None

    @property
    def max_bytes(self) -> int:
        return int(self._cache.max_bytes)

    @property
    def max_entries(self) -> int:
        return int(self._cache.max_entries)

    @property
    def bytes_used(self) -> int:
        return int(self._cache.bytes_used)

    @property
    def hits(self) -> int:
        return int(self._cache.hits)

    @property
    def misses(self) -> int:
        return int(self._cache.misses)

    @property
    def evictions(self) -> int:
        return int(self._cache.evictions)

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self):
        self._cache.clear()

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._cache.resize(max_bytes=max_bytes, max_entries=max_entries)

    def clear_counters(self) -> None:
        self._cache.clear_counters()

    def get(self, key):
        return self._cache.get(key)

    def put(self, key, value):
        self._cache.put(key, value, nbytes=_nbytes(value))
        return value

    def get_or_compute(self, key, compute):
        cached = self.get(key)
        if cached is not None:
            return cached, True
        start = perf_counter()
        value = compute()
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.last_eval_ms = elapsed_ms
        self.put(key, value)
        return value, False

    def diagnostics(self, status=CacheStatus.READY, message="", **extra):
        with self._cache.lock:
            total = int(self.hits) + int(self.misses)
            hit_rate = None if total == 0 else float(self.hits) / float(total)
            return CacheDiagnosticsSnapshot(
                status=status,
                message=message,
                entries=len(self._cache),
                bytes_used=int(self.bytes_used),
                max_bytes=int(self.max_bytes),
                hits=int(self.hits),
                misses=int(self.misses),
                evictions=int(self.evictions),
                last_eval_ms=self.last_eval_ms,
                hit_rate=hit_rate,
                **extra,
            )


def _nbytes(value):
    nbytes_method = getattr(value, "nbytes", None)
    if callable(nbytes_method):
        return int(nbytes_method())
    if hasattr(value, "image") and isinstance(value.image, np.ndarray):
        total = int(value.image.nbytes)
        histogram_data = getattr(value, "histogram_data", None)
        if isinstance(histogram_data, np.ndarray):
            total += int(histogram_data.nbytes)
        return total
    if hasattr(value, "data") and isinstance(value.data, np.ndarray):
        total = int(value.data.nbytes)
        histogram_data = getattr(value, "histogram_data", None)
        if isinstance(histogram_data, np.ndarray):
            total += int(histogram_data.nbytes)
        return total
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if np.isscalar(value):
        return int(np.asarray(value).nbytes)
    return 1
