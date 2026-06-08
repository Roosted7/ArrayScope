"""Small bounded LRU cache for display evaluation results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.cache_status import CacheDiagnosticsSnapshot, CacheStatus


@dataclass
class BoundedArrayCache:
    max_bytes: int
    max_entries: int

    def __post_init__(self):
        self.max_bytes = int(self.max_bytes)
        self.max_entries = int(self.max_entries)
        self._items = OrderedDict()
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.last_eval_ms = None

    def clear(self):
        self._items.clear()
        self.bytes_used = 0

    def get(self, key):
        if key not in self._items:
            self.misses += 1
            return None
        value, nbytes = self._items.pop(key)
        self._items[key] = (value, nbytes)
        self.hits += 1
        return value

    def put(self, key, value):
        nbytes = _nbytes(value)
        if key in self._items:
            _old_value, old_nbytes = self._items.pop(key)
            self.bytes_used -= old_nbytes
        self._items[key] = (value, nbytes)
        self.bytes_used += nbytes
        self._evict()
        return value

    def get_or_compute(self, key, compute):
        cached = self.get(key)
        if cached is not None:
            return cached, True
        start = perf_counter()
        value = compute()
        self.last_eval_ms = (perf_counter() - start) * 1000.0
        self.put(key, value)
        return value, False

    def diagnostics(self, status=CacheStatus.READY, message="", **extra):
        return CacheDiagnosticsSnapshot(
            status=status,
            message=message,
            entries=len(self._items),
            bytes_used=int(self.bytes_used),
            max_bytes=int(self.max_bytes),
            hits=int(self.hits),
            misses=int(self.misses),
            evictions=int(self.evictions),
            last_eval_ms=self.last_eval_ms,
            **extra,
        )

    def _evict(self):
        while self._items and (len(self._items) > self.max_entries or self.bytes_used > self.max_bytes):
            _key, (_value, nbytes) = self._items.popitem(last=False)
            self.bytes_used -= nbytes
            self.evictions += 1


def _nbytes(value):
    nbytes_method = getattr(value, "nbytes", None)
    if callable(nbytes_method):
        return int(nbytes_method())
    if hasattr(value, "image") and isinstance(getattr(value, "image"), np.ndarray):
        total = int(value.image.nbytes)
        histogram_data = getattr(value, "histogram_data", None)
        if isinstance(histogram_data, np.ndarray):
            total += int(histogram_data.nbytes)
        return total
    if hasattr(value, "data") and isinstance(getattr(value, "data"), np.ndarray):
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
