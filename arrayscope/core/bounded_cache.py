"""One bounded-cache core: budget accounting and eviction (roadmap Y3).

Every bounded cache in the project builds on this core, so there is exactly
one eviction/priority implementation. The core owns the item store, the
byte/entry budgets, the eviction loop, and the hit/miss/eviction counters.

Eviction order is pluggable through ``retention_key``: without one, the entry
that has gone longest without insert/touch is evicted first (LRU); with one,
the entry whose ``retention_key(key, value)`` sorts smallest is evicted first.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from threading import RLock


class BoundedCache:
    def __init__(
        self,
        *,
        max_bytes: int | None = None,
        max_entries: int | None = None,
        retention_key: Callable[[object, object], object] | None = None,
        on_evict: Callable[[object, object, int], None] | None = None,
    ) -> None:
        self.lock = RLock()
        self._items: OrderedDict[object, tuple[object, int]] = OrderedDict()
        self._retention_key = retention_key
        # Optional eviction hook (G7 two-level cache): called with
        # (key, value, nbytes) for each entry the budget evicts.  Default None
        # keeps eviction byte-identical for every existing caller.
        self._on_evict = on_evict
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.max_entries = None if max_entries is None else int(max_entries)
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        with self.lock:
            return len(self._items)

    def __contains__(self, key) -> bool:
        with self.lock:
            return key in self._items

    def get(self, key, *, touch: bool = True, count: bool = True):
        with self.lock:
            entry = self._items.get(key)
            if entry is None:
                if count:
                    self.misses += 1
                return None
            if touch:
                self._items.move_to_end(key)
            if count:
                self.hits += 1
            return entry[0]

    def peek(self, key):
        """Read without touching recency or counters."""

        return self.get(key, touch=False, count=False)

    def peek_many(self, keys) -> dict:
        """Read several keys under one lock without touching cache state."""

        requested = tuple(keys or ())
        with self.lock:
            return {key: self._items[key][0] for key in requested if key in self._items}

    def note_hit(self) -> None:
        with self.lock:
            self.hits += 1

    def note_miss(self) -> None:
        with self.lock:
            self.misses += 1

    def put(self, key, value, *, nbytes: int = 0) -> None:
        nbytes = max(0, int(nbytes))
        with self.lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.bytes_used -= old[1]
            self._items[key] = (value, nbytes)
            self.bytes_used += nbytes
            self._evict()

    def discard(self, key):
        with self.lock:
            entry = self._items.pop(key, None)
            if entry is None:
                return None
            self.bytes_used -= entry[1]
            return entry[0]

    def items(self) -> tuple:
        """Snapshot of (key, value) pairs, oldest first."""

        with self.lock:
            return tuple((key, entry[0]) for key, entry in self._items.items())

    def would_fit(self, nbytes: int) -> bool:
        with self.lock:
            return self.max_bytes is None or int(nbytes) <= self.max_bytes

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        with self.lock:
            if max_bytes is not None:
                self.max_bytes = int(max_bytes)
            if max_entries is not None:
                self.max_entries = int(max_entries)
            self._evict()

    def clear(self) -> None:
        with self.lock:
            self._items.clear()
            self.bytes_used = 0

    def clear_counters(self) -> None:
        with self.lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def _over_budget(self) -> bool:
        if self.max_entries is not None and len(self._items) > self.max_entries:
            return True
        return self.max_bytes is not None and self.bytes_used > self.max_bytes

    def _evict(self) -> None:
        while self._items and self._over_budget():
            if self._retention_key is None:
                key = next(iter(self._items))
            else:
                retention = self._retention_key
                key = min(self._items.items(), key=lambda item: retention(item[0], item[1][0]))[0]
            value, nbytes = self._items.pop(key)
            self.bytes_used -= nbytes
            self.evictions += 1
            if self._on_evict is not None:
                # Runs under the cache lock (RLock).  The hook must not re-enter
                # this cache; the G7 tier only touches its own separate store.
                self._on_evict(key, value, nbytes)


__all__ = ["BoundedCache"]
