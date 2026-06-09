"""In-memory cache for operation-stage array results."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

from arrayscope.operations.regions import RegionSpec, StageKey, region_contains, region_text


_PRIORITY_RANK = {
    "lowest": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "highest": 4,
}


@dataclass(frozen=True)
class StageCacheDiagnostics:
    entries: int
    bytes_used: int
    max_bytes: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float | None
    candidates_seen: int
    stores: int
    refused_over_budget: int
    last_hit: str = ""
    last_miss: str = ""
    last_store: str = ""
    last_refused: str = ""


@dataclass(frozen=True)
class StageValue:
    data: object
    region: RegionSpec
    stage_index: int
    nbytes: int
    priority: str
    recompute_cost: float = 0.0


class StageCache:
    def __init__(self, *, max_bytes: int, max_entries: int = 64):
        self._max_bytes = int(max_bytes)
        self._max_entries = int(max_entries)
        self._items: OrderedDict[StageKey, StageValue] = OrderedDict()
        self._bytes_used = 0
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.candidates_seen = 0
        self.stores = 0
        self.refused_over_budget = 0
        self.last_hit = ""
        self.last_miss = ""
        self.last_store = ""
        self.last_refused = ""

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def bytes_used(self) -> int:
        return self._bytes_used

    def note_candidate(self, summary: str = "") -> None:
        with self._lock:
            self.candidates_seen += 1
            if summary:
                self.last_miss = str(summary)

    def get(self, key: StageKey) -> StageValue | None:
        with self._lock:
            value = self._items.pop(key, None)
            if value is None:
                self.misses += 1
                self.last_miss = _key_summary(key)
                return None
            self._items[key] = value
            self.hits += 1
            self.last_hit = _key_summary(key)
            return value

    def get_containing(self, key: StageKey) -> StageValue | None:
        with self._lock:
            for candidate_key, value in list(self._items.items()):
                if (
                    candidate_key.document_key == key.document_key
                    and candidate_key.operation_prefix == key.operation_prefix
                    and candidate_key.dtype == key.dtype
                    and tuple(candidate_key.shape) == tuple(key.shape)
                    and region_contains(value.region, key.region, key.shape)
                ):
                    self._items.pop(candidate_key)
                    self._items[candidate_key] = value
                    self.hits += 1
                    self.last_hit = _key_summary(candidate_key)
                    return value
            self.misses += 1
            self.last_miss = _key_summary(key)
            return None

    def put(self, key: StageKey, value: StageValue) -> bool:
        with self._lock:
            nbytes = max(0, int(value.nbytes))
            if nbytes > self._max_bytes:
                self.refused_over_budget += 1
                self.last_refused = _key_summary(key)
                return False
            if key in self._items:
                old = self._items.pop(key)
                self._bytes_used -= int(old.nbytes)
            self._items[key] = value
            self._bytes_used += nbytes
            self.stores += 1
            self.last_store = _key_summary(key)
            self._evict()
            return True

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        with self._lock:
            if max_bytes is not None:
                self._max_bytes = int(max_bytes)
            if max_entries is not None:
                self._max_entries = int(max_entries)
            self._evict()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes_used = 0

    def clear_counters(self) -> None:
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.candidates_seen = 0
            self.stores = 0
            self.refused_over_budget = 0
            self.last_hit = ""
            self.last_miss = ""
            self.last_store = ""
            self.last_refused = ""

    def diagnostics(self) -> StageCacheDiagnostics:
        with self._lock:
            total = int(self.hits) + int(self.misses)
            hit_rate = None if total == 0 else float(self.hits) / float(total)
            return StageCacheDiagnostics(
                entries=len(self._items),
                bytes_used=int(self._bytes_used),
                max_bytes=int(self._max_bytes),
                hits=int(self.hits),
                misses=int(self.misses),
                evictions=int(self.evictions),
                hit_rate=hit_rate,
                candidates_seen=int(self.candidates_seen),
                stores=int(self.stores),
                refused_over_budget=int(self.refused_over_budget),
                last_hit=self.last_hit,
                last_miss=self.last_miss,
                last_store=self.last_store,
                last_refused=self.last_refused,
            )

    def _evict(self) -> None:
        while self._items and (len(self._items) > self._max_entries or self._bytes_used > self._max_bytes):
            key = self._eviction_key()
            value = self._items.pop(key)
            self._bytes_used -= int(value.nbytes)
            self.evictions += 1

    def _eviction_key(self):
        lowest_rank = min(_priority_rank(value.priority) for value in self._items.values())
        for key, value in self._items.items():
            if _priority_rank(value.priority) == lowest_rank:
                return key
        return next(iter(self._items))


def _priority_rank(priority: str) -> int:
    return _PRIORITY_RANK.get(str(priority), _PRIORITY_RANK["low"])


def _key_summary(key: StageKey) -> str:
    return (
        f"stage={len(tuple(key.operation_prefix))}, "
        f"region={region_text(key.region)}, dtype={key.dtype}, shape={tuple(key.shape)}"
    )
