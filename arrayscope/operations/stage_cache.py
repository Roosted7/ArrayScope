"""In-memory cache for operation-stage array results."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from math import log2
from time import perf_counter

from arrayscope.core.bounded_cache import BoundedCache
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
    last_lookup_ms: float | None = None
    last_lookup_hit: bool | None = None
    compute_claims: int = 0
    compute_wait_reuses: int = 0


@dataclass(frozen=True)
class StageValue:
    data: object
    region: RegionSpec
    stage_index: int
    nbytes: int
    priority: str
    recompute_cost: float = 0.0
    hit_count: int = 0
    last_access_counter: int = 0
    visible_reuse: bool = False
    prefetch_only: bool = False


class _InFlightStageCompute:
    __slots__ = ("event", "value")

    def __init__(self):
        self.event = threading.Event()
        self.value = None


class StageCache:
    def __init__(self, *, max_bytes: int, max_entries: int = 64):
        self._cache = BoundedCache(
            max_bytes=int(max_bytes),
            max_entries=int(max_entries),
            retention_key=lambda key, value: (self.retention_score(key, value), int(value.last_access_counter)),
        )
        self._lock = self._cache.lock
        self.candidates_seen = 0
        self.stores = 0
        self.refused_over_budget = 0
        self.last_hit = ""
        self.last_miss = ""
        self.last_store = ""
        self.last_refused = ""
        self.last_lookup_ms = None
        self.last_lookup_hit = None
        self._access_counter = 0
        self._in_flight: dict[StageKey, _InFlightStageCompute] = {}
        self.compute_claims = 0
        self.compute_wait_reuses = 0

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

    def note_candidate(self, summary: str = "") -> None:
        with self._lock:
            self.candidates_seen += 1
            if summary:
                self.last_miss = str(summary)

    def note_refused(self, summary: str = "") -> None:
        with self._lock:
            self.refused_over_budget += 1
            if summary:
                self.last_refused = str(summary)

    def get(self, key: StageKey) -> StageValue | None:
        start = perf_counter()
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self.last_miss = _key_summary(key)
                self.last_lookup_hit = False
                self.last_lookup_ms = (perf_counter() - start) * 1000.0
                return None
            value = self._touch_value(value)
            self.last_hit = _key_summary(key)
            self.last_lookup_hit = True
            self.last_lookup_ms = (perf_counter() - start) * 1000.0
            return value

    def get_containing(self, key: StageKey) -> StageValue | None:
        start = perf_counter()
        with self._lock:
            for candidate_key, value in self._cache.items():
                if (
                    candidate_key.document_key == key.document_key
                    and candidate_key.operation_prefix == key.operation_prefix
                    and candidate_key.dtype == key.dtype
                    and tuple(candidate_key.shape) == tuple(key.shape)
                    and region_contains(value.region, key.region, key.shape)
                ):
                    self._cache.get(candidate_key)
                    value = self._touch_value(value)
                    self.last_hit = _key_summary(candidate_key)
                    self.last_lookup_hit = True
                    self.last_lookup_ms = (perf_counter() - start) * 1000.0
                    return value
            self._cache.note_miss()
            self.last_miss = _key_summary(key)
            self.last_lookup_hit = False
            self.last_lookup_ms = (perf_counter() - start) * 1000.0
            return None

    def resident_items(self) -> tuple[tuple[StageKey, StageValue], ...]:
        """Snapshot resident stage entries for GUI-side hot reuse checks.

        This is intentionally read-only. Callers that need exact region
        matching should use ``get_containing`` with a planned key; hot montage
        retargets use this only to reattach already-resident stage truth without
        running per-tile planning on the GUI thread.
        """

        with self._lock:
            return tuple((key, value) for key, value in self._cache.items())

    def begin_compute(self, key: StageKey) -> bool:
        """Claim the in-flight computation for ``key``.

        Returns True when the caller becomes the single computer for this
        stage; False when another evaluation is already computing it and the
        caller should wait via :meth:`wait_for_compute` instead of duplicating
        the work.
        """
        with self._lock:
            if key in self._in_flight:
                return False
            self._in_flight[key] = _InFlightStageCompute()
            self.compute_claims += 1
            return True

    def finish_compute(self, key: StageKey, value: StageValue | None = None) -> None:
        """Publish the claimed computation's result (or failure) to waiters.

        Must be called exactly once per successful :meth:`begin_compute`,
        even on failure (with ``value=None``) so waiters can take over.
        """
        with self._lock:
            entry = self._in_flight.pop(key, None)
        if entry is not None:
            entry.value = value
            entry.event.set()

    def wait_for_compute(
        self,
        key: StageKey,
        *,
        should_abort=None,
        poll_s: float = 0.05,
        timeout_s: float = 60.0,
    ) -> tuple[bool, StageValue | None]:
        """Wait for an in-flight computation of ``key`` to finish.

        Returns ``(finished, value)``. ``(True, value)`` when the computer
        published a result, ``(True, None)`` when nothing is in flight or the
        computer failed (callers should retry :meth:`begin_compute`), and
        ``(False, None)`` on abort or timeout (callers may fall back to
        computing without a claim, which at worst restores the old
        duplicate-computation behavior).
        """
        with self._lock:
            entry = self._in_flight.get(key)
        if entry is None:
            return True, None
        deadline = perf_counter() + float(timeout_s)
        while not entry.event.wait(float(poll_s)):
            if should_abort is not None and should_abort():
                return False, None
            if perf_counter() >= deadline:
                return False, None
        value = entry.value
        if value is not None:
            with self._lock:
                self.compute_wait_reuses += 1
        return True, value

    def put(self, key: StageKey, value: StageValue) -> bool:
        with self._lock:
            nbytes = max(0, int(value.nbytes))
            if not self._cache.would_fit(nbytes):
                self.refused_over_budget += 1
                self.last_refused = _key_summary(key)
                return False
            self._access_counter += 1
            object.__setattr__(value, "last_access_counter", self._access_counter)
            self._cache.put(key, value, nbytes=nbytes)
            self.stores += 1
            self.last_store = _key_summary(key)
            return True

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._cache.resize(max_bytes=max_bytes, max_entries=max_entries)

    def clear(self) -> None:
        self._cache.clear()

    def clear_counters(self) -> None:
        with self._lock:
            self._cache.clear_counters()
            self.candidates_seen = 0
            self.stores = 0
            self.refused_over_budget = 0
            self.last_hit = ""
            self.last_miss = ""
            self.last_store = ""
            self.last_refused = ""
            self.last_lookup_ms = None
            self.last_lookup_hit = None
            self.compute_claims = 0
            self.compute_wait_reuses = 0

    def diagnostics(self) -> StageCacheDiagnostics:
        with self._lock:
            total = int(self.hits) + int(self.misses)
            hit_rate = None if total == 0 else float(self.hits) / float(total)
            return StageCacheDiagnostics(
                entries=len(self._cache),
                bytes_used=int(self.bytes_used),
                max_bytes=int(self.max_bytes),
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
                last_lookup_ms=self.last_lookup_ms,
                last_lookup_hit=self.last_lookup_hit,
                compute_claims=int(self.compute_claims),
                compute_wait_reuses=int(self.compute_wait_reuses),
            )

    def retention_score(self, key: StageKey, value: StageValue) -> float:
        del key
        priority = _priority_rank(value.priority) * 1000.0
        recompute = min(float(value.recompute_cost or 0.0), 10_000.0)
        hits = min(int(value.hit_count), 100) * 25.0
        visible = 500.0 if value.visible_reuse else 0.0
        prefetch_penalty = 600.0 if value.prefetch_only else 0.0
        byte_penalty = log2(max(int(value.nbytes), 1)) * 8.0
        age_penalty = max(0, int(self._access_counter) - int(value.last_access_counter)) * 0.01
        return priority + recompute + hits + visible - prefetch_penalty - byte_penalty - age_penalty

    def _touch_value(self, value: StageValue) -> StageValue:
        self._access_counter += 1
        object.__setattr__(value, "hit_count", int(value.hit_count) + 1)
        object.__setattr__(value, "last_access_counter", self._access_counter)
        return value


def _priority_rank(priority: str) -> int:
    return _PRIORITY_RANK.get(str(priority), _PRIORITY_RANK["low"])


def _key_summary(key: StageKey) -> str:
    return (
        f"stage={len(tuple(key.operation_prefix))}, "
        f"region={region_text(key.region)}, dtype={key.dtype}, shape={tuple(key.shape)}"
    )
