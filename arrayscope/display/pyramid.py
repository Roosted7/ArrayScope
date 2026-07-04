"""Qt-free LOD pyramid materialization core (ADR 0050).

This module owns the reduction algorithm, the pyramid cache key, and the
bounded byte-accounted cache with singleflight request bookkeeping.  It must
stay importable without Qt: workers reduce and admit, the GUI thread only
looks levels up.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np

from arrayscope.core.bounded_cache import BoundedCache


ALGO_VERSION = 1
"""Reduction algorithm version; part of every pyramid cache key."""


def reduce_box_mean(array, factor_xy: tuple[int, int]) -> np.ndarray:
    """Reduce the first two axes of ``array`` by per-axis box means.

    ``factor_xy`` is ``(factor_x, factor_y)`` where x reduces the width axis
    (axis 1) and y reduces the height axis (axis 0), matching the
    ``desired_factor_xy`` convention in :mod:`arrayscope.display.lod`.

    Rules:

    - factors must be powers of two (level identity is per-axis log2);
    - accumulation runs in float32 for real inputs and complex64 for complex
      inputs (texture-appropriate output precision);
    - non-divisible trailing edges average the partial box, so no padding
      values leak into the result;
    - trailing component axes (RGB(A) or RG two-component planes) are reduced
      per component;
    - integer inputs round back to the input dtype so RGB8 textures stay
      uploadable.
    """

    factor_x, factor_y = (int(factor_xy[0]), int(factor_xy[1]))
    if factor_x < 1 or factor_y < 1:
        raise ValueError("reduction factors must be positive")
    if factor_x & (factor_x - 1) or factor_y & (factor_y - 1):
        raise ValueError(f"reduction factors must be powers of two, got {(factor_x, factor_y)}")
    values = np.asarray(array)
    if values.ndim < 2:
        raise ValueError("box-mean reduction requires at least two axes")
    if values.ndim > 3:
        raise ValueError("box-mean reduction supports 2D data plus one trailing component axis")
    if np.iscomplexobj(values):
        accumulated = values.astype(np.complex64, copy=False)
    else:
        accumulated = values.astype(np.float32, copy=False)
    reduced = _reduce_axis(_reduce_axis(accumulated, factor_y, axis=0), factor_x, axis=1)
    if np.issubdtype(values.dtype, np.integer):
        info = np.iinfo(values.dtype)
        return np.clip(np.rint(reduced), info.min, info.max).astype(values.dtype)
    if np.iscomplexobj(values):
        return reduced.astype(np.complex64, copy=False)
    return reduced.astype(np.float32, copy=False)


def _reduce_axis(values: np.ndarray, factor: int, *, axis: int) -> np.ndarray:
    if factor <= 1 or values.shape[axis] <= 1:
        return values
    length = int(values.shape[axis])
    starts = np.arange(0, length, factor)
    sums = np.add.reduceat(values, starts, axis=axis, dtype=values.dtype)
    counts = np.diff(np.append(starts, length)).astype(np.float32)
    shape = [1] * values.ndim
    shape[axis] = len(starts)
    return sums / counts.reshape(shape)


@dataclass(frozen=True)
class PyramidLevelKey:
    """Identity of one materialized pyramid level (ADR 0050 key contract)."""

    source_id: object
    tile_id: object
    component: str
    level_xy: tuple[int, int]
    algo_version: int = ALGO_VERSION

    def __post_init__(self) -> None:
        level_x, level_y = (int(self.level_xy[0]), int(self.level_xy[1]))
        if level_x < 0 or level_y < 0:
            raise ValueError("pyramid levels must be non-negative")
        object.__setattr__(self, "component", str(self.component))
        object.__setattr__(self, "level_xy", (level_x, level_y))
        object.__setattr__(self, "algo_version", int(self.algo_version))

    @property
    def factor_xy(self) -> tuple[int, int]:
        return (2 ** int(self.level_xy[0]), 2 ** int(self.level_xy[1]))

    @property
    def level(self) -> int:
        return max(int(self.level_xy[0]), int(self.level_xy[1]))


class PyramidCache:
    """Bounded, byte-accounted pyramid level cache with singleflight requests.

    Workers ``admit`` completed levels; the GUI thread only performs
    dictionary lookups.  ``begin_pending``/``end_pending`` provide the
    singleflight bookkeeping so duplicate materialization requests for the
    same key coalesce into one scheduled reduction.
    """

    def __init__(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._cache = BoundedCache(max_bytes=max_bytes, max_entries=max_entries)
        self._pending: set[PyramidLevelKey] = set()
        self._by_source: dict[tuple[object, int, str], set[PyramidLevelKey]] = {}
        self._lock = RLock()

    def lookup(self, key: PyramidLevelKey):
        """Return the cached level array, counting hit/miss."""

        return self._cache.get(key)

    def peek(self, key: PyramidLevelKey):
        """Return the cached level array without touching counters/recency."""

        return self._cache.peek(key)

    def admit(self, key: PyramidLevelKey, array) -> np.ndarray:
        """Admit a completed level and clear its pending claim."""

        values = np.asarray(array)
        with self._lock:
            if self._cache.would_fit(int(values.nbytes)):
                self._cache.put(key, values, nbytes=int(values.nbytes))
                self._by_source.setdefault(self._source_group(key), set()).add(key)
            self._pending.discard(key)
        return values

    @staticmethod
    def _source_group(key: PyramidLevelKey) -> tuple[object, int, str]:
        return (key.source_id, int(key.tile_id), str(key.component))

    def resident_keys_for(self, source_id, tile_id, component) -> tuple[PyramidLevelKey, ...]:
        """All currently cached level keys for one semantic tile.

        The index is pruned lazily against the bounded cache, so evicted
        levels disappear on the next enumeration; no eviction hook is
        required and the GUI-thread cost stays a few dictionary probes.
        """

        group = (source_id, int(tile_id), str(component))
        with self._lock:
            keys = self._by_source.get(group)
            if not keys:
                return ()
            live = tuple(key for key in keys if self._cache.peek(key) is not None)
            if len(live) != len(keys):
                if live:
                    self._by_source[group] = set(live)
                else:
                    self._by_source.pop(group, None)
            return live

    def begin_pending(self, key: PyramidLevelKey) -> bool:
        """Claim a materialization request; False when already cached/claimed."""

        with self._lock:
            if key in self._pending or self._cache.peek(key) is not None:
                return False
            self._pending.add(key)
            return True

    def end_pending(self, key: PyramidLevelKey) -> None:
        """Release a claim without admitting (cancelled/superseded/failed)."""

        with self._lock:
            self._pending.discard(key)

    def pending(self, key: PyramidLevelKey) -> bool:
        with self._lock:
            return key in self._pending

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def max_bytes(self) -> int | None:
        return self._cache.max_bytes

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

    def __contains__(self, key) -> bool:
        return key in self._cache

    def resize(self, *, max_bytes: int | None = None, max_entries: int | None = None) -> None:
        self._cache.resize(max_bytes=max_bytes, max_entries=max_entries)

    def resident_level_counts(self) -> dict[int, int]:
        """Return {scalar level: cached entry count} for diagnostics."""

        counts: dict[int, int] = {}
        for key, _value in self._cache.items():
            level = int(getattr(key, "level", 0))
            counts[level] = counts.get(level, 0) + 1
        return counts

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._pending.clear()


__all__ = ["ALGO_VERSION", "PyramidLevelKey", "PyramidCache", "reduce_box_mean"]
