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


@dataclass(frozen=True)
class SourceGridBinIdentity:
    """Value identity of one globally anchored reduced source rectangle."""

    source_rect_yx: tuple[int, int, int, int]
    reduction_vector_xy: tuple[int, int]
    reducer: str = "mean"
    algo_version: int = ALGO_VERSION


@dataclass(frozen=True)
class SourceGridReduction:
    """Reduced values plus the native-source coverage of every sample."""

    values: np.ndarray
    source_rects: tuple[tuple[int, int, int, int], ...]
    identities: tuple[SourceGridBinIdentity, ...]
    grid_origin_yx: tuple[int, int]
    reduction_vector_xy: tuple[int, int]
    valid_source_rect_yx: tuple[int, int, int, int]


@dataclass(frozen=True)
class SourceGridPageIdentity:
    """Identity of one uniform stored-sample page and its valid footprint."""

    source_rect_yx: tuple[int, int, int, int]
    reduction_vector_xy: tuple[int, int]
    reducer: str = "mean"
    algo_version: int = ALGO_VERSION


@dataclass(frozen=True)
class SourceGridPage:
    """One page of reduced values with exact per-sample draw coverage."""

    identity: SourceGridPageIdentity
    source_rect_yx: tuple[int, int, int, int]
    values: np.ndarray
    draw_source_rects: tuple[tuple[int, int, int, int], ...]


def partition_source_grid_pages(
    reduction: SourceGridReduction,
    *,
    stored_page_shape: tuple[int, int],
) -> tuple[SourceGridPage, ...]:
    """Partition reduced samples without losing their native draw spans.

    Page alignment is global native-source alignment.  Values are grouped by
    the page containing their bin origin; clipped first/last pages keep their
    exact valid footprint in identity, and every sample retains its own
    source rectangle for later backend geometry construction.
    """

    values = np.asarray(reduction.values)
    if values.ndim < 2:
        raise ValueError("source-grid page partition requires a 2D reduction")
    stored_h, stored_w = (int(value) for value in stored_page_shape)
    if stored_h <= 0 or stored_w <= 0:
        raise ValueError(f"stored page shape must be positive, got {stored_page_shape}")
    level_x, level_y = reduction.reduction_vector_xy
    source_page_h = stored_h * (1 << int(level_y))
    source_page_w = stored_w * (1 << int(level_x))
    rect_grid = np.asarray(reduction.source_rects, dtype=np.int64).reshape(
        values.shape[0], values.shape[1], 4
    )
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            rect = rect_grid[row, column]
            page_origin = (
                (int(rect[0]) // source_page_h) * source_page_h,
                (int(rect[2]) // source_page_w) * source_page_w,
            )
            groups.setdefault(page_origin, []).append((row, column))
    pages: list[SourceGridPage] = []
    for page_origin in sorted(groups):
        positions = groups[page_origin]
        rows = tuple(sorted({row for row, _column in positions}))
        columns = tuple(sorted({column for _row, column in positions}))
        if len(positions) != len(rows) * len(columns):
            raise ValueError("source-grid page samples must form one rectangular block")
        row_slice = slice(rows[0], rows[-1] + 1)
        column_slice = slice(columns[0], columns[-1] + 1)
        page_rects = rect_grid[row_slice, column_slice].reshape(-1, 4)
        source_rect = (
            int(np.min(page_rects[:, 0])),
            int(np.max(page_rects[:, 1])),
            int(np.min(page_rects[:, 2])),
            int(np.max(page_rects[:, 3])),
        )
        identity = SourceGridPageIdentity(
            source_rect_yx=source_rect,
            reduction_vector_xy=tuple(reduction.reduction_vector_xy),
        )
        pages.append(
            SourceGridPage(
                identity=identity,
                source_rect_yx=source_rect,
                values=np.ascontiguousarray(values[row_slice, column_slice]),
                draw_source_rects=tuple(tuple(int(value) for value in rect) for rect in page_rects),
            )
        )
    return tuple(pages)


def reduce_source_grid_mean(
    array,
    *,
    source_origin_yx: tuple[int, int],
    valid_source_rect_yx: tuple[int, int, int, int],
    reduction_vector_xy: tuple[int, int],
    input_reduction_vector_xy: tuple[int, int] = (0, 0),
) -> SourceGridReduction:
    """Reduce on the global native-source grid, independent of window origin.

    The reduction vectors are absolute log2 steps in ``(x, y)`` order.
    ``source_origin_yx`` locates input sample ``[0, 0]`` in native source
    coordinates. Recursive reduction is accepted only for a fully aligned
    input grid, which makes the result identical to direct source reduction;
    clipped partial parents are rejected instead of making cache history part
    of the values.
    """

    values = np.asarray(array)
    if values.ndim < 2 or values.ndim > 3:
        raise ValueError("source-grid mean requires 2D data plus at most one component axis")
    level_x, level_y = _reduction_vector(reduction_vector_xy, name="reduction")
    input_level_x, input_level_y = _reduction_vector(
        input_reduction_vector_xy,
        name="input reduction",
    )
    if level_x < input_level_x or level_y < input_level_y:
        raise ValueError("output reduction must not be finer than the input reduction")
    factor_x, factor_y = (1 << level_x, 1 << level_y)
    input_factor_x, input_factor_y = (1 << input_level_x, 1 << input_level_y)
    source_y, source_x = (int(source_origin_yx[0]), int(source_origin_yx[1]))
    valid_y0, valid_y1, valid_x0, valid_x1 = (
        int(value) for value in valid_source_rect_yx
    )
    if valid_y1 <= valid_y0 or valid_x1 <= valid_x0:
        raise ValueError("valid source rectangle must be non-empty")
    input_y1 = source_y + int(values.shape[0]) * input_factor_y
    input_x1 = source_x + int(values.shape[1]) * input_factor_x
    if not (
        source_y <= valid_y0 < valid_y1 <= input_y1
        and source_x <= valid_x0 < valid_x1 <= input_x1
    ):
        raise ValueError("valid source rectangle lies outside the input sample coverage")
    if (input_level_x or input_level_y) and (
        source_y % input_factor_y
        or source_x % input_factor_x
        or valid_y0 % input_factor_y
        or valid_y1 % input_factor_y
        or valid_x0 % input_factor_x
        or valid_x1 % input_factor_x
    ):
        raise ValueError("recursive input and valid coverage must align to its source grid")

    y_rects = _source_grid_axis_rects(valid_y0, valid_y1, factor_y)
    x_rects = _source_grid_axis_rects(valid_x0, valid_x1, factor_x)
    output_shape = (len(y_rects), len(x_rects), *values.shape[2:])
    output_dtype = np.complex64 if np.iscomplexobj(values) else np.float32
    reduced = np.empty(output_shape, dtype=output_dtype)
    source_rects: list[tuple[int, int, int, int]] = []
    identities: list[SourceGridBinIdentity] = []
    accumulated = values.astype(output_dtype, copy=False)
    for out_y, (rect_y0, rect_y1) in enumerate(y_rects):
        local_y0 = (rect_y0 - source_y) // input_factor_y
        local_y1 = (rect_y1 - source_y) // input_factor_y
        for out_x, (rect_x0, rect_x1) in enumerate(x_rects):
            local_x0 = (rect_x0 - source_x) // input_factor_x
            local_x1 = (rect_x1 - source_x) // input_factor_x
            source_rect = (rect_y0, rect_y1, rect_x0, rect_x1)
            sample = accumulated[local_y0:local_y1, local_x0:local_x1]
            if sample.size == 0:
                raise ValueError("source-grid bin has no input samples")
            reduced[out_y, out_x] = np.mean(sample, axis=(0, 1), dtype=output_dtype)
            source_rects.append(source_rect)
            identities.append(
                SourceGridBinIdentity(
                    source_rect_yx=source_rect,
                    reduction_vector_xy=(level_x, level_y),
                )
            )
    return SourceGridReduction(
        values=reduced,
        source_rects=tuple(source_rects),
        identities=tuple(identities),
        grid_origin_yx=(
            (valid_y0 // factor_y) * factor_y,
            (valid_x0 // factor_x) * factor_x,
        ),
        reduction_vector_xy=(level_x, level_y),
        valid_source_rect_yx=(valid_y0, valid_y1, valid_x0, valid_x1),
    )


def _reduction_vector(value, *, name: str) -> tuple[int, int]:
    level_x, level_y = (int(value[0]), int(value[1]))
    if level_x < 0 or level_y < 0:
        raise ValueError(f"{name} steps must be non-negative")
    return (level_x, level_y)


def _source_grid_axis_rects(start: int, stop: int, factor: int) -> tuple[tuple[int, int], ...]:
    first = (int(start) // int(factor)) * int(factor)
    return tuple(
        (max(int(start), origin), min(int(stop), origin + int(factor)))
        for origin in range(first, int(stop), int(factor))
    )


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
        self._revision = 0

    @property
    def revision(self) -> int:
        """Monotonic counter bumped whenever residency can change.

        GUI-side callers memoize per-tile resident-level scans against this:
        a memo guarded by ``revision`` is exact because admissions (and the
        evictions they trigger inside the same ``put``), explicit clears, and
        resizes bump it, while lookups/peeks/claims leave it unchanged.
        """

        return self._revision

    def lookup(self, key: PyramidLevelKey):
        """Return the cached level array, counting hit/miss."""

        return self._cache.get(key)

    def peek(self, key: PyramidLevelKey):
        """Return the cached level array without touching counters/recency."""

        return self._cache.peek(key)

    def peek_many(self, keys) -> dict[PyramidLevelKey, np.ndarray]:
        """Snapshot several resident levels with one cache-lock acquisition."""

        return self._cache.peek_many(keys)

    def admit(self, key: PyramidLevelKey, array) -> np.ndarray:
        """Admit a completed level and clear its pending claim."""

        values = np.asarray(array)
        with self._lock:
            if self._cache.would_fit(int(values.nbytes)):
                self._cache.put(key, values, nbytes=int(values.nbytes))
                self._by_source.setdefault(self._source_group(key), set()).add(key)
                self._revision += 1
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
        with self._lock:
            self._revision += 1

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
            self._revision += 1


__all__ = [
    "ALGO_VERSION",
    "PyramidLevelKey",
    "PyramidCache",
    "SourceGridBinIdentity",
    "SourceGridPage",
    "SourceGridPageIdentity",
    "SourceGridReduction",
    "partition_source_grid_pages",
    "reduce_box_mean",
    "reduce_source_grid_mean",
]


def preview_level_for_tile_shape(tile_shape, *, target_edge: int = 48, min_level: int = 2, max_level: int = 6) -> int:
    """Retained-preview level for one tile shape (ADR 0050).

    Coarse enough that a whole stack stays a few megabytes, fine enough to
    scroll through recognizably: the smallest power-of-two level whose
    reduced edges do not undershoot ``target_edge``.
    """

    edge = max(int(tile_shape[0]), int(tile_shape[1]), 1)
    level = 0
    while (edge >> (level + 1)) >= max(1, int(target_edge)) and level < int(max_level):
        level += 1
    return max(int(min_level), level)
