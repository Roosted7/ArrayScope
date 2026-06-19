"""Semantic montage histogram and window/level source tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from arrayscope.core.window_levels import LevelSource, LevelSourceRank, normalize_bounds


PROVISIONAL_TILE_SAMPLE_LIMIT = 2048
REFINED_TILE_SAMPLE_LIMIT = 8192
EXACT_TILE_SAMPLE_LIMIT = 32768
AGGREGATE_SAMPLE_LIMIT = 262144


def montage_level_key(document_key, view_state, all_indices, colormap_lut) -> tuple[object, ...]:
    """Identity for semantic montage levels, independent of layout/viewport."""

    axis = view_state.montage_axis
    scope_state = view_state.with_montage_axis(axis, columns=None, indices=None, text=None)
    lut_key = None if colormap_lut is None else np.asarray(colormap_lut).tobytes()
    return (
        "montage_levels",
        document_key,
        scope_state,
        None if axis is None else int(axis),
        tuple(int(index) for index in all_indices),
        lut_key,
    )


@dataclass(frozen=True)
class TileLevelStats:
    source_index: int
    bounds: tuple[float, float]
    sample: np.ndarray
    refined: bool = False

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.sample).size)


@dataclass(frozen=True)
class MontageLevelStats:
    bounds: tuple[float, float] | None
    source_indices: frozenset[int]
    expected_indices: frozenset[int]
    rank: LevelSourceRank
    sample: np.ndarray | None = None
    refined: bool = False

    @property
    def coverage_rank(self) -> int:
        if self.rank == LevelSourceRank.NONE:
            return 0
        if self.rank in {LevelSourceRank.MONTAGE_COMPLETE, LevelSourceRank.MONTAGE_SAMPLED_FULL}:
            return 2
        return 1


class MontageLevelTracker:
    def __init__(self):
        self._tiles: dict[object, dict[int, TileLevelStats]] = {}
        self._expected: dict[object, frozenset[int]] = {}
        self._revisions: dict[object, int] = {}
        self._aggregate_cache: dict[object, tuple[int, frozenset[int], MontageLevelStats]] = {}

    def ensure(self, key: object, expected_indices: Iterable[int]) -> MontageLevelStats:
        expected = frozenset(int(index) for index in expected_indices)
        if self._expected.get(key) != expected:
            self._expected[key] = expected
            self._invalidate(key)
        self._tiles.setdefault(key, {})
        return self._stats_for_expected(key, expected)

    def update_from_tile(
        self,
        key: object,
        source_index: int,
        histogram_data: np.ndarray | None,
        image: np.ndarray,
        *,
        refined: bool = False,
        aggregate: bool = True,
    ) -> MontageLevelStats | None:
        expected = self._expected.get(key, frozenset())
        source = histogram_data if histogram_data is not None else image
        tile_stats = _sample_tile_stats(source, int(source_index), refined=bool(refined))
        if tile_stats is None:
            return self.stats_for(key) if aggregate else None
        by_source = self._tiles.setdefault(key, {})
        previous = by_source.get(int(source_index))
        if previous is None or bool(refined) or not previous.refined:
            by_source[int(source_index)] = tile_stats
            self._invalidate(key)
        return self.stats_for(key) if aggregate else None

    def best_source(self, key: object, *, explicit_auto: bool = False) -> LevelSource | None:
        del explicit_auto
        stats = self.stats_for(key)
        if stats is None:
            return None
        return self.source_for_stats(key, stats)

    def source_for_stats(self, key: object, stats: MontageLevelStats) -> LevelSource | None:
        if stats.bounds is None:
            return None
        return LevelSource(
            levels=stats.bounds,
            histogram_range=stats.bounds,
            rank=stats.rank,
            source_count=len(stats.source_indices),
            expected_count=len(stats.expected_indices),
            semantic_key=key,
        )

    def histogram_data_for_stats(self, stats: MontageLevelStats | None) -> np.ndarray | None:
        if stats is None or stats.sample is None or np.asarray(stats.sample).size == 0:
            return None
        return np.asarray(stats.sample, dtype=np.float32)

    def stats_for(self, key: object) -> MontageLevelStats | None:
        expected = self._expected.get(key)
        if expected is None:
            return None
        return self._stats_for_expected(key, expected)

    def as_dict(self) -> dict[object, MontageLevelStats]:
        return {key: self._stats_for_expected(key, expected) for key, expected in self._expected.items()}

    def _stats_for_expected(self, key: object, expected: frozenset[int]) -> MontageLevelStats:
        revision = int(self._revisions.get(key, 0))
        cached = self._aggregate_cache.get(key)
        if cached is not None and cached[0] == revision and cached[1] == expected:
            return cached[2]
        by_source = self._tiles.get(key, {})
        selected = [by_source[index] for index in sorted(expected) if index in by_source]
        if not selected:
            stats = MontageLevelStats(None, frozenset(), expected, LevelSourceRank.NONE, None, False)
        else:
            bounds = _union_tile_bounds(selected)
            sources = frozenset(stat.source_index for stat in selected)
            rank = self._rank_for(sources, expected)
            refined = bool(selected) and all(stat.refined for stat in selected)
            if rank == LevelSourceRank.MONTAGE_COMPLETE and refined:
                rank = LevelSourceRank.MONTAGE_SAMPLED_FULL
            sample = _aggregate_samples(tuple(stat.sample for stat in selected), AGGREGATE_SAMPLE_LIMIT)
            stats = MontageLevelStats(bounds, sources, expected, rank, sample, refined)
        self._aggregate_cache[key] = (revision, expected, stats)
        return stats

    def _invalidate(self, key: object) -> None:
        self._revisions[key] = int(self._revisions.get(key, 0)) + 1
        self._aggregate_cache.pop(key, None)

    def _rank_for(self, source_indices: Iterable[int], expected_indices: Iterable[int]) -> LevelSourceRank:
        sources = frozenset(int(index) for index in source_indices)
        expected = frozenset(int(index) for index in expected_indices)
        if not sources:
            return LevelSourceRank.NONE
        if expected and expected.issubset(sources):
            return LevelSourceRank.MONTAGE_COMPLETE
        return LevelSourceRank.MONTAGE_VISIBLE_SUBSET


def _sample_tile_stats(values, source_index: int, *, refined: bool) -> TileLevelStats | None:
    bounds = _finite_bounds(values)
    if bounds is None:
        return None
    sample = _finite_sample(values, limit=REFINED_TILE_SAMPLE_LIMIT if refined else PROVISIONAL_TILE_SAMPLE_LIMIT)
    return TileLevelStats(
        source_index=int(source_index),
        bounds=bounds,
        sample=sample.astype(np.float32, copy=False),
        refined=bool(refined or np.asarray(values).size <= EXACT_TILE_SAMPLE_LIMIT),
    )


def _finite_sample(values, *, limit: int) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return np.asarray((), dtype=np.float32)
    flat = array.reshape(-1)
    mask = np.isfinite(flat)
    finite = flat if bool(np.all(mask)) else flat[mask]
    if finite.size > int(limit):
        finite = _sparse_even_random_sample(finite, limit=int(limit))
    return np.asarray(finite, dtype=np.float32)


def _finite_bounds(values) -> tuple[float, float] | None:
    array = np.asarray(values)
    if array.size == 0:
        return None
    flat = array.reshape(-1)
    mask = np.isfinite(flat)
    finite = flat if bool(np.all(mask)) else flat[mask]
    if finite.size == 0:
        return None
    return normalize_bounds((float(np.min(finite)), float(np.max(finite))))


def _sparse_even_random_sample(finite: np.ndarray, *, limit: int) -> np.ndarray:
    limit = max(1, int(limit))
    values = np.asarray(finite)
    if values.size <= limit:
        return values
    even_count = max(1, limit // 2)
    random_count = max(0, limit - even_count)
    even_indices = np.linspace(0, values.size - 1, even_count, dtype=np.int64)
    if random_count:
        rng = np.random.default_rng(_sample_seed(values.size, limit))
        random_indices = rng.choice(values.size, size=min(random_count, values.size), replace=False)
        indices = np.unique(np.concatenate((even_indices, random_indices)))
    else:
        indices = np.unique(even_indices)
    if indices.size < limit:
        filler = np.linspace(0, values.size - 1, limit, dtype=np.int64)
        indices = np.unique(np.concatenate((indices, filler)))
    return values[np.sort(indices)[:limit]]


def _sample_seed(size: int, limit: int) -> int:
    return int((int(size) * 1_103_515_245 + int(limit) * 12_345) & 0xFFFFFFFF)


def _union_tile_bounds(stats: Iterable[TileLevelStats]) -> tuple[float, float] | None:
    lows = []
    highs = []
    for stat in stats:
        bounds = normalize_bounds(stat.bounds)
        if bounds is None:
            continue
        lows.append(bounds[0])
        highs.append(bounds[1])
    if not lows:
        return None
    return normalize_bounds((min(lows), max(highs)))


def _aggregate_samples(samples: tuple[np.ndarray, ...], limit: int) -> np.ndarray | None:
    non_empty = [np.asarray(sample, dtype=np.float32).reshape(-1) for sample in samples if np.asarray(sample).size]
    if not non_empty:
        return None
    limit = max(1, int(limit))
    total = sum(int(sample.size) for sample in non_empty)
    if total <= limit:
        return np.concatenate(non_empty)
    step = max(1, int(math.ceil(total / limit)))
    conceptual = np.arange(0, total, step, dtype=np.int64)[:limit]
    selected = []
    offset = 0
    for sample in non_empty:
        end = offset + int(sample.size)
        local = conceptual[(conceptual >= offset) & (conceptual < end)] - offset
        if local.size:
            selected.append(sample[local])
        offset = end
    return np.concatenate(selected) if selected else None
