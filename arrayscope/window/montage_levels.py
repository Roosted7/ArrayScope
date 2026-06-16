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

    def ensure(self, key: object, expected_indices: Iterable[int]) -> MontageLevelStats:
        expected = frozenset(int(index) for index in expected_indices)
        self._expected[key] = expected
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
    ) -> MontageLevelStats:
        expected = self._expected.get(key, frozenset())
        source = histogram_data if histogram_data is not None else image
        tile_stats = _sample_tile_stats(source, int(source_index), refined=bool(refined))
        if tile_stats is None:
            return self._stats_for_expected(key, expected)
        by_source = self._tiles.setdefault(key, {})
        previous = by_source.get(int(source_index))
        if previous is None or bool(refined) or not previous.refined:
            by_source[int(source_index)] = tile_stats
        return self._stats_for_expected(key, expected)

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
        by_source = self._tiles.get(key, {})
        selected = [by_source[index] for index in sorted(expected) if index in by_source]
        if not selected:
            return MontageLevelStats(None, frozenset(), expected, LevelSourceRank.NONE, None, False)
        bounds = _union_tile_bounds(selected)
        sources = frozenset(stat.source_index for stat in selected)
        rank = self._rank_for(sources, expected)
        refined = bool(selected) and all(stat.refined for stat in selected)
        if rank == LevelSourceRank.MONTAGE_COMPLETE and refined:
            rank = LevelSourceRank.MONTAGE_SAMPLED_FULL
        sample = _aggregate_samples(tuple(stat.sample for stat in selected), AGGREGATE_SAMPLE_LIMIT)
        return MontageLevelStats(bounds, sources, expected, rank, sample, refined)

    def _rank_for(self, source_indices: Iterable[int], expected_indices: Iterable[int]) -> LevelSourceRank:
        sources = frozenset(int(index) for index in source_indices)
        expected = frozenset(int(index) for index in expected_indices)
        if not sources:
            return LevelSourceRank.NONE
        if expected and expected.issubset(sources):
            return LevelSourceRank.MONTAGE_COMPLETE
        return LevelSourceRank.MONTAGE_VISIBLE_SUBSET


def _sample_tile_stats(values, source_index: int, *, refined: bool) -> TileLevelStats | None:
    sample = _finite_sample(values, limit=REFINED_TILE_SAMPLE_LIMIT if refined else PROVISIONAL_TILE_SAMPLE_LIMIT)
    if sample.size == 0:
        return None
    low = float(np.min(sample))
    high = float(np.max(sample))
    bounds = normalize_bounds((low, high))
    if bounds is None:
        return None
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
    if flat.size > EXACT_TILE_SAMPLE_LIMIT:
        step = max(1, int(math.ceil(flat.size / max(1, int(limit)))))
        flat = flat[::step]
    finite = flat[np.isfinite(flat)]
    if finite.size > int(limit):
        step = max(1, int(math.ceil(finite.size / max(1, int(limit)))))
        finite = finite[::step][: int(limit)]
    return np.asarray(finite, dtype=np.float32)


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
    combined = np.concatenate(non_empty)
    if combined.size <= int(limit):
        return combined
    step = max(1, int(math.ceil(combined.size / max(1, int(limit)))))
    return combined[::step][: int(limit)]
