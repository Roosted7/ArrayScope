"""Semantic montage window/level source tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from arrayscope.display.levels import finite_bounds
from arrayscope.core.window_levels import LevelSource, LevelSourceRank, normalize_bounds


@dataclass(frozen=True)
class MontageLevelKey:
    document_key: object
    view_state: object
    montage_axis: int
    indices: tuple[int, ...]
    colormap_key: object | None


@dataclass(frozen=True)
class MontageLevelStats:
    bounds: tuple[float, float] | None
    source_indices: frozenset[int]
    expected_indices: frozenset[int]
    rank: LevelSourceRank

    @property
    def coverage_rank(self) -> int:
        if self.rank == LevelSourceRank.NONE:
            return 0
        if self.rank == LevelSourceRank.MONTAGE_COMPLETE:
            return 2
        return 1


class MontageLevelTracker:
    def __init__(self):
        self._stats: dict[object, MontageLevelStats] = {}

    def ensure(self, key: object, expected_indices: Iterable[int]) -> MontageLevelStats:
        expected = frozenset(int(index) for index in expected_indices)
        stats = self._stats.get(key)
        if stats is None:
            stats = MontageLevelStats(None, frozenset(), expected, LevelSourceRank.NONE)
            self._stats[key] = stats
        elif stats.expected_indices != expected:
            stats = MontageLevelStats(
                stats.bounds,
                stats.source_indices,
                expected,
                self._rank_for(stats.source_indices, expected),
            )
            self._stats[key] = stats
        return stats

    def update_from_tile(self, key: object, source_index: int, histogram_data: np.ndarray | None, image: np.ndarray) -> MontageLevelStats:
        previous = self._stats.get(key)
        if previous is None:
            previous = MontageLevelStats(None, frozenset(), frozenset(), LevelSourceRank.NONE)
        source = histogram_data if histogram_data is not None else image
        bounds = normalize_bounds(finite_bounds(source))
        if bounds is None:
            return previous
        if previous.bounds is not None:
            bounds = (min(previous.bounds[0], bounds[0]), max(previous.bounds[1], bounds[1]))
        source_indices = frozenset(set(previous.source_indices) | {int(source_index)})
        stats = MontageLevelStats(
            bounds=bounds,
            source_indices=source_indices,
            expected_indices=previous.expected_indices,
            rank=self._rank_for(source_indices, previous.expected_indices),
        )
        self._stats[key] = stats
        return stats

    def best_source(self, key: object, *, explicit_auto: bool = False) -> LevelSource | None:
        stats = self._stats.get(key)
        if stats is None or stats.bounds is None:
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

    def stats_for(self, key: object) -> MontageLevelStats | None:
        return self._stats.get(key)

    def as_dict(self) -> dict[object, MontageLevelStats]:
        return dict(self._stats)

    def _rank_for(self, source_indices: Iterable[int], expected_indices: Iterable[int]) -> LevelSourceRank:
        sources = frozenset(int(index) for index in source_indices)
        expected = frozenset(int(index) for index in expected_indices)
        if not sources:
            return LevelSourceRank.NONE
        if expected and expected.issubset(sources):
            return LevelSourceRank.MONTAGE_COMPLETE
        return LevelSourceRank.MONTAGE_VISIBLE_SUBSET
