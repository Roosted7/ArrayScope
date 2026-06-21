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


def montage_level_key(document_key, view_state, all_indices=None, colormap_lut=None) -> tuple[object, ...]:
    """Identity for semantic montage levels, independent of coverage and presentation.

    ``all_indices`` describes the currently requested coverage population, not
    the scalar identity of a tile.  Keeping it out of the key lets panning,
    viewport expansion, and partial retargeting reuse already sampled tile
    statistics instead of resetting histogram/window state.  The user's
    selected montage population remains part of the semantic scope, because
    changing it changes which source population window/level should represent.
    LUTs likewise change colours rather than scalar values.
    """

    del all_indices, colormap_lut
    axis = view_state.montage_axis
    selected_indices = None if view_state.montage_indices is None else tuple(int(index) for index in view_state.montage_indices)
    scope_state = view_state.with_montage_axis(axis, columns=None, indices=selected_indices, text=None)
    return (
        "montage_levels",
        document_key,
        scope_state,
        None if axis is None else int(axis),
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
        self._summary_cache: dict[object, tuple[int, frozenset[int], MontageLevelStats]] = {}
        self._sample_accumulators: dict[object, tuple[frozenset[int], frozenset[int], np.ndarray | None]] = {}

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
            if previous is None and int(source_index) in expected:
                self._append_tile_sample(key, expected, tile_stats)
            else:
                self._sample_accumulators.pop(key, None)
        return self.stats_for(key) if aggregate else None

    def has_source(self, key: object, source_index: int, *, refined: bool = False) -> bool:
        """Return whether reusable statistics already exist for one source."""

        stats = self._tiles.get(key, {}).get(int(source_index))
        if stats is None:
            return False
        return bool(stats.refined) if refined else True

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

    def summary_for(self, key: object) -> MontageLevelStats | None:
        """Return bounds/rank/coverage without rebuilding aggregate samples."""

        expected = self._expected.get(key)
        if expected is None:
            return None
        revision = int(self._revisions.get(key, 0))
        cached = self._summary_cache.get(key)
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
            stats = MontageLevelStats(bounds, sources, expected, rank, None, refined)
        self._summary_cache[key] = (revision, expected, stats)
        return stats

    def as_dict(self) -> dict[object, MontageLevelStats]:
        return {key: self._stats_for_expected(key, expected) for key, expected in self._expected.items()}

    def _stats_for_expected(self, key: object, expected: frozenset[int]) -> MontageLevelStats:
        revision = int(self._revisions.get(key, 0))
        cached = self._aggregate_cache.get(key)
        if cached is not None and cached[0] == revision and cached[1] == expected:
            return cached[2]
        summary = self.summary_for(key)
        if summary is None or not summary.source_indices:
            stats = MontageLevelStats(None, frozenset(), expected, LevelSourceRank.NONE, None, False)
        else:
            sample = self._sample_for_expected(key, expected, summary.source_indices)
            stats = MontageLevelStats(
                summary.bounds,
                summary.source_indices,
                summary.expected_indices,
                summary.rank,
                sample,
                summary.refined,
            )
        self._aggregate_cache[key] = (revision, expected, stats)
        return stats

    def _invalidate(self, key: object) -> None:
        self._revisions[key] = int(self._revisions.get(key, 0)) + 1
        self._aggregate_cache.pop(key, None)
        self._summary_cache.pop(key, None)

    def _append_tile_sample(self, key: object, expected: frozenset[int], tile_stats: TileLevelStats) -> None:
        previous = self._sample_accumulators.get(key)
        if previous is None:
            self._sample_accumulators[key] = (
                expected,
                frozenset({int(tile_stats.source_index)}),
                np.asarray(tile_stats.sample, dtype=np.float32).reshape(-1),
            )
            return
        previous_expected, previous_sources, previous_sample = previous
        source = int(tile_stats.source_index)
        if previous_expected != expected or source in previous_sources:
            self._sample_accumulators.pop(key, None)
            return
        sample = np.asarray(tile_stats.sample, dtype=np.float32).reshape(-1)
        if previous_sample is None or np.asarray(previous_sample).size == 0:
            merged = sample
        elif sample.size == 0:
            merged = np.asarray(previous_sample, dtype=np.float32).reshape(-1)
        else:
            merged = _merge_incremental_samples(previous_sample, sample, AGGREGATE_SAMPLE_LIMIT)
        self._sample_accumulators[key] = (expected, frozenset((*previous_sources, source)), merged)

    def _sample_for_expected(self, key: object, expected: frozenset[int], sources: frozenset[int]) -> np.ndarray | None:
        cached = self._sample_accumulators.get(key)
        if cached is not None and cached[0] == expected and cached[1] == sources:
            return cached[2]
        by_source = self._tiles.get(key, {})
        selected = tuple(by_source[index].sample for index in sorted(expected) if index in by_source)
        sample = _aggregate_samples(selected, AGGREGATE_SAMPLE_LIMIT)
        self._sample_accumulators[key] = (expected, sources, sample)
        return sample

    def _rank_for(self, source_indices: Iterable[int], expected_indices: Iterable[int]) -> LevelSourceRank:
        sources = frozenset(int(index) for index in source_indices)
        expected = frozenset(int(index) for index in expected_indices)
        if not sources:
            return LevelSourceRank.NONE
        if expected and expected.issubset(sources):
            return LevelSourceRank.MONTAGE_COMPLETE
        return LevelSourceRank.MONTAGE_VISIBLE_SUBSET


def _sample_tile_stats(values, source_index: int, *, refined: bool) -> TileLevelStats | None:
    finite = _finite_values(values)
    if finite is None:
        return None
    bounds = normalize_bounds((float(np.min(finite)), float(np.max(finite))))
    sample = finite
    limit = REFINED_TILE_SAMPLE_LIMIT if refined else PROVISIONAL_TILE_SAMPLE_LIMIT
    if sample.size > int(limit):
        sample = _sparse_even_random_sample(sample, limit=int(limit))
    return TileLevelStats(
        source_index=int(source_index),
        bounds=bounds,
        sample=sample.astype(np.float32, copy=False),
        refined=bool(refined or np.asarray(values).size <= EXACT_TILE_SAMPLE_LIMIT),
    )


def _finite_sample(values, *, limit: int) -> np.ndarray:
    finite = _finite_values(values)
    if finite is None:
        return np.asarray((), dtype=np.float32)
    if finite.size > int(limit):
        finite = _sparse_even_random_sample(finite, limit=int(limit))
    return np.asarray(finite, dtype=np.float32)


def _finite_bounds(values) -> tuple[float, float] | None:
    finite = _finite_values(values)
    if finite is None:
        return None
    return normalize_bounds((float(np.min(finite)), float(np.max(finite))))


def _finite_values(values) -> np.ndarray | None:
    """Return finite flattened values with one finite-mask pass."""

    array = np.asarray(values)
    if array.size == 0:
        return None
    flat = array.reshape(-1)
    mask = np.isfinite(flat)
    if bool(np.all(mask)):
        return flat
    finite = flat[mask]
    return finite if finite.size else None


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


def _merge_incremental_samples(existing: np.ndarray, addition: np.ndarray, limit: int) -> np.ndarray:
    """Merge a new tile sample without revisiting every previous tile sample."""

    existing = np.asarray(existing, dtype=np.float32).reshape(-1)
    addition = np.asarray(addition, dtype=np.float32).reshape(-1)
    limit = max(1, int(limit))
    if existing.size == 0:
        return addition[:limit]
    if addition.size == 0:
        return existing[:limit]
    total = int(existing.size + addition.size)
    if total <= limit:
        return np.concatenate((existing, addition)).astype(np.float32, copy=False)
    keep_existing = min(existing.size, max(1, int(round(limit * (existing.size / total)))))
    keep_addition = max(0, limit - keep_existing)
    existing_indices = np.linspace(0, existing.size - 1, keep_existing, dtype=np.int64)
    if keep_addition <= 0:
        return existing[existing_indices].astype(np.float32, copy=False)
    addition_indices = np.linspace(0, addition.size - 1, keep_addition, dtype=np.int64)
    return np.concatenate((existing[existing_indices], addition[addition_indices])).astype(np.float32, copy=False)
