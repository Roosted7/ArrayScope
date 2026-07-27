"""Semantic montage histogram and window/level source tracking."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from arrayscope.core.window_levels import LevelSource, LevelSourceRank, normalize_bounds

PROVISIONAL_TILE_SAMPLE_LIMIT = 512
REFINED_TILE_SAMPLE_LIMIT = 8192
EXACT_TILE_SAMPLE_LIMIT = 32768
AGGREGATE_SAMPLE_LIMIT = 65536


class LevelEvidenceQuality(IntEnum):
    """Quality ordering for reusable montage level evidence."""

    NONE = 0
    ROUGH_PREVIEW = 1
    ROUGH_TARGET = 2
    REFINED = 3


# Refined first-frame evidence is worker-side NumPy sampling. Four sources per
# submission made a 60-tile PyQtGraph successor wait through 15 kernel/Qt
# round-trips (~1.2 s before the first atomic frame). Sixteen keeps the merge
# callback bounded while reducing the visible dependency to four handoffs.
# This is also the provisional first-pixel threshold: one refined batch is the
# minimum honest window source for a cold CPU-windowed scope larger than the
# batch (montage-entry blackout, 2026-07-18 dossier).
MONTAGE_LEVEL_STATS_FIRST_CPU_BATCH = 16


def montage_level_key(
    document_key, view_state, all_indices=None, colormap_lut=None
) -> tuple[object, ...]:
    """Identity for semantic montage levels, independent of coverage and presentation.

    ``all_indices`` describes the currently requested coverage population, not
    the scalar identity of a tile.  Keeping it out of the key lets panning,
    viewport expansion, and partial retargeting reuse already sampled tile
    statistics instead of resetting histogram/window state.  The user's
    selected montage population remains part of the semantic scope, because
    changing it changes which source population window/level should represent.
    LUTs likewise change colours rather than scalar values.
    """

    axis = view_state.montage_axis
    selected_indices = (
        None
        if view_state.montage_indices is None
        else tuple(int(index) for index in view_state.montage_indices)
    )
    scope_state = view_state.with_montage_axis(
        axis, columns=None, indices=selected_indices, text=None
    )
    return (
        "montage_levels",
        document_key,
        scope_state,
        None if axis is None else int(axis),
    )


@dataclass(frozen=True)
class TileLevelStats:
    source_index: int
    bounds: tuple[float, float] | None
    sample: np.ndarray
    refined: bool = False
    evidence_quality: LevelEvidenceQuality | int | str = LevelEvidenceQuality.ROUGH_TARGET

    def __post_init__(self) -> None:
        quality = _coerce_evidence_quality(self.evidence_quality)
        if bool(self.refined) and quality < LevelEvidenceQuality.REFINED:
            quality = LevelEvidenceQuality.REFINED
        object.__setattr__(self, "evidence_quality", quality)
        object.__setattr__(self, "refined", bool(quality >= LevelEvidenceQuality.REFINED))

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
    evidence_quality: LevelEvidenceQuality | int | str = LevelEvidenceQuality.NONE

    def __post_init__(self) -> None:
        quality = _coerce_evidence_quality(self.evidence_quality)
        if bool(self.refined) and quality < LevelEvidenceQuality.REFINED:
            quality = LevelEvidenceQuality.REFINED
        object.__setattr__(self, "evidence_quality", quality)
        object.__setattr__(self, "refined", bool(quality >= LevelEvidenceQuality.REFINED))

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
        self._sample_accumulators: dict[
            object, tuple[frozenset[int], frozenset[int], np.ndarray | None]
        ] = {}

    def ensure(self, key: object, expected_indices: Iterable[int]) -> MontageLevelStats:
        expected = self.ensure_expected(key, expected_indices)
        return self._stats_for_expected(key, expected)

    def ensure_expected(self, key: object, expected_indices: Iterable[int]) -> frozenset[int]:
        """Record expected coverage without rebuilding aggregate statistics."""

        expected = frozenset(int(index) for index in expected_indices)
        if self._expected.get(key) != expected:
            self._expected[key] = expected
            self._invalidate(key)
        self._tiles.setdefault(key, {})
        return expected

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
        if _tile_stats_is_improvement(tile_stats, previous):
            by_source[int(source_index)] = tile_stats
            self._invalidate(key)
            if aggregate and previous is None and int(source_index) in expected:
                self._append_tile_sample(key, expected, tile_stats)
            else:
                self._sample_accumulators.pop(key, None)
        return self.stats_for(key) if aggregate else None

    def update_from_stats(
        self,
        key: object,
        tile_stats: TileLevelStats,
        *,
        aggregate: bool = True,
    ) -> MontageLevelStats | None:
        expected = self._expected.get(key, frozenset())
        by_source = self._tiles.setdefault(key, {})
        source_index = int(tile_stats.source_index)
        previous = by_source.get(source_index)
        if _tile_stats_is_improvement(tile_stats, previous):
            by_source[source_index] = tile_stats
            self._invalidate(key)
            if aggregate and previous is None and source_index in expected:
                self._append_tile_sample(key, expected, tile_stats)
            else:
                self._sample_accumulators.pop(key, None)
        return self.stats_for(key) if aggregate else None

    def install_cohort(
        self,
        key: object,
        tile_stats: Iterable[TileLevelStats],
        *,
        expected_indices: Iterable[int],
    ) -> MontageLevelStats:
        """Install one complete round cohort as a single observable decision.

        The preview worker returns all of its tile statistics together.  Build
        the replacement map off to the side, validate exact population
        coverage, then advance the tracker revision once.  Consumers can
        therefore observe either the predecessor or the complete round source,
        never a GUI-loop accumulation of per-tile preview results.
        """

        expected = frozenset(int(index) for index in expected_indices)
        rows = tuple(tile_stats)
        by_source = {int(stats.source_index): stats for stats in rows}
        if len(by_source) != len(rows):
            raise ValueError("round level cohort repeats a source")
        if frozenset(by_source) != expected:
            missing = tuple(sorted(expected - frozenset(by_source)))
            extra = tuple(sorted(frozenset(by_source) - expected))
            raise ValueError(
                "round level cohort does not match its expected population: "
                f"missing={missing}, extra={extra}"
            )

        installed = {source_index: by_source[source_index] for source_index in sorted(expected)}
        self._expected[key] = expected
        self._tiles[key] = installed
        self._sample_accumulators.pop(key, None)
        self._invalidate(key)
        return self._stats_for_expected(key, expected)

    def widen_from_stats(self, key: object, tile_stats: TileLevelStats) -> bool:
        """Merge one shader-frame source without narrowing prior coverage."""

        source_index = int(tile_stats.source_index)
        previous = self._tiles.get(key, {}).get(source_index)
        candidate = tile_stats
        if previous is not None:
            previous_bounds = normalize_bounds(previous.bounds)
            candidate_bounds = normalize_bounds(candidate.bounds)
            if previous_bounds is not None:
                bounds = previous_bounds
                if candidate_bounds is not None:
                    bounds = normalize_bounds(
                        (
                            min(previous_bounds[0], candidate_bounds[0]),
                            max(previous_bounds[1], candidate_bounds[1]),
                        )
                    )
                sample = _merge_incremental_samples(
                    np.asarray(previous.sample),
                    np.asarray(candidate.sample),
                    REFINED_TILE_SAMPLE_LIMIT,
                )
                quality = max(
                    _coerce_evidence_quality(previous.evidence_quality),
                    _coerce_evidence_quality(candidate.evidence_quality),
                )
                candidate = TileLevelStats(
                    source_index=source_index,
                    bounds=bounds,
                    sample=sample,
                    refined=bool(quality >= LevelEvidenceQuality.REFINED),
                    evidence_quality=quality,
                )
        if not _tile_stats_is_improvement(candidate, previous):
            return False
        self._tiles.setdefault(key, {})[source_index] = candidate
        self._sample_accumulators.pop(key, None)
        self._invalidate(key)
        return True

    def record_vacuous_source(self, key: object, source_index: int) -> None:
        """Record refined evidence for a source with no finite values.

        An all-NaN/empty tile (e.g. a log-scaled constant plane) can never
        produce bounds; without an explicit record, level convergence waits
        on it forever (2026-07-05: pyqtgraph+resident auto-levels flush
        parked on one such source with rank stuck below SAMPLED_FULL).
        Vacuous evidence contributes no bounds and no sample — it only says
        "sampled; nothing to contribute", which is what completion means.
        """

        by_source = self._tiles.setdefault(key, {})
        previous = by_source.get(int(source_index))
        if previous is None or not previous.refined:
            by_source[int(source_index)] = TileLevelStats(
                source_index=int(source_index),
                bounds=None,
                sample=np.asarray((), dtype=np.float32),
                refined=True,
                evidence_quality=LevelEvidenceQuality.REFINED,
            )
            self._invalidate(key)

    def has_source(self, key: object, source_index: int, *, refined: bool = False) -> bool:
        """Return whether reusable statistics already exist for one source."""

        stats = self._tiles.get(key, {}).get(int(source_index))
        if stats is None:
            return False
        return bool(stats.refined) if refined else True

    def source_stats(self, key: object, source_index: int) -> TileLevelStats | None:
        return self._tiles.get(key, {}).get(int(source_index))

    def has_source_quality(
        self, key: object, source_index: int, quality: LevelEvidenceQuality | int | str
    ) -> bool:
        stats = self.source_stats(key, source_index)
        if stats is None:
            return False
        return _coerce_evidence_quality(stats.evidence_quality) >= _coerce_evidence_quality(quality)

    def best_source(self, key: object, *, explicit_auto: bool = False) -> LevelSource | None:
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
            evidence_quality=int(stats.evidence_quality),
        )

    def histogram_data_for_stats(self, stats: MontageLevelStats | None) -> np.ndarray | None:
        if stats is None or stats.sample is None or np.asarray(stats.sample).size == 0:
            return None
        return np.asarray(stats.sample, dtype=np.float32)

    def histogram_aggregate_snapshot(
        self,
        key: object,
    ) -> tuple[int, frozenset[int], frozenset[int], tuple[np.ndarray, ...]] | None:
        """Return read-only inputs for worker-side aggregate sampling."""

        summary = self.summary_for(key)
        if summary is None or not summary.source_indices:
            return None
        expected = self._expected.get(key, frozenset())
        by_source = self._tiles.get(key, {})
        samples = tuple(
            np.asarray(by_source[index].sample, dtype=np.float32).reshape(-1)
            for index in sorted(summary.source_indices)
            if index in by_source
        )
        return (
            int(self._revisions.get(key, 0)),
            expected,
            summary.source_indices,
            samples,
        )

    def cached_histogram_data(self, key: object) -> np.ndarray | None:
        """Return a current aggregate without deriving one on the caller."""

        summary = self.summary_for(key)
        if summary is None or not summary.source_indices:
            return None
        cached = self._sample_accumulators.get(key)
        if (
            cached is None
            or cached[0] != summary.expected_indices
            or cached[1] != summary.source_indices
        ):
            return None
        sample = cached[2]
        if sample is None or np.asarray(sample).size == 0:
            return None
        return np.asarray(sample, dtype=np.float32)

    def install_histogram_aggregate(
        self,
        key: object,
        *,
        revision: int,
        expected_indices: frozenset[int],
        source_indices: frozenset[int],
        sample: np.ndarray | None,
    ) -> bool:
        """Install a derived aggregate only for the exact live snapshot."""

        if int(self._revisions.get(key, 0)) != int(revision):
            return False
        summary = self.summary_for(key)
        if (
            summary is None
            or summary.expected_indices != expected_indices
            or summary.source_indices != source_indices
        ):
            return False
        self._sample_accumulators[key] = (
            expected_indices,
            source_indices,
            None if sample is None else np.asarray(sample, dtype=np.float32),
        )
        self._aggregate_cache.pop(key, None)
        return True

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
            stats = MontageLevelStats(
                None,
                frozenset(),
                expected,
                LevelSourceRank.NONE,
                None,
                False,
                LevelEvidenceQuality.NONE,
            )
        else:
            bounds = _union_tile_bounds(selected)
            sources = frozenset(stat.source_index for stat in selected)
            rank = self._rank_for(sources, expected)
            refined = bool(selected) and all(stat.refined for stat in selected)
            evidence_quality = min(
                (_coerce_evidence_quality(stat.evidence_quality) for stat in selected),
                default=LevelEvidenceQuality.NONE,
            )
            if rank == LevelSourceRank.MONTAGE_COMPLETE and refined:
                rank = LevelSourceRank.MONTAGE_SAMPLED_FULL
            stats = MontageLevelStats(
                bounds, sources, expected, rank, None, refined, evidence_quality
            )
        self._summary_cache[key] = (revision, expected, stats)
        return stats

    def as_dict(self) -> dict[object, MontageLevelStats]:
        return {
            key: self._stats_for_expected(key, expected) for key, expected in self._expected.items()
        }

    def _stats_for_expected(self, key: object, expected: frozenset[int]) -> MontageLevelStats:
        revision = int(self._revisions.get(key, 0))
        cached = self._aggregate_cache.get(key)
        if cached is not None and cached[0] == revision and cached[1] == expected:
            return cached[2]
        summary = self.summary_for(key)
        if summary is None or not summary.source_indices:
            stats = MontageLevelStats(
                None,
                frozenset(),
                expected,
                LevelSourceRank.NONE,
                None,
                False,
                LevelEvidenceQuality.NONE,
            )
        else:
            sample = self._sample_for_expected(key, expected, summary.source_indices)
            stats = MontageLevelStats(
                summary.bounds,
                summary.source_indices,
                summary.expected_indices,
                summary.rank,
                sample,
                summary.refined,
                summary.evidence_quality,
            )
        self._aggregate_cache[key] = (revision, expected, stats)
        return stats

    def _invalidate(self, key: object) -> None:
        self._revisions[key] = int(self._revisions.get(key, 0)) + 1
        self._aggregate_cache.pop(key, None)
        self._summary_cache.pop(key, None)

    def _append_tile_sample(
        self, key: object, expected: frozenset[int], tile_stats: TileLevelStats
    ) -> None:
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

    def _sample_for_expected(
        self, key: object, expected: frozenset[int], sources: frozenset[int]
    ) -> np.ndarray | None:
        cached = self._sample_accumulators.get(key)
        if cached is not None and cached[0] == expected and cached[1] == sources:
            return cached[2]
        by_source = self._tiles.get(key, {})
        selected = tuple(
            by_source[index].sample for index in sorted(expected) if index in by_source
        )
        sample = _aggregate_samples(selected, AGGREGATE_SAMPLE_LIMIT)
        self._sample_accumulators[key] = (expected, sources, sample)
        return sample

    def _rank_for(
        self, source_indices: Iterable[int], expected_indices: Iterable[int]
    ) -> LevelSourceRank:
        sources = frozenset(int(index) for index in source_indices)
        expected = frozenset(int(index) for index in expected_indices)
        if not sources:
            return LevelSourceRank.NONE
        if expected and expected.issubset(sources):
            return LevelSourceRank.MONTAGE_COMPLETE
        return LevelSourceRank.MONTAGE_VISIBLE_SUBSET


def _sample_tile_stats(
    values,
    source_index: int,
    *,
    refined: bool,
    exact: bool = True,
    evidence_quality: LevelEvidenceQuality | int | str | None = None,
) -> TileLevelStats | None:
    finite = _finite_values(values)
    if finite is None:
        return None
    bounds = normalize_bounds((float(np.min(finite)), float(np.max(finite))))
    sample = finite
    limit = REFINED_TILE_SAMPLE_LIMIT if refined else PROVISIONAL_TILE_SAMPLE_LIMIT
    if sample.size > int(limit):
        sample = _sparse_even_random_sample(sample, limit=int(limit))
    requested_quality = (
        None if evidence_quality is None else _coerce_evidence_quality(evidence_quality)
    )
    allow_exact_promotion = requested_quality != LevelEvidenceQuality.ROUGH_PREVIEW
    is_refined = bool(
        refined
        or (
            allow_exact_promotion
            and bool(exact)
            and np.asarray(values).size <= EXACT_TILE_SAMPLE_LIMIT
        )
    )
    quality = (
        LevelEvidenceQuality.REFINED
        if is_refined
        else _coerce_evidence_quality(requested_quality or LevelEvidenceQuality.ROUGH_TARGET)
    )
    return TileLevelStats(
        source_index=int(source_index),
        bounds=bounds,
        sample=sample.astype(np.float32, copy=False),
        refined=is_refined,
        evidence_quality=quality,
    )


def sample_tile_level_stats(
    values,
    source_index: int,
    *,
    refined: bool,
    evidence_quality: LevelEvidenceQuality | int | str | None = None,
) -> TileLevelStats | None:
    return _sample_tile_stats(
        values,
        int(source_index),
        refined=bool(refined),
        evidence_quality=evidence_quality,
    )


def provisional_tile_level_stats(
    values,
    source_index: int,
    *,
    evidence_quality: LevelEvidenceQuality | int | str = LevelEvidenceQuality.ROUGH_TARGET,
) -> TileLevelStats | None:
    return _sample_tile_stats(
        values,
        int(source_index),
        refined=False,
        exact=False,
        evidence_quality=evidence_quality,
    )


def tile_level_stats_with_quality(
    stats: TileLevelStats,
    quality: LevelEvidenceQuality | int | str,
    *,
    source_index: int,
) -> TileLevelStats:
    quality = _coerce_evidence_quality(quality)
    return TileLevelStats(
        source_index=int(source_index),
        bounds=stats.bounds,
        sample=np.asarray(stats.sample),
        refined=bool(quality >= LevelEvidenceQuality.REFINED),
        evidence_quality=quality,
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
    if np.iscomplexobj(array):
        array = np.abs(array).astype(np.float32, copy=False)
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
        indices = _sorted_unique_indices(np.concatenate((even_indices, random_indices)))
    else:
        indices = _sorted_unique_indices(even_indices)
    if indices.size < limit:
        filler = np.linspace(0, values.size - 1, limit, dtype=np.int64)
        indices = _sorted_unique_indices(np.concatenate((indices, filler)))
    return values[indices[:limit]]


def _sorted_unique_indices(indices: np.ndarray) -> np.ndarray:
    sorted_indices = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
    if sorted_indices.size < 2:
        return sorted_indices
    keep = np.empty(sorted_indices.shape, dtype=bool)
    keep[0] = True
    keep[1:] = sorted_indices[1:] != sorted_indices[:-1]
    return sorted_indices[keep]


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
    non_empty = []
    for sample in samples:
        values = np.asarray(sample, dtype=np.float32).reshape(-1)
        if values.size:
            non_empty.append(values)
    if not non_empty:
        return None
    limit = max(1, int(limit))
    total = sum(int(sample.size) for sample in non_empty)
    if total <= limit:
        return np.concatenate(non_empty)
    step = max(1, math.ceil(total / limit))
    selected = []
    offset = 0
    for sample in non_empty:
        end = offset + int(sample.size)
        # Conceptual selections are k * step. Intersect that progression with
        # this tile arithmetically instead of allocating/filtering the full
        # aggregate index vector once for every tile.
        first_k = (offset + step - 1) // step
        stop_k = min((end + step - 1) // step, limit)
        local = (
            np.arange(first_k, stop_k, dtype=np.int64) * step - offset
            if first_k < stop_k
            else np.asarray((), dtype=np.int64)
        )
        if local.size:
            selected.append(sample[local])
        offset = end
    return np.concatenate(selected) if selected else None


def aggregate_histogram_samples(samples: tuple[np.ndarray, ...]) -> np.ndarray | None:
    """Build the bounded montage histogram sample on a worker thread."""

    return _aggregate_samples(tuple(samples or ()), AGGREGATE_SAMPLE_LIMIT)


def _merge_incremental_samples(
    existing: np.ndarray, addition: np.ndarray, limit: int
) -> np.ndarray:
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
    keep_existing = min(existing.size, max(1, round(limit * (existing.size / total))))
    keep_addition = max(0, limit - keep_existing)
    existing_indices = np.linspace(0, existing.size - 1, keep_existing, dtype=np.int64)
    if keep_addition <= 0:
        return existing[existing_indices].astype(np.float32, copy=False)
    addition_indices = np.linspace(0, addition.size - 1, keep_addition, dtype=np.int64)
    return np.concatenate((existing[existing_indices], addition[addition_indices])).astype(
        np.float32, copy=False
    )


def _coerce_evidence_quality(value) -> LevelEvidenceQuality:
    if isinstance(value, LevelEvidenceQuality):
        return value
    if isinstance(value, str):
        try:
            return LevelEvidenceQuality[value]
        except KeyError:
            return LevelEvidenceQuality(value)
    return LevelEvidenceQuality(int(value))


def _tile_stats_is_improvement(candidate: TileLevelStats, previous: TileLevelStats | None) -> bool:
    if previous is None:
        return True
    candidate_quality = _coerce_evidence_quality(candidate.evidence_quality)
    previous_quality = _coerce_evidence_quality(previous.evidence_quality)
    if candidate_quality > previous_quality:
        return True
    if candidate_quality < previous_quality:
        return False
    if candidate.refined and not previous.refined:
        return True
    candidate_bounds = normalize_bounds(candidate.bounds)
    previous_bounds = normalize_bounds(previous.bounds)
    if candidate_bounds is None:
        return False
    if previous_bounds is None:
        return True
    contains_previous = bool(
        candidate_bounds[0] <= previous_bounds[0] and candidate_bounds[1] >= previous_bounds[1]
    )
    return bool(
        contains_previous
        and (candidate_bounds != previous_bounds or candidate.sample_count > previous.sample_count)
    )
