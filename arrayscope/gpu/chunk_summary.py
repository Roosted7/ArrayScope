"""Bounded per-chunk summaries and the ADR 0056 coverage frontier."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

import numpy as np

from arrayscope.gpu.keys import DataChunkKey


DEFAULT_HISTOGRAM_BINS = 64
DEFAULT_REPRESENTATIVE_SAMPLE_LIMIT = 512
HISTOGRAM_NORMALIZED_L1_TOLERANCE = 0.05


def chunk_summary_storage_nbytes(bins: int = DEFAULT_HISTOGRAM_BINS) -> int:
    """Accounted array storage for one summary (counts + bin edges)."""

    bins = max(1, int(bins))
    return int(
        bins * np.dtype(np.float64).itemsize
        + (bins + 1) * np.dtype(np.float32).itemsize
    )


@dataclass(frozen=True)
class ChunkHistogramSummary:
    """Small immutable distribution summary for one canonical data chunk."""

    key: DataChunkKey
    bounds: tuple[float, float] | None
    counts: np.ndarray = field(repr=False, compare=False)
    bin_edges: np.ndarray = field(repr=False, compare=False)
    stored_finite_count: int
    source_weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, DataChunkKey):
            raise TypeError("chunk summaries require a DataChunkKey")
        counts = np.ascontiguousarray(self.counts, dtype=np.float64)
        edges = np.ascontiguousarray(self.bin_edges, dtype=np.float32)
        if counts.ndim != 1 or edges.shape != (counts.size + 1,):
            raise ValueError("chunk histogram counts and edges disagree")
        if np.any(counts < 0.0) or not np.all(np.isfinite(counts)):
            raise ValueError("chunk histogram counts must be finite and non-negative")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("chunk histogram edges must be strictly increasing")
        counts.setflags(write=False)
        edges.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(
            self, "stored_finite_count", max(0, int(self.stored_finite_count))
        )
        object.__setattr__(self, "source_weight", max(0.0, float(self.source_weight)))
        if self.bounds is not None:
            low, high = (float(self.bounds[0]), float(self.bounds[1]))
            if not np.isfinite(low) or not np.isfinite(high) or high < low:
                raise ValueError("chunk summary bounds must be finite and ordered")
            object.__setattr__(self, "bounds", (low, high))

    @property
    def nbytes(self) -> int:
        return int(self.counts.nbytes + self.bin_edges.nbytes)


@dataclass(frozen=True)
class ChunkHistogramAggregate:
    """One bounded aggregate over a non-overlapping summary frontier."""

    bounds: tuple[float, float] | None
    counts: np.ndarray = field(repr=False, compare=False)
    bin_edges: np.ndarray = field(repr=False, compare=False)
    representative_sample: np.ndarray = field(repr=False, compare=False)
    source_weight: float
    frontier_keys: tuple[DataChunkKey, ...]

    def __post_init__(self) -> None:
        counts = np.ascontiguousarray(self.counts, dtype=np.float64)
        edges = np.ascontiguousarray(self.bin_edges, dtype=np.float32)
        sample = np.ascontiguousarray(self.representative_sample, dtype=np.float32)
        counts.setflags(write=False)
        edges.setflags(write=False)
        sample.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "bin_edges", edges)
        object.__setattr__(self, "representative_sample", sample)
        object.__setattr__(self, "source_weight", max(0.0, float(self.source_weight)))
        object.__setattr__(self, "frontier_keys", tuple(self.frontier_keys))


def summarize_chunk(
    key: DataChunkKey,
    values,
    *,
    weights=None,
    bins: int = DEFAULT_HISTOGRAM_BINS,
) -> ChunkHistogramSummary:
    """Compute one bounded summary from already-materialized scalar values.

    Complex values use magnitude for the canonical stored summary. A
    display-specific complex mapping supplies mapped scalar values instead;
    the render worker uses the shared shader-mapping function at that seam.
    """

    if not isinstance(key, DataChunkKey):
        raise TypeError("chunk summaries require a DataChunkKey")
    bins = max(1, int(bins))
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array).astype(np.float32, copy=False)
    scalar = np.asarray(array).reshape(-1)
    sample_weights = _broadcast_weights(array, weights).reshape(-1)
    finite = np.isfinite(scalar) & np.isfinite(sample_weights) & (sample_weights >= 0.0)
    finite_values = np.asarray(scalar[finite], dtype=np.float32)
    finite_weights = np.asarray(sample_weights[finite], dtype=np.float64)
    if finite_values.size == 0:
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
        counts = np.zeros(bins, dtype=np.float64)
        bounds = None
    else:
        low = float(np.min(finite_values))
        high = float(np.max(finite_values))
        bounds = (low, high)
        edge_low, edge_high = _histogram_edge_bounds(low, high)
        edges = np.linspace(edge_low, edge_high, bins + 1, dtype=np.float32)
        counts, _unused = np.histogram(
            finite_values,
            bins=np.asarray(edges, dtype=np.float64),
            weights=finite_weights,
        )
        counts = np.asarray(counts, dtype=np.float64)
    return ChunkHistogramSummary(
        key=key,
        bounds=bounds,
        counts=counts,
        bin_edges=edges,
        stored_finite_count=int(finite_values.size),
        source_weight=float(np.sum(finite_weights, dtype=np.float64)),
    )


def chunk_summary_frontier(summaries) -> tuple[ChunkHistogramSummary, ...]:
    """Return a non-overlapping best-available coverage frontier.

    Finer children replace a parent only as a complete native-rectangle set.
    Partial child coverage contributes nothing beside that parent.
    """

    unique: dict[DataChunkKey, ChunkHistogramSummary] = {}
    for summary in tuple(summaries or ()):
        if not isinstance(summary, ChunkHistogramSummary):
            raise TypeError("coverage frontiers require ChunkHistogramSummary values")
        if summary.key.rank != 2:
            raise ValueError("G6 coverage frontiers currently require 2D chunk keys")
        unique[summary.key] = summary
    rows = tuple(unique.values())
    selected_keys = set(chunk_key_frontier(tuple(row.key for row in rows)))
    return tuple(
        sorted(
            (row for row in rows if row.key in selected_keys),
            key=lambda item: (item.key.chunk_origin, item.key.lod.reduction),
        )
    )


def chunk_key_frontier(keys) -> tuple[DataChunkKey, ...]:
    """Return the ADR 0056 non-overlapping frontier for chunk identities.

    Native pages are the root of one derived reducer family, so they may mix
    with that family's reduced pages.  Multiple non-native reducers remain
    incompatible semantic evidence and are rejected loudly.
    """

    unique = tuple(dict.fromkeys(tuple(keys or ())))
    for key in unique:
        if not isinstance(key, DataChunkKey):
            raise TypeError("coverage frontiers require DataChunkKey values")
        if key.rank != 2:
            raise ValueError("G6 coverage frontiers currently require 2D chunk keys")
    if not unique:
        return ()
    base_families = {_summary_base_family(key) for key in unique}
    reducers = {key.lod.reducer for key in unique if not key.lod.is_native}
    if len(base_families) != 1 or len(reducers) > 1:
        raise ValueError("one chunk-summary frontier cannot mix value families")
    ordered = sorted(unique, key=_coarse_first_key_for_key)
    selected: list[DataChunkKey] = []
    excluded: set[DataChunkKey] = set()
    for candidate in ordered:
        if candidate in excluded:
            continue
        finer = tuple(
            other
            for other in unique
            if other != candidate
            and other not in excluded
            and _strictly_finer(other, candidate)
            and _rect_contains(_key_rect(candidate), _key_rect(other))
        )
        if finer and _rect_fully_covered(
            _key_rect(candidate), tuple(_key_rect(item) for item in finer)
        ):
            continue
        selected.append(candidate)
        candidate_rect = _key_rect(candidate)
        excluded.update(
            other
            for other in unique
            if other != candidate
            and _strictly_finer(other, candidate)
            and _rects_overlap(candidate_rect, _key_rect(other))
        )
    return tuple(sorted(selected, key=lambda key: (key.chunk_origin, key.lod.reduction)))


def aggregate_chunk_summaries(
    summaries,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    sample_limit: int = DEFAULT_REPRESENTATIVE_SAMPLE_LIMIT,
) -> ChunkHistogramAggregate:
    """Merge immutable summaries without reading their materialized arrays."""

    bins = max(1, int(bins))
    sample_limit = max(1, int(sample_limit))
    frontier = chunk_summary_frontier(summaries)
    populated = tuple(
        item
        for item in frontier
        if item.bounds is not None and item.source_weight > 0.0
    )
    if not populated:
        edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float32)
        return ChunkHistogramAggregate(
            bounds=None,
            counts=np.zeros(bins, dtype=np.float64),
            bin_edges=edges,
            representative_sample=np.asarray((), dtype=np.float32),
            source_weight=0.0,
            frontier_keys=tuple(item.key for item in frontier),
        )
    low = min(float(item.bounds[0]) for item in populated)
    high = max(float(item.bounds[1]) for item in populated)
    edge_low, edge_high = _histogram_edge_bounds(low, high)
    edges = np.linspace(edge_low, edge_high, bins + 1, dtype=np.float32)
    counts = np.zeros(bins, dtype=np.float64)
    for summary in populated:
        counts += _rebin_counts(summary.counts, summary.bin_edges, edges)
    source_weight = float(np.sum(counts, dtype=np.float64))
    representative = _representative_sample(counts, edges, limit=sample_limit)
    return ChunkHistogramAggregate(
        bounds=(low, high),
        counts=counts,
        bin_edges=edges,
        representative_sample=representative,
        source_weight=source_weight,
        frontier_keys=tuple(item.key for item in frontier),
    )


def representative_sample_from_histogram(
    counts,
    bounds: tuple[float, float],
    *,
    sample_limit: int = DEFAULT_REPRESENTATIVE_SAMPLE_LIMIT,
) -> np.ndarray:
    """Turn GPU/readback bins into the bounded sample used by level tracking."""

    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if counts.size == 0:
        return np.asarray((), dtype=np.float32)
    low, high = (float(bounds[0]), float(bounds[1]))
    edge_low, edge_high = _histogram_edge_bounds(low, high)
    edges = np.linspace(edge_low, edge_high, counts.size + 1, dtype=np.float32)
    return _representative_sample(counts, edges, limit=max(1, int(sample_limit)))


def _broadcast_weights(array: np.ndarray, weights) -> np.ndarray:
    if weights is None:
        return np.ones(array.shape, dtype=np.float64)
    result = np.asarray(weights, dtype=np.float64)
    if result.shape == array.shape:
        return result
    if array.ndim >= 3 and result.shape == array.shape[:2]:
        result = result[(...,) + (None,) * (array.ndim - 2)]
    try:
        return np.broadcast_to(result, array.shape)
    except ValueError as exc:
        raise ValueError(
            f"summary weights {result.shape} do not broadcast to values {array.shape}"
        ) from exc


def _histogram_edge_bounds(low: float, high: float) -> tuple[float, float]:
    if high > low:
        return low, high
    radius = max(abs(low) * 0.03, 0.5)
    return low - radius, high + radius


def _summary_base_family(key: DataChunkKey) -> tuple[object, ...]:
    return (
        key.document_generation,
        key.operation_key,
        key.dtype,
        key.representation,
    )


def _coarse_first_key_for_key(key: DataChunkKey) -> tuple[object, ...]:
    return (
        -prod(key.chunk_shape),
        -sum(int(value) for value in key.lod.reduction),
        key.chunk_origin,
    )


def _strictly_finer(candidate: DataChunkKey, parent: DataChunkKey) -> bool:
    child = tuple(
        int(candidate.lod.reduction[axis])
        if axis < len(candidate.lod.reduction)
        else 0
        for axis in range(candidate.rank)
    )
    coarse = tuple(
        int(parent.lod.reduction[axis])
        if axis < len(parent.lod.reduction)
        else 0
        for axis in range(parent.rank)
    )
    return bool(
        len(child) == len(coarse)
        and all(fine <= broad for fine, broad in zip(child, coarse))
        and child != coarse
    )


def _key_rect(key: DataChunkKey) -> tuple[int, int, int, int]:
    y0, x0 = key.chunk_origin
    height, width = key.chunk_shape
    return int(y0), int(y0 + height), int(x0), int(x0 + width)


def _rect_contains(outer, inner) -> bool:
    return bool(
        outer[0] <= inner[0]
        and inner[1] <= outer[1]
        and outer[2] <= inner[2]
        and inner[3] <= outer[3]
    )


def _rects_overlap(left, right) -> bool:
    return bool(
        max(left[0], right[0]) < min(left[1], right[1])
        and max(left[2], right[2]) < min(left[3], right[3])
    )


def _rect_fully_covered(parent, children) -> bool:
    clipped = tuple(
        (
            max(parent[0], child[0]),
            min(parent[1], child[1]),
            max(parent[2], child[2]),
            min(parent[3], child[3]),
        )
        for child in children
        if _rects_overlap(parent, child)
    )
    if not clipped:
        return False
    y_edges = sorted(
        {parent[0], parent[1], *(value for rect in clipped for value in rect[:2])}
    )
    x_edges = sorted(
        {parent[2], parent[3], *(value for rect in clipped for value in rect[2:])}
    )
    for y0, y1 in zip(y_edges, y_edges[1:]):
        if y0 < parent[0] or y1 > parent[1] or y1 <= y0:
            continue
        for x0, x1 in zip(x_edges, x_edges[1:]):
            if x0 < parent[2] or x1 > parent[3] or x1 <= x0:
                continue
            if not any(
                child[0] <= y0 and y1 <= child[1] and child[2] <= x0 and x1 <= child[3]
                for child in clipped
            ):
                return False
    return True


def _representative_sample(
    counts: np.ndarray, edges: np.ndarray, *, limit: int
) -> np.ndarray:
    total = float(np.sum(counts, dtype=np.float64))
    if total <= 0.0:
        return np.asarray((), dtype=np.float32)
    size = min(max(1, int(np.ceil(total))), max(1, int(limit)))
    positions = (np.arange(size, dtype=np.float64) + 0.5) * (total / float(size))
    cumulative = np.cumsum(np.asarray(counts, dtype=np.float64))
    indices = np.searchsorted(cumulative, positions, side="left")
    indices = np.clip(indices, 0, counts.size - 1)
    centers = (
        np.asarray(edges[:-1], dtype=np.float64)
        + np.asarray(edges[1:], dtype=np.float64)
    ) * 0.5
    return np.asarray(centers[indices], dtype=np.float32)


def _rebin_counts(counts, source_edges, target_edges) -> np.ndarray:
    """Conservatively distribute local-bin mass across aggregate bins.

    Per-chunk bins have local ranges. Treating every bin as a midpoint creates
    visible spikes when ranges differ; overlap-weighted transfer is the
    bounded histogram analogue of linear resampling and preserves total mass.
    """

    source_edges = np.asarray(source_edges, dtype=np.float64)
    target_edges = np.asarray(target_edges, dtype=np.float64)
    source_left = source_edges[:-1, np.newaxis]
    source_right = source_edges[1:, np.newaxis]
    target_left = target_edges[np.newaxis, :-1]
    target_right = target_edges[np.newaxis, 1:]
    overlap = np.maximum(
        0.0,
        np.minimum(source_right, target_right) - np.maximum(source_left, target_left),
    )
    widths = np.maximum(source_right - source_left, np.finfo(np.float64).tiny)
    transferred = np.sum(
        np.asarray(counts, dtype=np.float64)[:, np.newaxis] * (overlap / widths),
        axis=0,
    )
    lost = float(
        np.sum(counts, dtype=np.float64) - np.sum(transferred, dtype=np.float64)
    )
    if abs(lost) > np.finfo(np.float64).eps * max(1.0, float(np.sum(counts))):
        # Float32 edge roundoff can leave an endpoint-sized sliver. Preserve
        # mass in the nearest edge bin instead of silently losing evidence.
        transferred[-1 if lost > 0.0 else 0] += lost
    return transferred


__all__ = [
    "ChunkHistogramAggregate",
    "ChunkHistogramSummary",
    "HISTOGRAM_NORMALIZED_L1_TOLERANCE",
    "aggregate_chunk_summaries",
    "chunk_summary_frontier",
    "chunk_key_frontier",
    "chunk_summary_storage_nbytes",
    "representative_sample_from_histogram",
    "summarize_chunk",
]
