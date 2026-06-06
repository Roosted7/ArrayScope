"""Pure histogram helpers for ROI comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistogramSpec:
    bins: int = 128
    range_mode: str = "shared_visible"
    density: bool = False


@dataclass(frozen=True)
class HistogramResult:
    name: str
    counts: np.ndarray
    edges: np.ndarray


def histogram(values, spec: HistogramSpec | None = None, value_range=None, *, name=""):
    spec = HistogramSpec() if spec is None else spec
    finite = _finite_values(values)
    if finite.size == 0:
        edges = _empty_edges(spec, value_range)
        return HistogramResult(str(name), np.zeros(len(edges) - 1, dtype=float), edges)
    counts, edges = np.histogram(finite, bins=max(1, int(spec.bins)), range=value_range, density=bool(spec.density))
    return HistogramResult(str(name), counts.astype(float), edges.astype(float))


def comparison_histograms(named_value_sets, spec: HistogramSpec | None = None):
    spec = HistogramSpec() if spec is None else spec
    prepared = [(str(name), _finite_values(values)) for name, values in named_value_sets]
    value_range = None
    if spec.range_mode == "shared_visible":
        non_empty = [values for _name, values in prepared if values.size]
        if non_empty:
            combined = np.concatenate(non_empty)
            low = float(np.min(combined))
            high = float(np.max(combined))
            if low == high:
                low -= 0.5
                high += 0.5
            value_range = (low, high)
    return tuple(histogram(values, spec, value_range=value_range, name=name) for name, values in prepared)


def _finite_values(values):
    values = np.asarray(values).ravel()
    return values[np.isfinite(values)].astype(float, copy=False)


def _empty_edges(spec, value_range):
    bins = max(1, int(spec.bins))
    if value_range is None:
        value_range = (0.0, 1.0)
    return np.linspace(float(value_range[0]), float(value_range[1]), bins + 1)
