"""Pure adaptive histogram plot helpers.

This module deliberately has no Qt or pyqtgraph imports.  Widgets provide cheap
view facts; this module performs array sampling and binning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter

import numpy as np


MIN_HISTOGRAM_BIN_SCREEN_PX = 5
DEFAULT_HISTOGRAM_BIN_CAP = 500
DEFAULT_HISTOGRAM_TARGET_IMAGE_SIZE = 200


@dataclass(frozen=True)
class HistogramPlotRequest:
    data: np.ndarray
    source_identity: object
    histogram_bounds: tuple[float, float] | None
    visible_value_span: float | None
    pixel_extent: float
    bin_cap: int = DEFAULT_HISTOGRAM_BIN_CAP
    min_bin_screen_px: int = MIN_HISTOGRAM_BIN_SCREEN_PX
    generation: int = 0
    view_signature: object | None = None
    target_image_size: int = DEFAULT_HISTOGRAM_TARGET_IMAGE_SIZE


@dataclass(frozen=True)
class HistogramPlotResult:
    generation: int
    source_identity: object
    view_signature: object | None
    x: np.ndarray | None
    y: np.ndarray | None
    sampled_count: int = 0
    elapsed_ms: float = 0.0
    cancelled: bool = False

    @property
    def has_data(self) -> bool:
        return self.x is not None and self.y is not None and len(self.x) > 0


def compute_histogram_plot(request: HistogramPlotRequest) -> HistogramPlotResult:
    start = perf_counter()
    x = None
    y = None
    sampled_count = 0
    try:
        data = np.asarray(request.data)
        if data.size:
            sampled = sample_histogram_data(data, target_image_size=request.target_image_size)
            sampled = sampled[np.isfinite(sampled)]
            sampled_count = int(sampled.size)
            if sampled.size:
                bounds = finite_increasing_pair(request.histogram_bounds)
                if bounds is None:
                    low = float(np.nanmin(sampled))
                    high = float(np.nanmax(sampled))
                    if np.isfinite(low) and np.isfinite(high):
                        if high == low:
                            high = low + 1.0
                        bounds = (low, high)
                if bounds is not None:
                    low, high = bounds
                    span = high - low
                    if span > 0.0:
                        visible_span = request.visible_value_span
                        if visible_span is None or visible_span <= 0.0:
                            visible_span = span
                        visible_span = max(min(float(visible_span), span), np.finfo(float).eps)
                        pixel_extent = max(1.0, float(request.pixel_extent))
                        max_visible_bins = max(
                            2,
                            int(pixel_extent / max(1, int(request.min_bin_screen_px))),
                        )
                        visible_bin_width = visible_span / max_visible_bins
                        requested_bins = max(
                            2,
                            int(ceil(span / max(visible_bin_width, np.finfo(float).eps))),
                        )
                        cap = max(2, min(int(request.bin_cap), int(sampled.size)))
                        bins = max(2, min(requested_bins, cap))
                        counts, edges = np.histogram(sampled, bins=bins, range=(low, high))
                        x = edges[:-1]
                        y = counts
    except Exception:
        x = None
        y = None
        sampled_count = 0
    return HistogramPlotResult(
        generation=int(request.generation),
        source_identity=request.source_identity,
        view_signature=request.view_signature,
        x=x,
        y=y,
        sampled_count=sampled_count,
        elapsed_ms=(perf_counter() - start) * 1000.0,
    )


def sample_histogram_data(data: np.ndarray, *, target_image_size: int = DEFAULT_HISTOGRAM_TARGET_IMAGE_SIZE) -> np.ndarray:
    array = np.asarray(data)
    if np.iscomplexobj(array):
        array = np.abs(array).astype(np.float32, copy=False)
    if array.ndim < 2:
        return array.reshape(-1)
    step0 = max(1, int(ceil(array.shape[0] / float(target_image_size))))
    step1 = max(1, int(ceil(array.shape[1] / float(target_image_size))))
    return array[::step0, ::step1].reshape(-1)


def finite_increasing_pair(values) -> tuple[float, float] | None:
    if values is None:
        return None
    try:
        low, high = values
        low = float(low)
        high = float(high)
    except Exception:
        return None
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return (low, high)
