"""Small rendering-backend benchmarks for display hot paths."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.montage import MontageTileState


@dataclass(frozen=True)
class RenderingBenchmarkResult:
    name: str
    backend: str
    scenario: str
    elapsed_ms: float
    timing: ImageUploadTiming


def benchmark_rendering_backends() -> tuple[RenderingBenchmarkResult, ...]:
    """Compare PyQtGraph and VisPy display-update hot paths.

    These benchmarks intentionally expose deterministic work counters alongside
    elapsed time.  Tests should gate on the counters because Qt/OpenGL timing is
    highly platform dependent, especially under the offscreen platform.
    """

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    results = []
    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_scalar_level_preview(view))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_complex_tile_level_preview(view))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_clean_tile_flush(view))
        finally:
            view.close()

    return tuple(results)


def _benchmark_scalar_level_preview(view) -> RenderingBenchmarkResult:
    data = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    start = perf_counter()
    view._apply_histogram_preview_levels((0.25, 0.85))
    elapsed = _elapsed_ms(start)
    return _result(view, "scalar_level_preview", elapsed)


def _benchmark_complex_tile_level_preview(view) -> RenderingBenchmarkResult:
    rgb, histogram, geometry, sources = _tile_layer_inputs(tile_shape=(96, 96), columns=2)
    view.setMontageTileLayerPresentation(
        rgb,
        histogramData=histogram,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
    )
    start = perf_counter()
    view._apply_histogram_preview_levels((0.4, 1.0))
    elapsed = _elapsed_ms(start)
    return _result(view, "complex_tile_level_preview", elapsed)


def _benchmark_clean_tile_flush(view) -> RenderingBenchmarkResult:
    rgb, histogram, geometry, sources = _tile_layer_inputs(tile_shape=(96, 96), columns=2)
    view.setMontageTileLayerPresentation(
        rgb,
        histogramData=histogram,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
    )
    start = perf_counter()
    view.setMontageTileLayerPresentation(
        rgb,
        histogramData=histogram,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=(),
        montage_tile_source_ids=sources,
    )
    elapsed = _elapsed_ms(start)
    return _result(view, "clean_tile_flush", elapsed)


def _tile_layer_inputs(*, tile_shape=(32, 32), columns=2):
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    gap = 1
    height = tile_h
    width = columns * tile_w + (columns - 1) * gap
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    histogram = np.clip(x + y, 0.0, 1.0).astype(np.float32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.uint8(220)
    rgb[..., 1] = np.clip(255.0 * x, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255.0 * y, 0, 255).astype(np.uint8)
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((tile_h, tile_w, columns)).with_montage_axis(2, columns=columns, indices=tuple(range(columns)), text=":"),
        display_shape=(height, width),
        montage=MontageGeometry(indices=tuple(range(columns)), tile_shape=(tile_h, tile_w), columns=columns, rows=1, gap=gap),
        montage_tile_states=tuple(MontageTileState.LOADED for _ in range(columns)),
    )
    sources = {index: ("montage_tile", index) for index in range(columns)}
    return rgb, histogram, geometry, sources


def _result(view, scenario: str, elapsed_ms: float) -> RenderingBenchmarkResult:
    backend = str(getattr(view, "rendering_backend_name", type(view).__name__))
    return RenderingBenchmarkResult(
        name=f"{backend}_{scenario}",
        backend=backend,
        scenario=scenario,
        elapsed_ms=float(elapsed_ms),
        timing=view.lastImageUploadTiming(),
    )


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0
