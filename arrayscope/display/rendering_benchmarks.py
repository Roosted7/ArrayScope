"""Small rendering-backend benchmarks for display hot paths."""

from __future__ import annotations

from dataclasses import dataclass
import os
from time import perf_counter

import numpy as np

from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.montage import MontageTileState
from arrayscope.window.display_frame import DisplayTilePayload


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

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_large_complex_tiled_initial(view))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_one_dirty_tile_commit(view))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_pan_zoom_no_upload(view))
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
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(tile_shape=(96, 96), count=2, columns=2)
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=(),
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    start = perf_counter()
    view._apply_histogram_preview_levels((0.4, 1.0))
    elapsed = _elapsed_ms(start)
    return _result(view, "complex_tile_level_preview", elapsed)


def _benchmark_clean_tile_flush(view) -> RenderingBenchmarkResult:
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(tile_shape=(96, 96), count=2, columns=2)
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    start = perf_counter()
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=(),
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    elapsed = _elapsed_ms(start)
    return _result(view, "clean_tile_flush", elapsed)


def _benchmark_large_complex_tiled_initial(view) -> RenderingBenchmarkResult:
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=128, columns=16)
    start = perf_counter()
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    elapsed = _elapsed_ms(start)
    return _result(view, "large_complex_tiled_initial", elapsed)


def _benchmark_one_dirty_tile_commit(view) -> RenderingBenchmarkResult:
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=64, columns=8)
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    dirty_payloads = dict(payloads)
    image = np.array(payloads[3].image, copy=True)
    image[..., 0] = 64
    dirty_payloads[3] = DisplayTilePayload(3, payloads[3].source_index, image, payloads[3].histogram_data, ("montage_tile", 3, "dirty"))
    dirty_sources = dict(sources)
    dirty_sources[3] = dirty_payloads[3].source_id
    start = perf_counter()
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=(3,),
        montage_tile_source_ids=dirty_sources,
        montage_tile_payloads=dirty_payloads,
    )
    elapsed = _elapsed_ms(start)
    return _result(view, "one_dirty_tile_commit", elapsed)


def _benchmark_pan_zoom_no_upload(view) -> RenderingBenchmarkResult:
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=64, columns=8)
    view.resize(900, 700)
    view.show()
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=None,
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    view.setMontageTileLayerPresentation(
        placeholder,
        histogramData=None,
        histogramPlotData=None,
        geometry=geometry,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        rgb_already_windowed=False,
        montage_dirty_tiles=(),
        montage_tile_source_ids=sources,
        montage_tile_payloads=payloads,
    )
    start = perf_counter()
    view.getView().setRange(xRange=(0.0, 256.0), yRange=(0.0, 256.0), padding=0)
    view.getView().setRange(xRange=(64.0, 320.0), yRange=(64.0, 320.0), padding=0)
    elapsed = _elapsed_ms(start)
    return _result(view, "pan_zoom_no_upload", elapsed)


def _direct_tile_layer_inputs(*, tile_shape=(32, 32), count=2, columns=2):
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    gap = 1
    count = int(count)
    columns = int(columns)
    rows = int(np.ceil(count / columns))
    height = rows * tile_h + max(0, rows - 1) * gap
    width = columns * tile_w + max(0, columns - 1) * gap
    y = np.linspace(0.0, 1.0, tile_h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, tile_w, dtype=np.float32)[None, :]
    geometry = DisplayGeometry(
        view_state=ViewState.from_shape((tile_h, tile_w, count)).with_montage_axis(2, columns=columns, indices=tuple(range(count)), text=":"),
        display_shape=(height, width),
        montage=MontageGeometry(indices=tuple(range(count)), tile_shape=(tile_h, tile_w), columns=columns, rows=rows, gap=gap),
        montage_tile_states=tuple(MontageTileState.LOADED for _ in range(count)),
    )
    payloads = {}
    sources = {}
    for index in range(count):
        histogram = np.clip(x + y + index / max(1, count), 0.0, 1.0).astype(np.float32)
        rgb = np.empty((tile_h, tile_w, 3), dtype=np.uint8)
        rgb[..., 0] = np.uint8(220)
        rgb[..., 1] = np.clip(255.0 * x, 0, 255).astype(np.uint8)
        rgb[..., 2] = np.clip(255.0 * y, 0, 255).astype(np.uint8)
        source_id = ("montage_tile", index)
        payloads[index] = DisplayTilePayload(index, index, rgb, histogram, source_id)
        sources[index] = source_id
    placeholder = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (height, width, 3))
    return placeholder, None, geometry, sources, payloads


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


def assert_optional_perf_gates(results: tuple[RenderingBenchmarkResult, ...]) -> None:
    if os.environ.get("ARRAYSCOPE_PERF_ASSERT") != "1":
        return
    by_name = {result.name: result for result in results}
    assert by_name["vispy_clean_tile_flush"].elapsed_ms <= max(2.0, by_name["pyqtgraph_clean_tile_flush"].elapsed_ms * 0.5)
    assert by_name["vispy_pan_zoom_no_upload"].elapsed_ms <= 8.0
    assert by_name["vispy_one_dirty_tile_commit"].elapsed_ms <= 4.0


def run_optional_stress_benchmark() -> tuple[RenderingBenchmarkResult, ...]:
    if os.environ.get("ARRAYSCOPE_RUN_STRESS") != "1":
        return ()
    from pathlib import Path

    import nibabel as nib

    source = Path("data/_WIPDelRec-tT2_20260223150234_14.nii")
    if not source.exists():
        raise FileNotFoundError(
            "stress fixture not found: data/_WIPDelRec-tT2_20260223150234_14.nii"
        )
    data = np.asarray(nib.load(str(source)).get_fdata(dtype=np.float32))
    if data.ndim < 3:
        raise ValueError(f"stress fixture must be at least 3D, got {data.shape}")
    slab = data
    while slab.ndim > 3:
        slab = slab[..., 0]
    complex_data = np.fft.fftshift(np.fft.fftn(slab, axes=(0, 1)), axes=(0, 1)).astype(np.complex64)
    del complex_data
    return benchmark_rendering_backends()


def main() -> None:
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg

    pg.mkQApp()
    results = run_optional_stress_benchmark() or benchmark_rendering_backends()
    for result in results:
        timing = result.timing
        print(
            f"{result.name}: elapsed={result.elapsed_ms:.3f} ms "
            f"total={float(timing.total_ms or 0.0):.3f} ms "
            f"tile_upload={float(timing.tile_layer_upload_ms or 0.0):.3f} ms "
            f"bytes={timing.visible_bytes} "
            f"items={timing.tile_layer_visible_items}/"
            f"{timing.tile_layer_items_updated}/"
            f"{timing.tile_layer_items_skipped}"
        )
    assert_optional_perf_gates(results)


if __name__ == "__main__":
    main()
