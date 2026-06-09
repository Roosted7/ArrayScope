"""Small foundation benchmarks for operation-cost sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.roi import RoiGeometry, roi_statistics, roi_values
from arrayscope.display.montage import RenderedTile, make_montage_plan, make_montage_viewport_canvas
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.dim_ops import centered_fft
from arrayscope.operations.pipeline import CenteredFFT, Mean


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    shape: tuple[int, ...]
    dtype: str
    elapsed_ms: float
    peak_estimate_bytes: int | None
    output_shape: tuple[int, ...]
    output_dtype: str


def benchmark_raw_slice(shape=(64, 128, 128), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    start = perf_counter()
    output = data[shape[0] // 2, :, :]
    elapsed = _elapsed_ms(start)
    return _result("raw_slice", shape, dtype, elapsed, output, None)


def benchmark_fft_slice(shape=(32, 128, 128), dtype=np.float32, workers=1) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    cost = estimate_pipeline_cost(shape, np.dtype(dtype), (CenteredFFT(axis=0),))
    start = perf_counter()
    transformed = centered_fft(data, 0, workers=int(workers))
    output = transformed[shape[0] // 2, :, :]
    elapsed = _elapsed_ms(start)
    return _result("fft_slice", shape, dtype, elapsed, output, cost.estimated_peak_bytes)


def benchmark_montage_canvas(shape=(32, 128, 128), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

    class _State:
        image_axes = (1, 2)

        def __init__(self, source_shape):
            self.shape = tuple(source_shape)

        def with_slice(self, axis, index):
            del axis, index
            return self

        def with_montage_axis(self, axis):
            del axis
            return self

    plan = make_montage_plan(_State(shape), axis=0, indices=range(min(8, shape[0])), tile_shape=shape[1:], columns=4)
    tiles = tuple(
        RenderedTile(tile=tile, image=data[tile.source_index], histogram_data=data[tile.source_index].astype(np.float32), eval_ms=0.0, slab_shape=shape[1:], slab_nbytes=data[tile.source_index].nbytes)
        for tile in plan.tiles
    )
    start = perf_counter()
    canvas = make_montage_viewport_canvas(
        plan,
        tiles,
        viewport_shape=(256, 256),
        budget_bytes=64 * 1024 * 1024,
        dtype=np.dtype(dtype),
        include_histogram=True,
    )
    elapsed = _elapsed_ms(start)
    return _result("montage_canvas", shape, dtype, elapsed, canvas.data, int(canvas.data.nbytes + (0 if canvas.histogram_data is None else canvas.histogram_data.nbytes)))


def benchmark_roi_stats(shape=(512, 512), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    geometry = RoiGeometry(kind="rectangle", rect=(shape[1] * 0.25, shape[0] * 0.25, shape[1] * 0.5, shape[0] * 0.5))
    cost = estimate_pipeline_cost(shape, np.dtype(dtype), (Mean(axis=0),))
    start = perf_counter()
    values = roi_values(data, geometry)
    stats = roi_statistics(values)
    elapsed = _elapsed_ms(start)
    output = np.asarray([stats.mean if stats.mean is not None else np.nan], dtype=float)
    return _result("roi_stats", shape, dtype, elapsed, output, cost.estimated_peak_bytes)


def run_foundation_benchmarks() -> tuple[BenchmarkResult, ...]:
    return (
        benchmark_raw_slice(),
        benchmark_fft_slice(workers=1),
        benchmark_montage_canvas(),
        benchmark_roi_stats(),
    )


def _result(name, shape, dtype, elapsed_ms, output, peak_estimate_bytes) -> BenchmarkResult:
    output = np.asarray(output)
    return BenchmarkResult(
        name=name,
        shape=tuple(int(size) for size in shape),
        dtype=str(np.dtype(dtype)),
        elapsed_ms=float(elapsed_ms),
        peak_estimate_bytes=peak_estimate_bytes,
        output_shape=tuple(int(size) for size in output.shape),
        output_dtype=str(output.dtype),
    )


def _elapsed_ms(start) -> float:
    return (perf_counter() - start) * 1000.0
