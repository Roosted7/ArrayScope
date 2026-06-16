"""Small foundation benchmarks for operation-cost sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.roi import RoiGeometry, roi_statistics, roi_values
from arrayscope.core.view_state import ViewState
from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.montage import RenderedTile, make_montage_plan, make_montage_viewport_canvas
from arrayscope.operations.chunked_stage import plan_chunked_stage_materialization
from arrayscope.operations.coordinator import OperationCoordinator
from arrayscope.operations.cost import estimate_pipeline_cost
from arrayscope.operations.dim_ops import centered_fft
from arrayscope.operations.pipeline import CenteredFFT, Mean
from arrayscope.operations.slabs import plan_slab, request_for_image


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    shape: tuple[int, ...]
    dtype: str
    elapsed_ms: float
    peak_estimate_bytes: int | None
    output_shape: tuple[int, ...]
    output_dtype: str
    chunk_count: int = 1


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


def benchmark_large_rgb_montage_histogram_drag(shape=(768, 1024), dtype=np.float32) -> BenchmarkResult:
    y = np.linspace(0, 1, int(shape[0]), dtype=np.float32)[:, None]
    x = np.linspace(0, 1, int(shape[1]), dtype=np.float32)[None, :]
    histogram = (x + y).astype(np.float32)
    base = np.empty((int(shape[0]), int(shape[1]), 3), dtype=np.float32)
    base[..., 0] = 255.0 * x
    base[..., 1] = 255.0 * y
    base[..., 2] = 255.0 * (1.0 - x)
    start = perf_counter()
    output = rgb_display_for_levels(base, histogram, (0.25, 1.25))
    elapsed = _elapsed_ms(start)
    return _result("large_rgb_montage_histogram_drag", shape, dtype, elapsed, output, int(base.nbytes + histogram.nbytes))


def benchmark_tile_layer_clean_commit(shape=(8, 128, 128), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    state = _BenchmarkState(shape)
    plan = make_montage_plan(state, axis=0, indices=range(min(8, shape[0])), tile_shape=shape[1:], columns=4)
    sources = {int(tile.montage_index): ("montage_tile", int(tile.source_index)) for tile in plan.tiles}
    start = perf_counter()
    unchanged = sum(1 for tile in plan.tiles if sources.get(int(tile.montage_index)) == ("montage_tile", int(tile.source_index)))
    elapsed = _elapsed_ms(start)
    return _result("tile_layer_clean_commit", shape, dtype, elapsed, np.asarray([unchanged], dtype=np.int64), data.nbytes)


def benchmark_roi_pan_zoom_stability(shape=(512, 512), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    geometry = RoiGeometry(kind="rectangle", rect=(64.0, 64.0, 128.0, 96.0))
    starts = [(0, 0), (16, 8), (32, 16), (48, 24)]
    start = perf_counter()
    values = [roi_statistics(roi_values(data[y : y + 256, x : x + 256], geometry)).count for x, y in starts]
    elapsed = _elapsed_ms(start)
    return _result("roi_pan_zoom_stability", shape, dtype, elapsed, np.asarray(values, dtype=np.int64), None)


def benchmark_offscreen_roi_demand_compute(shape=(64, 128, 128), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    geometry = RoiGeometry(kind="rectangle", rect=(16.0, 16.0, 32.0, 32.0))
    start = perf_counter()
    slab = data[shape[0] // 2]
    stats = roi_statistics(roi_values(slab, geometry))
    elapsed = _elapsed_ms(start)
    return _result("offscreen_roi_demand_compute", shape, dtype, elapsed, np.asarray([stats.mean], dtype=np.float64), None)


def benchmark_fft_stage_warmup_chunked(shape=(24, 24, 17), dtype=np.float32) -> BenchmarkResult:
    coordinator = OperationCoordinator(np.zeros(shape, dtype=dtype), operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(coordinator.document.current_shape)
    plan = plan_slab(coordinator.document, request_for_image(state))
    candidate = tuple(candidate for candidate in plan.region_plan.cache_candidates if candidate.retain)[-1]
    start = perf_counter()
    chunk_plan = plan_chunked_stage_materialization(plan.region_plan, candidate, object(), target_chunk_bytes=4096)
    elapsed = _elapsed_ms(start)
    chunk_count = 1 if chunk_plan is None else len(chunk_plan.chunks)
    return _result(
        "fft_stage_warmup_chunked",
        shape,
        dtype,
        elapsed,
        np.asarray([chunk_count], dtype=np.int64),
        candidate.estimated_nbytes,
        chunk_count=chunk_count,
    )


def benchmark_fft_stage_warmup_unchunked(shape=(24, 24, 17), dtype=np.float32) -> BenchmarkResult:
    coordinator = OperationCoordinator(np.zeros(shape, dtype=dtype), operations=(CenteredFFT(axis=2),))
    state = ViewState.from_shape(coordinator.document.current_shape)
    plan = plan_slab(coordinator.document, request_for_image(state))
    candidate = tuple(candidate for candidate in plan.region_plan.cache_candidates if candidate.retain)[-1]
    start = perf_counter()
    chunk_plan = plan_chunked_stage_materialization(plan.region_plan, candidate, object(), target_chunk_bytes=1024 * 1024 * 1024)
    elapsed = _elapsed_ms(start)
    return _result(
        "fft_stage_warmup_unchunked",
        shape,
        dtype,
        elapsed,
        np.asarray([0 if chunk_plan is None else len(chunk_plan.chunks)], dtype=np.int64),
        candidate.estimated_nbytes,
    )


def benchmark_live_profile_offscreen_unloaded_tile(shape=(16, 128, 128), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    start = perf_counter()
    profile = data[:, shape[1] // 2, shape[2] // 2]
    elapsed = _elapsed_ms(start)
    return _result("live_profile_offscreen_unloaded_tile", shape, dtype, elapsed, profile, None)


def run_foundation_benchmarks() -> tuple[BenchmarkResult, ...]:
    return (
        benchmark_raw_slice(),
        benchmark_fft_slice(workers=1),
        benchmark_montage_canvas(),
        benchmark_roi_stats(),
        benchmark_large_rgb_montage_histogram_drag(),
        benchmark_tile_layer_clean_commit(),
        benchmark_roi_pan_zoom_stability(),
        benchmark_offscreen_roi_demand_compute(),
        benchmark_fft_stage_warmup_chunked(),
        benchmark_fft_stage_warmup_unchunked(),
        benchmark_live_profile_offscreen_unloaded_tile(),
    )


def _result(name, shape, dtype, elapsed_ms, output, peak_estimate_bytes, *, chunk_count: int = 1) -> BenchmarkResult:
    output = np.asarray(output)
    return BenchmarkResult(
        name=name,
        shape=tuple(int(size) for size in shape),
        dtype=str(np.dtype(dtype)),
        elapsed_ms=float(elapsed_ms),
        peak_estimate_bytes=peak_estimate_bytes,
        output_shape=tuple(int(size) for size in output.shape),
        output_dtype=str(output.dtype),
        chunk_count=int(chunk_count),
    )


def _elapsed_ms(start) -> float:
    return (perf_counter() - start) * 1000.0


class _BenchmarkState:
    image_axes = (1, 2)

    def __init__(self, source_shape):
        self.shape = tuple(source_shape)

    def with_slice(self, axis, index):
        del axis, index
        return self

    def with_montage_axis(self, axis):
        del axis
        return self
