"""Small foundation benchmarks for operation-cost sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from arrayscope.core.roi import RoiGeometry, roi_statistics, roi_values
from arrayscope.core.view_state import ViewState
from arrayscope.display.image_upload import rgb_display_for_levels
from arrayscope.display.model.frame import DisplayTilePayload
from arrayscope.display.montage import make_montage_plan
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
    retargeted_count: int = 0
    pop_count: int = 0
    fairness_count: int = 0


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


def benchmark_montage_tile_payloads(shape=(32, 128, 128), dtype=np.float32) -> BenchmarkResult:
    data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    state = ViewState.from_shape(shape).with_image_axes(1, 2)
    plan = make_montage_plan(state, axis=0, indices=range(min(8, shape[0])), tile_shape=shape[1:], columns=4)
    start = perf_counter()
    payloads = {
        int(tile.montage_index): DisplayTilePayload(
            int(tile.montage_index),
            int(tile.source_index),
            data[tile.source_index],
            data[tile.source_index].astype(np.float32),
            ("benchmark", int(tile.source_index), str(np.dtype(dtype))),
        )
        for tile in plan.tiles
    }
    elapsed = _elapsed_ms(start)
    payload_bytes = sum(
        int(payload.image.nbytes) + (0 if payload.histogram_data is None else int(payload.histogram_data.nbytes))
        for payload in payloads.values()
    )
    output = np.asarray([len(payloads)], dtype=np.int64)
    return _result("montage_tile_payloads", shape, dtype, elapsed, output, payload_bytes)


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
    state = ViewState.from_shape(shape).with_image_axes(1, 2)
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


def benchmark_montage_priority_queue_retarget(shape=(8, 8, 1024), dtype=np.float32) -> BenchmarkResult:
    from arrayscope.display.model.tile_priority import MontageTilePriorityQueue, TilePriorityContext

    state = ViewState.from_shape(shape).with_montage_axis(2, indices=tuple(range(shape[2])), columns=32, text=":")
    plan = make_montage_plan(state, axis=2, indices=tuple(range(shape[2])), tile_shape=shape[:2], columns=32)
    full_range = ((0.0, float(plan.geometry.columns * (shape[1] + plan.geometry.gap))), (0.0, float(plan.geometry.rows * (shape[0] + plan.geometry.gap))))
    context = TilePriorityContext.from_tiles(
        view_range=full_range,
        focus=(0.0, 0.0),
        visible_tiles=range(shape[2]),
        near_tiles=(),
    )
    queue = MontageTilePriorityQueue(plan.tiles, context=context, aging_after=8)
    retargeted = 0
    popped = 0
    start = perf_counter()
    for step in range(16):
        focus_x = float((step % 32) * (shape[1] + plan.geometry.gap) + shape[1] * 0.5)
        focus_y = float((step // 2 % 32) * (shape[0] + plan.geometry.gap) + shape[0] * 0.5)
        focus_index = int(step // 2 % 32) * int(plan.geometry.columns) + int(step % 32)
        retargeted += queue.set_context(
            TilePriorityContext.from_tiles(
                view_range=full_range,
                focus=(focus_x, focus_y),
                visible_tiles=range(shape[2]),
                near_tiles=(),
                priority_tiles=(focus_index,),
            ),
            max_items=64,
        )
        for _ in range(8):
            if queue.pop() is None:
                break
            popped += 1
    elapsed = _elapsed_ms(start)
    output = np.asarray([retargeted, popped, queue.fairness_pops], dtype=np.int64)
    return _result(
        "montage_priority_queue_retarget",
        shape,
        dtype,
        elapsed,
        output,
        None,
        retargeted_count=retargeted,
        pop_count=popped,
        fairness_count=queue.fairness_pops,
    )


def run_foundation_benchmarks() -> tuple[BenchmarkResult, ...]:
    return (
        benchmark_raw_slice(),
        benchmark_fft_slice(workers=1),
        benchmark_montage_tile_payloads(),
        benchmark_roi_stats(),
        benchmark_large_rgb_montage_histogram_drag(),
        benchmark_tile_layer_clean_commit(),
        benchmark_roi_pan_zoom_stability(),
        benchmark_offscreen_roi_demand_compute(),
        benchmark_fft_stage_warmup_chunked(),
        benchmark_fft_stage_warmup_unchunked(),
        benchmark_live_profile_offscreen_unloaded_tile(),
        benchmark_montage_priority_queue_retarget(),
    )


def _result(
    name,
    shape,
    dtype,
    elapsed_ms,
    output,
    peak_estimate_bytes,
    *,
    chunk_count: int = 1,
    retargeted_count: int = 0,
    pop_count: int = 0,
    fairness_count: int = 0,
) -> BenchmarkResult:
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
        retargeted_count=int(retargeted_count),
        pop_count=int(pop_count),
        fairness_count=int(fairness_count),
    )


def _elapsed_ms(start) -> float:
    return (perf_counter() - start) * 1000.0
