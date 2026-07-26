"""Small rendering-backend benchmarks for display hot paths."""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from arrayscope.core.frame_targets import FrameTarget
from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_REASON_NATIVE_SCALE
from arrayscope.display.model.frame import (
    DisplayTilePayload,
    TilePresentationDelta,
    TilePresentationState,
)
from arrayscope.display.montage import MontageTileState
from arrayscope.kernel import InlineWorkerBackend, Kernel, WorkItem
from arrayscope.kernel import Lane as WorkLane


@dataclass(frozen=True)
class RenderingBenchmarkResult:
    name: str
    backend: str
    scenario: str
    elapsed_ms: float
    timing: ImageUploadTiming
    first_frame_ms: float | None = None
    event_loop_drain_ms: float | None = None
    frame_count: int = 0
    ui_max_gap_ms: float | None = None
    commit_count: int = 1
    presentation_revision: int = 0
    presentation_stale_count: int = 0
    presentation_pending_count: int = 0
    presentation_settled: bool = True
    lod_desired_factor: int = 1
    lod_applied_factor: int = 1
    lod_desired_factor_xy: tuple[int, int] = (1, 1)
    lod_applied_factor_xy: tuple[int, int] = (1, 1)
    lod_source_texels_per_pixel_xy: tuple[float, float] = (0.0, 0.0)
    lod_policy: str = "native-only"
    lod_reason: str = "native-resolution texture is appropriate at the current scale"
    kernel_counters: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def submission_ms(self) -> float:
        """CPU time spent submitting the display update."""

        return float(self.elapsed_ms)


@dataclass(frozen=True)
class RenderingBenchmarkEnvironment:
    os: str
    platform: str
    python: str
    qt_api: str
    qt_version: str
    xdg_session_type: str
    wayland_display: str
    display: str
    qt_qpa_platform: str
    desktop_session: str
    xdg_current_desktop: str
    gpu_vendor: str = ""
    gpu_renderer: str = ""
    gpu_version: str = ""
    gpu_max_texture_size: int = 0
    gpu_max_texture_image_units: int = 0
    gpu_limits_source: str = ""
    gpu_limit_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _GpuLimits:
    max_texture_size: int = 0
    max_texture_image_units: int = 0
    vendor: str = ""
    renderer: str = ""
    version: str = ""
    source: str = ""
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderingBenchmarkSample:
    run: int
    timestamp: float
    environment: RenderingBenchmarkEnvironment
    result: RenderingBenchmarkResult


@dataclass(frozen=True)
class _ActionMeasurement:
    submission_ms: float
    first_frame_ms: float | None = None
    event_loop_drain_ms: float | None = None
    frame_count: int = 0
    ui_max_gap_ms: float | None = None


def benchmark_pyqtgraph_rendering(
    *, measure_presented: bool | None = None
) -> tuple[RenderingBenchmarkResult, ...]:
    """Measure the PyQtGraph display-update hot paths.

    ``elapsed_ms``/``submission_ms`` measures CPU-side setter submission only.
    Set ``measure_presented=True`` (or ``ARRAYSCOPE_BENCH_PRESENTED=1``) to also
    observe first-frame scheduling and Qt event-loop starvation. Deterministic
    tests should gate on work counters rather than wall-clock time. WGPU's
    renderer-specific scaling benchmarks live later in this module.
    """

    if measure_presented is None:
        measure_presented = os.environ.get("ARRAYSCOPE_BENCH_PRESENTED") == "1"
    results = []
    scenarios = (
        _benchmark_tiled_small_initial,
        _benchmark_tiled_large_initial,
        _benchmark_one_tile_montage_initial,
        _benchmark_multi_tile_montage_initial,
        _benchmark_scalar_level_preview,
        _benchmark_large_histogram_plot_refresh,
        _benchmark_complex_tile_level_preview,
        _benchmark_large_tile_level_preview,
        _benchmark_tile_level_uniform_update,
        _benchmark_clean_tile_flush,
        _benchmark_large_complex_tiled_initial,
        _benchmark_one_dirty_tile_commit,
        _benchmark_pan_zoom_no_upload,
        _benchmark_progressive_tile_stream,
    )
    try:
        results.extend(
            _run_view_benchmark(
                view_type,
                scenario,
                measure_presented=measure_presented,
            )
            for scenario in scenarios
            for view_type in (ImageView2D,)
        )
        return tuple(results)
    finally:
        _collect_benchmark_widgets()


def _run_view_benchmark(
    view_type, scenario, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    """Run one scenario and close its parentless Qt view."""

    view = view_type()
    try:
        return scenario(view, measure_presented=measure_presented)
    finally:
        view.close()
        view.deleteLater()
        view = None


def _collect_benchmark_widgets() -> None:
    """Release closed benchmark widgets and their parentless context menus.

    PyQtGraph signal cycles can retain a closed benchmark view until cyclic
    garbage collection. A complete benchmark run creates hundreds of menus and
    graphics helpers, so deferring cleanup to interpreter shutdown can make
    pytest appear to hang. Cleanup is outside every measured action.
    """

    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    if app is not None:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    gc.collect()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    if app is not None:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


def _benchmark_scalar_level_preview(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    data = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    _present_single_plane_benchmark_tiled(
        view, data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), histogramPlotData=data
    )
    measurement = _measure_action(
        view,
        lambda: view._apply_histogram_preview_levels((0.25, 0.85)),
        measure_presented=measure_presented,
    )
    return _result(view, "scalar_level_preview", measurement)


def _benchmark_tiled_small_initial(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    data = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float32).reshape(128, 128)
    measurement = _measure_action(
        view,
        lambda: _present_single_plane_benchmark_tiled(
            view, data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), histogramPlotData=data
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "tiled_small_initial", measurement)


def _benchmark_tiled_large_initial(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    state = ViewState.from_shape((1024, 1024)).with_image_axes(0, 1)
    planner = FramePlanner(internal_tile_shape=(256, 256))
    plan = planner.plan(
        target=FrameTarget(("normal-large", state.shape), None, None, "exact-visible"),
        view_state=state,
        display_shape=state.shape,
        backend_capabilities=image_view_backend_capabilities(view),
    )
    tile_state = _single_plane_tile_state(plan)
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=tile_state.payloads,
        active_tiles=plan.active_region_ids,
        planned_tiles=plan.planned_region_ids,
        near_tiles=plan.near_region_ids,
        force_refresh=True,
    )
    measurement = _measure_action(
        view,
        lambda: view.setTiledPresentation(
            geometry=plan.geometry,
            tile_state=tile_state,
            tile_delta=delta,
            histogramPlotData=None,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            rgb_already_windowed=False,
            tile_residency_budget_bytes=512 * 1024 * 1024,
            frame_plan=plan,
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "tiled_large_initial", measurement)


def _benchmark_one_tile_montage_initial(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(128, 128), count=1, columns=1
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "one_tile_montage_initial", measurement)


def _benchmark_multi_tile_montage_initial(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64), count=16, columns=4
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "multi_tile_montage_initial", measurement)


def _benchmark_large_histogram_plot_refresh(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    data = np.linspace(0.0, 1.0, 1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    _present_single_plane_benchmark_tiled(
        view, data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), histogramPlotData=data
    )
    measurement = _measure_action(
        view,
        lambda: view._refresh_histogram_plot(auto_level=False),
        measure_presented=measure_presented,
    )
    return _result(view, "large_histogram_plot_refresh", measurement)


def _benchmark_complex_tile_level_preview(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(96, 96), count=2, columns=2
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        dirty_tiles=(),
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.4, 1.0),
            histogramRange=(0.0, 1.0),
            dirty_tiles=(),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "complex_tile_level_preview", measurement)


def _benchmark_large_tile_level_preview(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64), count=96, columns=12
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.35, 0.95),
            histogramRange=(0.0, 1.0),
            dirty_tiles=(),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "large_tile_level_preview", measurement)


def _benchmark_tile_level_uniform_update(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64), count=24, columns=6
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.2, 0.9),
            histogramRange=(0.0, 1.0),
            dirty_tiles=(),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "tile_level_uniform_update", measurement)


def _benchmark_clean_tile_flush(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(96, 96), count=2, columns=2
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            dirty_tiles=(),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "clean_tile_flush", measurement)


def _benchmark_large_complex_tiled_initial(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64), count=128, columns=16
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "large_complex_tiled_initial", measurement)


def _benchmark_one_dirty_tile_commit(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64), count=64, columns=8
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )
    dirty_payloads = dict(payloads)
    image = np.array(payloads[3].image, copy=True)
    image[..., 0] = 64
    dirty_payloads[3] = DisplayTilePayload(
        3, payloads[3].source_index, image, payloads[3].histogram_data, ("montage_tile", 3, "dirty")
    )
    measurement = _measure_action(
        view,
        lambda: _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=dirty_payloads,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
            dirty_tiles=(3,),
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "one_dirty_tile_commit", measurement)


def _benchmark_pan_zoom_no_upload(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64), count=64, columns=8
    )
    view.resize(900, 700)
    view.show()
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
    )
    _present_benchmark_tiled(
        view,
        geometry=geometry,
        payloads=payloads,
        levels=(0.0, 1.0),
        histogramRange=(0.0, 1.0),
        dirty_tiles=(),
    )

    def pan_zoom():
        view.getView().setRange(xRange=(0.0, 256.0), yRange=(0.0, 256.0), padding=0)
        view.getView().setRange(xRange=(64.0, 320.0), yRange=(64.0, 320.0), padding=0)

    measurement = _measure_action(view, pan_zoom, measure_presented=measure_presented)
    return _result(view, "pan_zoom_no_upload", measurement)


def _benchmark_progressive_tile_stream(
    view, *, measure_presented: bool
) -> RenderingBenchmarkResult:
    """Measure UI-thread cost while representative tile batches arrive."""

    return _benchmark_progressive_tile_stream_configured(
        view,
        tile_shape=(64, 64),
        count=96,
        columns=12,
        batch_size=8,
        scenario="progressive_tile_stream",
        measure_presented=measure_presented,
    )


def _benchmark_progressive_tile_stream_configured(
    view,
    *,
    tile_shape: tuple[int, int],
    count: int,
    columns: int,
    batch_size: int,
    scenario: str,
    measure_presented: bool,
) -> RenderingBenchmarkResult:
    """Measure one complete progressive direct-tile presentation."""

    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=tile_shape,
        count=count,
        columns=columns,
    )
    batch_size = max(1, int(batch_size))
    timings: list[ImageUploadTiming] = []

    def stream_batches():
        for end in range(batch_size, len(payloads) + batch_size, batch_size):
            end = min(end, len(payloads))
            visible = {index: payloads[index] for index in range(end)}
            dirty_start = max(0, end - batch_size)
            upserts = (
                visible
                if end == batch_size
                else {index: payloads[index] for index in range(dirty_start, end)}
            )
            state = TilePresentationState(visible)
            delta = TilePresentationDelta(
                structure_revision=1,
                payload_revision=end,
                visibility_revision=end,
                level_revision=1,
                histogram_revision=1,
                viewport_revision=1,
                upserts=upserts,
                active_tiles=tuple(range(end)),
                planned_tiles=tuple(range(len(payloads))),
                near_tiles=tuple(range(min(len(payloads), end + batch_size))),
                force_refresh=end == batch_size,
            )
            view.setTiledPresentation(
                geometry=geometry,
                tile_state=state,
                tile_delta=delta,
                histogramPlotData=None,
                levels=(0.0, 1.0),
                histogramRange=(0.0, 1.0),
                rgb_already_windowed=False,
                tile_residency_budget_bytes=512 * 1024 * 1024,
            )
            timings.append(view.lastImageUploadTiming())
            if end >= len(payloads):
                break

    measurement = _measure_action(view, stream_batches, measure_presented=measure_presented)
    return _result(
        view,
        scenario,
        measurement,
        timing=_sum_upload_timings(timings),
        commit_count=len(timings),
    )


def _present_benchmark_tiled(
    view,
    *,
    geometry,
    payloads: dict[int, DisplayTilePayload],
    levels: tuple[float, float],
    histogramRange: tuple[float, float],
    dirty_tiles: tuple[int, ...] | None = None,
    histogramPlotData=None,
    rgb_already_windowed: bool = False,
    tile_residency_budget_bytes: int = 512 * 1024 * 1024,
):
    payloads = {int(tile): payload for tile, payload in dict(payloads).items()}
    if dirty_tiles is None:
        upserts = payloads
        revision = 1
        force_refresh = True
    elif dirty_tiles == ():
        upserts = {}
        revision = 2
        force_refresh = False
    else:
        dirty = tuple(int(tile) for tile in dirty_tiles)
        upserts = {tile: payloads[tile] for tile in dirty if tile in payloads}
        revision = 2
        force_refresh = False
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=revision,
        visibility_revision=revision,
        level_revision=revision,
        histogram_revision=1,
        viewport_revision=1,
        upserts=upserts,
        active_tiles=tuple(payloads),
        planned_tiles=tuple(payloads),
        near_tiles=tuple(payloads),
        near_tile_source_ids={tile: payload.source_id for tile, payload in payloads.items()},
        force_refresh=force_refresh,
    )
    return view.setTiledPresentation(
        geometry=geometry,
        tile_state=TilePresentationState(payloads),
        tile_delta=delta,
        histogramPlotData=histogramPlotData,
        levels=levels,
        histogramRange=histogramRange,
        rgb_already_windowed=rgb_already_windowed,
        tile_residency_budget_bytes=tile_residency_budget_bytes,
    )


def _present_single_plane_benchmark_tiled(
    view,
    data: np.ndarray,
    *,
    levels: tuple[float, float],
    histogramRange: tuple[float, float],
    histogramPlotData=None,
) -> object:
    data = np.asarray(data)
    tile_edge = (
        256
        if max(tuple(int(value) for value in data.shape[:2])) > 256
        else max(1, int(data.shape[0]), int(data.shape[1]))
    )
    state = ViewState.from_shape(data.shape[:2]).with_image_axes(0, 1)
    plan = FramePlanner(internal_tile_shape=(tile_edge, tile_edge)).plan(
        target=FrameTarget(("normal", data.shape), None, None, "exact-visible"),
        view_state=state,
        display_shape=data.shape[:2],
        backend_capabilities=image_view_backend_capabilities(view),
    )
    tile_state = _single_plane_tile_state_from_array(plan, data)
    return _present_benchmark_tiled(
        view,
        geometry=plan.geometry,
        payloads=tile_state.payloads,
        levels=levels,
        histogramRange=histogramRange,
        histogramPlotData=histogramPlotData,
    )


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
        view_state=ViewState.from_shape((tile_h, tile_w, count)).with_montage_axis(
            2, columns=columns, indices=tuple(range(count)), text=":"
        ),
        display_shape=(height, width),
        montage=MontageGeometry(
            indices=tuple(range(count)),
            tile_shape=(tile_h, tile_w),
            columns=columns,
            rows=rows,
            gap=gap,
        ),
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


def _single_plane_tile_state(frame_plan) -> TilePresentationState:
    payloads: dict[int, DisplayTilePayload] = {}
    height, width = tuple(int(value) for value in frame_plan.geometry.display_shape[:2])
    denom = max(1, height + width - 2)
    for region in frame_plan.regions:
        y_slice, x_slice = region.data_slices
        y0 = int(0 if y_slice.start is None else y_slice.start)
        y1 = int(height if y_slice.stop is None else y_slice.stop)
        x0 = int(0 if x_slice.start is None else x_slice.start)
        x1 = int(width if x_slice.stop is None else x_slice.stop)
        y = np.arange(y0, y1, dtype=np.float32)[:, None]
        x = np.arange(x0, x1, dtype=np.float32)[None, :]
        histogram = ((x + y) / float(denom)).astype(np.float32)
        image = np.empty((max(0, y1 - y0), max(0, x1 - x0), 3), dtype=np.uint8)
        image[..., 0] = np.clip(255.0 * histogram, 0, 255).astype(np.uint8)
        image[..., 1] = np.clip(255.0 * x / max(1, width - 1), 0, 255).astype(np.uint8)
        image[..., 2] = np.clip(255.0 * y / max(1, height - 1), 0, 255).astype(np.uint8)
        payloads[int(region.region_id)] = DisplayTilePayload(
            int(region.region_id),
            int(region.region_id),
            image,
            histogram,
            ("tiled_region", frame_plan.semantic_key, int(region.region_id)),
            semantic_data=histogram,
            semantic_histogram_data=histogram,
            source_shape=histogram.shape,
        )
    return TilePresentationState(payloads)


def _single_plane_tile_state_from_array(frame_plan, data: np.ndarray) -> TilePresentationState:
    payloads: dict[int, DisplayTilePayload] = {}
    source = np.asarray(data)
    for region in frame_plan.regions:
        y_slice, x_slice = region.data_slices
        tile = source[y_slice, x_slice, ...]
        payloads[int(region.region_id)] = DisplayTilePayload(
            int(region.region_id),
            int(region.region_id),
            tile,
            tile,
            (
                "tiled_region",
                frame_plan.semantic_key,
                int(region.region_id),
                tuple(tile.shape),
                str(tile.dtype),
            ),
            semantic_data=tile,
            semantic_histogram_data=tile,
            source_shape=tile.shape[:2],
        )
    return TilePresentationState(payloads)


def _result(
    view,
    scenario: str,
    measurement: _ActionMeasurement,
    *,
    timing: ImageUploadTiming | None = None,
    commit_count: int = 1,
) -> RenderingBenchmarkResult:
    backend = image_view_backend_capabilities(view).name
    result_timing = view.lastImageUploadTiming() if timing is None else timing
    pending_count = max(
        0, int(getattr(result_timing, "tile_layer_level_update_pending_items", 0) or 0)
    )
    applied_lod = max(1, int(getattr(result_timing, "tile_layer_lod_factor", 1) or 1))
    source_texels = max(
        0.0, float(getattr(result_timing, "tile_layer_source_texels_per_pixel", 0.0) or 0.0)
    )
    lod_policy = LOD_POLICY_NATIVE_ONLY if applied_lod == 1 else "backend-reported"
    lod_reason = (
        LOD_REASON_NATIVE_SCALE
        if applied_lod == 1
        else "backend timing reported a non-native applied LOD factor"
    )
    return RenderingBenchmarkResult(
        name=f"{backend}_{scenario}",
        backend=backend,
        scenario=scenario,
        elapsed_ms=float(measurement.submission_ms),
        timing=result_timing,
        first_frame_ms=measurement.first_frame_ms,
        event_loop_drain_ms=measurement.event_loop_drain_ms,
        frame_count=int(measurement.frame_count),
        ui_max_gap_ms=measurement.ui_max_gap_ms,
        commit_count=max(1, int(commit_count)),
        presentation_revision=0,
        presentation_stale_count=pending_count,
        presentation_pending_count=pending_count,
        presentation_settled=pending_count == 0,
        lod_desired_factor=applied_lod,
        lod_applied_factor=applied_lod,
        lod_desired_factor_xy=(applied_lod, applied_lod),
        lod_applied_factor_xy=(applied_lod, applied_lod),
        lod_source_texels_per_pixel_xy=(source_texels, source_texels),
        lod_policy=lod_policy,
        lod_reason=lod_reason,
        kernel_counters=_backend_commit_work_counters(
            backend=backend,
            scenario=scenario,
            commit_count=max(1, int(commit_count)),
        ),
    )


def _backend_commit_work_counters(
    *, backend: str, scenario: str, commit_count: int
) -> dict[str, dict[str, int]]:
    kernel = Kernel(InlineWorkerBackend())
    for index in range(max(1, int(commit_count))):
        target = FrameTarget(
            semantic_key=("benchmark", str(backend), str(scenario)),
            viewport_key=None,
            presentation_key=("backend-commit", int(index)),
            quality="exact-visible",
        )
        kernel.note_inline_work(
            WorkItem(
                key=("benchmark_backend_commit", str(backend), str(scenario), int(index)),
                lane=WorkLane.BACKEND_COMMIT,
                frame_target=target,
                supersession_key=(
                    "benchmark-backend-commit",
                    str(backend),
                    str(scenario),
                    int(index),
                ),
                supersession_value=int(index),
            )
        )
    return dict(kernel.diagnostics().lanes)


def _sum_upload_timings(timings) -> ImageUploadTiming:
    timings = tuple(timings)
    if not timings:
        return ImageUploadTiming(mode="progressive_tile_stream")

    def total(field):
        values = [getattr(timing, field) for timing in timings]
        finite = [float(value) for value in values if value is not None]
        return sum(finite) if finite else None

    last = timings[-1]
    return ImageUploadTiming(
        total_ms=total("total_ms"),
        visible_upload_ms=total("visible_upload_ms"),
        histogram_upload_ms=total("histogram_upload_ms"),
        histogram_bind_ms=total("histogram_bind_ms"),
        histogram_recompute_ms=total("histogram_recompute_ms"),
        level_sync_ms=total("level_sync_ms"),
        rgb_window_ms=total("rgb_window_ms"),
        tile_layer_upload_ms=total("tile_layer_upload_ms"),
        tile_layer_rgb_window_ms=total("tile_layer_rgb_window_ms"),
        profile_bounds_ms=total("profile_bounds_ms"),
        visible_bytes=sum(int(timing.visible_bytes) for timing in timings),
        visible_pixels=sum(int(timing.visible_pixels) for timing in timings),
        histogram_bytes=sum(int(timing.histogram_bytes) for timing in timings),
        histogram_pixels=sum(int(timing.histogram_pixels) for timing in timings),
        fast_same_object=all(bool(timing.fast_same_object) for timing in timings),
        mode="progressive_tile_stream",
        tile_layer_visible_items=int(last.tile_layer_visible_items),
        tile_layer_items_updated=sum(int(timing.tile_layer_items_updated) for timing in timings),
        tile_layer_items_skipped=sum(int(timing.tile_layer_items_skipped) for timing in timings),
        tile_layer_rgb_window_tiles=sum(
            int(timing.tile_layer_rgb_window_tiles) for timing in timings
        ),
        tile_layer_resident_items=int(last.tile_layer_resident_items),
        tile_layer_storage_capacity=int(last.tile_layer_storage_capacity),
        tile_layer_storage_rebuilds=sum(
            int(timing.tile_layer_storage_rebuilds) for timing in timings
        ),
        tile_layer_storage_evictions=sum(
            int(timing.tile_layer_storage_evictions) for timing in timings
        ),
        tile_layer_texture_uploads=sum(
            int(timing.tile_layer_texture_uploads) for timing in timings
        ),
        tile_layer_texture_upload_bytes=sum(
            int(timing.tile_layer_texture_upload_bytes) for timing in timings
        ),
        tile_layer_vertex_uploads=sum(int(timing.tile_layer_vertex_uploads) for timing in timings),
        tile_layer_level_updates=sum(int(timing.tile_layer_level_updates) for timing in timings),
        tile_layer_level_update_pending_items=sum(
            int(timing.tile_layer_level_update_pending_items) for timing in timings
        ),
        tile_layer_estimated_gpu_bytes=int(last.tile_layer_estimated_gpu_bytes),
        tile_layer_cpu_shadow_bytes=int(last.tile_layer_cpu_shadow_bytes),
        tile_layer_page_count=int(last.tile_layer_page_count),
        tile_layer_active_pages=int(last.tile_layer_active_pages),
        tile_layer_device_max_texture_size=int(last.tile_layer_device_max_texture_size),
        tile_layer_budget_bytes=int(last.tile_layer_budget_bytes),
        tile_layer_near_resident_items=int(last.tile_layer_near_resident_items),
        tile_layer_warm_resident_items=int(last.tile_layer_warm_resident_items),
        tile_layer_evicted_near_items=sum(
            int(timing.tile_layer_evicted_near_items) for timing in timings
        ),
        tile_layer_lod_level=int(last.tile_layer_lod_level),
        tile_layer_lod_factor=int(last.tile_layer_lod_factor),
        tile_layer_source_texels_per_pixel=float(last.tile_layer_source_texels_per_pixel),
        tile_layer_gutter_pixels=int(last.tile_layer_gutter_pixels),
        tile_layer_mipmap_updates=sum(int(timing.tile_layer_mipmap_updates) for timing in timings),
        tile_layer_mipmap_available=any(
            bool(timing.tile_layer_mipmap_available) for timing in timings
        ),
        tile_layer_complex_texture_uploads=sum(
            int(timing.tile_layer_complex_texture_uploads) for timing in timings
        ),
        tile_layer_shader_uniform_updates=sum(
            int(timing.tile_layer_shader_uniform_updates) for timing in timings
        ),
        cpu_complex_prep_ms=total("cpu_complex_prep_ms"),
        tile_layer_capacity_warning=str(last.tile_layer_capacity_warning),
    )


def _measure_action(view, action, *, measure_presented: bool) -> _ActionMeasurement:
    if not measure_presented:
        start = perf_counter()
        action()
        return _ActionMeasurement(submission_ms=(perf_counter() - start) * 1000.0)
    return _measure_presented_action(view, action)


def _measure_presented_action(view, action) -> _ActionMeasurement:
    """Measure Qt frame scheduling separately from setter submission."""

    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("presented rendering benchmarks require a QApplication")
    view.resize(900, 700)
    view.show()
    _request_view_update(view)
    for _ in range(3):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

    start_time: float | None = None
    frame_times: list[float] = []
    heartbeat_times: list[float] = []
    paint_target = view.graphicsView.viewport()

    def note_frame(*_args):
        if start_time is not None:
            frame_times.append(perf_counter())

    class PaintProbe(QtCore.QObject):
        def eventFilter(self, obj, event):
            if obj is paint_target and event.type() == QtCore.QEvent.Type.Paint:
                # Timer category: UI cosmetic. Benchmark paint probe
                # defers frame accounting until the current paint returns.
                QtCore.QTimer.singleShot(0, self, note_frame)
            return False

    paint_probe = PaintProbe(paint_target)
    paint_target.installEventFilter(paint_probe)

    loop = QtCore.QEventLoop()
    # Timer category: UI cosmetic. Benchmark heartbeat measures event-loop
    # gaps and does not participate in application data flow.
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(2)
    timeout_ms = max(50, int(os.environ.get("ARRAYSCOPE_BENCH_FRAME_TIMEOUT_MS", "1000")))
    quiet_ms = max(1, int(os.environ.get("ARRAYSCOPE_BENCH_FRAME_QUIET_MS", "8")))
    deadline: float | None = None

    def poll():
        now = perf_counter()
        heartbeat_times.append(now)
        if (frame_times and (now - frame_times[-1]) * 1000.0 >= quiet_ms) or (
            deadline is not None and now >= deadline
        ):
            loop.quit()

    heartbeat.timeout.connect(poll)
    heartbeat.start()
    try:
        start_time = perf_counter()
        heartbeat_times.append(start_time)
        action()
        submission_ms = (perf_counter() - start_time) * 1000.0
        deadline = perf_counter() + timeout_ms / 1000.0
        _request_view_update(view)
        loop.exec()
    finally:
        heartbeat.stop()
        paint_target.removeEventFilter(paint_probe)

    end_time = perf_counter()
    heartbeat_times.append(end_time)
    gaps = [(right - left) * 1000.0 for left, right in itertools.pairwise(heartbeat_times)]
    measurement_start = end_time if start_time is None else start_time
    return _ActionMeasurement(
        submission_ms=float(submission_ms),
        first_frame_ms=None if not frame_times else (frame_times[0] - measurement_start) * 1000.0,
        event_loop_drain_ms=(end_time - measurement_start) * 1000.0,
        frame_count=len(frame_times),
        ui_max_gap_ms=max(gaps, default=0.0),
    )


def _request_view_update(view) -> None:
    try:
        view.graphicsView.viewport().update()
    except Exception:
        view.update()


def assert_optional_perf_gates(results: tuple[RenderingBenchmarkResult, ...]) -> None:
    if os.environ.get("ARRAYSCOPE_PERF_ASSERT") != "1":
        return
    by_name = {result.name: result for result in results}
    required = (
        "pyqtgraph_clean_tile_flush",
        "pyqtgraph_pan_zoom_no_upload",
        "pyqtgraph_one_dirty_tile_commit",
        "pyqtgraph_large_complex_tiled_initial",
        "pyqtgraph_progressive_tile_stream",
    )
    missing_frames = [name for name in required if by_name[name].first_frame_ms is None]
    assert not missing_frames, (
        "performance gates require presented-frame measurements; set "
        f"ARRAYSCOPE_BENCH_PRESENTED=1 (missing: {missing_frames})"
    )
    max_ui_gap = float(os.environ.get("ARRAYSCOPE_PERF_MAX_UI_GAP_MS", "50"))
    max_first_frame = float(os.environ.get("ARRAYSCOPE_PERF_MAX_FIRST_FRAME_MS", "250"))
    for name in required:
        result = by_name[name]
        assert float(result.ui_max_gap_ms or 0.0) <= max_ui_gap, (
            f"{name} starved the Qt event loop for {result.ui_max_gap_ms:.2f} ms"
        )
        assert float(result.first_frame_ms or 0.0) <= max_first_frame, (
            f"{name} first frame took {result.first_frame_ms:.2f} ms"
        )


def benchmark_large_progressive_montage(
    *,
    tile_shape: tuple[int, int] = (336, 336),
    count: int = 272,
    columns: int = 17,
    batch_size: int = 8,
    measure_presented: bool | None = None,
) -> tuple[RenderingBenchmarkResult, ...]:
    """Benchmark the production-scale direct tiled presentation path.

    The defaults mirror the 30M-pixel montage from the motivating diagnostic
    dump.  This is intentionally opt-in: the generated RGB plus scalar tile
    payloads consume roughly 200 MiB before backend storage is counted.
    """

    if measure_presented is None:
        measure_presented = os.environ.get("ARRAYSCOPE_BENCH_PRESENTED") == "1"
    results = []
    view = ImageView2D()
    try:
        results.append(
            _benchmark_progressive_tile_stream_configured(
                view,
                tile_shape=tile_shape,
                count=count,
                columns=columns,
                batch_size=batch_size,
                scenario="stress_large_progressive_tile_stream",
                measure_presented=bool(measure_presented),
            )
        )
    finally:
        view.close()
    return tuple(results)


def run_optional_stress_benchmark() -> tuple[RenderingBenchmarkResult, ...]:
    if os.environ.get("ARRAYSCOPE_RUN_STRESS") != "1":
        return ()
    tile_shape = (
        _positive_env_int("ARRAYSCOPE_STRESS_TILE_HEIGHT", 336),
        _positive_env_int("ARRAYSCOPE_STRESS_TILE_WIDTH", 336),
    )
    count = _positive_env_int("ARRAYSCOPE_STRESS_TILE_COUNT", 272)
    columns = min(count, _positive_env_int("ARRAYSCOPE_STRESS_COLUMNS", 17))
    batch_size = _positive_env_int("ARRAYSCOPE_STRESS_BATCH_SIZE", 8)
    return benchmark_large_progressive_montage(
        tile_shape=tile_shape,
        count=count,
        columns=columns,
        batch_size=batch_size,
    )


def collect_benchmark_samples(
    *,
    runs: int = 1,
    stress: bool = False,
    measure_presented: bool | None = None,
) -> tuple[RenderingBenchmarkSample, ...]:
    samples = []
    runs = max(1, int(runs))
    for run in range(runs):
        results = (
            benchmark_large_progressive_montage(measure_presented=measure_presented)
            if stress
            else benchmark_pyqtgraph_rendering(measure_presented=measure_presented)
        )
        environment = rendering_benchmark_environment(results)
        timestamp = time.time()
        samples.extend(
            RenderingBenchmarkSample(
                run=run,
                timestamp=timestamp,
                environment=environment,
                result=result,
            )
            for result in results
        )
    return tuple(samples)


def write_benchmark_jsonl(path, samples) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_sample_record(sample), sort_keys=True) + "\n")


def rendering_benchmark_environment(results=()) -> RenderingBenchmarkEnvironment:
    qt_api = ""
    qt_version = ""
    try:
        from pyqtgraph.Qt import QT_LIB, QtCore

        qt_api = str(QT_LIB)
        qt_version = str(getattr(QtCore, "QT_VERSION_STR", "") or QtCore.qVersion())
    except Exception:
        pass
    limits = _gpu_limits_from_results(results)
    return RenderingBenchmarkEnvironment(
        os=platform.system(),
        platform=platform.platform(),
        python=sys.version.split()[0],
        qt_api=qt_api,
        qt_version=qt_version,
        xdg_session_type=os.environ.get("XDG_SESSION_TYPE", ""),
        wayland_display=os.environ.get("WAYLAND_DISPLAY", ""),
        display=os.environ.get("DISPLAY", ""),
        qt_qpa_platform=os.environ.get("QT_QPA_PLATFORM", ""),
        desktop_session=os.environ.get("DESKTOP_SESSION", ""),
        xdg_current_desktop=os.environ.get("XDG_CURRENT_DESKTOP", ""),
        gpu_vendor=str(getattr(limits, "vendor", "")),
        gpu_renderer=str(getattr(limits, "renderer", "")),
        gpu_version=str(getattr(limits, "version", "")),
        gpu_max_texture_size=int(getattr(limits, "max_texture_size", 0) or 0),
        gpu_max_texture_image_units=int(getattr(limits, "max_texture_image_units", 0) or 0),
        gpu_limits_source=str(getattr(limits, "source", "")),
        gpu_limit_warnings=tuple(getattr(limits, "warnings", ()) or ()),
    )


def _gpu_limits_from_results(results):
    for result in tuple(results or ()):
        timing = getattr(result, "timing", None)
        if timing is not None and int(
            getattr(timing, "tile_layer_device_max_texture_size", 0) or 0
        ):
            return _GpuLimits(
                max_texture_size=int(timing.tile_layer_device_max_texture_size),
                source="benchmark_timing",
            )
    return _GpuLimits()


def _sample_record(sample: RenderingBenchmarkSample) -> dict:
    record = asdict(sample)
    timing = record["result"]["timing"]
    record["result"]["timing"] = timing
    return record


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(1, int(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _format_optional_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}ms"


# ---- montage pan tile-count scaling sweep -----------------------------------
#
# A single fixed-size stress montage reports one point, and one point has no
# slope.  This sweep re-runs the same pan at several tile counts so
# "panning N tiles costs the same as panning 1 tile" becomes a measurable
# gradient rather than an assertion.
#
# Run it through the headless compositor, not an attached session:
#
#     python -m arrayscope.tools.headless_display -- \
#         env QT_QPA_PLATFORM=wayland \
#         python -m arrayscope.display.rendering_benchmarks --tile-counts 1,16,64,256,512
#
# That launcher pins the Mesa EGL vendor file.  A bare headless Weston sorts
# 10_nvidia.json first and silently measures the dGPU -- the slower path here
# -- and an attached session puts desktop activity inside the timing loop.
# Unset WAYLAND_DISPLAY first: the launcher reuses an active socket if it
# finds one, which lands the run back on the session compositor.
#
# The controlled variable is the *committed* tile count.  Zoom is held fixed
# (so the LOD level cannot move) and the camera is parked at the montage
# centre, so the visible tile set and per-frame drawn area are constant once
# the montage is larger than the viewport.  Only the total number of committed
# tiles changes between rows, which is exactly the quantity the two chains
# below iterate.

DEFAULT_PAN_SCALING_TILE_COUNTS = (1, 16, 64, 256, 512)

# Camera window measured in tile pitches; wide enough that the visible set is
# interior (and therefore constant) for every row above ~9 tiles.
_PAN_SCALING_VIEW_TILES = 2.5


@dataclass(frozen=True)
class MontagePanScalingRow:
    """One tile count's pan measurement, keyed for slope reading."""

    tile_count: int
    columns: int
    tile_shape: tuple[int, int]
    committed_tiles: int
    visible_tiles: int
    lod_levels: tuple[int, ...]
    frames: int
    frame_ms_median: float
    frame_ms_p95: float
    camera_tiles_calls: int
    camera_tiles_ms: float
    set_tiles_calls: int
    set_tiles_ms: float
    viewport_only_calls: int
    viewport_only_ms: float
    present_method: str = "bitmap"
    max_fps: float = 0.0
    draw_error: str = ""

    @property
    def chain_a_ms(self) -> float:
        """Per-presented-frame chain: camera mapping plus instance upload."""

        return float(self.camera_tiles_ms) + float(self.set_tiles_ms)

    @property
    def chain_a_ms_per_frame(self) -> float:
        return self.chain_a_ms / max(1, int(self.frames))

    @property
    def chain_b_ms(self) -> float:
        """Coalesced drag-retarget chain (16 ms timer, controller-owned)."""

        return float(self.viewport_only_ms)

    @property
    def chain_b_ms_per_frame(self) -> float:
        return self.chain_b_ms / max(1, int(self.frames))


class _PanChainProbe:
    """Count and time the two pan chains without perturbing them.

    Each hook is a pass-through wrapper installed on the unbound method for
    the duration of one sweep: it adds two ``perf_counter`` reads and returns
    the callee's value unchanged.  Nothing is rescheduled, coalesced, or
    deferred, so draw pacing is whatever the unmodified code does.  The
    alternative -- editing the measured modules -- would make results
    unattributable while other work is in flight in those same files.
    """

    def __init__(self) -> None:
        self.reset()
        self._restore: list[tuple[object, str, object]] = []

    def reset(self) -> None:
        self.camera_tiles_calls = 0
        self.camera_tiles_ms = 0.0
        self.set_tiles_calls = 0
        self.set_tiles_ms = 0.0
        self.viewport_only_calls = 0
        self.viewport_only_ms = 0.0
        self.retarget_calls = 0
        self.retarget_ms = 0.0
        self.interactive_schedule_calls = 0
        self.interactive_schedule_ms = 0.0
        self.ladder_calls = 0
        self.ladder_ms = 0.0

    def __enter__(self) -> _PanChainProbe:
        from arrayscope.display.wgpu_imageview2d import WgpuImageView2D
        from arrayscope.gpu.wgpu_executor import WgpuPlaneExecutor
        from arrayscope.window.display_presenter import DisplayPresentationMixin
        from arrayscope.window.frame_controller import FrameControllerMixin
        from arrayscope.window.frame_effects import FramePipelineEffects
        from arrayscope.window.frame_runtime import FrameRuntimeMixin

        self._wrap(WgpuImageView2D, "_wgpu_tile_instances", "camera_tiles")
        self._wrap(WgpuPlaneExecutor, "_set_tiles", "set_tiles")
        # Chain B lives on the controller mixin.  It is wrapped even when the
        # active transport cannot reach it, so a zero reads as a measured zero
        # rather than as missing instrumentation.  The import is unconditional
        # like the two above: this module is always present, and swallowing an
        # internal ImportError would hide real breakage (import-health guard).
        self._wrap(FrameControllerMixin, "_try_update_montage_viewport_only", "viewport_only")
        # ``retarget`` is chain B's real entry point; ``viewport_only`` is the
        # fast path it *tries* first.  Wrapping both keeps a retarget that fell
        # through to a full session rebuild from being reported as free.
        self._wrap(FrameRuntimeMixin, "retarget_montage_viewport", "retarget")
        # Non-zero only on the coalesced (pointer-gesture) bridge branch, which
        # is how the emitted table names the path it actually measured.
        self._wrap(
            DisplayPresentationMixin,
            "_schedule_interactive_montage_viewport_update",
            "interactive_schedule",
        )
        # The ladder snapshot under chain B -- the memoization target.  Sub-
        # attributing it here is what makes "how much of chain B is the
        # snapshot" answerable without touching the measured modules.
        self._wrap(FramePipelineEffects, "tile_states", "ladder")
        return self

    def __exit__(self, *_exc) -> None:
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()

    def _wrap(self, owner, name: str, counter: str) -> None:
        original = getattr(owner, name)
        probe = self

        def wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = (perf_counter() - start) * 1000.0
                setattr(probe, f"{counter}_ms", getattr(probe, f"{counter}_ms") + elapsed)
                setattr(probe, f"{counter}_calls", getattr(probe, f"{counter}_calls") + 1)

        self._restore.append((owner, name, original))
        setattr(owner, name, wrapper)


def benchmark_montage_pan_scaling(
    *,
    tile_counts: tuple[int, ...] = DEFAULT_PAN_SCALING_TILE_COUNTS,
    tile_shape: tuple[int, int] = (64, 64),
    steps: int = 40,
    pan_tile_pitches: float = 1.0,
    present_method: str = "bitmap",
    max_fps: float | None = None,
) -> tuple[MontagePanScalingRow, ...]:
    """Pan a montage of each tile count over a fixed world distance.

    Every row pans the same world distance at the same zoom, so a rising
    per-frame cost can only come from the tile count.

    ``present_method`` selects the presentation path, and the two are paced
    very differently.  ``"bitmap"`` goes through rendercanvas, whose
    ``ondemand`` scheduler throttles to ``max_fps=30`` -- a ~33 ms floor that
    swallows the chain cost whole.  ``"screen"`` bypasses rendercanvas for a
    native child driving its own swapchain, so frame time reflects the
    swapchain present mode instead.  Chain A per frame is comparable across
    both; wall-clock frame time is not.

    ``max_fps`` lifts that pace ceiling so frame time measures cost rather
    than waiting -- pass something far above the display rate (1000) to make
    this a throughput benchmark.  ``None`` keeps each path's shipping cadence,
    which is what a user actually experiences.
    """

    from arrayscope.display.wgpu_imageview2d import WgpuImageView2D, import_qrenderwidget

    import_qrenderwidget()
    rows: list[MontagePanScalingRow] = []
    with _PanChainProbe() as probe:
        for tile_count in tile_counts:
            try:
                rows.append(
                    _measure_montage_pan(
                        WgpuImageView2D,
                        probe,
                        tile_count=int(tile_count),
                        tile_shape=tile_shape,
                        steps=max(1, int(steps)),
                        pan_tile_pitches=float(pan_tile_pitches),
                        present_method=str(present_method),
                        max_fps=max_fps,
                    )
                )
            except Exception as exc:
                # A row that cannot render (the executor caps tiles at 512) must
                # be reported as failed, never dropped: an omitted row silently
                # flattens the slope the sweep exists to measure.
                rows.append(
                    _failed_pan_scaling_row(
                        tile_count=int(tile_count),
                        tile_shape=tile_shape,
                        error=f"{type(exc).__name__}: {exc}",
                        present_method=str(present_method),
                        max_fps=max_fps,
                    )
                )
    return tuple(rows)


def _failed_pan_scaling_row(
    *,
    tile_count: int,
    tile_shape: tuple[int, int],
    error: str,
    present_method: str = "bitmap",
    max_fps: float | None = None,
) -> MontagePanScalingRow:
    return MontagePanScalingRow(
        tile_count=tile_count,
        columns=max(1, int(np.ceil(np.sqrt(tile_count)))),
        tile_shape=(int(tile_shape[0]), int(tile_shape[1])),
        committed_tiles=0,
        visible_tiles=0,
        lod_levels=(),
        frames=0,
        frame_ms_median=0.0,
        frame_ms_p95=0.0,
        camera_tiles_calls=0,
        camera_tiles_ms=0.0,
        set_tiles_calls=0,
        set_tiles_ms=0.0,
        viewport_only_calls=0,
        viewport_only_ms=0.0,
        present_method=present_method,
        max_fps=0.0 if max_fps is None else float(max_fps),
        draw_error=error,
    )


def _measure_montage_pan(
    view_type,
    probe: _PanChainProbe,
    *,
    tile_count: int,
    tile_shape: tuple[int, int],
    steps: int,
    pan_tile_pitches: float,
    present_method: str = "bitmap",
    max_fps: float | None = None,
) -> MontagePanScalingRow:
    from pyqtgraph.Qt import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError("the pan scaling sweep requires a QApplication")

    columns = max(1, int(np.ceil(np.sqrt(tile_count))))
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(
        tile_shape=tile_shape, count=tile_count, columns=columns
    )
    tile_h, tile_w = (int(tile_shape[0]), int(tile_shape[1]))
    rows_count = int(np.ceil(tile_count / columns))
    pitch_x = tile_w + 1
    pitch_y = tile_h + 1
    span_x = _PAN_SCALING_VIEW_TILES * pitch_x
    span_y = _PAN_SCALING_VIEW_TILES * pitch_y
    pan_distance = float(pan_tile_pitches) * pitch_x
    # Park the camera at the montage centre so the pan stays interior.
    centre_x = 0.5 * columns * pitch_x
    centre_y = 0.5 * rows_count * pitch_y
    x_start = centre_x - 0.5 * span_x - 0.5 * pan_distance
    y_range = (centre_y - 0.5 * span_y, centre_y + 0.5 * span_y)

    view = view_type(present_method=present_method)
    view.resize(900, 700)
    view.show()
    try:
        # A screen request that quietly resolved to bitmap would be reported
        # as a screen measurement while carrying rendercanvas's 30 fps cap.
        active_present = str(view.wgpuPresentMethod())
        if active_present != present_method:
            raise RuntimeError(
                f"requested present method {present_method!r} resolved to "
                f"{active_present!r}: {view.wgpuPresentMethodFallbackReason()}"
            )
        applied_fps = _apply_pan_pace_ceiling(view, active_present, max_fps)
        _present_benchmark_tiled(
            view,
            geometry=geometry,
            payloads=payloads,
            levels=(0.0, 1.0),
            histogramRange=(0.0, 1.0),
        )
        viewbox = view.view
        viewbox.setRange(xRange=(x_start, x_start + span_x), yRange=y_range, padding=0)
        _drain_pan_frames(app, QtCore, view, int(view._wgpu_draw_count))
        for _ in range(5):
            app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)

        probe.reset()
        frame_ms: list[float] = []
        step_distance = pan_distance / steps
        for step in range(1, steps + 1):
            x0 = x_start + step * step_distance
            before = int(view._wgpu_draw_count)
            start = perf_counter()
            viewbox.setRange(xRange=(x0, x0 + span_x), yRange=y_range, padding=0)
            _drain_pan_frames(app, QtCore, view, before)
            frame_ms.append((perf_counter() - start) * 1000.0)

        instances = view._wgpu_tile_instances()
        committed = len((view._wgpu_committed or {}).get("tiles", {}))
        return MontagePanScalingRow(
            tile_count=tile_count,
            columns=columns,
            tile_shape=(tile_h, tile_w),
            committed_tiles=committed,
            visible_tiles=_visible_instance_count(instances, viewbox.viewRange()),
            lod_levels=tuple(sorted({int(tile.lod_level) for tile in instances})),
            frames=len(frame_ms),
            frame_ms_median=_median(frame_ms),
            frame_ms_p95=_percentile(frame_ms, 95.0),
            camera_tiles_calls=probe.camera_tiles_calls,
            camera_tiles_ms=probe.camera_tiles_ms,
            set_tiles_calls=probe.set_tiles_calls,
            set_tiles_ms=probe.set_tiles_ms,
            viewport_only_calls=probe.viewport_only_calls,
            viewport_only_ms=probe.viewport_only_ms,
            present_method=active_present,
            max_fps=applied_fps,
            draw_error=str(getattr(view, "_wgpu_last_draw_error", "") or ""),
        )
    finally:
        view.close()
        view.deleteLater()
        _collect_benchmark_widgets()


def _apply_pan_pace_ceiling(view, present_method: str, max_fps: float | None) -> float:
    """Lift the draw-pace ceiling so frame time measures cost, not waiting.

    Each path throttles somewhere different, and both knobs are the ones the
    code already exposes for this -- no scheduling logic is replaced, only the
    rate it aims for.  Returns the pace actually applied (0.0 when the
    shipping cadence is left alone) so the row records what produced it.
    """

    if max_fps is None:
        return 0.0
    rate = float(max_fps)
    canvas = getattr(view, "_wgpu_canvas", None)
    if canvas is None:
        raise RuntimeError("pan pace override needs a wgpu canvas")
    if present_method == "screen":
        # Documented setter: "Pin the pace (benchmarks and tests)".
        canvas.max_draws_per_second = rate
        return rate
    # Bitmap rides rendercanvas's ondemand scheduler; raise its max_fps and
    # keep the mode, so coalescing behaviour is unchanged.
    canvas.set_update_mode("ondemand", max_fps=rate)
    return rate


def _drain_pan_frames(app, QtCore, view, before: int, *, timeout_s: float = 2.0) -> bool:
    """Wait for the draw the camera change requested, without forcing one."""

    deadline = perf_counter() + timeout_s
    while perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        if int(getattr(view, "_wgpu_draw_count", 0) or 0) > before:
            return True
    return False


def _visible_instance_count(instances, view_range) -> int:
    """Instances whose world rect intersects the camera's view range.

    ``TileInstance.dst_rect`` stopped being a viewport-relative rect when
    panning moved to the camera uniform (e9956846): it is now the tile's
    camera-free WORLD rect.  Testing it against the unit viewport therefore
    counted only the tile straddling the world origin and reported ``vis = 1``
    for every montage size -- a constant column cannot show a cost that scales
    with the visible set, which is the whole question here.  The camera is the
    ViewBox's range, so visibility is the intersection against that.
    """

    (x_lo, x_hi), (y_lo, y_hi) = view_range
    visible = 0
    for tile in instances:
        x, y, w, h = tile.dst_rect
        if x + w > x_lo and x < x_hi and y + h > y_lo and y < y_hi:
            visible += 1
    return visible


def _median(values) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _percentile(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = round((percentile / 100.0) * len(ordered) + 0.5) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def _fit_ms_per_tile(points) -> float:
    """Least-squares slope of cost against tile count."""

    points = [(float(x), float(y)) for x, y in points]
    count = len(points)
    if count < 2:
        return 0.0
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xx = sum(x * x for x, _ in points)
    sum_xy = sum(x * y for x, y in points)
    denominator = count * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return 0.0
    return (count * sum_xy - sum_x * sum_y) / denominator


def pan_scaling_records(rows, *, environment=None) -> tuple[dict, ...]:
    """JSONL records keyed by tile count so a regression reads as a slope."""

    environment = rendering_benchmark_environment() if environment is None else environment
    timestamp = time.time()
    return tuple(
        {
            "benchmark": "montage_pan_scaling",
            "timestamp": timestamp,
            "tile_count": int(row.tile_count),
            "environment": asdict(environment),
            "row": asdict(row),
            "chain_a_ms": row.chain_a_ms,
            "chain_a_ms_per_frame": row.chain_a_ms_per_frame,
            "chain_b_ms": row.chain_b_ms,
            "chain_b_ms_per_frame": row.chain_b_ms_per_frame,
        }
        for row in rows
    )


def write_pan_scaling_jsonl(path, rows) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in pan_scaling_records(rows):
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def format_pan_scaling_table(rows) -> str:
    """Human-readable sweep summary ending in the figure under test."""

    header = (
        f"{'tiles':>6} {'commit':>6} {'vis':>4} {'lod':>5} {'frames':>6} "
        f"{'frame_med':>10} {'frame_p95':>10} {'A_ms/frame':>11} {'A_calls':>8} "
        f"{'B_ms/frame':>11} {'B_calls':>8}"
    )
    methods = sorted({str(row.present_method) for row in rows})
    paces = sorted({float(row.max_fps) for row in rows})
    pace = ", ".join("display native" if value <= 0.0 else f"{value:g} fps" for value in paces)
    lines = [f"present: {', '.join(methods)}   pace: {pace}", header, "-" * len(header)]
    for row in rows:
        lod = ",".join(str(level) for level in row.lod_levels) or "-"
        lines.append(
            f"{row.tile_count:>6} {row.committed_tiles:>6} {row.visible_tiles:>4} "
            f"{lod:>5} {row.frames:>6} {row.frame_ms_median:>10.3f} "
            f"{row.frame_ms_p95:>10.3f} {row.chain_a_ms_per_frame:>11.4f} "
            f"{row.camera_tiles_calls:>8} {row.chain_b_ms_per_frame:>11.4f} "
            f"{row.viewport_only_calls:>8}"
        )
    lines.extend(f"  !! tiles={row.tile_count}: {row.draw_error}" for row in rows if row.draw_error)
    # Only rows that actually rendered carry a cost; fitting a failed row's
    # zeros would drag the slope toward the answer the sweep is testing for.
    fitted = [row for row in rows if row.frames and not row.draw_error]
    chain_a_slope = _fit_ms_per_tile([(row.tile_count, row.chain_a_ms_per_frame) for row in fitted])
    frame_slope = _fit_ms_per_tile([(row.tile_count, row.frame_ms_median) for row in fitted])
    lines.append("")
    lines.append(f"chain A (per presented frame): {chain_a_slope * 1000.0:+.4f} us per added tile")
    lines.append(f"wall-clock frame time:        {frame_slope * 1000.0:+.4f} us per added tile")
    lines.append("chain A is the figure under test and should trend toward zero.")
    if any(float(row.max_fps) > 0.0 for row in fitted):
        lines.append(
            "Pace ceiling lifted, so frame time measures cost rather than "
            "waiting: it should track the chain A slope, and a gap between "
            "them is per-frame work this sweep is not attributing."
        )
    else:
        lines.append(
            "Wall-clock frame time is pace-quantised at the shipping cadence "
            "(~33 ms bitmap on rendercanvas's ondemand max_fps=30 cap, ~17 ms "
            "screen on the display rate) and cannot show the growth. Re-run "
            "with --pan-max-fps 1000 to make it a throughput measurement."
        )
    return "\n".join(lines)


# ---- montage pan chain B (viewport retarget) sweep --------------------------
#
# The sweep above measures chain A -- the per-presented-frame instance build --
# on a bare ``WgpuImageView2D``.  A bare widget has no controller, so chain B
# (``retarget_montage_viewport`` and the ladder snapshot under it) could never
# run there and reported a structural ``B_calls = 0``.  Chain B needs the real
# object graph: window, document, view state, and a live frame session.
#
# So this sweep opens an actual ArrayScope window per row and pans it with real
# Qt mouse press/move/release on the graphics viewport.  That matters for more
# than realism: ``ViewportBridge`` routes on ``QApplication.mouseButtons()``
# (viewport_bridge.py:83), so a programmatic ``setRange`` takes the IMMEDIATE
# uncoalesced branch while a user's drag takes the 16 ms COALESCED one.  Those
# are different code paths with different retarget counts, and a drag is the
# gesture the reported slideshow comes from.  QTest-posted button state does
# reach ``mouseButtons()`` on this platform (verified: 1 through the drag, 0
# after release), so no simulation or monkeypatch is needed -- the measured
# path is the shipping one.  ``bridge_path`` in the emitted table is derived
# from the probe, not asserted, so a future platform where this stops holding
# reports "immediate" rather than silently mislabelling a coalesced number.
#
# Two visibility regimes, because they answer different questions and the
# reported symptom lives in only one of them:
#
#   fit    -- the whole montage on screen (visible == committed).  This is the
#             reported slideshow: cost should track the VISIBLE set.
#   window -- a few tiles on screen out of a large montage.  This is the "only
#             compute what is needed" claim: if chain B is truly visible-scoped,
#             cost here stays flat as the montage grows behind the viewport.
#
# A per-retarget cost that is flat in committed tiles but linear in visible
# tiles is a completely different finding from one that is linear in both,
# which is why the table reports both counts on every row.

DEFAULT_CHAIN_B_TILE_COUNTS = (16, 64, 144, 256, 400)
CHAIN_B_REGIMES = ("fit", "window")

# Tile pitches across the viewport in the ``window`` regime -- small enough
# that the visible set stays a handful of tiles at every montage size.
_CHAIN_B_WINDOW_VIEW_TILES = 3.0
# QSettings namespace for the benchmark window.  Deliberately NOT the shipping
# application name: the sweep pins a backend and present method, and must not
# rewrite the developer's own persisted choices to do it.
_CHAIN_B_APPLICATION_NAME = "ArrayScopePanBench"


@dataclass(frozen=True)
class MontagePanChainBRow:
    """One (tile count, visibility regime) chain B measurement."""

    tile_count: int
    regime: str
    tile_shape: tuple[int, int]
    committed_tiles: int
    visible_tiles: int
    drag_steps: int
    retarget_calls: int
    retarget_ms: float
    viewport_only_calls: int
    viewport_only_ms: float
    interactive_schedule_calls: int
    ladder_calls: float
    ladder_ms: float
    present_method: str = "screen"
    error: str = ""

    @property
    def bridge_path(self) -> str:
        """Which ``ViewportBridge`` branch the measured retargets came from."""

        if not self.retarget_calls:
            return "none"
        return "coalesced" if self.interactive_schedule_calls else "immediate"

    @property
    def chain_b_ms_per_retarget(self) -> float:
        return float(self.retarget_ms) / max(1, int(self.retarget_calls))

    @property
    def ladder_ms_per_retarget(self) -> float:
        return float(self.ladder_ms) / max(1, int(self.retarget_calls))

    @property
    def ladder_calls_per_retarget(self) -> float:
        return float(self.ladder_calls) / max(1, int(self.retarget_calls))


def benchmark_montage_pan_chain_b(
    *,
    tile_counts: tuple[int, ...] = DEFAULT_CHAIN_B_TILE_COUNTS,
    regimes: tuple[str, ...] = CHAIN_B_REGIMES,
    tile_shape: tuple[int, int] = (64, 64),
    drag_steps: int = 12,
    present_method: str = "screen",
) -> tuple[MontagePanChainBRow, ...]:
    """Drag a real montage window and time the viewport-retarget chain."""

    rows: list[MontagePanChainBRow] = []
    with _PanChainProbe() as probe:
        for regime in regimes:
            for tile_count in tile_counts:
                try:
                    rows.append(
                        _measure_montage_pan_chain_b(
                            probe,
                            tile_count=int(tile_count),
                            regime=str(regime),
                            tile_shape=tile_shape,
                            drag_steps=max(1, int(drag_steps)),
                            present_method=str(present_method),
                        )
                    )
                except Exception as exc:
                    # Same rule as the chain A sweep: a row that could not be
                    # measured is reported as failed, never dropped.
                    rows.append(
                        MontagePanChainBRow(
                            tile_count=int(tile_count),
                            regime=str(regime),
                            tile_shape=(int(tile_shape[0]), int(tile_shape[1])),
                            committed_tiles=0,
                            visible_tiles=0,
                            drag_steps=0,
                            retarget_calls=0,
                            retarget_ms=0.0,
                            viewport_only_calls=0,
                            viewport_only_ms=0.0,
                            interactive_schedule_calls=0,
                            ladder_calls=0,
                            ladder_ms=0.0,
                            present_method=str(present_method),
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
    return tuple(rows)


def _chain_b_montage_data(*, tile_shape: tuple[int, int], count: int) -> np.ndarray:
    """(H, W, N) stack whose frames differ, so no tile can alias another."""

    tile_h, tile_w = int(tile_shape[0]), int(tile_shape[1])
    ramp = np.linspace(0.0, 1.0, tile_h * tile_w, dtype=np.float32).reshape(tile_h, tile_w)
    stack = np.empty((tile_h, tile_w, int(count)), dtype=np.float32)
    for index in range(int(count)):
        stack[:, :, index] = ramp + float(index)
    return stack


def _open_chain_b_window(data: np.ndarray, *, present_method: str):
    """Open a real ArrayScope window pinned to wgpu at ``present_method``."""

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from arrayscope.app.launch import _create_window
    from arrayscope.app.settings_state import ImageRenderingBackendChoice

    pg.mkQApp()
    QtCore.QCoreApplication.setOrganizationName("ArrayScope")
    QtCore.QCoreApplication.setApplicationName(_CHAIN_B_APPLICATION_NAME)
    settings = QtCore.QSettings()
    settings.clear()
    settings.setValue("image_rendering_backend", ImageRenderingBackendChoice.WGPU.value)
    settings.setValue("wgpu_present_method", str(present_method))
    settings.setValue("first_run_hints_dismissed", True)
    settings.sync()
    return _create_window(
        data,
        title="pan-chain-b",
        application_name=_CHAIN_B_APPLICATION_NAME,
    )


def _pump(app, QtCore, seconds: float) -> None:
    deadline = perf_counter() + float(seconds)
    while perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


def _wait_for_committed_tiles(app, QtCore, view, expected: int, *, timeout_s: float) -> int:
    """Pump until every tile is committed, so rows compare like with like."""

    deadline = perf_counter() + float(timeout_s)
    committed = 0
    while perf_counter() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
        committed = len((getattr(view, "_wgpu_committed", None) or {}).get("tiles", {}))
        if committed >= int(expected):
            return committed
    return committed


def _measure_montage_pan_chain_b(
    probe: _PanChainProbe,
    *,
    tile_count: int,
    regime: str,
    tile_shape: tuple[int, int],
    drag_steps: int,
    present_method: str,
) -> MontagePanChainBRow:
    from pyqtgraph.Qt import QtCore

    from arrayscope.display.backend_contract import image_view_backend_capabilities

    app, win = _open_chain_b_window(
        _chain_b_montage_data(tile_shape=tile_shape, count=tile_count),
        present_method=present_method,
    )
    try:
        _pump(app, QtCore, 0.4)
        win._set_view_state(win.view_state.with_montage_axis(2, text=":"))
        win.render(reason="pan-chain-b")
        view = win.img_view
        backend = image_view_backend_capabilities(view).name
        if backend != "wgpu":
            raise RuntimeError(f"expected the wgpu backend, got {backend!r}")
        active_present = str(view.wgpuPresentMethod())
        if active_present != present_method:
            raise RuntimeError(
                f"requested present method {present_method!r} resolved to "
                f"{active_present!r}: {view.wgpuPresentMethodFallbackReason()}"
            )
        committed = _wait_for_committed_tiles(app, QtCore, view, tile_count, timeout_s=30.0)
        _pump(app, QtCore, 0.5)

        viewbox = view.view
        _apply_chain_b_regime(viewbox, regime, tile_shape=tile_shape, tile_count=tile_count)
        _pump(app, QtCore, 1.0)

        probe.reset()
        _drag_viewport(app, QtCore, view, probe, steps=drag_steps)

        instances = view._wgpu_tile_instances()
        return MontagePanChainBRow(
            tile_count=tile_count,
            regime=regime,
            tile_shape=(int(tile_shape[0]), int(tile_shape[1])),
            committed_tiles=committed,
            visible_tiles=_visible_instance_count(instances, viewbox.viewRange()),
            drag_steps=drag_steps,
            retarget_calls=probe.retarget_calls,
            retarget_ms=probe.retarget_ms,
            viewport_only_calls=probe.viewport_only_calls,
            viewport_only_ms=probe.viewport_only_ms,
            interactive_schedule_calls=probe.interactive_schedule_calls,
            ladder_calls=probe.ladder_calls,
            ladder_ms=probe.ladder_ms,
            present_method=active_present,
        )
    finally:
        win.close()
        _pump(app, QtCore, 0.2)
        _collect_benchmark_widgets()


def _apply_chain_b_regime(viewbox, regime: str, *, tile_shape, tile_count: int) -> None:
    """Park the camera so the visible tile set is the regime's controlled variable."""

    if regime == "fit":
        # Whole montage on screen: visible == committed, the reported symptom.
        viewbox.autoRange(padding=0.02)
        return
    if regime != "window":
        raise ValueError(f"unknown chain B visibility regime: {regime!r}")
    # A few tiles on screen at the montage centre, independent of montage size.
    (x_lo, x_hi), (y_lo, y_hi) = viewbox.viewRange()
    centre_x = 0.5 * (x_lo + x_hi)
    centre_y = 0.5 * (y_lo + y_hi)
    span_x = _CHAIN_B_WINDOW_VIEW_TILES * (int(tile_shape[1]) + 1)
    span_y = _CHAIN_B_WINDOW_VIEW_TILES * (int(tile_shape[0]) + 1)
    viewbox.setRange(
        xRange=(centre_x - 0.5 * span_x, centre_x + 0.5 * span_x),
        yRange=(centre_y - 0.5 * span_y, centre_y + 0.5 * span_y),
        padding=0,
    )


def _drag_viewport(app, QtCore, view, probe: _PanChainProbe, *, steps: int) -> None:
    """Pan by a real left-button drag on the graphics viewport.

    Each move is followed by a bounded pump rather than a fixed sleep: the
    coalescing timer is 16 ms and single-shot, so waiting for the retarget it
    scheduled is what makes one move cost one retarget.  Nothing here changes
    the pacing -- the wait only observes it.
    """

    from PySide6.QtTest import QTest

    target = view.getView().scene().views()[0].viewport()
    centre = QtCore.QPoint(target.width() // 2, target.height() // 2)
    left = QtCore.Qt.MouseButton.LeftButton
    no_modifier = QtCore.Qt.KeyboardModifier.NoModifier
    QTest.mousePress(target, left, no_modifier, centre)
    app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
    try:
        for step in range(1, int(steps) + 1):
            before = probe.retarget_calls
            QTest.mouseMove(target, centre + QtCore.QPoint(4 * step, 0))
            deadline = perf_counter() + 1.0
            while perf_counter() < deadline:
                app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)
                if probe.retarget_calls > before:
                    break
    finally:
        QTest.mouseRelease(target, left, no_modifier, centre + QtCore.QPoint(4 * int(steps), 0))
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents)


def chain_b_records(rows, *, environment=None) -> tuple[dict, ...]:
    """JSONL records keyed by regime and tile count."""

    environment = rendering_benchmark_environment() if environment is None else environment
    timestamp = time.time()
    return tuple(
        {
            "benchmark": "montage_pan_chain_b",
            "timestamp": timestamp,
            "tile_count": int(row.tile_count),
            "regime": str(row.regime),
            "environment": asdict(environment),
            "row": asdict(row),
            "bridge_path": row.bridge_path,
            "chain_b_ms_per_retarget": row.chain_b_ms_per_retarget,
            "ladder_ms_per_retarget": row.ladder_ms_per_retarget,
        }
        for row in rows
    )


def write_chain_b_jsonl(path, rows) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in chain_b_records(rows):
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _slope_or_held(points) -> str:
    """Slope text, or an explicit refusal when the variable was held fixed.

    A regime that deliberately pins one variable (``window`` holds the visible
    set constant while the montage grows) yields a denominator near zero, and
    least squares will happily return a large, meaningless number for it.
    Publishing that alongside a real slope invites reading noise as a finding,
    so the degenerate case is named instead of dressed up.
    """

    # Ratio, not absolute spread: the ``window`` regime's visible set drifts
    # 16 -> 20 while the montage grows 25x, and that 1.25x wobble is nowhere
    # near enough leverage to attribute a per-tile cost to.
    spread = {round(float(x)) for x, _ in points}
    if len(spread) < 2 or max(spread) < 2 * max(1, min(spread)):
        return "n/a (held ~constant)"
    return f"{_fit_ms_per_tile(points) * 1000.0:+.2f} us"


def format_chain_b_table(rows) -> str:
    """Chain B summary: cost per retarget against committed and visible tiles."""

    header = (
        f"{'regime':>7} {'tiles':>6} {'commit':>6} {'vis':>5} {'B_calls':>8} "
        f"{'B_ms/retgt':>11} {'vp_only':>8} {'ladder/rt':>10} "
        f"{'ladder_ms':>10} {'path':>10}"
    )
    methods = sorted({str(row.present_method) for row in rows})
    lines = [f"present: {', '.join(methods)}   backend: wgpu", header, "-" * len(header)]
    lines.extend(
        f"{row.regime:>7} {row.tile_count:>6} {row.committed_tiles:>6} "
        f"{row.visible_tiles:>5} {row.retarget_calls:>8} "
        f"{row.chain_b_ms_per_retarget:>11.3f} {row.viewport_only_calls:>8} "
        f"{row.ladder_calls_per_retarget:>10.2f} "
        f"{row.ladder_ms_per_retarget:>10.3f} {row.bridge_path:>10}"
        for row in rows
    )
    lines.extend(f"  !! {row.regime}/{row.tile_count}: {row.error}" for row in rows if row.error)
    lines.append("")
    measured = [row for row in rows if row.retarget_calls and not row.error]
    for regime in sorted({row.regime for row in measured}):
        scoped = [row for row in measured if row.regime == regime]
        if len(scoped) < 2:
            continue
        commit_slope = _slope_or_held(
            [(row.committed_tiles, row.chain_b_ms_per_retarget) for row in scoped]
        )
        visible_slope = _slope_or_held(
            [(row.visible_tiles, row.chain_b_ms_per_retarget) for row in scoped]
        )
        costs = [row.chain_b_ms_per_retarget for row in scoped]
        lines.append(
            f"chain B [{regime}]: {commit_slope} per added COMMITTED tile, "
            f"{visible_slope} per added VISIBLE tile "
            f"({min(costs):.2f} -> {max(costs):.2f} ms/retarget across the sweep)"
        )
    lines.append("")
    lines.append(
        "B_calls > 0 is the acceptance signal: chain B is the coalesced "
        "drag-retarget path, reached here by real Qt mouse events rather than "
        "by setRange, which would take the bridge's immediate branch instead."
    )
    lines.append(
        "Flat in committed but linear in visible means chain B is already "
        "visible-scoped and the cost is the visible set; linear in both means "
        "it still walks the whole montage."
    )
    lines.append(
        "The fitted slopes are a summary, not a model: if the fit regime's "
        "cost rises faster than its own slope predicts, the growth is "
        "superlinear in the visible set and the endpoint ratio above is the "
        "honest figure to quote."
    )
    lines.append(
        "ladder_ms is the snapshot's share of B_ms/retgt -- the memoization "
        "target's own cost, sub-attributed so the surgery can be judged "
        "against what it actually removes."
    )
    return "\n".join(lines)


def _parse_tile_counts(raw: str) -> tuple[int, ...]:
    counts = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 1:
            raise ValueError(f"tile counts must be positive, got {value}")
        counts.append(value)
    if not counts:
        raise ValueError("no tile counts given")
    return tuple(counts)


def main(argv: tuple[str, ...] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--jsonl", type=str, default="")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--presented", action="store_true")
    parser.add_argument(
        "--tile-counts",
        type=str,
        default="",
        help=(
            "comma-separated montage tile counts to sweep the pan cost over, "
            "e.g. 1,16,64,256,512 (the executor caps tiles at 512)"
        ),
    )
    parser.add_argument("--pan-steps", type=int, default=40)
    parser.add_argument("--pan-tile-edge", type=int, default=64)
    parser.add_argument(
        "--pan-present-method",
        choices=("bitmap", "screen"),
        default="bitmap",
        help=(
            "presentation path for the pan sweep; bitmap carries "
            "rendercanvas's 30 fps ondemand cap, screen drives its own swapchain"
        ),
    )
    parser.add_argument(
        "--pan-max-fps",
        type=float,
        default=0.0,
        help=(
            "lift the draw-pace ceiling (e.g. 1000) so frame time measures "
            "cost instead of waiting; 0 keeps each path's shipping cadence"
        ),
    )
    parser.add_argument(
        "--chain-b-tile-counts",
        type=str,
        default="",
        help=(
            "comma-separated montage tile counts for the chain B "
            "(viewport-retarget) sweep, e.g. 16,64,144,256,400; opens a real "
            "window per row and pans it with real Qt mouse events"
        ),
    )
    parser.add_argument(
        "--chain-b-regimes",
        type=str,
        default=",".join(CHAIN_B_REGIMES),
        help=(
            "comma-separated visibility regimes for the chain B sweep: "
            "'fit' puts the whole montage on screen, 'window' leaves a few "
            "tiles visible on a large montage"
        ),
    )
    parser.add_argument("--chain-b-drag-steps", type=int, default=12)
    parser.add_argument("--chain-b-tile-edge", type=int, default=64)
    parser.add_argument(
        "--chain-b-present-method",
        choices=("bitmap", "screen"),
        default="screen",
    )
    args = parser.parse_args(argv)

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg

    pg.mkQApp()
    if args.chain_b_tile_counts:
        regimes = tuple(
            chunk.strip() for chunk in str(args.chain_b_regimes).split(",") if chunk.strip()
        )
        rows = benchmark_montage_pan_chain_b(
            tile_counts=_parse_tile_counts(args.chain_b_tile_counts),
            regimes=regimes or CHAIN_B_REGIMES,
            tile_shape=(int(args.chain_b_tile_edge), int(args.chain_b_tile_edge)),
            drag_steps=int(args.chain_b_drag_steps),
            present_method=str(args.chain_b_present_method),
        )
        if args.jsonl:
            write_chain_b_jsonl(args.jsonl, rows)
        print(format_chain_b_table(rows))
        return
    if args.tile_counts:
        rows = benchmark_montage_pan_scaling(
            tile_counts=_parse_tile_counts(args.tile_counts),
            tile_shape=(int(args.pan_tile_edge), int(args.pan_tile_edge)),
            steps=int(args.pan_steps),
            present_method=str(args.pan_present_method),
            max_fps=float(args.pan_max_fps) if args.pan_max_fps > 0.0 else None,
        )
        if args.jsonl:
            write_pan_scaling_jsonl(args.jsonl, rows)
        print(format_pan_scaling_table(rows))
        return
    stress = bool(args.stress or os.environ.get("ARRAYSCOPE_RUN_STRESS") == "1")
    presented = bool(args.presented or os.environ.get("ARRAYSCOPE_BENCH_PRESENTED") == "1")
    samples = collect_benchmark_samples(runs=args.runs, stress=stress, measure_presented=presented)
    if args.jsonl:
        write_benchmark_jsonl(args.jsonl, samples)
    results = tuple(sample.result for sample in samples)
    for result in results:
        timing = result.timing
        print(
            f"{result.name}: submit={result.submission_ms:.3f} ms "
            f"commits={result.commit_count} "
            f"first_frame={_format_optional_ms(result.first_frame_ms)} "
            f"ui_gap={_format_optional_ms(result.ui_max_gap_ms)} "
            f"frames={result.frame_count} "
            f"total={float(timing.total_ms or 0.0):.3f} ms "
            f"tile_upload={float(timing.tile_layer_upload_ms or 0.0):.3f} ms "
            f"bytes={timing.visible_bytes} "
            f"items={timing.tile_layer_visible_items}/"
            f"{timing.tile_layer_items_updated}/"
            f"{timing.tile_layer_items_skipped} "
            f"resident={timing.tile_layer_resident_items}/"
            f"{timing.tile_layer_storage_capacity} "
            f"texture_uploads={timing.tile_layer_texture_uploads}"
        )
    assert_optional_perf_gates(results)


if __name__ == "__main__":
    main()
