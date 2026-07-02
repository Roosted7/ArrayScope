"""Small rendering-backend benchmarks for display hot paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
from time import perf_counter

import numpy as np

from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.core.scheduler import FrameTarget
from arrayscope.core.work_graph import WorkGraph, WorkItem, WorkLane
from arrayscope.core.view_state import ViewState
from arrayscope.display.backend_contract import image_view_backend_capabilities
from arrayscope.display.frame_planner import FramePlanner
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.lod import LOD_POLICY_NATIVE_ONLY, LOD_REASON_NATIVE_SCALE
from arrayscope.display.montage import MontageTileState
from arrayscope.display.model.frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState


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
    work_graph_counters: dict[str, dict[str, int]] = field(default_factory=dict)

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


def benchmark_rendering_backends(*, measure_presented: bool | None = None) -> tuple[RenderingBenchmarkResult, ...]:
    """Compare PyQtGraph and VisPy display-update hot paths.

    ``elapsed_ms``/``submission_ms`` measures CPU-side setter submission only.
    Set ``measure_presented=True`` (or ``ARRAYSCOPE_BENCH_PRESENTED=1``) to also
    observe first-frame scheduling and Qt event-loop starvation.  Neither field
    is presented as GPU execution time because VisPy uploads execute asynchronously.
    Deterministic tests should gate on work counters rather than wall-clock time.
    """

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

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
        for scenario in scenarios:
            for view_type in (ImageView2D, VisPyImageView2D):
                results.append(
                    _run_view_benchmark(
                        view_type,
                        scenario,
                        measure_presented=measure_presented,
                    )
                )

        results.append(
            _run_view_benchmark(
                VisPyImageView2D,
                _benchmark_warm_residency_queue_scaling,
                measure_presented=measure_presented,
            )
        )
        return tuple(results)
    finally:
        _collect_benchmark_widgets()


def _run_view_benchmark(view_type, scenario, *, measure_presented: bool) -> RenderingBenchmarkResult:
    """Run one scenario and close its parentless Qt/VisPy view."""

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
    _present_single_plane_benchmark_tiled(view, data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), histogramPlotData=data)
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
        lambda: _present_single_plane_benchmark_tiled(view, data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), histogramPlotData=data),
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


def _benchmark_one_tile_montage_initial(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(128, 128), count=1, columns=1)
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


def _benchmark_multi_tile_montage_initial(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=16, columns=4)
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


def _benchmark_large_histogram_plot_refresh(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    data = np.linspace(0.0, 1.0, 1024 * 1024, dtype=np.float32).reshape(1024, 1024)
    _present_single_plane_benchmark_tiled(view, data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0), histogramPlotData=data)
    measurement = _measure_action(
        view,
        lambda: view._refresh_histogram_plot(auto_level=False),
        measure_presented=measure_presented,
    )
    return _result(view, "large_histogram_plot_refresh", measurement)


def _benchmark_complex_tile_level_preview(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(96, 96), count=2, columns=2)
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


def _benchmark_large_tile_level_preview(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=96, columns=12)
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


def _benchmark_tile_level_uniform_update(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=24, columns=6)
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
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(96, 96), count=2, columns=2)
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


def _benchmark_large_complex_tiled_initial(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=128, columns=16)
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
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=64, columns=8)
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
    dirty_payloads[3] = DisplayTilePayload(3, payloads[3].source_index, image, payloads[3].histogram_data, ("montage_tile", 3, "dirty"))
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
    _placeholder, _histogram, geometry, _sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=64, columns=8)
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


def _benchmark_progressive_tile_stream(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
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

    _placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(
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
            upserts = visible if end == batch_size else {index: payloads[index] for index in range(dirty_start, end)}
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


def _benchmark_warm_residency_queue_scaling(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    from arrayscope.display.backends.vispy.tiles import take_payload_batch

    active_count = 8
    total_count = 40
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(
        tile_shape=(64, 64),
        count=total_count,
        columns=8,
    )
    active = {index: payloads[index] for index in range(active_count)}
    state = TilePresentationState(payloads)
    delta = TilePresentationDelta(
        structure_revision=1,
        payload_revision=1,
        visibility_revision=1,
        level_revision=1,
        histogram_revision=1,
        viewport_revision=1,
        upserts=active,
        active_tiles=tuple(range(active_count)),
        planned_tiles=tuple(range(total_count)),
        near_tiles=tuple(range(total_count)),
        near_tile_source_ids=sources,
        force_refresh=True,
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
    layer = getattr(view, "_vispy_gpu_montage_layer", None)
    if layer is None:
        raise RuntimeError("VisPy warm-residency benchmark requires a GPU montage layer")
    warm_payloads = {index: payloads[index] for index in range(active_count, total_count)}
    stats = []

    def warm_batches():
        remaining = dict(warm_payloads)
        while remaining:
            batch, remaining = take_payload_batch(remaining, max_items=4, max_bytes=8 * 1024 * 1024)
            stats.append(
                layer.warm_residency(
                    payloads=batch,
                    geometry=geometry,
                    rgb_already_windowed=False,
                    tile_delta=delta,
                    tile_residency_budget_bytes=512 * 1024 * 1024,
                )
            )

    measurement = _measure_action(view, warm_batches, measure_presented=measure_presented)
    return _result(
        view,
        "warm_residency_queue_scaling",
        measurement,
        timing=_sum_gpu_stats(stats, mode="warm_residency_queue_scaling"),
        commit_count=len(stats),
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
    tile_edge = 256 if max(tuple(int(value) for value in data.shape[:2])) > 256 else max(1, int(data.shape[0]), int(data.shape[1]))
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
            ("tiled_region", frame_plan.semantic_key, int(region.region_id), tuple(tile.shape), str(tile.dtype)),
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
    pending_count = max(0, int(getattr(result_timing, "tile_layer_level_update_pending_items", 0) or 0))
    applied_lod = max(1, int(getattr(result_timing, "tile_layer_lod_factor", 1) or 1))
    source_texels = max(0.0, float(getattr(result_timing, "tile_layer_source_texels_per_pixel", 0.0) or 0.0))
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
        work_graph_counters=_backend_commit_work_counters(
            backend=backend,
            scenario=scenario,
            commit_count=max(1, int(commit_count)),
        ),
    )


def _backend_commit_work_counters(*, backend: str, scenario: str, commit_count: int) -> dict[str, dict[str, int]]:
    graph = WorkGraph()
    for index in range(max(1, int(commit_count))):
        target = FrameTarget(
            semantic_key=("benchmark", str(backend), str(scenario)),
            viewport_key=None,
            presentation_key=("backend-commit", int(index)),
            quality="exact-visible",
        )
        graph.complete_inline(
            WorkItem(
                key=("benchmark_backend_commit", str(backend), str(scenario), int(index)),
                lane=WorkLane.BACKEND_COMMIT,
                frame_target=target,
                supersession_key=("benchmark-backend-commit", str(backend), str(scenario), int(index)),
                supersession_value=int(index),
            )
        )
    return dict(graph.diagnostics().lanes)


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
        tile_layer_rgb_window_tiles=sum(int(timing.tile_layer_rgb_window_tiles) for timing in timings),
        tile_layer_resident_items=int(last.tile_layer_resident_items),
        tile_layer_storage_capacity=int(last.tile_layer_storage_capacity),
        tile_layer_storage_rebuilds=sum(int(timing.tile_layer_storage_rebuilds) for timing in timings),
        tile_layer_storage_evictions=sum(int(timing.tile_layer_storage_evictions) for timing in timings),
        tile_layer_texture_uploads=sum(int(timing.tile_layer_texture_uploads) for timing in timings),
        tile_layer_texture_upload_bytes=sum(int(timing.tile_layer_texture_upload_bytes) for timing in timings),
        tile_layer_vertex_uploads=sum(int(timing.tile_layer_vertex_uploads) for timing in timings),
        tile_layer_level_updates=sum(int(timing.tile_layer_level_updates) for timing in timings),
        tile_layer_level_update_pending_items=sum(int(timing.tile_layer_level_update_pending_items) for timing in timings),
        tile_layer_estimated_gpu_bytes=int(last.tile_layer_estimated_gpu_bytes),
        tile_layer_cpu_shadow_bytes=int(last.tile_layer_cpu_shadow_bytes),
        tile_layer_page_count=int(last.tile_layer_page_count),
        tile_layer_active_pages=int(last.tile_layer_active_pages),
        tile_layer_device_max_texture_size=int(last.tile_layer_device_max_texture_size),
        tile_layer_budget_bytes=int(last.tile_layer_budget_bytes),
        tile_layer_near_resident_items=int(last.tile_layer_near_resident_items),
        tile_layer_warm_resident_items=int(last.tile_layer_warm_resident_items),
        tile_layer_evicted_near_items=sum(int(timing.tile_layer_evicted_near_items) for timing in timings),
        tile_layer_lod_level=int(last.tile_layer_lod_level),
        tile_layer_lod_factor=int(last.tile_layer_lod_factor),
        tile_layer_source_texels_per_pixel=float(last.tile_layer_source_texels_per_pixel),
        tile_layer_gutter_pixels=int(last.tile_layer_gutter_pixels),
        tile_layer_mipmap_updates=sum(int(timing.tile_layer_mipmap_updates) for timing in timings),
        tile_layer_mipmap_available=any(bool(timing.tile_layer_mipmap_available) for timing in timings),
        tile_layer_complex_texture_uploads=sum(int(timing.tile_layer_complex_texture_uploads) for timing in timings),
        tile_layer_shader_uniform_updates=sum(int(timing.tile_layer_shader_uniform_updates) for timing in timings),
        cpu_complex_prep_ms=total("cpu_complex_prep_ms"),
        tile_layer_capacity_warning=str(last.tile_layer_capacity_warning),
    )


def _sum_gpu_stats(stats, *, mode: str) -> ImageUploadTiming:
    stats = tuple(stats)
    if not stats:
        return ImageUploadTiming(mode=mode)
    last = stats[-1]

    def total(field):
        values = [getattr(stat, field) for stat in stats]
        finite = [float(value) for value in values if value is not None]
        return sum(finite) if finite else None

    return ImageUploadTiming(
        total_ms=total("upload_ms"),
        tile_layer_upload_ms=total("upload_ms"),
        visible_bytes=0,
        visible_pixels=0,
        fast_same_object=False,
        mode=mode,
        tile_layer_visible_items=int(last.visible_items),
        tile_layer_items_updated=sum(int(stat.items_updated) for stat in stats),
        tile_layer_items_skipped=sum(int(stat.items_skipped) for stat in stats),
        tile_layer_resident_items=int(last.resident_items),
        tile_layer_storage_capacity=int(last.storage_capacity),
        tile_layer_storage_rebuilds=sum(int(stat.storage_rebuilds) for stat in stats),
        tile_layer_storage_evictions=sum(int(stat.storage_evictions) for stat in stats),
        tile_layer_texture_uploads=sum(int(stat.texture_uploads) for stat in stats),
        tile_layer_texture_upload_bytes=sum(int(stat.texture_upload_bytes) for stat in stats),
        tile_layer_vertex_uploads=sum(int(stat.vertex_uploads) for stat in stats),
        tile_layer_level_updates=sum(int(stat.level_updates) for stat in stats),
        tile_layer_estimated_gpu_bytes=int(last.estimated_gpu_bytes),
        tile_layer_cpu_shadow_bytes=int(last.cpu_shadow_bytes),
        tile_layer_page_count=int(last.page_count),
        tile_layer_active_pages=int(last.active_pages),
        tile_layer_device_max_texture_size=int(last.device_max_texture_size),
        tile_layer_budget_bytes=int(last.budget_bytes),
        tile_layer_near_resident_items=int(last.near_resident_items),
        tile_layer_warm_resident_items=int(last.warm_resident_items),
        tile_layer_evicted_near_items=sum(int(stat.evicted_near_items) for stat in stats),
        tile_layer_lod_level=int(last.lod_level),
        tile_layer_lod_factor=int(last.lod_factor),
        tile_layer_source_texels_per_pixel=float(last.source_texels_per_pixel),
        tile_layer_gutter_pixels=int(last.gutter_pixels),
        tile_layer_mipmap_updates=sum(int(stat.mipmap_updates) for stat in stats),
        tile_layer_mipmap_available=any(bool(stat.mipmap_available) for stat in stats),
        tile_layer_complex_texture_uploads=sum(int(stat.complex_texture_uploads) for stat in stats),
        tile_layer_shader_uniform_updates=sum(int(stat.shader_uniform_updates) for stat in stats),
        tile_layer_capacity_warning=str(last.capacity_warning),
    )


def _measure_action(view, action, *, measure_presented: bool) -> _ActionMeasurement:
    if not measure_presented:
        start = perf_counter()
        action()
        return _ActionMeasurement(submission_ms=(perf_counter() - start) * 1000.0)
    return _measure_presented_action(view, action)


def _measure_presented_action(view, action) -> _ActionMeasurement:
    """Measure Qt/VisPy frame scheduling separately from setter submission.

    This observes the first draw/paint callback and event-loop starvation.  It
    deliberately does not claim to be GPU execution time: VisPy's GL commands
    and texture uploads execute asynchronously, and a paint callback may still precede
    the final compositor scan-out.
    """

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
    draw_events = getattr(getattr(view, "_vispy_canvas", None), "events", None)
    draw_emitter = getattr(draw_events, "draw", None)
    paint_target = None
    paint_probe = None

    def note_frame(*_args):
        if start_time is not None:
            frame_times.append(perf_counter())

    if draw_emitter is not None:
        draw_emitter.connect(note_frame)
    else:
        paint_target = view.graphicsView.viewport()

        class PaintProbe(QtCore.QObject):
            def eventFilter(self, obj, event):
                if obj is paint_target and event.type() == QtCore.QEvent.Type.Paint:
                    QtCore.QTimer.singleShot(0, self, note_frame)
                return False

        paint_probe = PaintProbe(paint_target)
        paint_target.installEventFilter(paint_probe)

    loop = QtCore.QEventLoop()
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(2)
    timeout_ms = max(50, int(os.environ.get("ARRAYSCOPE_BENCH_FRAME_TIMEOUT_MS", "1000")))
    quiet_ms = max(1, int(os.environ.get("ARRAYSCOPE_BENCH_FRAME_QUIET_MS", "8")))
    deadline: float | None = None

    def poll():
        now = perf_counter()
        heartbeat_times.append(now)
        if frame_times and (now - frame_times[-1]) * 1000.0 >= quiet_ms:
            loop.quit()
        elif deadline is not None and now >= deadline:
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
        if draw_emitter is not None:
            try:
                draw_emitter.disconnect(note_frame)
            except Exception:
                pass
        if paint_target is not None and paint_probe is not None:
            paint_target.removeEventFilter(paint_probe)

    end_time = perf_counter()
    heartbeat_times.append(end_time)
    gaps = [
        (right - left) * 1000.0
        for left, right in zip(heartbeat_times, heartbeat_times[1:])
    ]
    measurement_start = end_time if start_time is None else start_time
    return _ActionMeasurement(
        submission_ms=float(submission_ms),
        first_frame_ms=None if not frame_times else (frame_times[0] - measurement_start) * 1000.0,
        event_loop_drain_ms=(end_time - measurement_start) * 1000.0,
        frame_count=len(frame_times),
        ui_max_gap_ms=max(gaps, default=0.0),
    )


def _request_view_update(view) -> None:
    canvas = getattr(view, "_vispy_canvas", None)
    if canvas is not None:
        canvas.update()
    try:
        view.graphicsView.viewport().update()
    except Exception:
        view.update()


def assert_optional_perf_gates(results: tuple[RenderingBenchmarkResult, ...]) -> None:
    if os.environ.get("ARRAYSCOPE_PERF_ASSERT") != "1":
        return
    by_name = {result.name: result for result in results}
    required = (
        "vispy_clean_tile_flush",
        "vispy_pan_zoom_no_upload",
        "vispy_tile_level_uniform_update",
        "vispy_one_dirty_tile_commit",
        "vispy_large_complex_tiled_initial",
        "vispy_progressive_tile_stream",
        "vispy_warm_residency_queue_scaling",
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

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    if measure_presented is None:
        measure_presented = os.environ.get("ARRAYSCOPE_BENCH_PRESENTED") == "1"
    results = []
    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
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
            else benchmark_rendering_backends(measure_presented=measure_presented)
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
        qt_version = str(QtCore.QT_VERSION_STR)
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
    try:
        from arrayscope.display.backends.vispy.tiles import query_gpu_device_limits
        from vispy import gloo

        limits = query_gpu_device_limits(gloo)
        if str(getattr(limits, "source", "")) != "fallback":
            return limits
    except Exception:
        pass
    for result in tuple(results or ()):
        timing = getattr(result, "timing", None)
        if timing is not None and int(getattr(timing, "tile_layer_device_max_texture_size", 0) or 0):
            from arrayscope.display.backends.vispy.tiles import GpuDeviceLimits

            return GpuDeviceLimits(max_texture_size=int(timing.tile_layer_device_max_texture_size), source="benchmark_timing")
    try:
        from arrayscope.display.backends.vispy.tiles import query_gpu_device_limits
        from vispy import gloo

        return query_gpu_device_limits(gloo)
    except Exception:
        from arrayscope.display.backends.vispy.tiles import GpuDeviceLimits

        return GpuDeviceLimits()


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


def main(argv: tuple[str, ...] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--jsonl", type=str, default="")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--presented", action="store_true")
    args = parser.parse_args(argv)

    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg

    pg.mkQApp()
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
