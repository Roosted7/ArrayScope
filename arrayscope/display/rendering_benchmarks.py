"""Small rendering-backend benchmarks for display hot paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from time import perf_counter

import numpy as np

from arrayscope.core.runtime_diagnostics import ImageUploadTiming
from arrayscope.core.view_state import ViewState
from arrayscope.display.geometry import DisplayGeometry, MontageGeometry
from arrayscope.display.imageview2d import ImageView2D
from arrayscope.display.montage import MontageTileState
from arrayscope.window.display_frame import DisplayTilePayload, TilePresentationDelta, TilePresentationState


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
    is presented as GPU execution time because VisPy uploads are deferred.
    Deterministic tests should gate on work counters rather than wall-clock time.
    """

    from arrayscope.display.vispy_imageview2d import VisPyImageView2D

    if measure_presented is None:
        measure_presented = os.environ.get("ARRAYSCOPE_BENCH_PRESENTED") == "1"
    results = []
    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_scalar_level_preview(view, measure_presented=measure_presented))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_complex_tile_level_preview(view, measure_presented=measure_presented))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_clean_tile_flush(view, measure_presented=measure_presented))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_large_complex_tiled_initial(view, measure_presented=measure_presented))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_one_dirty_tile_commit(view, measure_presented=measure_presented))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_pan_zoom_no_upload(view, measure_presented=measure_presented))
        finally:
            view.close()

    for view_type in (ImageView2D, VisPyImageView2D):
        view = view_type()
        try:
            results.append(_benchmark_progressive_tile_stream(view, measure_presented=measure_presented))
        finally:
            view.close()

    return tuple(results)


def _benchmark_scalar_level_preview(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    data = np.linspace(0.0, 1.0, 256 * 256, dtype=np.float32).reshape(256, 256)
    view.setImagePresentation(data, histogramData=data, levels=(0.0, 1.0), histogramRange=(0.0, 1.0))
    measurement = _measure_action(
        view,
        lambda: view._apply_histogram_preview_levels((0.25, 0.85)),
        measure_presented=measure_presented,
    )
    return _result(view, "scalar_level_preview", measurement)


def _benchmark_complex_tile_level_preview(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
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
    measurement = _measure_action(
        view,
        lambda: view._apply_histogram_preview_levels((0.4, 1.0)),
        measure_presented=measure_presented,
    )
    return _result(view, "complex_tile_level_preview", measurement)


def _benchmark_clean_tile_flush(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
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
    measurement = _measure_action(
        view,
        lambda: view.setMontageTileLayerPresentation(
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
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "clean_tile_flush", measurement)


def _benchmark_large_complex_tiled_initial(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
    placeholder, _histogram, geometry, sources, payloads = _direct_tile_layer_inputs(tile_shape=(64, 64), count=128, columns=16)
    measurement = _measure_action(
        view,
        lambda: view.setMontageTileLayerPresentation(
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
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "large_complex_tiled_initial", measurement)


def _benchmark_one_dirty_tile_commit(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
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
    measurement = _measure_action(
        view,
        lambda: view.setMontageTileLayerPresentation(
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
        ),
        measure_presented=measure_presented,
    )
    return _result(view, "one_dirty_tile_commit", measurement)


def _benchmark_pan_zoom_no_upload(view, *, measure_presented: bool) -> RenderingBenchmarkResult:
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
            view.setTiledMontagePresentation(
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


def _result(
    view,
    scenario: str,
    measurement: _ActionMeasurement,
    *,
    timing: ImageUploadTiming | None = None,
    commit_count: int = 1,
) -> RenderingBenchmarkResult:
    backend = str(getattr(view, "rendering_backend_name", type(view).__name__))
    return RenderingBenchmarkResult(
        name=f"{backend}_{scenario}",
        backend=backend,
        scenario=scenario,
        elapsed_ms=float(measurement.submission_ms),
        timing=view.lastImageUploadTiming() if timing is None else timing,
        first_frame_ms=measurement.first_frame_ms,
        event_loop_drain_ms=measurement.event_loop_drain_ms,
        frame_count=int(measurement.frame_count),
        ui_max_gap_ms=measurement.ui_max_gap_ms,
        commit_count=max(1, int(commit_count)),
    )


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
        tile_layer_estimated_gpu_bytes=int(last.tile_layer_estimated_gpu_bytes),
        tile_layer_cpu_shadow_bytes=int(last.tile_layer_cpu_shadow_bytes),
        tile_layer_page_count=int(last.tile_layer_page_count),
        tile_layer_active_pages=int(last.tile_layer_active_pages),
        tile_layer_device_max_texture_size=int(last.tile_layer_device_max_texture_size),
        tile_layer_budget_bytes=int(last.tile_layer_budget_bytes),
        tile_layer_near_resident_items=int(last.tile_layer_near_resident_items),
        tile_layer_warm_resident_items=int(last.tile_layer_warm_resident_items),
        tile_layer_evicted_near_items=sum(int(timing.tile_layer_evicted_near_items) for timing in timings),
        tile_layer_capacity_warning=str(last.tile_layer_capacity_warning),
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
    and texture uploads are deferred, and a paint callback may still precede
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
                    QtCore.QTimer.singleShot(0, note_frame)
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
        "vispy_one_dirty_tile_commit",
        "vispy_large_complex_tiled_initial",
        "vispy_progressive_tile_stream",
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
        from arrayscope.display.vispy_tiled_renderer import query_gpu_device_limits
        from vispy import gloo

        limits = query_gpu_device_limits(gloo)
        if str(getattr(limits, "source", "")) != "fallback":
            return limits
    except Exception:
        pass
    for result in tuple(results or ()):
        timing = getattr(result, "timing", None)
        if timing is not None and int(getattr(timing, "tile_layer_device_max_texture_size", 0) or 0):
            from arrayscope.display.vispy_tiled_renderer import GpuDeviceLimits

            return GpuDeviceLimits(max_texture_size=int(timing.tile_layer_device_max_texture_size), source="benchmark_timing")
    try:
        from arrayscope.display.vispy_tiled_renderer import query_gpu_device_limits
        from vispy import gloo

        return query_gpu_device_limits(gloo)
    except Exception:
        from arrayscope.display.vispy_tiled_renderer import GpuDeviceLimits

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
