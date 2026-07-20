"""Measure G6(b) rough versus refined-grade resident GPU evidence.

The benchmark deliberately separates preparation/upload from the measured
evidence pass.  ``rough`` mirrors the live phase-1 contract (64 bins, a
512-value representative sample, and the retained L2 preview population for
montages).  ``exact`` is the collapse candidate: every native resident texel,
the histogram widget's 500-bin cap, exact finite bounds, and an 8192-value
representative sample per semantic source.

The executor runs on the GUI thread, as it does in the live wgpu surface;
completion fencing and bounded readback/sample reconstruction run on a worker.
GPU timestamps cover the bounds and histogram compute passes only.  Uploads are
prepared before measurement, so a failing compute-only result is sufficient to
reject a phase-1 collapse; it cannot be rescued by adding materialization cost.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Thread
from time import perf_counter

import numpy as np

from arrayscope.display.model.montage_levels import REFINED_TILE_SAMPLE_LIMIT
from arrayscope.display.pyramid import reduce_box_mean
from arrayscope.gpu.chunk_summary import representative_sample_from_histogram
from arrayscope.gpu.command_protocol import (
    BindContentPlanes,
    ContentPlane,
    DispatchHistogram,
    EnsureChunkResident,
    FrameSubmission,
)
from arrayscope.gpu.keys import SCALAR_R32F
from arrayscope.gpu.wgpu_executor import PAGE, WgpuPlaneExecutor, plane_chunk_key

ROUGH_BINS = 64
EXACT_BINS = 500
ROUGH_SAMPLE_LIMIT = 512
PHASE1_BUDGET_MS = 16.0
FIRST_PIXEL_TARGET_MS = 2_000.0
LIVE_COMMIT_BATCH_SOURCES = 4
DEFAULT_DATA_PATH = Path(
    "/home/thomas/projects/ArrayScope/data/_WIPDelRec-tT2_20260223150234_14.nii"
)


@dataclass(frozen=True)
class Measurement:
    scenario: str
    variant: str
    repetition: int
    source_count: int
    resident_lod: int
    bins: int
    source_pixels: int
    batch_sources: int
    submit_wall_ms: float
    submit_max_ms: float
    fence_wall_ms: float
    resolve_wall_ms: float
    first_batch_wall_ms: float
    total_wall_ms: float
    gpu_compute_ms: float
    heartbeat_max_gap_ms: float
    reconstructed_values: int
    finite_weight: int


class _HeartbeatProbe:
    def __init__(self, QtCore):
        self._clock = perf_counter
        self._last = self._clock()
        self._gaps: list[float] = []
        self._timer = QtCore.QTimer()
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.setInterval(1)
        self._timer.timeout.connect(self._tick)

    def _tick(self) -> None:
        now = self._clock()
        self._gaps.append((now - self._last) * 1000.0)
        self._last = now

    def start_turn(self) -> None:
        self._gaps.clear()
        self._last = self._clock()
        self._timer.start()

    def finish_turn(self) -> float:
        self._tick()
        self._timer.stop()
        return max(self._gaps, default=0.0)


def _load_data(path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asanyarray(nib.load(str(path)).dataobj)
    if data.ndim != 3:
        raise ValueError(f"G6 benchmark requires one 3-D scalar dataset, got {data.shape}")
    return data


def _centered_sources(depth: int, count: int) -> tuple[int, ...]:
    count = max(1, min(int(depth), int(count)))
    start = (int(depth) - count) // 2
    return tuple(range(start, start + count))


def _scenario_sources(depth: int) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    return (
        ("single_slice", _centered_sources(depth, 1), 0),
        ("montage_60", _centered_sources(depth, 60), 2),
        ("montage_272", _centered_sources(depth, 272), 2),
    )


def _page_rows(
    values: np.ndarray,
    *,
    plane_identity: object,
    lod_level: int,
    source_shape: tuple[int, int],
) -> tuple[tuple[object, np.ndarray], ...]:
    factor = 1 << int(lod_level)
    reduced = (
        np.asarray(values, dtype=np.float32)
        if factor == 1
        else reduce_box_mean(values, (factor, factor))
    )
    rows = []
    grid_h = -(-int(reduced.shape[0]) // PAGE)
    grid_w = -(-int(reduced.shape[1]) // PAGE)
    for chunk_y in range(grid_h):
        for chunk_x in range(grid_w):
            block = reduced[
                chunk_y * PAGE : (chunk_y + 1) * PAGE,
                chunk_x * PAGE : (chunk_x + 1) * PAGE,
            ]
            page = np.zeros((PAGE, PAGE), dtype=np.float32)
            page[: block.shape[0], : block.shape[1]] = block
            key = plane_chunk_key(
                plane_identity,
                "g6b-benchmark",
                int(lod_level),
                int(chunk_x),
                int(chunk_y),
                dtype="float32",
                representation=SCALAR_R32F,
                plane_shape=source_shape,
            )
            rows.append((key, page))
    return tuple(rows)


def _prepare_executor(device, data, sources, rough_lod: int):
    source_shape = tuple(int(value) for value in data.shape[:2])
    exact_pages_per_source = -(-source_shape[0] // PAGE) * -(-source_shape[1] // PAGE)
    rough_shape = tuple(-(-value // (1 << int(rough_lod))) for value in source_shape)
    rough_pages_per_source = -(-rough_shape[0] // PAGE) * -(-rough_shape[1] // PAGE)
    pages_per_source = exact_pages_per_source + (
        0 if int(rough_lod) == 0 else rough_pages_per_source
    )
    executor = WgpuPlaneExecutor(
        pool_layers={SCALAR_R32F: len(sources) * pages_per_source},
        device=device,
    )
    plane_identities = {int(source): ("g6b-real-data", int(source)) for source in sources}
    executor.submit(
        FrameSubmission(
            0,
            (
                BindContentPlanes(
                    tuple(
                        ContentPlane(
                            plane_identities[int(source)],
                            "g6b-benchmark",
                            source_shape,
                            max_lod=int(rough_lod),
                            representation=SCALAR_R32F,
                        )
                        for source in sources
                    )
                ),
            ),
        )
    ).wait_completed()

    exact_keys: dict[int, tuple[object, ...]] = {}
    rough_keys: dict[int, tuple[object, ...]] = {}
    batch_commands = []
    generation = 1
    for offset, source in enumerate(sources):
        plane = np.asarray(data[:, :, int(source)], dtype=np.float32)
        exact_rows = _page_rows(
            plane,
            plane_identity=plane_identities[int(source)],
            lod_level=0,
            source_shape=source_shape,
        )
        rough_rows = (
            exact_rows
            if int(rough_lod) == 0
            else _page_rows(
                plane,
                plane_identity=plane_identities[int(source)],
                lod_level=int(rough_lod),
                source_shape=source_shape,
            )
        )
        exact_keys[int(source)] = tuple(key for key, _page in exact_rows)
        rough_keys[int(source)] = tuple(key for key, _page in rough_rows)
        unique_rows = dict((*exact_rows, *rough_rows))
        batch_commands.extend(EnsureChunkResident(key, page) for key, page in unique_rows.items())
        if len(batch_commands) >= 80 or offset == len(sources) - 1:
            report = executor.submit(FrameSubmission(generation, tuple(batch_commands)))
            report.wait_completed()
            generation += 1
            batch_commands.clear()
    return executor, exact_keys, rough_keys


def _measure_once(
    *,
    app,
    probe,
    executor,
    scenario: str,
    variant: str,
    repetition: int,
    keys_by_source,
    resident_lod: int,
    bins: int,
    sample_limit: int,
) -> Measurement:
    probe.start_turn()
    started = perf_counter()
    submit_times = []
    fence_wall_ms = 0.0
    resolve_wall_ms = 0.0
    gpu_compute_ms = 0.0
    reconstructed_values = 0
    finite_weight = 0
    first_batch_wall_ms = 0.0
    source_rows = tuple(keys_by_source.items())
    for batch_index, batch_start in enumerate(
        range(0, len(source_rows), LIVE_COMMIT_BATCH_SOURCES)
    ):
        batch_started = perf_counter()
        batch = source_rows[batch_start : batch_start + LIVE_COMMIT_BATCH_SOURCES]
        commands = tuple(
            DispatchHistogram(keys, bins=bins, lo=None, hi=None, mode="real")
            for _source, keys in batch
        )
        submit_started = perf_counter()
        report = executor.submit(
            FrameSubmission(
                10_000 + int(repetition) * 1_000 + int(batch_index),
                commands,
            )
        )
        submit_times.append((perf_counter() - submit_started) * 1000.0)
        resolved: dict[str, object] = {}

        def fence_and_resolve(report=report, resolved=resolved) -> None:
            fence_started = perf_counter()
            report.wait_completed()
            fence_finished = perf_counter()
            resolve_started = perf_counter()
            gpu_ms = 0.0
            sample_count = 0
            batch_finite_weight = 0
            for readback in report.histograms.values():
                counts, bounds = readback.resolve()
                batch_finite_weight += int(np.asarray(counts, dtype=np.uint64).sum())
                if bounds is not None:
                    sample_count += int(
                        representative_sample_from_histogram(
                            counts,
                            bounds,
                            sample_limit=int(sample_limit),
                        ).size
                    )
                gpu_ms += float(readback.gpu_elapsed_ms or 0.0)
            resolved.update(
                fence_wall_ms=(fence_finished - fence_started) * 1000.0,
                resolve_wall_ms=(perf_counter() - resolve_started) * 1000.0,
                gpu_compute_ms=gpu_ms,
                reconstructed_values=sample_count,
                finite_weight=batch_finite_weight,
            )

        worker = Thread(target=fence_and_resolve, name="g6b-histogram-readback")
        worker.start()
        while worker.is_alive():
            app.processEvents()
            time.sleep(0.0005)
        worker.join()
        app.processEvents()
        fence_wall_ms += float(resolved["fence_wall_ms"])
        resolve_wall_ms += float(resolved["resolve_wall_ms"])
        gpu_compute_ms += float(resolved["gpu_compute_ms"])
        reconstructed_values += int(resolved["reconstructed_values"])
        finite_weight += int(resolved["finite_weight"])
        if batch_index == 0:
            first_batch_wall_ms = (perf_counter() - batch_started) * 1000.0
    finished = perf_counter()
    heartbeat_max_gap_ms = probe.finish_turn()
    source_pixels = int(
        sum(
            key.chunk_shape[0] * key.chunk_shape[1]
            for keys in keys_by_source.values()
            for key in keys
        )
    )
    return Measurement(
        scenario=str(scenario),
        variant=str(variant),
        repetition=int(repetition),
        source_count=len(keys_by_source),
        resident_lod=int(resident_lod),
        bins=int(bins),
        source_pixels=source_pixels,
        batch_sources=LIVE_COMMIT_BATCH_SOURCES,
        submit_wall_ms=float(sum(submit_times)),
        submit_max_ms=float(max(submit_times, default=0.0)),
        fence_wall_ms=float(fence_wall_ms),
        resolve_wall_ms=float(resolve_wall_ms),
        first_batch_wall_ms=float(first_batch_wall_ms),
        total_wall_ms=(finished - started) * 1000.0,
        gpu_compute_ms=float(gpu_compute_ms),
        heartbeat_max_gap_ms=float(heartbeat_max_gap_ms),
        reconstructed_values=int(reconstructed_values),
        finite_weight=int(finite_weight),
    )


def _percentile(values, percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summaries(rows: tuple[Measurement, ...]) -> tuple[dict[str, object], ...]:
    summaries = []
    identities = sorted({(row.scenario, row.variant) for row in rows})
    for scenario, variant in identities:
        selected = tuple(row for row in rows if row.scenario == scenario and row.variant == variant)
        summary = {
            "scenario": scenario,
            "variant": variant,
            "repetitions": len(selected),
            "source_count": selected[0].source_count,
            "resident_lod": selected[0].resident_lod,
            "bins": selected[0].bins,
            "source_pixels": selected[0].source_pixels,
            "batch_sources": selected[0].batch_sources,
        }
        for field in (
            "submit_wall_ms",
            "submit_max_ms",
            "fence_wall_ms",
            "resolve_wall_ms",
            "first_batch_wall_ms",
            "total_wall_ms",
            "gpu_compute_ms",
            "heartbeat_max_gap_ms",
        ):
            values = [float(getattr(row, field)) for row in selected]
            summary[f"{field}_median"] = float(statistics.median(values))
            summary[f"{field}_p95"] = float(_percentile(values, 95.0))
            summary[f"{field}_max"] = float(max(values))
        summary["phase1_budget_ms"] = PHASE1_BUDGET_MS
        summary["first_pixel_target_ms"] = FIRST_PIXEL_TARGET_MS
        summary["fits_phase1_budget"] = bool(
            summary["submit_max_ms_max"] <= PHASE1_BUDGET_MS
            and summary["heartbeat_max_gap_ms_max"] <= PHASE1_BUDGET_MS
            and summary["first_batch_wall_ms_p95"] <= FIRST_PIXEL_TARGET_MS
        )
        summaries.append(summary)
    return tuple(summaries)


def run_benchmark(*, data_path: Path, repetitions: int, warmups: int) -> dict[str, object]:
    from arrayscope.app.qt_binding import prefer_pyside6

    prefer_pyside6()
    import pyqtgraph as pg
    import wgpu
    from pyqtgraph.Qt import QtCore, QtWidgets
    from wgpu.backends.wgpu_native.extras import set_instance_extras

    app = pg.mkQApp("ArrayScope G6(b) histogram benchmark")
    window = QtWidgets.QWidget()
    window.setWindowTitle("ArrayScope G6(b) evidence benchmark")
    window.resize(640, 360)
    window.show()
    for _ in range(10):
        app.processEvents()

    with contextlib.suppress(RuntimeError):
        set_instance_extras(backends=["Vulkan"])
    adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    if "timestamp-query" not in adapter.features:
        raise RuntimeError("G6(b) benchmark requires GPU timestamp-query support")
    device = adapter.request_device_sync(required_features=["timestamp-query"])
    data = _load_data(data_path)
    probe = _HeartbeatProbe(QtCore)
    rows: list[Measurement] = []
    for scenario, sources, rough_lod in _scenario_sources(data.shape[2]):
        executor, exact_keys, rough_keys = _prepare_executor(device, data, sources, rough_lod)
        variants = (
            ("rough", rough_keys, rough_lod, ROUGH_BINS, ROUGH_SAMPLE_LIMIT),
            ("exact", exact_keys, 0, EXACT_BINS, REFINED_TILE_SAMPLE_LIMIT),
        )
        for variant, keys, lod, bins, sample_limit in variants:
            for repetition in range(-int(warmups), int(repetitions)):
                measurement = _measure_once(
                    app=app,
                    probe=probe,
                    executor=executor,
                    scenario=scenario,
                    variant=variant,
                    repetition=repetition,
                    keys_by_source=keys,
                    resident_lod=lod,
                    bins=bins,
                    sample_limit=sample_limit,
                )
                if repetition >= 0:
                    rows.append(measurement)
        del executor
        gc.collect()
    summaries = _summaries(tuple(rows))
    by_identity = {(str(row["scenario"]), str(row["variant"])): row for row in summaries}
    exact_summaries = []
    for row in summaries:
        if row["variant"] != "exact":
            continue
        rough = by_identity[(str(row["scenario"]), "rough")]
        same_phase1_population = bool(int(row["resident_lod"]) == int(rough["resident_lod"]))
        row["same_phase1_resident_population"] = same_phase1_population
        row["collapse_eligible"] = bool(row["fits_phase1_budget"] and same_phase1_population)
        exact_summaries.append(row)
    return {
        "schema": "arrayscope.g6b-histogram-benchmark.v1",
        "decision": (
            "collapse"
            if exact_summaries and all(bool(row["collapse_eligible"]) for row in exact_summaries)
            else "keep-rough-then-refined"
        ),
        "phase1_budget_ms": PHASE1_BUDGET_MS,
        "qt_platform": str(app.platformName()),
        "qt_platform_env": os.environ.get("QT_QPA_PLATFORM"),
        "adapter": str(adapter.summary),
        "data_path": str(data_path.resolve()),
        "data_shape": [int(value) for value in data.shape],
        "data_dtype": str(data.dtype),
        "git_revision": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(("git", "status", "--porcelain"), text=True).strip()
        ),
        "repetitions": int(repetitions),
        "warmups": int(warmups),
        "measurements": [asdict(row) for row in rows],
        "summaries": list(summaries),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure G6(b) rough versus refined-grade GPU level evidence"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        data_path=Path(args.data),
        repetitions=max(1, int(args.repetitions)),
        warmups=max(0, int(args.warmups)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "summaries": result["summaries"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
