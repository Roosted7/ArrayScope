"""Realistic histogram/level evidence benchmark matrix.

This complements the real-window ``profile_montage_workflow`` gate. It
measures the pure production algorithms across dtype, storage layout,
distribution, and population size, then optionally runs the resident WGPU
benchmark on every requested physical adapter. Timing assertions intentionally
remain outside CI; deterministic correctness and route coverage do not.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from arrayscope.core.window_levels import (
    LevelSource,
    LevelSourceRank,
    WindowLevelController,
)
from arrayscope.display.histogram_plot import HistogramPlotRequest, compute_histogram_plot
from arrayscope.display.model.montage_levels import (
    MontageLevelTracker,
    aggregate_histogram_samples,
    sample_tile_level_stats,
)

DEFAULT_DATA_PATH = Path(
    "/home/thomas/projects/ArrayScope/data/_WIPDelRec-tT2_20260223150234_14.nii"
)
DEFAULT_DTYPES = ("uint8", "int16", "uint16", "float32", "float64", "complex64")
DEFAULT_LAYOUTS = ("contiguous", "strided", "npy_mmap")
DEFAULT_DISTRIBUTIONS = ("gradient", "outliers", "nonfinite")
DEFAULT_SOURCE_COUNTS = (1, 60, 272)
DEFAULT_ENGINES = ("cpu", "wgpu-low-power", "wgpu-high-performance")


@dataclass(frozen=True)
class CpuCase:
    dtype: str
    layout: str
    distribution: str
    source_count: int

    @property
    def name(self) -> str:
        return f"{self.dtype}:{self.layout}:{self.distribution}:{self.source_count}"


@dataclass(frozen=True)
class CpuMeasurement:
    case: str
    repetition: int
    dtype: str
    layout: str
    distribution: str
    source_count: int
    pixels_per_source: int
    source_prepare_ms: float
    evidence_build_ms: float
    prepared_install_ms: float
    aggregate_ms: float
    plot_ms: float
    total_ms: float
    exact_bounds: tuple[float, float] | None
    measured_bounds: tuple[float, float] | None
    bounds_correct: bool
    covered_sources: int
    histogram_populated: bool
    transition_flicker_free: bool


def benchmark_cases(suite: str) -> tuple[CpuCase, ...]:
    """Return a bounded matrix; exhaustive is deliberately opt-in."""

    suite = str(suite)
    if suite == "smoke":
        return (
            CpuCase("uint8", "contiguous", "gradient", 1),
            CpuCase("float32", "strided", "nonfinite", 4),
            CpuCase("complex64", "contiguous", "outliers", 4),
        )
    if suite == "exhaustive":
        return tuple(
            CpuCase(dtype, layout, distribution, count)
            for dtype in DEFAULT_DTYPES
            for layout in DEFAULT_LAYOUTS
            for distribution in DEFAULT_DISTRIBUTIONS
            for count in DEFAULT_SOURCE_COUNTS
        )
    if suite != "representative":
        raise ValueError(f"unknown histogram benchmark suite: {suite!r}")
    rows = [
        *(CpuCase(dtype, "contiguous", "gradient", 60) for dtype in DEFAULT_DTYPES),
        *(CpuCase("float32", layout, "outliers", 60) for layout in DEFAULT_LAYOUTS),
        *(CpuCase("float32", "strided", "gradient", count) for count in DEFAULT_SOURCE_COUNTS),
        CpuCase("float32", "contiguous", "nonfinite", 60),
        CpuCase("complex64", "strided", "nonfinite", 60),
    ]
    return tuple(dict.fromkeys(rows))


def _source_values(
    dtype: np.dtype,
    *,
    shape: tuple[int, int],
    source_index: int,
    distribution: str,
) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    values = 0.7 * x + 1.3 * y + 11.0 * int(source_index)
    if distribution == "outliers":
        values = np.sin(values / 13.0) * 300.0
        values[0, 0] = -20_000.0
        values[-1, -1] = 50_000.0
    elif distribution == "nonfinite":
        values = np.cos(values / 17.0) * 1_000.0
        values[0, 0] = np.nan
        values[0, 1] = np.inf
        values[0, 2] = -np.inf
    elif distribution != "gradient":
        raise ValueError(f"unknown distribution: {distribution!r}")
    if np.issubdtype(dtype, np.complexfloating):
        imag = np.flip(values, axis=1) * 0.25
        with np.errstate(invalid="ignore"):
            return np.asarray(values + 1j * imag, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        finite = np.nan_to_num(values, nan=0.0, posinf=info.max, neginf=info.min)
        return np.asarray(np.clip(finite, info.min, info.max), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _mapped_finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.abs(array)
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return flat[np.isfinite(flat)]


def _case_sources(case: CpuCase, shape: tuple[int, int], directory: Path):
    dtype = np.dtype(case.dtype)
    if case.layout == "npy_mmap":
        path = directory / f"{case.name.replace(':', '-')}.npy"
        mapped = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(int(case.source_count), *shape),
        )
        for source_index in range(case.source_count):
            mapped[source_index] = _source_values(
                dtype,
                shape=shape,
                source_index=source_index,
                distribution=case.distribution,
            )
        mapped.flush()
        del mapped
        source = np.load(path, mmap_mode="r")
        return tuple(source[index] for index in range(case.source_count))
    rows = []
    for source_index in range(case.source_count):
        if case.layout == "strided":
            backing = _source_values(
                dtype,
                shape=(shape[0] * 2, shape[1] * 2),
                source_index=source_index,
                distribution=case.distribution,
            )
            rows.append(backing[::2, ::2])
        elif case.layout == "contiguous":
            rows.append(
                _source_values(
                    dtype,
                    shape=shape,
                    source_index=source_index,
                    distribution=case.distribution,
                )
            )
        else:
            raise ValueError(f"unknown storage layout: {case.layout!r}")
    return tuple(rows)


def _flicker_free_transition(stats) -> bool:
    controller = WindowLevelController()
    predecessor = LevelSource(
        levels=(-2_000.0, 2_000.0),
        histogram_range=(-2_000.0, 2_000.0),
        rank=LevelSourceRank.MONTAGE_COMPLETE,
        source_count=len(stats),
        expected_count=len(stats),
        semantic_key=("predecessor",),
    )
    previous = predecessor
    for count in range(1, len(stats)):
        selected = stats[:count]
        bounds = (
            min(float(row.bounds[0]) for row in selected if row.bounds is not None),
            max(float(row.bounds[1]) for row in selected if row.bounds is not None),
        )
        candidate = LevelSource(
            levels=bounds,
            histogram_range=bounds,
            rank=LevelSourceRank.MONTAGE_VISIBLE_SUBSET,
            source_count=count,
            expected_count=len(stats),
            semantic_key=("successor",),
            evidence_quality=1,
        )
        current = controller.decide(previous=previous, candidate=candidate, mode="relative")
        if current.display_levels != predecessor.levels:
            return False
        previous = current
    complete_bounds = (
        min(float(row.bounds[0]) for row in stats if row.bounds is not None),
        max(float(row.bounds[1]) for row in stats if row.bounds is not None),
    )
    complete = LevelSource(
        levels=complete_bounds,
        histogram_range=complete_bounds,
        rank=LevelSourceRank.MONTAGE_COMPLETE,
        source_count=len(stats),
        expected_count=len(stats),
        semantic_key=("successor",),
        evidence_quality=2,
    )
    switched = controller.decide(previous=previous, candidate=complete, mode="relative")
    return bool(switched.semantic_key == ("successor",))


def measure_cpu_case(
    case: CpuCase,
    *,
    shape: tuple[int, int],
    repetition: int,
    directory: Path,
) -> CpuMeasurement:
    started = perf_counter()
    prepare_started = perf_counter()
    sources = _case_sources(case, shape, directory)
    source_prepare_ms = (perf_counter() - prepare_started) * 1000.0

    evidence_started = perf_counter()
    stats = tuple(
        sample_tile_level_stats(values, source_index, refined=False)
        for source_index, values in enumerate(sources)
    )
    stats = tuple(row for row in stats if row is not None)
    evidence_build_ms = (perf_counter() - evidence_started) * 1000.0

    tracker = MontageLevelTracker()
    key = ("histogram-benchmark", case.name, repetition)
    tracker.ensure_expected(key, range(case.source_count))
    install_started = perf_counter()
    for row in stats:
        tracker.update_from_stats(key, row, aggregate=False)
    summary = tracker.summary_for(key)
    prepared_install_ms = (perf_counter() - install_started) * 1000.0

    aggregate_started = perf_counter()
    snapshot = tracker.histogram_aggregate_snapshot(key)
    aggregate = None if snapshot is None else aggregate_histogram_samples(snapshot[3])
    aggregate_ms = (perf_counter() - aggregate_started) * 1000.0

    plot_started = perf_counter()
    plot = compute_histogram_plot(
        HistogramPlotRequest(
            data=np.asarray((), dtype=np.float32) if aggregate is None else aggregate,
            source_identity=key,
            histogram_bounds=None if summary is None else summary.bounds,
            visible_value_span=None,
            pixel_extent=320.0,
        )
    )
    plot_ms = (perf_counter() - plot_started) * 1000.0

    finite_rows = tuple(_mapped_finite(source) for source in sources)
    finite_rows = tuple(row for row in finite_rows if row.size)
    exact_bounds = (
        None
        if not finite_rows
        else (
            min(float(np.min(row)) for row in finite_rows),
            max(float(np.max(row)) for row in finite_rows),
        )
    )
    measured_bounds = None if summary is None else summary.bounds
    bounds_correct = bool(
        (exact_bounds is None and measured_bounds is None)
        or (
            exact_bounds is not None
            and measured_bounds is not None
            # Per-source constant arrays are deliberately expanded by
            # normalize_bounds so the display window has non-zero width.
            # The truthful invariant is containment of the exact finite
            # population, not bit-identical raw extrema.
            and measured_bounds[0] <= exact_bounds[0]
            and measured_bounds[1] >= exact_bounds[1]
        )
    )
    return CpuMeasurement(
        case=case.name,
        repetition=int(repetition),
        dtype=case.dtype,
        layout=case.layout,
        distribution=case.distribution,
        source_count=int(case.source_count),
        pixels_per_source=int(np.prod(shape)),
        source_prepare_ms=float(source_prepare_ms),
        evidence_build_ms=float(evidence_build_ms),
        prepared_install_ms=float(prepared_install_ms),
        aggregate_ms=float(aggregate_ms),
        plot_ms=float(plot_ms),
        total_ms=(perf_counter() - started) * 1000.0,
        exact_bounds=exact_bounds,
        measured_bounds=measured_bounds,
        bounds_correct=bounds_correct,
        covered_sources=0 if summary is None else len(summary.source_indices),
        histogram_populated=bool(plot.has_data),
        transition_flicker_free=_flicker_free_transition(stats),
    )


def _summarize_cpu(rows: tuple[CpuMeasurement, ...]) -> tuple[dict[str, object], ...]:
    summaries = []
    for case in sorted({row.case for row in rows}):
        selected = tuple(row for row in rows if row.case == case)
        first = selected[0]
        summaries.append(
            {
                "case": case,
                "dtype": first.dtype,
                "layout": first.layout,
                "distribution": first.distribution,
                "source_count": first.source_count,
                "repetitions": len(selected),
                **{
                    f"{field}_median": float(
                        statistics.median(float(getattr(row, field)) for row in selected)
                    )
                    for field in (
                        "source_prepare_ms",
                        "evidence_build_ms",
                        "prepared_install_ms",
                        "aggregate_ms",
                        "plot_ms",
                        "total_ms",
                    )
                },
                "correct": all(
                    row.bounds_correct
                    and row.covered_sources == row.source_count
                    and row.histogram_populated
                    and row.transition_flicker_free
                    for row in selected
                ),
            }
        )
    return tuple(summaries)


def run_benchmark(
    *,
    suite: str,
    engines: tuple[str, ...],
    shape: tuple[int, int],
    repetitions: int,
    data_path: Path = DEFAULT_DATA_PATH,
) -> dict[str, object]:
    unknown = set(engines).difference(DEFAULT_ENGINES)
    if unknown:
        raise ValueError(f"unknown histogram benchmark engines: {sorted(unknown)}")
    cpu_rows = []
    if "cpu" in engines:
        with tempfile.TemporaryDirectory(prefix="arrayscope-histogram-benchmark-") as tmp:
            directory = Path(tmp)
            for case in benchmark_cases(suite):
                cpu_rows.extend(
                    measure_cpu_case(
                        case,
                        shape=shape,
                        repetition=repetition,
                        directory=directory,
                    )
                    for repetition in range(max(1, int(repetitions)))
                )
    gpu = {}
    for engine, preference in (
        ("wgpu-low-power", "low-power"),
        ("wgpu-high-performance", "high-performance"),
    ):
        if engine not in engines:
            continue
        from arrayscope.tools.g6_histogram_benchmark import run_benchmark as run_gpu

        gpu[engine] = run_gpu(
            data_path=Path(data_path),
            repetitions=max(1, int(repetitions)),
            warmups=1,
            power_preference=preference,
        )
    cpu_tuple = tuple(cpu_rows)
    return {
        "schema": "arrayscope.histogram-pipeline-benchmark.v1",
        "suite": str(suite),
        "engines": list(engines),
        "shape": [int(shape[0]), int(shape[1])],
        "cpu": {
            "measurements": [asdict(row) for row in cpu_tuple],
            "summaries": list(_summarize_cpu(cpu_tuple)),
        },
        "gpu": gpu,
        "route_contract": {
            "prepared_page_summary": "reuse on every backend",
            "cpu_semantic_values": "sample when prepared evidence is absent and CPU values exist",
            "resident_gpu_pages": "dispatch only when neither reusable CPU evidence source exists",
        },
        "correct": bool(
            all(row["correct"] for row in _summarize_cpu(cpu_tuple))
            and all(
                all(bool(summary["fits_phase1_budget"]) for summary in report["summaries"])
                for report in gpu.values()
            )
        ),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "git_revision": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), text=True
            ).strip(),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("smoke", "representative", "exhaustive"),
        default="representative",
    )
    parser.add_argument(
        "--engines",
        default=",".join(DEFAULT_ENGINES),
        help="comma-separated cpu,wgpu-low-power,wgpu-high-performance",
    )
    parser.add_argument("--shape", default="336x336")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        height, width = (int(value) for value in str(args.shape).lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise SystemExit("--shape must be HEIGHTxWIDTH") from exc
    engines = tuple(value.strip() for value in str(args.engines).split(",") if value.strip())
    result = run_benchmark(
        suite=str(args.suite),
        engines=engines,
        shape=(height, width),
        repetitions=max(1, int(args.repetitions)),
        data_path=Path(args.data),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correct": result["correct"], "engines": result["engines"]}, indent=2))
    return 0 if bool(result["correct"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
