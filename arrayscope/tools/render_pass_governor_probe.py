"""Measure R5 render-pass pacing, throughput, and cost attribution.

This is a real-rendering probe. It always runs in a managed Weston compositor
and writes its distilled dossier to stdout; it never creates JSONL artifacts.
Run the same command from two revisions for a code-baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from statistics import median

from arrayscope.core.trace import TRACE, configure_trace
from arrayscope.tools.headless_display import is_headless_display, run_in_headless_display
from arrayscope.tools.interaction_budget import bounded_interaction_settle_timeout_s
from arrayscope.tools.profile_montage_workflow import run_profile_montage_workflow

_BACKENDS = ("pyqtgraph", "wgpu")
_COMPONENTS = (
    "payload_build_ms",
    "prepare_ms",
    "backend_apply_ms",
    "backend_pool_growth_ms",
    "backend_executor_initialization_ms",
    "acknowledge_ms",
    "state_publish_ms",
    "geometry_sync_ms",
)


@dataclass(frozen=True)
class PassSummary:
    backend: str
    repeat: int
    pass_kind: str
    required_tiles: int
    chunks: int
    admitted_items: int
    full_min_ms: float | None
    full_p50_ms: float | None
    full_p95_ms: float | None
    full_max_ms: float | None
    full_over_50: int
    steady_p50_ms: float | None
    steady_max_ms: float | None
    structural_pool_growth_ms: float
    structural_executor_initialization_ms: float
    atomic_chunks: int
    callback_work_items_per_s: float | None
    component_p50_ms: dict[str, float]


@dataclass(frozen=True)
class FillSummary:
    backend: str
    repeat: int
    required_tiles: int
    preview_complete_ms: float | None
    target_settle_ms: float | None
    preview_tiles_per_s: float | None
    target_tiles_per_s: float | None
    settlement_status: str
    presented_tiles: int
    exact_payload_tiles: int


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = round(max(0.0, min(1.0, float(fraction))) * (len(ordered) - 1))
    return ordered[index]


def _backend_schedule(backends: tuple[str, ...], repeats: int) -> tuple[tuple[int, str], ...]:
    """Interleave backends and reverse each second round to balance order."""

    rows = []
    for repeat in range(max(1, int(repeats))):
        order = backends if repeat % 2 == 0 else tuple(reversed(backends))
        rows.extend((repeat, backend) for backend in order)
    return tuple(rows)


def _positive_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def _fill_summary(record: dict[str, object], *, backend: str, repeat: int) -> FillSummary:
    required = max(1, int(record.get("requested_tile_count", 0) or 0))
    preview_ms = _positive_float(record.get("first_preview_payload_fill_ms"))
    if preview_ms is None:
        preview_ms = _positive_float(record.get("first_display_payload_fill_ms"))
    target_ms = _positive_float(record.get("required_target_settled_ms"))
    target_duration = (
        None if preview_ms is None or target_ms is None else max(0.0, target_ms - preview_ms)
    )
    return FillSummary(
        backend=backend,
        repeat=int(repeat),
        required_tiles=required,
        preview_complete_ms=preview_ms,
        target_settle_ms=target_ms,
        preview_tiles_per_s=(None if preview_ms is None else 1000.0 * float(required) / preview_ms),
        target_tiles_per_s=(
            None
            if target_duration is None or target_duration <= 0.0
            else 1000.0 * float(required) / target_duration
        ),
        settlement_status=(
            "censored"
            if target_ms is None
            else "within-5s"
            if bool(record.get("interaction_settle_within_budget", False))
            else "late"
        ),
        presented_tiles=int(record.get("active_presented_tile_count", 0) or 0),
        exact_payload_tiles=int(record.get("final_exact_payload_count", 0) or 0),
    )


def _pass_summary(
    events: list[dict[str, object]],
    *,
    backend: str,
    repeat: int,
    pass_kind: str,
    required_tiles: int,
) -> PassSummary:
    rows = [event for event in events if event.get("pass_kind") == pass_kind]
    elapsed = [float(event.get("elapsed_ms", 0.0) or 0.0) for event in rows]
    pool = [float(event.get("backend_pool_growth_ms", 0.0) or 0.0) for event in rows]
    initialization = [
        float(event.get("backend_executor_initialization_ms", 0.0) or 0.0) for event in rows
    ]
    steady = [
        max(0.0, total - pool_ms - initialization_ms)
        for total, pool_ms, initialization_ms in zip(elapsed, pool, initialization, strict=True)
    ]
    admitted = sum(int(event.get("pass_chunk_items", 0) or 0) for event in rows)
    callback_ms = sum(elapsed)
    return PassSummary(
        backend=backend,
        repeat=int(repeat),
        pass_kind=pass_kind,
        required_tiles=max(1, int(required_tiles)),
        chunks=len(rows),
        admitted_items=admitted,
        full_min_ms=min(elapsed) if elapsed else None,
        full_p50_ms=median(elapsed) if elapsed else None,
        full_p95_ms=_percentile(elapsed, 0.95),
        full_max_ms=max(elapsed) if elapsed else None,
        full_over_50=sum(value > 50.0 for value in elapsed),
        steady_p50_ms=median(steady) if steady else None,
        steady_max_ms=max(steady) if steady else None,
        structural_pool_growth_ms=sum(pool),
        structural_executor_initialization_ms=sum(initialization),
        atomic_chunks=sum(bool(event.get("pass_completed_atomically", False)) for event in rows),
        callback_work_items_per_s=(
            None if callback_ms <= 0.0 else 1000.0 * float(admitted) / callback_ms
        ),
        component_p50_ms={
            component: median(float(event.get(component, 0.0) or 0.0) for event in rows)
            for component in _COMPONENTS
        }
        if rows
        else {},
    )


def run_probe(
    *,
    backends: tuple[str, ...],
    repeats: int,
    tiles: int,
    tile_size: int,
    timeout_s: float,
) -> tuple[list[FillSummary], list[PassSummary]]:
    configure_trace(None, ring_events=65_536)
    fills: list[FillSummary] = []
    passes: list[PassSummary] = []
    for repeat, backend in _backend_schedule(backends, repeats):
        trace_start = max(
            (int(event.get("sequence", 0) or 0) for event in TRACE.snapshot()),
            default=0,
        )
        records = run_profile_montage_workflow(
            backend=backend,
            wgpu_present_method="screen",
            texture_codec="off",
            enable_coarse_rung=True,
            synthetic_scene="geometry",
            synthetic_shape=(int(tile_size), int(tile_size), int(tiles)),
            max_tiles=int(tiles),
            timeout_s=bounded_interaction_settle_timeout_s(None),
            cold_fill_observation_timeout_s=float(timeout_s),
            stages=("raw_full_tiled_montage",),
            repeat_index=int(repeat),
        )
        record = next(row for row in records if row.get("phase") == "raw_full_tiled_montage")
        fill = _fill_summary(record, backend=backend, repeat=repeat)
        fills.append(fill)
        events = [
            event
            for event in TRACE.snapshot()
            if event.get("kind") == "commit_batch"
            and event.get("phase") == "backend_complete"
            and int(event.get("sequence", 0) or 0) > trace_start
        ]
        passes.extend(
            _pass_summary(
                events,
                backend=backend,
                repeat=repeat,
                pass_kind=pass_kind,
                required_tiles=fill.required_tiles,
            )
            for pass_kind in ("preview", "target")
        )
    return fills, passes


def _fmt(value: float | None, *, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_markdown(fills: list[FillSummary], passes: list[PassSummary]) -> None:
    print(
        "| backend | run | preview complete ms | preview tiles/s | target settle ms | "
        "target tiles/s | final presented/exact | status |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in fills:
        print(
            f"| {row.backend} | {row.repeat} | {_fmt(row.preview_complete_ms)} | "
            f"{_fmt(row.preview_tiles_per_s)} | {_fmt(row.target_settle_ms)} | "
            f"{_fmt(row.target_tiles_per_s)} | {row.presented_tiles}/"
            f"{row.exact_payload_tiles} | {row.settlement_status} |"
        )
    print()
    print(
        "| backend | run | pass | chunks | items | full min/p50/p95/max ms | >50 | "
        "steady p50/max ms | pool/init ms | atomic | callback items/s |"
    )
    print("|---|---:|---|---:|---:|---|---:|---|---|---:|---:|")
    for row in passes:
        distribution = "/".join(
            _fmt(value)
            for value in (
                row.full_min_ms,
                row.full_p50_ms,
                row.full_p95_ms,
                row.full_max_ms,
            )
        )
        print(
            f"| {row.backend} | {row.repeat} | {row.pass_kind} | {row.chunks} | "
            f"{row.admitted_items} | {distribution} | {row.full_over_50} | "
            f"{_fmt(row.steady_p50_ms)}/{_fmt(row.steady_max_ms)} | "
            f"{_fmt(row.structural_pool_growth_ms)}/"
            f"{_fmt(row.structural_executor_initialization_ms)} | "
            f"{row.atomic_chunks} | {_fmt(row.callback_work_items_per_s)} |"
        )
    print()
    print("Component medians (ms):")
    for row in passes:
        components = ", ".join(
            f"{name.removesuffix('_ms')}={value:.2f}"
            for name, value in row.component_p50_ms.items()
        )
        print(f"- {row.backend} run {row.repeat} {row.pass_kind}: {components}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=(*_BACKENDS, "all"), default="all")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--tiles", type=int, default=272)
    parser.add_argument("--tile-size", type=int, default=336)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser


def main(argv: tuple[str, ...] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_argv = tuple(argv if argv is not None else sys.argv[1:])
    if not is_headless_display():
        return run_in_headless_display(
            (sys.executable, "-m", "arrayscope.tools.render_pass_governor_probe", *source_argv)
        )
    backends = _BACKENDS if args.backend == "all" else (str(args.backend),)
    fills, passes = run_probe(
        backends=backends,
        repeats=max(1, int(args.repeat)),
        tiles=max(1, int(args.tiles)),
        tile_size=max(1, int(args.tile_size)),
        timeout_s=max(1.0, float(args.timeout)),
    )
    if args.format == "json":
        print(
            json.dumps(
                {
                    "fills": [asdict(row) for row in fills],
                    "passes": [asdict(row) for row in passes],
                },
                sort_keys=True,
            )
        )
    else:
        _print_markdown(fills, passes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
