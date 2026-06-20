"""Pure helpers for summarizing ArrayScope diagnostics JSONL traces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from statistics import median
from typing import Iterable


_TIMING_PATHS = (
    ("render_timing", "last_render_sync_ms"),
    ("render_timing", "last_control_sync_ms"),
    ("render_timing", "last_planning_ms"),
    ("render_timing", "last_worker_queue_wait_ms"),
    ("render_timing", "last_evaluation_ms"),
    ("render_timing", "last_display_commit_ms"),
    ("render_timing", "last_set_image_ms"),
    ("render_timing", "last_levels_histogram_ms"),
    ("montage_timing", "last_viewport_plan_ms"),
    ("montage_timing", "last_cache_resolve_ms"),
    ("montage_timing", "last_stage_plan_ms"),
    ("montage_timing", "last_session_setup_ms"),
    ("montage_timing", "last_initial_commit_ms"),
    ("montage_timing", "last_tile_cache_lookup_ms"),
    ("montage_timing", "last_level_stats_ms"),
    ("montage_timing", "last_tile_payload_build_ms"),
    ("montage_timing", "last_canvas_commit_ms"),
    ("montage_timing", "last_visible_upload_ms"),
    ("montage_timing", "last_rgb_window_ms"),
    ("montage_timing", "last_tile_layer_upload_ms"),
)


@dataclass(frozen=True)
class TraceStall:
    sequence: int
    gap_ms: float
    recorded_at: str
    session_id: int | None
    loaded_tiles: int
    pending_tiles: int
    changed_timings_ms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class DiagnosticsTraceSummary:
    path: str
    backend: str
    snapshot_count: int
    duration_ms: float
    expected_interval_ms: int
    median_gap_ms: float
    max_gap_ms: float
    stalls: tuple[TraceStall, ...]
    maximum_timings_ms: tuple[tuple[str, float], ...]


def summarize_diagnostics_trace(
    path: str | Path,
    *,
    stall_threshold_ms: float | None = None,
) -> DiagnosticsTraceSummary:
    """Read a diagnostics JSONL file and summarize event-loop sampling stalls.

    Timing fields in the log are cumulative "last observation" values.  A
    stall therefore records only timing fields whose value changed at the
    snapshot following the gap; unchanged values are not presented as causes.
    """

    path = Path(path)
    records = tuple(_read_jsonl(path))
    start = next((record for record in records if record.get("event") == "start"), {})
    snapshots = tuple(record for record in records if record.get("event") == "snapshot")
    expected_interval_ms = max(1, int(start.get("interval_ms", 500) or 500))
    threshold = (
        max(1000.0, expected_interval_ms * 3.0)
        if stall_threshold_ms is None
        else max(0.0, float(stall_threshold_ms))
    )
    backend = str(
        start.get("config", {}).get("image_rendering_backend_actual")
        or start.get("config", {}).get("image_rendering_backend_selected")
        or "unknown"
    )
    if not snapshots:
        return DiagnosticsTraceSummary(
            path=str(path),
            backend=backend,
            snapshot_count=0,
            duration_ms=0.0,
            expected_interval_ms=expected_interval_ms,
            median_gap_ms=0.0,
            max_gap_ms=0.0,
            stalls=(),
            maximum_timings_ms=(),
        )

    timestamps = tuple(_parse_timestamp(record["recorded_at"]) for record in snapshots)
    gaps_ms = tuple(
        max(0.0, (timestamps[index] - timestamps[index - 1]).total_seconds() * 1000.0)
        for index in range(1, len(timestamps))
    )
    stalls: list[TraceStall] = []
    maxima: dict[str, float] = {}
    previous_diagnostics: dict[str, object] = {}
    for index, record in enumerate(snapshots):
        diagnostics = dict(record.get("diagnostics", {}) or {})
        current_timings = _timing_values(diagnostics)
        for name, value in current_timings.items():
            maxima[name] = max(float(value), float(maxima.get(name, value)))
        if index > 0:
            gap_ms = gaps_ms[index - 1]
            if gap_ms >= threshold:
                previous_timings = _timing_values(previous_diagnostics)
                changed = tuple(
                    sorted(
                        (
                            (name, value)
                            for name, value in current_timings.items()
                            if previous_timings.get(name) != value
                        ),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                )
                montage = dict(diagnostics.get("montage", {}) or {})
                stalls.append(
                    TraceStall(
                        sequence=int(record.get("sequence", index + 1)),
                        gap_ms=gap_ms,
                        recorded_at=str(record.get("recorded_at", "")),
                        session_id=_optional_int(montage.get("session_id")),
                        loaded_tiles=int(montage.get("loaded_tiles", 0) or 0),
                        pending_tiles=int(montage.get("pending_tiles", 0) or 0),
                        changed_timings_ms=changed,
                    )
                )
        previous_diagnostics = diagnostics

    maximum_timings = tuple(sorted(maxima.items(), key=lambda item: item[1], reverse=True))
    duration_ms = max(0.0, (timestamps[-1] - timestamps[0]).total_seconds() * 1000.0)
    return DiagnosticsTraceSummary(
        path=str(path),
        backend=backend,
        snapshot_count=len(snapshots),
        duration_ms=duration_ms,
        expected_interval_ms=expected_interval_ms,
        median_gap_ms=0.0 if not gaps_ms else float(median(gaps_ms)),
        max_gap_ms=0.0 if not gaps_ms else float(max(gaps_ms)),
        stalls=tuple(stalls),
        maximum_timings_ms=maximum_timings,
    )


def format_trace_summary_markdown(summary: DiagnosticsTraceSummary, *, timing_limit: int = 8) -> str:
    """Format a compact, review-friendly Markdown trace summary."""

    lines = [
        f"## {summary.backend}",
        "",
        f"- File: `{Path(summary.path).name}`",
        f"- Snapshots: {summary.snapshot_count}",
        f"- Trace duration: {summary.duration_ms / 1000.0:.3f} s",
        f"- Expected / median sample interval: {summary.expected_interval_ms} / {summary.median_gap_ms:.1f} ms",
        f"- Maximum sample gap: {summary.max_gap_ms:.1f} ms",
        f"- Stalls above threshold: {len(summary.stalls)}",
        "",
        "### Largest observed timing values",
        "",
    ]
    for name, value in summary.maximum_timings_ms[: max(0, int(timing_limit))]:
        lines.append(f"- `{name}`: {value:.3f} ms")
    if not summary.maximum_timings_ms:
        lines.append("- none")
    lines.extend(("", "### Sampling stalls", ""))
    if not summary.stalls:
        lines.append("- none")
    for stall in sorted(summary.stalls, key=lambda item: item.gap_ms, reverse=True):
        context = (
            f"sequence {stall.sequence}, gap {stall.gap_ms:.1f} ms, "
            f"session {stall.session_id}, loaded/pending {stall.loaded_tiles}/{stall.pending_tiles}"
        )
        lines.append(f"- {context}")
        for name, value in stall.changed_timings_ms[:5]:
            lines.append(f"  - changed `{name}`: {value:.3f} ms")
        if not stall.changed_timings_ms:
            lines.append("  - no tracked timing changed; the blocking work is still unattributed")
    return "\n".join(lines) + "\n"


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid diagnostics JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"diagnostics JSONL record at {path}:{line_number} is not an object")
            yield value


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid diagnostics timestamp: {value!r}") from exc


def _timing_values(diagnostics: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for section, field in _TIMING_PATHS:
        section_value = diagnostics.get(section, {})
        if not isinstance(section_value, dict):
            continue
        value = section_value.get(field)
        if isinstance(value, (int, float)):
            values[f"{section}.{field}"] = float(value)
    return values


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize ArrayScope diagnostics JSONL traces")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--stall-ms", type=float, default=None, help="override the stall threshold")
    args = parser.parse_args(argv)
    for index, path in enumerate(args.paths):
        if index:
            print()
        print(format_trace_summary_markdown(summarize_diagnostics_trace(path, stall_threshold_ms=args.stall_ms)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
