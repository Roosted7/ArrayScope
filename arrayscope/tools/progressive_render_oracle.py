"""Replay diagnostics JSONL snapshots against the progressive render contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProgressiveRenderViolation:
    """One contract violation at a one-based diagnostics snapshot index."""

    rule: str
    snapshot_index: int
    levels: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class ProgressiveRenderOracleResult:
    """Progressive render verdict for one diagnostics trace."""

    path: str
    snapshot_count: int
    violations: tuple[ProgressiveRenderViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class _Snapshot:
    index: int
    session_id: int | None
    levels: tuple[int, ...]
    target_level: int
    preview_level: int | None
    presented_tiles: int
    evidence_covered: int
    evidence_population: int
    level_updates: int


def replay_progressive_render_trace(path: str | Path) -> ProgressiveRenderOracleResult:
    """Read one diagnostics JSONL trace and evaluate its snapshot sequence."""

    path = Path(path)
    snapshots = tuple(_snapshots_from_records(_read_jsonl(path)))
    if not snapshots:
        raise ValueError(f"diagnostics trace has no snapshot events: {path}")
    return ProgressiveRenderOracleResult(
        path=str(path),
        snapshot_count=len(snapshots),
        violations=check_progressive_render_snapshots(snapshots),
    )


def check_progressive_render_snapshots(
    snapshots: Sequence[_Snapshot] | Sequence[dict[str, object]],
) -> tuple[ProgressiveRenderViolation, ...]:
    """Evaluate an already-decoded snapshot sequence.

    Dictionaries may be complete diagnostics records or their ``diagnostics``
    payloads. This form keeps synthetic regression tests independent of JSONL
    artifacts.
    """

    normalized = tuple(
        snapshot
        if isinstance(snapshot, _Snapshot)
        else _snapshot_from_mapping(snapshot, index=index)
        for index, snapshot in enumerate(snapshots, start=1)
    )
    violations = [violation for snapshot in normalized if (violation := _check_r1(snapshot))]
    violations.extend(_check_r3(normalized))
    return tuple(sorted(violations, key=lambda item: (item.snapshot_index, item.rule)))


def format_progressive_render_violations(result: ProgressiveRenderOracleResult) -> str:
    """Format the detailed, line-oriented verdict for one trace."""

    lines = [f"{Path(result.path).name}: {result.snapshot_count} snapshots"]
    if result.passed:
        lines.append("PASS: no R1/R3 violations")
    for violation in result.violations:
        levels = _format_levels(violation.levels)
        lines.append(
            f"FAIL {violation.rule} snapshot {violation.snapshot_index} "
            f"levels={levels}: {violation.description}"
        )
    return "\n".join(lines)


def format_progressive_render_summary(
    results: Sequence[ProgressiveRenderOracleResult],
) -> str:
    """Format a compact Markdown table for review evidence."""

    lines = [
        "| Trace | Snapshots | R1 | R3 | Verdict |",
        "|---|---:|---:|---:|:---:|",
    ]
    for result in results:
        r1_count = sum(violation.rule == "R1" for violation in result.violations)
        r3_count = sum(violation.rule == "R3" for violation in result.violations)
        verdict = "PASS" if result.passed else "FAIL"
        name = Path(result.path).name.replace("|", "\\|")
        lines.append(
            f"| `{name}` | {result.snapshot_count} | {r1_count} | {r3_count} | {verdict} |"
        )
    return "\n".join(lines)


def _check_r1(snapshot: _Snapshot) -> ProgressiveRenderViolation | None:
    levels = set(snapshot.levels)
    if len(levels) > 2:
        return ProgressiveRenderViolation(
            rule="R1",
            snapshot_index=snapshot.index,
            levels=snapshot.levels,
            description="more than two resident/presented LOD levels are on screen",
        )
    if len(levels) != 2:
        return None
    if snapshot.preview_level is None:
        return ProgressiveRenderViolation(
            rule="R1",
            snapshot_index=snapshot.index,
            levels=snapshot.levels,
            description="two on-screen levels exist in a round with no preview level",
        )
    expected = {snapshot.target_level, snapshot.preview_level}
    if levels != expected:
        return ProgressiveRenderViolation(
            rule="R1",
            snapshot_index=snapshot.index,
            levels=snapshot.levels,
            description=(
                "two on-screen levels are not the round preview/target pair "
                f"{_format_levels(tuple(sorted(expected)))}"
            ),
        )
    return None


def _check_r3(snapshots: Sequence[_Snapshot]) -> tuple[ProgressiveRenderViolation, ...]:
    violations: list[ProgressiveRenderViolation] = []
    run: list[_Snapshot] = []

    def finish_run() -> None:
        if len(run) < 2:
            return
        first = run[0]
        maximum = max(run, key=lambda snapshot: snapshot.presented_tiles)
        if (
            first.evidence_population <= 0
            or first.evidence_covered >= first.evidence_population
            or maximum.presented_tiles <= first.presented_tiles
            or maximum.presented_tiles <= first.evidence_covered
        ):
            return
        updates = sum(snapshot.level_updates for snapshot in run[1:])
        violations.append(
            ProgressiveRenderViolation(
                rule="R3",
                snapshot_index=maximum.index,
                levels=maximum.levels,
                description=(
                    "levels evidence stayed frozen at "
                    f"{first.evidence_covered}/{first.evidence_population} while "
                    f"presented tiles grew {first.presented_tiles}"
                    f"→{maximum.presented_tiles} ({updates} tile-layer level updates)"
                ),
            )
        )

    for snapshot in snapshots:
        same_run = bool(run) and (
            snapshot.session_id == run[-1].session_id
            and snapshot.evidence_covered == run[-1].evidence_covered
            and snapshot.evidence_population == run[-1].evidence_population
            and snapshot.presented_tiles >= run[-1].presented_tiles
        )
        if not same_run:
            finish_run()
            run = []
        run.append(snapshot)
    finish_run()
    return tuple(violations)


def _snapshots_from_records(records: Iterable[dict[str, object]]) -> Iterable[_Snapshot]:
    index = 0
    for record in records:
        if record.get("event") != "snapshot":
            continue
        index += 1
        yield _snapshot_from_mapping(record, index=index)


def _snapshot_from_mapping(mapping: dict[str, object], *, index: int) -> _Snapshot:
    diagnostics = mapping.get("diagnostics", mapping)
    if not isinstance(diagnostics, dict):
        raise ValueError(f"snapshot {index} diagnostics payload is not an object")
    montage = diagnostics.get("montage", {})
    timing = diagnostics.get("montage_timing", {})
    if not isinstance(montage, dict):
        raise ValueError(f"snapshot {index} montage diagnostics is not an object")
    if not isinstance(timing, dict):
        raise ValueError(f"snapshot {index} montage timing diagnostics is not an object")

    desired_factor = _required_int(montage, "tile_lod_desired_factor", index=index)
    if desired_factor <= 0 or desired_factor & (desired_factor - 1):
        raise ValueError(
            f"snapshot {index} tile_lod_desired_factor must be a positive power of two"
        )
    preview_level_value = _required_int(montage, "tile_lod_ladder_floor_level", index=index)
    preview_level = None if preview_level_value < 0 else preview_level_value
    levels = _resident_levels(montage.get("tile_lod_resident_tile_levels"), snapshot_index=index)
    return _Snapshot(
        index=index,
        session_id=_optional_int(montage.get("session_id")),
        levels=levels,
        target_level=desired_factor.bit_length() - 1,
        preview_level=preview_level,
        presented_tiles=_required_int(montage, "presented_tiles", index=index),
        evidence_covered=_required_int(
            montage, "semantic_evidence_covered_source_count", index=index
        ),
        evidence_population=_required_int(
            montage, "semantic_evidence_target_population", index=index
        ),
        level_updates=_optional_int(timing.get("tile_layer_level_updates")) or 0,
    )


def _resident_levels(value: object, *, snapshot_index: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"snapshot {snapshot_index} tile_lod_resident_tile_levels is not a sequence"
        )
    levels: set[int] = set()
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"snapshot {snapshot_index} has an invalid resident level/count row")
        level = _optional_int(row[0])
        count = _optional_int(row[1])
        if level is None or count is None or level < 0 or count < 0:
            raise ValueError(f"snapshot {snapshot_index} has an invalid resident level/count value")
        if count:
            levels.add(level)
    return tuple(sorted(levels))


def _required_int(mapping: dict[str, object], field: str, *, index: int) -> int:
    value = _optional_int(mapping.get(field))
    if value is None:
        raise ValueError(f"snapshot {index} has no integer {field}")
    return value


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_levels(levels: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(level) for level in levels) + "}"


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid diagnostics JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"diagnostics JSONL record at {path}:{line_number} is not an object"
                )
            yield value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="diagnostics JSONL trace")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="emit one Markdown verdict table instead of detailed violations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        results = tuple(replay_progressive_render_trace(path) for path in args.paths)
    except (OSError, ValueError) as exc:
        _parser().error(str(exc))
    if args.summary:
        print(format_progressive_render_summary(results))
    else:
        print("\n\n".join(format_progressive_render_violations(result) for result in results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
