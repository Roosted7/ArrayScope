"""Replay diagnostics JSONL snapshots against the progressive render contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path


@dataclass(frozen=True)
class ProgressiveRenderViolation:
    """One contract violation at a one-based diagnostics snapshot index."""

    rule: str
    snapshot_index: int
    levels: tuple[int, ...]
    level_counts: tuple[tuple[int, int], ...]
    description: str


@dataclass(frozen=True)
class ProgressiveRenderOracleResult:
    """Progressive render verdict for one diagnostics trace."""

    path: str
    snapshot_count: int
    violations: tuple[ProgressiveRenderViolation, ...]
    #: False when the trace carries no per-level upload counters, so R1 could
    #: not be evaluated at all. A clean R1 column on such a trace means
    #: "not checked", never "satisfied".
    r1_verifiable: bool = True

    @property
    def passed(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class _Snapshot:
    index: int
    # Diagnostic context only; session identity is never a round boundary.
    session_id: int | None
    round_id: str
    levels: tuple[int, ...]
    level_counts: tuple[tuple[int, int], ...]
    target_level: int
    preview_level: int | None
    presented_tiles: int
    visible_tiles: int
    evidence_reason: str
    evidence_covered: int
    evidence_population: int
    level_updates: int
    #: Cumulative uploads per LOD level. WGPU only -- empty elsewhere, which
    #: makes R1 unverifiable rather than satisfied.
    uploads_by_level: dict[int, int]


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
        r1_verifiable=r1_is_verifiable(snapshots),
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
    if not normalized:
        raise ValueError("progressive render snapshot sequence is empty")
    violations = list(_check_r2b(normalized))
    violations.extend(_check_r1(normalized))
    violations.extend(_check_r3(normalized))
    return tuple(sorted(violations, key=lambda item: (item.snapshot_index, item.rule)))


def format_progressive_render_violations(result: ProgressiveRenderOracleResult) -> str:
    """Format the detailed, line-oriented verdict for one trace."""

    lines = [f"{Path(result.path).name}: {result.snapshot_count} snapshots"]
    if not result.r1_verifiable:
        lines.append(
            "R1 UNVERIFIABLE: no per-level upload counters in this trace "
            "(production cannot be told from cache arrival)"
        )
    if result.passed:
        # Round attribution and observed floor immutability are now proofs.
        # Upload purpose/duplicates and R3 value containment remain invisible
        # in 500 ms snapshots, so a clean trace is still weaker than the full
        # progressive contract being satisfied.
        lines.append(
            "no R1/R2b/R3 violations detected "
            "(round attribution is proven; production purpose and R3 remain heuristic)"
        )
    for violation in result.violations:
        levels = _format_levels(violation.levels)
        counts = _format_level_counts(violation.level_counts)
        lines.append(
            f"FAIL {violation.rule} snapshot {violation.snapshot_index} "
            f"levels={levels} counts={counts}: {violation.description}"
        )
    return "\n".join(lines)


def format_progressive_render_summary(
    results: Sequence[ProgressiveRenderOracleResult],
) -> str:
    """Format a compact Markdown table for review evidence."""

    lines = [
        "| Trace | Snapshots | R1 | R2b | R3 | Verdict |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for result in results:
        r1_count = sum(violation.rule == "R1" for violation in result.violations)
        r2b_count = sum(violation.rule == "R2b" for violation in result.violations)
        r3_count = sum(violation.rule == "R3" for violation in result.violations)
        r1_cell = str(r1_count) if result.r1_verifiable else "n/a"
        verdict = "PASS" if result.passed else "FAIL"
        name = Path(result.path).name.replace("|", "\\|")
        lines.append(
            f"| `{name}` | {result.snapshot_count} | {r1_cell} | "
            f"{r2b_count} | {r3_count} | {verdict} |"
        )
    if not all(result.r1_verifiable for result in results):
        lines.append("")
        lines.append("`n/a` = no per-level upload counters in the trace; R1 was not checked.")
    return "\n".join(lines)


def _check_r2b(snapshots: Sequence[_Snapshot]) -> tuple[ProgressiveRenderViolation, ...]:
    """Prove that every observed snapshot of one round carries the same floors."""

    floors_by_round: dict[str, tuple[int, int | None]] = {}
    violations: list[ProgressiveRenderViolation] = []
    for snapshot in snapshots:
        floors = (snapshot.target_level, snapshot.preview_level)
        expected = floors_by_round.setdefault(snapshot.round_id, floors)
        if floors == expected:
            continue
        violations.append(
            ProgressiveRenderViolation(
                rule="R2b",
                snapshot_index=snapshot.index,
                levels=snapshot.levels,
                level_counts=snapshot.level_counts,
                description=(
                    f"round {snapshot.round_id} moved floors from "
                    f"P={expected[1]}, T={expected[0]} to "
                    f"P={snapshot.preview_level}, T={snapshot.target_level}"
                ),
            )
        )
    return tuple(violations)


def _check_r1(snapshots: Sequence[_Snapshot]) -> tuple[ProgressiveRenderViolation, ...]:
    """Flag PRODUCTION at a level outside the round's two floors.

    R1 bounds production, not residency. The instantaneous visible level set is
    deliberately unconstrained: a tile retained finer than the target floor, or
    a coarse tile shown for free before the preview pass, are both legal and
    common (R2). Asserting "at most two levels on screen" would reject the
    free-reuse rung that the contract requires -- across the recorded traces it
    fires on 142 snapshots, 114 of which are legal reuse.

    Production is only observable through the cumulative per-level upload
    counters, which exist on WGPU alone. Growth in a resident level count
    cannot serve as a substitute: a tile arriving from the pyramid or page
    cache raises the count without producing anything. So on backends with no
    upload counters this rule is unverifiable from a snapshot trace, and is
    reported as such rather than guessed at.
    """

    violations: list[ProgressiveRenderViolation] = []
    for previous, current in pairwise(snapshots):
        # This explicit derived identity upgrades boundary attribution from
        # the former session+floors proxy to proof. Upload purpose is still
        # absent, and duplicate production at a legal floor stays invisible.
        if current.round_id != previous.round_id:
            continue
        if not current.uploads_by_level or not previous.uploads_by_level:
            continue
        floors = {current.target_level}
        if current.preview_level is not None:
            floors.add(current.preview_level)
        produced = {
            level: count - previous.uploads_by_level.get(level, 0)
            for level, count in current.uploads_by_level.items()
            if count > previous.uploads_by_level.get(level, 0)
        }
        offenders = {level: count for level, count in produced.items() if level not in floors}
        if not offenders:
            continue
        detail = ", ".join(f"level {level}: +{count}" for level, count in sorted(offenders.items()))
        finer = [level for level in offenders if level < current.target_level]
        note = (
            " (finer than the target floor -- quality the round never asked for; "
            "speculative native warming lands here too and is not distinguishable "
            "until uploads carry their purpose)"
            if finer
            else " (a third rung between the preview and target floors)"
        )
        violations.append(
            ProgressiveRenderViolation(
                rule="R1",
                snapshot_index=current.index,
                levels=current.levels,
                level_counts=current.level_counts,
                description=(
                    f"uploads produced outside the round floors "
                    f"{_format_levels(tuple(sorted(floors)))}: {detail}{note}"
                ),
            )
        )
    return tuple(violations)


def r1_is_verifiable(snapshots: Sequence[_Snapshot]) -> bool:
    """Whether this trace carries the per-level upload counters R1 needs."""

    return any(snapshot.uploads_by_level for snapshot in snapshots)


def _check_r3(snapshots: Sequence[_Snapshot]) -> tuple[ProgressiveRenderViolation, ...]:
    return (*_check_r3_frozen_partial_evidence(snapshots), *_check_r3_inactive_evidence(snapshots))


def _check_r3_frozen_partial_evidence(
    snapshots: Sequence[_Snapshot],
) -> tuple[ProgressiveRenderViolation, ...]:
    violations: list[ProgressiveRenderViolation] = []
    run: list[_Snapshot] = []

    def finish_run() -> None:
        if len(run) < 2:
            return
        first = run[0]
        maximum = max(run, key=lambda snapshot: snapshot.presented_tiles)
        growth_steps = sum(
            current.presented_tiles > previous.presented_tiles
            for previous, current in pairwise(run)
        )
        if (
            first.evidence_population <= 0
            or first.evidence_covered >= first.evidence_population
            or maximum.presented_tiles <= first.presented_tiles
            or (growth_steps < 2 and maximum.presented_tiles <= first.evidence_covered)
        ):
            return
        violations.append(
            ProgressiveRenderViolation(
                rule="R3",
                snapshot_index=maximum.index,
                levels=maximum.levels,
                level_counts=maximum.level_counts,
                description=(
                    "levels evidence stayed frozen at "
                    f"{first.evidence_covered}/{first.evidence_population} while "
                    f"presented tiles grew {first.presented_tiles}"
                    f"→{maximum.presented_tiles} "
                    f"(snapshot level updates={maximum.level_updates})"
                ),
            )
        )

    for snapshot in snapshots:
        same_run = bool(run) and (
            snapshot.round_id == run[-1].round_id
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


def _check_r3_inactive_evidence(
    snapshots: Sequence[_Snapshot],
) -> tuple[ProgressiveRenderViolation, ...]:
    violations: list[ProgressiveRenderViolation] = []
    run: list[_Snapshot] = []

    def finish_run() -> None:
        if len(run) < 2:
            return
        first = run[0]
        maximum = max(run, key=lambda snapshot: snapshot.presented_tiles)
        evidence_started_later = any(
            snapshot.index > run[-1].index
            and snapshot.round_id == first.round_id
            and snapshot.evidence_population > 0
            for snapshot in snapshots
        )
        if (
            max(snapshot.visible_tiles for snapshot in run) <= 1
            or maximum.presented_tiles <= first.presented_tiles
            or not evidence_started_later
        ):
            return
        violations.append(
            ProgressiveRenderViolation(
                rule="R3",
                snapshot_index=maximum.index,
                levels=maximum.levels,
                level_counts=maximum.level_counts,
                description=(
                    "levels evidence remained inactive while presented tiles grew "
                    f"{first.presented_tiles}→{maximum.presented_tiles} "
                    f"(snapshot level updates={maximum.level_updates})"
                ),
            )
        )

    for snapshot in snapshots:
        same_run = bool(run) and (
            snapshot.round_id == run[-1].round_id
            and snapshot.evidence_reason == "inactive"
            and snapshot.evidence_population == 0
        )
        if snapshot.evidence_reason != "inactive" or snapshot.evidence_population != 0:
            finish_run()
            run = []
            continue
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
    preview_level_value = _required_int(montage, "tile_lod_round_preview_level", index=index)
    preview_level = None if preview_level_value < 0 else preview_level_value
    level_counts = _resident_level_counts(
        montage.get("tile_lod_resident_tile_levels"), snapshot_index=index
    )
    return _Snapshot(
        index=index,
        session_id=_optional_int(montage.get("session_id")),
        round_id=_required_string(montage, "render_round_id", index=index),
        levels=tuple(level for level, count in level_counts if count),
        level_counts=level_counts,
        target_level=_required_nonnegative_int(montage, "tile_lod_round_target_level", index=index),
        preview_level=preview_level,
        presented_tiles=_required_nonnegative_int(montage, "presented_tiles", index=index),
        visible_tiles=_required_nonnegative_int(montage, "visible_tiles", index=index),
        evidence_reason=_required_string(montage, "semantic_evidence_blocking_reason", index=index),
        evidence_covered=_required_nonnegative_int(
            montage, "semantic_evidence_covered_source_count", index=index
        ),
        evidence_population=_required_nonnegative_int(
            montage, "semantic_evidence_target_population", index=index
        ),
        level_updates=_nonnegative_int(
            timing.get("tile_layer_level_updates"),
            field="tile_layer_level_updates",
            index=index,
        ),
        uploads_by_level=_uploads_by_level(
            montage.get("wgpu_uploads_by_level"), snapshot_index=index
        ),
    )


def _uploads_by_level(value: object, *, snapshot_index: int) -> dict[int, int]:
    if value is None:
        return {}
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"snapshot {snapshot_index} wgpu_uploads_by_level is not a sequence")
    uploads: dict[int, int] = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError(f"snapshot {snapshot_index} has an invalid upload-by-level row")
        level = _optional_int(row.get("level"))
        count = _optional_int(row.get("uploads"))
        if level is None or count is None or level < 0 or count < 0:
            raise ValueError(f"snapshot {snapshot_index} has an invalid upload-by-level value")
        # One level can appear once per representation; the round's production
        # at that level is their sum.
        uploads[level] = uploads.get(level, 0) + count
    return uploads


def _resident_level_counts(value: object, *, snapshot_index: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"snapshot {snapshot_index} tile_lod_resident_tile_levels is not a sequence"
        )
    level_counts: dict[int, int] = {}
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"snapshot {snapshot_index} has an invalid resident level/count row")
        level = _optional_int(row[0])
        count = _optional_int(row[1])
        if level is None or count is None or level < 0 or count < 0:
            raise ValueError(f"snapshot {snapshot_index} has an invalid resident level/count value")
        if level in level_counts:
            raise ValueError(f"snapshot {snapshot_index} repeats resident level {level}")
        if count:
            level_counts[level] = count
    return tuple(sorted(level_counts.items()))


def _required_int(mapping: dict[str, object], field: str, *, index: int) -> int:
    value = _optional_int(mapping.get(field))
    if value is None:
        raise ValueError(f"snapshot {index} has no integer {field}")
    return value


def _required_nonnegative_int(mapping: dict[str, object], field: str, *, index: int) -> int:
    return _nonnegative_int(mapping.get(field), field=field, index=index)


def _required_string(mapping: dict[str, object], field: str, *, index: int) -> str:
    value = mapping.get(field)
    if not isinstance(value, str):
        raise ValueError(f"snapshot {index} has no string {field}")
    return value


def _nonnegative_int(value: object, *, field: str, index: int) -> int:
    result = _optional_int(value)
    if result is None or result < 0:
        raise ValueError(f"snapshot {index} has no nonnegative integer {field}")
    return result


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _format_levels(levels: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(level) for level in levels) + "}"


def _format_level_counts(level_counts: tuple[tuple[int, int], ...]) -> str:
    return "{" + ", ".join(f"{level}:{count}" for level, count in level_counts) + "}"


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
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        results = tuple(replay_progressive_render_trace(path) for path in args.paths)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.summary:
        print(format_progressive_render_summary(results))
    else:
        print("\n\n".join(format_progressive_render_violations(result) for result in results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
