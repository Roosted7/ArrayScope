import json
from pathlib import Path

import pytest

from arrayscope.tools.progressive_render_oracle import (
    check_progressive_render_snapshots,
    format_progressive_render_summary,
    format_progressive_render_violations,
    main,
    replay_progressive_render_trace,
)


def _snapshot(
    levels,
    *,
    session_id=1,
    target_level=2,
    preview_level=5,
    presented=1,
    covered=None,
    population=10,
    level_updates=0,
):
    if covered is None:
        covered = max(presented, population)
    return {
        "event": "snapshot",
        "diagnostics": {
            "montage": {
                "session_id": session_id,
                "tile_lod_resident_tile_levels": [[level, count] for level, count in levels],
                "tile_lod_applied_level": levels[0][0] if levels else 0,
                "tile_lod_desired_factor": 2**target_level,
                "tile_lod_ladder_floor_level": preview_level,
                "presented_tiles": presented,
                "visible_tiles": population,
                "semantic_evidence_blocking_reason": "ready",
                "semantic_evidence_covered_source_count": covered,
                "semantic_evidence_target_population": population,
            },
            "montage_timing": {"tile_layer_level_updates": level_updates},
        },
    }


@pytest.mark.parametrize(
    "snapshots",
    [
        [_snapshot([(5, 10)])],
        [_snapshot([(5, 10)]), _snapshot([(2, 10)])],
        [
            _snapshot([(5, 10)]),
            _snapshot([(2, 4), (5, 6)]),
            _snapshot([(2, 10)]),
        ],
    ],
    ids=["one-level", "preview-then-target", "preview-target-coexist"],
)
def test_legal_progressive_sequences_pass(snapshots):
    assert check_progressive_render_snapshots(snapshots) == ()


def test_one_target_level_passes_when_round_has_no_preview():
    assert check_progressive_render_snapshots([_snapshot([(2, 10)], preview_level=-1)]) == ()


@pytest.mark.parametrize(
    ("snapshots", "expected_index", "expected_levels"),
    [
        ([_snapshot([(0, 11), (2, 39), (5, 222)])], 1, (0, 2, 5)),
        (
            [
                _snapshot(
                    [(0, 15), (1, 31), (4, 2)],
                    target_level=1,
                    preview_level=4,
                )
            ],
            1,
            (0, 1, 4),
        ),
        (
            [
                _snapshot([(2, 4), (5, 6)]),
                _snapshot([(0, 1), (2, 4), (5, 5)]),
            ],
            2,
            (0, 2, 5),
        ),
        ([_snapshot([(0, 1), (2, 9)])], 1, (0, 2)),
    ],
    ids=[
        "fft-three-levels",
        "scalar-three-levels",
        "retained-third-level",
        "wrong-two-level-pair",
    ],
)
def test_illegal_level_sequences_fail_r1(snapshots, expected_index, expected_levels):
    violations = check_progressive_render_snapshots(snapshots)

    assert [(item.rule, item.snapshot_index, item.levels) for item in violations] == [
        ("R1", expected_index, expected_levels)
    ]


def test_frozen_evidence_during_growing_fill_fails_once():
    snapshots = [
        _snapshot([(2, 272)], presented=1, covered=32, population=272),
        _snapshot([(2, 272)], presented=100, covered=32, population=272),
        _snapshot([(2, 272)], presented=272, covered=32, population=272),
        _snapshot(
            [(2, 272)],
            presented=272,
            covered=272,
            population=272,
            level_updates=240,
        ),
    ]

    violations = check_progressive_render_snapshots(snapshots)

    assert len(violations) == 1
    assert violations[0].rule == "R3"
    assert violations[0].snapshot_index == 3
    assert "32/272" in violations[0].description
    assert "1→272" in violations[0].description


def test_replay_and_formatters_report_snapshot_index_and_markdown_table(tmp_path):
    path = tmp_path / "three-levels.jsonl"
    records = [{"event": "start"}, _snapshot([(0, 11), (2, 39), (5, 222)])]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    result = replay_progressive_render_trace(path)

    assert not result.passed
    assert "FAIL R1 snapshot 1 levels={0, 2, 5}" in format_progressive_render_violations(result)
    summary = format_progressive_render_summary([result])
    assert "| Trace | Snapshots | R1 | R3 | Verdict |" in summary
    assert "| `three-levels.jsonl` | 1 | 1 | 0 | FAIL |" in summary


def test_summary_cli_returns_nonzero_for_contract_violation(tmp_path, capsys):
    path = tmp_path / "three-levels.jsonl"
    path.write_text(
        json.dumps(_snapshot([(0, 11), (2, 39), (5, 222)])) + "\n",
        encoding="utf-8",
    )

    exit_code = main(["--summary", str(path)])

    assert exit_code == 1
    assert "| `three-levels.jsonl` | 1 | 1 | 0 | FAIL |" in capsys.readouterr().out


def test_replay_rejects_empty_trace(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text('{"event": "start"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no snapshot events"):
        replay_progressive_render_trace(path)


def test_recorded_progressive_render_traces_are_contract_clean():
    roots = (
        Path("tests/fixtures/progressive-render-contract"),
        Path("tests/artifacts/progressive-render-contract"),
    )
    paths = sorted(path for root in roots for path in root.glob("*.jsonl"))
    if not paths:
        pytest.skip("no recorded progressive-render JSONL fixtures are present")

    results = tuple(replay_progressive_render_trace(path) for path in paths)
    failures = tuple(result for result in results if not result.passed)

    assert not failures, "\n\n".join(
        format_progressive_render_violations(result) for result in failures
    )
