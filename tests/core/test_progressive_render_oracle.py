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
    uploads=None,
):
    if covered is None:
        covered = max(presented, population)
    montage = {
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
    }
    if uploads is not None:
        montage["wgpu_uploads_by_level"] = [
            {"level": level, "uploads": count, "representation": "scalar_r32f", "bytes": count * 16}
            for level, count in sorted(uploads.items())
        ]
    return {
        "event": "snapshot",
        "diagnostics": {
            "montage": montage,
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


def test_empty_synthetic_sequence_fails_closed():
    with pytest.raises(ValueError, match="sequence is empty"):
        check_progressive_render_snapshots([])


@pytest.mark.parametrize(
    "snapshots",
    [
        [
            _snapshot([(0, 11), (2, 39), (5, 222)], uploads={0: 4, 2: 10, 5: 30}),
            _snapshot([(0, 11), (2, 60), (5, 201)], uploads={0: 4, 2: 40, 5: 30}),
        ],
        [
            _snapshot([(0, 15), (1, 31), (4, 2)], target_level=1, preview_level=4, uploads={0: 9}),
            _snapshot(
                [(0, 15), (1, 35)], target_level=1, preview_level=4, uploads={0: 9, 1: 20, 4: 6}
            ),
        ],
        [
            _snapshot([(6, 8), (2, 2)], uploads={2: 2}),
            _snapshot([(6, 4), (5, 4), (2, 2)], uploads={2: 2, 5: 12}),
        ],
    ],
    ids=[
        "retained-finer-tiles-are-free-reuse",
        "three-visible-levels-but-only-floor-production",
        "coarse-free-reuse-rung-before-preview",
    ],
)
def test_visible_level_mixtures_are_legal_when_production_stays_on_the_floors(snapshots):
    """R1 bounds production, not residency.

    Every sequence here shows three or more distinct levels at once, which the
    contract explicitly permits: a tile retained finer than the target floor,
    or a coarse tile displayed for free before the preview pass, both cost
    nothing to show and must never be re-produced. Only uploads outside the
    round's two floors are violations.
    """

    assert check_progressive_render_snapshots(snapshots) == ()


@pytest.mark.parametrize(
    ("snapshots", "expected_index", "expected_fragment"),
    [
        (
            [
                _snapshot([(1, 44)], target_level=1, preview_level=4, uploads={1: 44}),
                _snapshot(
                    [(0, 6), (1, 44)], target_level=1, preview_level=4, uploads={0: 176, 1: 44}
                ),
            ],
            2,
            "level 0: +176",
        ),
        (
            [
                _snapshot([(5, 10)], uploads={5: 10}),
                _snapshot([(3, 5), (5, 5)], uploads={3: 5, 5: 10}),
            ],
            2,
            "third rung",
        ),
    ],
    ids=["produced-finer-than-target-floor", "produced-third-rung-between-floors"],
)
def test_production_outside_the_round_floors_fails_r1(snapshots, expected_index, expected_fragment):
    violations = check_progressive_render_snapshots(snapshots)

    assert [(item.rule, item.snapshot_index) for item in violations] == [("R1", expected_index)]
    assert expected_fragment in violations[0].description


def test_r1_is_not_checked_without_per_level_upload_counters(tmp_path):
    """A clean R1 column on a PyQtGraph trace means "not checked", not "passed".

    Residency alone cannot separate production from a page-cache arrival, so
    the oracle must refuse to guess rather than report a false PASS.
    """

    trace = tmp_path / "no-uploads.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(record)
            for record in [_snapshot([(0, 5), (2, 5)]), _snapshot([(0, 5), (2, 5)])]
        ),
        encoding="utf-8",
    )

    result = replay_progressive_render_trace(trace)

    assert result.r1_verifiable is False
    assert [item for item in result.violations if item.rule == "R1"] == []
    assert "n/a" in format_progressive_render_summary([result])
    assert "R1 UNVERIFIABLE" in format_progressive_render_violations(result)


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


def test_repeated_partial_evidence_freeze_fails_before_presented_exceeds_coverage():
    snapshots = [
        _snapshot([(2, 272)], presented=1, covered=32, population=272),
        _snapshot([(2, 272)], presented=10, covered=32, population=272),
        _snapshot([(2, 272)], presented=20, covered=32, population=272),
    ]

    violations = check_progressive_render_snapshots(snapshots)

    assert len(violations) == 1
    assert violations[0].rule == "R3"
    assert violations[0].snapshot_index == 3
    assert "1→20" in violations[0].description


def test_one_partial_evidence_step_within_covered_cohort_is_not_a_freeze():
    snapshots = [
        _snapshot([(2, 272)], presented=1, covered=32, population=272),
        _snapshot([(2, 272)], presented=20, covered=32, population=272),
    ]

    assert check_progressive_render_snapshots(snapshots) == ()


def test_inactive_evidence_during_growing_fill_fails_r3():
    snapshots = [
        _snapshot([(5, 40)], presented=40, covered=0, population=0),
        _snapshot([(5, 200)], presented=200, covered=0, population=0),
        _snapshot([(2, 72), (5, 200)], presented=272, covered=0, population=0),
        _snapshot([(2, 272)], presented=272, covered=32, population=272),
    ]
    for snapshot in snapshots[:3]:
        snapshot["diagnostics"]["montage"]["semantic_evidence_blocking_reason"] = "inactive"
        snapshot["diagnostics"]["montage"]["visible_tiles"] = 272
    snapshots[-1]["diagnostics"]["montage"]["semantic_evidence_blocking_reason"] = (
        "worker-in-flight"
    )

    violations = check_progressive_render_snapshots(snapshots)

    assert len(violations) == 1
    assert violations[0].rule == "R3"
    assert violations[0].snapshot_index == 3
    assert "inactive" in violations[0].description
    assert "40→272" in violations[0].description


def test_superseded_inactive_round_without_later_evidence_is_not_mislabeled():
    snapshots = [
        _snapshot([(5, 40)], presented=40, covered=0, population=0),
        _snapshot([(5, 200)], presented=200, covered=0, population=0),
        _snapshot([(2, 10)], session_id=2, presented=10, covered=10, population=10),
    ]
    for snapshot in snapshots[:2]:
        snapshot["diagnostics"]["montage"]["semantic_evidence_blocking_reason"] = "inactive"
        snapshot["diagnostics"]["montage"]["visible_tiles"] = 272

    assert check_progressive_render_snapshots(snapshots) == ()


_OFF_FLOOR_PRODUCTION = (
    _snapshot([(2, 39), (5, 222)], uploads={2: 39, 5: 222}),
    _snapshot([(0, 11), (2, 39), (5, 222)], uploads={0: 11, 2: 39, 5: 222}),
)


def test_replay_and_formatters_report_snapshot_index_and_markdown_table(tmp_path):
    path = tmp_path / "off-floor-production.jsonl"
    records = [{"event": "start"}, *_OFF_FLOOR_PRODUCTION]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    result = replay_progressive_render_trace(path)

    assert not result.passed
    assert result.r1_verifiable is True
    assert (
        "FAIL R1 snapshot 2 levels={0, 2, 5} counts={0:11, 2:39, 5:222}"
        in format_progressive_render_violations(result)
    )
    summary = format_progressive_render_summary([result])
    assert "| Trace | Snapshots | R1 | R3 | Verdict |" in summary
    assert "| `off-floor-production.jsonl` | 2 | 1 | 0 | FAIL |" in summary


def test_summary_cli_returns_nonzero_for_contract_violation(tmp_path, capsys):
    path = tmp_path / "off-floor-production.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in _OFF_FLOOR_PRODUCTION),
        encoding="utf-8",
    )

    exit_code = main(["--summary", str(path)])

    assert exit_code == 1
    assert "| `off-floor-production.jsonl` | 2 | 1 | 0 | FAIL |" in capsys.readouterr().out


def test_replay_rejects_empty_trace(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text('{"event": "start"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no snapshot events"):
        replay_progressive_render_trace(path)


def test_replay_rejects_malformed_required_snapshot_field(tmp_path):
    path = tmp_path / "malformed.jsonl"
    snapshot = _snapshot([(2, 1)])
    snapshot["diagnostics"]["montage"]["semantic_evidence_blocking_reason"] = None
    path.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="semantic_evidence_blocking_reason"):
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
