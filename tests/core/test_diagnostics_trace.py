import json

import pytest

from arrayscope.core.diagnostics_trace import format_trace_summary_markdown, summarize_diagnostics_trace


def _write_trace(path):
    records = [
        {
            "event": "start",
            "interval_ms": 500,
            "config": {"image_rendering_backend_actual": "vispy"},
        },
        {
            "event": "snapshot",
            "sequence": 1,
            "recorded_at": "2026-06-20T10:00:00+00:00",
            "diagnostics": {
                "render_timing": {"last_render_sync_ms": 2.0},
                "montage_timing": {"last_canvas_commit_ms": 1.0},
                "montage": {"session_id": 1, "loaded_tiles": 1, "pending_tiles": 3},
            },
        },
        {
            "event": "snapshot",
            "sequence": 2,
            "recorded_at": "2026-06-20T10:00:00.500000+00:00",
            "diagnostics": {
                "render_timing": {"last_render_sync_ms": 2.0},
                "montage_timing": {"last_canvas_commit_ms": 1.0},
                "montage": {"session_id": 1, "loaded_tiles": 2, "pending_tiles": 2},
            },
        },
        {
            "event": "snapshot",
            "sequence": 3,
            "recorded_at": "2026-06-20T10:00:03+00:00",
            "diagnostics": {
                "render_timing": {"last_render_sync_ms": 2100.0},
                "montage_timing": {"last_canvas_commit_ms": 4.0},
                "montage": {"session_id": 2, "loaded_tiles": 4, "pending_tiles": 0},
            },
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_trace_summary_detects_sampling_stall_and_changed_timing(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_trace(path)

    summary = summarize_diagnostics_trace(path)

    assert summary.backend == "vispy"
    assert summary.snapshot_count == 3
    assert summary.median_gap_ms == pytest.approx(1500.0)
    assert summary.max_gap_ms == pytest.approx(2500.0)
    assert len(summary.stalls) == 1
    stall = summary.stalls[0]
    assert stall.sequence == 3
    assert stall.session_id == 2
    assert stall.changed_timings_ms[0] == ("render_timing.last_render_sync_ms", 2100.0)


def test_trace_summary_markdown_marks_unattributed_stall(tmp_path):
    path = tmp_path / "trace.jsonl"
    _write_trace(path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["diagnostics"]["render_timing"]["last_render_sync_ms"] = 2.0
    records[-1]["diagnostics"]["montage_timing"]["last_canvas_commit_ms"] = 1.0
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    text = format_trace_summary_markdown(summarize_diagnostics_trace(path))

    assert "## vispy" in text
    assert "no tracked timing changed" in text


def test_trace_summary_rejects_invalid_jsonl(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n")

    with pytest.raises(ValueError, match="invalid diagnostics JSONL"):
        summarize_diagnostics_trace(path)
