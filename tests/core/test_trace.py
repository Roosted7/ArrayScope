import json


def test_trace_bus_writes_flat_jsonl_and_bounds_ring(tmp_path):
    from arrayscope.core.trace import TraceBus

    path = tmp_path / "trace.jsonl"
    bus = TraceBus()
    bus.configure(path, ring_bytes=250)
    for index in range(8):
        bus.emit("kernel_submit", task_seq=index, key=("tile", index))
    snapshot = bus.snapshot()
    bus.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 8
    assert rows[0]["schema_version"] == 1
    assert rows[-1]["sequence"] == 8
    assert all(row["kind"] == "kernel_submit" for row in rows)
    assert 0 < len(snapshot) < len(rows)


def test_trace_latency_matches_task_spans(tmp_path):
    from arrayscope.tools.trace_latency import analyze_trace_latency

    path = tmp_path / "trace.jsonl"
    rows = (
        {"kind": "kernel_submit", "task_seq": 7, "ts_ns": 1_000_000},
        {"kind": "kernel_start", "task_seq": 7, "ts_ns": 3_000_000},
        {"kind": "kernel_finish", "task_seq": 7, "ts_ns": 8_000_000},
        {"kind": "bridge_drain", "elapsed_ms": 1.5, "ts_ns": 9_000_000},
        {"kind": "input", "action": "phase_start", "ts_ns": 10_000_000},
        {"kind": "backend_ack", "tile": 3, "ts_ns": 14_000_000},
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = analyze_trace_latency(path)

    assert summary["kernel_queue_ms"]["p50"] == 2.0
    assert summary["kernel_run_ms"]["p50"] == 5.0
    assert summary["bridge_drain_ms"]["max"] == 1.5
    assert summary["input_to_first_ack_ms"]["p50"] == 4.0


def test_trace_verify_requires_exact_ack_for_final_visible_targets(tmp_path):
    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 3,
            "source_index": 41,
            "target_level": 2,
            "sequence": 1,
        },
        {
            "kind": "backend_ack",
            "tile": 3,
            "source_index": 41,
            "level": 4,
            "quality": "fallback",
            "accepted": True,
            "sequence": 2,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    failed = verify_trace(path)
    assert not failed["ok"]
    assert failed["violations"][0]["tile"] == 3

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "backend_ack",
                    "tile": 3,
                    "source_index": 41,
                    "level": 2,
                    "quality": "exact",
                    "accepted": True,
                    "sequence": 3,
                }
            )
            + "\n"
        )

    passed = verify_trace(path)
    assert passed["ok"]
    assert passed["acknowledged_targets"] == 1
    assert passed["acknowledgement_order"] == (3,)


def test_trace_verify_forgets_targets_released_from_scope(tmp_path):
    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 7,
            "source_index": 5,
            "target_level": 0,
            "sequence": 1,
        },
        {
            "kind": "lifecycle",
            "edge": "target_released",
            "tile": 7,
            "source_index": 5,
            "target_level": 0,
            "sequence": 2,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)
    assert result["ok"]
    assert result["required_targets"] == 0


def test_trace_verify_rejects_loud_stall_event(tmp_path):
    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 4,
            "source_index": 4,
            "target_level": 0,
            "sequence": 1,
        },
        {
            "kind": "stall",
            "session_id": 9,
            "owner_chain": {"required_unsettled": [4]},
            "sequence": 2,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)
    assert not result["ok"]
    assert {
        "invariant": "no_stall_events",
        "session_id": 9,
        "owner_chain": {"required_unsettled": [4]},
    } in result["violations"]


def test_trace_verify_rejects_empty_and_lifecycle_free_traces(tmp_path):
    from arrayscope.tools.trace_verify import verify_trace

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    result = verify_trace(empty)
    assert not result["ok"]
    assert result["violations"][0]["invariant"] == "trace_not_empty"

    no_lifecycle = tmp_path / "no_lifecycle.jsonl"
    no_lifecycle.write_text(
        json.dumps({"kind": "backend_ack", "tile": 0, "accepted": True, "sequence": 1}) + "\n"
    )
    result = verify_trace(no_lifecycle)
    assert not result["ok"]
    assert result["violations"][0]["invariant"] == "lifecycle_events_present"


def test_trace_verify_enforces_expected_target_count(tmp_path):
    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 1,
            "source_index": 9,
            "target_level": 0,
            "sequence": 1,
        },
        {
            "kind": "backend_ack",
            "tile": 1,
            "source_index": 9,
            "level": 0,
            "quality": "exact",
            "accepted": True,
            "sequence": 2,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert verify_trace(path, expect_targets=1)["ok"]

    mismatched = verify_trace(path, expect_targets=36)
    assert not mismatched["ok"]
    assert mismatched["violations"][0]["invariant"] == "required_target_count"
    assert mismatched["violations"][0]["observed"] == 1


def test_trace_latency_reports_finish_outcomes(tmp_path):
    from arrayscope.tools.trace_latency import analyze_trace_latency

    path = tmp_path / "trace.jsonl"
    rows = (
        {"kind": "kernel_finish", "task_seq": 1, "ts_ns": 1, "outcome": "completed"},
        {"kind": "kernel_finish", "task_seq": 2, "ts_ns": 2, "outcome": "superseded"},
        {"kind": "kernel_finish", "task_seq": 3, "ts_ns": 3, "outcome": "completed"},
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = analyze_trace_latency(path)

    assert summary["kernel_finish_outcomes"] == {"completed": 2, "superseded": 1}
