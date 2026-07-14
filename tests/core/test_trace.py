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
