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


def test_trace_verify_rejects_identity_rejected_commits(tmp_path):
    """Session-148 follow-up: a healthy replay never re-emits dead payloads.

    ``commit_batch``/``backend_complete`` events carry the tiles whose
    upserts the backend refused at the typed-identity gate.  Any non-empty
    ``identity_rejected`` in a whole-workflow replay means the session
    queued a payload that can never satisfy its own target — the re-emit
    loop behind the stale/empty-tile stall — and must fail verification
    even when every target eventually settles.
    """

    from arrayscope.tools.trace_verify import verify_trace

    rows = [
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 0,
            "source_index": 0,
            "target_level": 0,
            "sequence": 1,
        },
        {
            "kind": "commit_batch",
            "phase": "backend_complete",
            "session_id": 148,
            "revision": 7,
            "committed_upserts": [],
            "identity_rejected": [0],
            "delta_upserts": [0],
            "sequence": 2,
        },
        {
            "kind": "backend_ack",
            "tile": 0,
            "accepted": True,
            "source_index": 0,
            "level": 0,
            "quality": "exact",
            "sequence": 3,
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)
    assert not result["ok"]
    assert {
        "invariant": "no_identity_rejected_commits",
        "session_id": 148,
        "revision": 7,
        "identity_rejected": (0,),
    } in result["violations"]
    assert result["identity_rejected_commits"] == 1

    # The same trace with a clean commit verifies green.
    rows[1]["identity_rejected"] = []
    rows[1]["committed_upserts"] = [0]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    clean = verify_trace(path)
    assert clean["ok"]
    assert clean["identity_rejected_commits"] == 0


def test_trace_verify_rejects_exact_upsert_during_open_preview_pass(tmp_path):
    """A declared first-pixel pass cannot publish a new exact island."""

    from arrayscope.tools.trace_verify import verify_trace

    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 0,
            "source_index": 0,
            "target_level": 0,
            "sequence": 1,
        },
        {
            "kind": "commit_batch",
            "phase": "backend_complete",
            "session_id": 2,
            "revision": 4,
            "preview_pass_open_before": True,
            "exact_upserts_during_preview_pass": [0],
            "sequence": 2,
        },
        {
            "kind": "backend_ack",
            "tile": 0,
            "accepted": True,
            "source_index": 0,
            "level": 0,
            "quality": "exact",
            "sequence": 3,
        },
    )
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)

    assert not result["ok"]
    assert result["preview_pass_exact_commits"] == 1
    assert {
        "invariant": "preview_pass_precedes_exact_upserts",
        "session_id": 2,
        "revision": 4,
        "exact_upserts": (0,),
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


def test_trace_verify_flags_acknowledgement_churn_livelock(tmp_path):
    """The watchdog (V3) sees deadlocks; identical-ack churn is the livelock signal."""

    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 2,
            "source_index": 7,
            "target_level": 0,
            "sequence": 1,
        }
    ]
    rows.extend(
        {
            "kind": "backend_ack",
            "tile": 2,
            "source_index": 7,
            "level": 0,
            "quality": "exact",
            "accepted": True,
            "sequence": 2 + offset,
        }
        for offset in range(30)
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)
    assert not result["ok"]
    churn = [v for v in result["violations"] if v["invariant"] == "no_acknowledgement_churn"]
    assert churn and churn[0]["tile"] == 2 and churn[0]["identical_acks"] == 30

    # The same trace passes with the check disabled or a higher limit.
    assert verify_trace(path, max_identical_acks=0)["ok"]
    assert verify_trace(path, max_identical_acks=50)["ok"]


def test_trace_verify_flags_identical_commit_bail_livelock(tmp_path):
    """Ring 0 gate: a barrier cannot replan forever without a producer."""

    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 2,
            "source_index": 7,
            "target_level": 0,
            "sequence": 1,
        },
        {
            "kind": "backend_ack",
            "tile": 2,
            "source_index": 7,
            "level": 0,
            "quality": "exact",
            "accepted": True,
            "sequence": 2,
        },
    ]
    rows.extend(
        {
            "kind": "commit_bail",
            "outcome": "shader-atomic-successor-wait",
            "wakeup": "replan",
            "session_id": 57,
            "active_payloads": 10,
            "active_tiles": 15,
            "pending_upserts": 28,
            "dirty_payloads": 33,
            "sequence": 3 + offset,
            "ts_ns": 1000 + offset,
        }
        for offset in range(30)
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)
    assert not result["ok"]
    loops = [
        violation
        for violation in result["violations"]
        if violation["invariant"] == "no_identical_commit_bail_loop"
    ]
    assert loops == [
        {
            "invariant": "no_identical_commit_bail_loop",
            "outcome": "shader-atomic-successor-wait",
            "wakeup": "replan",
            "session_id": 57,
            "identical_commit_bails": 30,
            "limit": 25,
        }
    ]

    assert verify_trace(path, max_identical_commit_bails=0)["ok"]
    assert verify_trace(path, max_identical_commit_bails=50)["ok"]


def test_trace_verify_retains_ack_across_compatible_retarget(tmp_path):
    """Idempotent backends do not re-ack a retarget the payload already satisfies."""

    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 3,
            "source_index": 41,
            "target_level": 1,
            "sequence": 1,
        },
        {
            "kind": "backend_ack",
            "tile": 3,
            "source_index": 41,
            "level": 0,
            "quality": "exact",
            "accepted": True,
            "sequence": 2,
        },
        # Same identity re-required (e.g. a zoom/pan replan): no re-ack follows.
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 3,
            "source_index": 41,
            "target_level": 1,
            "sequence": 3,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert verify_trace(path, expect_targets=1)["ok"]


def test_trace_verify_drops_ack_across_incompatible_retarget(tmp_path):
    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 3,
            "source_index": 41,
            "target_level": 1,
            "sequence": 1,
        },
        {
            "kind": "backend_ack",
            "tile": 3,
            "source_index": 41,
            "level": 0,
            "quality": "exact",
            "accepted": True,
            "sequence": 2,
        },
        # The slot now names a different source: the old pixels cannot satisfy it.
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 3,
            "source_index": 55,
            "target_level": 1,
            "sequence": 3,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = verify_trace(path)
    assert not result["ok"]
    assert result["violations"][0]["invariant"] == "final_required_target_acknowledged"
    assert result["violations"][0]["tile"] == 3


def test_trace_verify_accepts_retained_satisfaction_edge(tmp_path):
    """`target_satisfied_retained` closes a target without a fresh backend ack."""

    from arrayscope.tools.trace_verify import verify_trace

    path = tmp_path / "trace.jsonl"
    rows = (
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 6,
            "source_index": 12,
            "target_level": 2,
            "sequence": 1,
        },
        {
            "kind": "lifecycle",
            "edge": "target_satisfied_retained",
            "tile": 6,
            "source_index": 12,
            "level": 0,
            "quality": "exact",
            "sequence": 2,
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert verify_trace(path, expect_targets=1)["ok"]

    # A retained edge that predates the current requirement does not count.
    stale = tmp_path / "stale.jsonl"
    stale.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "kind": "lifecycle",
                    "edge": "target_satisfied_retained",
                    "tile": 6,
                    "source_index": 12,
                    "sequence": 1,
                },
                {
                    "kind": "lifecycle",
                    "edge": "target_required",
                    "tile": 6,
                    "source_index": 12,
                    "target_level": 2,
                    "sequence": 2,
                },
            )
        )
    )
    assert not verify_trace(stale)["ok"]
