import json
from decimal import Decimal


def test_trace_bus_writes_flat_jsonl_and_bounds_ring(tmp_path):
    from arrayscope.core.trace import TraceBus

    path = tmp_path / "trace.jsonl"
    bus = TraceBus()
    bus.configure(path, ring_events=3)
    for index in range(8):
        bus.emit("kernel_submit", task_seq=index, key=("tile", index))
    snapshot = bus.snapshot()
    bus.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 8
    assert rows[0]["schema_version"] == 1
    assert rows[-1]["sequence"] == 8
    assert all(row["kind"] == "kernel_submit" for row in rows)
    # The ring keeps exactly the last `ring_events` events, and it keeps the
    # *newest* ones — a stall dump is only useful if it ends at the stall.
    assert len(snapshot) == 3
    assert [event["sequence"] for event in snapshot] == [6, 7, 8]


def test_live_trace_sink_hides_buffered_rows_until_close(tmp_path):
    """Characterize the sink's loss of live row visibility while buffered."""

    from arrayscope.core.trace import TraceBus

    path = tmp_path / "trace.jsonl"
    bus = TraceBus()
    bus.configure(path, ring_events=0)
    bus.emit("lifecycle", edge="fallback_ready", tile=7)

    assert path.read_text() == ""

    bus.close()
    assert len(path.read_text().splitlines()) == 1


def test_trace_ring_only_bus_dumps_complete_parseable_jsonl(tmp_path):
    """The ring-only bus (production's watchdog default) encodes at dump."""

    from arrayscope.core.trace import TraceBus

    bus = TraceBus()
    bus.configure(ring_events=4)
    assert bus.enabled
    for index in range(6):
        bus.emit("lifecycle", edge="target_required", tile=index, key=Decimal(index))
    dump = bus.dump(tmp_path / "nested" / "stall.trace.jsonl")
    bus.close()

    rows = [json.loads(line) for line in dump.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [3, 4, 5, 6]
    assert [row["tile"] for row in rows] == [2, 3, 4, 5]
    # A field JSON cannot represent still degrades to `repr` at the boundary.
    assert rows[0]["key"] == "Decimal('2')"


def test_trace_ring_can_drain_to_one_appended_jsonl_without_resetting_sequence(tmp_path):
    from arrayscope.core.trace import TraceBus

    path = tmp_path / "trace.jsonl"
    bus = TraceBus()
    bus.configure(ring_events=4)
    bus.emit("kernel_start", task_seq=1)
    bus.emit("kernel_finish", task_seq=1)
    bus.drain(path)
    assert bus.snapshot() == ()

    bus.emit("kernel_start", task_seq=2)
    bus.drain(path, append=True)
    bus.close()

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["kind"] for row in rows] == [
        "kernel_start",
        "kernel_finish",
        "kernel_start",
    ]


def test_trace_dump_survives_an_event_that_cannot_encode(tmp_path):
    """A pathological field costs its own row, not the whole stall dump.

    Moving the encode to dump time moves any encode failure there too, which
    is the moment the trace is the evidence.  The dump must stay complete and
    parseable line-for-line.
    """

    from arrayscope.core.trace import TraceBus

    cyclic: list[object] = []
    cyclic.append(cyclic)

    bus = TraceBus()
    bus.configure(ring_events=8)
    bus.emit("lifecycle", edge="target_required", tile=0)
    bus.emit("lifecycle", edge="fallback_ready", tile=1, payload=cyclic)
    bus.emit("stall", session_id=3)
    dump = bus.dump(tmp_path / "stall.trace.jsonl")
    bus.close()

    rows = [json.loads(line) for line in dump.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["kind"] for row in rows] == ["lifecycle", "lifecycle", "stall"]
    assert "trace_encode_error" in rows[1]
    assert "trace_encode_error" not in rows[0]
    assert "trace_encode_error" not in rows[2]


def test_trace_bus_without_ring_or_sink_is_disabled(tmp_path):
    from arrayscope.core.trace import TraceBus

    bus = TraceBus()
    bus.configure(ring_events=0)
    assert not bus.enabled
    bus.emit("lifecycle", edge="target_required", tile=0)
    assert bus.snapshot() == ()

    # A sink with no ring still writes every line and keeps no tail.
    path = tmp_path / "sink-only.jsonl"
    bus.configure(path, ring_events=0)
    assert bus.enabled
    bus.emit("lifecycle", edge="target_required", tile=0)
    bus.close()
    assert bus.snapshot() == ()
    assert len(path.read_text().splitlines()) == 1


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


def test_trace_verify_rejects_phase2_kernel_submit_while_coverage_is_open(tmp_path):
    """Progressive contract: compute phases are ordered at submission."""

    from arrayscope.tools.trace_verify import verify_trace

    rows = [
        {
            "kind": "lifecycle",
            "edge": "target_required",
            "tile": 0,
            "source_index": 8,
            "target_level": 1,
            "sequence": 1,
        },
        {
            "kind": "kernel_submit",
            "task_seq": 12,
            "lane": "display_preparation",
            "presentation_phase": 2,
            "coverage_pass_open": True,
            "session_id": 4,
            "tile_number": 0,
            "sequence": 2,
        },
        {
            "kind": "backend_ack",
            "tile": 0,
            "source_index": 8,
            "level": 1,
            "quality": "exact",
            "accepted": True,
            "sequence": 3,
        },
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    failed = verify_trace(path)
    assert not failed["ok"]
    assert failed["phase2_submits_during_coverage"] == 1
    assert {
        "invariant": "no_phase2_submit_during_coverage",
        "session_id": 4,
        "task_seq": 12,
        "lane": "display_preparation",
        "tile": 0,
    } in failed["violations"]

    rows[1]["coverage_pass_open"] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert verify_trace(path)["ok"]


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
    assert churn
    assert churn[0]["tile"] == 2
    assert churn[0]["identical_acks"] == 30

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


def _drive_lifecycle_trace(tmp_path, drive):
    """Run one lifecycle scenario with the real trace bus writing to a file."""

    from arrayscope.core.trace import close_trace, configure_trace

    path = tmp_path / "lifecycle-trace.jsonl"
    configure_trace(path)
    try:
        drive()
    finally:
        close_trace()
    return path


def _retained_edges(path):
    return [
        event
        for event in (json.loads(line) for line in path.read_text().splitlines())
        if event.get("kind") == "lifecycle" and event.get("edge") == "target_satisfied_retained"
    ]


def test_lifecycle_emits_retained_satisfaction_for_finer_fallback_ack(tmp_path):
    """A retained finer fallback settles its target with only fallback acks.

    ``_quality_lod_satisfies_target`` lets already-finer fallback pixels close
    a coarser demand, but the only backend acks on the bus carry fallback
    quality, which replay must not count as exact settlement.  Production owns
    saying `target_satisfied_retained`; without it the strongest invariant
    reads the run as never satisfied.
    """

    from arrayscope.presentation.tile_lifecycle import (
        TileLifecycle,
        TilePayloadRef,
        TileTarget,
    )
    from arrayscope.tools.trace_verify import verify_trace

    lc = TileLifecycle()
    fallback = TilePayloadRef(source_id="p0", quality="fallback", lod_level=0, source_index=5)

    def drive():
        lc.retarget(
            {0: TileTarget(tile_number=0, source_index=5, semantic_source_id="s5", lod_level=2)}
        )
        lc.fallback_ready(0, fallback)
        lc.commit_emitted({0: fallback})
        lc.backend_ack({0: fallback})

    path = _drive_lifecycle_trace(tmp_path, drive)

    assert lc.record(0).target_settled
    edges = _retained_edges(path)
    assert edges
    assert edges[0]["tile"] == 0
    result = verify_trace(path, expect_targets=1)
    assert result["ok"], result["violations"]


def test_lifecycle_emits_retained_satisfaction_on_confirmed_resident_commit(tmp_path):
    """Resident-retarget commits confirm without upserts and without acks."""

    from arrayscope.presentation.tile_lifecycle import (
        TileLifecycle,
        TilePayloadRef,
        TileTarget,
    )
    from arrayscope.tools.trace_verify import verify_trace

    lc = TileLifecycle()
    exact = TilePayloadRef(source_id="p1", quality="exact", lod_level=0, source_index=7)

    def drive():
        lc.retarget(
            {1: TileTarget(tile_number=1, source_index=7, semantic_source_id="s7", lod_level=0)}
        )
        lc.target_ready(1, exact)
        lc.commit_emitted({1: exact})
        lc.backend_presented_snapshot({1: "p1"})
        lc.presentation_confirmed([1])

    path = _drive_lifecycle_trace(tmp_path, drive)

    assert lc.record(1).target_settled
    assert _retained_edges(path)
    result = verify_trace(path, expect_targets=1)
    assert result["ok"], result["violations"]


def test_lifecycle_reaffirms_retained_satisfaction_across_compatible_retarget(tmp_path):
    """Each new requirement a retained payload closes gets its own edge.

    The backend never re-acks a retarget the resident payload already
    satisfies; the lifecycle is the only owner that can say the new
    requirement was closed by retention.
    """

    from arrayscope.presentation.tile_lifecycle import (
        TileLifecycle,
        TilePayloadRef,
        TileTarget,
    )
    from arrayscope.tools.trace_verify import verify_trace

    lc = TileLifecycle()
    fallback = TilePayloadRef(source_id="p0", quality="fallback", lod_level=0, source_index=5)

    def drive():
        lc.retarget(
            {0: TileTarget(tile_number=0, source_index=5, semantic_source_id="s5", lod_level=2)}
        )
        lc.fallback_ready(0, fallback)
        lc.commit_emitted({0: fallback})
        lc.backend_ack({0: fallback})
        # Coarse zoom replan: still strictly coarser than the retained level-0
        # fallback, so no work and no ack follow.
        lc.retarget(
            {0: TileTarget(tile_number=0, source_index=5, semantic_source_id="s5", lod_level=1)}
        )

    path = _drive_lifecycle_trace(tmp_path, drive)

    assert lc.record(0).target_settled
    assert len(_retained_edges(path)) == 2
    result = verify_trace(path, expect_targets=1)
    assert result["ok"], result["violations"]


def test_lifecycle_commit_path_reaffirmation_is_idempotent_per_requirement(tmp_path):
    """`note_retained_satisfaction` re-affirms settled scope without churn."""

    from arrayscope.presentation.tile_lifecycle import (
        TileLifecycle,
        TilePayloadRef,
        TileTarget,
    )

    lc = TileLifecycle()
    exact = TilePayloadRef(source_id="p1", quality="exact", lod_level=0, source_index=7)

    def drive():
        lc.retarget(
            {1: TileTarget(tile_number=1, source_index=7, semantic_source_id="s7", lod_level=0)}
        )
        lc.target_ready(1, exact)
        lc.commit_emitted({1: exact})
        lc.backend_presented_snapshot({1: "p1"})
        lc.presentation_confirmed([1])
        # The settled noop-commit bail re-affirms the whole required scope;
        # repeated polls must not spam the bus (unsettled tile 2 stays quiet).
        lc.note_retained_satisfaction((1, 2))
        lc.note_retained_satisfaction((1, 2))

    path = _drive_lifecycle_trace(tmp_path, drive)

    edges = _retained_edges(path)
    assert len(edges) == 1
    assert edges[0]["tile"] == 1


def test_trace_verify_retained_edge_quality_matches_lifecycle_settlement(tmp_path):
    """Fallback-quality retention closes only strictly coarser later demands.

    Mirrors ``_quality_lod_satisfies_target``: an equal-level fallback still
    owes exact target work, so a retained fallback edge must not survive a
    retarget to its own level.
    """

    from arrayscope.tools.trace_verify import verify_trace

    def rows(final_target_level):
        return (
            {
                "kind": "lifecycle",
                "edge": "target_required",
                "tile": 4,
                "source_index": 9,
                "target_level": 2,
                "sequence": 1,
            },
            {
                "kind": "lifecycle",
                "edge": "target_satisfied_retained",
                "tile": 4,
                "source_index": 9,
                "level": 1,
                "quality": "fallback",
                "sequence": 2,
            },
            {
                "kind": "lifecycle",
                "edge": "target_required",
                "tile": 4,
                "source_index": 9,
                "target_level": final_target_level,
                "sequence": 3,
            },
        )

    coarser = tmp_path / "coarser.jsonl"
    coarser.write_text("".join(json.dumps(row) + "\n" for row in rows(2)))
    assert verify_trace(coarser, expect_targets=1)["ok"]

    equal = tmp_path / "equal.jsonl"
    equal.write_text("".join(json.dumps(row) + "\n" for row in rows(1)))
    result = verify_trace(equal)
    assert not result["ok"]
    assert result["violations"][0]["invariant"] == "final_required_target_acknowledged"


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
