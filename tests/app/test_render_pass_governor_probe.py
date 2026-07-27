from arrayscope.tools.render_pass_governor_probe import (
    _backend_schedule,
    _fill_summary,
    _pass_summary,
)


def test_backend_schedule_is_interleaved_and_order_balanced():
    assert _backend_schedule(("pyqtgraph", "wgpu"), 2) == (
        (0, "pyqtgraph"),
        (0, "wgpu"),
        (1, "wgpu"),
        (1, "pyqtgraph"),
    )


def test_fill_summary_reports_wall_clock_and_incremental_target_throughput():
    summary = _fill_summary(
        {
            "requested_tile_count": 272,
            "first_preview_payload_fill_ms": 2000.0,
            "required_target_settled_ms": 6000.0,
            "interaction_settle_within_budget": False,
            "active_presented_tile_count": 272,
            "final_exact_payload_count": 272,
        },
        backend="wgpu",
        repeat=0,
    )

    assert summary.preview_complete_ms == 2000.0
    assert summary.target_settle_ms == 6000.0
    assert summary.preview_tiles_per_s == 136.0
    assert summary.target_tiles_per_s == 68.0
    assert summary.settlement_status == "late"
    assert summary.presented_tiles == 272
    assert summary.exact_payload_tiles == 272


def test_pass_summary_keeps_full_latency_and_splits_structural_time():
    summary = _pass_summary(
        [
            {
                "pass_kind": "target",
                "elapsed_ms": 100.0,
                "pass_chunk_items": 2,
                "backend_pool_growth_ms": 60.0,
                "backend_executor_initialization_ms": 5.0,
                "pass_completed_atomically": False,
            },
            {
                "pass_kind": "target",
                "elapsed_ms": 40.0,
                "pass_chunk_items": 8,
                "backend_pool_growth_ms": 0.0,
                "backend_executor_initialization_ms": 0.0,
                "pass_completed_atomically": False,
            },
        ],
        backend="wgpu",
        repeat=0,
        pass_kind="target",
        required_tiles=272,
    )

    assert summary.full_max_ms == 100.0
    assert summary.full_over_50 == 1
    assert summary.steady_max_ms == 40.0
    assert summary.structural_pool_growth_ms == 60.0
    assert summary.structural_executor_initialization_ms == 5.0
    assert summary.atomic_chunks == 0
