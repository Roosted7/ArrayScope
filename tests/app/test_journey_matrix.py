"""Fault-injection proof for every journey trajectory oracle."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from arrayscope.tools.journey_matrix import evaluate_gesture


def _artifacts(tmp_path):
    baseline = tmp_path / "baseline.png"
    changed = tmp_path / "changed.png"
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(baseline)
    pixels = np.zeros((20, 20, 3), dtype=np.uint8)
    pixels[5:15, 5:15] = 255
    Image.fromarray(pixels).save(changed)
    start_ns = 1_000_000_000
    end_ns = 1_600_000_000
    start = {
        "kind": "input",
        "action": "journey_gesture",
        "edge": "start",
        "journey": "cold_fill",
        "gesture_id": "cold_fill-1",
        "ts_ns": start_ns,
    }
    end = {
        "kind": "input",
        "action": "journey_gesture",
        "edge": "complete",
        "journey": "cold_fill",
        "gesture_id": "cold_fill-1",
        "ts_ns": end_ns,
        "session_desired_level": 1,
        "applied_level": 1,
    }
    commits = [
        {
            "kind": "commit_batch",
            "phase": "backend_complete",
            "ts_ns": 1_100_000_000,
            "sequence": 10,
            "delta_qualities": [[0, "preview", 2]],
            "delta_priority_ranks": [[0, 0]],
            "committed_upserts": [0],
            "max_upserts": 1,
            "coverage_pass_closed": True,
            "desired_level": 1,
            "applied_level": 2,
        },
        {
            "kind": "commit_batch",
            "phase": "backend_complete",
            "ts_ns": 1_300_000_000,
            "sequence": 20,
            "delta_qualities": [[1, "exact", 1]],
            "delta_priority_ranks": [[1, 1]],
            "committed_upserts": [1],
            "max_upserts": 1,
            "coverage_pass_closed": False,
            "desired_level": 1,
            "applied_level": 1,
        },
    ]
    timeline = [
        {
            "reason": "journey-start",
            "gesture_id": "cold_fill-1",
            "monotonic_ns": start_ns,
            "screenshot_saved": True,
            "screenshot_path": str(baseline),
            "camera_desired_level": 2,
            "session_desired_level": 2,
        },
        {
            "reason": "interval",
            "gesture_id": "cold_fill-1",
            "monotonic_ns": 1_150_000_000,
            "screenshot_saved": True,
            "screenshot_path": str(changed),
            "camera_desired_level": 1,
            "session_desired_level": 1,
            "applied_level": 2,
        },
        {
            "reason": "journey-end",
            "gesture_id": "cold_fill-1",
            "monotonic_ns": end_ns,
            "screenshot_saved": True,
            "screenshot_path": str(changed),
            "camera_desired_level": 1,
            "session_desired_level": 1,
            "applied_level": 1,
        },
    ]
    return (
        [start, *commits, end],
        timeline,
        {"gesture_id": "cold_fill-1", "start": start, "end": end},
    )


def _evaluate(trace, timeline, interval, *, backend="vispy"):
    return evaluate_gesture(trace, timeline, backend=backend, interval=interval)


def test_healthy_trajectory_fixture_exercises_all_oracles(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [[0, "preview", 2], [2, "preview", 2]]
    trace[1]["delta_priority_ranks"] = [[0, 0], [2, 1]]
    trace[1]["committed_upserts"] = [0, 2]
    trace[1]["max_upserts"] = 2
    trace[2]["delta_priority_ranks"] = []

    result = _evaluate(trace, timeline, interval)

    assert result["ok"]
    assert result["presentation"]["priority_correlation"] == pytest.approx(1.0)
    assert result["coverage_pass_observed"]


def test_scheduling_owner_close_event_satisfies_coverage_oracle(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["coverage_pass_closed"] = False
    trace.insert(
        2,
        {
            "kind": "scheduling_phase",
            "event": "coverage_closed",
            "ts_ns": 1_200_000_000,
            "sequence": 15,
        },
    )

    result = _evaluate(trace, timeline, interval)

    assert result["coverage_pass_observed"]
    assert result["level_convergence_ms_after_pass_close"] == 100.0
    assert result["ok"]


def test_wgpu_resident_zoom_in_does_not_require_a_payload_commit(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    interval["start"]["journey"] = "zoom_in"
    interval["start"]["gesture_id"] = "zoom_in-1"
    interval["end"]["journey"] = "zoom_in"
    interval["end"]["gesture_id"] = "zoom_in-1"
    interval["gesture_id"] = "zoom_in-1"
    for sample in timeline:
        sample["gesture_id"] = "zoom_in-1"
    trace = [event for event in trace if event.get("kind") != "commit_batch"]

    result = _evaluate(trace, timeline, interval, backend="wgpu")

    assert result["presentation"]["minimum_commits"] == 0
    assert result["presentation"]["commit_count"] == 0
    assert result["presentation"]["ok"]
    assert result["ok"]


def test_wgpu_descriptor_only_missed_redraw_keeps_freshness_red(tmp_path):
    """A wgpu zoom-out over resident content commits nothing by design, so
    the freshness oracle rests entirely on sampled pixels. An injected missed
    redraw — every gesture-tagged screenshot identical to the baseline —
    must stay red even now that the journey-end sample drains pending
    presentation draws first (the drain gives up bounded; it must never
    substitute for an actual repaint)."""

    trace, timeline, interval = _artifacts(tmp_path)
    interval["start"]["journey"] = "zoom_out"
    interval["start"]["gesture_id"] = "zoom_out-1"
    interval["end"]["journey"] = "zoom_out"
    interval["end"]["gesture_id"] = "zoom_out-1"
    interval["gesture_id"] = "zoom_out-1"
    baseline_path = timeline[0]["screenshot_path"]
    for sample in timeline:
        sample["gesture_id"] = "zoom_out-1"
        sample["screenshot_path"] = baseline_path
    trace = [event for event in trace if event.get("kind") != "commit_batch"]

    result = _evaluate(trace, timeline, interval, backend="wgpu")

    assert result["presentation"]["ok"]  # zero commits is legal for this cell
    assert result["first_new_pixels_ms"] is None
    assert not result["first_new_pixels_within_budget"]
    assert not result["ok"]


def test_matrix_declares_every_backend_journey_cell():
    from arrayscope.tools.journey_matrix import BACKENDS, DRIVER_RUNS, JOURNEYS, MIN_COMMITS

    assert BACKENDS == ("wgpu", "pyqtgraph", "vispy")
    assert set(MIN_COMMITS) == {(backend, journey) for backend in BACKENDS for journey in JOURNEYS}
    assert MIN_COMMITS[("pyqtgraph", "cold_fill")] >= 2
    assert "deep_zoom_far_scroll" in DRIVER_RUNS["zoom"][1]


def test_matrix_uses_checked_in_profile_session_fixture():
    from arrayscope.tools.journey_matrix import (
        PROFILE_SESSION_FIXTURE,
        _profile_driver_command,
    )

    assert PROFILE_SESSION_FIXTURE.name == "profile_montage_session.json"
    assert PROFILE_SESSION_FIXTURE.is_file()
    payload = json.loads(PROFILE_SESSION_FIXTURE.read_text(encoding="utf-8"))
    assert payload["panels"]["window_size"] == [1400, 940]
    command = _profile_driver_command(
        backend="wgpu",
        data="scan.nii",
        stages="montage_zoompan_scalar",
        output=Path("artifacts/wgpu/zoom"),
    )
    fixture_index = command.index("--session-fixture") + 1
    assert command[fixture_index] == str(PROFILE_SESSION_FIXTURE)
    assert command[fixture_index]


def test_driver_health_blocks_correctness_and_stall_diagnostics(tmp_path):
    from arrayscope.tools.journey_matrix import _driver_health

    metrics = tmp_path / "metrics.jsonl"
    stderr = tmp_path / "driver.stderr.log"
    metrics.write_text(
        json.dumps(
            {
                "phase": "montage_scroll_scalar",
                "presentation_settled": True,
                "required_target_settled": True,
                "stale_level_tiles": 0,
                "r8_gate_failures": [
                    {
                        "category": "correctness",
                        "gate": "slow_scroll_converged",
                        "evidence": 1,
                    },
                    {
                        "category": "performance",
                        "gate": "event_loop_heartbeat",
                        "evidence": 100.0,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("[arrayscope] STALL TILE PROBE: {'tile': 59}\n", encoding="utf-8")

    health = _driver_health(metrics, stderr)

    assert health["ok"] is False
    assert {row["reason"] for row in health["blocking_failures"]} == {
        "correctness_gate",
        "stall_tile_probe",
    }
    assert [row["gate"] for row in health["performance_diagnostics"]] == ["event_loop_heartbeat"]


def test_driver_health_keeps_screenshot_timing_reds_diagnostic(tmp_path):
    from arrayscope.tools.journey_matrix import _driver_health

    metrics = tmp_path / "metrics.jsonl"
    stderr = tmp_path / "driver.stderr.log"
    metrics.write_text(
        json.dumps(
            {
                "phase": "montage_zoompan_scalar",
                "presentation_settled": True,
                "required_target_settled": True,
                "stale_level_tiles": 0,
                "r8_gate_failures": [
                    {
                        "category": "performance",
                        "gate": "gui_callbacks_below_50ms",
                        "evidence": 75.0,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stderr.write_text("", encoding="utf-8")

    health = _driver_health(metrics, stderr)

    assert health["ok"] is True
    assert health["blocking_failures"] == []
    assert len(health["performance_diagnostics"]) == 1


@pytest.mark.parametrize(
    ("stderr", "reason"),
    [
        (
            "NotImplementedError: wgpu backend renders complex payloads as a single tile only",
            "complex_montage",
        ),
        (
            "NotImplementedError: wgpu backend supports linear shader scale only; got 'log'",
            "nonlinear_scale",
        ),
        (
            "NotImplementedError: wgpu backend supports linear shader scale only; got 'symlog'",
            "nonlinear_scale",
        ),
        (
            "NotImplementedError: wgpu RGB tile 0 payload does not fit rgb8 cleanly",
            "float_rgb",
        ),
    ],
)
def test_wgpu_recorded_loud_rejections_classify_as_unsupported(stderr, reason):
    from arrayscope.tools.journey_matrix import _wgpu_unsupported_reason

    assert _wgpu_unsupported_reason(stderr) == reason


def test_artifact_evaluation_reports_persisted_wgpu_rows_unsupported(tmp_path):
    from arrayscope.tools.journey_matrix import DRIVER_RUNS, JOURNEYS, evaluate_artifact_dir

    for run_name in DRIVER_RUNS:
        output = tmp_path / "wgpu" / run_name
        output.mkdir(parents=True)
        (output / "unsupported.json").write_text(
            '{"reason": "complex_montage"}\n', encoding="utf-8"
        )

    report = evaluate_artifact_dir(tmp_path)
    wgpu_rows = [row for row in report["rows"] if row["backend"] == "wgpu"]

    assert len(wgpu_rows) == len(JOURNEYS)
    assert all(row["status"] == "unsupported" for row in wgpu_rows)
    assert all(row["ok"] for row in wgpu_rows)


def test_phase_order_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace.insert(
        1,
        {
            "kind": "kernel_submit",
            "ts_ns": 1_050_000_000,
            "presentation_phase": 2,
            "coverage_pass_open": True,
            "lane": "display_preparation",
        },
    )

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert not result["phase_ordered"]


def test_bounded_priority_commit_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [[0, "preview", 2], [2, "preview", 2]]
    trace[1]["delta_priority_ranks"] = [[0, 3], [2, 2]]
    trace[1]["committed_upserts"] = [0, 2]
    trace[2]["delta_priority_ranks"] = [[1, 0]]

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert not result["presentation"]["bounded"]
    assert not result["presentation"]["priority_ordered"]


@pytest.mark.parametrize("backend", ["vispy", "wgpu"])
def test_gpu_zero_upload_rebind_is_exempt_from_item_cap(tmp_path, backend):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [[0, "preview", 2], [2, "preview", 2]]
    trace[1]["delta_priority_ranks"] = [[0, 0], [2, 1]]
    trace[1]["committed_upserts"] = [0, 2]
    trace[1]["uploads"] = 0
    trace[1]["upload_bytes"] = 0
    trace[1]["vertex_uploads"] = 0

    result = _evaluate(trace, timeline, interval, backend=backend)

    assert result["presentation"]["bounded"]
    assert result["presentation"]["cap_exemptions"] == [
        {
            "sequence": 10,
            "size": 2,
            "limit": 1,
            "reason": f"{backend}_zero_upload_rebind",
        }
    ]


@pytest.mark.parametrize("backend", ["vispy", "wgpu"])
def test_gpu_pixel_upload_cannot_claim_rebind_cap_exemption(tmp_path, backend):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [[0, "preview", 2], [2, "preview", 2]]
    trace[1]["delta_priority_ranks"] = [[0, 0], [2, 1]]
    trace[1]["committed_upserts"] = [0, 2]
    trace[1]["uploads"] = 1
    trace[1]["upload_bytes"] = 4096
    trace[1]["vertex_uploads"] = 0

    result = _evaluate(trace, timeline, interval, backend=backend)

    assert not result["presentation"]["bounded"]
    assert result["presentation"]["cap_exemptions"] == []


@pytest.mark.parametrize("backend", ["vispy", "wgpu"])
def test_gpu_mixed_resident_rebind_caps_only_reported_cold_tiles(tmp_path, backend):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [
        [0, "preview", 2],
        [1, "preview", 2],
        [2, "preview", 2],
    ]
    trace[1]["delta_priority_ranks"] = [[0, 0], [1, 1], [2, 2]]
    trace[1]["committed_upserts"] = [0, 1, 2]
    trace[1]["cold_upsert_tiles"] = [2]
    trace[1]["uploads"] = 1
    trace[1]["upload_bytes"] = 4096
    trace[1]["vertex_uploads"] = 0

    result = _evaluate(trace, timeline, interval, backend=backend)

    assert result["presentation"]["bounded"]
    assert result["presentation"]["cap_exemptions"] == [
        {
            "sequence": 10,
            "size": 3,
            "limit": 1,
            "cold_size": 1,
            "reason": f"{backend}_resident_rebind_with_bounded_cold_upserts",
        }
    ]


def test_gpu_mixed_resident_rebind_rejects_oversized_cold_subset(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [
        [0, "preview", 2],
        [1, "preview", 2],
        [2, "preview", 2],
    ]
    trace[1]["delta_priority_ranks"] = [[0, 0], [1, 1], [2, 2]]
    trace[1]["committed_upserts"] = [0, 1, 2]
    trace[1]["cold_upsert_tiles"] = [1, 2]
    trace[1]["uploads"] = 2
    trace[1]["upload_bytes"] = 8192
    trace[1]["vertex_uploads"] = 0

    result = _evaluate(trace, timeline, interval, backend="wgpu")

    assert not result["presentation"]["bounded"]
    assert result["presentation"]["cap_exemptions"] == []


def test_demand_freshness_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    for sample in timeline[1:]:
        sample["camera_desired_level"] = 0
        sample["session_desired_level"] = 2

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert not result["demand_fresh_within_budget"]


def test_early_old_demand_match_does_not_mask_stale_final_camera(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    timeline[1]["camera_desired_level"] = 2
    timeline[1]["session_desired_level"] = 2
    timeline[2]["camera_desired_level"] = 0
    timeline[2]["session_desired_level"] = 2

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert result["demand_fresh_ms_after_gesture"] is None
    assert not result["demand_fresh_within_budget"]


def test_demand_freshness_uses_transition_trace_when_sample_confirms(tmp_path):
    # The sampled timeline starves during the post-transition replan burst;
    # the product's lod_demand transition trace is the ground-truth
    # timestamp, honored only because a later sample confirms the state.
    trace, timeline, interval = _artifacts(tmp_path)
    trace.insert(
        2,
        {"kind": "lod_demand", "level": 1, "previous_level": 2, "ts_ns": 1_050_000_000},
    )

    result = _evaluate(trace, timeline, interval)

    assert result["demand_fresh_ms_after_gesture"] == 50.0
    assert result["demand_fresh_within_budget"]


def test_demand_freshness_transition_event_alone_cannot_pass(tmp_path):
    # An injected transition trace with no confirming sample stays red: the
    # oracle remains output-driven.
    trace, timeline, interval = _artifacts(tmp_path)
    trace.insert(
        2,
        {"kind": "lod_demand", "level": 1, "previous_level": 2, "ts_ns": 1_050_000_000},
    )
    for sample in timeline[1:]:
        sample["session_desired_level"] = 2

    result = _evaluate(trace, timeline, interval)

    assert result["demand_fresh_ms_after_gesture"] is None
    assert not result["demand_fresh_within_budget"]


def test_demand_freshness_late_transition_still_fails(tmp_path):
    # A genuinely late transition carries a late ground-truth timestamp.
    trace, timeline, interval = _artifacts(tmp_path)
    trace[-1]["ts_ns"] = 8_000_000_000
    interval["end"] = trace[-1]
    trace.insert(
        2,
        {"kind": "lod_demand", "level": 1, "previous_level": 2, "ts_ns": 7_050_000_000},
    )
    timeline[1]["session_desired_level"] = 2
    timeline[2]["monotonic_ns"] = 7_400_000_000

    result = _evaluate(trace, timeline, interval)

    assert result["demand_fresh_ms_after_gesture"] == 6_050.0
    assert not result["demand_fresh_within_budget"]


def test_demand_freshness_budget_starts_at_gesture_not_first_pixels(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[-1]["ts_ns"] = 7_000_000_000
    interval["end"] = trace[-1]
    timeline[1]["monotonic_ns"] = 5_900_000_000
    timeline[1]["camera_desired_level"] = 0
    timeline[1]["session_desired_level"] = 2
    timeline[2]["monotonic_ns"] = 6_100_000_000
    timeline[2]["camera_desired_level"] = 1
    timeline[2]["session_desired_level"] = 1

    result = _evaluate(trace, timeline, interval)

    assert result["demand_fresh_ms_after_gesture"] == 5_100.0
    assert not result["demand_fresh_within_budget"]


def test_first_new_pixels_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    late = deepcopy(timeline[1])
    late["monotonic_ns"] = 4_000_000_000
    timeline = [timeline[0], late]

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert result["first_new_pixels_ms"] == 3_000.0
    assert not result["first_new_pixels_within_budget"]


def test_missed_redraw_fault_injection_still_fails(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    baseline_path = timeline[0]["screenshot_path"]
    for sample in timeline[1:]:
        sample["screenshot_path"] = baseline_path

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert result["first_new_pixels_ms"] is None
    assert not result["first_new_pixels_within_budget"]
    assert result["demand_fresh_ms_after_gesture"] is None
    assert not result["demand_fresh_within_budget"]


def test_level_convergence_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[2]["ts_ns"] = 6_300_000_001
    trace[-1]["ts_ns"] = 6_300_000_001
    interval["end"] = trace[-1]

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert result["level_convergence_ms_after_pass_close"] > 5_000.0
    assert not result["level_converged_within_budget"]


def test_wgpu_cold_level_red_is_unsupported_only_with_identical_reference_red():
    from arrayscope.tools.journey_matrix import _classify_reference_blocked_wgpu_rows

    isolated_red = {
        "completed": True,
        "phase_ordered": True,
        "presentation": {"ok": True},
        "first_new_pixels_within_budget": True,
        "demand_fresh_within_budget": True,
        "coverage_pass_observed": False,
        "level_converged_within_budget": False,
    }
    rows = [
        {
            "backend": "vispy",
            "journey": "cold_fill",
            "status": "failed",
            "ok": False,
            "results": [dict(isolated_red)],
        },
        {
            "backend": "wgpu",
            "journey": "cold_fill",
            "status": "failed",
            "ok": False,
            "results": [dict(isolated_red)],
        },
    ]

    _classify_reference_blocked_wgpu_rows(rows)

    assert rows[0]["status"] == "failed"
    assert not rows[0]["ok"]
    assert rows[1]["status"] == "unsupported"
    assert rows[1]["ok"]
    assert rows[1]["unsupported_reasons"] == ["reference_vispy_cold_level_convergence_standing_red"]


def test_wgpu_cold_level_red_stays_failed_if_reference_has_another_oracle_red():
    from arrayscope.tools.journey_matrix import _classify_reference_blocked_wgpu_rows

    isolated_red = {
        "completed": True,
        "phase_ordered": True,
        "presentation": {"ok": True},
        "first_new_pixels_within_budget": True,
        "demand_fresh_within_budget": True,
        "coverage_pass_observed": False,
        "level_converged_within_budget": False,
    }
    reference_red = dict(isolated_red)
    reference_red["first_new_pixels_within_budget"] = False
    rows = [
        {
            "backend": "vispy",
            "journey": "cold_fill",
            "status": "failed",
            "ok": False,
            "results": [reference_red],
        },
        {
            "backend": "wgpu",
            "journey": "cold_fill",
            "status": "failed",
            "ok": False,
            "results": [isolated_red],
        },
    ]

    _classify_reference_blocked_wgpu_rows(rows)

    assert rows[1]["status"] == "failed"
    assert not rows[1]["ok"]


def test_wgpu_cold_level_red_stays_failed_on_backend_runtime_error():
    from arrayscope.tools.journey_matrix import (
        _classify_reference_blocked_wgpu_rows,
        _wgpu_cold_runtime_clean,
    )

    isolated_red = {
        "completed": True,
        "phase_ordered": True,
        "presentation": {"ok": True},
        "first_new_pixels_within_budget": True,
        "demand_fresh_within_budget": True,
        "coverage_pass_observed": False,
        "level_converged_within_budget": False,
    }
    rows = [
        {
            "backend": backend,
            "journey": "cold_fill",
            "status": "failed",
            "ok": False,
            "results": [dict(isolated_red)],
        }
        for backend in ("vispy", "wgpu")
    ]
    stderr = "GPUValidationError: Dimension Z value 2832 exceeds the limit of 2048"

    _classify_reference_blocked_wgpu_rows(
        rows,
        wgpu_runtime_clean=_wgpu_cold_runtime_clean(stderr),
    )

    assert rows[1]["status"] == "failed"
    assert not rows[1]["ok"]


def test_missing_coverage_close_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["coverage_pass_closed"] = False
    trace[1]["preview_pass_open_before"] = True

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert not result["coverage_pass_observed"]
    assert not result["level_converged_within_budget"]


def test_journey_report_names_the_ring_it_actually_ran_in(monkeypatch):
    """A headless-Weston verdict must not be filed as the developer's session.

    Both are real compositors with real GL, so both satisfy ground rule #1 —
    but a reader adjudicating a red row has to know which machine state
    produced it.
    """

    from arrayscope.tools.journey_matrix import _journey_ring

    offscreen = SimpleNamespace(offscreen_smoke=True)
    real = SimpleNamespace(offscreen_smoke=False)

    monkeypatch.delenv("ARRAYSCOPE_HEADLESS_DISPLAY", raising=False)
    assert _journey_ring(offscreen) == "offscreen-smoke"
    assert _journey_ring(real) == "real-wayland"

    monkeypatch.setenv("ARRAYSCOPE_HEADLESS_DISPLAY", "arrayscope-headless-1")
    assert _journey_ring(real) == "headless-weston"
    assert _journey_ring(offscreen) == "offscreen-smoke"
