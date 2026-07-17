"""Fault-injection proof for every journey trajectory oracle."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

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
    return [start, *commits, end], timeline, {"gesture_id": "cold_fill-1", "start": start, "end": end}


def _evaluate(trace, timeline, interval):
    return evaluate_gesture(trace, timeline, backend="vispy", interval=interval)


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


def test_matrix_declares_all_ten_backend_journey_cells():
    from arrayscope.tools.journey_matrix import BACKENDS, JOURNEYS, MIN_COMMITS

    assert set(MIN_COMMITS) == {
        (backend, journey) for backend in BACKENDS for journey in JOURNEYS
    }
    assert MIN_COMMITS[("pyqtgraph", "cold_fill")] >= 2


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


def test_vispy_zero_upload_rebind_is_exempt_from_item_cap(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [[0, "preview", 2], [2, "preview", 2]]
    trace[1]["delta_priority_ranks"] = [[0, 0], [2, 1]]
    trace[1]["committed_upserts"] = [0, 2]
    trace[1]["uploads"] = 0
    trace[1]["upload_bytes"] = 0
    trace[1]["vertex_uploads"] = 0

    result = _evaluate(trace, timeline, interval)

    assert result["presentation"]["bounded"]
    assert result["presentation"]["cap_exemptions"] == [
        {"sequence": 10, "size": 2, "limit": 1, "reason": "vispy_zero_upload_rebind"}
    ]


def test_vispy_pixel_upload_cannot_claim_rebind_cap_exemption(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["delta_qualities"] = [[0, "preview", 2], [2, "preview", 2]]
    trace[1]["delta_priority_ranks"] = [[0, 0], [2, 1]]
    trace[1]["committed_upserts"] = [0, 2]
    trace[1]["uploads"] = 1
    trace[1]["upload_bytes"] = 4096
    trace[1]["vertex_uploads"] = 0

    result = _evaluate(trace, timeline, interval)

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


def test_level_convergence_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[2]["ts_ns"] = 6_300_000_001
    trace[-1]["ts_ns"] = 6_300_000_001
    interval["end"] = trace[-1]

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert result["level_convergence_ms_after_pass_close"] > 5_000.0
    assert not result["level_converged_within_budget"]


def test_missing_coverage_close_oracle_fault_injection(tmp_path):
    trace, timeline, interval = _artifacts(tmp_path)
    trace[1]["coverage_pass_closed"] = False
    trace[1]["preview_pass_open_before"] = True

    result = _evaluate(trace, timeline, interval)

    assert not result["ok"]
    assert not result["coverage_pass_observed"]
    assert not result["level_converged_within_budget"]
