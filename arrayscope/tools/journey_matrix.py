"""Output-driven trajectory oracles for the montage journey gate matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np
from PIL import Image


JOURNEYS = ("cold_fill", "zoom_in", "zoom_out", "scroll_shuffle", "index_scroll")
BACKENDS = ("vispy", "pyqtgraph")
DRIVER_RUNS = {
    "cold": ("raw_full_tiled_montage", ("cold_fill",)),
    "scroll": ("montage_scroll_scalar", ("scroll_shuffle", "index_scroll")),
    "zoom": ("montage_zoompan_scalar", ("zoom_in", "zoom_out")),
}
FIRST_NEW_PIXELS_BUDGET_MS = 2_000.0
DEMAND_FRESHNESS_BUDGET_MS = 5_000.0
LEVEL_CONVERGENCE_BUDGET_MS = 5_000.0
MIN_PRIORITY_CORRELATION = 0.50
DRIVER_WATCHDOG_S = 180.0
MIN_COMMITS = {
    ("vispy", "cold_fill"): 2,
    ("pyqtgraph", "cold_fill"): 1,  # CPU-LUT single-pass contract.
    ("vispy", "zoom_in"): 1,
    ("pyqtgraph", "zoom_in"): 1,
    ("vispy", "zoom_out"): 0,  # Finer retained pixels need no payload commit.
    ("pyqtgraph", "zoom_out"): 0,
    ("vispy", "scroll_shuffle"): 2,
    ("pyqtgraph", "scroll_shuffle"): 2,
    ("vispy", "index_scroll"): 1,
    ("pyqtgraph", "index_scroll"): 1,
}


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _gesture_intervals(events: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    starts: dict[str, dict[str, object]] = {}
    intervals: list[dict[str, object]] = []
    for event in events:
        if event.get("kind") != "input" or event.get("action") != "journey_gesture":
            continue
        gesture_id = str(event.get("gesture_id", ""))
        if not gesture_id:
            continue
        if event.get("edge") == "start":
            starts[gesture_id] = event
        elif event.get("edge") == "complete" and gesture_id in starts:
            intervals.append({"gesture_id": gesture_id, "start": starts.pop(gesture_id), "end": event})
    intervals.extend(
        {"gesture_id": gesture_id, "start": start, "end": None}
        for gesture_id, start in starts.items()
    )
    return intervals


def _pixel_change_fraction(before_path: str | Path, after_path: str | Path) -> float:
    before = np.asarray(Image.open(before_path).convert("RGB"), dtype=np.int16)
    after = np.asarray(Image.open(after_path).convert("RGB"), dtype=np.int16)
    if before.shape != after.shape:
        return 1.0
    changed = np.max(np.abs(after - before), axis=-1) >= 3
    return float(np.count_nonzero(changed) / max(1, changed.size))


def _first_new_pixels(samples: list[dict[str, object]]) -> tuple[float | None, dict[str, object] | None]:
    baseline = next((row for row in samples if row.get("reason") == "journey-start"), None)
    if baseline is None or not bool(baseline.get("screenshot_saved", False)):
        return None, None
    baseline_path = baseline.get("screenshot_path")
    baseline_ns = int(baseline.get("monotonic_ns", 0) or 0)
    if not baseline_path or baseline_ns <= 0:
        return None, None
    for sample in samples:
        sample_ns = int(sample.get("monotonic_ns", 0) or 0)
        if sample_ns <= baseline_ns or not bool(sample.get("screenshot_saved", False)):
            continue
        path = sample.get("screenshot_path")
        if not path:
            continue
        try:
            fraction = _pixel_change_fraction(str(baseline_path), str(path))
        except (OSError, ValueError):
            continue
        if fraction >= 0.001:
            return (sample_ns - baseline_ns) / 1_000_000.0, sample
    return None, None


def _pearson(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return None
    x = np.asarray(values_x, dtype=np.float64)
    y = np.asarray(values_y, dtype=np.float64)
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _presentation_oracle(
    commits: list[dict[str, object]], *, backend: str, journey: str
) -> dict[str, object]:
    deltas = [event for event in commits if tuple(event.get("delta_qualities", ()) or ())]
    minimum = int(MIN_COMMITS[(backend, journey)])
    bounded_failures = []
    commit_ordinals: list[float] = []
    scheduling_ranks: list[float] = []
    presentation_ordinal = 0
    for event in deltas:
        delta = tuple(event.get("delta_qualities", ()) or ())
        limit = int(event.get("max_upserts", 0) or 0)
        reason = str(event.get("unbounded_reason", "") or "")
        if limit > 0 and len(delta) > limit:
            bounded_failures.append({"sequence": event.get("sequence"), "size": len(delta), "limit": limit})
        elif limit <= 0 and reason not in {"atomic_successor", "first_cpu_frame"}:
            bounded_failures.append({"sequence": event.get("sequence"), "size": len(delta), "limit": None})
        ranks = {
            int(tile): rank
            for tile, rank in tuple(event.get("delta_priority_ranks", ()) or ())
            if rank is not None
        }
        for row in delta:
            tile = int(row[0])
            if tile in ranks:
                commit_ordinals.append(float(presentation_ordinal))
                scheduling_ranks.append(float(ranks[tile]))
            presentation_ordinal += 1
    rho = _pearson(commit_ordinals, scheduling_ranks)
    correlation_applicable = len(set(commit_ordinals)) >= 2 and len(set(scheduling_ranks)) >= 2
    priority_ok = not correlation_applicable or (rho is not None and rho >= MIN_PRIORITY_CORRELATION)
    return {
        "minimum_commits": minimum,
        "commit_count": len(deltas),
        "bounded": not bounded_failures,
        "bounded_failures": bounded_failures,
        "priority_correlation": rho,
        "priority_correlation_applicable": correlation_applicable,
        "priority_ordered": priority_ok,
        "ok": len(deltas) >= minimum and not bounded_failures and priority_ok,
    }


def evaluate_gesture(
    trace_events: list[dict[str, object]],
    timeline: list[dict[str, object]],
    *,
    backend: str,
    interval: dict[str, object],
) -> dict[str, object]:
    start = dict(interval["start"])
    end = interval.get("end")
    journey = str(start.get("journey", ""))
    gesture_id = str(interval["gesture_id"])
    start_ns = int(start.get("ts_ns", 0) or 0)
    end_ns = int(dict(end or {}).get("ts_ns", 2**63 - 1) or (2**63 - 1))
    segment = [event for event in trace_events if start_ns <= int(event.get("ts_ns", 0) or 0) <= end_ns]
    commits = [
        event
        for event in segment
        if event.get("kind") == "commit_batch" and event.get("phase") == "backend_complete"
    ]
    samples = [row for row in timeline if str(row.get("gesture_id", "")) == gesture_id]
    first_new_ms, first_new_sample = _first_new_pixels(samples)

    phase2 = [
        event
        for event in segment
        if event.get("kind") == "kernel_submit"
        and int(event.get("presentation_phase", 0) or 0) == 2
        and bool(event.get("coverage_pass_open", False))
    ]

    demand_fresh_ms = None
    if first_new_sample is not None:
        output_ns = int(first_new_sample.get("monotonic_ns", 0) or 0)
        final_sample = next(
            (sample for sample in reversed(samples) if sample.get("reason") == "journey-end"),
            None,
        )
        final_camera = (
            None if final_sample is None else final_sample.get("camera_desired_level")
        )
        for sample in samples:
            sample_ns = int(sample.get("monotonic_ns", 0) or 0)
            if sample_ns < output_ns:
                continue
            camera = sample.get("camera_desired_level")
            session = sample.get("session_desired_level")
            if (
                final_camera is not None
                and camera is not None
                and session is not None
                and int(camera) == int(final_camera)
                and int(session) == int(final_camera)
            ):
                demand_fresh_ms = (sample_ns - start_ns) / 1_000_000.0
                break

    close_event = next(
        (event for event in commits if bool(event.get("coverage_pass_closed", False))),
        None,
    )
    coverage_open_observed = any(
        bool(event.get("coverage_pass_open", False))
        or bool(event.get("preview_pass_open_before", False))
        for event in segment
    )
    coverage_pass_observed = bool(close_event is not None or not coverage_open_observed)
    close_ns = (
        start_ns
        if close_event is None
        else int(close_event.get("ts_ns", start_ns) or start_ns)
    )
    convergence_events = (
        [
            event
            for event in commits
            if int(event.get("ts_ns", 0) or 0) >= close_ns
        ]
        if coverage_pass_observed
        else []
    )
    if end is not None and coverage_pass_observed:
        convergence_events.append(dict(end))
    level_convergence_ms = None
    for event in convergence_events:
        desired = event.get("desired_level", event.get("session_desired_level"))
        applied = event.get("applied_level")
        if desired is None or applied is None or int(applied) > int(desired):
            continue
        level_convergence_ms = (int(event.get("ts_ns", close_ns) or close_ns) - close_ns) / 1_000_000.0
        break

    presentation = _presentation_oracle(commits, backend=backend, journey=journey)
    result = {
        "backend": backend,
        "journey": journey,
        "gesture_id": gesture_id,
        "completed": end is not None,
        "phase2_submit_count_during_coverage": len(phase2),
        "phase_ordered": not phase2,
        "presentation": presentation,
        "first_new_pixels_ms": first_new_ms,
        "first_new_pixels_within_budget": (
            first_new_ms is not None and first_new_ms <= FIRST_NEW_PIXELS_BUDGET_MS
        ),
        "demand_fresh_ms_after_gesture": demand_fresh_ms,
        "demand_fresh_within_budget": (
            demand_fresh_ms is not None and demand_fresh_ms <= DEMAND_FRESHNESS_BUDGET_MS
        ),
        "coverage_pass_observed": coverage_pass_observed,
        "level_convergence_ms_after_pass_close": level_convergence_ms,
        "level_converged_within_budget": (
            level_convergence_ms is not None and level_convergence_ms <= LEVEL_CONVERGENCE_BUDGET_MS
        ),
    }
    result["ok"] = bool(
        result["completed"]
        and result["phase_ordered"]
        and presentation["ok"]
        and result["first_new_pixels_within_budget"]
        and result["demand_fresh_within_budget"]
        and result["coverage_pass_observed"]
        and result["level_converged_within_budget"]
    )
    return result


def evaluate_backend_run(
    trace_path: str | Path,
    timeline_path: str | Path,
    *,
    backend: str,
) -> dict[str, object]:
    events = read_jsonl(trace_path)
    timeline = read_jsonl(timeline_path)
    intervals = _gesture_intervals(events)
    instances = [
        evaluate_gesture(events, timeline, backend=backend, interval=interval)
        for interval in intervals
    ]
    rows = []
    for journey in JOURNEYS:
        selected = [item for item in instances if item["journey"] == journey]
        rows.append(
            {
                "backend": backend,
                "journey": journey,
                "instances": len(selected),
                "ok": bool(selected) and all(bool(item["ok"]) for item in selected),
                "results": selected,
            }
        )
    return {"backend": backend, "ok": all(bool(row["ok"]) for row in rows), "rows": rows}


def evaluate_artifact_dir(artifact_dir: str | Path) -> dict[str, object]:
    artifact_dir = Path(artifact_dir)
    rows = []
    for backend in BACKENDS:
        instances_by_journey = {journey: [] for journey in JOURNEYS}
        for run_name, (_stages, owned_journeys) in DRIVER_RUNS.items():
            output = artifact_dir / backend / run_name
            trace_path = output / "trace.jsonl"
            timeline_path = output / f"{backend}-visual-timeline.jsonl"
            if not trace_path.exists() or not timeline_path.exists():
                continue
            result = evaluate_backend_run(trace_path, timeline_path, backend=backend)
            for row in result["rows"]:
                journey = str(row["journey"])
                if journey in owned_journeys:
                    instances_by_journey[journey].extend(row["results"])
        rows.extend(
            {
                "backend": backend,
                "journey": journey,
                "instances": len(instances_by_journey[journey]),
                "ok": bool(instances_by_journey[journey])
                and all(bool(item["ok"]) for item in instances_by_journey[journey]),
                "results": instances_by_journey[journey],
            }
            for journey in JOURNEYS
        )
    return {"ok": all(bool(row["ok"]) for row in rows), "rows": rows}


def run_matrix(args) -> int:
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for backend in BACKENDS:
        for run_name, (stages, _owned_journeys) in DRIVER_RUNS.items():
            output = artifact_dir / backend / run_name
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "-m",
                "arrayscope.tools.profile_montage_workflow",
                "--backend",
                backend,
                "--data",
                str(args.data),
                "--session-fixture",
                "",
                "--stages",
                stages,
                "--trace",
                str(output / "trace.jsonl"),
                "--jsonl",
                str(output / "metrics.jsonl"),
                "--screenshot-dir",
                str(output),
                "--screenshot-interval-s",
                "0.1",
                "--timeout-s",
                "5",
            ]
            env = dict(os.environ)
            if args.offscreen_smoke:
                env["QT_QPA_PLATFORM"] = "offscreen"
            elif env.get("QT_QPA_PLATFORM") != "wayland":
                raise SystemExit("real journey ring requires QT_QPA_PLATFORM=wayland")
            stdout_path = output / "driver.stdout.log"
            stderr_path = output / "driver.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=Path(__file__).parents[2],
                        env=env,
                        check=False,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=DRIVER_WATCHDOG_S,
                    )
                    returncode = int(completed.returncode)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    returncode = None
                    timed_out = True
            if returncode not in (0, None):
                failures.append(
                    {
                        "backend": backend,
                        "run": run_name,
                        "returncode": returncode,
                        "stderr": str(stderr_path),
                        "gate_effect": "diagnostic_only",
                    }
                )
            elif timed_out:
                failures.append(
                    {
                        "backend": backend,
                        "run": run_name,
                        "returncode": None,
                        "stderr": str(stderr_path),
                        "timed_out_after_s": DRIVER_WATCHDOG_S,
                        "gate_effect": "incomplete_artifacts_fail_owned_cells",
                    }
                )
    report = evaluate_artifact_dir(artifact_dir)
    report["driver_failures"] = failures
    report["ring"] = "offscreen-smoke" if args.offscreen_smoke else "real-wayland"
    report["ok"] = bool(
        report["ok"]
        and not any("timed_out_after_s" in failure for failure in failures)
    )
    (artifact_dir / "journey-matrix.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify an existing artifact directory")
    verify.add_argument("artifact_dir")
    run = subparsers.add_parser("run", help="drive both backends then verify the matrix")
    run.add_argument("--artifact-dir", required=True)
    run.add_argument("--data", default="data/_WIPDelRec-tT2_20260223150234_14.nii")
    run.add_argument("--offscreen-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_matrix(args)
    report = evaluate_artifact_dir(args.artifact_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
