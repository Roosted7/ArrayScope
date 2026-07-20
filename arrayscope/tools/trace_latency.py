"""Summarize causal queue/run/drain latency from an ArrayScope event trace."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path

import numpy as np


def analyze_trace_latency(path: str | Path) -> dict[str, object]:
    events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    counts = Counter(str(event.get("kind", "")) for event in events)
    submitted = {
        int(event["task_seq"]): int(event["ts_ns"])
        for event in events
        if event.get("kind") == "kernel_submit" and event.get("task_seq") is not None
    }
    started = {
        int(event["task_seq"]): int(event["ts_ns"])
        for event in events
        if event.get("kind") == "kernel_start" and event.get("task_seq") is not None
    }
    finished = {
        int(event["task_seq"]): int(event["ts_ns"])
        for event in events
        if event.get("kind") == "kernel_finish" and event.get("task_seq") is not None
    }
    queue_ms = [
        (started[seq] - submitted[seq]) / 1_000_000.0 for seq in started.keys() & submitted.keys()
    ]
    run_ms = [
        (finished[seq] - started[seq]) / 1_000_000.0 for seq in finished.keys() & started.keys()
    ]
    drains = [
        float(event.get("elapsed_ms", 0.0) or 0.0)
        for event in events
        if event.get("kind") == "bridge_drain"
    ]
    acknowledgements = sorted(
        int(event["ts_ns"]) for event in events if event.get("kind") == "backend_ack"
    )
    input_to_ack_ms = []
    for event in events:
        if event.get("kind") != "input" or event.get("action") != "phase_start":
            continue
        start_ns = int(event["ts_ns"])
        index = bisect_left(acknowledgements, start_ns)
        if index < len(acknowledgements):
            input_to_ack_ms.append((acknowledgements[index] - start_ns) / 1_000_000.0)
    outcomes = Counter(
        str(event.get("outcome", "unknown"))
        for event in events
        if event.get("kind") == "kernel_finish"
    )
    return {
        "event_count": len(events),
        "event_kinds": dict(sorted(counts.items())),
        "kernel_finish_outcomes": dict(sorted(outcomes.items())),
        "kernel_queue_ms": _distribution(queue_ms),
        "kernel_run_ms": _distribution(run_ms),
        "bridge_drain_ms": _distribution(drains),
        "input_to_first_ack_ms": _distribution(input_to_ack_ms),
    }


def _distribution(values) -> dict[str, float | int | None]:
    data = np.asarray(tuple(values), dtype=np.float64)
    if not data.size:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": int(data.size),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "max": float(np.max(data)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Summarize an ArrayScope event trace")
    parser.add_argument("trace")
    args = parser.parse_args(argv)
    print(json.dumps(analyze_trace_latency(args.trace), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
