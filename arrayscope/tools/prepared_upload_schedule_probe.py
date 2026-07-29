"""Did preparing ahead delay the pixels a round was waiting for?

Preparation runs on the same worker pool as the tasks that produce pixels. It
is submitted at ``Priority.PREFETCH`` so it sorts behind them in the ready
heap — but priority only orders *selection*. Once a preparation task has been
handed to a worker, that worker is gone until the task returns, and no priority
or lane quota can take it back. If every worker is inside a preparation while a
visible producer sits ready, the producer waits, and a round whose paint order
is supposed to run outward from the viewport centre can be reordered by which
worker happened to free up first.

This probe answers whether that actually happened, from a recorded trace and
nothing else. The kernel already emits ``kernel_submit`` / ``kernel_start`` /
``kernel_finish`` with lane, priority, tile and task sequence, so the evidence
is a replay, not new instrumentation on the commit path.

The number to read is **blocking_worker_ms**: worker-time during which a
preparation was running while at least one outranking visible task had been
submitted and had not yet started. That is head-of-line blocking, and it is the
only way preparation can affect paint order. Zero means preparation never held
a worker a producer wanted, whatever else the trace shows.

It is a *union* over the producers waiting at each instant, deliberately. One
preparation with three producers queued behind it occupies one worker, not
three. An earlier version of this probe added the overlap once per waiting
producer and called the total worker time, so a single 10 ms task that two
producers were waiting behind reported 20 ms. Worker-time is still summed
*across* preparations, because two preparations running at once genuinely do
hold two threads.

**producer_wait_overlap_ms** is the other quantity, and it is the per-producer
sum on purpose: it answers "how much collective waiting coincided with
preparation", which is a fairness measure rather than a resource measure. The
two questions are different and their answers coincide only when at most one
producer is ever waiting at a time.

``--ack`` additionally reports the acknowledgement order against the tile
priority order the round asked for, so a reordering can be attributed to (or
cleared of) the blocking the same run recorded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field

# The lane preparation is submitted on. Tasks on other lanes are producers as
# far as this probe is concerned: it is deliberately blind to *what* they make.
PREPARATION_LANE = "display_preparation"
PREPARATION_KEY_PREFIX = "prepared-upload"


@dataclass
class _Task:
    """One task's life, assembled from its three kernel events."""

    seq: int
    lane: str = ""
    priority: int = -1
    tile: int = -1
    visible: bool = False
    key: object = None
    submit_ns: int = -1
    start_ns: int = -1
    finish_ns: int = -1
    outcome: str = ""

    @property
    def is_preparation(self) -> bool:
        """Identified by task key, never by lane.

        ``DISPLAY_PREPARATION`` is not a private lane: the LOD ladder submits
        ordinary rung work on it too. Classifying by lane counted those as
        preparations and — worse — removed them from the producer population
        they belong to, which is how an early run of this probe reported
        preparations on a build that has none.
        """

        key = self.key
        return isinstance(key, (list, tuple)) and bool(key) and key[0] == PREPARATION_KEY_PREFIX

    @property
    def ran(self) -> bool:
        return self.start_ns >= 0 and self.finish_ns >= 0


def _union_ms(intervals) -> float:
    """Wall time covered by a set of possibly overlapping ``(low, high)`` ns pairs."""

    spans = sorted((low, high) for low, high in intervals if high > low)
    if not spans:
        return 0.0
    total = 0
    current_low, current_high = spans[0]
    for low, high in spans[1:]:
        if low > current_high:
            total += current_high - current_low
            current_low, current_high = low, high
        else:
            current_high = max(current_high, high)
    total += current_high - current_low
    return total / 1e6


def _intersect(first, second):
    """Clip ``second`` against ``first``; both are ``(low, high)`` ns pairs."""

    low = max(first[0], second[0])
    high = min(first[1], second[1])
    return (low, high) if high > low else None


@dataclass
class ScheduleEvidence:
    """What the trace says about preparation's effect on producer start times."""

    preparations_submitted: int = 0
    preparations_started: int = 0
    preparations_dropped: int = 0
    preparation_outcomes: dict[str, int] = field(default_factory=dict)
    preparation_busy_ms: float = 0.0
    # The resource number: worker time held away from producers. Unioned within
    # each preparation, summed across them, because concurrent preparations
    # really do occupy separate threads.
    blocking_worker_ms: float = 0.0
    # The fairness number: collective producer waiting that coincided with a
    # running preparation. Summed per producer, so it may exceed the above.
    producer_wait_overlap_ms: float = 0.0
    blocking_intervals: int = 0
    worst_producer_delay_ms: float = 0.0
    delayed_producers: int = 0
    producers_started: int = 0

    def as_dict(self) -> dict:
        return {
            "preparations_submitted": self.preparations_submitted,
            "preparations_started": self.preparations_started,
            "preparations_dropped": self.preparations_dropped,
            "preparation_outcomes": dict(sorted(self.preparation_outcomes.items())),
            "preparation_busy_ms": round(self.preparation_busy_ms, 3),
            "blocking_worker_ms": round(self.blocking_worker_ms, 3),
            "producer_wait_overlap_ms": round(self.producer_wait_overlap_ms, 3),
            "blocking_intervals": self.blocking_intervals,
            "worst_producer_delay_ms": round(self.worst_producer_delay_ms, 3),
            "delayed_producers": self.delayed_producers,
            "producers_started": self.producers_started,
        }


def _load_tasks(events) -> dict[int, _Task]:
    tasks: dict[int, _Task] = {}
    for event in events:
        kind = event.get("kind")
        if kind not in ("kernel_submit", "kernel_start", "kernel_finish"):
            continue
        seq = int(event.get("task_seq", -1))
        if seq < 0:
            continue
        task = tasks.get(seq)
        if task is None:
            task = _Task(seq=seq)
            tasks[seq] = task
        ts = int(event.get("ts_ns", 0) or 0)
        if kind == "kernel_submit":
            task.submit_ns = ts
            task.lane = str(event.get("lane", "") or "")
            task.priority = int(event.get("priority", -1))
            task.tile = int(event.get("tile_number", -1))
            task.visible = bool(event.get("visible", False))
            task.key = event.get("key")
        elif kind == "kernel_start":
            task.start_ns = ts
            if not task.lane:
                task.lane = str(event.get("lane", "") or "")
            if task.priority < 0:
                task.priority = int(event.get("priority", -1))
            if task.tile < 0:
                task.tile = int(event.get("tile_number", -1))
        else:
            task.finish_ns = ts
            task.outcome = str(event.get("outcome", "") or "")
    return tasks


def analyze_schedule(events) -> ScheduleEvidence:
    """Head-of-line blocking charged to preparation, from kernel events alone."""

    tasks = _load_tasks(events)
    preparations = [task for task in tasks.values() if task.is_preparation]
    producers = [
        task
        for task in tasks.values()
        if not task.is_preparation and task.visible and task.submit_ns >= 0
    ]
    evidence = ScheduleEvidence(
        preparations_submitted=len(preparations),
        preparations_started=sum(1 for task in preparations if task.start_ns >= 0),
        preparations_dropped=sum(1 for task in preparations if task.start_ns < 0),
        producers_started=sum(1 for task in producers if task.start_ns >= 0),
    )
    for task in preparations:
        if task.outcome:
            evidence.preparation_outcomes[task.outcome] = (
                evidence.preparation_outcomes.get(task.outcome, 0) + 1
            )
    running = [task for task in preparations if task.ran]
    for task in running:
        evidence.preparation_busy_ms += (task.finish_ns - task.start_ns) / 1e6

    # A producer is "waiting" over [submit, start). Preparation blocks it over
    # whatever part of that window a preparation task was actually running, and
    # only when the producer outranks it — equal or worse priority would not
    # have been selected first anyway.
    waits = [
        (task.submit_ns, task.start_ns, task)
        for task in producers
        if task.start_ns > task.submit_ns >= 0
    ]
    for prep in running:
        window = (prep.start_ns, prep.finish_ns)
        overlaps = [
            clipped
            for wait_start, wait_end, producer in waits
            if producer.priority < prep.priority
            and (clipped := _intersect(window, (wait_start, wait_end))) is not None
        ]
        if not overlaps:
            continue
        evidence.blocking_intervals += 1
        # One worker, however many producers happened to be queued behind it.
        evidence.blocking_worker_ms += _union_ms(overlaps)
        evidence.producer_wait_overlap_ms += sum(high - low for low, high in overlaps) / 1e6

    per_producer: dict[int, float] = defaultdict(float)
    for wait_start, wait_end, producer in waits:
        # Unioned per producer as well: two preparations overlapping the same
        # wait at the same instant did not delay that producer twice.
        blocked = [
            clipped
            for prep in running
            if producer.priority < prep.priority
            and (clipped := _intersect((prep.start_ns, prep.finish_ns), (wait_start, wait_end)))
            is not None
        ]
        if blocked:
            per_producer[producer.seq] = _union_ms(blocked)
    if per_producer:
        evidence.delayed_producers = len(per_producer)
        evidence.worst_producer_delay_ms = max(per_producer.values())
    return evidence


def acknowledgement_report(events) -> dict:
    """First-ack order and the tiles the round asked to paint first."""

    order: list[int] = []
    seen: set[int] = set()
    for event in events:
        if event.get("kind") != "backend_ack" or not bool(event.get("accepted", False)):
            continue
        tile = int(event.get("tile", -1))
        if tile < 0 or tile in seen:
            continue
        seen.add(tile)
        order.append(tile)
    return {"acknowledgement_order": tuple(order)}


def read_events(path: str):
    events = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Charge producer start-time delay to prepared-upload tasks"
    )
    parser.add_argument("trace")
    parser.add_argument("--ack", action="store_true", help="also report acknowledgement order")
    args = parser.parse_args(argv)
    events = read_events(args.trace)
    report = analyze_schedule(events).as_dict()
    if args.ack:
        report.update({key: list(value) for key, value in acknowledgement_report(events).items()})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())


__all__ = ["ScheduleEvidence", "acknowledgement_report", "analyze_schedule", "read_events"]
