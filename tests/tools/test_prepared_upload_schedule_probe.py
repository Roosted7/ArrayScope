"""Ring 1. Charging producer delay to preparation, from kernel events alone.

The probe exists to answer one question with evidence instead of reasoning:
did preparing ahead hold a worker that a pixel-producing task was waiting for?
Its first version answered it wrongly, by classifying tasks on their lane —
`DISPLAY_PREPARATION` is not private to the hand-off, the LOD ladder submits
ordinary rung work on it — so it reported preparations on a build that has
none, and removed real producers from the population they belong to. That
misclassification is pinned here.
"""

from __future__ import annotations

from arrayscope.tools.prepared_upload_schedule_probe import (
    acknowledgement_report,
    analyze_schedule,
)

MS = 1_000_000


def _submit(seq, *, key, lane, priority, ts, visible=True, tile=-1):
    return {
        "kind": "kernel_submit",
        "task_seq": seq,
        "key": key,
        "lane": lane,
        "priority": priority,
        "visible": visible,
        "tile_number": tile,
        "ts_ns": ts,
    }


def _start(seq, *, ts):
    return {"kind": "kernel_start", "task_seq": seq, "ts_ns": ts}


def _finish(seq, *, ts, outcome="completed"):
    return {"kind": "kernel_finish", "task_seq": seq, "ts_ns": ts, "outcome": outcome}


def _preparation(seq, *, submit, start, finish, tile=0):
    return [
        _submit(
            seq,
            key=["prepared-upload", 1, tile, ["identity"]],
            lane="speculative_residency",
            priority=80,
            visible=False,
            tile=tile,
            ts=submit,
        ),
        _start(seq, ts=start),
        _finish(seq, ts=finish),
    ]


def _producer(seq, *, submit, start, finish, tile=0):
    return [
        _submit(
            seq,
            key=["rung", 1, tile],
            lane="visible_materialization",
            priority=10,
            visible=True,
            tile=tile,
            ts=submit,
        ),
        _start(seq, ts=start),
        _finish(seq, ts=finish),
    ]


def test_a_preparation_running_across_a_producers_wait_is_charged():
    """The producer was ready at 1 ms and did not start until 11 ms."""

    events = [
        *_preparation(1, submit=0, start=1 * MS, finish=11 * MS),
        *_producer(2, submit=1 * MS, start=11 * MS, finish=12 * MS),
    ]

    evidence = analyze_schedule(events)

    assert evidence.preparations_started == 1
    assert evidence.blocking_worker_ms == 10.0
    assert evidence.producer_wait_overlap_ms == 10.0
    assert evidence.delayed_producers == 1
    assert evidence.worst_producer_delay_ms == 10.0


def test_a_preparation_that_finishes_before_the_producer_is_ready_is_not_charged():
    events = [
        *_preparation(1, submit=0, start=0, finish=5 * MS),
        *_producer(2, submit=6 * MS, start=7 * MS, finish=8 * MS),
    ]

    evidence = analyze_schedule(events)

    assert evidence.preparations_started == 1
    assert evidence.blocking_worker_ms == 0.0
    assert evidence.delayed_producers == 0


def test_ladder_work_on_the_preparation_lane_is_a_producer_not_a_preparation():
    """The misclassification that made an unrelated build look instrumented.

    This task sits on `display_preparation` and is visible rung work. Counting
    it as a preparation invented hand-off activity where there was none; it
    must instead be part of the producer population that preparation can delay.
    """

    events = [
        _submit(
            1,
            key=["rung-step", 7, 3],
            lane="display_preparation",
            priority=10,
            visible=True,
            tile=3,
            ts=0,
        ),
        _start(1, ts=1 * MS),
        _finish(1, ts=2 * MS),
    ]

    evidence = analyze_schedule(events)

    assert evidence.preparations_submitted == 0
    assert evidence.producers_started == 1


def test_a_preparation_dropped_before_running_costs_no_worker_time():
    """Submitted and superseded is free; only a started task holds a thread."""

    events = [
        _submit(
            1,
            key=["prepared-upload", 1, 0, ["identity"]],
            lane="speculative_residency",
            priority=80,
            visible=False,
            ts=0,
        ),
        *_producer(2, submit=1 * MS, start=9 * MS, finish=10 * MS),
    ]

    evidence = analyze_schedule(events)

    assert evidence.preparations_submitted == 1
    assert evidence.preparations_dropped == 1
    assert evidence.blocking_worker_ms == 0.0


def test_a_producer_that_does_not_outrank_the_preparation_is_not_charged():
    """Only work the scheduler would genuinely have picked first counts."""

    events = [
        *_preparation(1, submit=0, start=0, finish=10 * MS),
        _submit(
            2,
            key=["warm", 1],
            lane="speculative_residency",
            priority=80,
            visible=True,
            ts=1 * MS,
        ),
        _start(2, ts=10 * MS),
        _finish(2, ts=11 * MS),
    ]

    assert analyze_schedule(events).blocking_worker_ms == 0.0


def test_acknowledgement_order_keeps_each_tiles_first_accepted_ack():
    events = [
        {"kind": "backend_ack", "tile": 5, "accepted": True},
        {"kind": "backend_ack", "tile": 2, "accepted": False},
        {"kind": "backend_ack", "tile": 3, "accepted": True},
        {"kind": "backend_ack", "tile": 5, "accepted": True},
    ]

    assert acknowledgement_report(events)["acknowledgement_order"] == (5, 3)


def test_one_preparation_blocking_two_producers_is_one_worker_not_two():
    """Worker time is a resource, so it is unioned, not summed per waiter.

    A single 10 ms preparation with two producers queued behind it occupied one
    thread for 10 ms. Adding the overlap once per waiting producer reported
    20 ms of "worker time" that no machine ever spent — the arithmetic behind
    the first 458 ms figure quoted for this seam.
    """

    events = [
        *_preparation(1, submit=0, start=1 * MS, finish=11 * MS),
        *_producer(2, submit=1 * MS, start=11 * MS, finish=12 * MS, tile=1),
        *_producer(3, submit=1 * MS, start=11 * MS, finish=12 * MS, tile=2),
    ]

    evidence = analyze_schedule(events)

    # One worker was held for 10 ms.
    assert evidence.blocking_worker_ms == 10.0
    # Two producers each waited 10 ms behind it: 20 ms of collective waiting.
    assert evidence.producer_wait_overlap_ms == 20.0
    assert evidence.delayed_producers == 2
    assert evidence.worst_producer_delay_ms == 10.0
    # And it is one blocking episode, not two.
    assert evidence.blocking_intervals == 1


def test_two_concurrent_preparations_hold_two_workers():
    """The counter-case: across preparations, worker time really does add."""

    events = [
        *_preparation(1, submit=0, start=1 * MS, finish=11 * MS, tile=0),
        *_preparation(2, submit=0, start=1 * MS, finish=11 * MS, tile=1),
        *_producer(3, submit=1 * MS, start=11 * MS, finish=12 * MS, tile=2),
    ]

    evidence = analyze_schedule(events)

    assert evidence.blocking_worker_ms == 20.0
    # The producer still only waited 10 ms, however many threads were busy.
    assert evidence.worst_producer_delay_ms == 10.0
    assert evidence.delayed_producers == 1


def test_partially_overlapping_preparations_do_not_double_charge_one_producer():
    """Per-producer delay is unioned across preparations too."""

    events = [
        *_preparation(1, submit=0, start=1 * MS, finish=7 * MS, tile=0),
        *_preparation(2, submit=0, start=4 * MS, finish=11 * MS, tile=1),
        *_producer(3, submit=1 * MS, start=11 * MS, finish=12 * MS, tile=2),
    ]

    evidence = analyze_schedule(events)

    # The union of [1,7) and [4,11) is 10 ms of wait, not 6 + 7 = 13 ms.
    assert evidence.worst_producer_delay_ms == 10.0
    # Worker time is the sum of the two tasks' blocking spans: 6 + 7.
    assert evidence.blocking_worker_ms == 13.0
