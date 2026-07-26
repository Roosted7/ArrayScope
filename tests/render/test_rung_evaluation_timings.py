"""Per-(rung, level) evaluation accounting — pure bookkeeping.

This counter exists because wall clock cannot settle the question it answers:
the reference machine's raw montage stage spreads 4.0-4.9 s run to run, so
"is reduced-input evaluation cheaper, and by how much" has to be counted
directly per rung and per level.
"""

from __future__ import annotations

import threading

from arrayscope.render.ladder import Rung
from arrayscope.render.stages import RungEvaluationTimings


def test_no_evaluations_reports_no_rows():
    # Never a row of zeros: a permanently-zero diagnostic reads as evidence.
    assert RungEvaluationTimings().rows() == ()


def test_calls_and_time_accumulate_per_rung_and_level():
    timings = RungEvaluationTimings()
    timings.record(Rung.FLOOR, 4, 2_000_000)
    timings.record(Rung.FLOOR, 4, 6_000_000)
    timings.record(Rung.DESIRED, 2, 1_500_000)

    assert timings.rows() == (
        {
            "rung": 0,
            "rung_name": "floor",
            "level": 4,
            "calls": 2,
            "discarded": 0,
            "total_ms": 8.0,
            "max_ms": 6.0,
        },
        {
            "rung": 2,
            "rung_name": "desired",
            "level": 2,
            "calls": 1,
            "discarded": 0,
            "total_ms": 1.5,
            "max_ms": 1.5,
        },
    )


def test_same_rung_at_different_levels_stays_separate():
    # The whole point: a rung's cost is a function of the level it ran at.
    timings = RungEvaluationTimings()
    timings.record(Rung.DESIRED, 2, 4_000_000)
    timings.record(Rung.DESIRED, 0, 40_000_000)

    rows = {(int(row["rung"]), int(row["level"])): row for row in timings.rows()}
    assert set(rows) == {(2, 2), (2, 0)}
    assert rows[(2, 0)]["total_ms"] == 40.0
    assert rows[(2, 2)]["total_ms"] == 4.0


def test_rows_are_ordered_coarse_rung_first():
    timings = RungEvaluationTimings()
    for rung in (Rung.EXACT, Rung.DESIRED, Rung.PREVIEW, Rung.FLOOR):
        timings.record(rung, 1, 1)

    assert [int(row["rung"]) for row in timings.rows()] == [0, 1, 2, 3]


def test_unknown_rung_value_degrades_to_its_number_instead_of_raising():
    timings = RungEvaluationTimings()
    timings.record(99, 3, 1_000_000)

    (row,) = timings.rows()
    assert row["rung"] == 99
    assert row["rung_name"] == "99"


def test_negative_elapsed_never_produces_negative_totals():
    # A non-monotonic clock reading must not corrupt a cumulative counter.
    timings = RungEvaluationTimings()
    timings.record(Rung.FLOOR, 4, -5)

    (row,) = timings.rows()
    assert row["calls"] == 1
    assert row["total_ms"] == 0.0


def test_concurrent_recording_loses_no_calls():
    """Worker threads record; the counter must not drop or double-count."""

    timings = RungEvaluationTimings()
    start = threading.Barrier(4)

    def worker(rung: int) -> None:
        start.wait()
        for _ in range(500):
            timings.record(rung, 2, 1_000)

    threads = [threading.Thread(target=worker, args=(rung,)) for rung in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = timings.rows()
    assert len(rows) == 4
    assert all(int(row["calls"]) == 500 for row in rows)
    assert all(row["total_ms"] == 0.5 for row in rows)


def test_discarded_evaluations_are_counted_beside_the_cost_that_spent_them():
    """Spent-and-thrown-away has to be separable from spent-and-used.

    Measured on the FFT montage: 8 level-1 evaluations at a 1041 ms mean, all
    discarded, inside a 5.5 s stage. The aggregate time alone cannot say that.
    """

    timings = RungEvaluationTimings()
    for _ in range(8):
        timings.record(Rung.DESIRED, 1, 1_040_000_000)
        timings.record_discarded(Rung.DESIRED, 1)
    timings.record(Rung.DESIRED, 2, 20_000_000)

    rows = {(int(row["rung"]), int(row["level"])): row for row in timings.rows()}
    assert rows[(2, 1)]["calls"] == 8
    assert rows[(2, 1)]["discarded"] == 8
    # Priced from the row itself: discarded x mean.
    mean_ms = float(rows[(2, 1)]["total_ms"]) / int(rows[(2, 1)]["calls"])
    assert int(rows[(2, 1)]["discarded"]) * mean_ms == 8320.0
    assert rows[(2, 2)]["discarded"] == 0


def test_a_discard_with_no_timed_call_still_reports_a_row():
    """A discard must never be silently dropped for lack of a timing partner."""

    timings = RungEvaluationTimings()
    timings.record_discarded(Rung.PREVIEW, 4)

    (row,) = timings.rows()
    assert (row["rung"], row["level"], row["calls"], row["discarded"]) == (1, 4, 0, 1)
    assert row["total_ms"] == 0.0
