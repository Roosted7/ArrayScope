"""The frame-cadence recorder must tell the two failure modes apart.

Phase 1 of ``docs/redesign/wgpu-frame-pacing-2026-07-21.md`` claims the
screen path matches the display's frequency while ignoring its phase.  That
claim is only worth acting on if the instrument can distinguish:

* a pacer that is drifting (interval slightly wrong — dossier defect 1),
* a pacer that is scattered but not drifting (defect 2),
* a pacer that is genuinely locked (the target state),

on synthetic sequences where the answer is known.  An instrument that
reported "unlocked" for a locked pacer would send phase 3 chasing a defect
that is not there, so the locked case is asserted as hard as the broken
ones.
"""

from __future__ import annotations

import math
import random

import pytest

from arrayscope.display.backends.wgpu.frame_timing import FrameTimingRecorder

PERIOD_MS = 1000.0 / 60.0
PERIOD_S = PERIOD_MS / 1000.0


def _present_series(recorder: FrameTimingRecorder, timestamps) -> None:
    for timestamp in timestamps:
        recorder.note_presented(timestamp, acquire_ms=0.1, present_ms=0.05)


def test_free_running_integer_ms_pacer_reads_as_unlocked_and_drifting():
    """The exact shape of the shipped pacer: 60 Hz rounded to whole ms.

    ``round(16.67) == 17`` per frame is a third of a millisecond of error
    that the old scheduler folded into the next deadline instead of
    correcting against an anchor, so the phase walks a whole period in
    under a second.
    """

    recorder = FrameTimingRecorder()
    step = round(PERIOD_MS) / 1000.0
    _present_series(recorder, [i * step for i in range(300)])

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    # Phases cover the cycle uniformly: no relationship to scan-out.
    assert snapshot["phase_lock_r"] < 0.05
    # ~0.333 ms lost per 16.67 ms frame -> ~20 ms of phase per second.
    assert snapshot["phase_drift_ms_per_s"] == pytest.approx(19.6, abs=1.0)
    # And yet the frame rate looks essentially perfect, which is the trap.
    assert snapshot["interval_ms_p50"] == pytest.approx(17.0, abs=0.01)


def test_locked_pacer_reads_as_locked():
    """A pacer holding a fixed offset into each period, with sub-ms jitter."""

    recorder = FrameTimingRecorder()
    jitter = random.Random(11)
    _present_series(
        recorder,
        [i * PERIOD_S + jitter.gauss(0.0, 0.0003) for i in range(300)],
    )

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert snapshot["phase_lock_r"] > 0.9
    assert snapshot["phase_spread_ms"] < 1.0
    assert abs(snapshot["phase_drift_ms_per_s"]) < 1.0


def test_a_scattered_pacer_reports_its_drift_as_unresolved():
    """Defect 2 alone: right average interval, wrong moment every time.

    Drift cannot be measured here and the instrument must say so rather
    than answer confidently.  With advances spread across half a period the
    median carries ~0.3 ms of standard error, which rescaling to per-second
    multiplies into tens of ms/s of noise — so this asserts on the spread,
    which is the honest signal, and deliberately does NOT assert that drift
    is near zero.  ``R`` alone cannot make this call: it is ~0 for both a
    scattered pacer and a steadily drifting one.
    """

    recorder = FrameTimingRecorder()
    jitter = random.Random(3)
    _present_series(
        recorder,
        [i * PERIOD_S + jitter.uniform(-PERIOD_S / 2.0, PERIOD_S / 2.0) for i in range(400)],
    )

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert snapshot["phase_lock_r"] < 0.3
    # Advances fill a large fraction of the period: drift is not resolvable.
    assert snapshot["phase_advance_spread_ms"] > PERIOD_MS / 4.0


def test_a_drifting_pacer_reports_its_drift_as_resolved():
    """The companion to the scattered case, and the reason it is separate.

    Identical ``R`` (~0), opposite verdict on whether the drift number
    means anything: every frame advances the phase by the same amount, so
    the spread collapses and the drift is trustworthy.
    """

    recorder = FrameTimingRecorder()
    step = round(PERIOD_MS) / 1000.0
    _present_series(recorder, [i * step for i in range(300)])

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert snapshot["phase_lock_r"] < 0.05
    assert snapshot["phase_advance_spread_ms"] < 0.1
    assert snapshot["phase_drift_ms_per_s"] == pytest.approx(19.6, abs=1.0)


def test_a_dropped_frame_is_not_reported_as_drift():
    """Missing a vblank costs a frame, but it does not move the phase.

    An interval of two periods lands on the same point in the refresh
    cycle.  Counting it as a full period of movement would make any hitchy
    but correctly-phased pacer look like it was drifting badly.
    """

    recorder = FrameTimingRecorder()
    timestamps = []
    timestamp = 0.0
    for index in range(200):
        timestamp += PERIOD_S * (2.0 if index % 20 == 0 else 1.0)
        timestamps.append(timestamp)
    _present_series(recorder, timestamps)

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert abs(snapshot["phase_drift_ms_per_s"]) < 0.5
    assert snapshot["phase_lock_r"] > 0.9
    # The dropped frames are still visible where they belong: the tail.
    assert snapshot["interval_ms_max"] == pytest.approx(2.0 * PERIOD_MS, abs=0.01)


def test_schedule_slip_measures_the_gap_from_intended_to_actual_wakeup():
    recorder = FrameTimingRecorder()
    recorder.note_scheduled(100.0)
    recorder.note_draw_started(100.004)  # 4 ms late
    recorder.note_presented(100.006, acquire_ms=0.2, present_ms=0.1)

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert snapshot["schedule_slip_ms_p50"] == pytest.approx(4.0, abs=0.01)
    # Frame cost is draw start -> present edge, not the slip before it.
    assert snapshot["frame_ms_p50"] == pytest.approx(2.0, abs=0.01)


def test_an_unscheduled_draw_records_no_slip():
    """Expose/resize draws bypass the pacer; they must not fake a slip of 0."""

    recorder = FrameTimingRecorder()
    recorder.note_draw_started(50.0)
    recorder.note_presented(50.002, acquire_ms=0.1, present_ms=0.05)

    assert recorder.snapshot(period_ms=PERIOD_MS)["schedule_slip_ms_max"] == 0.0
    assert recorder.snapshot(period_ms=PERIOD_MS)["frames_observed"] == 1


def test_history_is_bounded_but_the_total_is_not():
    recorder = FrameTimingRecorder(capacity=16)
    _present_series(recorder, [i * PERIOD_S for i in range(200)])

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert snapshot["window_frames"] == 16
    assert snapshot["frames_observed"] == 200


def test_reset_drops_history_and_any_half_recorded_frame():
    recorder = FrameTimingRecorder()
    _present_series(recorder, [i * PERIOD_S for i in range(10)])
    recorder.note_scheduled(1.0)
    recorder.note_draw_started(1.001)

    recorder.reset()
    # A present arriving after the reset must not be paired with the draw
    # that was in flight across it, or frame_ms would be a fiction.
    recorder.note_presented(99.0, acquire_ms=0.1, present_ms=0.05)

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)
    assert snapshot["window_frames"] == 1
    assert snapshot["frame_ms_max"] == 0.0


def test_empty_and_degenerate_inputs_do_not_explode():
    recorder = FrameTimingRecorder()
    empty = recorder.snapshot(period_ms=PERIOD_MS)
    assert empty["frames_observed"] == 0
    assert empty["phase_lock_r"] == 0.0
    assert empty["interval_ms_p95"] == 0.0

    # A zero/negative period is never legitimate, but a diagnostics read
    # must not be the thing that raises.
    _present_series(recorder, [i * PERIOD_S for i in range(5)])
    degenerate = recorder.snapshot(period_ms=0.0)
    assert degenerate["phase_lock_r"] == 0.0
    assert degenerate["phase_spread_ms"] == 0.0


def test_percentiles_interpolate_over_the_window():
    recorder = FrameTimingRecorder()
    # Intervals of exactly 1..100 ms, so the percentiles are arithmetic.
    timestamp = 0.0
    recorder.note_presented(timestamp, acquire_ms=0.0, present_ms=0.0)
    for step_ms in range(1, 101):
        timestamp += step_ms / 1000.0
        recorder.note_presented(timestamp, acquire_ms=0.0, present_ms=0.0)

    snapshot = recorder.snapshot(period_ms=PERIOD_MS)

    assert snapshot["interval_ms_p50"] == pytest.approx(50.5, abs=0.01)
    assert snapshot["interval_ms_p95"] == pytest.approx(95.05, abs=0.05)
    assert snapshot["interval_ms_max"] == pytest.approx(100.0, abs=0.01)


def test_phase_spread_saturates_rather_than_diverging():
    """``R`` near zero implies an unbounded circular deviation.

    Reporting ``inf`` milliseconds would poison any JSONL artifact that
    carries it, so the spread saturates at the quarter period where the
    "how unlocked" gradient has stopped meaning anything.
    """

    recorder = FrameTimingRecorder()
    jitter = random.Random(5)
    _present_series(recorder, [jitter.uniform(0.0, 10.0) for _ in range(500)])

    spread = recorder.snapshot(period_ms=PERIOD_MS)["phase_spread_ms"]
    assert math.isfinite(spread)
    assert spread == pytest.approx(PERIOD_MS / 4.0, abs=0.01)
