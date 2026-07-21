# wgpu screen present — frame pacing has frequency but no phase (2026-07-21)

**Status:** phase 1 landed (`c67a4730`) — the cadence is now observable and
input latching is confirmed. Nothing is measured on real hardware yet, and
the claim below stands unadjudicated until phase 2 reads it.

**Scope:** the wgpu `present_method: screen` path only
(`arrayscope/display/backends/wgpu/screen_canvas.py`). The bitmap path and
every other backend are untouched.

**One variable (Thomas, 2026-07-21).** Present mode stays pinned at
whatever the surface already chose. Swapping Fifo/Mailbox in the same
change would confound the only thing being measured. See "The Fifo
question is now downstream" — this constraint has a real consequence for
what the pacer can achieve, and it is better stated than discovered later.

**Baseline (Thomas, 2026-07-21).** Re-baseline against
`claude/tile-panning-performance-adfc0a`, not the original tip. Per-tile
CPU cost in the montage pan path was eliminated by moving the camera
transform into the vertex shader (`62f851f5`); frame time is now ~1.4 ms at
512 tiles and flat in tile count. Pacing is therefore no longer masked by
per-frame CPU work — which raises the value of this work and also means any
pre-`62f851f5` measurement is worthless as a control.

**Relation to the queue:** feeds row 3(d) promotion evidence. This is *not*
a stall fix — see "Why this is not the pacing the graveyard rejected".

## The claim

The screen path matches the display's **frequency** and ignores its
**phase**. Frequency-matched, phase-free presentation is not frame sync: it
puts every frame at an arbitrary and slowly-drifting offset from scan-out,
which costs both smoothness and latency while reporting a perfect fps.

## What the code does today

[`screen_canvas.py:209-218`](../../arrayscope/display/backends/wgpu/screen_canvas.py#L209)
is a free-running rate limiter:

```python
interval = 1000.0 / float(self.max_draws_per_second or 30.0)
elapsed_ms = (perf_counter() - self._last_draw_started) * 1000.0
delay_ms = 0 if elapsed_ms >= interval else round(interval - elapsed_ms)
QtCore.QTimer.singleShot(delay_ms, self, self._invoke_draw)
```

Three distinct defects, none of which an fps counter can see:

1. **No absolute reference, so error integrates.** The delay is computed
   from *elapsed since the last draw began*. Every `round()` to integer ms
   and every timer oversleep folds permanently into the next deadline;
   there is no vblank anchor to correct back toward. At 16.67 ms the
   sub-millisecond rounding residue alone walks the phase through a full
   frame in seconds. That walk is the beat frequency that makes a nominally
   60 fps pan step.
2. **Phase is uniformly distributed.** Even with a perfect timer, drawing
   at the right rate at an arbitrary offset means each frame lands
   somewhere random relative to scan-out. Frames near the boundary are
   shown a refresh late; the smoothness cost is the *variance*, not the
   mean.
3. **Latency is maximal by construction.** Waking as early as the interval
   permits samples input roughly a full refresh before it can be scanned
   out. The latency-optimal wake is as late as the render budget safely
   allows.

`_last_draw_started` is stamped at draw **start**
([`:227`](../../arrayscope/display/backends/wgpu/screen_canvas.py#L227)), so
frame cost is excluded from pacing entirely — a 12 ms frame and a 2 ms frame
schedule their successors identically.

Two incidentals found while mapping: `screen.refreshRate()` is queried on
every `request_draw` with no caching and no `screenChanged` hookup, and the
`or 30.0` fallback at
[`:215`](../../arrayscope/display/backends/wgpu/screen_canvas.py#L215) is
dead code from the pinned-at-30 era (the property already guarantees ≥ 1.0).

## The load-bearing insight: Mailbox has no clock

This is why present mode cannot be a separate question.

- Under **Fifo**, `get_current_texture()` blocks once the image chain is
  full, and it unblocks when the compositor releases an image — *at
  vblank*. That block is the ~15 ms GUI-thread cost recorded as the gate-B
  tier-1 caveat and the reason
  [`_prefer_mailbox`](../../arrayscope/display/backends/wgpu/screen_canvas.py#L362)
  exists. **It is also the only vblank timestamp we get.**
- Under **Mailbox**, acquire returns immediately and present is
  fire-and-forget. There is no backpressure and no timing feedback of any
  kind. Excess frames are silently discarded by the compositor.

So the current architecture chose the mode that structurally cannot be
phase-locked, and then paced it with the only tool left — a free-running
timer. The 15 ms Fifo block was read as pure waste. It is better understood
as **a synchronization signal being consumed on the GUI thread at the worst
possible moment**.

### The Fifo question is now downstream

Holding present mode fixed (the one-variable constraint above) has a
consequence worth stating plainly rather than discovering in phase 4: **on
a Mailbox surface the pacer cannot phase-lock at all**, because there is no
clock to lock to. That splits the work into two honest halves, and the
split is cleaner than the original single change:

1. **This dossier's change — an absolute schedule.** Anchor the deadline to
   a fixed monotonic grid instead of "previous draw + interval". That kills
   defect 1 outright: rounding and oversleep stop integrating, because
   every wake is computed from the anchor rather than from its predecessor.
   It needs no vblank information at all, so it is measurable with present
   mode untouched, and it is the whole of what phases 3–4 below cover.
2. **A later, separate change — a real clock.** True phase alignment
   (defect 2, and the latency win of defect 3) needs vblank timestamps,
   which on this stack means Fifo acquire or one of the rejected sources in
   the brainstorm table. That is its own dossier, its own one variable, and
   it is only worth opening if phase 2 shows residual misalignment *after*
   the absolute schedule has removed the drift.

The instrumentation cannot tell these apart by `phase_lock_r` alone — a
drifting pacer and a scattered one both read ~0 — which is exactly why
`phase_advance_spread_ms` exists. Phase 2 reads the two together: high
drift with tight spread means (1) is the fix and is sufficient; near-zero
drift with wide spread means (1) will not help and only (2) will.

## Clock-source brainstorm

| # | Source | Verdict |
|---|---|---|
| A | **PLL over Fifo acquire-return timestamps** | **Chosen.** Zero new plumbing — `acquire_ms` is already timed at [`wgpu_imageview2d.py:668-670`](../../arrayscope/display/wgpu_imageview2d.py#L668). Works on any surface that offers Fifo, which per WebGPU/Vulkan is all of them. |
| B | `wp_presentation_feedback` (real compositor timestamps) | **Rejected for v1.** Feedback is requested per-commit *before* the commit, and wgpu owns the commit inside `wgpuSurfacePresent`. We cannot interpose without patching wgpu-py. Strictly better data if upstream ever exposes it. |
| C | A second clock `wl_surface` with `wl_surface.frame` callbacks | **Rejected.** It buys a true vblank tick at the cost of an extra subsurface. The 2026-07-20 glitch dossier is a monument to what extra subsurfaces cost this window; paying that for a clock we can estimate is a bad trade. |
| D | `VK_KHR_present_wait` / `VK_GOOGLE_display_timing` | **Not available.** Not exposed by wgpu-py 0.31.1. Revisit on upgrade — this is the principled answer if it lands. |
| E | Fifo acquire on a worker thread | **Rejected for v1.** The GUI thread owns the device and the wgpu-py surface objects are not thread-safe on this path. Large blast radius for a benefit the deadline scheduler already gets. |
| F | `QScreen.refreshRate()` alone | **Retained as the fallback tier** — this is today's behavior, and it stays the floor. |

## Design: phase-locked deadline scheduling

One timer. Ground rule 12 forbids new pacing timers, and this adds none: it
changes what the *existing* `QTimer.singleShot` targets, from an elapsed
interval to an absolute deadline.

### 1. Estimator

Seed `period` from `QScreen.refreshRate()`; seed `phase` from the first
blocking acquire return. Maintain both with a phase-locked loop over
subsequent acquire-return timestamps, median-filtered against outliers.
Reuse `_ewma` from
[`core/latency_feedback.py:108`](../../arrayscope/core/latency_feedback.py#L108)
rather than growing a second EWMA implementation (ADR 0052's supersession
already named that module the one feedback model).

Track the residual — the spread of observed returns around the predicted
vblank. Sustained high residual means **no lock**.

### 2. Deadline

Wake at `next_vblank − render_budget`, where
`render_budget = ewma(frame_cost) + (p95 − p50)` jitter margin, computed
from timings we will already be collecting.

Because each wake recomputes from an *absolute* predicted vblank rather than
from the previous draw, error stops integrating — defect 1 dies at the root.
The draw lands in a consistent pre-vblank slot — defect 2. Input is latched
as late as the budget allows — defect 3.

The self-correcting property that makes this cheap: arriving on schedule
means acquire returns fast (an image is free), and the small residual wait
is still a phase sample. The estimator is fed by the thing it optimizes.

### 3. Fallback tiers, so compatibility is by construction

The estimator may only ever **narrow** the deadline. With no lock —
headless, offscreen, VRR, remote, software compositor, occluded window,
Mailbox-only surface — the deadline falls back to exactly today's
`last_draw + interval`. **Worst case is current behavior**, which is what
makes this safe to land before the evidence is complete.

Stall guard (ground rule 11 — the wait needs an owner): the deadline clamps
to `now + 2·period`, so a compositor that stops releasing images (hidden or
occluded window) cannot wedge the draw loop. The owner of the resumption is
the timer itself, never "a later expose will pick it up".

### 4. Input latching — CONFIRMED (2026-07-21)

The deadline move only removes latency if moving the draw later also moves
the *input sample* later; if the camera were latched when a redraw is
requested, deferring the draw toward vblank would add a frame of lag rather
than remove one. This was confirmed by test rather than by reading the call
graph — `test_camera_is_latched_when_the_paced_draw_runs_not_when_it_is_requested`
in `tests/display/test_wgpu_imageview2d.py` holds the pacer's deferral
window open, moves the camera inside it, and asserts which range reached
the executor. It also pins the coalescing case: a second camera move folded
into an already-pending draw still reaches the frame, so a fast pan does
not present a trail of stale cameras.

No production change is needed here; the property is now pinned so a future
refactor cannot quietly cost the latency this design is buying.

## Instrumentation must land first

The counters at
[`wgpu_imageview2d.py:311-316`](../../arrayscope/display/wgpu_imageview2d.py#L311)
keep only last + max for acquire and present. There is **no frame-interval
metric, no percentile, and no `request_draw`→`_invoke_draw` scheduling-
latency metric**. Nothing currently in the diagnostics would show phase
drift or a missed vblank — the defect claimed at the top of this dossier
cannot presently be seen, in either direction.

So phase 1 is measurement, and it is genuinely falsifiable: if the phase
residual under Fifo is not tight, no lock is possible, alternatives B/C/D
become the only routes, and this design is abandoned rather than tuned.

Landed in `c67a4730` as `FrameTimingRecorder`
(`arrayscope/display/backends/wgpu/frame_timing.py`): interval, frame cost,
schedule slip, acquire and present distributions, plus `phase_lock_r`,
`phase_drift_ms_per_s` and `phase_advance_spread_ms`.

**One design note worth carrying forward.** The phase readout took three
attempts and the tests pin each failure. Unwrapping phase to fit a slope
manufactures ~11 ms/s of drift out of a non-drifting pacer whenever jitter
approaches half a period; summing folded per-frame advances instead
random-walks to ~21 ms/s of noise. Drift is therefore a *median* advance,
which also makes dropped frames read correctly (two periods lands on the
same phase — zero movement, not a full period of it). Even the median has a
noise floor once advances spread out, hence `phase_advance_spread_ms`
shipping beside it. **Do not read drift without the spread**; a confident
drift number over wide spread is the instrument lying.

## Measurement plan

Paired runs under the headless-Weston launcher (never the attached session
— it puts desktop activity inside the timing loop), Intel iGPU, **≥3 paired
runs per cell**: the wgpu screen-present memory records single zoompan runs
differing by >2× on identical code, so no single-run delta is adjudicable.

**One variable: the pacer.** Present mode is held at whatever the surface
already selected, and is recorded per run so a cell is never compared
across modes by accident.

| | Free-running (today) | Absolute schedule |
|---|---|---|
| **present mode as-is** | baseline | the hypothesis |

Read out: `phase_drift_ms_per_s` with `phase_advance_spread_ms` (the pair
decides which defect is live), `phase_lock_r`, frame-interval p95/max,
`schedule_slip_ms` p95, event-loop heartbeat gap, acquire p95/max, warm
input dispatch. Fast-scroll p95 is the stable headline; zoompan needs the
≥3-run discipline.

**Harness note (2026-07-21):** the pan sweep is being changed separately to
drive a real pointer gesture so chain B becomes measurable on wgpu. Any
baseline captured before that lands is not comparable with one captured
after — record which side of it every run sits on.

## Why this is not the pacing the graveyard rejected

The graveyard buries several pacing changes with "Never — fix labels/owners,
not pacing", and ground rule 11's mechanical check is that a repeating bail
with no in-flight work is a defect rather than pacing. That check does not
fire here: nothing is stalled, no wakeup is lost, no owner is missing. The
claim is narrower and mechanical — presented frames are misaligned in time
with scan-out — and phase 1 makes it measurable before phase 3 acts on it.
If the residual shows the frames are already well-aligned, the claim is
simply false and the work stops.

## Phases and exit gates

| # | Step | Exit gate | Status |
|---|---|---|---|
| 1 | Frame-cadence instrumentation (interval, schedule slip, frame cost, phase) through `wgpuPresentationDiagnostics()`; confirm input latching | Metrics visible on a real-Wayland run; no measurable frame-path cost | **DONE** `c67a4730`; latching confirmed by test |
| 2 | Baseline capture on the current tip | Recorded drift + advance spread under `tests/artifacts/`; **the claim is confirmed or the dossier is closed as refuted** | next |
| 3 | Absolute-schedule pacer, no-lock fallback, stall guard | Unit-tested on synthetic sequences: lock, no-lock, drift rejection, occlusion clamp. **Does not land without the no-lock fallback.** Existing pacing tests stay green | |
| 4 | Paired re-measurement, ≥3 runs/cell | Frame-interval p95 and phase drift improve with no event-loop or acquire regression; **journey matrix green — this does not land without it** | |

The Fifo/clock-source question is deliberately *not* a phase here; it is a
separate dossier gated on what phase 2 finds (see "The Fifo question is now
downstream").

## Open questions

- **VRR.** Adaptive sync means the period is not fixed. Expected to surface
  as sustained no-lock, i.e. degrade to today's behavior — but this machine
  cannot exercise it, so it stays a caveat rather than a claim.
- **Multi-monitor.** Dragging between panels of different rates should
  reseed the estimator; the missing `screenChanged` hookup is the natural
  place, and is worth fixing in phase 1 regardless.
- **`QTimer.singleShot` defaults to `Qt.CoarseTimer.`** For deadline
  scheduling this should be `PreciseTimer`. Cheap, but it changes timing —
  it belongs in phase 3 with the rest of the pacing change, not smuggled
  into phase 1's measurement.
- **The resize-coalescing spill trade** (16.8 ms mean / 29.0 ms max,
  [`:253-262`](../../arrayscope/display/backends/wgpu/screen_canvas.py#L253))
  is still unadjudicated on pixels and interacts with any pacing change.
  Out of scope here; flagged so a phase-4 regression is not misattributed.
