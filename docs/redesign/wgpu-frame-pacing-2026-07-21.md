# wgpu screen present — frame pacing has frequency but no phase (2026-07-21)

**Status: the premise below is REFUTED as stated, and redirected.** Phase 1
landed (`c67a4730`); phase 2 measured it on real hardware and found the
pacer is not the binding constraint. **Do not implement phases 3–4.** The
measurement and the verdict are in "Phase 2 result" near the end; everything
between here and there is the original reasoning, preserved because the
refutation only makes sense against it.

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

## Phase 2 result (2026-07-21) — refuted, and redirected at the event loop

Three paired runs, headless Weston, Intel iGPU, current tip
(`6bb0a321`, i.e. after the vertex-shader pan fix), wgpu screen present,
Mailbox. Artifacts: `tests/artifacts/frame-pacing-baseline-2026-07-21/`
(gitignored; `baseline-run{1,2,3}.jsonl`).

Medians of three, with the per-run spread where it matters:

| metric | `montage_scroll_scalar` | `montage_zoompan_scalar` |
|---|---:|---:|
| interval p50 | **40.2 ms** (37–59) | **63.6 ms** (60–66) |
| interval p95 | 330.6 ms | 320.5 ms |
| **schedule slip p50** | **12.7 ms** (7.5–14.1) | **18.4 ms** (17.9–21.7) |
| schedule slip p95 | 70.4 ms | 125.9 ms |
| frame cost p50 | 1.8 ms | 1.3 ms |
| acquire p50 | 0.13 ms | 0.13 ms |
| phase R | 0.06 | 0.09 |
| advance spread | 8.4 ms | 7.9 ms |
| drift ms/s | 47.6 / 58.4 / **−0.9** | 6.3 / 26.1 / 5.4 |

**1. The pacer is not what is limiting the frame rate.** The target period
is 16.67 ms; the observed interval is 40–64 ms. Frames are not arriving
late because they are mis-phased, they are arriving late because they are
not being *run*. Schedule slip — the gap between the wakeup the pacer asked
for and the one it got — is 12.7–18.4 ms at the median, i.e. **as large as
the entire refresh period**, and 70–126 ms at p95.

**2. So defect 1's fix cannot help.** An absolute-grid deadline removes
accumulated drift, but you cannot land on a vblank boundary you are already
18 ms late for; the new deadline would be missed by exactly the same
margin. Phases 3–4 as written would have been effort spent on the smallest
term.

**3. The GPU side is genuinely cheap now.** Frame cost 1.3–1.8 ms p50 and
acquire 0.13 ms confirm the vertex-shader pan fix (`62f851f5`) removed the
per-tile CPU term, and that Mailbox acquire never blocks. The remaining
cost is not in producing a frame.

**4. Drift was never resolvable here, and the instrument said so.** Advance
spread is 7.9–8.4 ms against a quarter-period threshold of 4.17 ms, so the
drift column should be read as noise — and the runs prove it: scroll drift
came out 47.6, 58.4, and **−0.9 ms/s** on identical code, a sign flip. Had
phase 1 shipped drift without `phase_advance_spread_ms` beside it, run 1
alone would have "confirmed" a 47 ms/s drift and sent phase 3 chasing it.
This is the one design decision from phase 1 that paid for itself
immediately.

### What this redirects to

The dominant term is **event-loop occupancy**: the Qt event loop is not
free to run the paced draw when it is due. That is the same root as the
standing `gui_callbacks_below_50ms` / `event_loop_heartbeat` reds this
workload already carries, which means frame pacing was a symptom-level
reading of a scheduling problem — precisely the mistake the graveyard rows
below warn about. The next question is *what occupies the loop between
draws*, answered with a callback-attribution profile rather than a pacer.

### Limits of this measurement

- One workload family (montage scalar scroll/zoompan) on one machine.
  These stages mix cold fill with interaction, so some of the slip is fill
  work rather than steady-state interaction cost. A settled-montage,
  pure-interaction case has NOT been measured and could look different.
- Mailbox only. Under Mailbox there is no vblank clock at all, so `phase R`
  here measures our own emission times against an assumed period — it
  cannot distinguish "unlocked" from "unlockable". This is a limit of the
  isolation constraint, not a defect in it.
- Captured before the pan sweep was changed to drive a real pointer
  gesture; not comparable with runs taken after that lands.

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
| 2 | Baseline capture on the current tip | Recorded drift + advance spread under `tests/artifacts/`; **the claim is confirmed or the dossier is closed as refuted** | **DONE — REFUTED.** See "Phase 2 result" |
| 3 | Absolute-schedule pacer, no-lock fallback, stall guard | — | **CANCELLED** by phase 2: slip ≫ period, so the fix cannot reach the defect |
| 4 | Paired re-measurement, ≥3 runs/cell | — | **CANCELLED** with phase 3 |

The Fifo/clock-source question was gated on phase 2 finding residual
misalignment worth chasing. It did not, so that dossier is **not opened**:
there is no point acquiring a better clock for a draw that cannot be
scheduled on time.

Phase 1 stands on its own regardless — the instrumentation is what produced
this verdict, and it stays as the readout for whatever addresses the event
loop.

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
