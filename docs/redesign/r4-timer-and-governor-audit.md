# R4 — Timer and governor audit (no fixed timers, no phantom pacing)

**Goal:** every remaining `QTimer` is justified in one of three categories
or deleted; the resource governor shrinks to telemetry plus two knobs.

## Timer audit

Inventory (28 `QTimer(` constructions pre-redesign; regenerate with
`grep -rn "QTimer(\|singleShot" arrayscope --include='*.py'`). For each,
classify and act:

| category | rule | examples |
|---|---|---|
| **anti-hang fallback** | keep; must back off when idle and count its polls (bridge pattern) | kernel bridge fallback; the ONE watchdog assertion probe |
| **UI cosmetic** | keep; must not gate data flow | toast duration, slow-work overlay delay, double-click windows |
| **scheduling** | DELETE; replace with kernel completion events, deps, or capacity waiters | commit interval timers, stage-wait pumps, viewport update deferrals, priority retarget delays, drain fallbacks in per-controller copies, governor sampling reapply-on-edge |

R2 deletes most scheduling timers with their clusters; this plan sweeps the
rest: `render_coordinator.py` coalescing (keep — input coalescing at 16 ms
is interaction shaping, not work pacing; document it as cosmetic),
`montage_prefetch.py`, `render_prefetch.py`, `histogram_controller.py`,
`sync/bus.py` reconnect timers (anti-hang: justify + bound), and every
`singleShot` without a receiver context (the architecture guard already
forbids new ones).

Acceptance per file: a one-line comment on each surviving timer naming its
category, and an architecture test extension
(`tests/app/test_architecture_guards.py`) that fails on new `QTimer`
constructions outside an allowlist.

## Governor rescope (ADR 0052 superseded in part)

The governor's job collapses to: sample telemetry, then set

1. the bridge drain budget (`QtKernelBridge.set_budget_ms` /
   `set_max_items_per_drain`), and
2. kernel lane quotas (`set_lane_quota`) + commit batch bounds
   (`CommitBatch.max_items/max_bytes`).

Delete: per-controller worker clamps, per-channel `ui_work_decision`
plumbing through window (`_ui_work_decision`, work signatures,
conservative starts), decision rings in benchmark JSONL, and
`interact-edge reapplication` (the bridge budget applies per drain — no
edge timing needed). Keep the observation records (they feed the
benchmark tooling) and the interaction-state input.

R4 must also make kernel admission prune stale side work at the owner,
not in `level_stats` or via timers. The 2026-07-08 VisPy trace
(`arrayscope-diagnostics-20260708-151620.jsonl`) shows correct/presented
montage tiles at the slow scroll tail, but queued/running workers are
dominated by superseded `display_preparation`, `display_preview`,
`histogram_refinement`, and ROI/profile hover tasks. Visible presentation
and current-source preview work must preempt or cancel stale refinement
work between bursts; histogram tasks may run in idle gaps, but once a
newer visible target supersedes them, kernel ownership should drop them
before they can occupy workers for seconds.

Eviction/admission ranking should become one shared policy instead of
parallel backend and LOD guesses. The priority order is: stale semantic
targets first, then offscreen/out-of-index residents, then farthest near-ring
residents, and only under visible residency pressure demote active exact/finer
payloads to already-resident lower LOD. Ordinary camera zoom is not pressure;
it must not schedule lower-quality DESIRED materializations or wrapper swaps
while exact/finer current payloads fit.

Preview-level selection should become dynamic at the same owner boundary,
not a renderer-local constant. Inputs should include viewport demand, tile
shape, operation cost/capabilities, staged/intermediate availability, cache
budget, and memory pressure. The ladder still decides which rung is useful;
kernel admission decides what can run now. Startup/far-zoomed montages should
therefore request the cheapest correct first pixels, while spare capacity can
refine toward exact only after visible correctness work is admitted.

`core/gui_callback_budget.py` stays — it is the drain bound vocabulary.
`core/latency_feedback.py` shrinks to the EWMA + outlier suppression the
bridge budget uses. `compute_policy` lane worker counts become the initial
lane quotas.

## LOD / preview production policy (R3 field feedback, 2026-07-08)

R1–R3 land the *mechanism* — degraded-preview first-pixels (shown whenever a
tile would otherwise be black, even at native scale, labelled `quality=
"preview"`, replaced by target work without being cleared) and rough → hold →
refined level/histogram phasing. They keep a **fixed** preview LOD level and a
**static** preview-vs-target choice. R4 owns the adaptive policy at the same
admission boundary:

- **Backlog-driven preview choice.** Showing coarser-than-target pixels is a
  data-availability + latency decision, never "the viewport demanded reduced":
  - target-quality data already producible (a resident finer level to
    downsample, or ops staged) **and not running behind** (≤1 tile missing) →
    present target directly, no preview detour;
  - **running behind** (≥2 tiles missing) → downsample the operation output —
    even from higher-quality data — to blanket-fill fast;
  - **caught up** (≤1 behind) → stop downsampling and upgrade quality,
    **oldest low-quality tiles first**.
  "Behind" is the backlog the drain/commit budget already tracks; wire it in
  rather than inventing a new counter.
- **Never downgrade LOD.** A committed/uploaded tile is always kept (prefer
  target, then finer, then coarser); a presented tile is never swapped to a
  *lower* LOD except under memory pressure, and even then only as the last
  eviction choice when no other resident can be freed (extends the eviction
  ranking above). Zoom-in retains the coarser data as correct first-pixels and
  treats the newly-needed detail as *new* tiles (so if preview LOD still beats
  target and we are behind, preview shows first).
- **Never downgrade levels/histogram.** Level/histogram evidence is never
  replaced by lower-quality evidence, and is not re-sampled (rough or refined)
  when equal-or-better evidence already exists — the R1–R3 `has_source`/refined
  guards enforce this; R4 preserves it while moving refinement scheduling into
  kernel admission (drop stale side work at the owner, not in `level_stats`).
- **Dynamic preview level.** Until R4 the preview LOD level is a renderer-local
  constant; R4 makes it dynamic from viewport demand, tile shape, op
  cost/capabilities, staged availability, cache budget, and memory pressure
  (see the dynamic-preview-level paragraph above). The op cost/capabilities
  input, and the reduced-input eligibility gaps it must reason about, are
  audited in [`op-reduced-input-compat.md`](op-reduced-input-compat.md).
- **Move the R3 level-stats bridge into kernel admission.** The 2026-07-08
  VisPy stabilization keeps startup/reload levels correct by seeding rough
  evidence from currently displayed payloads at montage completion, keeps
  preview evidence provisional, and queues exact/current payloads for refined
  stats after visible completion. That bridge belongs to R4's owner model:
  the kernel should admit current semantic level evidence and refined
  histogram work as supersedable side work, with visible presentation deps
  outranking it, instead of `LevelStatsService` discovering displayed payloads
  from session completion.

Deferred to a focused follow-up (not strictly R4): the **PyQtGraph
2-quality-level** presentation. Because PyQtGraph bakes levels into pixels at
commit, its auto-levels currently crawl tile-by-tile as bounds grow, re-baking
tiles through the fill. It should instead capture a full-coverage *rough* level
estimate before the first CPU-LUT commit, show the preview-LOD tiles at those
stable rough levels, then apply one *refined* level update for the final-LOD
tiles. VisPy already does rough → hold → refined because its levels are a cheap
late GPU uniform.

## Exit gate

- Timer allowlist test green; grep shows no scheduling-category timers.
- Governor file count of decisions ≤ 2 knobs; ADR 0052 status updated to
  "superseded by ADR 0053 R4" with the rescope recorded.
- Idle CPU still 0% (settled-idle probe); scrub heartbeat gap ≤ 16 ms.
- Benchmarks within ±10% of R2/R3 numbers on both backends.
