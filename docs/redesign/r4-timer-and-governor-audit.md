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

## Exit gate

- Timer allowlist test green; grep shows no scheduling-category timers.
- Governor file count of decisions ≤ 2 knobs; ADR 0052 status updated to
  "superseded by ADR 0053 R4" with the rescope recorded.
- Idle CPU still 0% (settled-idle probe); scrub heartbeat gap ≤ 16 ms.
- Benchmarks within ±10% of R2/R3 numbers on both backends.
