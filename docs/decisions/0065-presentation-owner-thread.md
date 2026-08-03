# 0065 — The presentation-state owner thread

Date: 2026-08-03. Status: accepted, staged delivery. Companion to ADR 0064
(process-isolated compute) and the progressive-render contract's R5 endgame:
"the GUI thread should validate, submit, and release buffers, with preparation
owned by a worker."

## Context

Three optimization rounds cut the montage commit callback's p95 from 313 ms to
65 ms and the user still experiences "slideshow, not movie." The 2026-08-02
trace (`arrayscope-diagnostics-20260802-233644.jsonl`, 124 s, 336×336×272,
wgpu, process pool ON) explains why the medians were the wrong target:

- 708 montage commits at p50 28.7 ms. Every one blocks ⌈29/16.7⌉ = 2 frames.
  During fills they queue back-to-back: the montage drew at a median 8/6/4
  draws per second across the session's three phases (slice+small montage /
  full montage / FFT chain). Locked-60 requires no GUI callback over ~half a
  frame; the median callback is 3.4× that.
- The commit's biggest term on the live machine is the **presentation delta
  build** (`payload`, p50 11.25 ms) — ~3× its quiet-probe cost (5.9–6.8 ms),
  because Python bookkeeping on the GUI thread contends with everything else.
  The wgpu-bound remainder (`apply`+`set_image`) is 7–13 ms; the irreducible
  Qt/wgpu core measured from the term map is ~4–5 ms.
- A second front of equal size: `kernel_bridge_drain` ran completion **bodies**
  on the GUI thread for 14.8 s of the session (2 868 drains, p95 17.7 ms) —
  per-tile admissions 7.1 s, `montage-stage-plan` bodies 4.4 s at p50 13.7 ms
  (over half a frame in 199 of 269 observations), prefetch admits 2.0 s.

Every prior round refused to move the remaining terms because **presentation
state has no off-thread owner**: `FrameSession` + `TileLifecycle` are mutated
by the commit callback, by completion bodies inside the drain, and by the LOD
floor builders — all GUI-thread by convention, with no locks anywhere
(`window/`, `presentation/` grep: zero threading primitives).

The measurement lesson from round 4 stands: the governor's fitted fixed term
spreads 5.9–14.4 ms run-to-run — wider than the 8.33 ms frame boundary — so
callback-duration statistics cannot adjudicate this work. **Adjudication is
draw-gap structure**: draws/s during interaction and during fills, longest
blocked-frame run, zero-draw intervals while draw debt exists. Baseline from
trace 3: phase medians 8/6/4 draws/s. Target: median ≥30, zero-draw
intervals ≈ 0 while debt exists.

## Decision

One dedicated **presentation owner thread** in the GUI process becomes the
sole writer of montage presentation state (`FrameSession`, `TileLifecycle`,
the montage level tracker). It is fed by the kernel completion stream and by
GUI-posted messages, and it produces ready-to-submit transactions into a
keep-latest, identity-keyed mailbox. The GUI thread's montage work reduces to:
take transaction → validate identity → backend apply/submit → present → post
the acknowledgement inputs back to the owner.

This is thread ownership, not a feature — it ships unflagged, in revertable
stages. It does not need a process: the stutter is GUI-thread occupancy, and
a thread gets the bookkeeping off that thread with zero-copy handoff.

### Why the completion stream and the commit must move together

The mapping (2026-08-03, three read-only passes) established that session
state already has writers on both sides: `mark_materialized` and the
admission helpers run inside the bridge drain; `build_tile_presentation`,
`acknowledge_tile_presentation`, `mark_presented` run inside the commit
callback; `render/lod.py` floor builders write payloads from both. A design
that moves only one side creates a second writer. So the owner thread absorbs
**both** the montage completion bodies and the commit preparation, and the
migration bridge is one explicit session-order lock (below) until the last
GUI-side writer is converted to a message.

### The pipeline

```
kernel workers (threads or processes)
    │ CompletionQueue.put — runs under the kernel lock: O(1) route only
    ▼
RoutedCompletionQueue: montage-scoped events → owner deque + wake
                       everything else      → Qt bridge as today
    ▼
OWNER THREAD (presentation owner)
    kernel.dispatch_event(event)        ← same staleness re-classification
      └ _on_rung_done → apply_commit    ← admission, prepared-upload planning
    prepare_transaction():
      level-source decision, governor render-pass decision,
      build_tile_presentation, decide_presentation,
      level-stats merge (pure steps)
    → publish PreparedPresentationTransaction into keep-latest mailbox
    → post presentation-gate event to the GUI thread
    ▼
GUI THREAD (unchanged event loop)
    gate → mailbox.take(key) → validate (session_id, generation,
        level revision, base-revision chain) → backend apply →
        commit_tiled_delta (wgpu submit) → present
    Qt residue: committed-frame rebuild, axis flips, setImageStale,
        content-extent refresh, overlays from the published tile-state
        snapshot, resident-evidence probes
    → post AckInputs{report, levels, residency snapshot} to the owner
    ▼
OWNER THREAD
    acknowledge_tile_presentation, mark_presented, state publish
    → publish tile_states + diagnostics snapshots; prepare next transaction
```

Staleness is adjudicated by the **existing** five layers (kernel supersession
at dispatch, pipeline rung intent, session/gate identity, render round key,
level-round revision) — the owner adds no new staleness concept. The GUI-side
validate step reuses the atomic-prepared-transaction template
(`frame_effects._prepared_atomic_transaction_current`): session identity,
level revision, base-revision continuity, scope coverage. A stale transaction
is dropped; the owner rebuilds after the next acknowledgement — exactly the
prepared-uploads miss semantics, "a miss is a normal outcome."

### The mailbox

`PreparedUploadMailbox` generalises unchanged (nothing in it is
buffer-specific): keep-latest per key by generation, exact-key take, byte
bound, closed counter equations. The transaction key is
`(session_id, session generation, level revision)`; the payload is the frozen
delta + presentation decision + the identity tuple + the tile-state snapshot
the GUI residue needs. Keep-latest is correct here for the same reason it is
for uploads: two transactions under one key supersede, and only the newest
can validate.

### The GUI-affinity audit (what must not run on the owner)

From the maps, the complete list of GUI/wgpu touchpoints currently inside the
moved regions, each with its treatment:

| Touchpoint | Where | Treatment |
|---|---|---|
| `physical_resident_fn` = `img_view.tiledPayloadResident` | injected into the builder (`frame_effects.py:5970`), probed per candidate | **Residency snapshot**: GUI posts the resident source-id set with each AckInputs; builder consumes the snapshot. The logical fallback branch (`frame_session.py:314`) already defines behaviour when absent. |
| `view.tiledPayloadResident` in `_prepare_backend_uploads` | `frame_effects.py:1521` | same snapshot |
| `QTimer.singleShot` re-arms | `request_presentation` CPU-windowed path, backstops | owner posts a Qt event to a GUI receiver that owns the timer; timers never live on the owner thread |
| `img_view.getLevels()`, `last_tile_commit_report` | ack inputs | produced GUI-side after present, posted to owner (already frozen values) |
| `tile_states` / `tile_state_revision` | one live GUI read (`frame_runtime.py:170`), rest already snapshot | published snapshot in the transaction; the live read converts to the snapshot |
| committed-frame rebuild, `apply_axis_flips`, `setImageStale`, extent refresh, overlays, hover/dock refresh | commit tail | stays GUI, keyed off the transaction's snapshots |
| `ResourceGovernor` render-pass family | `begin_render_pass`/`decide_render_pass` in the build; observation recording in drains | governor gains one RLock around public entry points (it has none today); the owner becomes the only caller of the render-pass family, the GUI keeps lane/pressure families |
| `handler_error_hook` | dispatch | thread-safe already (trace bus locked); strict-mode re-raise surfaces on the owner and is forwarded as a posted error |

`LodPageCache` and `BoundedCache` are already RLock-guarded; `emit_trace` is
locked; `kernel.submit`/`dispatch_event` are RLock-protected and documented
thread-agnostic. `TileLifecycle` moves wholesale (pure functional core, "no
I/O, no Qt, no scheduling" by its own docstring).

### The migration bridge: one session-order lock

Until the last GUI-side session writer is converted (retargets, prune-stale,
data-change invalidation), a single `RLock` serializes owner pipeline sections
against GUI session sections, with contention counters published in
diagnostics. This is scaffolding, not architecture: the end state has no
GUI-side session sections and the lock's GUI-side acquisitions trend to zero.
Any stage may be reverted independently while the lock holds the invariant.

### Stages

1. **Infrastructure + completion routing.** Owner thread, routed completion
   queue, session-order lock, governor lock. Behaviour pins: admission
   counters and presented output byte-identical on the probe; contention
   counters published.

   **Shipped scope, corrected by measurement (2026-08-03).** Stage 1 routes
   only the `montage-stage-plan:` bodies — the single worst drain offender
   (p50 13.7 ms, over half a frame in 199 of 269 live observations,
   4.4 s/session), a family that runs ~6 times per fill and therefore cannot
   contend the migration lock. Routing the per-tile `frame:` completions too
   — the plan as first written — censored the 272-tile probe fill 3/3
   (33–196/272 against HEAD's 3/3 within 5 s), and the cause survived two
   stand-aside policies (strict GUI priority, then one-turn alternation —
   structurally identical timelines): the admission stream serializes under
   the one lock against GUI sections dominated by `retarget_frame_pipeline`,
   ~89 ms × 52 calls = 4.6 s of an 8.6 s fill, 65 % of GUI-side hold against
   1.3 s of owner work, with a feedback loop (late admissions → replans,
   retargets 31 → 52). That retarget is the R5 model rebuild stage 2 moves to
   the owner; the `frame:` routing clause follows it there. What did need
   fixing at any scope, and is fixed: lock *barging* (a Python lock hands off
   by race, so the owner's per-event release/re-acquire starved GUI waiters
   p50 55 ms / max 550 ms — the announce-and-stand-aside alternation in the
   owner is the repair), and the drain-borne reconciliations (capacity
   waiters, the rate-limited governor reapply) which only ran when the GUI
   bridge itself drained — a routed completion now wakes them through the
   bridge's `routedEventsDispatched`, without which a task parked on quota
   that a routed completion freed waits forever.
2. **Transaction preparation on the owner.** The gate takes prepared
   transactions from the mailbox; validate → apply → submit → residue on the
   GUI; AckInputs posted back; acknowledge/mark_presented/state-publish on the
   owner. The GUI montage callback shrinks to backend apply + Qt residue.
   The index-window retarget becomes an owner message in this stage (measured
   prerequisite, not an option — see stage 1's correction), and the `frame:`
   completion families route once the GUI sections have shrunk.
3. **Snapshot conversions.** Residency snapshot replaces both live
   `tiledPayloadResident` probes; the one live `tile_states` read converts;
   diagnostics snapshots read owner-published copies; GUI-side writers
   (retarget, prune, data-change) become owner messages; lock spans shrink.
4. **Deletion and re-measurement.** With callbacks small, the frame-knee
   pacing complexity in the governor is deleted, not repaired (its predicate
   was already unadjudicable). Re-measure draw-gap structure against trace 3;
   the user records trace 4 for ground truth.

### What this does not do

- It does not move `write_texture`, the wgpu executor, or any Qt object off
  the GUI thread.
- It does not change what a commit presents: every stage is pinned on
  byte-identical presented output and unchanged admission/acknowledgement
  counters, mutation-checked.
- It does not touch pyqtgraph's whole-frame handoff (R6 remainder) or the
  index-window retarget's model rebuild (`render_entry`, p50 56.5 ms live) —
  the latter is the next R5 item after the owner exists, and becomes an owner
  message rather than a GUI callback at that point.
- It does not introduce a new staleness mechanism; every drop is adjudicated
  by the existing supersession/round/identity keys.

## Consequences

- The montage GUI callback's target shape is: take + validate (sub-ms) +
  backend apply/submit (4–13 ms today, with `set_image` the floor) + Qt
  residue (~2 ms). The 11 ms delta build, the 2–5 ms state publish, the
  ~2×2.2 ms level-stats queues, and the 14.8 s/session of completion bodies
  leave the GUI thread entirely.
- GIL contention transfers from "everything against the GUI thread" to
  "owner thread against workers" — acceptable: the owner's work is the same
  work, now off the input/paint path, and the process pool (ADR 0064) keeps
  the heavy evaluation out of this process's GIL altogether.
- The governor's render-pass model keeps steering chunk sizes on the owner;
  its responsiveness price now protects the owner's cadence rather than the
  GUI thread, and the 50 ms R5 ceiling applies to what the GUI actually
  blocks on — the submit — which is what the contract always meant.
- Crash containment: an owner-thread exception is forwarded to the existing
  UI error hook; the thread is respawned with a fresh session snapshot the
  same way a session rebirth already works.
