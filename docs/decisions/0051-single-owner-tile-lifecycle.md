# 0051 — Single-owner tile lifecycle (presentation-pipeline rework)

**Status:** Accepted (2026-07-04). P1-P3 (presentation, semantic, and residency axes
authoritative, identity-aware acknowledgement, event-driven convergence) implemented on
`feature/lod-residency` as of 2026-07-06; later phases tracked below.

## Context

ADR 0050 delivered resident LOD, the presentation floor, the retained preview level, and the
background preview walk — and every one of those landings surfaced the same defect class:
**optimistic bookkeeping around ignored returns and acknowledge-only cleanup.** Concrete
instances fixed in one day: pyramid singleflight claims leaked on blocked admission; tile
evaluations lost on declined admission (28 tiles "loading" forever); stage request keys wedged
in `stage_fan_in.active_requests` (64 tiles waiting on a lost stage); `dirty_payloads` for
non-active tiles re-emitting upserts a viewport-scoped backend never accepts (~120
commits+draws/s idle loop); walk claims leaked into the shared pyramid on session replacement
(wedged wrong LOD on scrub-back); whole-atlas mipmap regeneration presenting a reused slot's
previous occupant (mipmaps now default-off, `ARRAYSCOPE_ATLAS_MIPMAPS=1`).

These are not independent bugs. They are the signature of **split ownership**: a tile's
lifecycle is stored as ~15 parallel collections on `MontageRenderSession` (`loading_tiles`,
`skipped_tiles`, `active_tile_requests`, `presented_tiles`, `rendered_tiles`,
`display_tile_payloads`, `dirty_payloads`, `pending_payload_upserts`, `pending_removals`,
`parked_dirty_payloads`, `pending_lod_requests`, `acknowledged_source_ids`, the
`pending_level_*` queues, `stage_fan_in.active_requests`, plus `MontageTileState`, a 4-value
enum that cannot express presented, parked, or refining), and four writers mutate them:
`montage_session` (bookkeeping methods), `frame_renderer` (~50 montage methods driven by
completion callbacks), `montage_lod` (decisions, floor, chains, claims, walk continuation),
and the backend (acknowledgement, atlas slots). Invariants like "a claim taken at plan time is
released on every terminal path" live in call flow spread across those writers, so every new
feature re-implements them and misses an edge.

The same split blocks the open roadmap items. X5c (viewport-scoped tiled scenes for normal
images) needs scope to be a first-class input, not montage-mode checks. X5d (region-first
materialization) needs a place to put "this region of this tile is evaluated" that is not yet
another parallel set. PyQtGraph/VisPy parity (Y2) needs semantic tile bookkeeping that is
provably backend-independent. Per-slot mip invalidation needs upload-granular derived-state
tracking. And the field-reported symptoms — stale LOD after scrubs, wrong-content tiles,
sessions opening wrongly scaled — are all "presentation state believed something the backend
never confirmed."

`TilePresentationState.acknowledge_delta` (X5b) already shows the working pattern: an
immutable state advanced only by backend-confirmed deltas. This ADR extends that pattern from
"which payloads are presented" to the whole tile lifecycle.

## Decision

One Qt-free module, `arrayscope/presentation/`, owns every tile's lifecycle through an
explicit state machine. All other components become **event sources or effect executors**;
none of them writes tile state.

### The tile record (three orthogonal axes)

A tile's condition is the product of three axes, each with its own small state set:

- **Semantic** (is the exact value computed?): `unplanned → planned → evaluating →
  evaluated(payload) | declined | skipped`. Region-first materialization (X5d) later refines
  `evaluated` to a region set without touching the other axes.
- **Residency** (which pyramid levels exist for this tile's source?): per-level
  `absent → claimed(owner) → materializing → resident | released`. Claims are recorded IN the
  record with their owner (evaluation, chain step, walk, ingest) — a leak becomes a queryable
  fact, not a mystery.
- **Presentation** (what does the backend show?): `unpresented → upsert_emitted(source_id,
  quality, level) → presented(acknowledged) | parked(reason)`, with `presented` carrying the
  acknowledged payload identity. `parked` replaces `parked_dirty_payloads`: an upsert declined
  by scope is parked at acknowledgement and re-armed by a scope event, never re-emitted blind.

### The state machine contract

`TileLifecycle` (one instance per session, records keyed by tile number) is a **functional
core**: every input is an explicit event, every output is an explicit effect list; it performs
no I/O, no Qt, no scheduling.

Events (the complete write surface):
`plan_applied`, `scope_changed(active, near, out)`, `evaluation_started/completed/declined`,
`stage_attached/completed/lost`, `lod_planned(chain)/step_completed/declined`,
`preview_admitted`, `upsert_emitted`, `commit_acknowledged(TileCommitReport)`,
`removal_acknowledged`, `session_replaced`, `level_target_changed`.

Effects (what callers execute):
`ScheduleEvaluation`, `ScheduleMaterialization(step)`, `EmitUpsert(tile, payload_ref,
quality)`, `EmitRemoval`, `ReleaseClaim(level_key, owner)`, `ReportDiagnostic`.

Rules the machine enforces structurally (each maps to a fixed defect):

1. **No optimistic transitions.** Nothing enters `presented` except
   `commit_acknowledged` with backend acceptance (X5b, now the only path).
2. **Claim balancing is an invariant, not a convention.** Every `claimed` entry must reach
   `resident` or `released`; `session_replaced` and every `declined` path emit the
   `ReleaseClaim` effects mechanically. A dangling claim is detectable by a single scan.
3. **Emit-once, park, re-arm.** An upsert is emitted once per (payload identity, scope
   epoch); out-of-scope acknowledgement parks it; only `scope_changed` re-arms.
4. **Interaction supersedes speculation.** Effects carry a priority class; the executor drops
   or defers speculative effects during interaction. The machine never assumes an effect ran —
   executors report back (`evaluation_declined`, not silence).
5. **Derived GPU state is invalidated at upload granularity.** The presentation axis records
   per-slot derived-state (mip) validity; a slot reuse or upsert marks exactly that slot
   dirty. Whole-atlas regeneration is forbidden (this is what re-enables mipmaps by default).
6. **Convergence is event-driven.** Every piece of backend evidence — a commit report, an
   inherited atlas state — has a consumer scheduled at the moment it arrives, never "the next
   commit that happens to run". An acknowledgement that reveals a backend/machine identity
   mismatch schedules its own bounded follow-up commit; a rebuilt session inherits the
   backend's identity map instead of starting blind. A repair that only runs on event X while
   the wedge forms after the last X is the defect, not bad luck. (Added 2026-07-05 after the
   settled-wedge field defect: `backend_stale_identities` steady at idle with zero dirty
   tiles, healed only by a lucky pan.)

### What the four writers become

- `montage_session`: keeps the plan, caches, and revision counters; per-tile collections are
  deleted phase by phase and replaced by views over `TileLifecycle` (e.g. `loading_tiles` ==
  records with semantic axis `evaluating`).
- `frame_renderer`: completion callbacks translate results into events and execute effects;
  it stops deciding state. Its scheduling stays (it owns lanes and budgets), but admission
  outcomes are always reported back as events.
- `montage_lod`: level selection, floor lookup, and chain planning become pure functions
  consulted by the machine when building effects; the walk becomes an event source whose
  admissions are ordinary claims with owner `walk`.
- backends (VisPy, PyQtGraph): consume `EmitUpsert`/`EmitRemoval` effects through the Y2
  surface contract and answer with `TileCommitReport`s — the single source of `presented`.
  Backend parity for the semantic bookkeeping is structural because the bookkeeping no longer
  lives in the backends.

### Test harness (accepted alongside, same rework)

A pytest-marked onscreen GPU suite (`-m gpu_interaction`, runs on real hardware, skipped
elsewhere — Xvfb/software-GL is not evidence, per X5) extends the heartbeat probes of
`/tmp/lod-baseline/verify_scrub.py` into assertions:

- **Responsiveness:** 10 ms heartbeat gap histogram per phase; pan/scrub max gap budget
  (~16 ms target, >50 ms sync step fails).
- **Content:** synthetic sources with per-tile analytically known patterns (tile index + level
  encoded in pixels); after each scripted interaction (open, pan, scrub, level drag,
  scrub-back), capture the canvas and assert every visible tile shows its own pattern at an
  acceptable level — "wrong window", "previous occupant", and "stale LOD" become failing
  pixels, not eyeballs. (VisPy `canvas.render()` needs the int-rounded `physical_size`
  subclass in this environment.)
- **Lifecycle:** after settle, assert the machine has no dangling claims, no `parked` tiles in
  active scope, no `evaluating` older than the settle window, and diagnostics
  `tile_lod_reason` derived from the PRESENTED level.

## Consequences

- The corner-case stream stops being whack-a-mole: new features add events/effects, and the
  six structural rules apply to them automatically.
- `MontageRenderSession` shrinks toward plan + caches + revisions; diagnostics read one record
  per tile instead of correlating 15 collections.
- X5c/X5d/X5e and PyQtGraph adoption land as machine extensions (scope events, region-refined
  semantic axis, backend-report-driven decisions) instead of new parallel writers.
- Cost: a migration period where the machine shadows legacy state with parity assertions
  before each collection is deleted; honest about the roadmap rule — this is incremental
  conformance, not a big-bang rewrite.

### Wedge triage (diagnostics contract)

The Lifecycle diagnostics line classifies any stale-presentation report; check in this order:

1. `lifecycle_identity_rejections > 0` — a false-acknowledgement door exists and the machine
   caught it (working as designed; find and close the door). Persistent identical rejections
   end in resignation, so a non-converging backend stays bounded and visible.
2. `backend_stale_identities > 0` at idle with `dirty_payload_tiles == 0` — evidence arrived
   but nothing consumed it: a missing convergence trigger (rule 6 violation).
3. `backend_stale_identities > 0` with dirty tiles churning — the backend cannot converge;
   check resignations before suspecting the machine.
4. `stall_repairs > 0` / `last_stall_signature` — the watchdog rescued a dead pump. Every
   rescue is a bug report (a completion path exited without rescheduling), not a fix;
   machine-derived dispatch (P2) deletes the class and demotes the watchdog to an assertion.

## Phases

- **P1 (landed 2026-07-04):** `arrayscope/presentation/tile_lifecycle.py` machine + exhaustive
  Qt-free unit tests; wired as shadow (events mirrored from existing paths, parity asserted in
  diagnostics); presentation axis becomes authoritative (emit-once/park/re-arm and
  acknowledgement move in; `parked_dirty_payloads` deleted).
- **P2 (core landed 2026-07-05):** identity-aware acknowledgement is the machine invariant —
  `commit_acknowledged` compares backend slot identities (`presented_identities`) against the
  machine's emitted identities and RETURNS the confirmed set; every legacy pop consumes that
  verdict (one decision site; subsumes the three false-ack doors patched per-site on
  2026-07-04/05).  A pair rejected `IDENTITY_RESIGN_AFTER` times is resigned: the machine
  records the backend's identity as the presented truth (never our emit), bounding retries
  against a non-converging backend while `backend_stale_identities` keeps the wedge visible.
  Semantic axis authoritative for results: `rendered_tiles` is `LifecycleRenderedTiles`, a
  collection that IS the event source (set → `evaluation_completed`, remove →
  `evaluation_dropped`), so direct fixture writes stay correct and park eligibility reads
  `EVALUATED` — the `parkable_tiles` crutch is gone.  Build-side drawn-identity reconciliation
  defers to the machine's presented identity (no duplicate convergence loops).
  Rule 6 wired the same day (settled-wedge field defect): every acknowledgement queries
  `backend_identity_mismatch_tiles()` and schedules a coalesced follow-up commit when
  non-empty (`_montage_identity_repair_commits` counts them; bounded by resignation and
  per-pair attempt limits, terminated when a no-op commit yields no report); session
  replacement inherits the backend identity map; the machine records explicit
  `TileRecord.resigned` `(wanted, shown)` pairs, cleared by fresh evaluation — the build-side
  skip and the repair query match only those.
  **Session-rebirth cost (landed 2026-07-05 #3):** both candidate cures shipped.  A same-key
  re-render reuses the live session outright (no rebirth, no flush; a converged re-render is a
  pinned no-op), and an index-window scrub step with identical layout geometry retargets the
  session in place — new `session_id`/`key` (so stale completions park exactly as on a
  rebirth) while the lifecycle machine, backend acknowledgement state, and drawn payloads
  survive; source changes route through the ordinary `mark_materialized`/demotion seams and
  the budgeted flush converges them.  Payload wrapper construction is bounded by the same
  admission budget that caps uploads (unbuilt tiles stay dirty; the backlog continuation is
  the consumer, per rule 6), and the seed-only previous-frame/retained-store scans are skipped
  once a session has presented.  Sessions with pending level refinement still rebirth until
  the machine owns level convergence.  Kill switch `ARRAYSCOPE_DISABLE_SESSION_RETARGET`;
  counters `_montage_session_reuses` / `_montage_session_retargets` / retarget-reject reasons.
  Warm scrub step: ~50 → ~36 ms measured; the remaining cost was the delta-commit walk itself
  (VisPy layer update, wrapper seeding, priority ordering, full-image apply).
  **Field-verify gate: passed (2026-07-05, manual).** Verdict "good enough for now" with
  known short-lived inconsistencies attributed to the split ownership this rework is
  removing; the counters above stay the triage vocabulary for regression reports.
  **Machine-derived dispatch (landed 2026-07-05 #4):** `presentation/dispatch.py` is the one
  Qt-free decision site — `derive_montage_dispatch` reads the session/machine records and
  returns every pump they imply; `_dispatch_montage_work` executes it and every montage event
  edge ends there (tile done/error, stage done/stale/error, LOD level ready, result flush,
  deferred planning, viewport retarget, interaction-quiet).  A declined admission always
  leaves a wakeup armed (`EvaluationController.notify_when_capacity`, fired by the next drain
  that processes any completion) — this closed the dead-pump field freeze, whose four root
  causes were: the tile-admission decline stopping the drain with no re-arm; gesture-deferred
  pending records racing the update-pending flag; already-presented dirty payloads never
  clearing (endless no-op commits at idle); and blocked LOD materializations releasing claims
  with no consumer (tiles stuck on a coarse level until an unrelated pan).  Orphaned loading
  records are requeued by the derivation itself (never when stage records exist — waiting
  tiles belong to their stage).  The 1 Hz watchdog is an ASSERTION now: a zero-progress tick
  logs loudly, counts `stall_repairs` (asserted 0 in the GPU harness), and rescues via the
  ordinary dispatch, never a bespoke repair.
  **Sets-as-views + stage fan-in events (landed 2026-07-05 #6):** `loading_tiles`,
  `active_tile_requests`, and `skipped_tiles` are machine views — `TileRecord` carries
  `load_intent` / `request_active` / `stage_key`, the session attributes are set-like view
  objects whose mutations are events, and the machine clears load intent mechanically when a
  backend confirms an EVALUATED tile's payload (rule 1), when a tile parks out of scope
  (rule 3), or when it is skipped.  A preview/floor acceptance while the tile is still
  EVALUATING keeps it loading (exact content is still owed).  Stage fan-in stays the queue
  implementation, but every mutation (merge/activate/release/fail) reconciles the machine's
  per-record tile↔stage binding (`stage_bindings_replaced`; replacement sites route through
  `attach_stage_fan_in`), so "loading without a request because a stage owns it" is a record
  fact.
  **Auto-levels wedge (fixed 2026-07-05, same landing):** the tile-layer auto-levels wait
  path was a triple rule violation, reproduced at `stall_repairs` 8–10 per
  pyqtgraph+resident workflow run and now 0 across repeated runs:
  (a) the legacy `loading_tiles` entry of a confirmed-presented tile had no owner, holding
  the wait open forever (the machine view dissolves this class); (b) the parked flush's
  evidence producer was not always armed — parking now marks the session level scan, the
  dispatch derivation gained `level_evidence` and pumps the cached-stats continuation, the
  scan restarts a full pass on new arrivals (a tile materializing behind the cursor fell
  through a completed pass permanently), and a source with no finite values records
  *vacuous refined evidence* (`record_vacuous_source`) instead of being re-queued forever;
  (c) the watchdog signature could not see level-evidence or stale-level-drain progress and
  reported healthy budgeted drains as stalls (both are in the signature now, and an armed
  level-stats timer counts as scheduled work).  The evidence drain also no longer schedules
  a full presentation commit per budget slice while a flush is parked (~68 no-op commits
  per 272-tile drain).
  **Retarget level-pending fallback removed (same landing):** sessions with pending level
  refinement no longer rebirth on an index-window scrub.  The old reject guarded against a
  blind re-upsert loop that machine-gated emission (emit-once + identity-aware ack with
  bounded resignation) has made impossible; stale-level drains are budgeted, prioritized,
  and watchdog-visible, and the retarget resets the per-window evidence scan counters.
  Level convergence values themselves still live in `PresentationGenerationTracker` — the
  machine owns visibility and pumping, not yet the per-tile level axis.
  **Delta-commit walk cost (landed 2026-07-06):** warm retarget commits are delta-proportional
  enough for the P2 gate: wrapper seeding now receives the dirty-tile delta and reuses resident
  base identities before any LOD/texture lookup, and one-shot upsert/stale-level ordering uses a
  pure priority helper instead of constructing mutable queues.  Fair AC-power reruns against the
  clean base measured the no-cProfile warm scrub path at 15.7 ms mean / 15.0 ms p50 / 21.4 ms
  worst, with commit slices 8.4–13.7 ms and payload-build slices 3.7–5.9 ms.  The changed tree
  measured 14.8 ms mean / 14.0 ms p50 / 24.3 ms worst, with commit slices 7.3–13.5 ms and
  payload-build slices 2.5–5.1 ms.  cProfile-confirmed worst-case commit spikes narrowed under
  instrumentation (clean commit slices 17.7–32.7 ms, changed 22.9–30.5 ms), while outer profiled
  step means were effectively noise-equivalent.  Corrected scrub-fastpath probe: 59 steps,
  mean 9.4 ms, worst 21.8 ms, heartbeat p95 14.9 ms, max 48.5 ms, `stall_repairs=0`.
  No new kill switch.
- **P2-adjacent (landed 2026-07-05, the few-Hz-scroll cure):** rule 4 applied at the
  architecture level — the interaction path is cheap by construction.  Dimension scrubbing
  notes viewport interaction (it was invisible to every gate); during a burst, stage planning
  is deferred, superseded work (mid-burst steps present floors/cached payloads only; only the
  landing step plans, so superseded steps take no claims at all); evaluator keys and pyramid
  resident-level scans are memoized behind semantic identity + a pyramid revision counter.
  Scrub steps: ~216 → ~23 ms (uncached burst) / ~50 ms (cached rebuild — remaining P2 target).
- **P3 (landed 2026-07-06):** residency axis authoritative for demanded-level materialization.
  `TileRecord.levels` now carries per-level phase, owner, request metadata, and release order
  (`claimed → materializing → resident|released`).  The session's LOD materialization queue is a
  lifecycle-backed view over claimed records instead of an owning list: planning records all chain
  claims with `ClaimOwner.CHAIN`, draining marks them materializing, worker completion marks
  admitted levels resident, and stale/blocked/error paths execute `ReleaseClaim` effects against
  the shared pyramid.  `release_session_claims` is a machine scan (`session_replaced`) over live
  non-resident claims, so chain intermediates and final requested levels release by record owner
  on session replacement; the old flush-path settle repair was removed because LOD pumps are now
  implied by machine-owned claims and dispatch/capacity wakeups.
- **P4:** per-slot derived-state tracking in backends; re-enable atlas mipmaps by default.
- **P5:** PyQtGraph tiled backend consumes the same effects; X5e benchmark matrix on both.

## Alternatives considered

- **Keep patching edges.** Rejected: six instances of the same defect class in one day; each
  fix adds call-flow invariants that the next feature must rediscover.
- **Actor/queue per tile.** Rejected: thousands of tiles; the machine is one object with
  per-tile records precisely to keep scans (repair, claims audit) O(N) and allocation-free.
- **Own state in the backend.** Rejected: Y2 proved semantic bookkeeping must live above the
  backends; PyQtGraph adoption doubles any backend-owned state.

## Related records

ADR 0044 (viewport-scoped residency), 0045 (deferred-callback generation guards), 0046
(evidence-first), 0047 (auto backend), 0050 (resident LOD; the defect inventory), roadmap X5b–e,
Y1 (render contract — `TileLifecycle` consumes its staleness vocabulary, never reinvents it).
