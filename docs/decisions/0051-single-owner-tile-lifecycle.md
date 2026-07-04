# 0051 — Single-owner tile lifecycle (presentation-pipeline rework)

**Status:** Accepted (2026-07-04). Phase 1 (lifecycle core + presentation-axis ownership +
shadow parity) implemented on `feature/lod-residency`; later phases tracked below.

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
  five structural rules apply to them automatically.
- `MontageRenderSession` shrinks toward plan + caches + revisions; diagnostics read one record
  per tile instead of correlating 15 collections.
- X5c/X5d/X5e and PyQtGraph adoption land as machine extensions (scope events, region-refined
  semantic axis, backend-report-driven decisions) instead of new parallel writers.
- Cost: a migration period where the machine shadows legacy state with parity assertions
  before each collection is deleted; honest about the roadmap rule — this is incremental
  conformance, not a big-bang rewrite.

## Phases

- **P1 (this change):** `arrayscope/presentation/tile_lifecycle.py` machine + exhaustive
  Qt-free unit tests; wired as shadow (events mirrored from existing paths, parity asserted in
  diagnostics); presentation axis becomes authoritative (emit-once/park/re-arm and
  acknowledgement move in; `parked_dirty_payloads` deleted).
- **P2:** semantic axis authoritative — `loading_tiles`, `active_tile_requests`,
  `skipped_tiles` become views; stage fan-in bookkeeping reports through events.
- **P3:** residency axis authoritative — pyramid claims/chains/walk admissions carry owners;
  `release_session_claims` becomes a machine scan; delete `pending_lod_requests`.
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
