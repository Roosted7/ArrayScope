# Plan 03 — P3: residency axis authoritative

**Status:** landed 2026-07-06. Read `README.md` ground rules first.

## Background

ADR 0051's tile record has three axes; two are authoritative (presentation since P1, semantic
since P2). The **residency axis** — per pyramid level:
`absent → claimed(owner) → materializing → resident | released` — is specified but the legacy
paths still own the truth: pyramid claims/chains/walk admissions are tracked in
`montage_lod`/session collections, `release_session_claims` is an imperative cleanup,
`pending_lod_requests` is a parallel set, and the settle repair is a flush-path patch.

P3 (ADR 0051 Phases): claims/chains/walk admissions carry owners in the record;
`release_session_claims` becomes a machine scan; delete `pending_lod_requests`; the settle
repair becomes machine-driven convergence. Rule 2 becomes structural: "a claim taken at plan
time is released on every terminal path" stops being a call-flow convention — a dangling claim
becomes a queryable fact.

**Why this matters beyond hygiene:** the ADR 0050 defect inventory (claims leaked on blocked
admission, walk claims leaked into the shared pyramid on session replacement → wedged wrong
LOD on scrub-back) is exactly this axis. X5d region-first materialization needs this axis as
its landing zone.

## Step 1 — Inventory the legacy writers (read-only day)

Grep in the worktree and list every touch point with file:line in a scratch note:

1. `pending_lod_requests` — every reader/writer (to be deleted; readers become machine views).
2. `release_session_claims` — definition + all callers (becomes a machine scan).
3. Claim take/release sites: pyramid singleflight claims, chain-step claims, walk admissions,
   ingest-reduction claims — grep `claim` in `montage_lod`, the materializer, and the walk;
   note each site's OWNER (evaluation / chain / walk / ingest) and its terminal paths
   (completed / declined / superseded / session replaced).
4. The settle repair on the flush path (grep for the repair the flush performs; ADR 0051 P3
   names it) — record what evidence it consumes and what it fixes.
5. Existing machine surface: `TileRecord` fields and events in
   `presentation/tile_lifecycle.py` — `lod_planned(chain)/step_completed/declined`,
   `preview_admitted`, `ReleaseClaim(level_key, owner)`, `ScheduleMaterialization(step)`
   already exist in the contract; check which are wired vs stubs.

Deliverable: a table (site → owner → events it should emit → effect it should execute).
Do not change code yet.

## Step 2 — Shadow phase (parity, the P1 pattern)

1. Add per-level residency entries with owners to `TileRecord`; mirror every legacy claim
   take/release into machine events from the existing sites (no behavior change).
2. Add a parity assertion in diagnostics: machine residency view == legacy collections
   (`pending_lod_requests`, live claim sets), following how P1 shadowed presentation state.
3. Add a **dangling-claim scan** (single O(N) pass: any `claimed`/`materializing` entry whose
   owner has no live work) — expose as a diagnostics counter; assert 0 after settle in the GPU
   harness lifecycle checks.
4. Suite + GPU harness + a manual scrub/zoom session must show zero parity violations. Soak:
   run the wedge-repro workflow and one onscreen probe (hands-off) and check the counter.

## Step 3 — Authoritative flip, one writer at a time (one commit each)

Order chosen so each step deletes its legacy state immediately after the machine takes over:

1. **Evaluation/chain claims:** `lod_planned`/`lod_step_completed`/`lod_step_declined` become
   the only bookkeeping; declines emit `ReleaseClaim` mechanically (rule 2); delete the legacy
   chain-claim tracking.
2. **Walk admissions:** the walk becomes an event source; its admissions are ordinary claims
   with owner `walk` (ADR 0051 "what the four writers become"). Walk supersession/session
   replacement releases via the machine — this deletes the "walk claims leaked into shared
   pyramid" class structurally.
3. **`release_session_claims` → machine scan:** `session_replaced` emits `ReleaseClaim` for
   every live claim of that session; the imperative function body becomes a call into the scan
   (then inline/delete it).
4. **Delete `pending_lod_requests`:** replace remaining readers with a machine view (records
   with a claimed/materializing level), same set-like view pattern as
   `loading_tiles`/`active_tile_requests` (sets-as-views landing, f8b00bff).
5. **Settle repair → convergence:** whatever evidence the flush-path repair consumed must get
   a consumer scheduled AT ARRIVAL (rule 6) — most likely `lod_level_ready`/materialization
   completion edges feeding `derive_montage_dispatch` so the dispatch derivation implies the
   pump. Then delete the flush-path patch. If the repair ever fires in the shadow soak, that
   run tells you exactly which event lacked a consumer.

Constraints: every step keeps `derive_montage_dispatch` the ONE decision site (extend the
derivation, never add a side pump); admission declines must still arm capacity waiters; no
optimistic release (a release without a terminal event is the old defect class).

## Step 4 — Verify

1. Qt-free machine unit tests: claim-balance invariant (every `claimed` reaches
   `resident|released` on every terminal path incl. `session_replaced`, `declined`,
   supersession), owner attribution, view parity. Extend the existing exhaustive
   `tile_lifecycle` test module.
2. Full suite `-n 16`; GPU harness with the NEW dangling-claim assertion + existing
   `stall_repairs==0`.
3. Wedge repro: 0 stalls. Scrub-back stale-LOD spot check (the historical symptom of leaked
   walk claims): scrub away and back on a montage; tiles must refine to the demanded level.
4. Probes: `verify_scrub_fastpath.py` and warm-scrub numbers unchanged from Plan 01's result
   (P3 is bookkeeping ownership, not a perf change — any regression is a bug).

## Step 5 — Docs + memory

1. ADR 0051: P3 phase entry → landed paragraph (what moved into the machine, what was deleted,
   counters added, before/after diagnostics).
2. `docs/roadmap.md`: advance the X5 active queue; update `docs/current-state.md` only for the
   high-level tile-lifecycle state.
3. Commit; update Claude memory: tip, queue, the dangling-claim scan as a new triage tool.
