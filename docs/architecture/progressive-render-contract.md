# Progressive render contract

What a montage fill is allowed to put on screen, and who decides it.

This document is normative. Every rule below is a **correctness** rule: it holds
regardless of how slow the machine is, how large the montage is, or how far
behind the scheduler falls. Performance work may choose *which* legal state to
land in; it may never choose an illegal one. Where a rule costs time, the cost
is paid — a fill that is correct and slow is a performance bug, a fill that is
fast and violates a rule below is a defect.

## Vocabulary

- **Round** — one settled view target. A scroll step, a slice change, or a
  zoom change starts a new round. Tiles retained from the previous round are
  still part of the new round's visible set.
- **Level** — LOD reduction exponent. **Higher level means coarser**; level 0
  is native. "Finer than L" means a level numerically below L.
- **Preview floor `P`** / **target floor `T`** — the round's two quality
  floors, `P` coarser than `T`. Both are **minimums**, not exact levels: a tile
  at level `≤ P` satisfies the preview floor, a tile at level `≤ T` satisfies
  the target floor.
- **Production** — computing, reducing or uploading tile data. Displaying data
  that is already resident is *not* production.
- **Presented tile** — a tile whose pixels are currently on screen.
- **Round levels** — the window/levels the round's pixels are mapped through.

## R1 — Two production passes per round, and only two

A round runs at most **two** production passes: one preview pass to the floor
`P`, then one target pass to the floor `T`. No third rung exists — no
"slightly better than preview but not yet target", and nothing finer than `T`.

The bound is on **production**, not on what is on screen. The count of distinct
levels visible at any instant is unconstrained, because retained data is free
to show (R2). What is forbidden is *producing* a level outside `{P, T}` for the
round, and producing anything for a tile that already satisfies the relevant
floor.

Concretely, within one round no tile may be produced:

- at a level finer than `T` — the round did not ask for that quality;
- at a level between `P` and `T` — that is a third rung;
- at `P` when it already sits at `≤ P`, or at `T` when it already sits at `≤ T`
  (see R2).

> Observed violation: with target floor `T = 1`, WGPU scalar kept uploading
> level 0 — 176 uploads in one snapshot interval, then 32, 12, 16, 52 — for a
> round that only demanded level 1. Level 0 accumulated 680 uploads / 178 MB
> across the trace. Producing finer than the target floor is waste, whether or
> not the result is legal to display.

## R2 — Floors are minimums: show what you have, never redo it

`P` and `T` are quality floors. A tile that already meets or exceeds a floor is
**skipped** in that pass — never re-produced, never re-uploaded, never
re-baked. A tile resident at level `≤ P` is skipped by the preview pass; a tile
resident at level `≤ T` is skipped by the target pass as well, and is already
final for the round.

This is what lets a round cost almost nothing when the view barely moved, and
it is a correctness rule, not an optimization: re-producing a tile that already
satisfies the floor is forbidden work, and re-uploading it can only make the
image worse (it briefly replaces good pixels with equal-or-worse ones).

**Quality never regresses.** No tile is ever replaced by a coarser version of
itself. If a tile is resident at level 0 and the round's floors are `P = 5`,
`T = 2`, that tile is already better than the round will ever ask for: it is
displayed as-is and takes no part in either pass.

### The free-reuse rung

A tile that is resident and ready but *coarser* than `P` may be displayed
**immediately, before the preview pass runs**. This is display-only and must
cost zero production. It is strictly better than showing nothing: the user gets
a blurry, roughly-scaled image at once, which the preview pass then replaces.

This is why the visible level set may legitimately hold three or more distinct
levels at once — say a retained level 0 from a zoomed-in view, a coarse
retained level 6, and this round's `P = 4` — while the round still runs only
the two passes R1 allows. Displaying that mixture is correct. Producing it is
not.

> The instantaneous visible level set is therefore **not** an invariant and
> must not be asserted as one. The oracle checks production, not residency.

### Skipping production is not skipping presentation

R2 governs **production**. It says nothing about whether a tile reaches the
screen, and conflating the two strands tiles permanently.

A tile whose resident level already satisfies the round's floor is correctly
skipped for production — and if it is not currently presented, it still needs a
**presentation step**. Residency satisfies R2; it does not satisfy the
lifecycle's physical first-pixel obligation. The ladder must therefore plan
presentation-only work for a tile that has pixels but is not showing them,
rather than refusing it as "already covered".

> Observed violation: zoom in, then zoom out. Tiles retained at level 0 from
> the zoomed-in view satisfy the new coarse target floor, so every replan
> refused all of them with *"tile already has committable coverage"* — 112
> replans, that counter climbing 4 344 → 34 808 — while `ledger_target_ready`
> held 137, `schedulable` and `running` were 0, the kernel was idle, and
> `tasks_submitted` and `commit_batches` were frozen. The montage settled at
> 135 of 272 forever. **No stall fired**, because the ledger considered those
> tiles ready and therefore fine.
>
> A test asserting only that the *retained* tiles stay visible passes through
> this untouched. Coverage of an expansion must assert that the montage
> **completes**, and must reproduce the zoom-in-then-out sequence — expanding
> without first zooming in never creates the finer-than-floor residency that
> triggers it.

## R2b — The floors are one number each per round

A round has **one** preview floor and **one** target floor, chosen once, before
any tile of that round is scheduled, and **fixed for the life of that round**.

Per-tile derivation of either floor is forbidden. Retention affects whether a
tile is *skipped* (R2); it never affects what the floor *is*.

Two corrections to an earlier reading of this rule, both established by
measurement rather than inspection:

- **The risk is temporal, not spatial.** An earlier draft said two tiles in the
  same round must not land on different floors "because their individual demand
  or retention state differed". That mechanism does not exist: one planning
  pass hands every tile the same demand and policy, so a single pass cannot
  produce heterogeneous floors. What R2b actually forbids is the floor *moving*
  between planning passes that belong to the same round.
- **A session is not a round.** Session identity is not a usable round key in
  either direction. Across the recorded traces one session spans up to 14
  distinct view scales, and elsewhere 25 sessions cover 14 scales — sessions
  are sometimes coarser than rounds and sometimes finer. The floor legitimately
  tracks the continuous viewport scale, so it changes whenever the view target
  changes; observing a floor change inside one session is therefore **not**
  evidence of a violation, and it was misread as one before this note.

**Round identity is now explicit.** It is a structural key over the semantic
session key (document/operation revision and display state), exact view range,
viewport shape, montage plan geometry/source population, and display axes.
Visible/resident tile population is work state and is deliberately excluded.
So are demand, admission, and scheduler epochs: a change to any of those that
leaves the settled view target unchanged is another planning pass in the same
round and must reuse its latched floors. A genuine change to the structural
target starts a new round and permits both floors to be selected again.

## R3 — Levels never clip what is drawn

The round levels in force at any instant must **contain** the true value range
of every tile presented at that instant.

Round levels may be provisional, and may be refined toward the final window as
evidence accumulates. Refinement may only widen coverage over the presented
set; it may never leave a presented tile outside the window it is drawn
through.

Corollaries, by backend:

- **PyQtGraph** bakes levels into the image at upload. It therefore needs the
  round's levels **before the first tile of the round is drawn**. A tile
  uploaded under levels that later change must be re-uploaded, so the levels
  must be settled first. "Settled" means final for the round, not merely
  seeded.
- **WGPU** applies levels in the shader and can restate them per frame. It may
  begin drawing before levels are final, but at every frame the levels uniform
  must already cover every tile in that frame. Adding a tile to the presented
  set and widening the levels to include it must happen in the **same** commit,
  never in that order across two frames.

> Observed violation: PyQtGraph FFT presented all 272 tiles over ~21 s with
> levels evidence frozen at 32 of 272 sources, then converged the levels
> afterwards. WGPU FFT filled 272 tiles with evidence `inactive`, and only
> started computing levels once the last tile was presented.

### The levels tolerance ladder

"Contain" is not "equal". A tile whose baked levels differ from the round
levels is **acceptable within a tolerance that tightens as the round
progresses**. This is what lets PyQtGraph reuse committed tiles the way WGPU
reuses bound ones: PyQtGraph cannot restate levels after commit, but it does
not have to, as long as the mismatch is inside the tolerance for that stage.

| Stage | Tolerance on each bound | Rationale |
|---|---|---|
| Free-reuse rung (pre-preview, R2) | ±20% | Better to show something blurry and mis-scaled than nothing; it is replaced within one pass anyway |
| Preview pass | ±10% | Keeps the preview pass dedicated to tiles that are truly missing or truly mis-scaled |
| Target pass (final) | ±2% | The settled image; the tightest the round ever gets |

Tolerance is evaluated per bound (lower and upper independently) as a fraction
of the round's level span. A tile inside the tolerance for the current stage is
**not** re-produced — this is a skip condition in the sense of R2, and it
composes with the LOD floors: a tile is redone only if it fails the level floor
*or* falls outside the stage tolerance.

The percentages above are the starting policy, not physical constants. They
belong to the round planner as named policy values, tunable in one place — not
scattered as literals at the tile sites that read them.

Two rules bound the ladder:

- **The final stage is not optional.** A round that settles must end with every
  presented tile inside the target tolerance. Tolerance buys time during the
  fill; it never survives into the settled image.
- **Tolerance never overrides R3.** A tile may sit at slightly different levels
  from the round window, but it may never be *clipped* by it — the window must
  still contain the tile's value range. Tolerance is about how exactly the
  mapping matches, not about whether data falls outside it.

## R4 — Every backend and dtype has a preview pass, or has none

A pipeline either supports the preview pass for a given backend and dtype, or
it does not. What is forbidden is the middle state: skipping the preview and
letting the **target** pass become the first pass on a montage large enough to
have needed one.

If a backend cannot preview a dtype, that is a gap to close, not a fallback to
ship. Until it is closed, the pipeline must not silently degrade into an
unbatched full-quality first pass.

### The preview pass does not depend on reduced *input*

Whether the operation pipeline can consume reduced input is a question about
**how cheaply a preview tile is produced**. It is not a question about whether
the round has a preview pass. Those are separate, and conflating them is how
whole pipelines lost their first pass.

The pass exists to put a *complete* image on screen quickly. That value comes
from the ordering — every tile at `P` before any tile at `T` — and it survives
even when producing a coarse tile costs exactly what the target costs. A round
that cannot reduce its input must still run the preview pass, served by
evaluating natively and reducing the **output** for a cheap upload or bake.

This is not double work, because R2 already forbids it from being double work.
A native evaluation sits at level 0, which satisfies every floor the round has;
the target pass must therefore **skip** that tile and re-present from what is
already resident rather than evaluate it again. One evaluation, presented
coarse first and refined after — the preview cost is the extra presentation,
not an extra computation.

> Observed violation: `reduced_input_available == False` removed the FLOOR rung
> entirely, so an operation change or a source reload with any non-narrowable
> operation active jumped straight to target quality. The ladder already
> supports this case — `RungStep.reduce_from_native` exists precisely for it —
> but the admission gate refused the rung the step was built to serve.

> Observed violation: PyQtGraph complex (FFT) has no preview atlas
> ("reduced RGB payload format" absent), so its first and only pass runs at
> target level over every tile — the "starts at too high a quality" and
> "very very slow" report.

## R5 — All bulk work is chunked and governed

Any operation over the round's tile set is chunked under a governor, on both
passes. A pass may complete in one unchunked burst **only** if the governor
predicts it fits; "preview is cheap" is not a licence to bypass the governor,
it is a prediction the governor must verify.

> Observed violation: scalar preview presents every tile in one step
> irrespective of cost; the PyQtGraph FFT second pass is unchunked and freezes
> the UI until it completes.

### 50 ms is a ceiling on blocking, not a target update rate

The 50 ms number is the **maximum a GUI-thread callback may block**, and
nothing else. It was read for a while as a target cadence, and a scheduler
built to hit it collapsed: it shrank the cohort toward one tile, which did not
shorten the callback because the dominant cost was fixed rather than per-item,
and it shrank the budget it was measuring against at the same time. Both levers
saturated and the fill never recovered.

So the governor does **not** minimize distance to 50 ms. It fits a measured
cost model — fixed, per-item, per-byte — and minimizes one smooth objective:
predicted fill time, plus a responsiveness price that is zero below 18 ms,
rises gently to 45 ms, and then climbs quadratically, plus a price for
extrapolating beyond the cohort sizes it has actually measured.

Two consequences that look wrong and are not:

- **Deliberately exceeding 50 ms can be correct.** If one tile costs 49.9 ms
  and each additional tile costs 0.1 ms, the objective picks ~136 tiles at
  ~63 ms, because two chunks of 63 ms beat 272 chunks of 50 ms by two orders of
  magnitude of fill time. R5 still *reports* the overrun; optimization sees the
  whole smooth curve rather than a cliff.
- **Model risk is charged once per decision, not per chunk.** Extrapolating to
  an unmeasured cohort size is a property of taking the step, not of every
  chunk that follows. Multiplying it by the chunk count made a few milliseconds
  of risk read as hundreds and pinned the cohort at whatever size had already
  been measured — the same bias toward tiny cohorts, arriving by a different
  route.

### Cadence is a product property, not a latency bound

A montage that completes in less total time but appears all at once feels worse
than one that visibly progresses. Some update cadence is wanted for its own
sake. It is not obtained by capping chunk size: it belongs in the objective as
a preference, alongside fill time and responsiveness.

### The budget is a bookkeeping budget, not a compute budget

Array compute is already off the GUI thread. What remains in a 50 ms callback
is presentation state, payload iteration, residency bookkeeping, geometry, and
GPU resource allocation — one measured warm WGPU callback was 107.1 ms of which
50.9 ms was pool growth, and 63–69% of a commit was iterating every presented
payload, so an empty-delta commit still cost ~90 ms.

That is why no chunk size satisfies R5 on its own: **chunking work whose cost
does not depend on the chunk cannot help.** The fixed term must come down
(commit cost proportional to the delta) and what remains should not be on the
GUI thread at all — that thread should validate, submit, and release buffers,
with preparation owned by a worker.

## R6 — Falling behind degrades quality, never liveness

When the scheduler cannot keep up with incoming rounds, it sheds work in this
order, and never blocks:

1. Drop the target pass. Draw preview only.
2. If preview cannot keep up, **move the tiles that already exist** to their
   new positions and draw the newly exposed area as black; after a short
   timeout, a placeholder.
3. Fill correctly once the input settles.

Freezing the montage until idle is never a legal response to load. Stale pixels
that move with the view are better than correct pixels that stop moving.

> Observed violation: WGPU fast scroll freezes every tile until idle, then
> replaces them all at once.

## Ownership

Most of the above failed because the decision was made at the wrong altitude.
The binding assignment:

| Decision | Owner | Scope |
|---|---|---|
| Whether this round previews at all | round planner | per round |
| The preview floor `P` (R2b) | round planner | per round, one value |
| The target floor `T` (R2b) | round planner | per round, one value |
| Round levels / window (R3) | round levels owner | per round, one value |
| Stage tolerance percentages (R3) | round planner | per round, named policy |
| Whether a given tile is skipped (R2) | ladder | per tile, **reads** the round floors and tolerance |
| Which rung a given tile still needs | ladder | per tile |
| Chunk sizes and pacing (R5) | governor | per pass |
| Load shedding (R6) | scheduler | per round |
| Baking or binding levels into pixels | backend adapter | per tile, **reads** the round value |

Both altitude errors have now been repaired:

- **Preview floor — repaired.** `render.lod.selected_lod_factor()` chooses the
  preview and target floors from the round demand and latches that demand plus
  both values against the structural round key. The immutable render intent
  carries the id and floors through the pipeline to `LodLadder.plan()`; the
  ladder policy owns no fallback value, and omitting either floor fails
  loudly. The ladder still decides, per tile, whether that tile is *skipped* —
  that is correctly per-tile work (R2) — but evaluation and rung planning now
  read the round floors unchanged rather than re-deriving them.
- **Round levels — repaired.** `LevelStatsService` claims the decision when a
  preview rung is admitted, then installs the complete preview cohort as one
  tracker revision. The cohort rows carry worker-prepared bounds and samples,
  so no per-source slab sweep or duplicate full operation/FFT runs beside that
  preview.
  For the admitted montage-axis orthonormal FFT, the displayed pages box-mean
  the display axes before the operation pipeline, while the same cohort worker
  scans every native input value and derives the conservative L1 envelope
  `max(sum(abs(x), axis) / sqrt(N))`. The profiler's admitted
  FFT/shift/IFFT chain is a phase modulation and uses the native maximum
  magnitude. These are complete native-resolution bounds rather than sparse
  samples, so they cover R2 free reuse without a native FFT. The ladder's
  commuting tile-local predicate, exact operation shape, transform axis, and
  linear shader scale are executable preconditions; an unproved chain keeps
  the transform-once-native route. When the preview floor is already native,
  the one native input read supplies both the envelope and the one displayed
  transform. PyQtGraph still waits for the complete cohort decision before its
  first bake.
  PyQtGraph holds its first bake until the complete round source exists and
  keeps that value for the round. WGPU may widen the installed value from
  current-round target evidence, but does so before the corresponding tile is
  marked dirty. Pipelines with no usable preview cohort retain the semantic
  source-slab owner as an explicit fallback after coverage closes.

  That fallback deliberately parks while coverage is open, which is the shape
  of the regression reverted in `61bb5f1a` — and is only safe for the opposite
  reason. The reverted park held levels at a seed batch *while tiles were being
  presented*; this one runs only in the `preview-cohort-pending` window, whose
  whole premise is that nothing has been baked yet. **The park is legal exactly
  as long as that premise holds**, so it is pinned by test rather than left to
  inspection.

A backend adapter may decide *how* to apply a value (bake vs bind). It may
never decide *what* the value is, and it may never be the place a round-level
invariant is first enforced.

## R7 — Speculative residency is post-settle work

Uploading data the *current* round does not display — whole-plane warming for a
future crop, breadth for a future scroll — is **prefetch**. It runs after the
round settles, on an idle/prefetch lane, and it yields to any new round
immediately. It is never admitted alongside the preview or target pass.

This is not a priority preference. Speculative traffic dominates the upload
path: across the two recorded WGPU traces, level 0 is **82% and 84% of all
upload bytes** (891 MB of 1089 MB; 381 MB of 453 MB) while the levels the
rounds actually displayed account for under 11% each. Letting that share
compete with the fill is the difference between a montage that completes and
one that empties out.

> Observed violation: on the 272-tile WGPU FFT round the preview pass reached
> 272/272 presented, and then — as whole-plane level-0 warming began (+64,
> +192, +128, +128, +128, +192 uploads in consecutive intervals) — presented
> collapsed to 183, then 77, and took four more seconds to climb back to 272.
> The montage visibly emptied out *after* it was already complete.

### Warming does not have to be native

The mechanism is legitimate: `canonical_plane_view_state` lets an anchored crop
be presented out of whole-plane pages instead of a crop-local upload, so
whole-plane residency is what makes cropped X/Y indexing fast.

But it does not follow that the warm must be at level 0. Tile identity on a
windowable axis is **source-anchored**: the tile grid aligns to
source-coordinate multiples and the content key names the whole plane with the
anchored axis' window stripped. A window shift from `50:100` to `51:101` is
therefore already a hit **at whatever level the whole-plane pages exist** — the
reduction bins are anchored to the source grid, not to the window origin, so
they do not re-phase when the window slides. Shift-invariance comes from
anchoring, not from native resolution.

Two consequences for the prefetch ladder:

- **Breadth before depth.** Extending whole-plane coverage at a coarse level
  buys the same shift-invariance for a fraction of the traffic — a whole plane
  at level 2 is 16× smaller than at level 0. Prefetch should widen coverage at
  the levels a near-future view would actually display before it deepens
  anything.
- **Native warming is the narrow case, and goes last.** It is justified only
  where arbitrary levels must be re-derived on the GPU without a re-upload. It
  is not the general answer to crop shifts.

Note what warming cannot buy: on a *non-anchored* axis (an FFT along it) the
window stays folded into the content key, so a shift misses by construction —
and native residency does not rescue that, because the operation has to be
recomputed regardless.

## Acceptance

A change to the progressive path is accepted when the invariant oracle reports
no R1/R2b/R3 violation across the recorded interaction traces, and the relevant
suites are green. Latency medians are reported alongside, but a median may
never be traded for an invariant.

One caveat on what the oracle can see. The 500 ms diagnostics snapshots carry
residency (`tile_lod_resident_tile_levels`) and, on WGPU only, cumulative
per-level upload counts (`wgpu_uploads_by_level`). Residency alone cannot tell
production from a cache arrival, so R1 is checked from upload counters where
they exist and reported as unverifiable where they do not. Nothing distinguishes
a deliberate native-warm upload from over-production; until uploads are tagged
with their purpose, level-0 traffic during a coarser round is reported as
*suspected* over-production, not asserted as a violation.

Tagging uploads with their purpose (round production vs speculative warm) is
the change that turns this from a suspicion into a decidable rule, and it makes
R7 checkable at the same time: warm traffic inside a round's fill window is a
violation regardless of its level.

## Status, 2026-07-28

Where each rule stands after the preview-first recovery program. Written for
whoever picks this up next; update it in place rather than appending.

| Rule | State |
|---|---|
| R1 two production passes | Enforced. Oracle keys on round id; `authoritative_round_identity_present` now passes, so a red R1 is informative again. |
| R2 floors are minimums | Enforced for skip and free reuse. The **tolerance ladder is not implemented** — see below. |
| R2b one floor pair per round | Enforced. Round identity is explicit and both floors latch to it. |
| R3 levels never clip | Enforced. Round levels come from the preview cohort, with an analytic envelope that is exact for realistic k-space. |
| R4 preview for every backend/dtype | Green on both backends, including PyQtGraph complex and pipelines that cannot reduce their input. |
| R5 chunked and governed | Governed. The per-commit **bookkeeping** is now delta-proportional; the per-commit **aggregates** are not. See below. |
| R6 shed quality, never liveness | **Not implemented.** WGPU fast scroll still freezes until idle. |
| R7 speculative residency is post-settle | **Not implemented.** Level 0 was 82–84% of upload bytes and ran during the fill. |

Open work, roughly in dependency order:

1. **Presentation cost proportional to the delta.** Partly done — the
   whole-montage *bookkeeping* is gone, the whole-montage *aggregates* are
   not. What is left is an incrementally accumulated montage histogram owner
   and maintained presentation truth; see "What a bounded commit still costs"
   below for why neither is a loop that can be tightened in place.
2. **Presentation bookkeeping off the GUI thread.** The thread should validate,
   submit and release buffers; preparation belongs to a worker, behind a
   mailbox that keeps the latest prepared frame and drops stale ones.
3. **R6 load shedding** on WGPU fast scroll.
4. **R7 post-settle prefetch**, with the warm ladder breadth-first at coarse
   levels rather than native-first.
5. **The R3 tolerance ladder** (±20% free-reuse, ±10% preview, ±2% target),
   which lets PyQtGraph reuse committed tiles the way WGPU reuses bound ones.
6. **Tag uploads with their purpose**, which turns the oracle's suspected
   over-production into a decidable rule and makes R7 checkable.

### What a bounded commit still costs

Measured at 272 tiles, the field scale, against the parent revision.

The **bookkeeping** that made a commit O(montage) is gone. Four walks were
keyed to the montage and are now keyed to the delta, all pinned by counting
tests in `tests/render/test_commit_cost_scaling.py`. For a one-tile commit,
against the parent revision those counters read: 272 and 816 rebuilt tile
layout regions (PyQtGraph and WGPU — WGPU asked three times per commit), 272
rebuilt GPU tile instances, and 818 re-derived pin ownerships over a 272-page
resident set. All four are now bounded by the delta.

End to end, the total GUI-thread commit cost of a full 272-tile fill in 32-tile
cohorts fell from 79.1 ms to 70.9 ms on WGPU (−10%) and from 66.8 ms to 64.3 ms
on PyQtGraph (−4%). Those are means of six order-balanced interleaved rounds
per revision; on WGPU every after-round beat every before-round, on PyQtGraph
one round overlapped.

Interleaving is not optional here. Run sequentially — all of one revision, then
all of the other — the same benchmark reported −19% and −18%, because this
machine drifts by more than the effect over the minutes such a comparison
takes. Alternate revisions within one sweep and balance the order, or the
number measures the clock, not the change.

What remains is **whole-montage aggregates**, not bookkeeping, and it is the
larger half on PyQtGraph. After this change, a bounded PyQtGraph commit at 272
tiles still spends ~28% rebuilding the montage-wide histogram source and ~24%
re-reading presented-tile identity from Qt, one `isVisible()` call per tile.
Neither is a loop that can be made cheaper in place:

- The histogram source is a montage-wide aggregate with no owner outside the
  commit. It is rebuilt because any tile's pixels changing genuinely changes
  it. Making it delta-proportional means an incrementally accumulated owner —
  or accepting a coarser repaint cadence during a fill, which changes what the
  histogram widget shows and is a product decision, not a performance one.
- Presented-tile identity is read from Qt because `state.visible` and the
  item's own flag are deliberately allowed to diverge. Maintaining that truth
  instead of re-reading it is sound, but it moves a correctness invariant into
  bookkeeping, and this module exists because optimistic bookkeeping is the
  defect class that strands tiles.

> Two traps, both paid for once here. **The fitted `fixed_ms` the governor
> steers on is too noisy on this machine to adjudicate a change of this size**:
> across four repeats it ranged 4.8–59.2 ms before and 0.0–40.5 ms after, a
> spread larger than the effect, because the fit often has two or three design
> points. Read it as direction, never as a result; the counting tests and the
> in-process fill benchmark are the evidence.
>
> **A commit benchmark that reuses one object cannot see the app.** Two
> separate caches were justified on benchmarks whose regime made them free,
> and both cost more than they saved in the real workflow. A histogram-reuse
> cache measured on a *fixed* tile population made a fill slower, because
> reuse can never succeed while tiles are still arriving and the attempt was
> paid for regardless — 336 payload inspections per commit where one pass is
> 168. A layout cache keyed on object *identity* measured a 9% hit rate (40
> hits, 412 misses) against the app, which rebuilds geometry every step, while
> eager derivation made each of those misses more expensive than having no
> cache at all; keyed by value and derived lazily it is 443 hits / 8 misses.
> Benchmark the regime the app is actually in, and reuse the objects the app
> actually reuses.

### How to verify a change here

- `python -m arrayscope.tools.progressive_render_oracle --summary TRACE.jsonl`
  replays recorded diagnostics against R1/R2b/R3. A clean run means "no
  violation detected by a snapshot heuristic", never "the contract holds".
- `arrayscope.tools.profile_montage_workflow` gates its exit on an in-process
  R1–R5/R7 verdict. Medians are report-only and must stay that way.
- `arrayscope.tools.render_pass_governor_probe` reports wall time, throughput,
  chunk distributions and cost attribution — use it rather than building
  another harness.
- Run `tests/ui` and **diff the sorted failure list against the base commit**.
  A raw count cannot distinguish a stall from xdist noise, and this ring
  carries genuine flakes: re-run any delta serially before believing it.
  `tests/gpu` and `tests/ui` contend for the GPU — never run them concurrently.
