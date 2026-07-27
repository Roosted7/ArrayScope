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

## R2b — The floors are one number each per round

A round has **one** preview floor and **one** target floor, chosen once, before
any tile of that round is scheduled.

Per-tile derivation of either floor is forbidden. Two tiles in the same round
must never land on different preview levels because their individual demand or
retention state differed. Retention affects whether a tile is *skipped* (R2);
it never affects what the floor *is*.

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

> Observed violation: PyQtGraph complex (FFT) has no preview atlas
> ("reduced RGB payload format" absent), so its first and only pass runs at
> target level over every tile — the "starts at too high a quality" and
> "very very slow" report.

## R5 — All bulk work is chunked and governed

Any operation over the round's tile set is chunked under a governor, on both
passes. A pass may complete in a single unchunked burst **only** if that burst
fits inside one 50 ms budget.

This applies equally to the preview pass. A preview that appears all at once is
correct only when it genuinely cost less than 50 ms; the same code path must
chunk when it costs more. "Preview is cheap" is not a licence to bypass the
governor — it is a prediction the governor must verify.

> Observed violation: scalar preview presents every tile in one step
> irrespective of cost; the PyQtGraph FFT second pass is unchunked and freezes
> the UI until it completes.

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

Two specific altitude errors are in scope for the recovery work:

- **Too low.** The preview level is currently computed per tile inside the
  ladder's `plan()`, from that tile's own demand and retention. A round-level
  property is being decided tile by tile. It must be lifted to the round planner
  and passed down. Note the ladder still decides, per tile, whether that tile is
  *skipped* — that is correctly per-tile work (R2), and it must read the round's
  floors rather than re-derive them.
- **Too low.** The round levels are currently resolved by a batched sweep over
  *source slabs*, two at a time, running beside the tile pipeline and competing
  with it for workers. A single round-scoped value is being assembled from
  ~136 independent worker results. It must become one round-owned decision,
  and it should be derived from the preview cohort — those tiles already
  contain the round's data at reduced resolution, which is exactly and cheaply
  the evidence the window needs. The separate source-slab sweep re-reads and
  re-evaluates the data (including re-running the FFT) to learn something the
  preview pass has already computed.

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
no R1/R3 violation across the recorded interaction traces, and the relevant
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
