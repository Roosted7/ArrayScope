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
- **Preview pass** — an optional, coarser, faster pass over the round's tiles.
- **Target pass** — the pass at the round's desired LOD. Always required.
- **Presented tile** — a tile whose pixels are currently on screen.
- **Round levels** — the window/levels the round's pixels are mapped through.

## R1 — At most two qualities on screen

At every instant, the distinct LOD levels among presented tiles number **at
most two**. When there are two, they are exactly the round's preview level and
the round's target level.

There is no third rung. There is no "slightly better than preview but not yet
target". A tile retained from an earlier round at some older level counts
against this bound like any other presented tile: if it does not sit at this
round's preview or target level, it must be re-presented at one of them or
dropped.

> Observed violation: WGPU FFT held levels `{0, 2, 5}` simultaneously
> (11 tiles retained at native level 0 across a preview-then-target fill of
> 272 tiles), and WGPU scalar held `{0, 1, 4}` through fast scroll.

## R2 — The preview level is one number per round

A round has **one** preview level, chosen once, before any tile of that round
is scheduled. Every tile that takes the preview pass uses that level.

Per-tile derivation of the preview level is forbidden. Two tiles in the same
round must never land on different preview levels because their individual
demand or retention state differed.

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
| The preview level (R2) | round planner | per round, one value |
| The target level | round planner | per round, one value |
| Round levels / window (R3) | round levels owner | per round, one value |
| Which rung a given tile still needs | ladder | per tile |
| Chunk sizes and pacing (R5) | governor | per pass |
| Load shedding (R6) | scheduler | per round |
| Baking or binding levels into pixels | backend adapter | per tile, **reads** the round value |

Two specific altitude errors are in scope for the recovery work:

- **Too low.** The preview level is currently computed per tile inside the
  ladder's `plan()`, from that tile's own demand and retention. A round-level
  property is being decided tile by tile, which is mechanically how three
  qualities end up on screen at once. It must be lifted to the round planner
  and passed down.
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

## Acceptance

A change to the progressive path is accepted when the invariant oracle reports
no R1/R3 violation across the recorded interaction traces, and the relevant
suites are green. Latency medians are reported alongside, but a median may
never be traded for an invariant.
