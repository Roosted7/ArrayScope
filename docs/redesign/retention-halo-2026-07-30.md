# One retention halo — and the measured rejection of warming it

Dated 2026-07-30. Follows
[render-eagerness-2026-07-30.md](render-eagerness-2026-07-30.md), which mapped
the three inconsistent halos (§1) and the prefetch that is not a halo-filler
(§5). This document records what was unified, what the numbers said about
warming the halo from prefetch, and why that half was **not** shipped.

---

## 1. What the halo is now

Three constants in `arrayscope/display/montage.py`, and one derivation
function that every consumer calls:

| Name | Rings | What it decides |
|---|---|---|
| `VISIBLE_MARGIN_TILES` | 0 | What the frame owes the screen. Widening it widens first-pixel latency. |
| `COVERAGE_MARGIN_TILES` | 1 | What we ask to be **produced**. Each tile costs an evaluation, a reduction and an upload. |
| `RETENTION_HALO_TILES` | 2 | What we **keep and warm**. Scheduling order, which in-flight work survives a move, which payload eviction gives up last. Costs bookkeeping, not production. |

`viewport_tiles(plan, view_range=, viewport_shape=, margin_tiles=)` is the one
derivation. The margin was never the whole policy: it snaps the camera rect
outward to whole-tile bounds and clips it to the montage *first*, and a caller
that skipped that step got a different set for the same camera and the same
margin.

**Why retention is two rings and coverage is one.** A pan of one tile step
turns the old ring 1 into the new drawn set, so a one-ring halo is spent
exactly by the move it exists to absorb. At two rings the old ring 2 still
leads the new viewport. This is asserted directly in
`test_retention_halo_survives_the_pan_step_that_coverage_only_absorbs`, not
argued.

### The differences that remain, and why

- **Coverage stays at 1 while retention is 2.** Coverage buys production;
  retention buys bookkeeping. They are priced differently, so they are sized
  differently.
- **The resize path asks for coverage 0, retention 2.** A resize is a live
  drag: it must not enlarge the production request for a camera the user is
  still moving. It has no reason to drop the halo.
- **A non-persistent-residency backend gets coverage 0, retention 2.** A
  backend that cannot hold a tile past the viewport should not pay to produce
  one; retention costs nothing either way. This branch is also what a *failed
  capability probe* returns (`image_view_backend_capabilities` answers with a
  bare default for a `None` view), so leaving the halo out of it made the halo
  vanish during teardown rather than on some deliberately halo-less backend.
  Both shipped backends declare `persistent_tile_residency`.

### The two accidents that are gone

1. **`near` meant two different sets.** The planner used margin 1 on the raw
   camera rect; the session used margin 2 on the snapped rect. Both fill
   `TilePresentationDelta.near_tiles` (planner on the retarget-only
   presentation path, session on every commit) and
   `LodAdmissionScope.near_tile_numbers` (planner first, session as fallback),
   so what `near` denoted depended on which path last produced it. Measured on
   the 272-tile field montage zoomed to a few tiles: **the planner called 15
   tiles near while the session protected 45.**

2. **The resize path's silent 0/0.** Traced to "Reduce VisPy zoom path
   further" (`19d36f45`), alongside `_cancel_vispy_speculative_work` and
   `priority_retarget_limit=1`. Both companions were deleted with VisPy
   (`54d1fd5b`, `934081fb`); this line stayed. Its cost was the halo half, not
   the coverage half: `retarget_viewport` feeds `near` straight into
   `retarget_tile_priority`, so **every window resize replaced the session's
   one priority context with an empty NEAR band** — flattening the ordering
   that prefetch candidates and per-commit upsert admission both read — until
   some later camera move happened to rebuild it.

### What widening the planner cost

The planner's `near` went from margin 1 to margin 2 (and from the raw rect to
the snapped one), which is a widening and therefore owes a number rather than
an argument. Same harness as §3 — 272 tiles at 128×128, camera on ~2.5 tiles,
fresh window per trial, ABBA, n=6 per arm. Arm A removes the halo everywhere
past the drawn set (a *stronger* revert than main, which kept the session at
margin 2); arm B is the unified halo.

| | A (no halo) | B (unified) |
|---|---|---|
| `near_region_ids` / priority NEAR band | 15 | **45** |
| pan `items_updated` | 9 | **9** |
| pan `image_replacements` | 9 | **9** |
| pan commits | 5 | **5** |
| pan `upload_ms` | 3.8 | 3.7 |
| idle max GUI stall ms | 0.12 | 0.14 |

`items_updated` was exactly 9 in **all twelve trials**, in both arms; commits
were 5 in eleven of twelve. Tripling what the halo denotes changes no pan work
and no GUI stall, which is what "retention costs bookkeeping, not production"
predicts — `near` is a set-membership bool per region and feeds only ordering
and retention bookkeeping, never an evaluation. Cold-fill time was too noisy on
this machine to adjudicate (6.0–16.9 s across both arms, medians 8.0 vs 8.4);
it is dominated by machine load, not by the arm, and no claim is made about it.

---

## 2. Prefetch ordering: already halo-first, once the band is populated

`_candidate_tiles` ranks by `tile_priority_key`, whose leading component is
`TilePriorityBand`, and the NEAR band *is* the retention halo. So the idle
walk already visits halo tiles before the rest of the montage — conditional on
the session's priority context carrying a halo, which is exactly what the
resize accident above destroyed. No ordering change was needed; the fix was to
stop blanking the band.

Asserted end-to-end in
`test_idle_prefetch_walks_the_retention_halo_before_the_rest_of_the_montage`,
on a montage where the halo is a genuine subset.

---

## 3. Warming the halo from prefetch: measured, rejected

### The gap that looked real

A display-cache hit ends a tile's walk (`montage_prefetch.py`, the `hit`
branch) before reaching `_warm_prefetched_tiled_residency`. So prefetch can
only ever warm **backend** residency for tiles that are display-cache
*misses* — and a montage that opens fit-to-window has already evaluated every
tile, so after the first fill every candidate is a hit and the walk warms
nothing for the rest of the session. Measured directly: on a 272-tile montage
at 64×64 tiles, after the initial fit and a zoom to 15 visible tiles,
`prefetch_scheduled = 0`, with all 272 tiles display-cached and the 30-tile
ring **0 of 30 backend-resident**.

The change tried: inside the halo, spend the idle batch on warming residency
from the already-cached payload — no evaluation, no worker, no lane — behind
every existing gate (`blocked_interaction`, busy, memory pressure, the 1/2/4
idle batch limit).

### What it measured

272 tiles at 128×128 float32, 800×800 offscreen viewport, camera on ~2.5
tiles (15 visible, 30-tile ring), fixed idle budget, then a pan of 40% of the
span. **Interleaved in-process, ABBA-balanced, fresh window per trial**, n=6
per arm. Arm A reproduces the pre-change `hit` branch.

| | A (main) | B (warm) |
|---|---|---|
| halo warms performed | 0 | 6 |
| ring tiles resident after idle | 0 / 30 | **2 / 30** |
| pan `items_updated` | **9** | **25** |
| pan `image_replacements` | **9** | **25** |
| pan commits | **4** | **18.5** |
| pan `upload_ms` | **4.4** | **15.8** |
| pan total ms | 537 | 634 |
| idle max GUI stall ms | 0.22 | 0.91 |

Medians. The signs are consistent across all six trials in each arm (five of
six arm-B trials report exactly 25 `items_updated`, five of six arm-A trials
exactly 9).

**The pan got 2.8× more expensive, and the halo barely warmed.** This is a net
loss and it was reverted.

### Why — and it was predicted

`_warm_prefetched_tiled_residency` reaches `session._ensure_display_tile_payload`,
which **writes into `session.display_tile_payloads`** (`frame_session.py:2486`).
Halo tiles are outside `planned_numbers` (the margin-0 visible set minus
skipped), so the next commit's stale-payload sweep (`frame_session.py:3691`)
pops each one and queues it as a `pending_removal`. Every warmed halo tile
becomes churn on the following commit.

This is verbatim §3.3 of the eagerness map:

> Retaining an out-of-view tile keeps it outside `planned_numbers`, which makes
> that branch unconditionally take the *remove* path. A retention change that
> does not also widen this predicate will retain tiles that the very next
> commit removes again, and will measure as flat.

It does not measure flat — it measures worse, because each retained-then-
removed payload costs a commit.

### What the prefetch half needs before it can work

The warm is on the wrong side of a seam it does not own. It needs **either**:

- a non-mutating payload build, so speculation can hand the backend a payload
  without depositing it in the session's commit-scoped map; **or**
- the `may_remove_visible` / `planned_numbers` predicate widened to the
  retention halo — which is the in-flight work retaining out-of-view payloads
  on the WGPU path.

Until one of those lands, warming the halo from prefetch makes panning worse.

> **Re-checked 2026-07-30 after out-of-view retention landed, and it is still
> blocked** — see
> [data-change-invalidation-2026-07-30.md](data-change-invalidation-2026-07-30.md)
> §5. Retention does not qualify as the second option above: `_may_retain_out_of_view`
> requires the backend to be **drawing** the payload already, which a speculative
> warm never is. Two new facts from that check. First, after a data change the
> halo is not warmed at all and the reason is upstream of the warm — prefetch has
> no invitation left once the montage stops committing (`prefetch_scheduled = 0`,
> ring 0 of 24 through a second of idle). Second, adding that invitation and
> bisecting isolates the cost precisely: with the invitation kept and only
> `_warm_prefetched_tiled_residency` disabled, the retention test's extra
> whole-montage republication per pan leg disappears. The display-cache half of
> the walk is free; the residency warm is the whole price.

---

## 4. What this did not touch

- **`FrameRegion.active` still disagrees with `FrameSession.visible_tiles`.**
  Same camera, same margin 0: the planner reported **6** tiles where the
  session reported **15**, because the planner passes the raw camera rect and
  the session passes the snapped one. Only `near` was unified here; `active`
  drives presentation scope on the retarget-only path and widening it is a
  separate, riskier change. Recorded, not fixed.

  > **Settled the same day, and the diagnosis above is wrong.** The snap
  > contributes nothing on that camera: the whole 6-against-15 gap is a tile
  > stride the viewport rect added that no constant named, so the session got
  > a ring the planner did not. `active` now goes through `viewport_tiles`
  > like everything else, at its own named margin of zero rings, because
  > giving it the drawn set's ring makes completion wait on off-screen tiles
  > and the montage stops converging. The margin constants are all one larger
  > and denote the same sets. See
  > [viewport-rect-convention-2026-07-30.md](viewport-rect-convention-2026-07-30.md).
- **`LodAdmissionScope.near_tile_numbers` is written and never read.** It is
  constructed in `frame_runtime.py` and normalized in `stages.py`; no
  production code consults it. Material for queue row 12.
- Neither backend's eviction predicate consults the halo. That is the standing
  gap §1 of the eagerness map named, and it belongs to the PyQtGraph victim
  selection and WGPU payload retention work.

### A latent flake this work walked into

`tests/ui/test_roi_inspection_interactions.py::test_hidden_inspection_panel_uses_tiled_frame_payloads_and_opening_populates_dock`
fails intermittently, serially, on **both** main and this branch. Measured on
the same three-test invocation, alone, one process at a time: **main 2 of 4
failed, this branch 2 of 6.** Inherited, not introduced.

It is the PyQtGraph twin of the strict-xfailed WGPU hidden-ROI defect in
[queue.md](../queue.md), which the queue records as passing. The shape fits the
same trigger: the test waits on `_compute_roi_inspection_snapshot` having been
called once and then immediately asserts `_roi_info_panel is not None`, but the
panel is populated from `setRoiInfoRows` off `_set_committed_display_frame`
(`display_presenter.py:527`), which is a different event-loop turn. Under load
the assert can land first.

A second `--since` run flagged a different single test,
`tests/ui/test_resident_crop_rebind.py::test_rebind_and_evaluation_paths_settle_identical_levels`
— in the file `main` is documented to carry 19 of its 25 pre-existing reds in.
Same treatment, serially, one process at a time: **main 2 of 5 failed, this
branch 1 of 3** -- comparable rates, and neither arm is clean. The failing runs take ~28 s
against ~10 s for the passing ones, so it is a settle timeout, not a wrong
answer.

That the two `--since` runs flagged *disjoint* sets is itself the signal: a
regression does not move.

Two other tests in the same `--since` selection —
`test_wgpu_expanded_montage_never_hides_retained_sixty` and
`test_single_slice_to_full_montage_presents_around_retained_exact_tile[scalar]`
— reported only under `-n auto` and pass serially on both branches: the
documented `tests/gpu` + `tests/ui` contention, reproduced here by running two
pytest processes at once, which is worth not doing.

### One correction to the eagerness map

§5 says the residency warm "warms one tile then immediately hides it
(`tiles.py:2134`) straight back into the reuse pool". The hide is real; the
pool return is not. `tiles.py` calls `_remove_from_direct_reuse_pool(item_state)`
and `_touch_resident_state(item_state)` immediately after, with a comment
saying warm residency is deliberately kept out of the generic pool. Also, §1's
table implies the halo is conditional on `persistent_tile_residency`;
`PYQTGRAPH_CAPABILITIES` declares it `True`, so both shipped backends were
already on the margin-1/2 policy and only the fallback branch was halo-less.

---

## 5. Reproducing

Both A/Bs are in-process, ABBA-balanced, fresh window per trial. The first
version of the harness was wrong in two ways worth recording: it reused one
window, so ring residency accumulated monotonically across trials (0 → 28 and
never reset), and the pan cost saturated to zero after the third trial. Both
made every arm look identical for reasons that had nothing to do with the
arms. A null run — twelve trials with the arms wired to the same code — is
what exposed it, and is worth running before trusting any A/B here.
