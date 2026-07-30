# The viewport rect convention — settled, and why `active` keeps its own margin

Dated 2026-07-30. Closes the two questions
[retention-halo-2026-07-30.md](retention-halo-2026-07-30.md) §4 left open and
[render-eagerness-2026-07-30.md](render-eagerness-2026-07-30.md) §1 opened:
whether `FrameRegion.active` should go through the snapped rect, and whether
the snapped convention is the right one given that it was discontinuous at
tile boundaries. They turned out to be one question with one answer, and the
answer is not the one either document expected.

---

## 1. The convention

> **`viewport_tiles(plan, view_range=, margin_tiles=N)` is the derivation: the
> tiles the camera rect touches, plus `N` whole-tile rings. The margin is the
> whole policy.**

Everything else about the derivation is convention, and both conventions are
now fixed so that neither can contribute a ring of its own:

1. **The rect is a function of the camera and the margin alone.** It took a
   `viewport_shape` from `f58cf98f` until now, and that parameter could never
   change the answer — the only expression reading it was
   `max(stride, min(viewport_px, stride))`, which is `stride`. It is gone. A
   parameter inside the one derivation that provably cannot change its answer
   is worse than no parameter, because the next caller sizes something to it.

2. **Snapping the rect outward to whole-tile bounds cannot change which tiles
   it touches.** `_expand_rect_to_tile_bounds` and `tiles_intersecting` now
   read one span computation (`MontagePlan._axis_span`), so the snap is
   idempotent by construction. This is the half that makes "the raw camera
   rect" and "the snapped camera rect" stop being two questions.

Four margins, all relative to the camera, all named:

| Constant | Rings | The question it answers |
|---|---:|---|
| `ACTIVE_MARGIN_TILES` | 0 | What is on screen. Gates completion. |
| `VISIBLE_MARGIN_TILES` | 1 | What the frame draws. |
| `COVERAGE_MARGIN_TILES` | 2 | What we ask to be produced. |
| `RETENTION_HALO_TILES` | 3 | What we keep and warm. |

**These are one larger than the numbers they replace and denote exactly the
same sets.** The rect used to add one tile stride per side that no constant
named, so `VISIBLE_MARGIN_TILES = 0` meant "the camera plus one ring" and
every margin below it was off by the same one. The rebase is arithmetic, not
policy: `montage_rect_for_viewport` now takes the margin explicitly and
`_rect_for_view_range` multiplies by it instead of hard-coding one stride.

---

## 2. The two things that were wrong, and what each was worth

### 2.1 The undeclared ring — worth the whole 6-against-15 discrepancy

§4 of the halo document recorded that `FrameRegion.active` reported **6** tiles
where `FrameSession.visible_tiles` reported **15** for one camera on the
272-tile field montage, at the same nominal margin 0, and attributed it to the
planner passing the raw rect where the session passed the snapped one.

That attribution is wrong. The snap contributes nothing on this camera. The
entire gap is the stride the rect added and no constant named: the session
went through the rect and got the ring, the planner bypassed the rect and did
not. Calling it a snapping difference is what made it look like a convention
problem rather than an accounting one.

### 2.2 The boundary — real, and not a performance defect

`_expand_rect_to_tile_bounds` solved the strict overlap test
`tile_x1 > rx0 and tile_x0 < rx1` while `tiles_intersecting` solves the
inclusive one. The two differ on exactly one input — a rect edge landing on an
integer multiple of the stride — and there the strict form drops a tile the
inclusive form keeps, which is a whole ring. On the field montage that is a
rect of 84 against 101 and a halo of 49 against 64, pinned by
`test_a_camera_edge_on_a_tile_boundary_widens_the_halo_by_a_whole_ring`.

**The claim that this jittered the halo during ordinary camera motion does not
survive measurement.** Over 400 camera spans × 300 drag steps at three tile
sizes, the strict form changes the halo set across a drag within 1% as often
as the inclusive one (16 px tiles: 2789 vs 2793; 64 px: 630 vs 630; 128 px:
317 vs 316), and neither produces a non-monotone dip except on a handful of
spans where both edges hit the boundary condition together. The strict form
was monotone too — it stepped one pixel later. A camera 2.5 tiles wide covers
3 or 4 tile columns depending on its sub-tile phase, and no rect convention
removes that.

So the boundary was fixed for what it costs elsewhere: it is what made
snapping non-idempotent, and a non-idempotent snap is what entitles a consumer
to hold a second convention. §4 below shows the one place where it does pay.

---

## 3. What changed about what is visible

The load-bearing risk, since `montage_rect_for_viewport` is shared with
`FrameSession.visible_tiles`. Measured directly: for a one-pixel-per-step drag
crossing five tile boundaries, every camera position, both conventions, all
four margins, three tile sizes.

| Set | Steps where the answer differs (16 / 64 / 128 px tiles) | Direction |
|---|---|---|
| on screen (`active`) | 5/85, 5/325, 5/645 | **+3 tiles**, never fewer |
| drawn (`visible`) | 10/85, 10/325, 10/645 | **+5 tiles**, never fewer |
| coverage | 10/85, 10/325, 10/645 | **+7 tiles**, never fewer |
| retention halo | 10/85, 10/325, 10/645 | **+9 tiles**, never fewer |

**The convention never narrows a set.** It widens by exactly one tile column,
on two of the steps around each tile boundary the drag crosses — 11.8% of
steps at 16 px tiles, 3.1% at 64 px, 1.6% at 128 px, i.e. two pixel positions
per stride regardless of tile size. Everywhere else the two conventions agree
tile for tile. The `active` row differs at only one step per boundary rather
than two, and for a different reason: it changed from the raw float camera to
the rect's whole-pixel rounding, so it gains a column when the camera edge
lands within a pixel of a boundary.

Nothing that was drawn stops being drawn, and nothing that was produced stops
being produced. That is the whole answer to "what is visible everywhere".

### The montage still completes, and first pixels did not regress

`profile_montage_workflow`, synthetic geometry scene, 256×256×64 at 8 columns,
PyQtGraph, ABBA across checkouts (main / this branch / this branch / main),
n=2 per arm. Sequential single-shot processes, so these are a **no-regression
check, not an A/B** — no improvement is claimed from them.

| | A (main) | B (this branch) |
|---|---|---|
| `raw_full_tiled_montage` first tile ms | 384.6, 381.9 | 369.9, 344.7 |
| `raw_full_tiled_montage` initial fill ms | 394.9, 393.4 | 393.5, 356.0 |
| `raw_full_tiled_montage` full refined ms | 974.2, 1040.1 | 1005.2, 987.1 |
| `raw_full_tiled_montage` tiles | 64/64, 64/64 | 64/64, 64/64 |
| `montage_zoompan_scalar` first tile ms | 979.1, 974.7 | 935.1, 921.6 |
| `montage_zoompan_scalar` tiles | 60/60, 60/60 | 60/60, 60/60 |

Both arms complete every tile in every run, and the progressive-render
invariant verdicts are identical between arms (`R2, R4, R7` on the fill;
`R1, R2, R5, R7` on zoom/pan, with the extra `R2b` and the
`deep_zoom_far_scroll` R8 entry appearing on the *first* run of either arm and
not the second — warm-up, not the convention). The zoom/pan initial fill
ranges 15.5–18.8 s across all four runs with no arm ordering, so it is
dominated by the machine and no claim is made about it.

---

## 4. The drag: where the unified derivation actually pays

The instrument is `FramePlanner`'s whole-tuple region reuse
(`montage_region_set_hits` / `_misses`), which is keyed on the *pair*
`(active ids, near ids)`. A drag of five tile boundaries, one pixel per step,
arms interleaved ABBA in one process, swept across seven sub-tile start
phases. The counts are identical at 16, 64 and 128 px tiles — this is grid
arithmetic, not timing.

| start phase | region-set misses, main → new |
|---|---|
| +0 px | 15 → **10** |
| +1, +3, +7 px, +stride/3 px | 16 → **11** |
| +stride/2 px | 16 → **10** |
| +stride−1 px | 15 → **11** |

**Fewer misses in every phase tested — roughly a third fewer — and the misses
that remain are exactly the boundary steps.** The reason is the pair. On main,
`active` transitioned at the float camera boundary and `near` transitioned at
the rect's, one pixel apart, so each tile boundary cost *three* whole-tuple
misses instead of two. Routing both through one derivation aligned the two
transitions. This is the same lesson the architecture guard's docstring
records from the convergence latch: the defect is invisible from either site
and only exists in the pair.

The residual two misses per boundary are not removable by any convention: the
set genuinely changes when the camera crosses a tile.

---

## 5. `FrameRegion.active` — through the one derivation, at its own margin

`active` now comes from `viewport_tiles(margin_tiles=ACTIVE_MARGIN_TILES)`, so
the architecture guard's exemption for `_plan_montage` is **removed**. So is
the exemption for `montage_viewport._nearest_visible_tile`, which was granted
on the grounds that it had no `viewport_shape` to snap with — the rect never
read `viewport_shape`, so that exemption was never buying anything.
`_RAW_RECT_TILE_SET_EXEMPTIONS` is down to `viewport_tiles` itself.

**It did not take the drawn set's ring, and that is measured rather than
argued.** Unifying `active` onto `VISIBLE_MARGIN_TILES` is the obvious move —
both sets answer "what does this camera show" — and it fails:

- `test_montage_tile_count_increase_preserves_manual_zoom_when_not_near_auto`
  times out at five seconds on **both** backends. The dump reports
  `active=(0, 1, 5, 6), unsettled=(5, 6)`, `commit_outcome='level-evidence-wait'`:
  the montage waits forever on level evidence for the two ring tiles it just
  adopted as required.
- `test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope` then
  fails in the framebuffer oracle with *6 of 12 required tiles* reporting
  `tile rect is off-target or degenerate` — the oracle cannot sample them
  because they are not on screen.

`FramePlan.active_region_ids` is what `FrameSession.required_tile_numbers`
reads, and that is documented as "the one set that admission, evidence, and
completion must render". Slack in it is not conservatism, it is a completion
obligation on tiles nobody can see. **Slack belongs in the drawn set.**

So the answer to §4's question is the second of the three shapes the brief
offered — do not give `active` the other consumers' margin — but the reason is
not the one that was expected. It is not that `active` does not need the
snapping; snapping was never the difference. It is that the ring the other
consumers carry is *slack for a camera that has moved since the plan was
built*, and the set that decides whether the frame is finished must not carry
slack.

---

## 6. What this did not settle

- **The drawn set's ring is inherited, not justified.** It has been in the
  rect since `f58cf98f` — "Phase 4e.p2: Implement viewport-sized montage
  rendering with memory budget management" — with no recorded reason, so every
  measurement this display has ever taken was taken with it. Naming it changed
  nothing; *removing* it would narrow the drawn, coverage and retention sets by
  a ring each and is a separate adjudication with its own evidence. The
  plausible reason it exists is that the backends hide every tile outside the
  drawn set, so a drawn set with no slack turns any plan/camera lag into a
  hole — plausible, not established.

- **`TilePresentationDelta.active_tiles` is still filled from two sets.** The
  planner fills it on the retarget-only path (`display_presenter.py:476`) at
  `ACTIVE_MARGIN_TILES`; the session fills it on every commit
  (`build_tile_presentation`) from `visible_tiles`, at `VISIBLE_MARGIN_TILES`.
  Those are now a *named* ring apart rather than an accidental one, and §5 says
  they cannot simply be equalised, because the field is doing two jobs:
  presentation scope, which wants slack, and the required set, which must not
  have it. Splitting them is the real repair and is not attempted here.

- **`_axis_span`'s inclusive bound treats each tile's exclusive right edge as
  inclusive**, which is harmless when the montage gap is ≥ 1 and makes
  adjacent tiles overlap by one unit when it is 0. `viewport_tiles` therefore
  takes the margin rect *before* the snap, which sidesteps it; the snap's own
  idempotence is asserted on the production geometry only.

- **`montage_rect_for_viewport`'s empty-intersection fallback** used to return
  a 1×1 rect at the origin, which is not a smaller answer but a different one
  — it teleports an off-montage camera's answer to the opposite corner. It now
  keeps the camera's own rect. Nothing depended on the old behaviour while
  `active` bypassed the function; routing `active` through it made the
  fallback answer "what does an off-montage camera owe the screen", where the
  truthful answer is nothing.

---

## 7. Tests

- `test_a_camera_edge_on_a_tile_boundary_widens_the_halo_by_a_whole_ring` is
  replaced by
  `test_a_camera_edge_on_a_tile_boundary_answers_the_same_as_half_a_pixel_past_it`,
  which asserts the new convention rather than the defect, at 272 tiles where
  the halo is a genuine subset.
- `test_snapping_a_rect_to_tile_bounds_cannot_change_which_tiles_it_touches`
  is the convention itself, swept one pixel at a time through several strides
  — the resolution the defect lived at, and invisible at any coarser step.
- `test_the_planner_and_the_session_derive_their_camera_sets_the_same_way`
  asserts the *derivation*, not a literal: `active` is exactly the margin-0
  answer and a proper subset of the drawn set.
- `test_a_camera_edge_that_only_touches_a_tile_still_requires_it` moved from
  the planner to `tests/display/test_montage.py` at margin 0. At the planner
  it had stopped being falsifiable — the tile a camera edge merely touches is
  inside every ring the plan carries, so no boundary convention could fail it.
- `test_multi_tile_montage_marks_active_and_near_regions_from_viewport` moved
  from a 2×3 montage to 9×9, because at 2×3 the retention halo is the whole
  montage and the assertion had become "all of them".

## 8. Reproducing

The pure-function A/B — arm A reconstructs main's composition (strict snap,
undeclared ring, margins one smaller) inside the current process, so both arms
run the same interpreter and the same plan objects, ABBA per phase — is
`rect_ab.py` in the session scratchpad; it is thirty lines of `old_expand`
plus the current module and is quicker to rewrite than to find. The
cross-checkout profiler runs are the command in §3 run from
`/home/thomas/projects/ArrayScope` and from the branch worktree alternately;
the machine's load average was 2.6–5.9 throughout, and the zoom/pan fill is
load-dominated at that level, which is why only the fill-stage numbers are
quoted as evidence.
