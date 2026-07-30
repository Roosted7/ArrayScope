# Data-change invalidation: hide what is superseded, remove it later

Dated 2026-07-30. Held back until out-of-view retention existed on both
backends; it now does, and the seam it opened is real. Closes the "deferred
invalidation on data change" item left open by
[out-of-view-retention](out-of-view-retention-2026-07-30.md) §5, and builds on
the vocabulary in [render-eagerness](render-eagerness-2026-07-30.md) §6.

The requirement, in the user's words:

> When the data changes (ops, reload, changed indexing, different view mode) we
> need to invalidate and hide the out-of-view tiles, to ensure a quick pan never
> reveals stale tiles. **However, updating the actually currently visible tiles
> has priority over this** — if the extra out-of-view bookkeeping would cause any
> slowdown on what is visible, it waits. In-view must complete before
> out-of-view updates. Additionally, after that and when idle, the prefetch
> system should start getting those nearby tiles loaded.

Three parts. Two are done and measured. The third is measured and **blocked**,
with the blocker identified precisely; it is not shipped and §5 says why.

§8 is a follow-up of the same date: the one path §4 left open — PyQtGraph's
compact preview atlas — measured, and closed by masking it per tile.

---

## 1. The seam nobody was watching: `retarget_index_window`

The map named `invalidate_tiled_presentation(hide_pixels=True)` on a rejected
session transition as the blunt teardown. That is not where the defect lives.

**A montage data change usually never reaches that call at all.** When document,
layout geometry and every presentation input match — which is true of a slice
step, a montage index-window move, an axis-range crop, and a transpose under
canonical orientation — the controller takes `retarget_index_window`
(`frame_controller.py:1301`), an *in-place* rebirth: the `FrameSession` object
survives with a new `session_id` and key, and **nothing tells the surface to
stop drawing the predecessor**. The surface keeps its instances until some later
commit happens to narrow them.

Measured on a settled 64-source WGPU montage zoomed to 20 visible tiles, with
out-of-view retention holding the other 44:

| | before the change | after `render()` returned |
|---|---:|---:|
| tiles the surface is drawing | 64 | **64** |
| of which out of view | 44 | **44** |

A pure camera move onto that region — `setRange` only, no renderer call, so no
commit and therefore no per-tile predicate anywhere in the path — reproduced the
**previous slice pixel for pixel**: 100.0 % of the render target identical to a
capture of the same region taken before the change, mean lit brightness 51 where
the correct value is 210. The instances stayed drawn for ~310 ms.

Two things make this a WGPU defect specifically:

- instances are world-space, so a pan rewrites one camera uniform and nothing
  else (`TileInstance`, `command_protocol.py:174`) — there is no CPU code in the
  path to consult a staleness predicate;
- the pages those instances name are keyed by content, so the successor's upload
  goes to *different* pages and the old ones keep the old texels.

`cpu_item` is safe here by construction: it hides each `ImageItem` as its tile
leaves the viewport, so the same pan draws background. That is a property worth
keeping rather than a defect, and it is now pinned (§4).

### Why the retention that just landed made this worth doing now

Before out-of-view retention, a WGPU commit narrowed the presented set to the
viewport, so a bridge kept ~20 instances alive. With retention the presented set
is the viewport *plus* the retained halo — 42 at 64 tiles, 260 at 272 — so the
same seam now leaves an order of magnitude more superseded pixels standing.

---

## 2. The ordering: the removals were riding the visible path

The out-of-view half of the retarget is not free, and it is not a one-off. Every
tile whose source moved and is out of view took the miss branch, which ends in
`pending_removals.add(index)`. A queued removal re-enters the `removals` tuple of
**every** commit until the backend acknowledges it, and on a settled montage they
do not all clear.

Measured on 272 tiles, 24 visible, one slice step, counting only the commits that
ran *before* `visible_plan_complete()`:

| | commits before first visible pixels | out-of-view removals carried | visible upserts |
|---|---:|---:|---:|
| WGPU | 9 | **992** | 52 |
| PyQtGraph | 11 | **744** | 13 |

Every single one of those removals is work for a tile the user is not looking
at, performed on the path that owes the screen its new pixels. That is exactly
what the requirement forbids.

### What changed

`FrameSession.deferred_out_of_view_invalidations`. The retarget's miss branch
keeps all the cheap slot bookkeeping inline — popping `rendered_tiles`,
`tile_source_ids`, `display_tile_payloads`, forgetting the level generation,
`presentation_discarded` — because leaving any of that undone would let a later
pass treat superseded data as current. Only the removal is deferred.

The drain is bounded (`DEFERRED_OUT_OF_VIEW_DRAIN_BATCH = 32`), gated on
`visible_plan_complete()` and on interaction, deterministic (lowest tile first,
so a partially drained montage is reproducible), and self-limiting: a tile can be
in the queue at most once, and a tile the camera comes back to leaves it instead
of racing its own replacement upsert.

It has two owners, both of which gate identically because the drain gates
itself: the prefetch entry (invited after every commit) and
`_settle_montage_visible_plan_if_complete` — which is the moment the ordering
gate opens, and is what makes the queue converge on a montage that has stopped
committing. Without that second owner the tail sat at 248 forever once the
montage settled.

### Measured

Same 272-tile montage, single-arm:

| out-of-view removals reaching a commit before `visible_plan_complete()` | before | after |
|---|---:|---:|
| WGPU | 992 | **0** |
| PyQtGraph | 744 | **0** |

Asserted directly, on both backends, in
`test_data_change_serves_the_visible_set_before_any_out_of_view_removal`.

---

## 3. What it cost the visible path — nothing measurable

Interleaved in-process A/B, one process, one window, arms alternated by the
runtime `FrameSession.defer_out_of_view_removals` flag. 272 tiles at 64 px, 24
visible, eight slice indices each measured once per arm per pass with the arm
order mirrored between passes (so the second visit's display-cache warmth
cancels), four passes, **n = 16 per arm per run, three runs**. Metric is
first-pixel latency for the visible set: the wall time from the data change to
`visible_plan_complete()`.

| median, ms | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| WGPU inline (main) | 42.6 | 59.0 | 35.1 |
| WGPU deferred | 47.9 | 52.8 | 35.0 |
| PyQtGraph inline (main) | 37.1 | 33.5 | 33.3 |
| PyQtGraph deferred | 38.4 | 35.1 | 37.9 |

The direction flips between runs on WGPU. The distributions are bimodal — a fast
cluster around 30–45 ms and a slow one at 90–400 ms, which is cold versus
display-cached slices — and both arms land 9 fast / 7 slow in every run, so the
median is decided by where the boundary happens to fall. Pooling the fast
clusters across all three runs gives **36 ms inline vs 38 ms deferred on WGPU**
and **≈33 vs ≈36 ms on PyQtGraph**. Fully overlapping ranges.

**The honest statement is that first-pixel latency is unchanged within noise.**
The measured result of this change is the counter in §2, not a millisecond.

One tail observation, recorded because it repeated and not claimed as a result:
the **inline** arm produced a ~5.1 s outlier in all three PyQtGraph runs
(5148, 5147, 5125 ms — the settle cap), and the deferred arm's worst trial in
those same runs was 224, 199 and 214 ms. n=3, on a machine carrying background
load from another checkout's test run throughout; suggestive, not evidence.

---

## 4. Staleness: one predicate, not two

Nothing here adds a second notion of stale. The seam **hides**; it never
resurrects. A tile it hid returns only through the ordinary identity-checked
upsert path, and the existing guards are what refuse superseded pixels if
anything asks for them meanwhile:

- `acknowledged_identity_satisfies_target` gates `_reshow_retained_active_tiles`
  on `cpu_item`, and a superseded state's acknowledged identity names the old
  `semantic_key`;
- `payload_compatible_with_tile`'s `semantic_generation` guard
  (`montage_payload_cache.py:169`, ADR 0050) is what the WGPU retention
  predicate already consults per commit;
- the retained payload store is keyed by the evaluator's tile key, which embeds
  the document and the operation, so an ops change misses it outright.

### The precision that turned out to be load-bearing

The first version passed the *viewport* and let the backend hide everything
outside it. That reads as equivalent and is not:
`retarget_index_window` is also reached by a **camera move** — `frame_session_key`
includes the viewport plan — and hiding on one of those undoes out-of-view
retention entirely. It was caught by
`test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope` going from 0
to 1 whole-montage republication per pan leg. The contract now takes the
superseded tile *set*, which for the retarget is exactly the tiles that retarget
just deferred: a camera move that supersedes nothing hides nothing.

### The one place a superseded tile could still reach the screen — closed

PyQtGraph's compact preview atlas is a single item covering many tiles. The
first version of `hide_tiles` dropped it only when no surviving tile still
depended on it, and otherwise kept it drawing and counted
`_superseded_atlas_retentions`. That was named as the one path on `cpu_item`
where a superseded tile could still reach the screen. It is now masked per
tile instead, and §8 records what that cost and how often it bit.

---

## 5. Prefetch: checked, and **not** unblocked

[retention-halo-2026-07-30.md](retention-halo-2026-07-30.md) §3 measured warming
the halo from prefetch as a 2.8× pan regression and reverted it, naming as the
blocker that a warm writes into `session.display_tile_payloads` for a tile
outside `planned_numbers`. It listed out-of-view retention as one of the two
things that would unblock it. Retention has landed, so this was checked rather
than assumed.

**First, a different gap, measured.** After a data change the halo is not warmed
at all — and not because the warm fails, but because **prefetch is never
invited**. Every invitation hangs off a commit, so a montage that finished
converging after a slice step has none left. On a settled 272-tile montage with
a 24-tile halo ring: `prefetch_scheduled = 0` and the ring **0 of 24**
display-cached through a second of idle, with the last recorded prefetch
decision stale from before the change.

**Second, adding the invitation reproduces the rejected regression.** Inviting
the walk from `_settle_montage_visible_plan_if_complete` — the exact moment the
product rule names — cost
`test_wgpu_pan_away_and_back_reuses_the_retained_presentation_scope` one
whole-montage republication per pan leg, **0 → 1**.

**Third, bisected.** With the invitation kept and only
`_warm_prefetched_tiled_residency` disabled, the republication goes away and the
test passes. So the display-cache half of the walk is free; the **backend
residency warm** is the whole cost.

Retention cannot rescue it, and the reason is in the predicate rather than in the
measurement: `_may_retain_out_of_view` requires
`backend_identities[tile] == tile_ack_identity(payload)` — the backend must
already be *drawing* the payload. A warm never is, by definition. The retention
that landed protects tiles the commit already presented; it has nothing to say
about a tile speculatively deposited into the payload map.

**Not shipped.** The invitation alone is a regression, and the only way to take
it today is to stop warming backend residency — which is not a gate to add but a
production path to retire, and every prefetch candidate is out of scope by
construction (`_candidate_tiles` excludes the visible set, `planned_numbers` is
the visible set), so gating it is deleting it. That is a separate adjudication
with its own measurement, not a line in this change. The next increment needs
either a non-mutating payload build for speculation, or the retirement of the
residency warm — which is the same fork
[retention-halo-2026-07-30.md](retention-halo-2026-07-30.md) §3 ended on, now
with the second branch closed off and the first one measured as cheap.

---

## 6. Coverage

- `tests/ui/test_montage_data_change_invalidation.py` — the live path. The WGPU
  case is a physical oracle: a pure camera move onto the superseded region, read
  back from the executor target and compared against a settled capture of the
  previous data, carrying its own mutant (the same region settled on the new data
  must NOT match, or the comparison could never fail). The `cpu_item` case is
  labelled in its own docstring as a **guard, not a reproduction**. The ordering
  case runs 272 tiles against 24 visible on both backends. All three assert the
  montage **completes** — the R2 pathology quoted in the progressive-render
  contract (settled at 135 of 272 forever, no stall firing) passes straight
  through an assertion that only checks what stayed visible.
- `tests/window/test_deferred_out_of_view_invalidation.py` — queue, bound,
  determinism, release-on-return, and that repeated data changes cannot grow the
  queue past the tile count.
- `tests/display/test_presented_visibility_equivalence.py` — the layer contract:
  `hide_tiles` hides the named tiles, leaves the rest drawing, and keeps states
  and pixels.

Two fixture traps had to be cleared first, both of which had already produced a
green result that could not fail:

- **Pre-warming the target slice makes the defect vanish.** With every visible
  tile a display-cache hit, the commit inside `render()` publishes the successor
  immediately, which narrows the drawn set as a side effect and closes the stale
  window before a test can look at it. The fixture now warms the pipeline on a
  *third* slice the tests never display again, so the measured change is still a
  cache miss.
- **A tile-local pixel difference is invisible at montage scale.** The first
  fixture distinguished slices by mirroring each tile in x — a difference of at
  most 95 against a global auto-window span of 315 000, i.e. under one grey
  level. The slices now differ by source ORDER, so the value distribution is
  identical (an auto-window change cannot be mistaken for a content change) while
  every tile's brightness differs.

And one fixture cost, paid rather than hidden. At 272 tiles with 24 visible, the
cold settle after a data change sat right on the interaction cap and these tests
joined the documented `tests/gpu` + `tests/ui` contention casualties (about one
run in three timed out even serially). The two stale-reveal tests moved to a
64-tile grid — they need a superseded out-of-view set and a legible render
target, not a big one — and the ordering test keeps 272 with a 9-tile viewport
and its target slice warmed for the VISIBLE set only, which does not touch the
subject: cached tiles reach the retarget for the visible display tiles alone, so
every out-of-view tile takes the miss branch either way, and the test's own
"measuring nothing" guard fails outright if that stops being true. Nine serial
repetitions of the ordering test and eight of the whole file are clean; the
mutant check (the runtime flag off) still fails three of the four.

## 7. Reproduce

Both A/Bs are the runtime `FrameSession.defer_out_of_view_removals` flag toggled
between legs in one process. Counters used: `deferred_out_of_view_deferred` /
`_drained` / `_returned`, `_wgpu_binding_full_republications`, and a wrapper
around `build_tile_presentation` that classifies each delta's removals against
`visible_tile_numbers`. No per-commit instrumentation was added to production.
Every number here was taken on a machine whose runner banner reported 4–10 of 16
cores in use by another checkout's test run; the counters are unaffected by that,
the milliseconds are why §3 claims nothing.

---

## 8. The preview atlas, closed (2026-07-30, follow-up)

§4 left one path open and named it honestly. This closes it, and the first
result is that the hole is smaller than it reads.

### How often it actually bit: never, in any settled sequence

`_superseded_atlas_retentions` was never non-zero in a realistic sequence, and
the reason is structural rather than lucky:

- out-of-view payload retention is `gpu_atlas`-only
  (`FrameController._retains_out_of_view_payloads`), so on `cpu_item`
  `delta.active_tiles` **is** the viewport;
- every commit already narrows the atlas to `active_tiles ∩ requested_active`
  (`_update_direct_payload_presentation`), so an atlas member outside the
  viewport does not survive a commit.

Sampled continuously — 1 195 samples over a zoom ladder (full → half → quarter
→ corner → full, each leg settled), four settled slice steps, and six mid-fill
scrub rounds on a 272-source montage — the set `preview_atlas_active_tiles −
visible_tile_numbers` was **empty at every single sample**, six `hide_tiles`
calls landed with **no** atlas up, and the counter finished at **0**.

The one state that does produce it is a camera move with **no commit** between
it and the data change, while the atlas is still up:

```
atlas up after 5 pumps: active=23 visible=272
after camera-only pan:  atlas_active=23
  hide_tiles(n=244) atlas_up=True active=23 stale=10 -> retention+=1
after data change: visible=28 atlas_oov=10 drawn_oov=10 retentions=1
```

That is a user panning or zooming a large montage while its first fill is still
arriving and then stepping the slice — narrow, but a real interaction and not a
contrivance, and 10 superseded tiles kept drawing.

### What was built, and why that shape

**Per-tile masking inside the atlas item.** Not because it was the most
promising of the three shapes but because it turned out not to be new code:
`_CompactPreviewAtlasItem.paint` already iterates `active_tiles` and draws each
member's own parts, and `set_active_tiles` is the narrowing every ordinary
commit already performs. Withdrawing the superseded members is that same
operation, so the branch collapses from *drop-or-retain* to one line.

The two alternatives were both worse for stated reasons. **Splitting the atlas**
buys nothing the paint loop does not already give. **Dropping it whenever any
member is superseded** was the simplest and is the wrong trade: the atlas exists
to avoid 256+ individual items, and a data change on a large montage is exactly
when that cannot be paid back on the path that owes the screen new pixels —
while the retained-and-counted alternative it replaced was the one that put
stale pixels on screen. Masking is the only shape that refuses both.

An atlas masked down to nothing is hidden rather than removed, which keeps its
pages for the way back and matches the seam's contract: hide, never remove.

`_superseded_atlas_retentions` is deleted with the branch it counted.

### The visible-path cost: nothing measurable, and one commit either way

Interleaved in-process A/B, one process, arms alternated per pass with the order
mirrored between passes. The control arm is main's `hide_tiles` verbatim. The
regime is the only one where the two arms differ at all — atlas up, superseded
out-of-view members present at the data change. Metric is first-pixel latency
for the visible set (data change → `visible_plan_complete()`), the commits that
ran before it, and the time inside `hide_tiles` itself.

| median | run 1 (n=6/arm) | run 2 (n=8/arm) |
|---|---:|---:|
| first pixel, main | 192.6 ms | 179.9 ms |
| first pixel, masked | 182.7 ms | 182.0 ms |
| commits before first pixel, both arms | **1** | **1** |
| `hide_tiles` ms, main | 2.098 | 2.030 |
| `hide_tiles` ms, masked | 2.067 | 1.924 |

The direction flips between runs and the ranges overlap completely, so the
honest statement is again that **first-pixel latency is unchanged within
noise**. The commit count is identical and exact. `hide_tiles` itself is
dominated by the ~200 per-tile `_hide_tile` calls either way; the atlas branch
is a set difference in both arms, and the extra `update()` only marks an already
installed item dirty — the repaint that follows draws *fewer* `drawImage` calls
than the one before it, so masking cannot make the paint more expensive.

Both runs' banners reported 4–5 of 16 cores in use by another checkout.

### Coverage

- `tests/ui/test_montage_data_change_invalidation.py::test_cpu_item_preview_atlas_stops_drawing_what_a_data_change_superseded`
  — the live path, and a pixel oracle: after the data change the camera moves
  back onto the superseded block with no renderer call and the viewport raster
  must be scene background. Against the unfixed code it fails with **90.1 % of
  the viewport not background**. Its regime is load-bearing twice over — the
  atlas needs ≥ 256 planned tiles *and* sources large enough that fitting the
  montage demands a reduced LOD, so the file's existing 32 px grids cannot
  reproduce it at either count. It carries its own mutant (the same block, once
  settled on the new data, must be lit and must not match the old capture) and
  asserts the montage completes.
- `tests/display/test_pyqtgraph_preview_atlas.py::test_hide_tiles_masks_superseded_atlas_members_and_keeps_the_survivors`
  — the layer contract at 256 tiles, also as pixels: the masked member's patch
  of the real PyQtGraph scene goes to background while a survivor keeps painting
  its value out of the same item, which is what separates masking from dropping.

Both fail on their pixel assertion against main, not on bookkeeping — the
assertion order was changed once for exactly that reason, because the first
draft's membership check fired first and would have let a green result rest on
something other than the screen. Five serial repetitions of the whole
`test_montage_data_change_invalidation.py` file are clean, and `--since main`
reaches 86 tests with none left to run.

One oracle trap cleared: `_lit_fraction`, the WGPU render target's "brighter
than black", does not carry over. PyQtGraph's scene background is a light grey
(236, 238, 240), so it reads 1.0 for an empty viewport — the first version of
the test failed identically with and without the fix. The PyQtGraph oracle asks
the view for its own `backgroundBrush()` colour, which is what the Qt-raster CPU
reference already does.
