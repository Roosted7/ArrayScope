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

### The one place a superseded tile can still reach the screen

PyQtGraph's compact preview atlas is a single item covering many tiles, so it
cannot be hidden per tile. `hide_tiles` drops it only when no surviving tile
still depends on it; when it is load-bearing it stays, and
`_superseded_atlas_retentions` counts it. Hiding a load-bearing atlas would blank
the bridge this seam exists to preserve, which is a worse failure than the one it
would fix.

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
