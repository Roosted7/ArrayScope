# Render-side eagerness — what a viewport change costs today

Dated 2026-07-30. **Map only.** Nothing here is a decision; every numbered
finding is a citation of current `main` behaviour, written so the work that
follows can be scoped without re-deriving it. Where a claim is an inference
rather than a read line, it says so.

The compute/operations side is deliberately lazy: work happens when a result
is demanded. The render side is the opposite in three distinct ways, and they
are usually conflated:

1. **Teardown on a pure camera move** — a pan or zoom that changes no data
   destroys work that was still valid.
2. **Whole-montage recomputation per camera move** — bookkeeping that is
   O(all tiles) or O(all lifecycle records) even when one tile changed.
3. **No retention halo** — the strict viewport is the only protected set, so
   the tile you are about to pan onto is the first one thrown away.

The normative constraints on any change here are in
[`../architecture/progressive-render-contract.md`](../architecture/progressive-render-contract.md).
R2's "floors are minimums: show what you have, never redo it" is the rule this
document is measuring the implementation against; R2's *"Skipping production
is not skipping presentation"* subsection is the trap every retention change
must clear.

---

## 1. The three tile sets, and their three inconsistent halos

`MontagePlan.tiles_intersecting(view_range, margin_tiles=N)`
([`montage.py:140`](../../arrayscope/display/montage.py)) is the only primitive.
The rect it receives is already snapped outward to whole-tile bounds
(`_expand_rect_to_tile_bounds`, `montage.py:434`), so `margin_tiles=0` still
means "every tile with any overlap".

| Consumer | margin | Source |
|---|---|---|
| `FrameSession.visible_tiles` / `active` | **0** | `frame_session.py:1601` |
| `coverage` (addition candidates) | **1** | `montage_viewport.py:720` |
| `near` (priority band, warm residency) | **2** | `montage_viewport.py:720` |
| `FramePlanner._plan_montage` `near` | **1** | `frame_planner.py:277` |
| `_retarget_montage_resize_payloads` | **0 / 0** | `frame_controller.py:455` |
| `_prune_stale_montage_tile_work` | 2 | `frame_controller.py:1608` |

Two different things are called `near` (margin 2 in the session, margin 1 in
the planner), and the resize path silently uses no halo at all. Whatever the
right retention radius is, it is currently three numbers.

**Neither backend's eviction predicate consults any of them.** `near_tiles` /
`near_tile_source_ids` reach the PyQtGraph layer and are used for exactly one
thing: the `near_resident_items` diagnostics counter
(`backends/pyqtgraph/tiles.py:2175`).

---

## 2. PyQtGraph (`cpu_item` residency)

### 2.1 Leaving the viewport is cheap — and correct

The sweep at `tiles.py:1708` hides every `_states` entry outside the final
`active` set via `_hide_tile` (`tiles.py:2300`): `setVisible(False)`, and the
state is pushed onto `_direct_reuse_pool`. **Pixels, `ImageItem`, scene
membership and `rgb_base`/`hist_source` are all kept.** This part already does
what the lazy principle asks.

### 2.2 Returning is cheap — when the state survived

`_direct_state_matches` → `resident_current=True` (`tiles.py:1483`) forces
`dirty=False` and suppresses upload (`tiles.py:1523`, `1541`). Re-showing is
`setPos` + `_apply_item_lod_scale` + `_set_item_visible(True)`. **No
`setImage`, no QImage rebuild, no scene insertion, no LUT re-application.**
`RetainedTiledPayloadStore.resolve(lod_factor=None)` plus `reuse_any_lod`
(`montage_payload_cache.py:88`, `frame_controller.py:1335`) is the ADR 0050
mechanism that already lets a retained coarse tile satisfy a finer round —
i.e. the "the tile I have is good enough for preview" case the product wants
is implemented at the payload layer.

### 2.3 The defect: the victim policy is LIFO, so it destroys exactly the
### tile you are panning back to

A newly visible tile with no `_states` entry hijacks a pooled state:

```python
# tiles.py:1579  (and again at :2060)
item_state = self._pop_direct_reuse_pool() if self._direct_reuse_pool else None
```

```python
# tiles.py:2554
def _pop_direct_reuse_pool(self, index: int = -1) -> TileLayerItemState:
```

`index=-1` pops the **most recently hidden** state — the tile that just left
the viewport, i.e. the single most likely tile to be needed next. Its
`_states` entry is dropped (`tiles.py:1598`), its source key unregistered, its
`ImageItem` moved to the new slot and overwritten by `_set_tile_data`
(`tiles.py:1622`). Its pixels are gone.

This fires on ordinary pan/scroll into fresh tiles **with no data change and
with no budget pressure** — it is slot occupancy, not memory, that triggers
it. Note the asymmetry: `_prune_resident_items` (`tiles.py:2586`) pops
`index=0`, i.e. it evicts oldest-first, which is correct. The two policies
disagree, and the hot one is the wrong one.

> **Fixed.** `_hijack_direct_reuse_state` now takes the pool head, preferring
> a state whose tile is not in the delta's near set, with an explicit fallback
> to the coldest protected state so a hijack can never fail into a fresh
> `ImageItem` allocation. Two things learned while measuring it, both of which
> narrow where this defect can be observed at all:
>
> 1. **The hijack is unreachable once every tile has a `_states` entry.**
>    `_hide_tile` keeps that entry, so a montage that has been presented whole
>    even once — a fit-to-view first frame — never hijacks again. An A/B built
>    on "present the montage, then zoom in and pan" measures exactly flat.
> 2. **On a balanced translation pan the pool is exactly one departure cohort
>    deep.** Entering count equals leaving count, so the pool is drained and
>    refilled every step and head and tail name the same column; every victim
>    policy destroys the same set. Pure pan measures flat too, in both
>    directions and into the montage edge.
>
> The policy only bites where the pool holds more than one cohort when a fresh
> tile arrives — a **zoom-in**, which retires many tiles and admits none. Panning
> away and back after a zoom-in (24×24 montage, 10×10 view, zoom to 4×4, six
> steps out and six back, twice) rebuilds **64 → 32 tiles, −50%**, with the whole
> saving on the return legs: 4 tiles rebuilt per step becomes 0. Interleaved
> in-process, order-balanced, `image_replacements` from `TileLayerUpdateStats`.

### 2.4 Secondary: a re-entering tile is hidden for one commit — **fixed**

`resolve_items` is restricted to `requested_upserts` whenever the delta
carries any upsert (`tiles.py:1254`). A tile that just re-entered the viewport
but is not itself an upsert therefore had no `drawable_payloads` entry and was
hidden — then re-shown on a later commit once `rearm_for_scope`
(`tile_lifecycle.py:1624`) re-dirtied it. Retained pixels existed the whole
time.

`_reshow_retained_active_tiles` now presents such a tile in the commit that
admits it back into `active`. It is a presentation step, not production: the
R2 subsection *"Skipping production is not skipping presentation"* is the rule
being served. A candidate qualifies only when the retained state is what the
commit would have drawn anyway — `acknowledged_identity_satisfies_target`
(the stale-pixel guard, unchanged: an outrun state still waits for its
upsert), `_direct_tile_binding_stale`/`_direct_tile_geometry_changed` both
negative, levels matching, and an item that actually holds an image. Measured
on a 4×4-tile window over an 8×8 montage, panning one column out and back
while one overlapping tile re-publishes (interleaved in-process, order
balanced, 12 trials × 6 pan-backs per arm): **2 → 1 commits to put pixels
under every visible tile, and 4 → 0 tile-commits hidden.**

Three regimes are deliberately left on the old path, all conservative (hide,
wait for the upsert) rather than wrong: `force_refresh` deltas, states whose
levels moved while they were out of view (`update_levels` skips non-visible
states, so their pixels are windowed to the old levels and this commit carries
no upsert to re-window them), and page-backed payloads, whose drawn source id
comes from the assembled payload rather than the one in `tile_payloads`.

### 2.5 Budget paths (correct in shape, blind to the halo)

- `_prune_rgb_source_cache` (`tiles.py:2506`): strips `rgb_base`/`hist_source`
  from pooled states past a **hardcoded 128 MiB** (`tiles.py:47`, no setting).
  Oldest-first. Only pooled states.
- `_prune_resident_items` (`tiles.py:2586`): full eviction past
  `tile_residency_budget_bytes`. Oldest-first, excludes `active_tiles` — but
  `active_tiles` is the margin-0 visible set, so **the halo is evicted before
  anything else**. Also, the `continue` branch at `tiles.py:2592` removes a
  still-visible state from the reuse pool without evicting it, silently
  dropping its warm-reuse claim.
  **Both fixed**: near tiles are now evicted after strangers, and an
  unevictable candidate is skipped in place rather than popped. The budget is
  unchanged as a hard ceiling — a near tile still goes the moment it is the
  last candidate, which is covered by its own test.
- Falling out of `_prune_rgb_source_cache` costs a returning tile the full
  `_set_tile_data_from_current_source` path (`tiles.py:2449`); `update_levels`
  skips non-visible states (`tiles.py:1936`), so a tile that returns after a
  levels change pays `rgb_display_for_levels` over the whole tile plus a full
  `setImage` (`tiles.py:2409`).

### 2.6 PyQtGraph never takes the converged early-out

`viewport_invariant_key` (`frame_session.py:1532`) lets an unchanged viewport
skip the entire retarget. It is gated on `persistent_gpu_tile_residency_backend`
(`frame_effects.py:5300`), which requires `shader_windowing` **and**
`tile_residency_kind in {"gpu_atlas", "none"}`. PyQtGraph is
`shader_windowing=False`, `cpu_item` (`backend_contract.py:44`) and fails on
both counts, so every camera nudge runs the full pass described in §4 — on the
backend that can least afford it.

Whether the latch is *safe* on `cpu_item` is genuinely open (the CPU path
re-windows on the CPU, so a level change is not viewport-invariant the way it
is under shader windowing). It has not been adjudicated; it has only been
excluded.

---

## 3. WGPU (`gpu_atlas` residency)

### 3.1 The GPU side is already lazy — commendably so

Eviction is reachable **only** from `_ensure_free_pool_layer`
(`wgpu_executor.py:2252`), and only when the target pool has no free layer and
no headroom below its (monotone, never-shrinking) layer ceiling. It evicts one
LRU unpinned victim from that one pool. There is no code anywhere that walks
the resident set dropping out-of-view pages, and no byte-budget trim on the
GPU side at all.

A pure pan/zoom submits a camera uniform and nothing else
(`wgpu_imageview2d.py:1271`, `wgpu_executor.py:3294` short-circuits
`UpdateTileInstances` on tuple identity). `invalidate_tiled_presentation`
documents explicitly that "executor page-table residency is retained"
(`wgpu_imageview2d.py:2746`).

Pins are the presentation-committed page set
(`replace_bound_content_pin_set`, `wgpu_executor.py:4070`), **not** viewport
visibility. So off-screen montage pages stay pinned as long as they are in the
current commit — which is the right instinct, and is precisely what §3.2
undoes one level up.

### 3.2 The defect is one level up: the session drops out-of-view payloads

```python
# frame_session.py:3689
for stale in tuple(self.display_tile_payloads):
    if int(stale) not in materialized:
        ...  # keep only if in planned_numbers and payload_is_current
        self.display_tile_payloads.pop(stale)
        self.pending_removals.add(stale)
        self.dirty_payloads.pop(stale)
        self.pending_payload_upserts.pop(stale)
        self.level_generation.forget_tile(stale)
```

`planned_numbers` is `visible_tile_numbers - skipped` (`frame_session.py:3660`)
— the margin-0 set. `materialized` is `rendered_tiles ∪ reconciled`, and on the
GPU-atlas backend `rendered_tiles` is kept near-empty **by design**, with an
explicit comment (`frame_session.py:1622`). So on WGPU every payload that
leaves the strict viewport is popped and queued as a physical removal, while
its GPU pages sit resident and unreferenced.

Returning therefore costs a full re-commit: re-plan, re-admit, re-instance,
re-bind. Not a re-upload — the pages are usually still there — but the
per-commit cost measured in the
[per-commit dossier](per-commit-transaction-count-2026-07-26.md) is
whole-montage by construction (63–69% of it is
`_apply_backend_tiled_presentation` iterating every presented payload, and an
empty-delta commit still costs ~90 ms). That is the lag spike, and the pages
being resident is exactly why it looks unexplained.

### 3.3 The guard that will bite any fix here

`frame_session.py:3808-3860` compares each drawn slot's identity against the
current payload and, on mismatch, calls `presentation_discarded` and adds to
`stale_identity_removals` — **but only when `tile not in planned_numbers or
may_remove_visible(tile)`**. Retaining an out-of-view tile keeps it outside
`planned_numbers`, which makes that branch unconditionally take the *remove*
path. A retention change that does not also widen this predicate will retain
tiles that the very next commit removes again, and will measure as flat.

---

## 4. Whole-montage work per camera move (both backends)

Everything below runs on every non-short-circuited viewport change, regardless
of how many tiles actually changed.

| Site | What it does per pan |
|---|---|
| `FramePlanner._plan_montage` (`frame_planner.py:285`) | constructs a fresh `FrameRegion` for **every tile in the montage**. Only the `active`/`planned`/`near` booleans differ between pans; `bounds`, `data_slices`, `view_state`, `materialization_key` are pan-invariant. Not memoized (`frame_controller.py:141`). |
| `FramePlan.__post_init__` (`frame_planner.py:71`) | four more full passes: `_active_region_ids`, `_planned_region_ids`, `_near_region_ids`, and `scene_region_signature` — the last builds a tuple-of-floats per region. |
| `TileLifecycle.retarget` (`tile_lifecycle.py:591`) | walks **every record ever created** (records are never removed — `tile_lifecycle.py:470`) and allocates a fresh `TileTarget` for each non-visible one, clearing `task_claim`/`stage_producer_key` and emitting a trace event. O(session history), not O(montage). |
| ~~`_priority_ordered_tile_delta`~~ | **Closed.** Was a full-montage `prioritize_tile_numbers` sort keyed on `(id(plan), id(context))` while `retarget_tile_priority` replaced `_priority_context` on every retarget. `retarget_tile_priority` now keeps the incumbent context object when the new one would rank every tile identically, and both consumers share one memo (`montage_priority_ranks`). See §4.1. |
| ~~`_stage_consumer_scheduling_rank`~~ | **Closed** with the row above; it no longer sorts for itself. |
| `sync_lifecycle_scope` (`frame_session.py:1323`) | O(all payloads) scan, called **twice** per retarget (`frame_controller.py:1499` and `frame_session.py:1638`). |
| `_update_montage_tile_overlays_for_plan` (`frame_runtime.py:736`) | cache key includes the viewport rect, so it misses every pan; the loop body is cheap (skipped tiles only). |

Already correctly scoped, for contrast: `mark_ladder_swaps_for_current_demand`
(`lod.py:1141`), `additions` filtering (`frame_controller.py:1511`),
`_rearm_required_first_pixel_payloads` (`frame_session.py:1336`),
`level_only_drain` (`frame_session.py:4440`), and `make_montage_plan`'s
`lru_cache(maxsize=4)` returning an identical plan object across pans
(`montage.py:209`) — which is what makes the `id(plan)`-keyed caches viable in
the first place.

This section is consistent with the measured
[index-scroll anatomy](../../docs/redesign/) finding that de-duplicating
per-tile work, not GPU or upload work, produced the −25%.

### 4.1 What the priority-rank rows actually cost, measured

The two closed rows were adjudicated on numbers rather than on the shape of
the key, because a value-keyed cache with a poor hit rate is worse than no
cache. Both measurements are on this machine, 2026-07-30.

**How often does a retarget change the ordering?** Instrumented run of
`tests/ui/{montage_interactions, montage_scroll_settling,
montage_scroll_level_retention, viewport_interactions, r8_first_pixel_phasing,
prefetch, lod_demand_freshness}`: **225 retargets, every one of them from
`retarget_viewport` (`frame_session.py:1641`)** — no hover or
`refresh_montage_priority_targets` retarget fired at all — of which **66
(29.3%) produced an ordering-identical context**. Those are replans with a
still camera, which is what a fill under a stationary viewport looks like on a
non-persistent-residency backend, where `frame_controller.py:1459`'s converged
early-out does not apply. Over the same run the commit boundary was reached
1436 times with a non-empty delta, i.e. **6.4 orderings per retarget**, and the
shipping `(id(plan), id(context))` key already hit 81% of them — the defect was
never "misses every commit", it was "one whole-montage sort per retarget,
including the retargets that changed nothing".

**What one avoided sort is worth.** In-process A/B on a 272-tile montage
(17×16), interleaved ABBA × 12, replaying that measured population at scale
(100 retargets, 29% still, 640 commit orderings):

| | rank-memo hit rate | whole-montage sorts | time in `prioritize_tile_numbers` |
|---|---|---|---|
| shipping | 84.4% (540/640) | 100 | 110.2 ms median (min 80.8) |
| retained context | **89.5%** (573/640) | **67** | **72.5 ms** median (min 60.9) |

Cost per sort is 1.10 vs 1.08 ms — the same in both arms, which is the check
that the win is *fewer* sorts and not cheaper ones. The enclosing loop
(retarget + ordering) moved 480.8 → 432.9 ms median, but at a standard
deviation of ~100 ms that number is not separable from machine noise and is
recorded here only so nobody re-derives it and claims it.

All 640 emitted orders were byte-identical between the two arms across 30
distinct permutations, which is the property that matters: the memo decides
*when* to resort, never *what* the order is.

**Not done, and why.** `_priority_ordered_tile_delta` still ranks the whole
montage to order a delta that is usually a handful of tiles. Ranking only the
upserts would give the identical relative order (band, focus rank and distance
are unchanged, and the `sequence` tie-break is monotone in tile number either
way) for O(k log k) instead of O(N log N) — but the absolute rank values are
published on the delta as `priority_ranks`, traced as `delta_priority_ranks`,
and read by the journey matrix (`journey_matrix.py:255`), and
`_stage_consumer_scheduling_rank` genuinely needs ranks comparable across the
whole montage. Changing what that number means is a separate adjudication.

Untouched for the same reason of scope: `render/effects.py:1011` runs a third
full-montage `prioritize_tile_numbers` inside `tile_states`, the ladder
snapshot previously measured at 93% of the drag-retarget chain. It ranks a
different candidate set and has no memo at all.

---

## 5. Prefetch: exists, but is not a halo-filler

`schedule_near_viewport_montage_prefetch` (`montage_prefetch.py:78`) warms the
*display payload* into the evaluator cache and optionally warms backend
residency (`_warm_prefetched_tiled_residency`, `montage_prefetch.py:664`) —
one tile at a time, each immediately hidden again (`tiles.py:2134`).

Candidates are **all plan tiles minus visible/rendered/loading/skipped**
(`montage_prefetch.py:465`) — the whole montage, priority-ordered, not the
halo. Gates, in order: `blocked_interaction` (hard no-op mid-drag),
session currency, evaluator presence, `_busy` (which demands a confident
scroll direction), a speculative share limit of 0/1/2 tiles, an 80% display
cache ceiling, governor memory pressure, and an idle batch limit of 1/2/4
tiles.

So the machinery for "when idle, get the nearby tiles ready" exists and is
correctly gated on interaction. What it lacks is (a) a halo-first candidate
ordering and (b) any coupling to backend retention — it warms and then
immediately drops the tile back into the same reuse pool that §2.3 will hijack.

`ImageView2D` does **not** override `_after_tiled_commit`
(`imageview2d.py:967` is a no-op base), so commits never warm the halo either.

---

## 6. What already implements "invalidate but keep"

Worth knowing before adding anything: the vocabulary exists.

- `TileLifecycle.target_invalidated` (`tile_lifecycle.py:1093`) — clears
  claims, demotes `presented_quality` to `"preview"`, leaves the tile
  presented.
- `remember_presentable` / `presentable_payloads` (`tile_lifecycle.py:906`,
  used at `frame_session.py:3697`) — "dirty means *prepare replacement*, not
  *remove the visible slot*".
- `may_remove_visible` (`tile_lifecycle.py:1051`) — refuses to turn a pending
  removal into a physical one for an active tile whose backend payload matches.
- `presentation_discarded` (`tile_lifecycle.py:1613`) — clears the
  presentation axis only, keeping `presentable_payloads` and `levels`.
- Park / `rearm_for_scope` / `_unpark` — a tile leaving scope stays PARKED and
  is re-dirtied on return.
- `payload_compatible_with_tile` (`montage_payload_cache.py:169`) — refuses a
  retained payload whose `semantic_generation` is stale. **This is the
  existing answer to "a quick pan must never reveal stale tiles"**, and any
  deferred-invalidation design should be built on it rather than beside it.

Blunt teardown, for contrast: `hide_all()` (`tiles.py:1152`) via
`invalidate_tiled_presentation(hide_pixels=True)`
(`frame_controller.py:886`) on a rejected session transition — it does not
distinguish visible from out-of-view.

---

## 7. Render-area code with no production caller

Reported here because §1–§6 required reading these paths anyway; the
adjudication belongs to [queue row 12](../queue.md).

- `_discard_direct_reuse_pool` (`tiles.py:2560`) — tests only.
- `_remove_tile` (`tiles.py:2324`) — no callers at all.
- `MontageTilePriorityQueue.prune` (`tile_priority.py:131`) — tests only;
  `discard` is reachable only through it.
- `ChunkStore` / `SlotPool` (`gpu/chunk_store.py`) — re-exported by
  `gpu/__init__.py:18`, used only by `tests/gpu/`. Its eager `ensure`-time
  eviction loop (`chunk_store.py:136`) is **not** the wgpu residency model and
  must not be read as documentation of one.
- `reset_surface` / `clearMontageTileLayer` (`imageview2d.py:1200`, `:2890`) —
  no production caller; the WGPU total-wipe `reset_tiled_residency`
  (`wgpu_imageview2d.py:2762`) is reachable only through them and `clear()`.

---

## 8. Summary of the gap

| Principle | Compute side | Render side today |
|---|---|---|
| Do work only when demanded | yes | GPU pages: yes. Everything above them: no. |
| Keep a valid result | yes | PyQtGraph keeps pixels, and no longer hands them to a LIFO victim policy (§2.3, fixed). WGPU keeps pages, then drops the payload that references them (§3.2). |
| Cost proportional to the change | yes | Five whole-montage passes per camera nudge (§4). |
| Protect what is about to be needed | n/a | Three inconsistent halo definitions (§1). One of them now reaches both PyQtGraph eviction predicates (§2.3, §2.5); the wgpu side and the unification are untouched. |
