# Render-side eagerness — what a viewport change costs today

Dated 2026-07-30. **Map, plus one adjudication.** Every numbered finding is a
citation of current `main` behaviour, written so the work that follows can be
scoped without re-deriving it. Where a claim is an inference rather than a read
line, it says so. §2.6 is the exception: it was a question when this document
was written and is now a decision with its evidence, so nobody has to
re-derive that either.

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

> **Superseded in part, same day.** §1's halo inconsistency and §5's
> "prefetch is not a halo-filler" have both been acted on; see
> [retention-halo-2026-07-30.md](retention-halo-2026-07-30.md) for the unified
> halo, two corrections to what is written below, and the measured **rejection**
> of warming the halo from prefetch. §2, §3, §4, §6 and §7 stand as written.
>
> **The margin column in §1's table is one too small in every row**, and the
> quoted-block correction below misdiagnoses why: the rect adds a tile stride
> that no constant named, and that stride — not the snap — is the whole
> 6-against-15 gap. The convention is settled in
> [viewport-rect-convention-2026-07-30.md](viewport-rect-convention-2026-07-30.md),
> where the constants are rebased to denote the same sets honestly.

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

> Two corrections found while unifying them. **The margin is not the whole
> difference**: the session derives its set from a rect that
> `montage_rect_for_viewport` has snapped outward to whole-tile bounds and
> clipped to the montage, while the planner passes the raw camera rect
> straight through. On the 272-tile montage zoomed to a few tiles that made
> the planner's *margin-0* set 6 tiles where the session's was 15 — a
> discrepancy this table cannot show. (`near` is now unified; `active` is
> not.) **And the halo was never conditional on `persistent_tile_residency`**
> as the policy in `montage_viewport.py` implies: `PYQTGRAPH_CAPABILITIES`
> declares that flag `True`, so both shipped backends were already on the
> margin-1/2 policy. The halo-less branch is what a *failed capability probe*
> returns.

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

### 2.6 PyQtGraph never took the converged early-out — adjudicated, and now it does

`viewport_invariant_key` (`frame_session.py:1532`) lets an unchanged viewport
skip the entire retarget. It used to be gated on
`persistent_gpu_tile_residency_backend`, which requires `shader_windowing`
**and** `tile_residency_kind in {"gpu_atlas", "none"}`. PyQtGraph is
`shader_windowing=False`, `cpu_item` (`backend_contract.py:44`) and failed on
both counts, so every camera nudge ran the full pass described in §4 — on the
backend that can least afford it.

**Adjudicated 2026-07-30: the latch is safe on `cpu_item`, and the key needs no
additional captures.** The gate is now two predicates that answer two
questions (`frame_effects.py`):

- `persistent_gpu_tile_residency_backend` — *does a commit here upsert into a
  shader-sampled page store, so the bounded GPU commit policy applies?*
  Unchanged, and still what `settled_payloads_are_known` reads (a third
  question again: whose map holds a settled tile).
- `viewport_convergence_latch_backend` — *can this backend move the camera over
  retained tiles without the engine restating anything per tile?* Persistent
  residency, `kind in {"gpu_atlas", "cpu_item", "none"}`, no shader-windowing
  requirement. WGPU pans by camera uniform over world-rect instances;
  PyQtGraph pans by the ViewBox transform over retained `ImageItem` scene
  positions. Shader windowing is about who applies *levels*, which is a
  different axis.

#### Why the CPU path is not the hazard it looked like

The obvious candidate mechanism — under shader windowing a levels change is
viewport-invariant because the GPU re-maps, whereas the CPU path re-windows
pixels — **does not exist on the production path**:

- `ImageView2D._update_montage_tile_levels` (`imageview2d.py:3139`) *raises*:
  "PyQtGraph tiled level changes must use governed tiled presentation
  commits". `MontageTileLayer.update_levels` (`tiles.py:1918`), whose loop
  skips non-visible states, has **only test callers** — so "a levels change
  skips the tiles that are not visible" cannot strand a PyQtGraph tile.
- `set_lookup_table` (`tiles.py:1124`) iterates **all** states and is reached
  from `_backend_display_lut_changed` (`imageview2d.py:3151`) without touching
  the session at all, so no session-level early-out can skip it.

More generally, every state change that can alter PyQtGraph's baked pixels
either reborns the session — `window_mode`, `user_levels_override`, colormap
LUT identity, dtype, rgb and geometry are all `_reject(...)` cases at
`frame_controller.py:1050-1078`, and a reborn session starts with
`_viewport_converged_key = None` — or owns its own commit:

- user and automatic levels: `display_presenter.py:717-745` commits directly
  through `apply_montage_presentation` / `commit_frame_session_presentation`;
- refined level evidence, whose PyQtGraph branch is inside the code the latch
  skips (`frame_controller.py:1857`): `retarget_frame_pipeline` calls
  `_finish_frame_session_if_complete` itself (`frame_runtime.py:438`), so that
  branch has an owner off the viewport path.

And the latch cannot be *earned* while anything is outstanding:
`mark_ladder_swaps_for_current_demand` dirties any visible tile whose payload
is not `exact` (`lod.py:1175`), which sets `lod_swap_ready`, which is one of
the three conditions at `frame_controller.py:1539`.

#### Evidence

A differential oracle — one scripted interaction run twice, once latched and
once not, comparing baked pixels, LUT, levels, LOD scale and geometry per
visible tile — is in
[`tests/ui/test_viewport_convergence_latch.py`](../../tests/ui/test_viewport_convergence_latch.py).
The script interleaves sub-tile pans with three levels changes, two colormap
changes, an automatic window, a zoom-in and two boundary-crossing pans. It
carries its own control (latch vs latch) and its own mutant (a key that forgets
the visible tile set, which must be caught).

Two measurement traps had to be cleared first, and both produced a confident
wrong answer before the control did:

- **Sampling mid-chunk is not a defect.** At 272 tiles the level re-bake is
  chunked under the governor, so an immediate post-step sample catches the two
  arms at different points in the same sequence. The latch-vs-latch control
  diverges there too. The oracle therefore drains each step and excludes the
  steps that never settle — which are the same three in both arms (both
  explicit levels changes and the automatic window; a standing PyQtGraph
  condition, not a latch effect).
- **A physical capture of a second window in the same process measures the
  first window's leftovers.** Arm two inherits arm one's QSettings, so its
  colormap and levels differ; a latch-vs-latch control diverged on **58% of
  pixels** that way. One arm per process fixes it.

Physical truth, real Weston output (`arrayscope.tools.headless_display
--exact-window`, 880×660, 272 tiles, one arm per process, three arms:
latch, latch again as control, no latch):

| Comparison | Settled steps compared | Worst differing pixels |
|---|---:|---|
| CONTROL: latch vs latch | 22 of 26 | **0** of 580 800 |
| A/B: latch vs no latch | 22 of 26 | **0** of 580 800 |

Byte-identical compositor output on every settled step. The four excluded
steps are the ones at least one arm reported as never settling (both explicit
levels changes, the automatic window, and — in the *unlatched* arm only — the
nudge after the automatic window).

An earlier sweep of the same harness showed 1093 differing pixels in both the
control and the A/B, with a byte-identical diff mask confined to
`y ∈ [643, 655]` — the 13-row status strip. Worth recording because it is the
shape the answer takes when the harness itself is the source: the A/B number
equalled the control's exactly.

One asymmetry worth recording, because it points the opposite way from the
suspicion: the *unlatched* build has **more** steps that never settle. Its
retarget re-queues PyQtGraph's refined-level pass on every camera nudge
(`frame_controller.py:1857`), which re-opens level work the latched build had
already closed. The test's settle assertion is therefore one-sided — the latch
may never *prevent* a settle; settling more often is not a defect.

#### Cost removed

272-tile PyQtGraph montage, fully settled and fully on screen, sub-tile pans.
Both arms in one process, interleaved, block order ABBA-balanced, n=72 per arm
per run, three runs:

| | latch off | latch on |
|---|---:|---:|
| median retarget | 31.2 / 31.2 / 30.8 ms | 1.09 / 1.07 / 1.02 ms |

**≈31 ms → ≈1.1 ms per camera nudge, −96.6%, ~29×.** The residual is the O(1)
key check plus the prefetch schedule. The first nudge of each block pays a full
pass to re-earn the latch (median ~30 ms unlatched, ~1.1 ms latched) and is
excluded from both steady states.

#### Two things the key still does not capture, and why that is fine

- `viewport_invariant_key` accepts `viewport_shape` and ignores it. That is
  redundancy rather than a hole: a resize rescales the span
  (`_resize_preserving_screen_zoom_spans`, `montage_viewport.py:741`) *and*
  re-keys `make_montage_plan`, so `id(plan)` and the span both move.
- A sub-tile pan changes the R2b structural round key (it includes the exact
  view range), so the latched pass never re-selects the round floors. Nothing
  is produced under the latch, so no floor is used; the next unlatched pass
  latches fresh floors.

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

### 3.2 The defect is one level up: the session narrows the presentation scope

**Closed 2026-07-30 —
[out-of-view-retention](out-of-view-retention-2026-07-30.md).** That dossier
also corrects the mechanism below: the payload drop is real and fires at 272
tiles, but on a mid-sized montage nothing is dropped at all and the whole effect
comes from `TilePresentationDelta.active_tiles` shrinking to the viewport. A
tile absent from `active_tiles` disappears from `_wgpu_committed["tiles"]`
without any removal being emitted, and breaks the executor's
`previous_commit["tiles"] <= payloads` precondition for applying a delta.
Measured: full republications per pan cycle 4 → 0 (64 tiles) and 6 → 2 (272
tiles), GPU uploads unchanged at zero in both arms.


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

### 3.3 The guard that will bite any fix here — checked, not widened

`frame_session.py:3808-3860` compares each drawn slot's identity against the
current payload and, on mismatch, calls `presentation_discarded` and adds to
`stale_identity_removals` — **but only when `tile not in planned_numbers or
may_remove_visible(tile)`**. Retaining an out-of-view tile keeps it outside
`planned_numbers`, which makes that branch unconditionally take the *remove*
path. A retention change that does not also widen this predicate will retain
tiles that the very next commit removes again, and will measure as flat.

The landed change does **not** widen it. Retention is refused unless the backend
is drawing exactly the payload the session still holds, which puts every
retained tile on that branch's `current_identity == shown_identity` early
continue, so it cannot fire for one. The price is that retention erodes when
background work supersedes an out-of-view wrapper, and each shrink costs one
republication; see §5 of the retention dossier.

---

## 4. Whole-montage work per camera move (both backends)

Everything below runs on every non-short-circuited viewport change, regardless
of how many tiles actually changed. Since the §2.6 adjudication, a *settled*
montage short-circuits all of it on both backends; what remains here is the
cost of every camera move that crosses a tile boundary or a zoom threshold, and
of every camera move at all while the montage is still filling.

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

**Closure log.** The first two rows (`_plan_montage` and
`FramePlan.__post_init__`) are addressed in
[frame-plan-region-reuse-2026-07-30.md](frame-plan-region-reuse-2026-07-30.md):
the pan-invariant region skeleton and its derived signature are built once
per montage plan, measured 85–93% off the planner's per-retarget cost at 272
tiles. The remaining rows are open.

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

> Correction: the hide is real, the drop into the reuse pool is not.
> `tiles.py` calls `_remove_from_direct_reuse_pool(item_state)` and
> `_touch_resident_state(item_state)` immediately after, with a comment saying
> warm residency is deliberately kept out of the generic pool so the next warm
> batch cannot pop and retarget it. The sharper defect is that the walk's
> display-cache `hit` branch returns before ever reaching the warm, so on a
> montage that opened fit-to-window — every tile already evaluated — prefetch
> warms backend residency for nothing at all. Closing that gap was
> **measured and rejected**: see
> [retention-halo-2026-07-30.md](retention-halo-2026-07-30.md) §3.

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

> **Acted on, and one correction to the target.** See
> [data-change-invalidation-2026-07-30.md](data-change-invalidation-2026-07-30.md).
> The blunt teardown named above is *not* where a montage data change goes: a
> slice step, an index-window move, an axis-range crop and a canonical-orientation
> transpose all take `retarget_index_window`, an in-place rebirth that never
> reaches `invalidate_tiled_presentation` at all — so nothing told the surface to
> stop drawing the predecessor, and a pure camera move reproduced the previous
> slice **100 % pixel-identical** on `gpu_atlas`. Superseded tiles are now hidden
> at that seam (`hide_superseded_tiled_presentation`, hide-only, proportional to
> what changed) and their removals deferred behind `visible_plan_complete()`:
> out-of-view removals riding the commits before first visible pixels go 992 → 0
> (WGPU) and 744 → 0 (PyQtGraph) at 272 tiles, with first-pixel latency unchanged
> within noise.

---

## 7. Render-area code with no production caller

Reported here because §1–§6 required reading these paths anyway; the
adjudication belongs to [queue row 12](../queue.md).

- `_discard_direct_reuse_pool` (`tiles.py:2560`) — tests only.
- `_remove_tile` (`tiles.py:2324`) — no callers at all.
- `MontageTileLayer.update_levels` (`tiles.py:1918`) — tests only. Its
  production entry point `ImageView2D._update_montage_tile_levels`
  (`imageview2d.py:3139`) raises on purpose, because PyQtGraph level changes go
  through the governed commit instead. Worth naming because its visible-only
  loop reads like the mechanism that would make the §2.6 latch unsafe, and it
  is not reachable to be one.
- `MontageTilePriorityQueue.prune` (`tile_priority.py:131`) — tests only;
  `discard` is reachable only through it.
- `ChunkStore` / `SlotPool` (`gpu/chunk_store.py`) — re-exported by
  `gpu/__init__.py:18`, used only by `tests/gpu/`. Its eager `ensure`-time
  eviction loop (`chunk_store.py:136`) is **not** the wgpu residency model and
  must not be read as documentation of one.
- `reset_surface` / `clearMontageTileLayer` (`imageview2d.py:1200`, `:2890`) —
  no production caller; the WGPU total-wipe `reset_tiled_residency`
  (`wgpu_imageview2d.py:2762`) is reachable only through them and `clear()`.

### 7.1 Adjudicated 2026-07-30 — all six deleted

Every entry above was rot, and every one of them was orphaned by a named
commit rather than by drift:

| Entry | Orphaned by | Survivor |
|---|---|---|
| `_remove_tile` | `02bc08ae` "Add proper residency to PyQtGraph" | `_evict_resident_state` (budget-governed, refuses a visible state) |
| `_discard_direct_reuse_pool` | never had one | `clear()` and `_prune_resident_items` |
| `prune` / `discard` | `934081fb` "Delete duplicate FrameSession tile-work queue" | none needed — production builds the queue from a candidate set and drains it |
| `ChunkStore` / `SlotPool` | never adopted | `PageTable` + the executor's own layer pools |
| `clearMontageTileLayer` | `da54e9db` "Improve VisPy atlas re-usage" | `hide_tiled_presentation` / `invalidate_tiled_presentation` |
| `reset_surface` | born unreachable in `ff0fa59d` | none — see below |

Two findings worth keeping:

- **`reset_surface` was speculative contract, not decayed contract.** It was
  introduced with the `ImageSurface` protocol and never had a caller in
  `arrayscope/`. The event class that would justify it does not exist either:
  the tree has no device-lost, context-loss, or surface-lost handling anywhere,
  and no runtime backend switch. Its one product-visible artifact was
  `presentation_diagnostics()["last_reset_reason"]`, which only a test read.
- **The residue is `clear()`.** With `reset_surface` and
  `clearMontageTileLayer` gone, `reset_tiled_residency` is reachable only from
  `ImageViewShell.clear()` / `WgpuImageView2D.clear()` — and `clear()` itself
  has no caller in `arrayscope/`, `tools/`, or `tests/`. That is the same
  cluster one level down. It is left standing deliberately: `reset_tiled_residency`
  is a declared backend-contract member with its own contract test
  (`test_reset_tiled_residency_survives_and_recommits`), so retiring the reset
  boundary is a contract decision, not a dead-code sweep. Next row-12 candidate.

A third finding is about the guard rather than the code: `ChunkStore` and
`SlotPool` were invisible to `tools/dead_code.py` **because** `gpu/__init__.py`
imported them. The guard rightly refuses to count an `__all__` entry as a
caller, but it does count the `from … import …` line that feeds it. A
re-export-only package `__init__` therefore launders test-only code into
"reachable" — the same rot one level up, and the reason this module had to be
found by reading rather than by the gate.

---

## 8. Summary of the gap

| Principle | Compute side | Render side today |
|---|---|---|
| Do work only when demanded | yes | GPU pages: yes. Everything above them: no. |
| Keep a valid result | yes | PyQtGraph keeps pixels, and no longer hands them to a LIFO victim policy (§2.3, fixed). WGPU keeps pages, then drops the payload that references them (§3.2). |
| Cost proportional to the change | yes | Five whole-montage passes per camera nudge (§4). |
| Protect what is about to be needed | n/a | Three inconsistent halo definitions (§1). One of them now reaches both PyQtGraph eviction predicates (§2.3, §2.5); the wgpu side and the unification are untouched. |
