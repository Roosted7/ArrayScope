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

### 2.4 Secondary: a re-entering tile is hidden for one commit

`resolve_items` is restricted to `requested_upserts` whenever the delta
carries any upsert (`tiles.py:1254`). A tile that just re-entered the viewport
but is not itself an upsert therefore has no `drawable_payloads` entry and is
hidden at `tiles.py:1422` — then re-shown on a later commit once
`rearm_for_scope` (`tile_lifecycle.py:1624`) re-dirties it. Retained pixels
exist the whole time.

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
| `_priority_ordered_tile_delta` (`frame_effects.py:3936`) | full-montage `prioritize_tile_numbers` sort, cached on `(id(plan), id(context))`. `retarget_tile_priority` replaces `_priority_context` with a new object on every retarget (`frame_session.py:5239`), so **the cache misses every pan by construction**. |
| `_stage_consumer_scheduling_rank` (`frame_effects.py:4625`) | same full-montage sort, no cache. |
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
| Keep a valid result | yes | PyQtGraph keeps pixels, then hands them to a LIFO victim policy (§2.3). WGPU keeps pages, then drops the payload that references them (§3.2). |
| Cost proportional to the change | yes | Five whole-montage passes per camera nudge (§4). |
| Protect what is about to be needed | n/a | Three inconsistent halo definitions, none of which reaches an eviction predicate (§1). |
