# Out-of-view presentation retention on `gpu_atlas` (2026-07-30)

Closes §3.2 of [render-eagerness](render-eagerness-2026-07-30.md) for the WGPU
backend. **Landed.** The measured effect is a counter, not a millisecond: full
whole-montage republications per pan cycle fall from a median of 4 to 0 at 64
tiles and 6 to 2 at 272. The apply-time saving is inside the noise at the scale
this could be measured at, and is reported as such rather than as a win.

## 1. What the defect actually was — one correction to the map

§3.2 named the drop at `frame_session.py:3689` (every payload leaving the strict
viewport is popped and queued for physical removal). That is real, but it is not
the whole mechanism and on a mid-sized montage it does not fire at all.

Measured on a settled 36-tile scalar montage, zoomed to 9–12 visible tiles:
`display_tile_payloads` held **36** on both revisions. Nothing was dropped —
`materialized` contains every tile with a `RenderedTile`, and this montage has
one per tile. What shrank was `TilePresentationDelta.active_tiles`, from 36 to
the visible 9.

That is sufficient on its own, because `active_tiles` is the presentation scope:

- `TilePresentationState.active_payloads(delta)` filters the acknowledged state
  by `active_tiles`, and that mapping *is* `montage_tile_payloads`;
- `_apply_backend_tiled_presentation` builds `committed_tiles` from
  `work_payloads` only, so a tile absent from `payloads` simply vanishes from
  `_wgpu_committed["tiles"]` — **no removal has to be emitted for the backend to
  forget it**;
- and its absence breaks `set(previous_commit["tiles"]) <= set(payloads)`, the
  precondition at `wgpu_imageview2d.py:1674` that decides `incremental_delta`.
  Fail it and the commit republishes the whole presented set instead of applying
  the delta.

So the correct statement of the defect is **the session narrows the presentation
scope to the viewport**, and the payload drop is the same defect's other half —
which does fire at 272 tiles, where payloads collapsed from 272 to 12/16/20.

## 2. The change

`FrameSession.retain_out_of_view_presentation_scope` runs last in
`build_tile_presentation`, after every other pass has already run on the visible
scope, and widens `active_tiles` with out-of-view tiles that pass a narrow
predicate. Nothing else widens — not `planned`, not the lifecycle work scope,
not the rearm/upsert/priority passes, not the level-staleness scan. A retained
tile is a tile the commit does no work for; a test asserts no retained tile
appears in `delta.upserts`.

`retain_out_of_view_payloads` is set at session construction from
`tile_residency_kind == "gpu_atlas" and persistent_tile_residency`. PyQtGraph
(`cpu_item`) is out of scope and pinned off by test.

### The predicate

A tile is retained only when **the backend is drawing exactly the payload the
session still holds**, and the payload is still semantically right for the slot:

| Clause | Why |
|---|---|
| `backend_identities[tile] == tile_ack_identity(payload)` | physical truth, and see §3 |
| plan tile and lifecycle target both name `payload.source_index` | the slot has not been repointed |
| `payload.source_id in record.presentable_payloads` | `TileLifecycle.retarget` prunes that map when a slot is reused for another source — the same evidence `may_remove_visible` reads |
| `payload_compatible_with_tile(payload, tile.view_state, ...)` | the existing `semantic_generation` staleness guard (ADR 0050, `montage_payload_cache.py:169`) |

Deliberately **not** required: target quality. A retained coarse tile is legal to
keep showing (R2's free-reuse rung); requiring `payload_is_current` would refuse
every preview tile, which on a large montage is most of the out-of-view set.

The predicate is re-evaluated for every held tile on every build, not only for
new candidates. A decision taken once at departure goes stale — a montage index
scroll repoints a slot at another source without the retention entry knowing.
That costs O(retained) dict lookups plus one `semantic_generation` tuple per
commit, sub-millisecond at 272 tiles.

### The ceiling and its eviction order

Retention holds CPU-side payload wrappers past the viewport, so it gets its own
stated bound: **50 % of `tile_residency_budget_bytes`**
(`RETAINED_OUT_OF_VIEW_BUDGET_SHARE`), summed over `DisplayTilePayload.nbytes`
stamped once at departure and maintained incrementally. A session with no
residency budget retains nothing.

Eviction order is **least recently visible first** — the tile that left the
viewport longest ago. That is the opposite of the PyQtGraph victim policy
§2.3 already found wrong, and eviction takes the ordinary drop path, so an
evicted tile ends up exactly where it would have been without retention: out of
the scope and queued for physical removal.

The ceiling did not bind in either measured regime: 22 retained tiles held
1.4 MB and 260 held 2.6 MB, against a 134 MB ceiling. It is a safety bound, not
an operating constraint on this hardware. Its behaviour is pinned by test at a
forced ceiling instead.

## 3. The §3.3 guard, checked before measuring

`frame_session.py:3853` discards a drawn slot whose identity differs from its
current payload when `tile not in planned_numbers or may_remove_visible(tile)`.
A retained out-of-view tile satisfies **both** halves — `may_remove_visible`
returns `True` for any record with `target.visible == False`. Retaining without
addressing this would have the next commit remove the tile again and the whole
change would measure flat.

**The predicate was not widened.** Requiring the backend to be drawing exactly
the held payload puts every retained tile on that pass's `current_identity ==
shown_identity` early continue, so the branch cannot fire for a retained tile by
construction. What that costs is stated in §5.

## 4. Measured

Interleaved in-process A/B: one process, one window, one settled montage, arms
alternated `base, retain, retain, base, base, retain, …` by toggling the runtime
flag. Sequential single-shot processes have inverted the sign four times in this
repo; this comparison never crosses a process boundary.

64-source montage, 8 columns, 160 px tiles, offscreen WGPU. One cycle =
home → away-x → home → away-y → home, 20 tiles visible throughout. Six rounds
per arm, medians with the full value list:

| per pan cycle | base | retain |
|---|---|---|
| **full republications** | **4** `[4,4,4,4,5,5]` | **0** `[0,0,0,0,1,1]` |
| incremental commits | 1 `[0,0,0,1,1,4]` | 4 `[0,2,4,4,7,8]` |
| mapping-only fast path | 0 `[0,0,0,0,0,6]` | 3 `[2,2,3,3,4,4]` |
| GPU uploads | 0 | 0 |
| `_apply_backend_tiled_presentation` total | 31.1 ms `[13.4–37.3]` | 28.7 ms `[6.9–39.7]` |
| presented tiles at rest | 20 | 42 |

The republication distributions do not overlap. Everything else does.

**Read the counters, not the milliseconds.** `apply_ms` is flat within noise, and
the reason is the regime, not the change: a "full republication" here republishes
the *presented* set, which without retention is 20 tiles. The 63–69 %-of-a-commit
figure in the [per-commit dossier](per-commit-transaction-count-2026-07-26.md)
was measured at 272 presented tiles. Retention trades four cheap 20-tile
republications for one plus several delta and mapping-only commits over a
42-tile presented set: the transaction *classes* move decisively, the wall clock
at this scale does not. Wall time per cycle is unusable — most rounds are
dominated by the offscreen fill's settle timeout.

**GPU uploads are zero in both arms**, which is the point: the pages were
resident all along (`_ensure_free_pool_layer` is the executor's only eviction
route). Retention buys back the *binding*, not the bytes.

The same probe at 272 sources / 96 px, four rounds per arm:

| per pan cycle | base | retain |
|---|---|---|
| full republications | 6 `[4,5,6,6]` | 2 `[0,1,2,3]` |
| incremental commits | 2 `[1,2,2,4]` | 4 `[4,4,4,8]` |
| mapping-only fast path | 0 `[0,0,0,6]` | 3 `[1,2,3,4]` |
| GPU uploads | 0 | 0 |
| `_apply_backend_tiled_presentation` total | 40.0 ms `[36.3–48.9]` | 36.8 ms `[26.8–40.7]` |

Here the ranges do overlap at the edges (base min 4, retain max 3), so the
medians separate and the distributions do not. The direction is the same; four
rounds is what that fixture affords, because it does not settle inside the
offscreen budget and every leg spends its timeout.

Two observations from a separate cold before/after pass on that fixture,
recorded because they are the regime the map describes and are **not** claims:

- without retention the payload set genuinely collapses — 272 held payloads
  became 12/16/20 — which is §3.2's mechanism firing as written, at a scale the
  64-tile fixture never reaches;
- after the four-leg pan the base revision ended with
  `required_target_unsettled_tiles() == (19,)` and never completed, while the
  retained revision ended at `()`. n=1 on a loaded machine; suggestive of the R2
  "settled at 135 of 272 forever" family, not evidence of a fix for it.

## 5. What this costs, and what it does not do

- **Retention erodes.** Background work supersedes an out-of-view payload
  wrapper; the identity clause then refuses the tile and it leaves the scope,
  and a shrink republishes once. This is why the retain arm shows 1 rather than
  0 in two of six rounds. Retaining the *acknowledged* payload instead of the
  current one would remove the erosion — and would require widening the §3.3
  predicate. That is the next increment, deliberately not taken here.
- **A full republication is now larger.** Retention raises the presented set, so
  the republications that do happen cost more each. The counter table above is
  the net.
- **Pinned GPU pages.** Retained tiles stay in `committed_page_keys`, so
  `replace_bound_content_pin_set` pins them. Bounded by construction: retention
  only holds tiles of the current plan, and the fully-zoomed-out view already
  pins every one of them — `ensure_raw_pool_capacity` and
  `planned_tile_count(..., minimum=len(payloads))` already size the pool for that
  state. But the byte ceiling bounds payload memory, not the pin set, and the
  [pool-layer leak](wgpu-pool-layer-leak-2026-07-26.md) is the standing reminder
  that a residency budget equal to the working set has a failure mode.
- **R3 gets more conservative, not less.** `_shader_presented_payload_bounds`
  already unions bounds over every `display_tile_payloads` entry, so more
  presented tiles widen the containment the clamp checks. The round's level
  *source* is whole-montage already and does not move.
- **Untouched:** the PyQtGraph victim policy (§2.3), the three inconsistent halo
  radii (§1), deferred invalidation on data change, and the whole-montage
  per-camera-move passes (§4). The atomic-successor builder keeps its own
  `active_tiles == upserts` contract and is not widened; after an atomic handoff
  the retention set is rebuilt from scratch by the next general build.

## Reproduce

The arms are the runtime `FrameSession.retain_out_of_view_payloads` flag, so the
A/B is a scratch pytest module that toggles it between legs and reads
`_wgpu_binding_full_republications` / `_wgpu_binding_incremental_commits` /
`_wgpu_binding_fast_path_commits` and `executor.uploads_total` — all pre-existing
counters. No per-commit instrumentation was added; `apply_ms` came from wrapping
`_apply_backend_tiled_presentation` in the probe, not in production. Regression
coverage is `tests/ui/test_montage_out_of_view_retention.py` (live WGPU pan) and
`tests/window/test_out_of_view_retention.py` (predicate, ceiling, eviction
order).
