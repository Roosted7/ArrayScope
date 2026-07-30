# Frame-plan region reuse — closing the first row of §4

Dated 2026-07-30. Closes the `FramePlanner._plan_montage` and
`FramePlan.__post_init__` rows of
[render-eagerness §4](render-eagerness-2026-07-30.md). The other rows of that
table (the priority-rank cache that can never hit, the lifecycle retarget
sweep, the twice-per-retarget `sync_lifecycle_scope`) are untouched.

## What a camera move now costs

A pan changes `active` and `near`. Nothing else in a montage `FrameRegion`
is a function of the camera:

| field | source | camera-dependent |
|---|---|---|
| `region_id` | `tile.montage_index` | no |
| `source_index` | `tile.source_index` | no |
| `bounds` | `tile.x0/y0/width/height` | no |
| `data_slices` | `tile.height/width` | no |
| `view_state` | `tile.view_state` | no |
| `materialization_key` | `(semantic_key, "montage", axis, source_index)` | no |
| `planned` | every tile, unconditionally | no |
| `active`, `near` | `tiles_intersecting(view_range, …)` | **yes** |

So the invariant part is built once per montage into a
`_MontageRegionSkeleton` and reused, along with `scene_region_signature` and
`_planned_region_ids`, which are pure functions of it.

## The key, and the evidence it is stable

`(id(montage_plan), target.semantic_key, axis)` — exactly the three inputs
the table above names.

The identity component is only viable because `make_montage_plan` returns the
*same* object across pans (`montage.py:209`, `lru_cache(maxsize=4)`). That
premise was measured, not assumed: this repo has already been burnt by an
identity key on an object the app quietly rebuilt every step, which measured
a 9% hit rate (see the commit-cost instruments dossier).

A real `ArrayScopeWindow` with a 272-tile montage, driven through the actual
viewport machinery — `setRange` on the live view, the app's own retarget
path — over twelve pans:

| planner input | distinct values over 12 pans |
|---|---|
| `id(montage_plan)` | **1** |
| `target.semantic_key` | **1** |
| `view_state.montage_axis` | **1** |
| `id(view_state)` | **1** |
| `len(montage_plan.tiles)` | **1** |
| `id(target)` | 12 |
| `view_range` | 12 |

`target` is a fresh object every retarget, which is why it cannot be the key
even though `target.semantic_key` can. Live planner counters over setup plus
the same twelve pans: 14 skeleton hits / 2 misses (0.875); 11/12 restricted
to the pans themselves.

## Two levels, and why the second one exists

1. **Skeleton** — the pan-invariant regions, `scene_region_signature` and
   `_planned_region_ids`. Bounded at **4**, matching `make_montage_plan`'s
   own `lru_cache(maxsize=4)`: a fifth entry could never hit, because the
   plan it was keyed on would already be gone. The skeleton holds a strong
   reference to that plan, so an `id` in the key can never name a different
   object.
2. **Whole regions tuple** — up to **8 per skeleton**, keyed on the
   `(active ids, near ids)` pair. `tiles_intersecting` snaps outward to whole
   tiles, so a drag step smaller than a tile resolves to the identical sets.

Between them, each tile memoizes its `(active, planned, near)` variants — at
most 8 by construction, 2–3 in practice — so a genuinely new tile set costs
272 dict lookups rather than 272 dataclass constructions.

## Measurements

In-process, interleaved, order-balanced (a sequential single-shot A/B has
inverted the sign four times in this repo). 272 tiles, 17 columns, 16×16
tiles, a 4×4-tile camera. 40 rounds. Median ms per retarget, reusing planner
vs the same planner with `montage_region_cache_size=0`.

| pan regime | skeleton hit | tuple hit | before | after | delta |
|---|---|---|---|---|---|
| 1 px drag step, 60 pans | 99.96% | 99.84% | 1.3617 | 0.0921 | **−93.2%** |
| half-tile steps, 24 pans | 99.90% | 25.00% | 1.3545 | 0.1572 | **−88.4%** |
| every tile set distinct | 99.90% | 0.00% | 1.0929 | 0.1643 | **−85.0%** |

The bottom row is the honest floor: the whole-tuple layer cannot hit there at
all, and the skeleton alone still removes 85%. Traced allocation over one pan
sequence: 227.7 KiB → 13.1 KiB (1 px drag) / 40.2 KiB (the other two).

The whole-tuple layer was A/B'd on its own rather than assumed — same
planner, same 60-pan pixel drag, only `MONTAGE_FLAG_SET_CACHE_SIZE` differs:

| tuple cache | hit rate | median ms/pan |
|---|---|---|
| 8 | 96.7% | 0.0862 |
| 0 | 0% | 0.1500 |

Machine load was elevated and shared during these runs; the ratios are stable
across regimes and the absolute numbers are not the claim.

## What would make this wrong

- `make_montage_plan` ceasing to return an identical object across pans. The
  hit rate collapses to 0 and every plan is rebuilt — correct, just no
  faster than before. The reuse tests assert the hit rate precisely so this
  degrades loudly instead of silently.
- A montage plan whose `montage_index` stops being the tile's position.
  `region_ids_in_order` sorts the membership set instead of filtering all
  regions, which is only equivalent for a dense plan; the skeleton checks
  density once at build and keeps the full filter for anything else.
- `planned` becoming conditional on this path. It is structurally baked into
  the skeleton.

## Left alone deliberately

The halo radii. The planner's `near` uses `margin_tiles=1` while the
session's uses 2 and the resize path uses 0 — the §1 inconsistency. This
change preserves all three exactly; unifying them belongs to the sibling
chip.
