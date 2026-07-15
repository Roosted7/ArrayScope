# G5 source-grid sparse pyramid contract — 2026-07-16

**Status:** authoritative implementation contract for the remaining ADR 0056
work; pure page resolution/pins, source-grid mean geometry, and the VisPy
CPU-resolution/physical-truth seam implemented.
This closes the dangling “route-canonicalization” reference in the GPU handoff
before production code chooses an accidental second reduction route.

## One canonical reduction route

Ingest reduction, ladder materialization, cached derivation, and later
GPU-from-resident generation must all request the same source-grid page plan.
The plan is pure integer geometry and contains:

- document/operation generation, representation, dtype, reducer family, and
  anisotropic reduction vector;
- native-source footprint and valid input footprint for every logical page;
- stored-sample rectangle, global bin origin, and source samples per stored
  sample on every axis;
- clipped boundary coverage, gutter/sample expansion, and the exact transform
  from a target page's sample coordinates to the resolved resident page.

No caller may independently start reduction bins at its window-local zero.
`reduce_box_mean(array)` remains a low-level numeric primitive only; the
source-grid planner decides which samples each bin owns before any numeric
backend runs.

## Identity and boundary rules

Logical `DataChunkKey` page geometry is expressed in native source coordinates.
At any LOD, the physical page stores a uniform sample extent while its source
footprint grows by the reduction vector. Full aligned interior bins therefore
keep the same identity across overlapping windows.

Partial boundary bins are different values when their valid sample coverage
differs, even if they occupy the same nominal global bin. Their identity must
include the clipped valid footprint (directly or through a value-source key).
They are not reusable merely because a screen window moved by one pixel.

Gutters expand input coverage but do not move the global bin origin. Recursive
parent-to-child generation is legal only when grid origin, reducer lineage,
sufficient statistics, and valid coverage prove that the composition equals a
direct reduction. Otherwise the canonical route derives from the nearest valid
source, never from cache history.

## Page-table resolution and pins

A target virtual page resolves once on the CPU to either:

- the exact resident page; or
- the finest compatible resident ancestor whose source footprint covers the
  target, including actual key/LOD, physical slot, target-to-resident sample
  scale and offset, and that binding's generation.

Document/operation generation, representation, dtype, spatial coverage, and
reducer family must all match. Anisotropic reductions use componentwise
ancestry; reducer mismatch never aliases. The shader consumes the resolved
binding and does not walk an ancestor ladder per fragment.

Pins are owner-scoped sets, replaced atomically. Several active target pages
may share one coarse ancestor; one target leaving must not unpin coverage still
owned by another. Slot compaction/reuse refreshes binding generations so an old
resolution can never sample a new occupant. When all capacity is pinned,
refinement is denied and coarse coverage stays resident.

## Physical truth

A coarse ancestor drawn for a fine target is acknowledged as the actual coarse
LOD and fallback quality. It must never acknowledge the requested fine/exact
identity. Finer arrival changes only the page-table resolution; fine eviction
falls back to pinned coarse coverage without a black frame. Explicit missing
display is allowed only when no compatible resident ancestor exists.

## Ordered slices and gates

1. Pure `arrayscope/gpu` page geometry, ancestor resolution, binding
   generations, and owner-scoped pins.
2. VisPy consumption of resolved pages with actual-LOD physical truth and
   never-black fine-arrival/fine-eviction gates.
3. Ladder/cache migration from whole-plane `PyramidLevelKey` identity to
   logical `DataChunkKey` pages.
4. Source-grid reduction binning: origins 101 and 102 share aligned interior
   pages, keep clipped boundaries distinct, upload only boundary pages at
   factor > 1, and match the direct CPU source-grid oracle.
5. Reducer families and phase-cancellation correctness, followed by real
   Wayland GL certification.

## Implementation progress

The pure model and first VisPy consumption slice now stand without scheduler
coupling:

- `PageTable.resolve` returns exact or finest compatible covering residency,
  including actual key/slot, target-to-resident sample transform, and binding
  generation; unbinding fine content immediately exposes coarse fallback;
- slot remap/reuse mints new binding generations;
- `ChunkStore.replace_pins(owner, keys)` atomically owns coverage without one
  consumer unpinning another consumer's shared ancestor;
- `reduce_source_grid_mean` plans global anisotropic bins, reports native-source
  coverage/identity per sample, shares aligned interiors across origins 101/102,
  rejects clipped recursive inputs, and matches a direct CPU oracle;
- anchored atlas chunks now use canonical `DataChunkKey` identities instead of
  backend-private tuples; the mixed atlas page table deliberately excludes
  legacy whole-tile keys from logical ancestor lookup;
- `TextureAtlasPool.resolve_page_targets` performs one bounded CPU resolution
  pass, changes mappings/pins only, and never uploads or schedules. It rebinds
  fine arrival and fine removal in place, reports explicit missing only when no
  compatible page exists, and refreshes bindings after slot remap;
- owner pins are honored by every atlas eviction route, including speculative
  warm placement and superseded-page reclamation;
- physical presentation rows report target key, actual key/LOD, exact versus
  fallback quality, and binding generation; presented identity is the actual
  resident page, never the requested fine page;
- 111 focused GPU/pyramid/VisPy tests pass. The next slice is the live
  ladder/cache migration that emits logical page targets into this seam.

The ladder-side target planner is also now explicit:

- `render.lod.plan_lod_page_targets` is a Qt-free deterministic transform
  from content identity, native source rect, anisotropic reduction, and
  uniform stored-page shape to canonical `DataChunkKey` targets;
- factor-2 windows starting at 101 and 102 share the aligned interior page
  and keep both clipped boundaries distinct; desired mean-family identity
  stays separate from a coarser physical resolution;
- the planner is not yet attached to `DisplayTilePayload`. Current reduced
  ladder values are still binned from window-local zero, and attaching global
  target names to those texels would be false identity. The next slice must
  migrate materialization and boundary draw geometry together before feeding
  these targets into the VisPy resolver.

The pure boundary-geometry half of that atomic slice is now implemented:

- `partition_source_grid_pages` groups globally reduced samples into uniform
  stored-page classes while retaining the exact native-source rectangle of
  every sample for draw construction;
- clipped first/last factor-2 bins retain width one, aligned interiors retain
  width two, and the flattened draw spans cover every valid source coordinate
  exactly once;
- shifted windows share only a complete aligned interior page identity and
  attach byte-identical values to it. Boundary page identities remain distinct.

The remaining live step is to carry these page values and spans through the
ladder cache/payload contract and have VisPy build grouped quads from the spans.

## Rejected shortcuts

- backend-private tuple keys as the permanent pyramid identity;
- per-fragment ancestor walks or sampled-zero missingness;
- active exact chunks as a substitute for owner-scoped coarse pins;
- pinning every coarse page forever;
- acknowledging a coarse physical fallback as the fine target;
- wrapping a `DataChunkKey` in the legacy whole-tile residency tuple;
- treating `max_texture_size` as a total VRAM/page-count budget (pin-pressure
  denial gates use an explicit byte budget; without one the atlas may grow);
- treating previous-screen retention as data-level coverage;
- renaming window-origin reductions without changing their bins;
- stamping source-grid target keys onto window-origin reduced values;
- drawing a clipped boundary page as one uniformly stretched quad when its
  first/last partial bins have different native widths;
- reusing clipped boundary values across different valid footprints;
- preserving `PyramidLevelKey` behind a compatibility shim;
- deferring anisotropy or collapsing complex modes to one reducer family;
- changing scheduler or `prepare_rung` ownership to land an identity,
  reduction-grid, and residency feature.
