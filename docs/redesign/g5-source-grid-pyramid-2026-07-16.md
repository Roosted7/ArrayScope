# G5 source-grid sparse pyramid contract — 2026-07-16

**Status:** authoritative implementation contract for the remaining ADR 0056
work; pure page resolution/pins and source-grid mean geometry implemented.
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

The first pure slice is now standing without Qt, VisPy, or scheduler coupling:

- `PageTable.resolve` returns exact or finest compatible covering residency,
  including actual key/slot, target-to-resident sample transform, and binding
  generation; unbinding fine content immediately exposes coarse fallback;
- slot remap/reuse mints new binding generations;
- `ChunkStore.replace_pins(owner, keys)` atomically owns coverage without one
  consumer unpinning another consumer's shared ancestor;
- `reduce_source_grid_mean` plans global anisotropic bins, reports native-source
  coverage/identity per sample, shares aligned interiors across origins 101/102,
  rejects clipped recursive inputs, and matches a direct CPU oracle;
- 74 focused GPU/pyramid tests pass. The next slice is VisPy consumption of
  these resolutions with actual-coarse physical acknowledgement.

## Rejected shortcuts

- backend-private tuple keys as the permanent pyramid identity;
- per-fragment ancestor walks or sampled-zero missingness;
- active exact chunks as a substitute for owner-scoped coarse pins;
- pinning every coarse page forever;
- acknowledging a coarse physical fallback as the fine target;
- treating previous-screen retention as data-level coverage;
- renaming window-origin reductions without changing their bins;
- reusing clipped boundary values across different valid footprints;
- preserving `PyramidLevelKey` behind a compatibility shim;
- deferring anisotropy or collapsing complex modes to one reducer family;
- changing scheduler or `prepare_rung` ownership to land an identity,
  reduction-grid, and residency feature.
