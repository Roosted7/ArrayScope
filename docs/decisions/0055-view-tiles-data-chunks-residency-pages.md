# ADR 0055: Separate view tiles, data chunks, and residency pages

- **Status:** Accepted (design); implementation staged as the G-program in
  [`docs/proposals/gpu-engine-plan.md`](../proposals/gpu-engine-plan.md)
- **Date:** 2026-07-15
- **Branch note:** authored on `codex/gpu-engine`; renumber on integration if
  a parallel branch has claimed 0055 (see the parallel-cowork ADR collision
  convention).
- **Related:** ADR 0037, 0038, 0040, 0041, 0046, 0049, 0050, 0051, 0053, 0054

## Context

"Everything presented is a tile" is the shared semantic across both rendering
backends, and it should stay. But the code currently conflates three concepts
that only *happen* to be 1:1 today:

1. **View tile** — a rectangular output region of the montage/frame
   (`tile_number`, `region_id`).
2. **Data chunk** — an N-dimensional block of evaluated array values.
3. **Residency page** — a physical allocation in backend memory (an atlas
   slot on VisPy, an `ImageItem` on PyQtGraph).

The 1:1 binding is load-bearing in three places:

- `display/frame_planner.py` (`_single_tile_regions`, `_plan_montage`): each
  view tile owns exactly one `data_slices` chunk.
- `display/region_source.py` (`EagerDisplayRegionSource.read_region`): the CPU
  materializes exactly that chunk into a `DisplayTilePayload`.
- `display/backends/vispy/tiles.py` (`TextureAtlasPool.tile_slots`) and
  `display/backends/pyqtgraph/tiles.py` (`MontageTileLayer._states`): one
  uniform-shape residency slot per tile number.

The costs are measurable and user-visible:

- **Same-extent slice-window shifts re-upload everything.** Scrolling a
  displayed-axis window from `X=100:200` to `X=101:201` keeps geometry
  identical, but `montage_tile_semantic_key` and the per-tile evaluator keys
  fold in `axis_range_indices`, so every tile misses, re-materializes on CPU
  workers, and re-uploads — while ~99% of the required values are already
  resident on the GPU one texel over.
- **Non-montage views rebirth the whole session on any index change**
  (`frame_controller._maybe_retarget_frame_session`, the `"no-axis"` branch),
  even when the source values are unchanged and resident.
- **Montage index scroll is fast only because of a special case**
  (`retarget_index_window` remaps resident payloads by source identity). That
  optimization is the degenerate instance of a general rule — "presented
  output re-references resident data" — which the 1:1 model cannot express
  for any other interaction.
- The VisPy backend already proves the value of decoupling presentation from
  residency: levels/LUT/complex-mode are shader uniforms
  (`GpuWindowedTileVisual`), so those changes cost zero uploads. Data identity
  is the remaining coupled axis.

## Decision

1. **Three formal concepts, three keys.** Introduce a Qt-free residency
   vocabulary (new `arrayscope/gpu/` package):
   - `ViewTileKey` — output identity: presentation, tile number/region,
     montage coordinates. Owned by the shared semantic layer.
   - `DataChunkKey` — value identity:
     `(document_generation, operation_key, lod, chunk_origin, chunk_shape,
     dtype, representation)`. Independent of where or whether it is drawn.
   - `ResidencyPageKey` / page table — physical placement: which pool, page,
     and slot currently hold a chunk, with residency generations and
     explicit "not resident" state.

2. **The relationship is many-to-many, mediated by a page table.** Many view
   tiles may sample shared chunks; one view tile may sample several chunks;
   one chunk may span pages. `frame_planner`'s 1:1 output remains valid as
   the *initial degenerate mapping*, not as an invariant. No shared-layer
   code may assume a view tile owns its payload.

3. **Backends receive a logical view description plus chunk references.**
   The `DisplayTiledPresentation` contract gains chunk-level identity so a
   backend can satisfy a view tile from already-resident chunks. Materialized
   numpy payloads remain one way to *source* a chunk — not the definition of
   a tile.

4. **Raw values are the canonical GPU representation.** Scalar `r32f`,
   complex `rg32f` (both already exist as atlas storage modes). Display
   mapping — levels, scale, LUT, complex component — stays in the shader;
   changing it never changes residency. "Raw" means *the earliest
   representation the GPU backend can use correctly*: for unsupported
   operations that is still the CPU-evaluated result, so GPU operation
   kernels are an optimization path, never a prerequisite.

5. **The GPU engine is insulated from VisPy.** `arrayscope/gpu/` owns chunk
   keys, page tables, pool policy, transfer planning, and (later) compute
   and compression. The VisPy layer consumes it for canvas/context/shader
   integration. Replacing VisPy must not replace the residency architecture.

6. **No new scheduler.** Residency requests, chunk materialization, and
   upload work flow through the existing kernel/pipeline/lifecycle
   (ADR 0053 explicitly forbids parallel scheduling systems). The page table
   is a data structure consulted by existing owners, not an actor.

7. **The backends diverge below the semantic seam, never above it.** Shared:
   view meaning, expected tiles, acceptable quality, generations and
   convergence, levels/histogram semantics (ADR 0054), interaction. Divergent:
   materialization, storage layout, indexing, residency, compression,
   histogram execution, rendering mechanics. PyQtGraph keeps the simple
   CPU pipeline (slice → map → `ImageItem`); the GPU endpoint compiles the
   same logical view into chunk residency + shader sampling.

8. **Time-to-first-image stays first-class.** Whole-array residency for small
   arrays is a background convergence target with explicit priority ordering
   (rough-level samples → visible output → adjacent indexes → remainder),
   never a first-frame prerequisite.

## Consequences

Positive:

- Same-extent window shifts, fixed-index scrolls over resident data, montage
  reshuffles, and axis reversals become coordinate/page-table updates with
  boundary-only uploads.
- Histogram/levels can later become compute reductions over the chunk store
  (sharing the shader's complex-mapping function) without re-modeling their
  inputs — ADR 0054's evidence phases already treat them as
  payload-independent.
- Compression, sparse residency, and GPU operation kernels get a home that
  does not require another foundational rewrite.
- The existing atlas becomes the first page-pool implementation rather than
  dead weight.

Costs:

- A third identity axis (chunk) joins tile identity and presentation
  identity; key derivations must stay consistent with the evaluator's cache
  keys (the `montage_tile_key`/`montage_tile_key_batch` equivalence problem
  now has a third participant).
- Chunk layout vs. displayed-axis order is a real problem (strided gathers);
  the plan addresses it with size-tiered policies, not a single canonical
  layout.
- PyQtGraph and VisPy internals drift further apart; the backend contract and
  its conformance tests carry more weight.

## Rejected alternatives

- **Two independent viewers per backend.** Duplicates axis/index semantics,
  quality policy, generations, and interaction; history (ADR 0038–0040)
  shows that drift becomes user-visible bugs.
- **Hardware sparse textures as the foundation.** Device-dependent support
  and page granularity; software virtual texturing provides the same
  behavior portably. Hardware sparse remains an optional later backend
  (G-program stage G7).
- **Compression first.** Compressing after upload cannot reduce the upload;
  codec policy for scientific dtypes needs the chunk store to exist first.
- **Requiring GPU kernels for all operations before switching.** Blocks the
  entire endpoint on the hardest 10%; the "earliest correct representation"
  rule lets CPU-evaluated chunks flow through the same store.
