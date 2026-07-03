# 0049 — Out-of-core and lazy array sources

**Status:** Accepted (2026-07). First slice implemented: source protocol, budgeted
read seam, and memory-mapped `.npy`/`.cfl` adapters.

## Context

Every load path materialized the full array into process memory before a window
opened: `io/file_interpreters.load_path` returned an eager `np.ndarray`, and
`BartLoader` even copied an already memory-mapped `.cfl` into RAM. Files larger
than memory could not be opened at all, and files near the memory budget paid a
long, unbounded decode before the first frame.

The evaluation pipeline above the base array was already region-first: the
runtime region planner (ADR 0026) computes `required_input_region` before any
read, slab evaluation is cooperative and cancellable (ADR 0012), and stage and
display caches are bounded (ADRs 0016, 0027, Y3). The one missing piece was the
bottom of the stack: base data was read with direct ndarray indexing
(`apply_region(document.base_data, …)` in `operations/slabs.py`), which assumes
an in-memory array and gives transport/decoding no explicit boundary. The X5d
exit gate also asks for "a clear extension point for memory-mapped/chunked
sources" under region-first materialization, and the roadmap item is explicit:
request planning, cancellation, and memory budgets must stay above the source
adapter so "lazy" does not mean unbounded transport or decoding.

## Decision

Introduce an explicit source protocol at the bottom of the evaluation stack and
route every base-data read through one budgeted seam.

- **`core/array_source.py` (Qt-free).** `ArraySource` is a protocol exposing
  `shape`, `dtype`, `nbytes`, an optional `chunk_shape` hint, a `label`, and
  `read_region(index_spec, *, cancellation_token=None) -> np.ndarray`, where
  `index_spec` holds one item per axis (int, slice, or tuple of indices). An
  adapter transports and decodes exactly the region it is asked for and returns
  in-memory data that does not alias the backing store. `NdArraySource` adapts
  any ndarray-like backing store, including `np.memmap`.
- **`LazySourceArray` is the document base-data proxy.** `ArrayDocument`
  continues to hold one `base_data` object; for lazy sources that object is a
  `LazySourceArray`, which exposes `shape`/`dtype`/`nbytes` so planning,
  document keys, cost estimates, and memory policy inputs work unchanged
  without reading, and delegates explicit region reads to the source. Implicit
  whole-array conversion (`np.asarray` and friends) goes through
  `materialize()`, which refuses beyond a byte budget instead of silently
  decoding everything. Baking a document (`materialize`) remains possible for
  in-budget sources and is refused with a clear error beyond it.
- **One read seam, budgets above the adapter.**
  `operations/source_read.read_base_region` is the only place evaluation reads
  base data. Eager ndarrays keep the existing direct indexing. Lazy sources get:
  cancellation check, byte estimate of the planned region
  (`region_nbytes` on the already-planned `required_input_region`), refusal via
  `SourceReadRefused` when the estimate exceeds the lane-aware budget
  (prefetch lane uses the prefetch budget, other lanes the visible-render
  budget, floored at a module default), then one explicit `read_region` call.
  The three base reads in `operations/slabs.py` (plain, stage materialization,
  stage-cache evaluation) all go through this seam; a test guard keeps direct
  `base_data` indexing out of the tree.
- **First adapters: memory-mapped files.** `io/lazy_sources.open_memmap_source`
  maps `.npy` (`np.load(mmap_mode="r")`) and BART `.cfl` (Fortran-order
  `np.memmap`). `load_path(lazy="auto")` opens supported files lazily at or
  above a threshold (25% of available memory, floored at 64 MiB) and eagerly
  below it; `lazy=True/False` forces either behavior, and unmappable `.npy`
  files (object/pickled) fall back to eager loading under `auto`. Loaded-lazy
  windows are labeled `[…, lazy]`.

Chunked stores (Zarr/HDF5-like) are a later adapter behind the same protocol:
they implement `read_region` plus a real `chunk_shape` hint, and inherit the
seam's budgets and cancellation without new pipeline code.

## Consequences

- Files larger than memory open instantly; decoding cost is paid per planned
  region, bounded by lane budgets, and cancellable between reads.
- The region planner's `required_input_region` is now also the transport
  request. Nothing below the planner can widen a read.
- A lazy read that would exceed its budget fails loudly (`SourceReadRefused`)
  instead of paging the machine; the refusal carries requested and budget
  bytes for diagnostics.
- Slab, stage-cache, chunked, and montage evaluation are source-agnostic; all
  existing value/shape tests run identically over lazy documents.
- Repeated reads of the same region re-decode from the map. That is the stage
  and display caches' job; the adapter stays cache-free by design.
- Hover/scalar inspection over lazy sources performs small explicit reads;
  exact values stay independent of display state, as required by X5.
- `np.asarray(document.base_data)` on a huge lazy source now raises instead of
  hanging. Call sites that legitimately need full materialization must call
  `LazySourceArray.materialize` and handle refusal.

## Alternatives considered

### Hand `np.memmap` directly to `ArrayDocument`

Rejected. It works mechanically (memmap is an ndarray), but every full-array
NumPy call silently decodes the entire file, exactly the unbounded behavior the
roadmap forbids, and chunked stores could never fit the same seam.

### Dask or another lazy-array framework

Rejected for the core seam. The pipeline already owns planning, budgets,
cancellation, and caching; a task-graph library would duplicate all four and
blur ownership. Third-party lazy containers can still be wrapped as adapters.

### Cache inside the source adapter

Rejected. Bounded reuse is owned by the stage/display caches above the seam
(one eviction implementation, Y3). Adapter-level caches would create a second,
invisible memory consumer.

### Async/queued source reads

Deferred. Reads are synchronous inside worker lanes today, which preserves the
existing scheduler semantics. Queued transport belongs with the X5e
LOD/source-provided-level work once acknowledged-residency evidence exists.

## Migration

1. Done in the first slice: protocol + `NdArraySource` + `LazySourceArray`,
   budgeted `read_base_region` seam, memmap `.npy`/`.cfl` adapters, `lazy="auto"`
   loading, value-parity and refusal tests.
2. Next candidates: a chunked (Zarr/HDF5-like) adapter with a real
   `chunk_shape`; wiring the dataset selectors (`.h5`/`.npz`/`.mat`) to lazy
   sources; chunk-aligned region planning hints; surfacing refusals in the UI
   as actionable guidance rather than a generic error.
3. Region-first display materialization (X5d) can consume `read_region`
   directly when huge single planes stop requiring a full display image.

## Related records

- ADR 0012: lazy slab evaluation.
- ADR 0016: evaluation scheduler and memory budget.
- ADR 0025: operation capabilities and region contracts.
- ADR 0026: runtime region planner.
- ADR 0027: in-memory stage cache.
- ADR 0046: evidence-first performance strategy (X5d extension point).
