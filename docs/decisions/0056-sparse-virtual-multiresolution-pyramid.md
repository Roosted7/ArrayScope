# ADR 0056: Sparse virtual multiresolution pyramid

- **Status:** Accepted; canonical source-grid planning, CPU materialization,
  shared logical-page caching, backend resolution, and physical-truth wiring
  implemented on the G5 landing candidate. Final real-Wayland/stress
  acceptance is tracked by row 1 of [`docs/queue.md`](../queue.md).
- **Date:** 2026-07-15
- **Branch note:** authored on `codex/gpu-engine`; renumber on integration if
  a parallel branch claimed 0056.
- **Related:** ADR 0041, 0050, 0051, 0054, 0055

## Context

ADR 0055 separated view tiles, data chunks, and residency pages, and the
G3/G4a implementation made chunk residency content-keyed and warm across
window shifts and index scrolls. The LOD system, however, still treats
reduced representations as *versions of screen tiles* (ladder rungs bound to
tile numbers), which leaves known gaps: reduced-LOD planes re-upload whole on
window shifts, fallback is arbitrated per tile rather than per data region,
and every display mode shares one reduction even though complex reductions
are mode-dependent (`abs(mean(z)) ≠ mean(abs(z))` — phase cancellation can
zero a mean whose samples all have high magnitude).

## Decision

Make the LOD system **logically sparse now**; hardware sparse textures are
never a prerequisite (they also cannot express our fallback semantics —
zero-on-missing is indistinguishable from a genuine zero in scientific data).

1. **LOD belongs to data identity, not to screen tiles.** A reduced
   representation is an independently addressable logical chunk:
   `(document_revision, operation_revision, reduction_vector, reducer,
   chunk_coordinates, dtype, representation)`. It must never contain
   viewport position, screen-tile identity, levels, colormap, or physical
   slot identity.

2. **Anisotropic reduction from the start.** Identity carries a per-axis
   `reduction_vector` (log2 steps per axis, `0` = native). Implementations
   may request only isotropic levels initially, but the key model permits
   `(4, 1)`-style reductions so a 16×-compressed X axis need not blur Y.

3. **Reducer is semantic identity.** `mean`, `RMS`, `mean(abs)`, power,
   circular-phase, and complex-colour reducers produce *different data*.
   Mode-specific derived LOD families share raw L0 chunks and are generated
   lazily within a budget; the family in use is named in the chunk key.
   Display-only approximate reducers are permitted but recorded as such in
   the key. Recursive parent→child generation is allowed only when the
   reducer is associative or sufficient statistics (count/sum/sum-of-squares/
   vector-sum) ride along.

4. **Uniform plane-pixel pages across LODs.** A physical page holds a fixed
   sample extent (e.g. 256²) at any reduction; a coarser page simply covers
   more source. Pools are keyed by storage format, never by LOD. Uploads are
   uniformly sized, eviction is format-local, and one coarse page substitutes
   for many fine ones.

5. **Best-resident-ancestor resolution in the page table.** A lookup for a
   target-LOD virtual page resolves directly to the best currently resident
   representation (slot, actual LOD, coordinate scale/offset, generation) —
   no per-fragment ladder walks. Entries carry generations so an evicted and
   reused slot can never be sampled through a stale mapping. When a finer
   page arrives, only the page-table entry updates.

6. **Pinned coarse coverage; never-black fallback.** For every actively
   rendered region some coarser representation stays pinned. Eviction order:
   cold fine pages, speculative pages, distant target pages — coarse
   fallback coverage last. Missing-page display uses explicit residency
   metadata, never a sampled zero.

7. **Histogram/levels consume a non-overlapping coverage frontier.** Every
   page carries/derives a small summary; when children replace a parent, the
   parent's contribution is removed before the children's are added, so each
   source region contributes exactly once at its best available
   representation (extends ADR 0054's evidence ordering to page granularity).

8. **LOD generation is sourced where cheapest** (precomputed pyramid, CPU,
   GPU-from-resident-chunks, persistent cache) and never uploads L0 merely to
   derive a coarse level obtainable directly.

9. **Bricks may retain non-displayed depth** (e.g. 128×128×4×4): backend
   policy, not semantic identity — chunk keys stay N-D-capable, transfer
   sizing stays private to the backend.

**Implementation clarification (2026-07-16):** exact residency and resolvable
coverage are deliberately different queries. Producers, singleflight claims,
prefetch, and lifecycle `RESIDENT` admission require every exact planned
`DataChunkKey`; a compatible coarse ancestor is presentation fallback only and
must not suppress finer materialization. Floor selection ranks the actual
resolved physical LOD, not a hypothetical target rung. VisPy and PyQtGraph both
delegate ancestor choice to `PageTable.resolve`, including anisotropic ranking
and full value-family rejection (including reducer and gutter). A page-cache claim attaches to the complete
requested set, touches shared resident members before admitting missing
boundaries, and therefore evicts outgoing pages before the requested set's own
interior. An exact set larger than the configured cache is ineligible until the
budget changes; it is not repeatedly materialized into an impossible cache.
Reduced page values remain presentation-qualified unless a payload also carries
explicit native semantic data; exact histogram/ROI/measurement/export reads
must not silently consume display fallback.

The page table's scale/offset is a nominal aligned-grid transform. Clipped
boundary bins require the immutable target and actual page-plan draw blocks;
both backends use those exact source rectangles rather than uniformly
stretching the nominal transform. A complete same-source target already
resolvable through physical pages rebinds atomically during pan/zoom, including
while the interaction remains active; only a physically cold successor may be
deferred to the stop edge.

## Consequences

Positive: the reduced-LOD re-upload gap closes by construction (uniform
pages + content keys); fallback becomes a data-level guarantee instead of a
per-tile negotiation (the black-tiles failure class loses its mechanism);
complex display modes get semantically correct overviews; compression (G7)
and hardware sparse allocators remain drop-in physical options.

Costs: derived families multiply residency pressure (mitigated: lazy,
budgeted, LRU within the family); sufficient-statistics pages cost memory
(only exact reducers need them; display-approx reducers are the default and
say so in their key); migrating the ladder is staged — the ladder keeps
owning *policy* (acceptable quality, convergence) while chunk identity moves
to the data layer, mirroring how ADR 0055 was landed without a rewrite.

## Rejected alternatives

- **Hardware sparse textures as the foundation** — mip-tail and
  missing-read semantics are implementation-defined; zero-on-missing is
  semantically unusable; sparse hardware solves none of LOD choice,
  priorities, reducers, generations, or histograms.
- **Scalar LOD levels in identity** — bakes isotropy into keys and forces a
  later migration exactly when many consumers exist.
- **One pyramid from averaged real/imag channels for all modes** — wrong for
  magnitude/phase/power displays (phase cancellation).
- **Per-fragment resident-ancestor search in the shader** — divergent and
  redundant; the page table already knows the answer.
