# Proposals

Proposals describe a concrete design that has not yet become an accepted architecture decision or active roadmap commitment.

- [Axis information](axis-info.md): names, units, coordinates, spacing, and orientation. A first internal `AxisInfo` model exists, but broad IO/UI propagation remains incremental work.
- [Tensor exploration engine — endpoint architecture](tensor-engine-endpoint.md): the
  deadline-driven engine the G-program converges toward — resource-broker-as-kernel-evolution,
  operation classes, frame-quality controller, precision tiers, adaptive compression,
  renderer-protocol strategy. Direction record; the G-plan stays the executable program.
- [GPU engine implementation plan (G-program)](gpu-engine-plan.md): staged route from the
  [ADR 0055](../decisions/0055-view-tiles-data-chunks-residency-pages.md) three-way
  view-tile/data-chunk/residency-page separation to a virtual, streamed, optionally
  compressed N-D GPU data store. Active on `codex/gpu-engine`.
- [LOD multi-resolution implementation plan](lod-multires-implementation-plan.md): historical —
  implemented by [ADR 0050](../decisions/0050-async-multi-resolution-tile-residency.md) and
  [ADR 0051](../decisions/0051-single-owner-tile-lifecycle.md); retained for its cache-key and
  storage-class rationale.

A proposal should state the user problem, ownership, compatibility/migration, testing, and why it is not yet an ADR. Move it to an ADR only when the direction is accepted; move implementation work to the roadmap only with an exit gate.
