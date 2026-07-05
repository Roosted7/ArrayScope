# Proposals

Proposals describe a concrete design that has not yet become an accepted architecture decision or active roadmap commitment.

- [Axis information](axis-info.md): names, units, coordinates, spacing, and orientation. A first internal `AxisInfo` model exists, but broad IO/UI propagation remains incremental work.
- [LOD multi-resolution implementation plan](lod-multires-implementation-plan.md): historical —
  implemented by [ADR 0050](../decisions/0050-async-multi-resolution-tile-residency.md) and
  [ADR 0051](../decisions/0051-single-owner-tile-lifecycle.md); retained for its cache-key and
  storage-class rationale.

A proposal should state the user problem, ownership, compatibility/migration, testing, and why it is not yet an ADR. Move it to an ADR only when the direction is accepted; move implementation work to the roadmap only with an exit gate.
