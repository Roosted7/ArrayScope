# Proposals

Proposals describe a concrete design that has not yet become an accepted architecture decision or active roadmap commitment.

- [Operations as a platform](operations-as-a-platform.md): the successor to the custom-operations
  program — a native everyday toolbox (demoting the NumPy-trivial sigpy/BART ops that pay a
  complex128 promotion or a subprocess for arithmetic), one definition format so any operation can
  be read and duplicated into an editable copy, the hidden subprocess bridge promoted to a
  user-editable command runtime with named execution environments, discovered instead of declared
  shapes, and input slots for ROIs and second arrays. Decisions in
  [ADR 0060](../decisions/0060-operation-definitions-runtimes-and-discovered-shapes.md).
- [Axis information](axis-info.md): names, units, coordinates, spacing, and orientation. A first internal `AxisInfo` model exists, but broad IO/UI propagation remains incremental work.
- [Tensor exploration engine — endpoint architecture](tensor-engine-endpoint.md): the
  deadline-driven engine the G-program converges toward — resource-broker-as-kernel-evolution,
  operation classes, frame-quality controller, precision tiers, adaptive compression,
  renderer-protocol strategy. Direction record; the G-plan stays the executable program.
- [GPU engine implementation plan (G-program)](gpu-engine-plan.md): staged route from the
  [ADR 0055](../decisions/0055-view-tiles-data-chunks-residency-pages.md) three-way
  view-tile/data-chunk/residency-page separation to a virtual, streamed, optionally
  compressed N-D GPU data store. Active on `codex/gpu-engine`.
- [Tensor ops on the GPU — G8 candidate](tensor-ops-g8.md): the T0–T4 ladder for running
  operations as GPU compute over the ADR 0057 command protocol ("upload the source once,
  derive on the GPU"), with explicit triggers, the FFT-scroll kill-gate at T1, and the
  CPU-as-value-oracle trust split. Explicitly not now; named so the deferred experiment is
  designed instead of improvised.
- [wgpu renderer experiment](wgpu-renderer-experiment.md): the wgpu-py evidence program that
  produced the ADR 0057 backend (gates, measurements, native-Wayland recipe).
- [wgpu shader legibility and filtering](wgpu-shader-legibility.md): four stages making the
  wgpu frame self-describing — pixel grid, NaN/missing-page/clip trust signals, montage
  guides and slice labels, honest minification filtering, per-pixel value labels. Stage A is
  implemented (default off); B–D are the live part. Argues explicitly *against* GPU mipmaps
  for this backend, since hardware mip generation cannot express the pyramid's semantic
  reducers.
- [GPU-port continuation](gpu-port-continuation.md): continuation record for the GPU-engine
  port sessions.
- [LOD multi-resolution implementation plan](lod-multires-implementation-plan.md): historical —
  implemented by [ADR 0050](../decisions/0050-async-multi-resolution-tile-residency.md) and
  [ADR 0051](../decisions/0051-single-owner-tile-lifecycle.md); retained for its cache-key and
  storage-class rationale.

A proposal should state the user problem, ownership, compatibility/migration, testing, and why it is not yet an ADR. Move it to an ADR only when the direction is accepted; move implementation work to the roadmap only with an exit gate.
