# Second-axis crop retained-LOD handoff — 2026-07-30

## Decision

Present the already-resident crop-local plane as a display-only fallback, then
refine to the round target through the ordinary governed ladder. Do not warm a
future crop level: that would produce a rung the round did not request and
violate progressive-render R1.

This is a narrow transition rule. It applies only when a page-backed montage
already has one displayed axis cropped and the successor adds the other crop as
a strict sub-rectangle. Ordinary crop rebind, exact CPU-backed semantics,
canonical source-plane reuse, and one-axis crop behavior keep their incumbent
paths.

## Mechanism and ownership

- `FrameSession` re-expresses the predecessor's reduced descriptor arrays over
  the narrower current source rectangle. The wrapper is preview quality and
  retains the predecessor anchor only as physical page provenance.
- WGPU proves the predecessor crop-local page keys resident and binds them with
  the successor source-origin offset. This is a zero-upload mapping reuse, not
  new production.
- The deferred stage completion seeds 272 rebinds in bounded visible-path
  callbacks. The complete rebound scope publishes first; target admission and
  render-pass commits then use governor-owned four-item caps.
- A one-shot transition latch prevents later refinement replans from
  re-installing the fallback over exact target payloads.

## Managed-Weston evidence

The exact 336×336×272 seeded field fixture ran in eight fresh managed-Weston
processes, order-balanced and interleaved across `main` and this change, with
both `(0, 1)` and `(1, 0)` image-axis orientations.

| Metric | `main` (4 runs) | retained fallback (4 runs) |
|---|---:|---:|
| First current-session pixel, median | 2586 ms | 1320 ms |
| Maximum governed GUI callback, median | 244 ms | 32 ms |
| Worst governed GUI callback | 293 ms | 35 ms |
| Rebind result | 0/272; `pages_not_resident=272` | 272/272 crop-local subsets |

The first-current-pixel latency improves 49%. Every changed run stays below the
R5 50 ms callback ceiling and reports zero stall assertions. Full target
settlement is deliberately background quality work and is slower in this
prototype (median about 9.3 s versus 5.8 s on `main`); that is accepted here as
the R6-style liveness trade: retained pixels arrive early and no GUI callback
waits for target convergence. It is not a claim that R6 is generally
implemented.

The committed field gate additionally requires, in both orientations:

- all 272 retained preview/fallback ACKs precede the first exact ACK;
- full eventual settlement;
- zero stall probes;
- final WGPU framebuffer agreement with the CPU semantic reference.

The existing resident-crop UI module remains 21 green with exactly its four
clean-`main` inherited level-reanchoring reds.
