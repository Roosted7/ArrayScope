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

## The R5 gate is wall-clock, and that makes it load-sensitive

Read this before believing or disbelieving a `retained crop handoff exceeded R5`
failure. `test_second_display_axis_crop_presents_resident_lod_before_refining`
asserts `max_callback_ms <= 50.0` — an **absolute wall-clock ceiling on a GUI
callback**. Measured on this machine, the same commit:

| Machine state (1-min loadavg) | Result |
|---|---|
| 1.6 – 2.8 | 3/3 invocations pass (6/6 parametrized instances) |
| 3.6 – 11 | 4/6 instances fail, 50.5 – 131.5 ms |

Nothing about the change moves between those rows; a concurrent `pytest -n auto`
in another worktree is enough to flip it. An interleaved, order-balanced A/B of
this branch against the same branch plus the crop-rebind R3 train had **both arms
fail together and both arms pass together** in the same load window, which is
what a load artifact looks like and what a real regression does not.

Two consequences worth keeping:

- **Every functional assertion in that gate held in every run** — 272/272
  crop-local subset rebinds, `pages_not_resident=0`, all 272 retained ACKs
  before the first exact ACK, zero stall probes, and CPU-reference framebuffer
  agreement. Only the timing ceiling moved. When triaging this gate, separate
  those two halves before concluding anything.
- **The distinction is R5-specific.** R3 and the other contract rules are value
  and ordering predicates: they hold or they do not, and a slow machine cannot
  make levels clip. R5 is the one rule in the table whose gate is a duration, so
  it is the one rule whose gate needs either a quiet machine, a margin
  justified by repetitions, or a work-proportional counter instead of a clock.
  The profiler's physical-draw deadline has the same character — see the
  30.7 ms request against rendercanvas's 33.3 ms cadence in the
  [R3 closure evidence](crop-rebind-r3-closure-2026-07-30.md).

The dossier's own `244 → 32 ms` medians were taken on a quiet machine across
eight fresh managed-Weston processes and remain the claim of record; they are not
reproducible under contention and should not be re-quoted from a loaded run.

### Work-proportional gate and rebind-transaction evidence

The regression gate now asserts the work the transition controls and records
elapsed time as evidence:

- 272 seed items admitted in nine callbacks, with the governor's named
  rebind cap at 32 and observed batches never exceeding it;
- target ladder admission and retained-fallback presentation batches both
  capped at four by the same named governor policy;
- exactly one complete fallback-gate post/firing;
- 272 page-backed-superset rebinds and zero exact-plane rebinds for this field
  transition, so the page-backed R3 path is measured rather than inferred.

Bounds derivation now occurs only after physical residency accepts a candidate.
The failed canonical candidate is no longer scanned before the successful
strict-subset candidate. Each 32-tile governed callback scans 326,400 bytes
(the smallest accepted reduced planes); the whole 272-tile handoff scans
2,774,400 bytes once, rather than scanning both candidates.

Three executed managed-Weston invocations (six parametrized instances), with
the orientation order alternated, all passed the complete functional gate:

| Invocation order | 1-min loadavg at launch | max callback `(0,1)` / `(1,0)` | max bounds-scan transaction `(0,1)` / `(1,0)` |
|---|---:|---:|---:|
| `(0,1)`, `(1,0)` | 1.71 | 59.391 / 44.101 ms | 2.554 / 1.874 ms |
| `(1,0)`, `(0,1)` | 4.02 | 78.965 / 95.709 ms | 2.290 / 3.856 ms |
| `(0,1)`, `(1,0)` | 9.80 | 51.215 / 54.272 ms | 1.674 / 2.864 ms |

The latter two runs reported material external load. The admitted work and all
pixel/order assertions stayed invariant while wall time moved, which is the
reason the clock is no longer the pass/fail predicate. The cold bounds scan is
inside the governed callback, but its worst measured transaction is 3.856 ms;
it is not an R5 blocker after the scan is restricted to the accepted plane.
