# WGPU cropped-axis page resolution (2026-07-23)

## Product requirement

A displayed-axis crop is a view over array values, not a new array allocation.
The GPU endpoint should retain useful source-aligned chunks and compile each
tile to the resident pages plus source coordinates it needs. The relationship
is intentionally many-to-many:

- one tile may sample one or several resident pages;
- one resident page may serve one or several tiles or successor frames;
- a one-pixel crop/index change uploads only a genuinely cold boundary page;
- presentation-only changes never materialize or upload source values.

This is the existing ADR 0055/0056 contract. A dedicated crop-shape fast path
would preserve the wrong abstraction and leave the next geometry combination
untested.

## Red-first benchmark

`profile_montage_workflow` now runs the same crop matrix in both
`display_x_axis_slice` and `display_y_axis_slice`, on both maintained
backends:

1. one centered 100-sample displayed-axis crop;
2. both displayed axes cropped;
3. independent one-pixel moves on the primary and secondary axes;
4. diagonal motion and a return;
5. an exact 256-sample source-page edge, boundary crossing, and return;
6. odd 99x101 extents and a one-pixel successor;
7. X/Y role swap and return;
8. montage to current/+1/current single-slice and montage restore.

Every crop successor records settlement, committed-frame currency, physical
tile coverage, upload delta, binding-cache hit/miss delta, and page fan-in.
The WGPU gate requires zero uploads after the source montage has made the
needed pages resident.

## Baseline on `19e757c4` plus benchmark-only changes

Real Wayland, Intel low-power Vulkan adapter, scalar float32 NIfTI
`(336, 336, 272)`, 50 montage tiles:

| Stage | One-axis crop | Both-axis crop | First axis -1 | Second axis +1 |
|---|---:|---:|---:|---:|
| X primary | 864 ms | 727 ms | 632 ms | stalled at 3208 ms |
| Y primary | 984 ms | 812 ms | 639 ms | stalled at 3181 ms |

All observed crop successors performed zero WGPU uploads. The failure is
therefore not cold transfer pressure: after the second displayed dimension
moves, lifecycle targets remain planned/unpresented while no evaluator work is
active and the committed frame stays on the predecessor.

The opt-in cProfile run on the failing X stage recorded 33.2 million calls.
The largest crop-path Python costs were repeated physical-truth diagnostics
(`tileTruthPhysicalRows`: 1.93 s cumulative), presentation commits
(`_present_tile_delta`: 6.21 s cumulative), chunk histogram sample
reconstruction (`representative_sample_from_histogram`: 3.72 s cumulative),
and lifecycle settlement scans (0.73 s cumulative). WGPU payload binding was
0.43 s cumulative; uploads remained zero. GPU histogram resolves dominated
worker cumulative time and shutdown overlap, so profiler attribution and
plain timing remain separate evidence.

Artifacts:

- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu-trace.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu.cprofile`
- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu-cprofile.jsonl`

## Correctness root cause and owner fix

The cropped payloads were physically resident. The presentation coordinator
nevertheless required a second, session-local `(source_id, levels)` warm
marker before it allowed the atomic successor to bind. Fifty already-resident
tiles therefore entered 25 two-tile low-priority warm callbacks. The callbacks
performed no uploads, but their serialized replan cycle exceeded the
interaction deadline and left the successor planned but unpresented.

Physical residency is now the sole warm owner whenever a backend exposes
`tiledPayloadResident` or `tiledPayloadCommitSlotOwned`. The historical marker
remains only as a fallback for backends without a physical predicate. The
existing marked-but-evicted test remains green: a stale marker cannot override
lost residency.

The real-Wayland WGPU rerun completed all 11 X-primary and all 11 Y-primary
crop states with current committed frames, zero uploads, and zero
`hidden-warm-residency-wait` trace events. This closes the freeze, but not the
performance work: resident-only cells still take up to 3.65 s and GUI
callbacks exceed the 50 ms bar.

Artifacts:

- `/tmp/arrayscope-xy-crop-matrix-owner-fix-wgpu.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-owner-fix-wgpu-trace.jsonl`

## Performance iterations

The post-correctness cProfile showed that the 1 ms continuity oracle rebuilt
the complete WGPU physical-row and page-pool diagnostic tree on every sample.
An allocation-light physical count now checks only the committed tiles' page
keys. Rich diagnostics remain available on their explicit path. On the same
real-Wayland X matrix this reduced total crop settlement from 13.89 s to
8.59 s and the worst cell from 3.65 s to 1.83 s.

The next cProfile then exposed 508 synchronous queue-buffer reads for resident
histogram evidence: each of 254 per-tile dynamic histograms independently read
counts and bounds. `WgpuPlaneExecutor` now preserves every histogram result
but copies all deferred results in a `FrameSubmission` into one staging buffer.
Resolving all tile evidence performs one queue read per frame submission.
The X matrix fell again to 6.34 s total and 0.87 s worst-cell settlement.
Both iterations retain complete/current physical coverage and zero crop
uploads.

Artifacts:

- `/tmp/arrayscope-xy-crop-matrix-light-continuity-wgpu.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-light-continuity-wgpu.cprofile`
- `/tmp/arrayscope-xy-crop-matrix-batched-histogram-wgpu.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-batched-histogram-wgpu-trace.jsonl`

## Final backend-parity gate

The committed implementation was rerun on real Wayland with both displayed
axis roles in one fresh process per backend. Every one of the 22 successors
settled on a current committed frame. Neither trace contained a stall or
hidden-warm-wait edge.

| Backend | X matrix total / worst | Y matrix total / worst | WGPU uploads |
|---|---:|---:|---:|
| WGPU | 6.16 s / 0.87 s | 6.36 s / 0.92 s | 0 |
| PyQtGraph | 4.76 s / 0.91 s | 5.65 s / 1.03 s | n/a |

The correctness and per-interaction settlement gates are green. The complete
R8 rows remain red only on the standing 50 ms GUI-callback bar (observed
process maxima 463–498 ms WGPU and 380–616 ms PyQtGraph); this slice does not
claim that pacing bar.

Artifacts:

- `/tmp/arrayscope-xy-crop-final-wgpu.jsonl`
- `/tmp/arrayscope-xy-crop-final-wgpu-trace.jsonl`
- `/tmp/arrayscope-xy-crop-final-pyqtgraph.jsonl`
- `/tmp/arrayscope-xy-crop-final-pyqtgraph-trace.jsonl`

## Implementation direction

Replace geometry-specific binding selection with one resolver:

1. derive the requested source rectangle and value family;
2. query source-aligned page coverage at each acceptable resident LOD;
3. select the best complete non-overlapping coverage;
4. emit page references plus exact source-to-page draw blocks;
5. request materialization only for uncovered source regions.

The resolver, not montage/non-montage or one-axis/two-axis branches, must own
the decision. Lifecycle and level evidence should consume its immutable result
instead of repeatedly reconstructing page keys or semantic samples on the GUI
thread.

This dossier will be updated with each bounded optimization and its before /
after evidence.

## 2026-07-24 stale-crop and full-pool follow-up

The expanded all-dimension journey reproduced a worse failure than slow
settlement: after shifting both displayed-axis windows and then scrolling
other dimensions, the lifecycle could report a fully current 50/50 frame
while physical tiles sampled several predecessor crop origins.

Three mutable-owner leaks formed the stale-pixel chain:

1. an in-place `FrameSession` retarget updated `view_state` but retained the
   session's original `source_anchoring`;
2. asynchronous preview completion derived its page/source anchor from that
   mutable live session instead of the immutable rendered tile snapshot; and
3. a queued presentation-gate event derived its owner generation when it ran,
   so an event posted by a predecessor could act for its successor.

The session retarget now replaces source anchoring before payload restamping,
preview work derives anchoring from its immutable tile state, and every queued
presentation gate carries the owner generation captured at post time.
Physical WGPU diagnostics expose the committed source origin and extent.

The regression no longer stops at lifecycle identities. In both maintained
backends it keeps both displayed axes cropped, applies fast/slow scrolls over
all dimensions, and compares every settled checkpoint with CPU semantic
pixels. WGPU also checks session/payload/physical source-origin agreement;
deterministic screenshots are retained when requested.

The full 272-plane `(336, 336, 272)` run exposed a related capacity-accounting
bug. A reduced preview texture occupies one page, but the commit deliberately
replaces it with four reusable 256-square native pages. Pool admission counted
the preview (272 pages), then tried to install `272 × 4 = 1088` native pages
under a 1024-page retention preference and failed once every resident page
was bound. Admission now counts the physical replacement, treats byte policy
as retention rather than correctness, and reserves the immutable frame-plan
working set in one copy-preserving growth. The offscreen Vulkan full-file
repro completed 272/272 with 1088 resident/pinned pages and no exhaustion;
pool growth fell from 15 resizes / 2.88 GB copied to one resize / 6 MB copied.
This is diagnostic GPU evidence; the real-Wayland ring remains the acceptance
gate for presentation timing and compositor pixels.

Two adjacent capacity/fallback defects were fixed in the same owner:

- the byte policy is one cross-representation retention budget, apportioned
  across scalar, complex, RGB, and windowable-RGBA pools; it is no longer
  interpreted as a target to fill independently in every representation;
- a cold, factor-misaligned crop wider than one stored page now assembles its
  already-materialized logical pages into one bounded local upload. Failure to
  use canonical resident pages is therefore not a fatal geometry error.

The all-stage real-data run on 50 tiles kept every X/Y all-dimension physical
checkpoint green on WGPU and PyQtGraph (12 checks per stage, no pixel or source
origin failures). It remains honestly red on standing performance and
window/level-continuity bars in unrelated FFT/zoom/scroll stages; the one
recorded stall was a committed-frame-stale diagnostic with no required tile
unsettled, not a page-pool exhaustion. Those bars are not claimed by this
correctness slice.

Artifacts:

- `/tmp/arrayscope-full272-noscreens.SzQoXT/result.jsonl`
- `/tmp/arrayscope-full272-planned.DmR2Vl/result.jsonl`
- `/tmp/arrayscope-physical-visual-fixed.I2rsbq/`
- `/tmp/arrayscope-profile-final.L4Mxmx/result.jsonl`
- `/tmp/arrayscope-profile-final.L4Mxmx/trace.jsonl`

## 2026-07-25 born-cropped views: warm the canonical plane, then rebind

The resident-crop rebind landed with a structural hole that only the field
workflow hit. The 2026-07-25 engagement investigation proved it: a view that is
cropped from the first frame onward uploads only crop-local pages, whose plane
identity folds their own source rect, so `tiledPayloadResident` refused every
shifted window and the seed declined with `pages_not_resident` on every tile of
every step. The rebind was inert for exactly the workflow it was built for.

The fix is a production decision, not a binding one. Anchoring is granted only
for display axes the operation chain commutes with, and
`source_anchoring_for_view` therefore strips those axes' windows before it keys
the content — the content key is already a promise about the WHOLE plane.
`canonical_plane_residency_source` redeems it: when the capability is on and a
displayed-axis crop is active, the tile's evaluation produces the window-free
plane, which rides along as `native_residency_data` and replaces the crop-local
page upload in the commit. The presented pages, texture, and window-exact level
evidence are unchanged; the widened array is residency data only.

Bindings and accounting followed:

- `_wgpu_reusable_native_texture` qualified a warm plane by whole-plane
  `source_rect` as well as shape. A cropped payload presents a sub-rect of a
  plane it fully owns, and the caller already binds the window's source origin
  into it, so the shape against `anchor.plane_shape` is the whole test. (A
  window-local `semantic_data` cannot slip through it: that matches
  `plane_shape` only when the window IS the plane.)
- capacity, upload bytes, and the commit cohort keyed the warm on
  `lod.level > 0`. A cropped level-0 payload warms too, so all three now ask
  one predicate — carries the whole plane, presents less than all of it.
- the CPU-reference oracle read `native_residency_data` as the payload's own
  values. It now takes the payload's source window out of the plane.

Budget: widening is refused when one montage of whole planes would exceed half
the tile-residency byte policy (`CANONICAL_PLANE_WARM_BUDGET_SHARE`). The warm
plane replaces the crop-local upload rather than adding to it, so the question
is the montage's whole physical working set, not one extra plane. On the field
fixture that admits 297 planes against a planned 100.

Measured, offscreen Vulkan, `(336, 336, 272)` float32 raw scalar, 100 montage
tiles, both displayed axes cropped from the first frame, scrubbing the primary
displayed dimension one row per step:

| capability | cold born-cropped fill | producers / step | ms / step |
|---|---:|---:|---:|
| off (crop-local) | 3707 ms | 100 | 2460–6548 |
| on (canonical warm) | 3698 ms | 0 | 300–365 |

First-pixel latency does not pay for the scrub's speed: the widened evaluation
replaces the crop-local one instead of adding a second pass, and measured on
this fixture a whole-plane tile evaluation costs the same 0.6 ms as its 200×200
crop (both are fixed-overhead bound, not pixel bound).

Still open, and deliberately not claimed here: an EXACT (level-0) born-cropped
montage. Its payloads carry window-local CPU semantics, so the rebind needs a
whole plane in the canonical-plane memo to re-slice, and the exact path does not
yet carry `native_residency_data` (that would mean widening
`_evaluate_native_tile_result`'s result through `RenderedTile`, and one native
plane cannot serve an RGB `image` and a value `semantic_data` at once).
Measured on a 64×64×60 fixture, 50 tiles: 50 producers and 598–630 ms per step,
declining with `no_reslicable_plane` — physically rebindable, semantically not.

## 2026-07-25 exact born-cropped views: the same plane, re-cut on the CPU

The exact path now carries that widened plane. It rides the ordinary tile result
(`EvaluationResult` → `store_montage_tile_result` → `RenderedTile` →
`_ensure_display_tile_payload`) as `native_residency_data`, so one array does
both jobs: the commit warms the window-invariant pages with it, and the memo
re-cuts this tile's window-local CPU planes out of it on the next step.

The representation split is resolved by refusal, not by carrying two planes.
The widening is granted only when the evaluation's display plane IS its value
plane — `data is semantic_data is lod_source_data`, with no separately scaled
histogram — which is the shader-display shape, and the memo re-checks the same
property on the committed payload before it re-slices
(`canonical_plane_payload_for`). A CPU-colormapped RGB `image` and a value
`semantic_data` are two different planes; those views keep their ordinary
evaluation instead of being served an approximation. A histogram plane is not
invented from a plane that never carried one.

The memo's admission moved with it. It used to pin every whole-plane payload of
the committed frame on sight; a cropped payload's carried plane is admissible
too, but the committed frame right after a document or operation change still
holds the RETIRED identity's payloads, so pinning on sight would silently
re-fill the memo that change had just released. Admission now happens inside the
rebind, at the one point where the predecessor's content key has already been
proven to be this window's own. Correctness never depended on it — the
content-key check refuses to re-slice a stale plane — but the release accounting
did.

Measured, offscreen Vulkan, `(64, 64, 60)` float32 raw scalar, 50 exact tiles,
both displayed axes cropped from the first frame, scrubbing one row per step,
fresh process per row:

| capability | cold born-cropped fill | producers / step | ms / step |
|---|---:|---:|---:|
| off (crop-local) | 1997–3107 ms | 50 | 908–2265 |
| on (canonical warm) | 2362–4787 ms | 0 | 98–212 |

Unlike the reduced path, the first fill does pay something here: the exact
window's own evaluation is still needed for the presented payload and its
window-exact level evidence, so the plane is a second pass rather than a
replacement. This harness' cold-fill variance is wide (whichever row runs first
in a process costs ~3.5 s either way), so only the direction is claimed.

Known and unclaimed: a backend without source-anchored page residency pays that
second pass and cannot use it. PyQtGraph's `payload_resident` matches on the
acknowledged tile identity and the physical source array, both of which a rebind
changes, so it declines every candidate. That is how the reduced path already
behaves with the capability on; the capability remains the WGPU resident-crop
fast path, default off.

The single-slice presentation gets this on the same terms — the seed serves both
presentations through one canonical source-plane identity — so a born-cropped
single-slice scrub now rebinds too. The test that pinned the opposite
(`test_single_slice_cold_crop_scrub_falls_back_to_evaluation`) was retired
because its premise, a born-cropped window nothing can rebind, no longer exists;
fallback remains covered by the capability-off, budget-decline, and
content-key-change tests.
