# Plan 04 — Preview-quality reduced display/evaluation, then exact refinement

**Status:** partially implemented on 2026-07-06. Read `README.md` ground rules first.

Implemented so far: the preview payload quality contract, reduced-input evaluation for
`lod-commuting` tiled montage pipelines, RGB preview histogram planes for PyQtGraph re-windowing,
axis-aware reduced-input evaluator support, native-output-reduced opaque evaluator support, and
GPU/zoom-threshold regressions. The current scheduler only launches previews for pipelines that
remove work in the per-tile implementation. Non-display transform previews are semantically valid
but must be shared/batched across the requested tile window before scheduler enablement; the direct
per-tile route duplicated FFT work and did not present before exact payloads in the 2026-07-06
profile.

## Background

Plan 02 proved that PyQtGraph can benefit from resident display LOD once level changes consume
display-sized payloads: the FFT level loop improved from 7193 ms to 3012 ms. It also proved the
remaining blocker: cold raw/FFT settle still regresses because the current path evaluates native
display pixels first and then reduces them. That adds worker work instead of replacing work.

ADR 0050 already names the required contract: reduced-input display evaluation must be a
`quality="preview"` payload with exact semantic planes explicitly absent, followed by the native
`quality="exact"` result through ordinary supersession. Do not wire reduced-input evaluation
without that preview-then-refine contract.

## Goal

Make first presentation fast and honest on large tiled scenes, especially on PyQtGraph and low-end
devices:

- present a reduced display payload instead of black tiles/placeholders when exact work would miss
  the interaction budget;
- evaluate `lod-commuting` display pipelines on reduced input where that removes work;
- keep `lod-opaque` pipelines exact for the transform, then reduce their output for presentation;
- stream native exact payloads through the normal lifecycle as soon as they are ready;
- keep hover, ROI, profiles, level statistics, and exports on exact/native sources only.

## Step 0 — Measure the current split

1. Re-run the Plan 02 PyQtGraph native/resident workflow once per arm to confirm the current
   baseline still matches the recorded medians closely enough.
2. cProfile the cold raw and FFT settle paths with resident LOD enabled. Attribute time to:
   native display evaluation, display reduction, histogram/display-stat preparation, stage/op work,
   payload construction, and backend presentation.
3. Record whether each hot pipeline is `lod-commuting`, `lod-transforming`, or `lod-opaque`.

## Step 1 — Add the payload-quality contract

1. Represent display payload quality explicitly: preview payloads draw pixels but have no exact
   semantic planes, while exact payloads carry native semantic data as today.
2. Make exact-value consumers refuse preview payloads deterministically and request/await the exact
   source through existing region-limited evaluation paths.
3. Add lifecycle and diagnostics visibility: preview-presented count, preview superseded by exact,
   exact wait/refusal counters, and applied preview level.

## Step 2 — Wire reduced display evaluation only where it removes work

1. For `lod-commuting` display pipelines, reduce input first and evaluate the display result at the
   demanded preview level.
2. For `lod-transforming` pipelines, map the demanded level through the region planner before
   display evaluation. Non-display transform support must fan out one shared reduced transform to
   the requested tile window; do not schedule one transform preview worker per tile.
3. For `lod-opaque` pipelines, keep native transform semantics and reduce the output for display.
   This is an evaluator fallback until there is evidence that scheduling it beats placeholders
   without duplicating exact work.
4. Do not reduce exact semantic planes. A preview tile must never masquerade as exact inspection
   data.

## Step 3 — Presentation and scheduling

1. Prefer an already-resident preview over a placeholder when exact work is not ready.
2. Schedule exact refinement immediately after or in parallel with preview work, subject to the
   existing visible/interaction budgets.
3. Supersede preview with exact only through backend-acknowledged commits. A failed exact commit
   must leave the preview visible.
4. Keep speculative preview work bounded by `WorkGraph` lanes; do not add a new idle scheduler.

## Step 4 — Verification

1. Focused tests: preview payload refuses exact reads; exact supersedes preview; placeholder is not
   shown when a valid preview exists; PyQtGraph and VisPy share the semantic contract.
2. Workflow evidence: PyQtGraph resident cold raw/FFT settle no worse than native within the agreed
   tolerance, while retaining the >2x level-loop win from Plan 02.
3. GPU harness: green with `stall_repairs==0`; add/extend content assertions for zoom across a LOD
   threshold.
4. Manual check on a low-end or throttled device when available: first visible response, level drag,
   scrub/pan heartbeat, and exact ROI/hover behavior.

## Step 5 — Docs + memory

1. ADR 0050: move "Reduce-before-ops and preview-then-refine" from design note to implemented
   status with before/after numbers.
2. `docs/current-state.md`: keep only the high-level state and link to the roadmap/ADR for detail.
3. `docs/roadmap.md`: advance the X5 active queue.
4. Append to the Claude `arrayscope-lod-residency` memory; do not delete historical notes.
