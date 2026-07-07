# Plan 04 — Preview-quality reduced display/evaluation, then exact refinement

**Status:** partially implemented and evidence-updated on 2026-07-06. Read `README.md`
ground rules first. **The active continuation is
[Plan 05](05-preview-floor-machine.md)** — the VisPy preview floor, done through the lifecycle
machine (the in-flight session-counter approach is the ADR 0051 defect class; Plan 05 §"The
root cause to fix").

Implemented so far: the preview payload quality contract, reduced-input evaluation for
`lod-commuting` tiled montage pipelines, RGB preview histogram planes for PyQtGraph re-windowing,
axis-aware reduced-input evaluator support, native-output-reduced opaque evaluator support, and
GPU/zoom-threshold regressions. The benchmark tool now separates first payload, first complete
fill, and settled frame. PyQtGraph tile-layer commits now use dynamic priority/chunking for both
cold dirty payloads and level-only re-windowing, so presentation must not fall back to all-at-once
or tile-id row/column order. The current scheduler only launches previews for pipelines that remove
work in the per-tile implementation. Shared non-display transform previews are implemented behind
`ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW`, but remain default-off: the direct shared FFT path competed
with exact work and regressed full settle in the 2026-07-06 profiles.

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
   the requested tile window; do not schedule one transform preview worker per tile. The first
   shared route is intentionally feature-gated until preview work has its own lower-priority queue
   or controller so it cannot delay exact visible fills.
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
5. Keep presentation commits on the same priority model as rendering. Cold tile-layer upserts and
   level-only re-windowing must use dynamic chunk limits and `TilePriorityContext`; sorted tile-id
   fallbacks visibly produce first-row/first-column fills and are regressions.

## Step 4 — Verification

1. Focused tests: preview payload refuses exact reads; exact supersedes preview; placeholder is not
   shown when a valid preview exists; PyQtGraph and VisPy share the semantic contract.
2. Workflow evidence: PyQtGraph resident cold raw/FFT settle no worse than native within the agreed
   tolerance, while retaining the >2x level-loop win from Plan 02.
3. Workflow evidence must distinguish request-to-first-payload, request-to-first-complete-fill, and
   request-to-settled-frame. Record whether the run is offscreen or visible and whether the machine
   is on battery.
4. GPU harness: green with `stall_repairs==0`; add/extend content assertions for zoom across a LOD
   threshold.
5. Manual check on a low-end or throttled device when available: first visible response, level drag,
   scrub/pan heartbeat, and exact ROI/hover behavior.

## Step 5 — Docs + handoff

1. ADR 0050: move "Reduce-before-ops and preview-then-refine" from design note to implemented
   status with before/after numbers.
2. `docs/current-state.md`: keep only the high-level state and link to the roadmap/ADR for detail.
3. `docs/roadmap.md`: advance the X5 active queue.
4. Record conclusions, found issues, and technical debt in the live docs. Append external memory
   notes only when that is explicitly requested for the session.

## 2026-07-06 conclusions

- The initial preview-quality payload contract is usable: exact consumers refuse preview payloads,
  RGB preview floors carry display histogram planes, and native exact refinement supersedes through
  backend acknowledgement.
- `ARRAYSCOPE_SHARED_TRANSFORM_PREVIEW=1` proves the shared non-display route can be wired, but it is
  not a default-quality scheduler path. On battery/offscreen and visible Wayland runs, shared FFT
  preview work competed with exact tiles and made full settle slower. The next design should give
  transform previews a separate lower-priority preview queue or controller, similar in spirit to the
  existing level/histogram side lanes.
- First-complete-fill is the honest perceptual milestone for large tiled scenes. Settled-frame timing
  remains necessary for correctness, but it hides whether the user saw a progressively filling image.
  The workflow profiler now reports request-to-first-payload, payload-to-fill, and
  first-visible-tile-to-full-visible-fill separately.
- PyQtGraph tile-layer presentation must be budgeted even when the change is a cold dirty payload, not
  only when a stale level target exists. Without that limit a complex/FFT montage can appear in one
  large commit.
- Level-only presentation order is part of the priority system. Appending `sorted(tile_id)` level
  updates caused visible first-row/first-column refinement; backend continuation order now follows
  the session delta priority order.
- Scalar PyQtGraph first fill looked slower than FFT because progressive tile commits rebuilt
  aggregate histogram samples in the tile-presentation hot path. Deferring aggregate histogram plot
  sample construction after the first display commit restored the expected visible-fill order in
  `tests/artifacts/x5-item1-pyqtgraph-visible-hist-first-only.jsonl`: scalar visible fill ~0.71 s,
  FFT visible fill ~1.12 s on visible Wayland.
- Feedback-governor diagnostics now include branch details and per-batch renderer-stage details
  (`payload`, `prepare`, `apply`, `ack`, `geometry`, `overlay`, etc.) so future oscillations can be
  attributed to the actual hot-path stage instead of guessed from total elapsed time.
- The pacing unification in "Stabilize PyQtGraph LOD tile pacing" silently dropped the
  resident-retarget bypass: routing everything through `TileAdmissionQueue.admit` made
  `max_items` cap zero-cost retargets too (`test_resident_retarget_upserts_bypass_cold_priority_cap`
  regressed while the commit's own new pacing test asserted the opposite). Resolved 2026-07-06 as
  backend truth: remaps never charge the byte budget, and whether they bypass the ITEM cap is
  declared by the caller — instant on persistent GPU-residency backends (`free_fn`, VisPy atlas
  remap, default), paced in priority order where re-level rebuilds items
  (`pace_resident_retargets=True`, the PyQtGraph tile-layer limits). Lesson: when a bypass is
  replaced by a unified path, the invariant it enforced must move into the unified path's
  contract explicitly — and two tests asserting opposite semantics must name whose backend
  truth they pin.
