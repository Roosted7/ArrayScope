# Current state

**Snapshot: `main`, 2026-07-16.** The GPU engine branch (`codex/gpu-engine`)
is fully merged; `main` is the single line of development again. Keep this
file a *short* snapshot — history belongs to the archives, direction to
[`queue.md`](queue.md). Update by replacement, not by layering dated
correction blocks.

## Architecture (what stands)

- One execution kernel (`arrayscope/kernel/`), one render pipeline
  (`arrayscope/render/`), one tile lifecycle machine (ADR 0051);
  orchestration on `RenderOrchestrator` over
  `frame_controller/frame_session/frame_effects/frame_runtime` (ADR 0045).
- The GPU engine (ADR 0055/0056, G1–G5 slice 1): Qt-free `arrayscope/gpu/`
  (chunk keys/grid, page table, chunk store); content-keyed chunked VisPy
  residency with window-shift/scroll-back/warm fast paths; reduced-LOD
  uniform plane-pixel pages; physical presentation truth as a standing
  audited invariant.
- Visible-truth machinery: schema-v1 trace bus, `trace_verify` invariants
  (stalls, ack churn, identity-rejected commits), the V3 stall watchdog,
  import-health and architecture guards.

## Verified behavior (real-display, 2026-07-15/16)

- ±1 window shift uploads boundary strips only; scroll-back and revisit =
  0 uploads; warm prefetch never disturbs residents; FFT-along-shifted-axis
  correctly re-uploads (negative gate).
- The 2026-07 field-defect classes — black tiles, wrong fill order, orange
  complex tiles (two causes), identity-aliasing starvation, retained-slice
  staleness, deferred-stage lost wakeups — are each closed with a
  repro-first gate and a dossier ([redesign/README.md](redesign/README.md)).

## Known open work

The ordered list with exit gates is [`queue.md`](queue.md). Headlines:
extreme-churn convergence xfail (step 1), G5 remainder (step 2), the
performance bars — FFT scroll is ~4 fps vs the ~17 fps scalar target
(step 3), then G6/renderer-protocol/G7. Standing debt: trace
`target_satisfied_retained`, framebuffer-to-CPU oracle, complex64 PyQtGraph
deadlock, ImageViewShell duplication, montage key-owner consolidation.

## Material risks

1. **Complexity debt.** The renderer successor is ~10,800 lines across six
   modules on one object; `FrameSession` has ~100+ fields; residency/
   visibility/priority facts still live in several owners. Every fix should
   reduce owner count ([ground rules](ground-rules.md) #2).
2. **Acceptance is machine-bound.** Rings 3–4 (stress, real-GL) run only on
   this machine by hand; CI is offscreen software-GL. A display-lane
   regression can merge green ([testing/README.md](testing/README.md)).
3. **Hardware evidence is Linux-only**; per-OS backend/LOD defaults await
   X5e. The histogram adapter remains sensitive to private PyQtGraph API.

## What is working well

- The Qt-free semantic core (`core/`, `operations/`), operation pipeline,
  slicing, profiles, ROI, linked-window sync.
- Suite health: ~2081 passed / 24 skipped in ~124 s, parallel by default,
  with strong doc-to-test traceability (tests cite their ADR/dossier).
