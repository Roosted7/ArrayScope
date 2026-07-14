# Current state

**Snapshot:** `main`, 2026-07-14, after linear redesign/review rebase.
**[Codex 2026-07-14 — history correction]** No redesign integration merge
remains; completed queue and review commits are a single linear sequence.
**[Codex 2026-07-14 — post-V4 update]** The redesign execution record and
performance rules remain in [docs/redesign/README.md](redesign/README.md);
the live product ordering has returned to [docs/roadmap.md](roadmap.md).

## Architecture (what the rewrite delivered, R1–R7)

- One execution kernel (`arrayscope/kernel/`): priorities, dependencies,
  staleness, one worker→GUI bridge. WorkGraph and the eight FIFO pools are
  gone.
- One render pipeline (`arrayscope/render/`): typed stage contracts,
  `LodLadder` planner, kernel-backed `FramePipeline`; a normal image is a
  one-region plan, a montage a multi-region plan.
- One tile state machine (`presentation/tile_lifecycle.py`, ADR 0051).
- `frame_renderer.py` is deleted; orchestration lives on
  `RenderOrchestrator` (`window/render.py`) over
  `frame_controller/frame_session/frame_effects/frame_runtime`.

## Viewer-truth fixes now on `main`

1. Required-tile admission, completion, evidence, and presentation consume
   the canonical `FrameSession.required_tile_numbers()` owner; the V1
   one-index/boundary scenario has correct pixels on both backends.
2. One canonical tile rank reaches kernel execution and progressive cold
   presentation; the V2 disjoint-scroll scenario paints center-out.
3. Dead prefetch imports and their silent broad-exception fallbacks are gone.
4. A stranded required tile is no longer silent: after two idle seconds it
   emits a `stall` owner-chain event, writes the bounded trace, and shows a
   persistent diagnostic.
5. Acknowledged montage extent changes replay AUTO/FIT camera intent against
   successor geometry; backend bounds changes no longer become USER input,
   and an actual USER camera is preserved.

## Known open work

1. **Performance bars remain unmet.** The frozen T1 baseline exceeds the
   50 ms callback, 16 ms heartbeat, and 15 ms warm-input commitments; FFT
   scroll remains the primary throughput target. Follow the measured
   P-program in the roadmap.
2. **The broad suite is red.** The final pre-integration non-GPU run reported
   42 failures and 2 teardown errors; 41 failures reproduced serially.
   Stale deleted-owner assertions coexist with real coalescer, levels,
   viewport/ROI, cache-rebind, and transition behavior debt.
3. Hardware evidence remains Linux-only; the histogram adapter remains
   sensitive to private PyQtGraph API.

## Material risks

1. **Complexity debt is the top risk.** The renderer successor totals
   ~10,800 lines across six modules on one object; `FrameSession` has ~106
   fields; the same fact (residency, visibility, priority) lives in several
   owners. Every fix that doesn't reduce owner count tends to create the
   next bug.
2. **Suite/acceptance split.** The visible-truth harness now exists and is
   authoritative for pixels, but the broad offscreen suite must be brought
   back to truth without reintroducing superseded ownership models.
3. **Performance work can regress truth.** Every P-step therefore needs
   before/after measurements plus the real-display pixel/trace gates.

## What is working well

- The Qt-free semantic core (`core/`, `operations/`), operation pipeline,
  slicing, profiles, ROI, linked-window sync — untouched by the churn and
  solid.
- Kernel and ladder semantics are pinned by fast Qt-free tests (~0.5 s).
