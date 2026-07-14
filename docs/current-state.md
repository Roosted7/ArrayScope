# Current state

**Snapshot:** `redesign` branch, 2026-07-14 (post R1–R7 architecture, post
R8 closure). The active queue and ground rules live in
[docs/redesign/README.md](redesign/README.md); the July course reset and its
rationale are in [docs/redesign/retro-2026-07.md](redesign/retro-2026-07.md).
`main` (6fa5c758) holds the pre-redesign state.

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

## Known broken (user-visible) and in the queue

1. **Persistent black tiles** — admission/completion/evidence scoping
   disagree about which tiles must render; a stranded tile also deadlocks
   the level-evidence pass. Dossier:
   [redesign/black-tiles-and-priority.md](redesign/black-tiles-and-priority.md).
2. **Priority rendering order** — per-tile viewport distance never reaches
   kernel ordering; three drifted rankers. Same dossier.
3. **Dead prefetch paths** — `montage_prefetch.py` silently imports the
   deleted `frame_renderer` module.
4. **Performance bars unmet** — GUI callback/heartbeat/warm-scrub bars and
   the FFT-scroll throughput target (~4 fps vs ~17 fps scalar) are restated
   in the redesign README and addressed by the P-steps after the merge;
   measurement lands first (T1: marathon benchmark harness +
   [tracing pipeline](redesign/tracing-pipeline.md)).

## Material risks

1. **Complexity debt is the top risk.** The renderer successor totals
   ~10,800 lines across six modules on one object; `FrameSession` has ~106
   fields; the same fact (residency, visibility, priority) lives in several
   owners. Every fix that doesn't reduce owner count tends to create the
   next bug.
2. **Acceptance gap.** Until the visible-truth harness exists, unit/offscreen
   green does not imply the screen is right.
3. **Branch divergence.** `redesign` is 116 commits ahead of `main`; merge
   (V4) as soon as V1–V2 land.
4. Hardware evidence remains Linux-only; the histogram adapter remains
   sensitive to private PyQtGraph API.

## What is working well

- The Qt-free semantic core (`core/`, `operations/`), operation pipeline,
  slicing, profiles, ROI, linked-window sync — untouched by the churn and
  solid.
- Kernel and ladder semantics are pinned by fast Qt-free tests (~0.5 s).
