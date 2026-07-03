# Current state

**Snapshot:** ArrayScope v33/Y1–Y3 plus v34 review adjustments, 2026-07-02.
The v30/X1–X4 control-plane work is preserved; v32 moved render orchestration
off the window into a composed `RenderOrchestrator` and fixed the crash class
that motivated it; Y1–Y3 then unified the staleness vocabulary, the shared
backend shell logic, UI state mirroring, and cache eviction. The v34 review
keeps that route, fixes low-level boundedness/copying issues in retained tiled
payloads and VisPy warm residency, and reframes X5 as an evidence-first
physical strategy gate. Evidence: [v32 composition audit](reviews/v32-composition-audit.md),
[ADR 0045](decisions/0045-render-orchestrator-composition.md),
[ADR 0046](decisions/0046-evidence-first-performance-strategy.md), and the
Y1–Y3/X5 entries in the [roadmap](roadmap.md).

## Maturity map

| Area | State | Notes |
|---|---|---|
| Basic launch, slicing, image/line display | Established | Broad automated coverage; real-hardware checks still owed after v32. |
| Dimension roles, ranges, flips/FFT shift | Established | |
| Reversible operation document/recipes | Established | Optimizer preserves public step history. |
| Region planning, stage cache, cost/memory estimates | Substantial | Qt-free and well covered. |
| Profiles and ROI inspection | Substantial | Shared pointer capture owns ROI/profile semantics on both backends. |
| Histogram and window/level | Substantial | Shared preview driver in the shell; PyQtGraph binding isolated in an adapter, still version-sensitive. |
| Frame planning and tiled presentation | Established | One `FramePlanner`/typed tiled path for single images, large planes, montages. The applied `MontagePlan` is the layout source of truth. |
| Render orchestration | Consolidated (Y1) | One `RenderOrchestrator` owns render state; `window/render_contract.py` owns staleness/tokens; admission decisions are observable in `WorkGraph` counters. |
| Progressive montage | Advanced | Control-plane models extracted and exercised. |
| Backend shell sharing | Consolidated (Y2) | Shared semantic drivers live in `ImageViewShell` behind small hooks; one tile-stats contract; surface parity tests run on both backends. |
| UI state sync | Declarative (Y3) | `ViewStateBinder` mirrors `ViewState` into widgets; one registration per control; guarded by architecture tests. |
| Caches | Unified (Y3) | `core/bounded_cache.py` is the one eviction/priority implementation under the display, stage, and payload caches. Retained tiled payloads are bounded before large inserts in the v34 review branch. |
| PyQtGraph backend | Fallback default (ADR 0047) | Bounded CPU/item convergence; large item counts and level re-window drains remain costly (X5a fixed starvation: 272-tile level drag never converged, now ~4.3 s). Selected by `auto` wherever hardware GL is absent or traces are missing. |
| VisPy backend | Auto-selected on Linux hardware GL (ADR 0047) | X5a Linux traces: first frame faster in every scenario (1.4–13×); level changes are uniform-only (272-tile level drag ~0.26 s vs ~8 s). Still unstable under software GL (Xvfb/llvmpipe) — do not treat CI GL runs as evidence. |
| LOD | Native-only policy | Desired vs applied factor reported separately; multi-resolution waits on ADR 0041/X5. |
| Diagnostics/benchmarks | Good | Work-graph counters, JSONL, benchmark records; profilers drive the production window composition. |
| Test suite | Repaired (v32) + contract coverage (Y2) | Host-independent, no `sys.modules` replacement; `test_imagesurface_contract.py` pins cross-backend semantics; architecture guards pin the Y1/Y3 invariants. |
| Documentation/ADRs | Updated through ADR 0047 | Roadmap gates Y1–Y3 recorded as done; X5a Linux traces published in `reviews/x5a-hardware-telemetry-linux-wayland.md`; X5b–X5e remain open. |

## What is working well

- The Qt-free semantic core (`core/`, `operations/`) is cleanly layered and
  well tested; nothing there needed structural change in the v32 audit.
- The extracted control-plane models (`FramePlanner`, `WorkGraph`,
  `TileAdmissionQueue`, `PresentationGenerationTracker`,
  `LevelConvergenceStrategy`, `StageFanInState`) are the right shape; v32
  built on them rather than replacing them.
- Render state has a single owner and a single staleness vocabulary
  (`render_contract`). Timer lifetime is structural: every deferred callback
  carries a receiver context, enforced by an architecture guard.
- The tiled pipeline is the only semantic presentation path; its memory
  protection (montage/tile budgets, skip warnings) replaced the old
  refuse/degrade decisions and is covered by tests. Physical storage remains a
  strategy decision below the shared surface.
- Cross-backend semantics are pinned by contract tests; the Y2 pass found and
  fixed two real one-backend-only forks (close-path interaction cancel, hidden
  presentation mode).

## Material risks

1. **Hardware evidence is still absent (X5).** Nothing measured in this
   repository under Xvfb/software GL says anything about real GPU latency,
   texture limits, Wayland, or interaction feel. VisPy claims remain
   unproven either way.
2. **Histogram adapter remains version-sensitive** to private PyQtGraph API.
3. **Physical upload paths are still per-backend** (`setImage`,
   `setTiledPresentation`, `setupUI`, the two `tiles.py`). That is by design
   (CPU items vs. GPU atlas), but changes there are only covered by the
   contract tests at the semantic level, not mechanically.
4. **The composed `RenderOrchestrator` is still too large for comfortable
   performance debugging.** The ownership boundary is correct, but future
   splits should follow measured workflows (residency, presentation commit,
   prefetch/resource policy), not another broad mechanical rewrite.
5. **Semantic tiled unification can become a physical pessimization** if small
   frames are forced through the same atlas/quad machinery as huge scenes. X5
   must measure singleton/direct physical strategies under the same semantic
   contract.
6. **Real-hardware regression checks after the Y1–Y3 refactors are owed**;
   the automated suites all run offscreen.

## Current direction

The structural gates are done enough to stop broad refactoring. Current work is
X5: base GPU, backend-default, singleton/direct fast-path, viewport-residency,
and multi-resolution decisions on measured device behavior, not headless runs.
Ordered gates and exit criteria are in the [roadmap](roadmap.md).
