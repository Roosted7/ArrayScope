# Current state

**Snapshot:** ArrayScope v32 composition line, reviewed 2026-07-02. The v30/X1–X4
control-plane work is preserved; v32 moved render orchestration off the window
into a composed `RenderOrchestrator`, fixed the crash class that motivated it,
and deleted the orphaned pre-tiles decision layer. Evidence:
[v32 composition audit](reviews/v32-composition-audit.md),
[ADR 0045](decisions/0045-render-orchestrator-composition.md).

## Maturity map

| Area | State | Notes |
|---|---|---|
| Basic launch, slicing, image/line display | Established | Broad automated coverage; real-hardware checks still owed after v32. |
| Dimension roles, ranges, flips/FFT shift | Established | |
| Reversible operation document/recipes | Established | Optimizer preserves public step history. |
| Region planning, stage cache, cost/memory estimates | Substantial | Qt-free and well covered. |
| Profiles and ROI inspection | Substantial | Shared pointer capture owns ROI/profile semantics on both backends. |
| Histogram and window/level | Substantial | PyQtGraph binding isolated in an adapter; still version-sensitive. |
| Frame planning and tiled presentation | Established | One `FramePlanner`/typed tiled path for single images, large planes, montages. v32: the applied `MontagePlan` is the layout source of truth. |
| Render orchestration | Restructured (v32) | One `RenderOrchestrator` owns render state; window keeps semantic state + thin API. Internal split of the orchestrator and token unification are the next gates (Y1). |
| Progressive montage | Advanced | Control-plane models extracted and exercised. |
| PyQtGraph backend | Production default | Bounded CPU/item convergence; large item counts remain costly. |
| VisPy backend | Experimental | Promising; real-hardware evidence is the X5 gate. Unstable under software GL (Xvfb/llvmpipe) — do not treat CI GL runs as evidence. |
| LOD | Native-only policy | Desired vs applied factor reported separately; multi-resolution waits on ADR 0041/X5. |
| Diagnostics/benchmarks | Good | Work-graph counters, JSONL, benchmark records; stage-warmup reporting removed with the dead path. |
| Test suite | Repaired (v32) | Host-independent (pinned memory snapshot), no `sys.modules` replacement, drifted/removed-path tests rewritten or deleted; fakes model the composition. |
| Documentation/ADRs | Updated for v32 | Roadmap gates renumbered: Y1–Y3 structural, X5 hardware evidence. |

## What is working well

- The Qt-free semantic core (`core/`, `operations/`) is cleanly layered and
  well tested; nothing there needed structural change in the v32 audit.
- The extracted control-plane models (`FramePlanner`, `WorkGraph`,
  `TileAdmissionQueue`, `PresentationGenerationTracker`,
  `LevelConvergenceStrategy`, `StageFanInState`) are the right shape; v32
  built on them rather than replacing them.
- Render state now has a single owner. Timer lifetime is structural
  (orchestrator is a `QObject` child of the window; deferred callbacks carry a
  receiver context), which removed an entire crash class at teardown/close.
- The tiled pipeline is the only presentation path; its memory protection
  (montage/tile budgets, skip warnings) replaced the old refuse/degrade
  decisions and is covered by tests.

## Material risks

1. **Parallel token schemes remain (Y1).** The orchestrator still contains
   several revision counters and staleness patterns that predate `WorkGraph`.
   They are now in one namespace, which makes the unification tractable, but
   until Y1 lands a fix can still pick the wrong guard.
2. **Backend duplication (Y2).** ~1,200 lines are implemented twice across
   the PyQtGraph and VisPy view classes; divergence between the two `tiles.py`
   files is a standing source of "works on one backend" bugs.
3. **Manual UI sync (Y3).** 17 `_sync_*` methods mirror `ViewState` into ~50
   widgets; every new control is a chance to miss one path.
4. **Profiling tools drift (Y3).** `tools/profile_montage_workflow.py`
   re-implements window composition; its numbers can silently diverge from the
   product.
5. **Hardware evidence is still absent (X5).** Nothing measured in this
   repository under Xvfb/software GL says anything about real GPU latency,
   texture limits, Wayland, or interaction feel. VisPy claims remain
   unproven either way.
6. **Histogram adapter remains version-sensitive** to private PyQtGraph API.

## Current direction

Keep the semantic core and control-plane models. Finish ownership: Y1 (one
generation contract, admission only through `WorkGraph`), Y2 (backend
de-duplication against `ImageSurface`), Y3 (declarative UI binding, tools on
production composition, one cache core). Only then spend effort on X5 hardware
evidence, which gates LOD and any backend-default change. Ordered gates and
exit criteria are in the [roadmap](roadmap.md).
