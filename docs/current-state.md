# Current state

**Snapshot:** `redesign` branch, 2026-07 (post R2 stabilization). ADR 0053
accepted; R1 landed; **R2 is code-complete but NOT closed** — its first
completion attempt shipped four architectural defects (per-completion
commit storm, camera-key churn, deps-as-ordering, per-tile-native FFT
floors) plus symptom patches; those are reverted/fixed (a23fb2b2,
4464a6e4) and the R2 exit gate (benchmark bars) must now be re-measured.
App background execution runs through one kernel plus one Qt bridge; the
montage data path runs through the render pipeline with an
admission/presentation split (bounded commit per event-loop turn behind a
coalescing gate). R2b work items and the red-test ledger live in
[known-red.md](redesign/known-red.md); plans R3–R5
([docs/redesign/](redesign/README.md)) dissolve the remaining LOD and
docs/test debt. R4's timer/governor implementation pass has landed, with
manual idle/scrub/benchmark evidence still owed before the gate closes.
`main` (6fa5c758) holds the pre-redesign state described in this file's git
history.

## Why the redesign (one paragraph)

Scheduling was split across five cooperating systems (WorkGraph
bookkeeping, eight FIFO QThreadPools, governor timers, dispatch derivation,
~28 pacing timers), most fan-in ran on the GUI thread in budgeted slices,
and `FrameRenderMixin` had grown to 6,100 lines/150 methods. Priorities and
dependencies were decorative at execution level. Regressions ping-ponged
because staleness, pacing, and ownership were re-decided at every call
site. ADR 0053 records the decisions: own Dask-inspired kernel, threads
sized to cores behind a swappable backend, both display backends
first-class via capabilities, core-green test bar with a known-red ledger.

## Maturity map (redesign-relevant rows; others unchanged from main)

| Area | State | Notes |
|---|---|---|
| Execution kernel | **Driving the app** | `arrayscope/kernel/`: real priorities/deps/staleness, lane quotas, one GUI fan-in bridge; R1 routes the former controller submissions through this scheduler. |
| Modular pipeline | **Driving montage** | `arrayscope/render/`: typed stage contracts, kernel-backed MontagePipeline, Qt-free evaluation effects, tile-state snapshots, stage deps, and commit effects route the live montage render path. |
| Unified LOD ladder | **Adopted in montage** | `render/ladder.py`: FLOOR→PREVIEW→DESIRED→EXACT pure planner now feeds montage through the pipeline; demanded-level convergence wakes via lifecycle/pipeline completions instead of a side request list. |
| Tile lifecycle | Unchanged owner | ADR 0051 machine stays; the pipeline feeds it rung and backend-ack events instead of frame_renderer result pumps. |
| Legacy orchestration | **Dissolving** | WorkGraph is deleted and `window/evaluation_controller.py` is import-only. `frame_renderer.py` is below the R2 gate with clusters B/C/E moved out or deleted; `window/montage_lod.py` is deleted, and level-stat maintenance now lives under `render/level_stats.py` with refinement admitted through the kernel. ADR 0054 records the rough-preview / rough-target / refined level-evidence ordering. |
| Vocabulary | Canonical in kernel | `WorkLane`/`EvalPriority` are compat aliases of kernel `Lane`/`Priority`. |
| Hygiene | Done (first pass) | Kill switches, P3 fallbacks, tmp_probes deleted; 3 probes live in `tools/probes/`. |
| Baseline health | Focused R4/core green; broad offscreen reds resolved | Core/operations, R4-focused architecture/governor/kernel/render/window tests, and the formerly red offscreen display/window/UI sentinels are green after the preview-to-target wakeup fix. Benchmark/manual/GPU evidence still needs re-measure before closing the redesign gate. |

## What is working well

- The Qt-free semantic core (`core/`, `operations/`), the lifecycle
  machine, the frame planner, and the surface contract tests — the
  redesign builds on them, none needed structural change.
- Kernel semantics are pinned by fast Qt-free tests (0.5 s), giving the
  R1–R3 ports a hard behavioral floor.

## Material risks

1. **R2 closes only on re-measured benchmark bars** (ground rule 8: gates
   are immutable). The pre-fix JSONLs are failure evidence, not baselines.
2. **Window-level/histogram correctness remains a gate.** The rough-preview /
   rough-target / refined evidence ordering is implemented and covered by
   focused tests, but R2 still needs the fresh benchmark/manual/GPU evidence
   before closure.
3. **Hardware evidence still Linux-only** (X5c–X5e untouched by the
   redesign; Windows/macOS traces owed).
4. **Histogram adapter remains version-sensitive** to private PyQtGraph
   API (unchanged).
5. **One temporary dual path remains**: montage level stats still use the
   existing bounded timer path until R3's LevelStatsService.

## Current direction

Finish R2b evidence collection (overlay/ROI timing, diagnostics formatter,
complex windowing seam, benchmark re-measure against the unchanged R2 gate),
then R3→R5 in order (`docs/redesign/README.md` owns the queue), then the X5
evidence gates in `docs/roadmap.md`.
