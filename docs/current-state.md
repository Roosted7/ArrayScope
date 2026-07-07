# Current state

**Snapshot:** `redesign` branch, 2026-07-07. ADR 0053 accepted; R1 is
landed, so app background execution now runs through one kernel plus one
Qt bridge. The modular-pipeline/LOD-ladder nucleus is landed; plans R2–R5
([docs/redesign/](redesign/README.md)) dissolve the remaining legacy
orchestration. `main` (6fa5c758) holds the pre-redesign state described in
this file's git history.

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
| Modular pipeline | **Nucleus** | `arrayscope/render/`: typed stage contracts, MontagePipeline scheduling skeleton with one-place supersession; effects (evaluation/commit) are R2 ports. |
| Unified LOD ladder | **New, tested** | `render/ladder.py`: FLOOR→PREVIEW→DESIRED→EXACT pure planner; replaces four scattered decision sites at R3. |
| Tile lifecycle | Unchanged owner | ADR 0051 machine stays; pipeline feeds it events instead of frame_renderer. |
| Legacy orchestration | **Dissolving** | WorkGraph is deleted and `window/evaluation_controller.py` is import-only. `frame_renderer`/`montage_lod` remain R2/R3 deletion targets with a [method-by-method map](redesign/frame-renderer-map.md). |
| Vocabulary | Canonical in kernel | `WorkLane`/`EvalPriority` are compat aliases of kernel `Lane`/`Priority`. |
| Hygiene | Done (first pass) | Kill switches, P3 fallbacks, tmp_probes deleted; 3 probes live in `tools/probes/`. |
| Baseline health | Green, with known slow wedge | R1 validation: `pytest tests -q -n 16 --ignore=tests/gpu_interaction` → 1696 passed, 3 skipped; GPU harness → 6 passed. The FFT transform-preview wedge is pre-existing and tracked in [known-red](redesign/known-red.md) for R2/R3. |

## What is working well

- The Qt-free semantic core (`core/`, `operations/`), the lifecycle
  machine, the frame planner, and the surface contract tests — the
  redesign builds on them, none needed structural change.
- Kernel semantics are pinned by fast Qt-free tests (0.5 s), giving the
  R1–R3 ports a hard behavioral floor.

## Material risks

1. **R2 is the big port** (evaluation + commit effects). Mitigations:
   golden-output tests before deletion, cluster-per-commit, benchmark
   bars in every exit gate.
2. **Hardware evidence still Linux-only** (X5c–X5e untouched by the
   redesign; Windows/macOS traces owed).
3. **Histogram adapter remains version-sensitive** to private PyQtGraph
   API (unchanged).
4. **Dual paths during R2** are allowed in exactly one place (level stats,
   until R3) — watch it doesn't grow.

## Current direction

Execute R2→R5 in order (`docs/redesign/README.md` owns the queue), then
return to the X5 evidence gates in `docs/roadmap.md`.
