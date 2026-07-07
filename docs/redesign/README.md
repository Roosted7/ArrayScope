# Redesign plans — kernel, pipeline, LOD ladder

**Date:** 2026-07-07. **Branch:** `redesign` (worktree `.worktrees/redesign`).
**Decision record:** [ADR 0053](../decisions/0053-execution-kernel-and-modular-pipeline.md).
**Source of truth for priority:** this README's table; `docs/roadmap.md`
defers to it while the redesign is in flight.

## What already landed on `redesign`

- `arrayscope/kernel/` — the execution kernel (scheduler, workers,
  completions, Qt bridge), 32 tests. Priorities/dependencies/staleness are
  now real at execution level.
- `arrayscope/render/` — typed stage contracts, the unified LOD ladder
  (pure planner, 11 tests), and the kernel-backed `MontagePipeline`
  skeleton (5 tests) with `PipelineEffects` integration points.
- Lane/priority vocabulary canonicalized in the kernel (legacy modules are
  compat aliases).
- Plan-time viewport content extent (fixes the fit-unlock regression,
  bisected to 2995d039).
- Hygiene deletions: kill switches, ADR 0051 P3 fallbacks, `tmp_probes/`
  (3 keepers now in `tools/probes/`).

## Plan queue (execute in order)

| Plan | What | Size | Blocked by |
|---|---|---|---|
| [R1](r1-kernel-adoption.md) | All background execution on the kernel; delete the 8 controllers + WorkGraph | L | — |
| [R2](r2-pipeline-integration.md) | MontagePipeline live: port evaluation/commit effects, dissolve frame_renderer clusters B/C/E | XL | R1 |
| [R3](r3-lod-ladder-adoption.md) | Ladder replaces montage_lod planning; one pyramid store; ops once per rung; PyQtGraph parity via capabilities | L | R2 |
| [R4](r4-timer-and-governor-audit.md) | Every QTimer justified or deleted; governor shrinks to telemetry + two knobs | M | R2 |
| [R5](r5-test-and-docs-truth-pass.md) | Delete wrong-path tests; docs/current-state truth pass; known-red ledger emptied | M | R2–R4 |

A plan is done only when its exit gate passes and the source code it
replaces is DELETED. "Both paths work" is not done — old remnants subtly
guide us back to the wrong path.

## Ground rules (read before ANY plan)

1. **Thomas's bar:** the GUI event loop never hangs noticeably. Interaction
   outranks all speculation. Any synchronous GUI-thread step >50 ms is a
   bug; pan/scrub heartbeat max gap target ~16 ms.
2. **GUI thread is a gateway** (ADR 0053): submit, drain, apply bounded
   commits, update widgets. Evaluation/reduction/stats/planning are kernel
   tasks. New timers are forbidden except anti-hang fallbacks or UI
   cosmetics — justify in a comment or don't merge.
3. **One owner per state.** Tile state: `TileLifecycle`. Quality
   progression: `LodLadder`. Execution + staleness: the kernel. If you need
   a new collection, you are probably duplicating one of them — stop.
4. **Backends branch on capabilities, never names.** Both backends stay
   first-class; VisPy exploits GPU residency/uniforms, PyQtGraph gets
   bounded CPU equivalents.
5. **Test bar:** Qt-free suites (core, operations, presentation, kernel,
   render) green at every commit. window/display breakage allowed ONLY with
   a [known-red.md](known-red.md) entry naming the fixing/deleting step.
6. **Delete aggressively, with the replacement.** Port commits delete the
   old methods AND their pacing tests. Never leave a compatibility shim.
7. **One cluster per commit; measure after each.** Commit messages: what +
   why + numbers (suite counts, before/after timings).

## Environment & commands (unchanged mechanics, new locations)

- Python: `~/miniconda3/envs/arrayscope/bin/python` (host conda env; the
  Cowork sandbox cannot load PyQt6 — GUI/GPU work runs on the host, e.g.
  via Desktop Commander).
- **Full suite** (~35 s, parallel):
  `cd ~/projects/ArrayScope/.worktrees/redesign && ~/miniconda3/envs/arrayscope/bin/python -m pytest tests -q -n 16 --ignore=tests/gpu_interaction`
  Known parallel-only flakes (pass alone; ignore unless newly consistent):
  `test_selecting_fft_workers_updates_settings`,
  `test_resource_governor_applies_worker_and_callback_limits`,
  `test_compute_policy_configures_stage_and_montage_lanes`,
  teardown of `test_montage_ready_display_payloads_commit_immediately`.
- **Kernel/render suites only** (fast TDD loop, no Qt):
  `… -m pytest tests/kernel tests/render tests/presentation -q -n 0`
- **GPU harness** (~20 s, real hardware, asserts `stall_repairs==0`):
  `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland … -m pytest tests/gpu_interaction -n 0`
- **Workflow benchmark:**
  `… -m arrayscope.tools.profile_montage_workflow --backend {vispy|pyqtgraph} --montage-lod-policy {resident|native-only} [--json-out FILE]`
  `STALL WATCHDOG` on stderr must stay 0.
- **Probes:** `tools/probes/` (`verify_scrub_fastpath.py`,
  `profile_cached_rebuild.py` — ONSCREEN, tell Thomas hands-off,
  `verify_stale.py`).
- **No fixed sleeps for waiting.** Foreground `start_process` + blocking
  `read_process_output`; sentinel-line waits for >2 min jobs; >3× historical
  runtime = hung.
- **PROBE TRAP:** `with_montage_axis(axis, text=...)` does NOT set the index
  window — pass `indices=range(...)`.

## Debugging gotchas (carried over, still true)

- `print()` in app code is swallowed under pytest/Qt — append to a /tmp
  file. py-spy can't keep up with 3.14 — use cProfile or JSONL diagnostics.
- JSONL wedge evidence lives in the STATIC TAIL of the file.
- `pkill -f "pytest tests"` in a DC shell kills the shell itself.
- VisPy offscreen `canvas.render()` needs int-rounded `physical_size`;
  `QTimer.singleShot` needs the 3-arg receiver form.
- Committing from a Cowork sandbox: delete stale `*.lock` under `.git`
  first.

## Definition of done (every plan)

1. Suites per rule 5; GPU harness green incl. `stall_repairs==0`.
2. The replaced code is deleted; grep proves no references remain.
3. Numbers in the commit message; ADR 0053 status table updated.
4. Claude memory updated (`arrayscope-lod-residency` note: queue position,
   tip, new gotchas).
