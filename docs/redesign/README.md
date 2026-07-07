# Redesign plans — kernel, pipeline, LOD ladder

**Date:** 2026-07-07. **Branch:** `redesign` (worktree `.worktrees/redesign`).
**Decision record:** [ADR 0053](../decisions/0053-execution-kernel-and-modular-pipeline.md).
**Source of truth for priority:** this README's table; `docs/roadmap.md`
defers to it while the redesign is in flight.

## What already landed on `redesign`

- `arrayscope/kernel/` — the execution kernel (scheduler, workers,
  completions, Qt bridge), 33 tests. Priorities/dependencies/staleness are
  now real at execution level.
- **R1 kernel adoption** — app submissions now share one `Kernel` and one
  `QtKernelBridge`; WorkGraph is deleted; `window/evaluation_controller.py`
  is an import-only surface over `arrayscope/kernel/eval_adapter.py`.
  Validation: 1696 passed / 3 skipped in the full non-GPU suite at `-n 16`;
  GPU harness 6 passed. The vispy/resident FFT preview wedge reproduced and
  remains in [known-red.md](known-red.md) for R2/R3.
- `arrayscope/render/` — typed stage contracts, the unified LOD ladder
  (pure planner), and the kernel-backed `MontagePipeline` now driving the
  live montage path through evaluation effects, stage dependencies, commit
  batches, lifecycle acknowledgement, and backend presentation.
- **R2 pipeline integration** — `frame_renderer.py` is 1,888 lines; clusters
  B/C/E are deleted or extracted, stage fan-in is kernel-dependency based,
  the watchdog is diagnostics-only (`stall_assertions`), and the FFT
  transform-preview GPU harness scenario is covered.
- Lane/priority vocabulary canonicalized in the kernel (legacy modules are
  compat aliases).
- Plan-time viewport content extent (fixes the fit-unlock regression,
  bisected to 2995d039).
- Hygiene deletions: kill switches, ADR 0051 P3 fallbacks, `tmp_probes/`
  (3 keepers now in `tools/probes/`).

## Plan queue (execute in order)

| Plan | What | Size | Blocked by |
|---|---|---|---|
| [R2](r2-pipeline-integration.md) | MontagePipeline live: port evaluation/commit effects, dissolve frame_renderer clusters B/C/E | Done | R1 |
| [R3](r3-lod-ladder-adoption.md) | Ladder replaces montage_lod planning; one pyramid store; ops once per rung; PyQtGraph parity via capabilities | L | R2 |
| [R4](r4-timer-and-governor-audit.md) | Every QTimer justified or deleted; governor shrinks to telemetry + two knobs | M | R2 |
| [R5](r5-test-and-docs-truth-pass.md) | Delete wrong-path tests; docs/current-state truth pass; known-red ledger emptied | M | R3–R4 |

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
8. **Exit gates are hard and immutable.** A plan is not done while its
   gate fails; never edit a gate to match a result (this happened once —
   see the R2b section of the R2 plan — and cost a full stabilization
   pass).
9. **No symptom patches.** Clamping workers/batches, adding pacing, or
   editing a harness because "it reduces the symptom" requires a written
   root-cause note first. The R2 freezes were a per-completion commit
   storm, camera-key churn, deps-as-ordering, and per-tile-native FFT
   floors — none of which worker clamps could fix.
10. **Deps are not ordering; cameras are not identity.** Kernel deps
    fail-propagate (data dependencies only); viewport keys never belong in
    task keys or supersession values
    (`test_camera_only_retarget_never_invalidates_rung_work` pins this).
11. **Payload semantics ride payload metadata** (`quality`,
    `texture_kind`, levels) — never dtype/shape sniffing in a backend.

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
- **GPU harness** (~20 s, real hardware, asserts `stall_assertions==0`):
  `ARRAYSCOPE_GPU_TESTS=1 XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland … -m pytest tests/gpu_interaction -n 0`
- **Workflow benchmark:**
  `… -m arrayscope.tools.profile_montage_workflow --backend {vispy|pyqtgraph} --montage-lod-policy {resident|native-only} [--jsonl FILE]`
  `STALL ASSERTION` on stderr must stay 0.
  R2 evidence is saved as
  `tests/artifacts/r2-profile-montage-workflow-{pyqtgraph,vispy}-{resident,native-only}.jsonl`.
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

1. Suites per rule 5; GPU harness green incl. `stall_assertions==0`.
2. The replaced code is deleted; grep proves no references remain.
3. Numbers in the commit message; ADR 0053 status table updated.
4. Claude memory updated (`arrayscope-lod-residency` note: queue position,
   tip, new gotchas).
