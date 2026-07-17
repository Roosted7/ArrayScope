# Graveyard — rejected approaches, do not re-derive

One row per experiment or design that was tried and rejected with evidence,
or vetoed after serious study. Purpose: stop the circling. Before starting a
performance or scheduling experiment, search this file; if your idea matches
a row, read the linked evidence first and satisfy the "retry only if"
condition before writing code.

**Convention:** the commit that reverts an experiment adds a row here in the
same commit. Keep rows to two lines; link the full record (program log,
dossier, review, or artifact path). Never delete rows.

## Scheduling / presentation admission

| Date | Tried | Why rejected | Retry only if | Evidence |
|---|---|---|---|---|
| 2026-07-14 | **P1: narrowed prefetch-busy predicate** (marathon's `required_target_settled()` form) | No FFT gain; scalar scroll +26%; PyQtGraph freeze unchanged | A trace shows prefetch suppression on the critical path | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-14 | **P5: coalesced completion drain** (empty-edge notification + timer-paced continuations, 3 capacity-wake variants) | All three variants stranded 36/36 exact targets at preview on the real VisPy gate; the bridge flood cannot be isolated from the capacity/refill protocol by changing `CompletionQueue` alone | The pipeline refill contract is redesigned and proven on a real display | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-14 | **P6 variants: pacing every range replay; reusing the frame-viewport timer; uncommitted preview session as viewport truth** | Delayed V1 boundary reveal / stranded 36 V2 targets / made V2 order-dependent | Never — derive tiled viewport work from `_committed_display_frame` only | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-15 | **P9: one-tile presentation commits** (removing the learned batch floor) | FFT scroll +44%, scalar +88%; stranded 60/60 at exact LOD 4; amplifies fixed per-commit cost without supplying evaluation refill | Never as admission batching; fix transaction ownership instead | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-15 | **P9: completion-owned admission refill** | VisPy veto: scalar scroll +94%, cold fill +17%, visibly mixed FFT tiles (PyQtGraph had improved) | Backend transaction ordering passes the corrected V1 real-pixel gate on BOTH backends; a clean lifecycle trace is insufficient | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-15 | **P9: byte-cap tuning** (1 MiB → 4 MiB persistent cohort) | Item cap was binding, not bytes; scroll regressed | A trace first shows the byte cap truncating an otherwise larger admitted cohort | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-15 | **P9: generic eight-item admission floors** (unconditional, and `not display_committed`) | Traded continuity for throughput; `display_committed` is not a plan phase | An explicit plan-phase signal exists to scope the floor | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-15 | **Plan-wide physical barrier on every per-tile rung** (generalizing the shared-transform gate) | Passed 419 offscreen tests, then stranded 45/272 ordinary PyQtGraph raw tiles live | Never — the barrier owns only the shared path that bypasses `FramePipeline` | [V-record](redesign/archive/v-program-execution-record-2026-07.md) |
| 2026-07-16 | **Timer/session/warming shortcuts for retained-slice staleness** | The real cause was a mislabeled first-presentable `DESIRED` rung quota-blocking successors, not rebirth or uploads | Never — fix labels/owners, not pacing | [dossier](redesign/slice-retention-staleness-2026-07-16.md) |

## Correctness / lifecycle

| Date | Tried | Why rejected | Retry only if | Evidence |
|---|---|---|---|---|
| 2026-07-18 | **Commit-side refinement withholding** (quality gate in `_quality_pass_admissible_upserts` holding exact upserts for covered tiles until plan-wide coverage) | Presentation must NEVER withhold better ready data — the "boom" upgrade is the ideal; the real barrier is phase-2 COMPUTE not starting during phase 1 (Thomas). The gate also stuck open on PyQtGraph and froze LOD upgrades after zoom | Never — enforce phase separation at work submission, not at commit | rendering.md progressive contract |
| 2026-07-14 | **Deleting CPU atomic successors wholesale** (V2) | PyQtGraph discarded 59 compatible slots and visibly rebuilt 12→60 | Never — atomicity is semantic: it preserves *compatible pixels*, nothing more | [V-record](redesign/archive/v-program-execution-record-2026-07.md) |
| 2026-07-14 | **Harness-side second completion model** (reconstructing "done" from six queues + plan length) | Diverged from the live owner; the harness never settled on a converged frame | Never — read `visible_plan_complete()` / lifecycle truth | [V-record](redesign/archive/v-program-execution-record-2026-07.md) |
| 2026-07-14 | **Watchdog that silently repairs** (`release_idle_evaluation_claims()` when idle) | Changed live state, erased the owner chain it was investigating, retried the same architecture | Never — watchdogs are evidence-only | [V-record](redesign/archive/v-program-execution-record-2026-07.md) |
| 2026-07-15 | **P2: committed-frame `level_source`** | Regressed V2 center-out 14/16 → 4/16; relative-window mapping error; no maturity rule for when level evidence becomes committed truth | The maturity/semantic rule is designed first | [P-log](redesign/archive/p-program-log-2026-07.md) |
| 2026-07-15 | **Forced extra complex copies / disabling two-quality shared preview** (psychedelic-tiles hypotheses) | The atlas already owns a defensive copy; deterministic numeric tests prove shared-preview values; the real cause was stale page uniforms | Never | [V-record](redesign/archive/v-program-execution-record-2026-07.md) |
| 2026-07-16 | **Synthesizing payloads for a future montage window** via `_ensure_display_tile_payload` (scroll-direction warming) | Manufactures lifecycle/presentation state instead of warming evaluated content | Never — warm evaluated chunks through the backend seam | [continuation brief](proposals/gpu-port-continuation.md) |
| 2026-07 | **Refined levels from preview evidence, unconditionally** | Violates the deliberate backend split: shader-windowing backends refine in place; CPU-LUT backends must wait for final evidence | Never unconditionally — gate on `image_view_backend_capabilities(...).shader_windowing` | ADR 0050 area; `LevelStatsService._preview_evidence_can_refine` |

## Architecture / process

| Date | Tried | Why rejected | Retry only if | Evidence |
|---|---|---|---|---|
| 2026-07-14 | **R8-style certification via internal counters** | Six weeks of green gates while the screen showed black tiles; fixes narrowed predicates until the counters lied | Never — pixels on a real display are the gate | [retro](redesign/retro-2026-07.md) |
| 2026-07-15 | **G4b: session-reuse surgery** (avoid `FrameSession` rebirth on scrub) | Measured on real data: scrub-step render 10.2 ms mean / 13.8 p95 — rebirth is not the bottleneck | A heavier dataset shows rebirth on the critical path | [continuation brief](proposals/gpu-port-continuation.md) |
| 2026-07-15 | **Datoviz as the renderer** | CPU-readback Qt backend, per-request upload copies, no protocol compute | The 8 recorded upstream questions resolve favorably | [tensor-engine-endpoint](proposals/tensor-engine-endpoint.md) |
| 2026-07-15 | **Pygfx as the renderer** | Would introduce a second scene graph beside the semantic command protocol | Never | [tensor-engine-endpoint](proposals/tensor-engine-endpoint.md) |
| 2026-07-15 | **GPU op kernels (flip/crop/conjugate) as an early goal** | Operations stay on the CPU; the engine consumes evaluated planes — chunk space = evaluated-value space | Evidence-gated G6+ experiment only | ADR 0055 scope note |
| ongoing | **py-spy profiling on this machine** | Unprivileged sampling falls behind indefinitely (ptrace_scope=1) | Machine config changes | [testing/README](testing/README.md) |

Older rejected work (pre-2026-07) lives in the review docs and
[`redesign/archive/`](redesign/archive/); this ledger starts at the 2026-07
course reset.
