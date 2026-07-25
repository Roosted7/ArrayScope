# Redesign — program record and dossier index

**Status: the program is closed and merged.** The R1–R7 architecture
rewrite, the V0–V4 visible-truth queue, and the P1–P9 measured performance
program all landed linearly on `main` (V4: 2026-07-14; the GPU engine that
absorbed the P-program's endpoint: 2026-07-16).

- **The active queue is [`../queue.md`](../queue.md).** Nothing in this
  directory orders current work.
- The ground rules this program produced were promoted to
  [`../ground-rules.md`](../ground-rules.md) and remain standing law.
- Rejected experiments are indexed in [`../graveyard.md`](../graveyard.md).
- Execution records:
  [`archive/v-program-execution-record-2026-07.md`](archive/v-program-execution-record-2026-07.md)
  (V0–V4/T1 and the P checkpoints as logged here) and
  [`archive/p-program-log-2026-07.md`](archive/p-program-log-2026-07.md)
  (the P1–P9 log as it accumulated in the roadmap). R1–R8 logs, known-red
  ledger, and R8_STATUS are in [`archive/`](archive/).

## What this program was

The rewrite delivered one kernel, one pipeline, one lifecycle machine
(`frame_renderer.py` deleted). The R8 certification program that followed
was closed because its gates measured internal counters while the screen
stayed wrong — the post-mortem is [retro-2026-07.md](retro-2026-07.md) and
is required reading before designing any acceptance gate. The V-program
fixed the user-visible truth (black tiles, priority order, silent dead
paths, loud non-convergence); the P-program measured performance causes one
at a time and, by rejection after rejection (see the graveyard), isolated
the structural cost that the GPU engine now solves by construction.

## Live dossiers (evidence + implementation contracts, still referenced)

These are **not archive**: queue steps and tests cite them.

| Dossier | Owns |
|---|---|
| [black-tiles-and-priority.md](black-tiles-and-priority.md) | Root-cause dossier behind V1/V2 (file:line evidence) |
| [tracing-pipeline.md](tracing-pipeline.md) | Trace-event spine design (schema v1, verify/latency tools) |
| [marathon-salvage.md](marathon-salvage.md) | Salvage audit of the `redesign-r8-marathon` worktree; Tier-2/3 items still undecided |
| [coverage-stall-2026-07-15.md](coverage-stall-2026-07-15.md) | Shared-fanout coverage-refill stall (fixed; exit gate recorded) |
| [stale-empty-tiles-2026-07-16.md](stale-empty-tiles-2026-07-16.md) | Identity-aliasing starvation livelock (fixed) + open follow-ups feeding queue step 1 |
| [slice-retention-staleness-2026-07-16.md](slice-retention-staleness-2026-07-16.md) | Retained-transition replacement latency (fixed at the rung-label owner) |
| [g5-source-grid-pyramid-2026-07-16.md](g5-source-grid-pyramid-2026-07-16.md) | **Authoritative implementation contract** for the remaining ADR 0056 work (queue row 1) |
| [retro-2026-07.md](retro-2026-07.md) | Why the circling happened; the rules that stop it |
| [fill-throughput-2026-07-18.md](fill-throughput-2026-07-18.md) | 272-tile raw-fill "stall" adjudication: O(tiles²) throughput collapse, not a lost wakeup (fixed at the roots) |
| [wgpu-field-stalls-2026-07-18.md](wgpu-field-stalls-2026-07-18.md) | wgpu field stalls 259-1/1-1: physical first-pass quality drift (fixed `43287f8`) |
| [demand-freshness-cold-fill-2026-07-19.md](demand-freshness-cold-fill-2026-07-19.md) | AUTO-camera demand-freshness cold_fill red: ViewportBridge dropped entry camera intent (live path fixed `6fd0c262`; unit-gate fixture still open) |
| [wgpu-frame-pacing-2026-07-21.md](wgpu-frame-pacing-2026-07-21.md) | wgpu screen frame pacing: cadence instrumentation landed (`c67a4730`), and the phase-locked-pacer premise **measured and refuted** — schedule slip (12–18 ms p50) is as large as the whole refresh period, so the limit is event-loop occupancy, not phase |
| [index-window-retarget-cost-2026-07-25.md](index-window-retarget-cost-2026-07-25.md) | Montage index-window remap cost: the delta premise **measured and refuted** (a +1 step leaves 0 of 100 slots unchanged), two per-tile redundancies removed for ≈ −25%, and the remaining slices characterised |

**Dossier convention:** a dossier is created when a field defect or design
slice needs evidence that outlives one commit. It records symptom → trace
evidence → root cause → fix owner → exit gate → follow-ups, and it is
updated (not duplicated) when follow-ups land. When a dossier's last
follow-up closes, move it to [`archive/`](archive/).
