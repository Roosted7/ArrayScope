# The queue — what to do next

**This is the only active queue.** If any other document claims to order
current work, that document is stale — fix it to point here.
[`roadmap.md`](roadmap.md) says *why* this order serves the mission;
this file says *what, in what order, and when it counts as done*.

**Rules for this file** (they exist because the last three queues drowned):

1. Update rows **in place**. When a step lands, move its row to *Done* below
   with one line of result + a link to the evidence. Never append status
   blockquotes or execution logs here — those go in the commit message, a
   dossier under [`redesign/`](redesign/), or a dated review.
2. Every step names its **exit gate in the ring that can actually see the
   failure** ([testing/README.md](testing/README.md)). "Code exists" and
   "offscreen suite green" are not completion.
3. A rejected/reverted attempt gets a [`graveyard.md`](graveyard.md) row in
   the reverting commit.
4. Re-order only with a stated reason in the commit message.

## Now (2026-07-16, in order)

| # | Step | Exit gate |
|---|---|---|
| 1 | **G5 remainder — sparse multiresolution pyramid** (in flight on `claude/g5-sparse-pyramid`). Migrate the live ladder from whole-plane keys to logical chunks feeding the resolution seam; finish source-grid materialization and reducer families. The implementation contract is [`redesign/g5-source-grid-pyramid-2026-07-16.md`](redesign/g5-source-grid-pyramid-2026-07-16.md) — use its route canonicalization, never infer a second reduction route. | Contract doc's exit gates; reduced-LOD revisit = 0 uploads stays green; real-GL gates green; live churn ring (`tests/stress/test_interaction_convergence.py`, Wayland) green on the branch; a `trace_verify` identical-`commit_bail`-loop invariant exists and is green; one user field session on the branch without a stall dump (field stalls 2026-07-17: `/tmp/arrayscope-stall-{9,63,90,57}-1.trace.jsonl` — the 10:20 session-57 atomic-successor-wait livelock is the open reproducer) |
| 2 | **Performance-bars program on the engine.** The bars (below) are the product promise. One measured cause at a time, before/after real-Wayland harness evidence per commit; a step that regresses a bar is reverted and buried in the graveyard. | Bars trend green in `profile_montage_workflow` on real Wayland, both backends (PyQtGraph at 2× allowance) |
| 3 | **G6 — GPU histogram/levels.** Per-chunk summaries over the chunk store with the ADR 0056 coverage frontier; GPU LOD generation from resident chunks. | Levels/histogram converge from chunk summaries; no GUI-thread aggregation; real-GL gate |
| 4 | **Renderer protocol + wgpu Experiment A.** Formalize the backend-neutral semantic command table ([tensor-engine-endpoint](proposals/tensor-engine-endpoint.md)); wgpu-py vertical slice (real QRenderWidget; test `present_method` screen vs bitmap on Wayland). QRhiWidget+native runtime is the recorded production candidate. | Command table maps 1:1 onto existing seams; Experiment A renders the montage scenario on real Wayland |
| 5 | **G7 — compressed transport.** Codec ladder, measured topology; ZFP-class first. After G6. | Measured end-to-end win on real data |

## Performance bars (commitments, not history — restored from R2/R4/R8D)

- GUI callbacks < **50 ms** always; event-loop heartbeat gap ≤ **16 ms**
  during fills and scrub; warm scrub input ≤ **15 ms**; settled-idle CPU 0%.
- **#1 throughput target:** fast montage FFT index scroll ~4 fps → toward
  the ~17 fps scalar rate (2026-07-09 measurement, realistic human scroll).
- Benchmark deltas stay within ±10% of the frozen baseline unless a step
  improves them. PyQtGraph gets 2× the VisPy allowance (it targets
  headless/remote use); both backends stay first-class for correctness.

## Standing lane — test hardening & debt (parallel-safe, any order)

Safe to pick up alongside the numbered queue; each is self-contained.

- **Emit `target_satisfied_retained`** when a retained compatible payload
  closes a target — until then `trace_verify`'s strongest invariant
  (`final_required_target_acknowledged`) stays tolerated in the stress
  matrix, i.e. effectively off.
- **Framebuffer-to-CPU reference oracle + fault injection** (the
  visible-truth gap named in
  [testing/stress-and-trace-strategy.md](testing/stress-and-trace-strategy.md)).
- **complex64 PyQtGraph presentation deadlock** (deterministic; strict xfail
  in the stress matrix).
- **4 pre-existing `tests/gpu_interaction` baseline failures** (P9-era) —
  re-triage now that the engine is merged.
- **ImageViewShell duplication** (`imageview2d` 2723 / `vispy_imageview2d`
  ~1840 lines).
- **Consolidate montage cache-key derivation** behind one parity-tested
  owner (`evaluator.py` scalar + batch forms; keep the
  `montage_key_batch_fallbacks` guard).
- **Audit `_resident_source_matches_expected(source, None) → True`**
  (controller-side expected-source coverage during session switches).
- **Branch/worktree cleanup** — delete merged branches and stale worktrees
  (list in the 2026-07-16 restructure notes; `redesign-r8-marathon` stays
  read-only until its Tier-2/3 salvage is decided per
  [redesign/marathon-salvage.md](redesign/marathon-salvage.md)).

## Done (most recent first — one line each, evidence linked)

- 2026-07-16 — **Churn-convergence stall net closed** (members 4+5 of the
  deferred-stage lost-wakeup family: stale-render-generation discard/resubmit
  livelock in stage-plan/stage-value completions; exact-pass candidacy
  starvation for non-exact payloads at the target level). Commit chain and
  stage-plan callbacks now emit loud bail/decision trace events; the live
  churn scenario converges 3/3 in ~23 s and its xfail is removed. Dossier:
  [stale-empty-tiles-2026-07-16](redesign/stale-empty-tiles-2026-07-16.md).
- 2026-07-16 — PyQtGraph identity-rejected upserts made loud (`6f95ce70`).
- 2026-07-16 — Session-148 identity-aliasing follow-ups: canonical full
  ranges, per-tile ack-vs-target coverage, re-commit backoff, trace_verify
  invariant (`37979222`; dossier
  [stale-empty-tiles-2026-07-16](redesign/stale-empty-tiles-2026-07-16.md)).
- 2026-07-16 — Orange floor tiles (per-session preview metadata vs
  persistent pyramid cache) + three deferred-stage lost-wakeups (`18a207fb`).
- 2026-07-16 — Identity-aliasing starvation stall root-caused and fixed
  (`dff723b4`).
- 2026-07-16 — Montage scroll-direction GPU warming; retained-slice
  staleness fixed at the rung-label owner (dossier
  [slice-retention-staleness-2026-07-16](redesign/slice-retention-staleness-2026-07-16.md)).
- 2026-07-15 — G1–G5 slice 1 of the GPU engine landed and real-GL verified;
  physical presentation truth standing invariant; see
  [proposals/gpu-engine-plan.md](proposals/gpu-engine-plan.md) status and
  the [continuation brief](proposals/gpu-port-continuation.md).
- 2026-07-14 — V0–V4 visible-truth program closed and merged; execution
  record in
  [redesign/archive/v-program-execution-record-2026-07.md](redesign/archive/v-program-execution-record-2026-07.md).
- 2026-07-14/15 — P1–P9 measured performance program; log in
  [redesign/archive/p-program-log-2026-07.md](redesign/archive/p-program-log-2026-07.md);
  its open cause (per-step 60-slot rebind) is solved structurally by the
  GPU engine's chunked residency.
