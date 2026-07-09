# 0052 — UI-work pacing governor (superseded)

**Status:** Superseded by [ADR 0053](0053-execution-kernel-and-modular-pipeline.md) R4
(2026-07-09). This record remains as historical context for the removed UI-work
governor design.

## Supersession

R4 chose the smaller owner model instead of implementing the proposed channel
registry and decision pipeline below:

- `ResourceGovernor` keeps telemetry/observation records and publishes only the
  live knobs still owned outside static policy: bridge drain budget and commit
  batch bounds.
- Kernel lane quotas are the execution throttle; compute policy supplies the
  initial lane quota values and the kernel owns stale-work pruning/admission.
- GUI callback vocabulary remains in `core/gui_callback_budget.py`.
- `core/latency_feedback.py` is the EWMA/outlier-suppression model used by the
  bridge/commit budgets, not a per-controller regression controller.
- Per-controller worker clamps, `decide_ui_work`, work signatures,
  conservative cold starts, benchmark decision rings, and interaction-edge
  reapplication timers are deleted.

## Context

The `ResourceGovernor` UI-work path started as a small EWMA batch-sizer and has become a
control system grown by patches. As of the PyQtGraph pacing landing:

- `decide_ui_work` stacks ~10 adjustment stages (EWMA sizing, overhead+marginal item model,
  byte model, four distinct recovery/backoff scalers, idle single-item floor, conservative
  cold-start cap), each appending detail strings.
- Five interacting channels (`montage_present_total`, `montage_cold_commit`, `montage_commit`,
  `tile_layer_commit`, `texture_upload`) with per-call-site channel choices; the first
  persistent-commit batch now takes the min of two channels' decisions at a call site.
- Cold start lives in a side dict keyed by channel, released by a hard-coded work-class set —
  the VisPy work classes were missing until 2026-07-06, so VisPy channels never left batch 1.
- Two near-identical cost-class reset helpers (`_reset_tile_layer_feedback_if_needed`,
  `_reset_persistent_tile_feedback_if_needed`) hold channel lists in code.
- Diagnostics: a 4096-entry decision ring serialized into every benchmark phase record.

Each piece is individually justified (the correlation gate on the regression models is right;
cold start prevents first-commit gaps; the recovery scalers fixed measured oscillations). The
shape is the problem: invariants live in call flow, exactly the defect class ADR 0051 removed
from tile state. For pacing the analogous failures are oscillation, channels wedged
conservative, and unexplainable batch decisions.

## Decision (proposed)

Make pacing a model with named invariants instead of a patch stack:

1. **Channel registry.** One table declares each channel: its work classes, backend scope,
   cost-class signature inputs, cold-start policy, and which models may apply. Call sites stop
   choosing channel combinations ad hoc (the min-of-two-channels first-commit rule becomes a
   declared relationship, e.g. `montage_present_total.cold_cap_by = montage_cold_commit`).
2. **Per-channel state object.** EWMAs, variances, covariances, outlier streaks, cold-start
   phase — one dataclass, owned by the channel, reset through one seam keyed by the cost-class
   signature. The side dicts and duplicated reset helpers disappear.
3. **Decision pipeline with named stages.** `decide_ui_work` becomes an ordered list of pure
   stages (state × request → adjustment + reason). The stage list is data; tests can assert
   the pipeline, and the details strings fall out of stage names.
4. **Invariants, property-tested:** batch/byte floors always respected; backoff and recovery
   factors bounded and monotone (no oscillation under stationary observations — property
   test with synthetic observation streams); regression models apply only above the
   correlation gate (keep r ≥ 0.25); cold start is a channel-state phase that every
   channel-relevant work class can release, never a work-class allowlist in code.
5. **Diagnostics budget.** Decisions and observations share one bounded ring; full dumps into
   benchmark records are opt-in (`--governor-trace` or env), not the default JSONL payload.

## Phases

- **G1 (mechanical):** extract the per-channel state object + single reset seam; registry
  table; no behavior change (pin with before/after decision-trace equality on recorded
  benchmark JSONL).
- **G2:** stage pipeline + invariant property tests; delete the duplicated helpers.
- **G3 (evidence):** re-run the workflow benchmarks on both backends; only then consider
  behavior changes (e.g. replacing recovery scalers with a single bounded controller).

## Alternatives considered

- **Keep patching.** Rejected: the 2026-07-06 cold-start bug (VisPy never released) is the
  predictable result; the next backend or work class will repeat it.
- **Full PID/queueing controller now.** Deferred: no evidence the current model class is
  insufficient once invariants hold; G3 decides.

## Related records

ADR 0046 (evidence-first), ADR 0051 (single-owner lifecycle — the same defect-class argument
for tile state), Plan 04/05 (the workloads that stress pacing).
