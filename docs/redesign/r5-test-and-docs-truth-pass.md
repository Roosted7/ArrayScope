# R5 — Test pruning and docs truth pass

**Goal:** the suite pins the new architecture (and nothing else); the docs
describe reality; the known-red ledger is empty.

## Test pruning rules

Delete a test when ANY of these hold (list every deletion in the commit
message with the rule number):

1. It pins deleted machinery (WorkGraph counters, per-controller drains,
   pacing-governor batch decisions, commit-interval timing, dispatch
   derivation, stage-wait pumps).
2. It asserts an implementation accident, not a contract — e.g.
   "visible pool has exactly one thread", "drain processes ≤ N per timer
   tick". The contract versions (ordering, boundedness, no-starvation)
   already live in `tests/kernel` / `tests/render`.
3. It re-tests kernel guarantees through the UI (staleness, supersession,
   dependency ordering) with heavy window fixtures. Keep ONE end-to-end
   smoke per behavior; delete the rest.

Keep and, where thin, strengthen: surface contract tests
(`test_imagesurface_contract.py`), lifecycle conformance
(`tests/presentation`), exact-inspection-independent-of-LOD tests,
architecture guards (extended in R4), GPU interaction harness, and the
golden evaluation-output tests added in R2.

## Docs truth pass

- `docs/architecture.md`: system map gains kernel + render packages;
  ownership and placement-guide tables updated; "known architectural debt"
  rewritten (frame_renderer entries removed as they no longer exist).
- `docs/architecture/scheduling-and-memory.md`: rewrite around the kernel
  (lanes/quotas/staleness/bridge); delete the WorkGraph and per-controller
  drain sections.
- `docs/current-state.md`: new snapshot + maturity rows for kernel,
  pipeline, ladder.
- `docs/roadmap.md`: redesign section marked done with dates; X5
  evidence gates (viewport-scoped normal images X5c, region-first X5d,
  cross-OS matrix X5e) become the next queue, re-expressed against the new
  module names.
- Archive `docs/plans/lod-remaining-work/` (all five plans are landed or
  absorbed: 01/02/03/05 done, 04 absorbed into R3) to
  `docs/archive/plans/`, leaving a pointer README.
- ADR 0050/0051/0052 status sections note what the redesign changed;
  ADR 0053 status table marked complete.
- `docs/redesign/known-red.md`: every entry resolved (fixed or its test
  deleted under a rule above); the file then states "empty" with the date.

## Exit gate

- Full suite + GPU harness green with zero known-red entries.
- Suite runtime ≤ pre-redesign (~35 s at `-n 16`) — deletions should pay
  for the new suites.
- `docs/` contains no reference to WorkGraph, EvaluationController,
  frame_renderer, or montage_lod except in ADR history and archives.
