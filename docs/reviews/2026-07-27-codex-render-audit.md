# Codex render branch audit

Date: 2026-07-27

Branch: `claude/review-codex-rendering-0fa4f6`

Product tip audited: `492e8710`

Baseline: local `main` at `cc97bd24`

## Executive summary

The branch does not satisfy the normative progressive-render contract. More
seriously for a review branch, its acceptance surfaces can report success while
R1, R3, R4, R5, and R7 are violated.

I confirmed **13 wrong-behavior tests**:

- **12 rendering/profiler tests** contradict or fail to require the contract:
  9 were added or materially modified by this branch, and 3 are unchanged
  incumbent tests that should have changed.
- **1 branch-added trace test** deliberately pins a harmful unrelated side
  effect.

This count is deliberately narrow. It excludes low-level atlas representation
tests that remain useful if an owner above the adapter supplies a bounded,
complete transaction. It also excludes tests that are merely too weak unless
they positively certify a non-compliant behavior.

The audit found **six additional ownership-altitude errors**, beyond the two
already identified in the review request. Including those known errors, the
branch has eight material ownership errors in the audited rendering path.

The profiler's new diagnostics are partly worth retaining, but its pass/fail
gate is not. Keep the exact required-set, generation, ACK, task-order, physical
pixel, and event-loop evidence. Replace the latency/order certification with
contract invariants, make timing report-only, and reduce the synthetic fixture
bulk only after the invariant gate exists.

At the product tip, `main..HEAD` was 50 files, 7,170 insertions, and 578
deletions. Twenty-two test files changed. The two profiler files account for
1,580 gross added lines: 694 in
`arrayscope/tools/profile_montage_workflow.py` and 886 in
`tests/app/test_profile_montage_workflow.py`.

I reviewed every added or modified test hunk in those 22 test files against
R1–R7, then traced each relevant assertion into its production owner. I also
searched unchanged tests around the contradicted level-readiness and residency
paths; those are the source of the three incumbent false-confidence tests
called out below.

## P0 findings

### 1. CONFIRMED — The profiler can certify a run that violates the contract

Files:

- `arrayscope/tools/profile_montage_workflow.py:7941`
- `arrayscope/tools/profile_montage_workflow.py:9556`
- `arrayscope/tools/profile_montage_workflow.py:10489`
- `tests/app/test_profile_montage_workflow.py:1430`
- `tests/core/test_progressive_render_oracle.py:313`

What is wrong:

`_r8_certification()` certifies latency, coarse-before-target ordering, and
selected workflow facts. It does not consume the progressive-render oracle and
does not require per-round production purpose, exactly one preview and target
pass, level containment for the committed presented set, or bounded callback
work. The command's exit status is based on `r8_gate_passed`, so a run may exit
successfully with R1 or R3 violated. The test at line 1430 constructs such a
passing phase from summary/order/timing fields without any production identity
or tile-value containment evidence.

The only recorded-trace oracle test is optional and skips when no JSONL path is
provided. Because committed JSONL artifacts are forbidden, that is not an
ordinary repository gate.

Why it matters:

This is the branch's main acceptance harness. A green result gives false
confidence precisely for the defects the branch was meant to fix.

Recommended action:

Make an invariant verdict mandatory for the profiler's success exit:
round/generation identity; purpose-tagged production; one P and one T at most;
reuse-floor checks; commit-time levels plus every presented tile's value bounds;
and measured callback/chunk evidence. Keep T1/T2/B medians as report-only
diagnostics. Do not permit an invariant gate to skip because a recorded artifact
is absent; feed it the live event stream or an in-memory distilled trace.

Wrong-behavior test:

- `test_r8_certification_passes_complete_semantic_and_responsive_phase`
  (materially modified on the branch)

### 2. CONFIRMED — PyQtGraph is allowed to present against partial level evidence

Files:

- `arrayscope/window/frame_effects.py:4618`
- `arrayscope/window/frame_effects.py:4687`
- `arrayscope/render/level_stats.py:2077`
- `tests/window/test_montage_backend.py:1722`
- `tests/window/test_montage_backend.py:1790`
- `tests/window/test_montage_backend.py:1845`

What is wrong:

`tile_layer_first_pixels_wait_for_level_source()` stops waiting after the first
CPU/refined evidence batch. The separate level-stat publication path likewise
publishes after a minimum source-count threshold. Neither decision proves that
the chosen levels contain the value range of every tile in the presented
transaction.

Three tests positively assert this partial-evidence behavior:

- one refined batch is enough to unblock first pixels;
- a partial seed with mixed preview evidence is enough;
- a first CPU histogram batch publishes partial bounds.

The first and third tests already existed on `main`; the branch should have
changed them when adopting R3. The mixed-evidence test was added by the branch
and extends the wrong contract.

Why it matters:

R3 requires levels to contain every presented tile and requires PyQtGraph to
have the final round levels before its first draw. Partial evidence makes
brightness change as the montage fills and can clip values that arrive after
the seed batch.

Recommended action:

Replace source-count readiness with one round-owned levels transaction whose
bounds cover the exact committed tile set. Add the contract's absolute and
relative containment tolerances and assert PyQtGraph has final round levels
before first presentation. Preserve a red regression test for mixed dtypes and
outlier tiles arriving late.

Wrong-behavior tests:

- `test_pyqtgraph_current_predicate_accepts_partial_refined_first_batch`
  (unchanged incumbent)
- `test_pyqtgraph_current_predicate_accepts_partial_seed_with_mixed_preview_evidence`
  (branch-added)
- `test_first_cpu_histogram_currently_publishes_partial_refined_batch`
  (unchanged incumbent)

### 3. CONFIRMED — The “one compact preview” path bypasses the governor without a 50 ms proof

Files:

- `arrayscope/render/pipeline.py:148`
- `arrayscope/render/pipeline.py:433`
- `arrayscope/window/frame_effects.py:4461`
- `arrayscope/window/frame_effects.py:4483`
- `arrayscope/window/frame_effects.py:4898`
- `arrayscope/window/frame_effects.py:4940`
- `arrayscope/display/backends/pyqtgraph/tiles.py:1581`
- `arrayscope/display/wgpu_imageview2d.py:1550`
- `arrayscope/display/wgpu_imageview2d.py:3381`
- `tests/render/test_pipeline.py:610`
- `tests/render/test_pipeline.py:635`
- `tests/window/test_montage_backend.py:3604`
- `tests/display/test_pyqtgraph_preview_atlas.py:217`

What is wrong:

The pipeline submits the entire 256–512 tile preview as one worker item and one
completion batch. Both PyQtGraph and WGPU effect limits then override the normal
item cap and deadline for the aggregate. PyQtGraph refuses a prefix and builds,
packs, and converts the entire transaction in the GUI commit. WGPU similarly
builds the atlas on the presentation path.

The tests assert a 272-tile single worker/completion transaction, a cap bypass
with no deadline, and zero PyQtGraph acknowledgement for a large prefix. None
requires measured evidence that every unchunked callback completes in less than
50 ms, and there is no chunked fallback.

Why it matters:

R5 permits one atomic-looking preview burst only when all bulk work is chunked
through the governor or each unchunked callback is proven below 50 ms. Compact
storage is a representation choice, not permission for unbounded compute or GUI
work.

Recommended action:

Keep the compact physical representation, but route materialization and packing
through governor-owned chunks. Stage immutable pages off the GUI thread and
make the final swap demonstrably cheap, or retain a measured sub-50 ms fast path
with an automatic chunked fallback. Test both the fast and fallback paths,
including loaded parallel execution.

Wrong-behavior tests:

- `test_full_preview_scope_uses_one_worker_and_one_completion_batch`
- `test_full_preview_scope_batches_missing_tiles_beside_retained_exact_coverage`
- `test_pyqtgraph_full_preview_cap_bypass_requires_explicit_aggregate_marker`
- `test_large_preview_prefix_is_not_acknowledged_as_physical_coverage`

All four were added by the branch.

### 4. CONFIRMED — PyQtGraph complex/FFT is exempted from the required preview

Files:

- `arrayscope/tools/profile_montage_workflow.py:5635`
- `arrayscope/tools/profile_montage_workflow.py:5650`
- `arrayscope/display/backends/pyqtgraph/tiles.py:1685`
- `tests/app/test_profile_montage_workflow.py:2234`

What is wrong:

The profiler sets `coarse_target_preview_required` false for the PyQtGraph FFT
case. The backend defers the compact preview for complex data, and the
certification test asserts that the absence of a preview is not a failure.

Why it matters:

R4 is backend- and dtype-explicit: if a reduced preview is required, a missing
backend path cannot silently turn the target into the first pass. This is one of
the observed defects named by the contract.

Recommended action:

Keep the gate red until a reduced RGB/complex preview is supported, or explicitly
declare that backend/dtype pipeline unsupported. Do not encode target-first as a
passing exemption.

Wrong-behavior test:

- `test_r8_certification_currently_exempts_deferred_pyqtgraph_complex_preview`
  (branch-added)

### 5. CONFIRMED — The progressive “oracle” cannot prove R1 or R3

Files:

- `arrayscope/tools/progressive_render_oracle.py:42`
- `arrayscope/tools/progressive_render_oracle.py:93`
- `arrayscope/tools/progressive_render_oracle.py:141`
- `arrayscope/tools/progressive_render_oracle.py:206`
- `tests/core/test_progressive_render_oracle.py:173`
- `tests/core/test_progressive_render_oracle.py:211`

What is wrong:

The snapshot model has counts, resident levels, floors, evidence counts, and
WGPU upload totals. It has no round generation, production purpose, per-tile
production identity, committed presented tile IDs, per-tile value ranges, or
the levels used by a particular commit.

Consequently:

- R1 only detects upload growth outside the current `{P, T}` set. It cannot
  detect duplicate production at P or T, more than one P/T pass, or distinguish
  production from speculative residency.
- Pairwise upload deltas can span a floor change because only session ID is a
  boundary.
- R3 detects frozen or inactive evidence by count. It never checks the required
  value-range containment.
- The test at line 211 explicitly passes a partial-evidence sequence for lack
  of a third count sample; this is a characterization of the heuristic, not an
  R3 result.

Why it matters:

The formatter emits “PASS: no R1/R3 violations,” a much stronger statement than
the captured data supports.

Recommended action:

Capture purpose-tagged production events keyed by round and tile, and
commit-time events containing the exact presented set, round levels, and tile
value bounds. Until then, label these checks as heuristic suspects and report
R1/R3 as unverifiable rather than passed.

This finding does not add the renamed count-heuristic characterization to the
13-test total: after the safe mechanical rename/docstring correction it no
longer claims to be an R3 acceptance test. Its underlying oracle remains too
weak.

### 6. CONFIRMED — Target production performs speculative native residency during the round

Files:

- `arrayscope/render/effects.py:392`
- `arrayscope/render/effects.py:423`
- `arrayscope/render/effects.py:2012`
- `arrayscope/window/frame_effects.py:306`
- `arrayscope/window/frame_effects.py:4825`
- `tests/render/test_effects.py:306`
- `tests/ui/test_montage_scroll_settling.py:178`
- `tests/window/test_montage_backend.py:3326`

What is wrong:

The target pass can carry a full native plane sidecar and warm canonical native
pages. Tests assert that reduced target work carries this speculative plane,
that a target pass warms pages for later crop rebind, and that native-source
prefetch is admitted in bounded cohorts during a gesture.

The mid-gesture cohort test is unchanged from `main`; it should have changed
when R7 made post-settle ownership normative.

Why it matters:

R7 makes speculative residency post-settle work. A bounded cohort is still
incorrectly timed if it competes with the active preview/target round, and the
piggybacked target work makes R1 production purpose ambiguous.

Recommended action:

Remove speculative native warming from preview and target production. Schedule
purpose-tagged, breadth-before-depth prefetch only after settlement and only
when visible work is not starved.

Wrong-behavior tests:

- `test_reduced_target_currently_carries_speculative_native_plane`
  (branch-added)
- `test_wgpu_target_pass_currently_warms_native_pages_for_crop_rebind`
  (branch-added)
- `test_wgpu_native_source_prefetch_stays_in_bounded_two_tile_cohorts_mid_gesture`
  (unchanged incumbent)

## P1 findings

### 7. CONFIRMED — Six additional decisions are made at the wrong ownership altitude

Files:

- `arrayscope/render/pipeline.py:433`
- `arrayscope/window/frame_effects.py:4461`
- `arrayscope/window/frame_effects.py:4618`
- `arrayscope/window/frame_effects.py:4898`
- `arrayscope/display/backends/pyqtgraph/tiles.py:1581`
- `arrayscope/display/wgpu_imageview2d.py:1550`
- `arrayscope/render/effects.py:392`
- `arrayscope/window/frame_effects.py:306`

The two already-known errors are not rediscovered here: preview level is chosen
per tile in `arrayscope/render/ladder.py`, and round levels are assembled from
per-slab worker results in `arrayscope/render/level_stats.py`.

The six additional errors are:

1. `FramePipeline._submit_preview_batch()` chooses whole-pass task granularity.
   Chunk size and continuation belong to the governor.
2. The PyQtGraph and persistent/WGPU effect-limit helpers override the
   governor's item and deadline limits for an aggregate. A presentation effect
   must not redefine pacing policy.
3. The PyQtGraph backend infers required-set completeness and refuses a prefix.
   The adapter should apply an explicit complete or staged transaction, not
   decide what completeness means.
4. The WGPU backend infers completeness from
   `len(preview_payloads) == planned_count` before selecting an atlas
   representation. The round owner should state transaction semantics; the
   backend may choose only how to apply them.
5. `tile_layer_first_pixels_wait_for_level_source()` decides, with
   backend-specific policy, when round levels are semantically sufficient. A
   round-level owner must publish final/containing levels; the adapter should
   consume them.
6. The target effect decides whether to piggyback canonical-plane warming.
   Speculative residency admission belongs to a post-settle scheduler/prefetch
   lane, not tile evaluation or frame application.

Why it matters:

These placements make one semantic decision emerge independently at pipeline,
effect, and backend levels. That is the same shape as the known per-tile
preview-level and per-slab level-stat defects.

Recommended action:

Create one round plan that owns floors, exact required set, semantic levels,
production purpose, and governed continuations. Pass explicit immutable
transactions downward. Backends own texture/item/atlas mechanics only.

### 8. CONFIRMED — Generic live trace output is no longer live and can be lost on a crash

Files:

- `arrayscope/core/trace.py:31`
- `arrayscope/core/trace.py:82`
- `tests/core/test_trace.py:27`

What is wrong:

The branch changes the file sink to a 64 KiB user-space buffer. The new test
positively asserts that an emitted row is invisible until close. A process
crash can lose the buffered tail, and an external observer cannot follow the
nominally live trace.

Why it matters:

This is unrelated to preview/LOD/levels and weakens diagnostics precisely when
a crash or hang makes the final events valuable. An in-process ring does not
recover an unflushed external trace after process death.

Recommended action:

Restore observable per-event or explicit-boundary flushing for the generic live
sink. If profiler write overhead is material, use an asynchronous writer or a
profiler-specific buffered mode with documented durability semantics.

Wrong-behavior test:

- `test_live_trace_sink_hides_buffered_rows_until_close` (branch-added)

### 9. CONFIRMED divergence; SUSPECTED optimistic timing — The profiler suppresses production range-change signals

Files:

- `arrayscope/tools/profile_montage_workflow.py:1142`
- `arrayscope/tools/profile_montage_workflow.py:1207`
- `arrayscope/tools/profile_montage_workflow.py:4167`
- `tests/app/test_profile_montage_workflow.py:1317`

What is wrong:

`_hold_fit_for_montage_build()` blocks ViewBox signals while fitting and then
requests a render explicitly. This is confirmed to differ from the user's
ordinary fit/range-change path. The degree to which it improves the reported
latency is suspected rather than measured in this audit.

Why it matters:

The harness may avoid production session churn or callbacks and then certify a
synthetic action as a user-visible journey.

Recommended action:

Retain this as an explicitly labelled isolated cold-fill diagnostic if useful.
Use the unsuppressed production action for acceptance and compare both paths
before attributing a latency improvement.

## P2 findings and missing gates

### 10. CONFIRMED — Existing tests gave false confidence and required contract gates are absent

Files:

- `tests/window/test_montage_backend.py:1722`
- `tests/window/test_montage_backend.py:1845`
- `tests/core/test_progressive_render_oracle.py:81`
- `tests/core/test_progressive_render_oracle.py:313`
- `tests/app/test_profile_montage_workflow.py:1430`

Tests that should have changed but did not:

- The two incumbent PyQtGraph partial-level tests in finding 2.
- The incumbent mid-gesture native-prefetch test in finding 6.

Tests or gates still missing:

- R1/R2: purpose-tagged per-round production that proves at most one P and one
  T pass and proves reuse never regresses.
- R3: exact containment for every presented tile, with the contract's absolute
  and relative tolerances, including PyQtGraph final levels before first draw
  and WGPU atomic add-plus-widen.
- R4: an explicit backend × dtype preview matrix with no target-first
  exemptions.
- R5: callback elapsed-time evidence, governor continuation/chunk evidence, and
  a tested fallback when the compact burst exceeds 50 ms.
- R6: load-shedding tests that preserve liveness and forbid black/empty freeze
  while visible work remains.
- R7: purpose-tagged post-settle prefetch admission and starvation checks.

The repository does **not** need a bound on the number of distinct quality
levels visible at once. The contract explicitly permits retained finer tiles
and free coarse reuse, so three or more visible levels can be legal.
`tests/core/test_progressive_render_oracle.py:81` is right to preserve such a
mixture. What must be bounded is production: no more than the round's P and T
passes.

Recommended action:

Make the contract invariants the primary assertions and leave latency,
resident-level mixtures, and heuristic evidence counts as diagnostics. A
fixture should not be called “complete” or “certified” unless it contains the
identities and ranges needed to prove the claim.

### 11. CONFIRMED bulk; recommendation — Retain evidence plumbing, replace the gate, then reduce fixtures

Files:

- `arrayscope/tools/profile_montage_workflow.py` (10,495 lines total; +694/-92)
- `tests/app/test_profile_montage_workflow.py` (4,063 lines total; +886/-1)

Worth keeping:

- exact required-tile-set and generation/session correlation;
- ACK and task-order evidence as diagnostics;
- physical-pixel/reference and event-loop counters;
- fail-closed parsing/formatting concepts from the separate oracle;
- bounded in-memory trace retention, provided generic live-trace durability is
  restored.

Unjustified or misleading in its current role:

- latency medians and coarse order as acceptance rather than report-only data;
- a second certification implementation disconnected from the normative
  oracle;
- backend/dtype exemptions that turn missing preview into green;
- signal-suppressed synthetic actions used as user-journey gates;
- hundreds of lines of repeated synthetic event dictionaries that encode
  schema scaffolding but not contract truth.

Why it matters:

The added size raises review and maintenance cost without raising the
acceptance gate's sensitivity to the named defects. More timing cases around an
incomplete verdict make the green result look stronger while leaving the
contract unmeasured.

Recommended action:

Do not delete the profiler wholesale. First replace the exit gate with the
mandatory invariants described in finding 1. Then consolidate repeated event
fixtures into typed/table-driven builders, remove duplicated order
certification, and split synthetic microbenchmarks from user-journey
acceptance. The useful result should be a smaller evidence producer feeding one
contract oracle, not a large latency gate beside an optional oracle.

## Safe mechanical corrections made during this audit

Commit `a4fe8cff` changes only comments, docstrings, and misleading test names.
It labels the current violations instead of describing them as intended
contract behavior. No assertion or execution path was changed.

Focused validation for that commit:

- 7 directly affected tests passed.
- `/home/thomas/miniconda3/envs/arrayscope/bin/ruff check ... --select F821,E9`
  passed for all touched Python files.
- `ruff format --check` passed for all touched Python files.
- `git diff --check` passed.

No render defect was fixed as part of this review.
