# Slice retention replacement latency — 2026-07-16

**Status:** source fix and observability verified on both backends, including
real Wayland GL.
**Evidence:** `arrayscope-diagnostics-20260715-235200.jsonl` in the main
repository checkout. This dossier owns the main-branch scheduling lane change;
the GPU continuation remains authoritative for queue order.

## Symptom and measurement

Retain-until-replace removed the black flash from single-slice scrubbing, but
the predecessor plane stayed drawn far longer than the materialization and GPU
cost justified:

- 34 of 74 sampled snapshots (45.9%) reported backend content different from
  the desired successor; the longest sampled stale interval was 2.50 s;
- 298 session transitions produced only 137 draws, with 249 display-cache
  misses;
- all 235 completed `DISPLAY_PREPARATION` admissions were initially blocked by
  quota, and no `DISPLAY_PREVIEW` work was admitted;
- the VisPy atlas was already hot (200 warm residents, zero upload time);
- evaluation median/p95 was 3.71/6.33 ms and commit median/p95 was 2.25/5.96
  ms;
- adjacent-slice prefetch attempted 48 times, all cost-blocked, and scheduled
  nothing.

The stale interval was therefore scheduler latency, not evaluation, commit,
session-rebirth, or GPU-upload latency.

## Root cause

Dimension scrubbing holds viewport interaction active until 120 ms after the
latest input. The resource governor intentionally parks
`DISPLAY_PREPARATION` during that interval while leaving one
`DISPLAY_PREVIEW` worker available for correctness pixels. When a single-tile
successor has no useful reduced floor, its first and only presentable rung is
named `DESIRED`; the ladder classified every `DESIRED` step as preparation.

The pipeline already has the correct finer gate: native work during interaction
is allowed only when retained/stage-backed source data proves it is cheap, and
cold native evaluation stays deferred. The coarse lane classification ran
before that proof could matter, so the successor waited for interaction quiet
despite an already-materialized source stage.

## Source fix

`LodLadder` now classifies work by semantic role. `DESIRED` remains
`DISPLAY_PREPARATION` when any current, ready, resident, floor, or preview
payload can provide first pixels. When `DESIRED` is the target's first and only
presentable rung, it uses `DISPLAY_PREVIEW`. The existing pipeline interaction
gate still rejects cold native work and admits retained/stage-backed extraction.
No governor quota, interaction timer, or session lifecycle rule changed.

Field observability records retention transition/replacement counts, active
age, and last/max physical replacement latency in diagnostics JSONLs. Trace
events `slice_retention_started` and `slice_retention_replaced` bracket the
same physical-acknowledgement interval and include cache/stage/upload evidence.

## Rejected paths

- **Shorten the 120 ms quiet timer:** tunes a symptom and reintroduces planning
  churn during continuous input.
- **Reuse the frame session:** already closed by measurement; rebirth cost was
  not the stale interval.
- **Warm more GPU planes:** the capture already had 200 warm residents and zero
  upload time.
- **Globally unblock `DISPLAY_PREPARATION`:** would admit refinement churn while
  interacting instead of only the successor's correctness pixels.
- **Disable retained pixels:** restores the black flicker this mechanism fixed.
- **Prefetch-only treatment:** cannot repair ownership of first-pixel work after
  a cache miss.
- **Reverse-axis UI fixture as "stage-backed":** rejected by the failing-first
  gate because the operation leaves no reusable whole-source stage for the
  successor. It is correctly a cold-native control. Centered FFT over the
  scrub axis is the deterministic retained-stage fixture.

## Exit gates

- Ladder: first/only `DESIRED` uses `DISPLAY_PREVIEW`; refinement after an
  existing floor/preview stays `DISPLAY_PREPARATION`.
- Pipeline: with interaction active and preparation quota zero,
  retained/stage-backed native first pixels run; cold native remains deferred.
- UI: pin interaction active without a fixed wait, keep the predecessor drawn
  until the successor's physical acknowledgement, and prove replacement occurs
  before interaction is released.
- Backend parity: PyQtGraph and VisPy retention gates; VisPy repeated under real
  Wayland GL with `ARRAYSCOPE_GPU_TESTS=1`.
- Broad render/window/UI slice, compileall, F821/E9, and diff checks green before
  commit.

## Verification

- 41 focused ladder/pipeline/retention gates passed offscreen;
- 396 render/window/retention/diagnostics tests passed in the broad focused
  slice;
- the VisPy retained-stage replacement and retain-without-blank gates passed
  with `ARRAYSCOPE_GPU_TESTS=1 QT_QPA_PLATFORM=wayland`;
- the full suite passed: 2081 passed, 24 skipped;
- compileall, ruff `F821,E9`, and `git diff --check` passed.
