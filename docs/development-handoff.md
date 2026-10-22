# Development handoff — 2026-10-22

The user requested winddown and preservation for tomorrow. Review the pending
validation and CI investigations before new features. Broad renewal and changes
to obsolete guidance remain authorized; explain decisions and preserve guarantees.

## Checkpoint and branch status

The checkout is `/workspace/ArrayScope`, on `dev`. Product-source checkpoint:
`f1287d73`. CI `db4665e7` was cherry-picked as `6d58ea96`; child-report tracking
followed as `fc17902b`, whose 114-case integrated scoped check passed. The
handoff commit adds documentation only. Resolve its exact head from the
[`dev` history](https://github.com/Roosted7/ArrayScope/commits/dev); the chat
checkpoint also records the verified remote head. Cherry-picks have different
hashes from their source branches.

| Branch | Status at handoff |
| --- | --- |
| [dev](https://github.com/Roosted7/ArrayScope/tree/dev) | Completed product/test/CI slices and handoff landed together in bounded commits; 114-case scoped check green. New hosted CI verdict pending. |
| `dev-streaming` (local only) | Source head `886d7145`; writer, strict engine, CLI and docs integrated through `f1287d73`. |
| `dev-test-snapshots` (local only) | `abd211ea`, integrated as `23cbfd18`; focused proof below. |
| `dev-ci` (local only) | `db4665e7`, integrated as `6d58ea96`; hosted matrix has not been validated by local checks. |
| [dev-gpu-ci](https://github.com/Roosted7/ArrayScope/tree/dev-gpu-ci) | **Pushed**, WIP investigation checkpoint `14e51dd5`; unresolved GPU failures, no production fix. |
| [dev-gui-ci](https://github.com/Roosted7/ArrayScope/tree/dev-gui-ci) | **Pushed**, WIP checkpoint `e97e7c6e`; completed fixture fixes integrated separately, remaining investigations open. |

Earlier batch, history and HDF5 changes are in `dev`; guides and focused proof
are in [queue-done.md](queue-done.md). Worktrees live under `/workspace/work/`.
All source worktrees have clean tracked files. No implementation slice remains
awaiting local integration. The two open investigations are preserved as notes,
not shipped fixes; untracked scratch logs and local Vulkan assets are excluded.
Bring those investigation branches up to the landed `dev` before resuming code.

GitHub CI access through the connected GitHub tools is confirmed, even though
the separate local `gh` REST credential returns 403. Read the
[latest dev runs](https://github.com/Roosted7/ArrayScope/actions?query=branch%3Adev)
before rerunning jobs. WIP pushes also started their older branch workflows;
those results describe the WIP base, not the landed `dev` fixes.

## What is implemented

- Launcher accessibility, labelled examples and recent files; searchable
  metadata-first dataset browsing; shared recipe clipboard/import validation.
- Atomic exact-region export with worker planning, cost consent, progress,
  cancellation and captured source/definition checks. Selected and full-derived
  export are separate discoverable menu actions.
- Bounded operation Undo/Redo, source-epoch resets, backing-buffer accounting,
  rank-successor history and whole-stack synchronization. Text editors retain
  their own Undo/Redo shortcuts.
- Qt-free recipe replay for read-only NPY and explicitly selected HDF5 input;
  strict built-in recipes, metadata-only dry-run, alias refusal and truthful
  SIGINT behavior across the publication boundary. See [recipe-replay.md](recipe-replay.md).
- Shared lazy HDF5 catalogs and per-read handles across GUI, Python and CLI;
  conservative decode estimates, identity checks, Cartesian indexing and descriptor
  transport. External links/storage and VDS are refused; see [hdf5-data.md](hdf5-data.md).
- Opt-in incremental **values-only NPY** export through the same atomic writer.
  It preserves global shape, dtype, endian order and squeezing without assembling
  the whole result. The first supported operation subset is deliberately strict:
  FFTShift, SplitAxis, plugins and other unproven mappings are refused before
  planner allocation. A 64 MiB working estimate is a target, not an RSS cap.
- Command/process/shared-memory cleanup, stable ROI statistics, launched-window
  collection, import identity, settings isolation and trace-observer fixes.

## Validation: preserve the distinctions

The latest completed broad run was at **`fe4ddb9f`**:
**1 failed, 5,068 passed, 321 skipped in 611.31 seconds**. It was not all green.
Log: `/workspace/work/arrayscope-development/integrated-release-tests.log`.
The failure was
`tests/ui/test_montage_interactions.py::test_montage_tile_count_increase_preserves_manual_zoom_when_not_near_auto[pyqtgraph]`.
The subsequent test-observer correction recognizes a resident published scope
covering the actual camera scope; seven focused cases, including missing/stale/
coarse/hidden/uncovered-content controls, passed. No rendering deadline or
production renderer changed. At **`fc17902b`**, the subsequent integrated
**scoped** run passed **114 cases in 14.94 seconds**: streaming/writer/resource,
actual CLI, CI reporting, shared memory, relay, camera observer and dependency
guards. There has been no later whole-suite pass. Ruff/format covered 715 files;
compile/dead-code/timing passed. An offline-built wheel at `fc17902b` matched
all 316 production Python files. Four fresh isolated wheel-direct CLI children
passed NPY and mixed-endian complex HDF5 dry-run/stream checks against independent
NumPy, with Qt/GPU imports blocked and source/recipe hashes unchanged. This is
a small-input packaging smoke, separate from the large constrained-resource test.
Wheel SHA256: `801a8204162958c1abd11610e29e105964b85ae0f6f2afc6a442446b0de712fa`.
No package was published. Local report:
`/workspace/work/arrayscope-development/wheel-night-smoke-fc17902b/report.json`.

Additional bounded evidence, each with its own scope:

- Export relay native destruction was deterministically observed on a worker
  thread. Application ownership plus a pending/running/released worker lease
  now destroys it on the GUI thread, including dropped completions and deleted
  receivers. Seven lifecycle cases and the combined 24-case export gate passed;
  the original cancellation passed 96 repetitions after pre-fix stress crashed
  three workers across 27 interactions. This supports the fix without proving
  that this was the only possible native crash cause.
- Incremental writer plus incumbent saver: **57 passed in 2.39 seconds**.
  A separate Linux constrained-process experiment exported exact 256 MiB output
  with only 96 MiB additional address-space allowance; a full-output allocation
  control failed. This proves that workload, not a universal memory bound.
- Mutation-revision test: all **78 public methods** remain reflected; unknown
  parameters still fail and the minimum-mutating-method guard remains.
  Comparable cProfile call time changed **137.06 → 22.48 seconds**; NumPy array
  repr calls changed **103,168 → 0**. Normal focused file: **13 passed in
  20.09 seconds**, target test 10.59 seconds. Negative controls detect unclocked
  nested/new/ledger fields, equal-sum pixel permutations and a new public method.
  Profiling artifacts are in `/workspace/work/arrayscope-test-snapshots/work/`.
- CI reporting has focused real-child tests and a real local JUnit wiring check;
  the changed CI workflow passed actionlint 1.7.7. Hosted platform outcomes remain
  separate evidence. No real-Windows execution or hardware-GPU acceptance was
  established in this winddown.
- Shared-memory compatibility checks passed 72 focused cases on Python 3.12 and
  six on Python 3.13; these are not complete interpreter-matrix runs.
- Precise child dependency declarations reduced a recorded selection from
  5,390 to 383 cases; this is selection evidence, not a wall-time claim. Xdist
  file scheduling now preserves duration ordering instead of undoing it.
  One exact duplicate viewport test was consolidated; the remaining case count
  has not been certified minimal. Measure expensive repeated setup and distinct
  failure coverage before further consolidation.

## First work tomorrow

1. Read the coordinator's final head/push and wheel result before rerunning
   anything. [queue.md](queue.md) has been updated for the integrated work.
   Inspect CI logs, JUnit failures, timing CSV and summaries before another broad
   sweep by default. Retain failures instead of weakening their assertions.
2. Resume [GPU WIP notes](https://github.com/Roosted7/ArrayScope/blob/dev-gpu-ci/docs/work-in-progress/gpu-ci.md).
   Workspace-local Mesa/llvmpipe reproduces three pixel/BC5 failures; the isolated
   exact histogram passes. A failed eviction assertion can leave module-shared
   Scene state polluted for later tests. Prove that ordered cascade, then separate
   canonical readback, page tables, framebuffer coordinates and codec error.
   Preserve tolerances; software Vulkan is correctness smoke, not hardware evidence.
3. Resume [GUI WIP notes](https://github.com/Roosted7/ArrayScope/blob/dev-gui-ci/docs/work-in-progress/gui-ci.md).
   Open items are data-change ordering with legal retained scopes, intermittent
   cropped-scroll settling, and a WGPU draw/commit-overlap fixture precondition.
   Capture camera targets, acknowledgements, work and owned timers **before**
   teardown. Distinguish an incorrect observer from stranded visible work.
4. Continue reducing measured test cost and duplicate setup while preserving
   distinct hazards and negative controls. Current CI consolidates exhaustive
   strict coverage, uses cached selection or explicit smoke for fast feedback,
   and seeds a complete GUI map only on explicit workflow dispatch.
5. Extend streaming eligibility only through bounded region algebra and exact
   numerical/resource proof. Keep metadata-only admission, globally disjoint
   reads, source validation and atomic cancellation. Do not replace a clear
   refusal with whole-source probing or a whole-output allocation.
6. After correctness and CI stabilize, revisit maintained-backend journeys on
   real hardware. Any native planning boundary needs a fresh profile and measured
   benefit; the historical candidate represented only about 6% of its workflow.

## Unstarted ideas, not promises or active work

- Incremental NPZ output and a GUI streaming-export experience.
- Bounded FFTShift/Split region algebra; compressed-chunk alignment and honest
  repeated decode/read-cost accounting. An earlier eight-chunk prototype read
  109,158,400 bytes for 13,644,800 bytes of source data with a leading FFT or
  off-leading transpose; a chunkable label alone cannot establish linear work.
- Measured font-cache reuse and an audit of near-duplicate expensive tests.
- Optional content-hash provenance; lazy NPZ/MAT input where formats permit it.
- Plugin/environment warm-worker exchange and historical R7/R8 work in [queue.md](queue.md).
- Headless plugins need Qt-free read-only definition resolution and explicit
  source cost/transport contracts. Progressive acquisition/export needs a
  source-owned write generation or lock; a shallow document snapshot does not
  freeze pixel buffers.
- Rust only at a measured bulk boundary, with packaging and scientific parity proof.
- UX hypotheses requiring user-journey design and validation: keyboard/screen-reader
  audits across launcher/browser/operations; guided failure recovery with actionable
  cost/refusal messages and detailed expert plans; shareable analysis bundles;
  named-axis workflows and safer operation discovery. These are unimplemented ideas.

Use each checkout's own `tools/env` and venv. Coordinate a shared maximum of two
focused pytest workers; use an absolute private `TESTMON_DATAFILE`. IPC checks
need Unix-socket binding permission in the sandbox. Preserve untracked `work/`
logs where useful; commits/WIP notes are durable. Do not restart every branch's
tests or treat focused green as a full-suite result.
