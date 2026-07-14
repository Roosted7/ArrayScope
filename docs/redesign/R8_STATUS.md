# R8 status

Updated: 2026-07-14

## Scope and stop condition

This checkpoint covers R8 viewer truth and convergence only. Scheduling,
throughput policy, and LOD admission remain out of scope. The work started from
clean R7+UI commit `906e5c3c`; `.worktrees/redesign-r8-marathon` remains
reference-only.

Work is stopped at the required two-hypothesis boundary. Two changes improved
machine-checkable invariants but did **not** eliminate the visually observed
wrong complex tiles:

1. `b49f67c3` aligned predicted and acknowledged texture kinds with the
   storage each backend physically draws.
2. `09729a6` prevented retained wrappers from carrying a prior source's typed
   tile identity after a semantic retarget.

Both causes were real and their focused regressions are green. Neither is a
complete explanation for the orange complex-tile failure described below. No
further rendering experiment should begin without treating that failure as a
new hypothesis.

## Landed invariant slices

- `500bcdc1` restores the R7 frame API coherence needed to run the viewer.
- `44b77892` introduces typed semantic tile target and acknowledgement
  identities.
- `257e5be2` enforces target/acknowledgement compatibility immediately before
  either backend draws.
- `bc8b7477` removes the stale paced-admission keyword that crashed the
  presentation gate.
- `adf81fd4` attaches live presentation effects before level side work can
  schedule a commit.
- `b49f67c3` separates scalar storage, complex RG32F storage, and PyQtGraph's
  physical RGB8 output.
- `db2aaa3f` adds the six-pattern adversarial complex fixture.
- `b6ece11` replaces the rejected global truth HUD with a dedicated
  `tile_truth_overlay.py` layer and one spatially attached label per visible
  tile on both backends. Each label records target/acknowledged source,
  texture kind, real/imag upload-plane identity, complex mapping, LOD,
  semantic generation, levels generation, and DRAW/LOAD state.
- `09729a6` rejects retained slot wrappers with the wrong semantic source and
  rebuilds typed identity when a compatible wrapper is retargeted.
- `687fa60` fixes the montage-prefetch completion callback to use the
  canonical frame-session staleness guard and adds a real-orchestrator
  completion-path regression plus a stale-name architecture guard.
- `e759738` updates the profiling tool to read
  `win.renderer._frame_session`, removes retired tile-ledger diagnostics, and
  makes capped real-fixture runs select the center of axis 2.

## Green automated evidence

Synthetic truth and transition coverage:

- The adversarial fixture covers constant-magnitude phase ramp,
  constant-phase magnitude ramp, real-only, imaginary-only, zero, and a known
  source signature.
- PyQtGraph's complex RGB8 output matches the CPU reference.
- VisPy's complex RG32F upload matches the exact real/imag planes.
- Back-to-back semantic transitions on both backends hide unacknowledged
  successors and expose placeholders until compatible acknowledgements arrive.
- The retained-wrapper regression reproduces a source-3 wrapper carrying a
  source-0 typed identity and proves the rebuilt identity, semantic key, and
  upload-plane pointer match the current source.

Recent validation:

- `tests/display`: 525 passed.
- Focused retained identity and residency tests: 7 passed.
- Complex synthetic fixture and transition tests: 4 passed.
- Broad LOD-residency plus complex slice: 139 passed, with the known baseline
  `test_retarget_index_window_demotes_misses_with_immediate_invalidation`
  failure.
- Prefetch completion, canonical guard, and ready-display focused tests:
  3 passed.
- Broad prefetch/architecture slice: 74 passed, with two unrelated existing
  architecture-guard failures (retired profiler ledger text before `e759738`,
  and a `QTimer.singleShot` allowlist count in `window/main.py`). The retired
  profiler ledger failure is green after `e759738`.
- Profiling-tool tests after the ownership update: 22 passed, 2 opt-in real
  profiler smokes skipped.

## Failed visual certification

The 2026-07-14 real-display runs used the bundled NIfTI on Wayland with six
tiles and a raw -> centered-FFT complex -> raw transition. Lifecycle
diagnostics reported all six tiles complete and drawable with matching target
and acknowledged source identities:

- PyQtGraph: `scalar_r32f -> rgb8 -> scalar_r32f`.
- VisPy: `scalar_r32f -> complex_rg32f -> scalar_r32f`.

Those counters are **not** acceptance evidence. Visual observation found:

- PyQtGraph retained wrongly rendered complex tiles.
- VisPy flashed the same defect.
- Empty background appeared as high-magnitude orange phase where it should be
  black.

The run also selected edge slices `0:6`, which are too empty for reliable
visual comparison. That fixture choice is rejected. All future capped
real-file checks must use slices centered on axis 2. For the bundled
`(336, 336, 272)` file and six tiles, that means indices `133:139`.

An earlier invocation of the real workflow with
`QT_QPA_PLATFORM=offscreen` is invalid and discarded. Real rendering,
framebuffer, visual, performance, Wayland, and GPU claims must never use the
offscreen platform. Offscreen remains valid only for deterministic tests that
do not claim real rendering behavior.

## What the failure proves

Typed lifecycle agreement is necessary but not sufficient. At least one of
these statements can currently be false while a tile is marked DRAW:

- the bytes actually bound/drawn are the bytes whose plane identity was
  acknowledged;
- the backend's active texture interpretation matches the acknowledged
  texture kind;
- the active scalar/complex shader or CPU mapping matches the current source;
- the active levels generation belongs to the same committed presentation;
- backend acknowledgement occurs only after those physical and presentation
  bindings are active.

In particular, scalar-only pixels must never be drawn through a complex
viewing mapping. Scalar components derived from complex source data are valid
only when the complex-storage shader explicitly selects the correct component.

## Next single hypothesis

Instrument the backend commit boundary—not the scheduler—with the smallest
possible physical draw record per tile: actual bound texture kind and
dtype/shape, real/imag upload-plane identities, active scalar/complex mapping
mode, active levels generation, and the identity returned in the backend
commit report. Reproduce on the centered six-slice fixture and compare the
first wrong orange tile with a black tile before changing code.

If that record shows physical binding and acknowledgement diverge, fix only
the acknowledgement timing/source. If they agree, reject that hypothesis and
inspect the shared complex magnitude/levels mapping next. Do not optimize
throughput or change scheduling policy during either step.

## Remaining R8 gates

- Centered real-fixture standalone visual check on PyQtGraph and VisPy.
- Centered raw -> complex -> raw back-to-back transition with no wrong flash.
- Synthetic adversarial fixture and relevant broad slice after the eventual
  cause fix.
- Only then proceed through the R8C semantic/quality/presentation/viewport
  matrix and R8D measurement work.

## 2026-07-14 stop checkpoint: two rejected hypotheses

Work stopped after the second rejected hypothesis, as required by the R8
workflow.

1. **Rejected: the last PyQtGraph tile contains a wrong committed complex
   plane.** On the centered `133:139` standalone fixture, every exact complex
   payload, magnitude plane, CPU phase RGB base, histogram source, and final
   `ImageItem` pixel matched an independently chunked reference, including
   source 138. The wrong orange tile was not caused by that committed plane.
2. **Rejected: directly replacing `ViewState.montage_indices` is a sufficient
   synchronous viewport-retarget test boundary.** The attempted regression
   remained on the same active `FrameSession` for ten seconds. The test setup
   did not prove the intended session-transition boundary and must not be used
   as evidence.

The first back-to-back raw -> FFT/FFTShift/iFFT probe instead found the
concrete PyQtGraph defect: after the complex target became current, all old
raw `ImageItem`s remained visible even though their typed acknowledgements did
not satisfy the complex targets. Hiding all physical tile mappings at every
new frame-session activation removes that unacknowledged visibility while
retaining residency.

The corresponding real VisPy run then exposed a separate identity alias: an
opaque source key for source 136 could hit a resident entry whose typed
acknowledged identity still named source 0. Source-key-only fast paths skipped
the upload and treated the stale atlas slot as presented. All VisPy reuse,
warm-skip, active-mapping, and view-cache paths now also require the typed
acknowledged identity to equal the payload acknowledgement.

Current evidence before resuming:

- Focused synthetic identity, surface invalidation, and semantic-transition
  tests: 4 passed.
- Full display slice in the maintained conda environment: 528 passed.
- Relevant display/UI/window slice: 105 passed with the pre-existing
  scheduling-policy assertion that expects `max_free_retargets == 24` while
  production returns 12. Scheduling is intentionally out of scope.
- Actual Wayland centered four-phase smoke on both backends (`raw0`, complex
  FFT/FFTShift/iFFT, `raw1`, complex FFT/FFTShift/iFFT): all phases settled,
  no unsafe visible identities, black empty background, matching alternating
  checkerboard phase, and tile sources 133 through 138.

Next single hypothesis after this checkpoint: a viewport retarget must be
driven through the canonical viewport controller (or wait for the initial
progressive session to become fully complete) before asserting the atomic
presentation boundary. Do not change production code for that test until the
real control-flow boundary is demonstrated.

### Resolution after the stop checkpoint

The canonical control flow was demonstrated: an index-window change mutates
the existing `FrameSession` through `retarget_index_window`; session object
replacement is neither expected nor required. Waiting for the active plan to
name sources 2 through 7 reproduced the invariant violation before any fix:
PyQtGraph tile slot 4 remained visibly acknowledged as source 4 while its
current lifecycle target named source 6.

The in-place retarget now invalidates visible backend mappings immediately
before mutating lifecycle targets, using the same residency-preserving surface
operation as full session activation. The cross-backend regression passes on
both the deterministic offscreen adapter path and the actual Wayland display.
The final relevant broad slice is 615 passed (`tests/display`, both transition
regressions, and `test_montage_backend.py` excluding only the known scheduling
policy assertion). Compileall, Ruff `F821,E9`, and `git diff --check` are also
green.
