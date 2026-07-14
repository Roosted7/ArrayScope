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

## 2026-07-14 R8C stop checkpoint: transition-matrix audit

Work stopped again after two rejected R8C assumptions.

1. **Rejected: the existing lifecycle suite already expresses the R8C
   transition contract.** It mixed opaque source IDs with typed backend
   acknowledgements, treated active targets as synonymous with drawn
   payloads, stopped convergence before `note_committed()`, and expected the
   session to duplicate the surface's physical-removal ownership. Those
   tests have been modernized in four green, test-only commits.
2. **Rejected: every VisPy atlas slot on a visible page is a drawn tile.** A
   real Wayland layout-reflow probe reported sources 0 through 2 as unsafe
   with no current target, but those rows came from resident-only pool slots.
   VisPy draws only the tile keys in the page's active payload mapping; page
   visibility alone is insufficient evidence. This probe result is discarded
   and must be repeated against active vertices/page payloads.

The audit did find one real viewport defect before stopping:
`FrameSession.retarget_viewport()` compared
`MontageGeometry.layout_identity`, a property that does not exist. Both sides
therefore evaluated as `None`, every column/row reflow appeared unchanged,
and queued tiles retained predecessor coordinates. Comparing the typed
`MontageGeometry` values directly fixes both focused regressions. The full
frame-session and viewport-priority slice is 104 passed.

Next single hypothesis after this checkpoint: when the Wayland VisPy probe
restricts its acknowledgement scan to `_page_payloads_by_index`, every drawn
tile after the FFT/FFTShift/iFFT reflow will have a current compatible target.
Do not change production code unless that corrected physical draw set still
contains a mismatch.

### Invalidated reflow visual probe

The corrected draw-set identity scan itself reported zero mismatches, but its
Wayland geometry was not acceptable visual evidence. It deliberately resized
the outer window from `1200x520` to `520x900` to force a 10-column -> 1-column
reflow; the second size left the actual image viewport effectively obscured by
docks. That run is retained only as backend-state evidence and discarded as a
visual smoke. Repeat at realistic desktop sizes and assert a usable viewport
extent before accepting the reflow visually.

The automated geometry comparison evidence remains valid: the synthetic
complex, complete frame-session, viewport-priority, semantic-transition, and
cross-backend viewport-transition slice is 115 passed. No scheduling or
throughput policy changed.

### Valid realistic reflow smoke

The replacement actual-Wayland smoke kept the outer window landscape at
`1200x820 -> 1050x820`; the measured image viewport remained usable at
`741x585 -> 591x585`. PyQtGraph and VisPy both reflowed the centered 12-tile
FFT/FFTShift/iFFT fixture from 5 columns to 4, settled all 12 drawn tiles, and
reported zero drawn acknowledgement/target mismatches. The per-tile truth
overlay was enabled. Screenshots are `/tmp/r8c-realistic-pyqtgraph.png` and
`/tmp/r8c-realistic-vispy.png` (ephemeral local evidence, not repository
artifacts).

## 2026-07-14 R8C stop checkpoint: VisPy resize fight

Work stopped after two rejected hypotheses, as required by the R8 workflow.
The report is a real Wayland-only VisPy resize fight: during a manual main
window resize, the outer window widely jumps back toward older geometry.
PyQtGraph did not show the same top-level fight in the user's comparison,
although its missing/stuck-tile convergence remains a separate open gate.

1. **Rejected: the VisPy canvas or camera feeds a stale size back into the
   main window.** A real Wayland event trace recorded the top-level
   `QWindow`, main widget, `VisPySurface`, display container, Qt viewport, and
   VisPy native canvas. The top-level window jumped first; the container,
   viewport, and native canvas then followed the accepted top-level size
   exactly. No child canvas resize preceded or requested the outer jump.
2. **Rejected: an active strong-Wayland canvas-preservation correction fights
   the user resize.** A second real Wayland trace recorded every
   `CanvasPreserveController` event. No preservation transaction started and
   no correction, strong constraint, or nudge ran. Instead, each oscillation
   was preceded by `canvas_preserver.cancel()` while the window repeatedly
   returned to the saved `982x706` viewport geometry.

The probe used a centered six-tile synthetic fixture with
FFT -> FFTShift -> iFFT and the persisted VisPy/strong-Wayland settings. It
was run onscreen; no offscreen rendering evidence is used. Representative
outer-window transitions were `1135x1013 -> 1441x907 -> 600x800 ->
1441x953`, followed by repeated returns toward `1441x907` while the compositor
accepted intermediate drag sizes. The saved viewport session at reproduction
time named viewport shape `706x982`, which matches the repeatedly restored
canvas extent.

Next single hypothesis after this checkpoint: an unreleased viewport-
continuity transaction treats every VisPy resize as an incomplete layout
restore and calls `resize_to_dockless_viewport_shape()` again. Instrument only
the continuity transaction generation/flags and the restore call sites; do
not change camera, rendering, scheduling, or canvas-preservation policy until
that call chain is proven.

## 2026-07-14 R8C stop checkpoint: hidden first-commit tiles

The first resize-correction smoke was invalid despite reporting six physically
drawn tiles. Its live camera covered only one tile (`x=-3..13`,
`y=-0.68..12.68`) while the six-tile montage covered `32x25`. The session had
planned against a `493x447` viewport, the settled viewport was `619x741`, and
no viewport correction remained pending. Backend visibility counters alone
therefore did not prove visible framebuffer coverage.

Those six tiles were not prefetch. The session's intended fitted camera made
all six part of `visible_tile_numbers`; the live camera failed to apply that
intent, so visible-lane work became wasted from the user's perspective.

Two narrowly separated findings followed:

1. **Insufficient by itself: notify a montage resize while `image is None`.**
   The shared `ArrayScopeGraphicsView.resizeEvent()` dropped every resize
   before the first tiled image commit. Forwarding that resize updated the
   session viewport but did not restore the camera.
2. **Confirmed cause: the same programmatic resize demoted AUTO to USER.**
   `QGraphicsView` emitted a range change before the semantic resize handler,
   converting `AUTO_UNTOUCHED` into `USER` around the predecessor one-tile
   range. Signal-blocking the complete no-image resize transaction preserves
   AUTO intent and lets the renderer retarget against the settled viewport.

Commit `305aa69` contains only that shared production fix and two
backend-parametric regressions. The original immediate, centered synthetic
FFT -> FFTShift -> iFFT Wayland smoke was repeated on PyQtGraph and VisPy.
Both settled with six visible/drawn tiles, a session viewport of `619x741`,
and a full-montage camera. Screenshots are
`/tmp/r8-resize-pyqtgraph.png` and `/tmp/r8-resize-vispy.png` (ephemeral local
evidence, not repository artifacts).

### Newly exposed convergence stop

Modernizing the broader interaction tests exposed a separate ADR 0042
violation: increasing an already-committed montage source set while manually
zoomed calls the legacy visible-fraction "autofit rescue". An experiment that
disabled that rescue only for a committed manual camera preserved the camera,
but uncovered a lost-work state and is not ready to commit:

- committed geometry and the current plan both named sources `0..19`;
- the manual camera and session range agreed;
- visible tiles were `{0, 1, 5, 6}`;
- pending tiles were `{6, 7, 11, 12}`;
- tile 6 remained a loading placeholder;
- loading tiles, active requests, attached stage requests, viewport
  continuations, and deferred-stage obligations were all empty;
- `visible_plan_complete()` remained false after ten seconds.

The illegal auto-fit was masking this convergence failure by expanding the
visible plan and taking a different work path. Do not change scheduling or
admission policy. The next single hypothesis must trace how
`retarget_index_window()` leaves pending tile 6 without a materialization
owner after the hot/deferred stage decision. Prove the missing ownership
handoff with a focused real-pipeline regression before changing code.

The marathon worktree contains a useful but unported content-extent contract:
`setViewportContentExtent()` returns whether the extent changed and
`refreshViewportContentExtentIntent()` reapplies AUTO/FIT only after backend
acknowledgement. Port tests before considering any of that code; it is not an
explanation for the ownerless pending tile above.

## 2026-07-14 R8C stop checkpoint: onscreen completion asymmetry

The apparent ownerless tile above was a scope mismatch, not a missing
materialization handoff. `FramePlan.active_region_ids` correctly admitted only
the physically onscreen tiles, while `visible_plan_complete()` incorrectly
required the broader coverage ring to settle. Commit `359f618` changes only
that completion predicate to use the existing `onscreen_target_settled()`
contract. Its focused invariant and the complete frame-session file are green
(61 tests). No admission, prefetch, queue, or scheduling policy changed.

The read-only `redesign-r8-marathon` comparison contains the same stale
`visible_plan_complete()` predicate, so there is no completion fix to port. It
does contain a narrower prefetch-busy check based on onscreen settlement; keep
that as a later salvage-matrix candidate, not as part of this truth slice.

The committed-manual-camera experiment was then repeated onscreen on the real
Wayland session for both backends. VisPy converged and preserved the camera;
PyQtGraph did not. Two hypotheses were insufficient to explain the full
PyQtGraph asymmetry, so work stops here before another experiment:

1. **Partially confirmed, insufficient: coverage-ring completion was the
   blocker.** The focused lifecycle invariant is fixed, and the VisPy path now
   converges. PyQtGraph still timed out.
2. **Confirmed geometry defect, insufficient as the full explanation:
   boundary-only tiles were classified onscreen.** With camera range ending at
   `y=3`, tile 5 begins exactly at `y=3` and has zero visible area, yet the
   inclusive montage intersection named active regions `(0, 5)`. A focused
   planner regression proves that zero-area edge contact must not be active;
   the strict-intersection change passes the 75-test planner, montage, and
   frame-session slice. The real PyQtGraph interaction still timed out, while
   VisPy passed.

The last pre-change PyQtGraph probe recorded
`onscreen_target_unsettled_tiles() == (5,)`, no stale levels, active regions
`(0, 5)`, coverage `{0, 1, 5, 6}`, and a camera of approximately
`x=-0.212..2.212, y=0..3`. The next investigation must first record the same
minimal tuple after strict intersection: active ids, target-unsettled ids,
backend acknowledged ids, committed geometry identity, and plan geometry
identity. Do not modify admission or scheduling. If the only remaining failure
is committed-geometry publication, test that ownership boundary before code.

## 2026-07-14 R8C stop checkpoint: semantic level evidence has no owner

The strict-intersection probe reduced the physical active set to `(0,)`.
PyQtGraph then reported no onscreen unsettled tiles and the correct exact scalar
acknowledgement for tile 0, while the committed geometry remained the
predecessor `0..4` and the session plan named `0..19`. The last commit outcome
was `level-evidence-wait`, not a tile materialization or backend-ack wait.

The evidence tracker was exhausted but incomplete:

- pending level tiles: empty;
- scan remaining: zero;
- evidence worker in flight: false;
- rank: `MONTAGE_VISIBLE_SUBSET`;
- sampled sources: `{0, 1, 2, 5, 6, 10}`.

ADR 0032 requires montage levels and histograms to describe the semantic index
population, not only current canvas pixels. Advancing the `0..19` committed
geometry with this partial range would therefore render tile 0 with the wrong
semantic level generation. Keeping the predecessor is truthful but violates
convergence because no current work owner can produce evidence for the other
sources.

Two test-first hypotheses did not close the end-to-end gate:

1. **Valid invariant, insufficient:** CPU atomic successor readiness now has a
   focused regression requiring only `onscreen_tile_numbers()`, rather than the
   coverage ring. The helper test passes, but the real PyQtGraph transition
   still parks on semantic level evidence.
2. **Valid gate, insufficient:** the marathon worktree's narrower montage
   prefetch busy check allows coverage-only backlog once the onscreen target is
   settled. Its focused test passes, but the real raw-data transition schedules
   no additional semantic evidence and still has the identical six-source
   tracker state after ten seconds.

Both insufficient production experiments and their focused helper tests were
removed after this checkpoint; neither remains in the working tree. Do not
extend the prefetch or admission policy next. The read-only ownership trace
proved that raw montages are explicitly rejected by generic montage prefetch
(`blocked_no_stage`) and that the level-evidence service only scans already
rendered payloads. Decide whether full-range level evidence belongs to a
dedicated bounded histogram lane rather than display-tile prefetch. The
marathon branch has no implementation for that missing-source evidence owner;
its histogram changes only move aggregation of already-collected per-source
samples off the GUI thread.
