# R8 current status

Updated: 2026-07-14

R8 remains limited to viewer truth and convergence. General throughput,
scheduling optimization, and R8D performance work remain blocked. The complete
chronological investigation is preserved in
[`archive/r8-investigation-2026-07-14.md`](archive/r8-investigation-2026-07-14.md).

## Gate status

- **R8A pixel truth:** complete.
- **R8B complex truth:** complete.
- **R8C.1 semantic transitions:** complete.
- **R8C.2 viewport/convergence boundaries:** complete.
- **R8C.3 semantic level-evidence ownership:** complete.
- **R8C.4 committed manual-camera policy:** complete.
- **R8C transition matrix:** complete for both backends on real Wayland.
- **Full R8 certification:** pending the repository-wide UI/app exit gate.
- **Marathon salvage audit:** deferred until the full R8 gate passes.
- **R8D performance work:** blocked.

The earlier orange-background complex-rendering failure is resolved. Both
backends enforce typed target/acknowledgement compatibility, distinguish scalar,
complex, and RGB texture storage, and use placeholders when a current semantic
target lacks compatible physical pixels. The synthetic adversarial complex
fixture and scalar/complex back-to-back transitions are green.

## R8C.3 semantic evidence owner

`LevelStatsService` now owns one explicit semantic-evidence path independent of
tile visibility, texture residency, and generic montage prefetch. `FrameSession`
contains the immutable evidence target and bounded progress tracker, but no
scheduling state was added to `TileLifecycle`.

The evidence evaluator:

- accepts an immutable document/view snapshot, bounded source batch, sampling
  limits, cancellation token, and compatible materialization/stage context;
- samples raw sources deterministically and evaluates derived sources through
  the operation planner and stage contracts, including montage-axis-coupled
  operations;
- returns per-source `TileLevelStats` and work accounting without constructing
  display images, RGB payloads, rendered tiles, holders, textures, atlas
  entries, or presentation state;
- reuses compatible stages and rejects stale results before tracker, cache,
  metadata, or presentation publication.

The initial policy remains deliberately bounded: 8,192 sampled pixels per
source, 65,536 aggregate histogram samples, at most 16 sources in the blocking
CPU batch, and 2 in background refinement. Work stays on the existing histogram
kernel lane and priority. GUI continuations merge at most one batch and advance
cursor/count progress without rescanning the complete source population.

Rendered payload evidence may seed the tracker, but visible-subset evidence can
never satisfy full semantic completion. PyQtGraph retains predecessor geometry
until refined CPU-window evidence covers the target population. VisPy may
present valid rough evidence and continues to the same full refined population.

The admission trace exposed two narrow liveness defects in the existing lane:
blocking PyQtGraph evidence needed one histogram worker after visible work
drained, and result-backlog alone incorrectly parked VisPy histogram work even
when no runnable visible work existed. The governor now accounts for those two
conditions without moving evidence into visible work or changing its lane and
priority.

Diagnostics expose the target population, covered source count/sample, pending
batches, in-flight generation, configured source/pixel bounds, and the precise
blocking reason.

## R8C.4 committed manual-camera policy

The ADR 0042 guard landed separately after R8C.3. Auto-fit rescue no longer
overwrites a committed manual montage camera when the montage axis is unchanged.
Initial slice-to-montage auto-fit and explicit fit-mode behavior remain green.

## Certification evidence

Focused deterministic tests cover raw sparse evidence, complex channel/scale
mapping without RGB construction, NaN/vacuous sources, stage-cache reuse,
montage-axis-coupled operations, bounded progress, semantic supersession,
diagnostics, and both backend commit contracts.

The real Wayland certification used a 20-source `384x640` landscape dataset in
a `1200x820` window with a measured image viewport and a committed manual camera
showing exactly source 0. The `0..4` to `0..19` transition passed for PyQtGraph
and VisPy without `QT_QPA_PLATFORM=offscreen`: camera state was preserved, all
20 semantic sources reached `MONTAGE_SAMPLED_FULL`, and no offscreen source was
evaluated, uploaded, or acknowledged as display work. The live channel,
complex-mode, operation, axes, and viewport-retarget matrix also passed for both
backends.

Validation recorded on this branch:

- `tests/core tests/operations`: 385 passed with default parallelism;
- `tests/display tests/window`: 830 passed with default parallelism;
- live Wayland manual-camera transition: 2 passed;
- live Wayland R8C transition/complex matrix: 19 passed;
- compileall, focused Ruff `F821,E9`, and `git diff --check`: green.

The complete default-parallel `tests/ui tests/app` exit gate is not green on the
clean redesign baseline. The first isolated failures are an independent
diagnostics cache-text assertion and stale `_montage_session` test ownership;
the latter belongs to the preserved canonical `_frame_session` migration in the
handoff stash. Those changes, along with unrelated ROI follow-up, remain in the
stash and were not bundled into R8C.3 or the camera guard.

## Next bounded slice

Keep R8D and the marathon salvage audit blocked. Reconcile the preserved
canonical frame-session migration and unrelated UI/app baseline failures as a
separate test-first slice, then rerun the full non-GPU exit gate. Do not weaken
the R8 certification checklist or treat the focused R8C matrix as the full R8
exit gate.
