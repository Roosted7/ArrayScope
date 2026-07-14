# R8 current status

Updated: 2026-07-14

R8 remains limited to viewer truth and convergence. General throughput,
scheduling optimization, and R8D performance work remain blocked. The complete
chronological investigation is preserved in
[`archive/r8-investigation-2026-07-14.md`](archive/r8-investigation-2026-07-14.md).

## Gate status

- **R8A typed identity:** complete.
- **R8B physical complex storage/mapping:** complete.
- **R8B.2 first-pixel level truth:** reopened and active.
- **R8C.1 semantic transitions:** reopened for source-window continuity.
- **R8C.2 viewport/convergence boundaries:** complete.
- **R8C.3 semantic evidence owner:** implementation landed; phasing
  certification incomplete.
- **R8C.4 committed manual-camera policy:** complete.
- **R8C transition matrix:** incomplete.
- **UI/app baseline cleanup:** deferred.
- **R8D broad optimization:** blocked; measurement and deterministic work
  counters are allowed.

Typed target/acknowledgement compatibility and distinct scalar, complex, and RGB
texture storage remain landed. New user-visible evidence reopens certification:
VisPy can physically draw first-pass complex tiles before a valid rough level
generation is applied, and source-window retargets discard compatible committed
overlap instead of remapping it by semantic source identity.

## Permanent R8 invariants

- A first physical VisPy tile must have a valid acknowledged level generation.
- First-pass rough evidence updates shader levels while tiles arrive.
- Rough histogram publication occurs at first-pass completion.
- Later quality passes do not repeat rough sampling.
- A source-window retarget preserves and remaps every compatible committed
  source.
- Placeholders are permitted only for targets with no compatible source.
- One-index scrolling must not create a full-black frame.

## R8C.3 semantic evidence owner — implementation landed, phasing incomplete

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

The owner remains the canonical statistics-only path for refinement and missing
semantic sources. R8B.2 must connect first-pass payload evidence to that same
tracker before physical presentation, continuously apply rough VisPy levels as
the first pass arrives, publish its rough histogram at first-pass completion,
and start refined semantic evidence only after the final display pass. Already
sampled sources must not be evaluated twice.

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

Close R8B.2 first-pixel level phasing and R8C.1 source-window continuity with
deterministic tests before production changes. After those tests fail, inspect
the marathon worktree read-only only for source remapping, physical residency
identity, session generation, emitted-versus-acknowledged semantics, atomic
successor presentation, and first-pass level/histogram phasing. UI/app baseline
cleanup and generic R8D optimization remain deferred until both user-visible
failures are closed.
