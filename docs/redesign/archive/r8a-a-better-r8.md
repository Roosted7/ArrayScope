# A better R8 sequence

## Current gate order

- **R8A pixel truth:** complete.
- **R8B complex truth:** complete.
- **R8C.1 semantic transitions:** complete.
- **R8C.2 viewport/convergence boundaries:** complete.
- **R8C.3 semantic level-evidence ownership:** active.
- **R8C.4 committed manual-camera policy:** blocked on R8C.3.
- **R8C final transition certification:** pending.
- **Marathon salvage audit:** deferred until all R8C gates pass.
- **R8D performance work:** blocked.

Truth remains the first priority, followed by convergence, responsiveness,
throughput, and cleanup. No later gate may weaken an earlier one.

## R8A — Viewer truth firewall — complete

Every tile target and backend acknowledgement carries one typed semantic
identity. Presentation state such as levels/LUT remains separate from source
pixels. Immediately before drawing:

`acknowledged identity == committed target identity`

An explicitly proven compatible lower-quality fallback is also valid. Any
other tile is a loading placeholder. Scalar, complex RG32F, and display-ready
RGB storage are distinct, and incompatible scalar/complex transitions cannot
mix inside a committed frame.

## R8B — Synthetic complex truth — complete

The deterministic adversarial fixture covers constant-magnitude phase ramps,
constant-phase magnitude ramps, real-only values, imaginary-only values,
zeros, and known source-index signatures. PyQtGraph and VisPy are checked for
texture kind, real/imag plane identity, complex mapping, values, LOD, levels
generation, and backend acknowledgement. The per-tile truth overlay reports
the same facts at each tile's screen position.

The earlier orange-background complex-rendering failure is resolved. Scalar to
complex to scalar transitions and out-of-order completions use placeholders
instead of stale pixels.

## R8C — Transition and convergence

Semantic transitions—operation, source population, channel, axes, and complex
mode—are atomic. Quality transitions for the same semantic source may refine
progressively and monotonically.

### R8C.1 — Semantic transitions — complete

Certified invariants:

- no stale source appears in a new slot;
- labels and values describe the acknowledged pixels;
- incompatible successors become placeholders;
- compatible lower-quality fallbacks are explicit;
- out-of-order completion cannot clear a current obligation.

### R8C.2 — Viewport and convergence boundaries — complete

The physical onscreen set gates visible completion. Coverage-ring and
boundary-only tiles do not masquerade as visible obligations. First-commit
resize intent and later user resize ownership are preserved consistently across
PyQtGraph and VisPy.

### R8C.3 — Semantic level-evidence ownership — active

Certification invariant:

> Semantic level and histogram evidence has one explicit, bounded owner
> independent of tile visibility and texture residency. Offscreen sources
> required for semantic statistics are evidence work, never visible tile work
> or generic prefetch.

The owner must:

- cover the complete expected semantic source population;
- perform bounded, observable evidence work without texture upload;
- reuse immutable per-source evidence behind generation guards;
- publish one truthful levels/histogram generation;
- wake a parked presentation exactly when evidence becomes sufficient;
- converge without another user action;
- leave visible admission, generic prefetch, and LOD policy unchanged.

### R8C.4 — Committed manual-camera policy — blocked on R8C.3

Once a montage camera is committed as manual, source-population growth must not
silently auto-fit it. The stronger two-backend transition cannot be certified
until PyQtGraph can obtain full semantic level evidence for the successor.

### R8C final transition certification — pending

Certification requires the centered synthetic fixture, the real fixture
standalone, back-to-back semantic transitions, and the relevant broad test
slice on both backends. Acceptance is zero unsafe visible pixels, zero orange
background failures, correct levels/histogram generation, and reliable
convergence.

### Marathon salvage audit — deferred

Only after every R8C gate passes, compare the marathon worktree for session
generation, physical residency identity, source remapping,
emitted/acknowledged semantics, atomic successor presentation, and physical
draw completion. Add salvage tests before porting code. Do not import its
throughput or scheduling experiments during R8C.

## R8D — Performance — blocked

Do not freeze or optimize the benchmark until R8A–R8C final certification is
green. Then freeze the fixture, window geometry, stage definitions, event-loop
pumping, metrics, thresholds, and baseline commit before optimizing one
measured cause at a time. Keep the hard 50 ms callback gate and define the
16 ms target once before using it.
