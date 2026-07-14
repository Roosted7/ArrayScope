# A better R8 sequence


## R8A — Viewer truth firewall

Do not begin with performance.

Introduce one typed, inspectable identity for every tile target and acknowledgement. Avoid tuples assembled differently in multiple modules.

Conceptually it should contain:

```
document_generation
operation_key
source_index
image_axes
axis_flips
channel
complex_mapping
texture_kind
semantic_generation
lod
```

Presentation state such as levels/LUT should be a separate typed key, with backend capabilities defining whether it can change without new pixels.

Every backend acknowledgement must return the exact identity it physically accepted. Immediately before drawing:

`acknowledged identity == committed target identity`

If that is false, draw a placeholder.

Also forbid a committed complex frame from mixing incompatible texture kinds or complex mappings. A scalar-to-complex or complex-mode transition should be frame-atomic.

## R8B — Synthetic complex truth tests

Use a tiny, deterministic 4–8 tile dataset before the real NIfTI workflow:

constant magnitude with a phase ramp;
constant phase with a magnitude ramp;
real-only values;
imaginary-only values;
zeros;
known source-index signatures.

For both backends, verify:

every tile uses the expected texture kind;
GPU/display samples agree with a CPU reference at selected pixels;
magnitude and phase are correct;
hover and ROI inspection return native values;
scalar → complex → scalar transitions never mix generations;
out-of-order completions produce placeholders, not stale values.

Add a debug overlay showing, per tile:

```
slot
target source
acknowledged source
texture kind
channel/complex mode
LOD
semantic generation
levels generation
```

That would likely make the screenshot defect obvious in a single run.

## R8C — Transition and convergence

Test semantic transitions separately from quality transitions.

Semantic transitions—operation, source window, channel, axes, complex mode—must commit atomically.

Quality transitions—same source and semantics, better LOD—may be progressive and monotonic.

Tests should prove:

- no stale source appears in a new slot;
- no new labels describe old pixels;
- no compatible tile becomes blank;
- incompatible tiles become placeholders;
- every ready target eventually presents without another user event;
- out-of-order completion cannot clear an obligation;
- settlement means zero remaining visible obligations.

## R8D — Performance

Only after R8A–C are green should the benchmark be frozen.

Freeze:

- fixture;
- window geometry;
- stage definitions;
- event-loop pumping;
- metrics;
- pass/fail thresholds;
- baseline commit.

Then optimize one measured cause at a time.

The throughput design should favor:

- fan-in of worker completions by frame generation;
- one edge-triggered GUI wake rather than one signal per tile;
- building an immutable transaction once;
- offscreen preparation;
- a cheap final swap;
- no O(tile-count) reconstruction for a one-tile update;
- no adaptive controller that interprets a fixed page-rebuild cost as a per-tile cost.

Keep the hard 50 ms callback bar and decide once what the 16 ms target means—maximum, p95, or interaction-only p95. Do not change that interpretation during optimization.