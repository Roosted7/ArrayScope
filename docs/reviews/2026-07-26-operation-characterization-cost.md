# Operation characterization cost — 2026-07-26

## Verdict

Bundle D's normal probe is bounded and worth paying once. For a three-dimensional
input it makes seven shape calls (one compact base plus two independent
variations per non-singleton axis) over at most 1,550 input elements when the
real array is large. The process-local cache reduces a repeated lookup to about
5 microseconds on this machine.

A shape relation that does not fit is deliberately a different cost class. If
the real signature is small, its bounded base probe is already exact. If it is
large, `ArrayDocument` makes the first real whole-array call to learn only that
signature, marks the operation unpredictable, and routes subsequent evaluation
through the ordinary whole-array `cache_stage`. That fallback is not bounded
and can be not worth it for an expensive command or a very large array. It is
still preferable to either refusing a correct operation or extrapolating a
shape rule without evidence. Diagnostics report bounded-probe and real-whole
costs separately.

## Method

Environment: `conda run -n arrayscope`, editable import verified at
`/home/thomas/projects/ArrayScope/.worktrees/discovered-shapes-bundle-d`.
Measurements used `perf_counter_ns`, 300 cold-cache repetitions per operation,
and input signature `(2048, 2048, 512)` / `float32`. Each cold repetition reset
the characterization cache; the warm measurement immediately repeated the same
signature.

| operation | bounded calls / elements | cold median | cold p95 | warm median |
| --- | ---: | ---: | ---: | ---: |
| identity | 7 / 1,550 | 201.8 µs | 388.4 µs | 4.6 µs |
| two-axis pad | 7 / 1,550 | 492.7 µs | 798.8 µs | 5.1 µs |
| last-axis decimate | 7 / 1,550 | 663.3 µs | 1,505.1 µs | 4.8 µs |
| region-capable `a * 2 + 1` | 7 / 1,550 | 467.0 µs | 847.6 µs | not measured |

The wall times include fitting and, for the region-capable row, the unified
region-conformance samples. They are local CPU numbers, not a promise for
subprocess runtimes. Command startup can dominate by orders of magnitude; the
bounded element count does not make process launch cheap.

## Rules and refusal boundary

The fitter accepts only:

- identity;
- axis removal;
- one-to-one axis permutation;
- fixed output axes;
- a constant per-axis offset (pad/crop);
- an exact rational scale, or an unambiguous floor/ceil/round-half-up rational
  scale.

Each output axis may depend on at most one input axis, and one input axis may
not feed two output axes. Ambiguous rounding behavior, coupled axes, changing
rank, inconsistent dtype, and any observation outside those families are not
extrapolated. They become exact/unpredictable for the input signature and use
OPAQUE whole-array execution with a cache stage.

## Correctness gates

- The cache key contains operation id, bound axis and parameters, input
  shape/dtype, and dynamic source identity. Python files use absolute path plus
  nanosecond mtime; command definitions can supply their template text through
  the same `source_identity` field.
- A linked-file edit changes the key and reimports/reprobes.
- A runtime shape or dtype mismatch is withheld before presentation, replaces
  the fitted entry with an exact unpredictable entry, and increments the
  invalidation counter.
- A decimating user operation was exercised through the real offscreen
  PyQtGraph montage tile path: six discovered output slices settled, their
  committed values matched source indices `0, 2, 4, 6, 8, 10`, and the opaque
  stage-cache candidate/store was observed.
