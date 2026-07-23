# WGPU cropped-axis page resolution (2026-07-23)

## Product requirement

A displayed-axis crop is a view over array values, not a new array allocation.
The GPU endpoint should retain useful source-aligned chunks and compile each
tile to the resident pages plus source coordinates it needs. The relationship
is intentionally many-to-many:

- one tile may sample one or several resident pages;
- one resident page may serve one or several tiles or successor frames;
- a one-pixel crop/index change uploads only a genuinely cold boundary page;
- presentation-only changes never materialize or upload source values.

This is the existing ADR 0055/0056 contract. A dedicated crop-shape fast path
would preserve the wrong abstraction and leave the next geometry combination
untested.

## Red-first benchmark

`profile_montage_workflow` now runs the same crop matrix in both
`display_x_axis_slice` and `display_y_axis_slice`, on both maintained
backends:

1. one centered 100-sample displayed-axis crop;
2. both displayed axes cropped;
3. independent one-pixel moves on the primary and secondary axes;
4. diagonal motion and a return;
5. an exact 256-sample source-page edge, boundary crossing, and return;
6. odd 99x101 extents and a one-pixel successor;
7. X/Y role swap and return;
8. montage to current/+1/current single-slice and montage restore.

Every crop successor records settlement, committed-frame currency, physical
tile coverage, upload delta, binding-cache hit/miss delta, and page fan-in.
The WGPU gate requires zero uploads after the source montage has made the
needed pages resident.

## Baseline on `b23f42dc` plus benchmark-only changes

Real Wayland, Intel low-power Vulkan adapter, scalar float32 NIfTI
`(336, 336, 272)`, 50 montage tiles:

| Stage | One-axis crop | Both-axis crop | First axis -1 | Second axis +1 |
|---|---:|---:|---:|---:|
| X primary | 864 ms | 727 ms | 632 ms | stalled at 3208 ms |
| Y primary | 984 ms | 812 ms | 639 ms | stalled at 3181 ms |

All observed crop successors performed zero WGPU uploads. The failure is
therefore not cold transfer pressure: after the second displayed dimension
moves, lifecycle targets remain planned/unpresented while no evaluator work is
active and the committed frame stays on the predecessor.

The opt-in cProfile run on the failing X stage recorded 33.2 million calls.
The largest crop-path Python costs were repeated physical-truth diagnostics
(`tileTruthPhysicalRows`: 1.93 s cumulative), presentation commits
(`_present_tile_delta`: 6.21 s cumulative), chunk histogram sample
reconstruction (`representative_sample_from_histogram`: 3.72 s cumulative),
and lifecycle settlement scans (0.73 s cumulative). WGPU payload binding was
0.43 s cumulative; uploads remained zero. GPU histogram resolves dominated
worker cumulative time and shutdown overlap, so profiler attribution and
plain timing remain separate evidence.

Artifacts:

- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu-trace.jsonl`
- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu.cprofile`
- `/tmp/arrayscope-xy-crop-matrix-baseline-wgpu-cprofile.jsonl`

## Implementation direction

Replace geometry-specific binding selection with one resolver:

1. derive the requested source rectangle and value family;
2. query source-aligned page coverage at each acceptable resident LOD;
3. select the best complete non-overlapping coverage;
4. emit page references plus exact source-to-page draw blocks;
5. request materialization only for uncovered source regions.

The resolver, not montage/non-montage or one-axis/two-axis branches, must own
the decision. Lifecycle and level evidence should consume its immutable result
instead of repeatedly reconstructing page keys or semantic samples on the GUI
thread.

This dossier will be updated with each bounded optimization and its before /
after evidence.
