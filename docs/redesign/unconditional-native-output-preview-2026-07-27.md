# Unconditional native-output preview evidence — 2026-07-27

## Scope

Progressive render contract R4 requires a preview pass even when the operation
pipeline cannot consume reduced input. The measured workload is the bundled
336×336×272 NIfTI with a displayed-axis `FFTShift`: reduction before that
reindex is refused, so FLOOR must evaluate each tile natively and reduce only
its output. The displayed-axis FFT and reindex refusal tests remain the
commutation gate; this change does not admit either transform to the
reduced-input route.

## Evaluation count

Backend-parametrized tests wrap the native evaluator and require exactly one
call for FLOOR. They then drive the target rung from the retained native
`RenderedTile` while replacing `evaluate_target_tile` with a function that
raises: both WGPU and PyQtGraph refine without a second evaluation.

| Backend contract | FLOOR native evaluations per tile | target evaluations per tile |
|---|---:|---:|
| WGPU | 1 | 0 |
| PyQtGraph | 1 | 0 |

## Real-Wayland 272-tile result

Fresh headless-Weston processes used the real NIfTI, the displayed-axis
`FFTShift`, the product-default preview arm, and the repository's five-second
hard interaction limit. JSONL and event traces stayed under `/tmp`.

| Backend | Revision | R4 | preview-complete ACK | target settle | presented at 5 s |
|---|---|---|---:|---:|---:|
| WGPU | `3970f5e4` parent | red, no preview ACKs | absent | >5 s | 138/272 |
| WGPU | candidate | green, 272/272 preview ACKs | 2452 ms | >5 s | 272/272 |
| PyQtGraph | `3970f5e4` parent | red, no preview ACKs | absent | >5 s | 129/272 |
| PyQtGraph | candidate | green, 272/272 preview ACKs | 3178 ms | >5 s | 272/272 |

The in-process gate's R1/R2 counts did not regress: WGPU retained its standing
R1/R2 failures and PyQtGraph retained its one standing R2 failure. Target
settlement remains red on both backends. WGPU also reports the standing
off-floor native uploads and level-containment failures once the previously
missing preview makes the full cohort observable; this change does not claim
to close those independent R1/R3/R5 queue items.

The standalone snapshot oracle was not run against the profiler event traces:
they contain `pipeline_plan`/kernel/commit events, not diagnostics snapshot
events, and the oracle correctly rejects that input rather than returning a
vacuous verdict.
