# R5 bulk render governor evidence — 2026-07-27

## Ownership

`ResourceGovernor.decide_render_pass()` owns the item cap, byte cap, fixed
32 ms control target, 50 ms R5 evidence threshold, and the preview/target
feedback channels. It measures
`elapsed = fixed + item*n + byte*MiB + cohort²*n²`, reports residual/R²
evidence, and separates item from byte cost only when those axes varied
independently enough to identify both.

Steering minimizes predicted total fill time plus one continuous callback
latency price: zero through 18 ms, gently quadratic through 45 ms, and a
value/slope-continuous stronger quadratic above 45 ms. Fifty milliseconds is
therefore a reported violation point, not a cap or a controller mode switch.
Model extrapolation is also continuous: up to 1.1× the measured item/byte
range is free, then an exponential evidence-distance price makes 2×
noticeable, 10× expensive, and 100× practically unavailable without making
any point a hard yes/no boundary. The 32/50 ms targets never adapt downward.

`FramePipeline` produces preview tiles through its ordinary governed
continuation instead of one whole-round worker task. Presentation effects pass
the governor decision through unchanged. Backends may stop earlier at the
deadline; they cannot widen the cap or deadline.

PyQtGraph prefixes use ordinary hidden-per-item staging. The compact atlas is
still a complete hidden transaction and is never exposed as partial truth, but
production cannot enter that atomic branch unless the governor admits the
entire planned set in one measured chunk. WGPU has the same full-set admission
guard for its preview atlas.

Every backend completion emits `pass_kind`, `pass_chunk_items`,
`pass_chunk_budget_ms`, `pass_chunk_within_50ms`, and
`pass_completed_atomically`, plus payload-build, backend, acknowledgement,
state-publication, geometry, governor fixed/item/byte/quadratic fit,
fit residual, predicted callback, latency price, and extrapolation price.
`presentation_build_timing` further attributes governed payload construction.
WGPU additionally measures structural executor initialization and texture-pool
growth at their owners. They remain inside the full callback/R5 evidence, but
the steady render-pass cost sample subtracts them: a one-time capacity
transition is not evidence that the next cohort's fixed, count, or byte cost
changed.

`python -m arrayscope.tools.render_pass_governor_probe` is the reusable
acceptance probe. It enters managed Weston itself, balances backend order when
`--backend all` is requested, reports true completed-tile throughput from
wall-clock milestones separately from callback work-item throughput, and
distills full and structural-compensated chunk distributions without writing
JSONL artifacts.

## Weston measurements

All runs used the managed private Weston compositor, the deterministic geometry
scene, a 272-tile plan, and the final code in this branch. The timing probe
observed for up to 30 s so a censored baseline remained measurable; the
product/test interaction gate remains 5 s.

| Backend/pass | Repeat | chunks | full min / p50 / p95 / max (ms) | >50 | structural pool / init total (ms) | atomic |
|---|---:|---:|---:|---:|---:|---:|
| PyQtGraph scalar preview | 0 | 17 | 4.7 / 22.0 / 39.2 / 43.0 | 0 | 0 / 0 | 0 |
| PyQtGraph scalar target | 0 | 15 | 4.6 / 22.6 / 48.5 / 60.1 | 1 | 0 / 0 | 0 |
| PyQtGraph scalar preview | 1 | 12 | 3.7 / 22.7 / 35.1 / 36.5 | 0 | 0 / 0 | 0 |
| PyQtGraph scalar target | 1 | 16 | 18.4 / 33.1 / 59.5 / 77.4 | 3 | 0 / 0 | 0 |
| WGPU scalar preview, screen | 0 | 9 | 9.7 / 46.7 / 1088.7 / 1088.7 | 4 | 0 / 433.5 | 0 |
| WGPU scalar target, screen | 0 | 18 | 4.5 / 46.7 / 61.4 / 68.5 | 6 | 28.2 / 0 | 0 |
| WGPU scalar preview, screen | 1 | 13 | 12.3 / 31.2 / 38.1 / 45.3 | 0 | 22.2 / 6.7 | 0 |
| WGPU scalar target, screen | 1 | 15 | 25.3 / 46.7 / 61.4 / 107.1 | 6 | 50.9 / 0 | 0 |

The 1089 ms WGPU row is one cold callback, not an atomic pass. Of it, 434 ms
is explicitly attributed to first-executor setup; the still-unattributed cold
remainder stays in the controller sample and is visible optimization debt.
The warm target maxima include one-time pool growth: repeat 1's 107.1 ms full
callback becomes 61.4 ms after subtracting its measured 50.9 ms pool
transition. The controller does not react to that transition, while the R5
trace still reports the full 107.1 ms violation.

### Wall time and throughput

Preview throughput is `272 / preview-complete`; target throughput is
`272 / (target-settle - preview-complete)`.

| Backend/arm | Repeat | preview complete | target settle | preview tiles/s | target tiles/s |
|---|---:|---:|---:|---:|---:|
| WGPU pre-f11 baseline | 0 | 1.101 s | 5.553 s | 247.0 | 61.1 |
| WGPU pre-f11 baseline | 1 | 0.997 s | 3.791 s | 272.8 | 97.3 |
| WGPU current | 0 | 1.387 s | 4.732 s | 196.1 | 81.3 |
| WGPU current | 1 | 1.305 s | 4.521 s | 208.5 | 84.6 |
| PyQtGraph pre-f11 baseline | 0–1 | >30 s | >30 s | censored | censored |
| PyQtGraph current | 0 | 1.997 s | 6.262 s | 136.2 | 63.8 |
| PyQtGraph current | 1 | 2.095 s | 5.713 s | 129.8 | 75.2 |

WGPU preview median is 1.346 s versus 1.049 s pre-f11: a 28.3% regression.
Target-settle median is 4.627 s versus 4.672 s pre-f11: load-neutral at −1.0%
with substantial run variance. Against Thomas's earlier 4.4 s reference, the
current 4.52–4.73 s is still a 0.12–0.33 s (3–8%) regression; the 4.4 s result
has not been recovered. PyQtGraph now completes where pre-f11 remained censored
after 30 s, but its 5.71–6.26 s target settlement still fails the product's
5 s interaction bar.

Current commands were
`python -m arrayscope.tools.render_pass_governor_probe --backend <backend>
--repeat 2`. The pre-f11 worktree used the same 272-tile geometry workflow and
30 s observation window; that revision predates the pass-kind trace fields, so
its wall-clock/throughput baseline is comparable but it cannot supply the new
per-pass chunk distribution.

## Result and remaining red evidence

No preview or target pass completed atomically in these runs. The former
PyQtGraph compact-preview whole-atlas burst and FFT whole-active-set
republication are gone: the backend resolves only the admitted upserts and
retains already drawn items.

PyQtGraph preview is green in these runs, but target still has 60–77 ms
outliers. WGPU no longer performs
whole-active-set binding publication on ordinary deltas: stable plane prefixes,
tile instances, level evidence, and committed state advance only for admitted
upserts. Its pool capacity is still derived from the physical byte working set;
incremental commits grow that existing executor in place instead of skipping
the capacity owner. Bound-page pins name committed coverage rather than every
resident LOD family.

R5 remains honestly red for the WGPU cold callback, warm backend/pool-growth
callbacks, PyQtGraph scalar-target outliers, and the previously measured
indivisible PyQtGraph FFT item updates. The governor records and prices the
steady components, excludes measured one-time structural transitions from its
cost fit, never lowers the target, and does not pretend item shrinkage can
repair fixed cost. No measured preview or target pass completed atomically.
